"""IP-Adapter for PixArt-Sigma — decoupled cross-attention for image conditioning.

Architecture:
  - ImageProjectionModel (Perceiver Resampler): CLIP features → fixed-length token sequence
  - IPAdapterAttnProcessor2_0: custom attention processor with parallel image cross-attention
  - IPAdapterWrapper: wraps PixArtTransformer2DModel to inject IP-Adapter functionality

The IP-Adapter adds a parallel cross-attention path alongside the existing text cross-attention
in each transformer block. Text and image conditioning are kept separate, enabling three modes:
  1. Text-only (zero image tokens)
  2. Image-only (zero text embeddings)
  3. Text + Image (both active)

Dimension flow:
  Text path:  T5 (B,300,4096) → caption_projection → (B,300,1152) → attn2 to_k/to_v
  Image path: CLIP (B,257,1280) → ImageProjection → (B,num_tokens,1152) → to_k_ip/to_v_ip
  Output:     text_attn_output + ip_scale * ip_attn_output
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: float = 4.0):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim, bias=False),
            nn.GELU(),
            nn.Linear(inner_dim, dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PerceiverAttention(nn.Module):
    """Single Perceiver cross-attention layer: queries attend to latent + input sequence."""

    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5

        self.norm_latents = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, latents: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latents: learnable query tokens (B, num_tokens, dim)
            context: input features, typically CLIP (B, seq_len, dim)
        """
        latents = self.norm_latents(latents)
        context = self.norm_context(context)

        b, n, _ = latents.shape
        h = self.heads

        q = self.to_q(latents).view(b, n, h, self.dim_head).transpose(1, 2)
        kv_input = torch.cat([context, latents], dim=1)
        kv = self.to_kv(kv_input)
        k, v = kv.chunk(2, dim=-1)
        s = kv_input.shape[1]
        k = k.view(b, s, h, self.dim_head).transpose(1, 2)
        v = v.view(b, s, h, self.dim_head).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(b, n, -1)
        return self.to_out(out)


class ImageProjectionModel(nn.Module):
    """Perceiver Resampler — projects variable-length CLIP features to fixed-length tokens.

    Follows the IP-Adapter architecture: learnable queries cross-attend to CLIP features
    through multiple Perceiver layers, producing a compact token representation that
    aligns with the transformer's inner_dim for direct use in decoupled cross-attention.
    """

    def __init__(
        self,
        cross_attention_dim: int = 1152,
        clip_embed_dim: int = 1280,
        num_tokens: int = 16,
        depth: int = 4,
        dim_head: int = 64,
        heads: int = 16,
        ff_mult: float = 4.0,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.cross_attention_dim = cross_attention_dim

        self.proj_in = nn.Linear(clip_embed_dim, cross_attention_dim)
        self.norm_in = nn.LayerNorm(cross_attention_dim)

        self.latents = nn.Parameter(
            torch.randn(num_tokens, cross_attention_dim) * 0.02
        )

        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(cross_attention_dim, dim_head=dim_head, heads=heads),
                FeedForward(cross_attention_dim, mult=ff_mult),
            ]))

        self.norm_out = nn.LayerNorm(cross_attention_dim)

    def forward(self, clip_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clip_features: (B, seq_len, clip_embed_dim), e.g. (B, 257, 1280) for CLIP ViT-H

        Returns:
            ip_tokens: (B, num_tokens, cross_attention_dim)
        """
        x = self.proj_in(clip_features)
        x = self.norm_in(x)

        b = x.shape[0]
        latents = self.latents.unsqueeze(0).expand(b, -1, -1)

        for attn, ff in self.layers:
            latents = attn(latents, x) + latents
            latents = ff(latents) + latents

        return self.norm_out(latents)


class IPAdapterAttnProcessor2_0(nn.Module):
    """Custom attention processor with decoupled cross-attention for IP-Adapter.

    Replaces the default processor on attn2 (cross-attention) blocks.
    Computes standard text cross-attention + parallel image cross-attention,
    combining them: output = text_output + ip_scale * ip_output.

    The ip_hidden_states are set dynamically via the _ip_hidden_states attribute
    before each transformer forward pass.
    """

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: int,
        ip_hidden_dim: int,
        scale: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale

        self.to_k_ip = nn.Linear(ip_hidden_dim, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(ip_hidden_dim, hidden_size, bias=False)

        nn.init.zeros_(self.to_v_ip.weight)

        self._ip_hidden_states: torch.Tensor | None = None

    def __call__(
        self,
        attn: "Attention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        head_dim = attn.head_dim if hasattr(attn, "head_dim") else query.shape[-1] // attn.heads
        inner_dim = key.shape[-1]
        num_heads = attn.heads

        query = query.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        text_output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0,
        )
        text_output = text_output.transpose(1, 2).contiguous().view(batch_size, -1, inner_dim)

        # IP-Adapter parallel cross-attention
        ip_hidden_states = self._ip_hidden_states
        if ip_hidden_states is not None:
            ip_key = self.to_k_ip(ip_hidden_states)
            ip_value = self.to_v_ip(ip_hidden_states)

            ip_key = ip_key.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
            ip_value = ip_value.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

            ip_output = F.scaled_dot_product_attention(
                query, ip_key, ip_value, dropout_p=0.0,
            )
            ip_output = ip_output.transpose(1, 2).contiguous().view(batch_size, -1, inner_dim)
            text_output = text_output + self.scale * ip_output

        hidden_states = text_output
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(b, c, h, w)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def set_ip_adapter_processors(
    transformer: nn.Module,
    ip_hidden_dim: int,
    scale: float = 1.0,
) -> dict[str, IPAdapterAttnProcessor2_0]:
    """Replace attn2 processors in all transformer blocks with IP-Adapter processors.

    Returns:
        Dict mapping processor name → IPAdapterAttnProcessor2_0 instance
    """
    ip_processors = {}
    attn_procs = {}

    for name, module in transformer.attn_processors.items():
        if name.endswith("attn2.processor"):
            attn_block_name = name.replace(".processor", "")
            parts = attn_block_name.split(".")
            attn_module = transformer
            for part in parts:
                attn_module = getattr(attn_module, part)

            hidden_size = attn_module.to_q.out_features

            processor = IPAdapterAttnProcessor2_0(
                hidden_size=hidden_size,
                cross_attention_dim=hidden_size,
                ip_hidden_dim=ip_hidden_dim,
                scale=scale,
            )

            if ip_hidden_dim == hidden_size:
                with torch.no_grad():
                    processor.to_k_ip.weight.copy_(attn_module.to_k.weight)

            attn_procs[name] = processor
            ip_processors[name] = processor
        else:
            attn_procs[name] = module

    transformer.set_attn_processor(attn_procs)

    num_ip = len(ip_processors)
    total_params = sum(p.numel() for proc in ip_processors.values() for p in proc.parameters())
    logger.info(
        f"IP-Adapter processors set on {num_ip} attn2 blocks, "
        f"new params: {total_params:,} ({total_params * 2 / 1024**2:.1f} MB in bf16)"
    )
    return ip_processors


class IPAdapterWrapper(nn.Module):
    """Wraps PixArtTransformer2DModel with IP-Adapter image projection and processors.

    This wrapper:
    1. Projects CLIP features through the ImageProjectionModel (Resampler)
    2. Sets ip_hidden_states on all IP-Adapter attention processors
    3. Calls the original transformer forward
    4. Cleans up ip_hidden_states after forward

    The wrapper is transparent for text-only mode (ip_hidden_states=None).
    """

    def __init__(
        self,
        transformer: nn.Module,
        image_proj: ImageProjectionModel,
        ip_processors: dict[str, IPAdapterAttnProcessor2_0],
        ip_scale: float = 1.0,
    ):
        super().__init__()
        self.transformer = transformer
        self.image_proj = image_proj
        self._ip_processors = nn.ModuleDict(ip_processors)

        self.set_ip_adapter_scale(ip_scale)

    def set_ip_adapter_scale(self, scale: float):
        """Dynamically adjust the IP cross-attention scale on all processors."""
        for proc in self._ip_processors.values():
            proc.scale = scale

    @property
    def config(self):
        return self.transformer.config

    @property
    def dtype(self):
        return self.transformer.dtype

    @property
    def device(self):
        return self.transformer.device

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
        ip_hidden_states: torch.Tensor | None = None,
        **kwargs,
    ):
        """Forward pass with optional IP-Adapter image conditioning.

        Args:
            hidden_states: noisy latent (B, C, H, W)
            timestep: diffusion timestep
            encoder_hidden_states: T5 text embeddings (B, seq, 4096)
            encoder_attention_mask: text attention mask
            ip_hidden_states: CLIP features (B, clip_seq, clip_dim) or
                              pre-projected tokens (B, num_tokens, inner_dim).
                              If None, IP-Adapter is disabled (text-only mode).
        """
        if ip_hidden_states is not None:
            if ip_hidden_states.shape[-1] != self.image_proj.cross_attention_dim:
                ip_tokens = self.image_proj(ip_hidden_states)
            else:
                ip_tokens = ip_hidden_states

            for proc in self._ip_processors.values():
                proc._ip_hidden_states = ip_tokens
        else:
            for proc in self._ip_processors.values():
                proc._ip_hidden_states = None

        output = self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            **kwargs,
        )

        return output

    def clear_ip_hidden_states(self):
        """Clear cached ip_hidden_states on all processors.

        Must be called AFTER backward() when gradient checkpointing is enabled,
        because checkpoint recomputation re-runs each block's forward and reads
        _ip_hidden_states from the processor.
        """
        for proc in self._ip_processors.values():
            proc._ip_hidden_states = None

    def get_ip_adapter_state_dict(self) -> dict:
        """Extract all IP-Adapter trainable parameters for saving."""
        state = {}

        for k, v in self.image_proj.state_dict().items():
            state[f"image_proj.{k}"] = v

        for proc_name, proc in self._ip_processors.items():
            for k, v in proc.state_dict().items():
                safe_name = proc_name.replace(".", "_")
                state[f"ip_processors.{safe_name}.{k}"] = v

        return state

    def load_ip_adapter_state_dict(self, state_dict: dict):
        """Load IP-Adapter parameters from a saved state dict."""
        image_proj_state = {
            k.replace("image_proj.", ""): v
            for k, v in state_dict.items()
            if k.startswith("image_proj.")
        }
        if image_proj_state:
            self.image_proj.load_state_dict(image_proj_state)

        for proc_name, proc in self._ip_processors.items():
            safe_name = proc_name.replace(".", "_")
            prefix = f"ip_processors.{safe_name}."
            proc_state = {
                k.replace(prefix, ""): v
                for k, v in state_dict.items()
                if k.startswith(prefix)
            }
            if proc_state:
                proc.load_state_dict(proc_state)


def build_ip_adapter(
    transformer: nn.Module,
    clip_embed_dim: int = 1280,
    num_tokens: int = 16,
    resampler_depth: int = 4,
    ip_scale: float = 1.0,
    dim_head: int = 64,
    heads: int | None = None,
) -> IPAdapterWrapper:
    """Build and attach IP-Adapter to a PixArt Transformer.

    Determines inner_dim from the transformer's first attn2 block,
    constructs the ImageProjectionModel and IP-Adapter processors,
    and returns the wrapped transformer.

    Args:
        heads: Number of attention heads in the Perceiver Resampler.
            If None, auto-computed as inner_dim // dim_head to avoid
            dimension mismatch between inner_dim and heads * dim_head.
    """
    inner_dim = None
    for name in transformer.attn_processors:
        if "attn2" in name:
            parts = name.replace(".processor", "").split(".")
            attn_module = transformer
            for part in parts:
                attn_module = getattr(attn_module, part)
            inner_dim = attn_module.to_q.out_features
            break

    if inner_dim is None:
        raise ValueError("Could not find attn2 blocks in transformer")

    if heads is None:
        heads = inner_dim // dim_head

    logger.info(
        f"Building IP-Adapter: inner_dim={inner_dim}, clip_embed_dim={clip_embed_dim}, "
        f"num_tokens={num_tokens}, depth={resampler_depth}, heads={heads}, ip_scale={ip_scale}"
    )

    image_proj = ImageProjectionModel(
        cross_attention_dim=inner_dim,
        clip_embed_dim=clip_embed_dim,
        num_tokens=num_tokens,
        depth=resampler_depth,
        dim_head=dim_head,
        heads=heads,
    )

    ip_processors = set_ip_adapter_processors(
        transformer, ip_hidden_dim=inner_dim, scale=ip_scale,
    )

    wrapper = IPAdapterWrapper(
        transformer=transformer,
        image_proj=image_proj,
        ip_processors=ip_processors,
        ip_scale=ip_scale,
    )

    total_new = sum(p.numel() for p in image_proj.parameters())
    total_new += sum(
        p.numel() for proc in ip_processors.values() for p in proc.parameters()
    )
    logger.info(f"IP-Adapter total new parameters: {total_new:,}")

    return wrapper
