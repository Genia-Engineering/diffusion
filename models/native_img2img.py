"""Native Img2Img for PixArt-Sigma — 8-channel concatenation + multi-layer residual injection.

Architecture:
  Input:  [noisy_latent (B,4,H,W), reference_latent (B,4,H,W)]
          → torch.cat → (B,8,H,W)
          → modified PatchEmbed(Conv2d 8→1152) → hidden_states (B,N,1152)

  Residual Injection:
          reference_latent → ref_proj(Conv2d 4→1152) → ref_tokens (B,N,1152)
          Every `injection_interval` transformer blocks:
              hidden_states += injection_gate_i(ref_tokens)

  All new modules (PatchEmbed extra channels, ref_proj, injection gates) are
  zero-initialized to preserve pretrained behavior at initialization.

  Cross-attention receives a learnable null embedding (T5 is fully discarded).
"""

import logging

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from diffusers.models.modeling_outputs import Transformer2DModelOutput

logger = logging.getLogger(__name__)


def expand_patch_embed_to_8ch(transformer: nn.Module) -> None:
    """Expand PatchEmbed input from 4 to 8 channels, zero-initializing new channels."""
    old_proj = transformer.pos_embed.proj
    device = old_proj.weight.device
    dtype = old_proj.weight.dtype

    new_proj = nn.Conv2d(
        8, old_proj.out_channels,
        kernel_size=old_proj.kernel_size,
        stride=old_proj.stride,
        padding=old_proj.padding,
        bias=old_proj.bias is not None,
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        new_proj.weight.zero_()
        new_proj.weight[:, :4] = old_proj.weight
        if old_proj.bias is not None:
            new_proj.bias.copy_(old_proj.bias)

    transformer.pos_embed.proj = new_proj

    logger.info(
        f"PatchEmbed expanded: Conv2d(4→8, {old_proj.out_channels}, "
        f"k={old_proj.kernel_size}, s={old_proj.stride})"
    )


class NativeImg2ImgPixArtWrapper(nn.Module):
    """Wraps PixArtTransformer2DModel for Native Img2Img generation.

    Takes over the transformer's block loop to insert residual injection points.
    The inner transformer must already have its PatchEmbed expanded to 8 channels.
    """

    def __init__(self, transformer: nn.Module, injection_interval: int = 4):
        super().__init__()
        self.transformer = transformer
        self.injection_interval = injection_interval

        inner_dim = transformer.inner_dim
        ps = transformer.config.patch_size
        num_layers = transformer.config.num_layers
        num_injections = num_layers // injection_interval

        self.ref_proj = nn.Conv2d(4, inner_dim, kernel_size=ps, stride=ps)
        self.ref_norm = nn.LayerNorm(inner_dim)

        self.injection_gates = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(inner_dim),
                nn.Linear(inner_dim, inner_dim),
            )
            for _ in range(num_injections)
        ])

        self.null_embed = nn.Parameter(torch.zeros(1, 1, inner_dim))

        self._zero_init_new_modules()

        logger.info(
            f"NativeImg2ImgPixArtWrapper: inner_dim={inner_dim}, "
            f"num_layers={num_layers}, injection_interval={injection_interval}, "
            f"num_injections={num_injections}"
        )

    def _zero_init_new_modules(self):
        nn.init.zeros_(self.ref_proj.weight)
        nn.init.zeros_(self.ref_proj.bias)
        for gate in self.injection_gates:
            nn.init.zeros_(gate[1].weight)
            nn.init.zeros_(gate[1].bias)

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
        noisy_latent: torch.Tensor,
        reference_latent: torch.Tensor,
        timestep: torch.Tensor,
        added_cond_kwargs: dict[str, torch.Tensor] | None = None,
        return_dict: bool = True,
    ) -> Transformer2DModelOutput | tuple:
        """
        Args:
            noisy_latent:      (B, 4, H, W) noisy target latent
            reference_latent:  (B, 4, H, W) clean reference (color block map) latent
            timestep:          (B,) scaled timestep values
            added_cond_kwargs: resolution/aspect_ratio conditions for adaln_single
            return_dict:       whether to wrap output in Transformer2DModelOutput
        """
        tf = self.transformer
        ps = tf.config.patch_size

        # 1. Concatenate input channels
        hidden_states = torch.cat([noisy_latent, reference_latent], dim=1)

        # 2. Compute reference tokens for injection
        ref_tokens = self.ref_proj(reference_latent)
        ref_tokens = ref_tokens.flatten(2).transpose(1, 2)
        ref_tokens = self.ref_norm(ref_tokens)

        # 3. PatchEmbed (8-ch input)
        batch_size = hidden_states.shape[0]
        height = hidden_states.shape[-2] // ps
        width = hidden_states.shape[-1] // ps
        hidden_states = tf.pos_embed(hidden_states)

        # 4. Timestep conditioning
        timestep_emb, embedded_timestep = tf.adaln_single(
            timestep, added_cond_kwargs,
            batch_size=batch_size, hidden_dtype=hidden_states.dtype,
        )

        # 5. Null cross-attention (T5 discarded)
        encoder_hidden_states = self.null_embed.expand(batch_size, -1, -1)
        encoder_attention_mask = None

        # 6. Transformer blocks with residual injection
        injection_idx = 0
        for i, block in enumerate(tf.transformer_blocks):
            if torch.is_grad_enabled() and tf.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    None,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep_emb,
                    None,
                    None,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep_emb,
                    cross_attention_kwargs=None,
                    class_labels=None,
                )

            if (i + 1) % self.injection_interval == 0 and injection_idx < len(self.injection_gates):
                hidden_states = hidden_states + self.injection_gates[injection_idx](ref_tokens)
                injection_idx += 1

        # 7. Output head (norm + modulation + proj_out + unpatchify)
        shift, scale = (
            tf.scale_shift_table[None]
            + embedded_timestep[:, None].to(tf.scale_shift_table.device)
        ).chunk(2, dim=1)
        hidden_states = tf.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale.to(hidden_states.device)) + shift.to(hidden_states.device)
        hidden_states = tf.proj_out(hidden_states)
        hidden_states = hidden_states.squeeze(1)

        hidden_states = hidden_states.reshape(
            -1, height, width, ps, ps, tf.out_channels
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            -1, tf.out_channels, height * ps, width * ps
        )

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    @staticmethod
    def _gradient_checkpointing_func(block, *args):
        """Adapter for torch.utils.checkpoint that works with the block's forward."""
        def _forward(*inputs):
            return block(
                inputs[0],
                attention_mask=inputs[1],
                encoder_hidden_states=inputs[2],
                encoder_attention_mask=inputs[3],
                timestep=inputs[4],
                cross_attention_kwargs=inputs[5],
                class_labels=inputs[6],
            )
        return checkpoint(_forward, *args, use_reentrant=False)

    def get_new_module_state_dict(self) -> dict:
        """Extract state dict for newly added modules only (for checkpoint saving)."""
        state = {}
        for k, v in self.ref_proj.state_dict().items():
            state[f"ref_proj.{k}"] = v
        for k, v in self.ref_norm.state_dict().items():
            state[f"ref_norm.{k}"] = v
        for i, gate in enumerate(self.injection_gates):
            for k, v in gate.state_dict().items():
                state[f"injection_gates.{i}.{k}"] = v
        state["null_embed"] = self.null_embed.data
        return state

    def load_new_module_state_dict(self, state_dict: dict) -> None:
        """Load state dict for newly added modules."""
        ref_proj_state = {k.replace("ref_proj.", ""): v for k, v in state_dict.items() if k.startswith("ref_proj.")}
        if ref_proj_state:
            self.ref_proj.load_state_dict(ref_proj_state)

        ref_norm_state = {k.replace("ref_norm.", ""): v for k, v in state_dict.items() if k.startswith("ref_norm.")}
        if ref_norm_state:
            self.ref_norm.load_state_dict(ref_norm_state)

        for i, gate in enumerate(self.injection_gates):
            prefix = f"injection_gates.{i}."
            gate_state = {k.replace(prefix, ""): v for k, v in state_dict.items() if k.startswith(prefix)}
            if gate_state:
                gate.load_state_dict(gate_state)

        if "null_embed" in state_dict:
            self.null_embed.data.copy_(state_dict["null_embed"])


def build_native_img2img_model(
    transformer: nn.Module,
    injection_interval: int = 4,
) -> NativeImg2ImgPixArtWrapper:
    """Perform model surgery and build the Native Img2Img wrapper.

    1. Expands PatchEmbed from 4→8 channels (zero-init new channels)
    2. Wraps transformer with residual injection modules
    """
    expand_patch_embed_to_8ch(transformer)
    wrapper = NativeImg2ImgPixArtWrapper(transformer, injection_interval)
    return wrapper
