"""PixArt-Sigma ControlNet-XS — 轻量级条件控制适配器。

ControlNet-XS 核心特点:
  1. 薄型控制分支 — control_dim = base_dim * size_ratio (~4.7% 参数)
  2. 双向信息流 — base→control 投射 + control→base 零初始化反馈
  3. 无文本 cross-attention — 文本语义由 base 分支处理，经投射间接传递
  4. 单次联合前向传播 — adapter 融合进 base transformer 循环

核心组件:
  PixArtControlNetXSConditionEncoder — 专用 CNN 编码器 (直接输出 control_dim tokens)
  PixArtControlNetXSBlock            — 薄型 transformer block (self-attn + FFN, 无 cross-attn)
  PixArtControlNetXSAdapter          — 完整 adapter (XS blocks + 双向投射 + 条件编码)
  PixArtControlNetXSTransformerModel — 包装 frozen transformer + trainable adapter 的联合前向
"""

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import PixArtTransformer2DModel
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import get_2d_sincos_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormSingle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Condition Encoder
# ---------------------------------------------------------------------------

class PixArtControlNetXSConditionEncoder(nn.Module):
    """专用 CNN 条件图编码器 — 直接映射到 (B, N, control_dim) token 序列。

    流程: pixel (B,3,H,W) → CNN backbone 8x下采样 → patch_proj → (B,N,control_dim)
    patch_proj 零初始化确保训练初期不干扰 base 模型。
    """

    def __init__(
        self,
        in_channels: int = 3,
        control_dim: int = 288,
        patch_size: int = 2,
        mid_channels: int = 64,
    ):
        super().__init__()
        self.patch_size = patch_size

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, mid_channels, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )

        self.patch_proj = nn.Conv2d(
            mid_channels, control_dim,
            kernel_size=patch_size, stride=patch_size,
        )
        nn.init.zeros_(self.patch_proj.weight)
        nn.init.zeros_(self.patch_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) → (B, N, control_dim)"""
        x = self.backbone(x)
        x = self.patch_proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x


# ---------------------------------------------------------------------------
# XS Transformer Block (self-attn + FFN, no cross-attn)
# ---------------------------------------------------------------------------

class _FeedForward(nn.Module):
    """Simple GELU feed-forward matching PixArt's default activation."""

    def __init__(self, dim: int, inner_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        inner_dim = inner_dim or dim * 4
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PixArtControlNetXSBlock(nn.Module):
    """薄型 transformer block — self-attention + FFN, AdaLN-Single 时间步调制。

    与 base BasicTransformerBlock(ada_norm_single) 同构但:
      - dim 更小 (control_dim)
      - 无 cross-attention (文本语义经 base→control 投射间接获取)
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        attention_bias: bool = True,
        num_embeds_ada_norm: int = 1000,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.gradient_checkpointing = False

        # AdaLN-Single: 6 modulation values (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

        # Self-Attention
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
        )

        # Feed-Forward
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.ff = _FeedForward(dim, dropout=dropout)

    def _forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
        ).chunk(6, dim=1)

        # Self-Attention
        norm_hidden = self.norm1(hidden_states)
        norm_hidden = norm_hidden * (1 + scale_msa) + shift_msa
        attn_output = self.attn1(norm_hidden)
        attn_output = gate_msa * attn_output
        hidden_states = hidden_states + attn_output

        # Feed-Forward
        norm_hidden = self.norm2(hidden_states)
        norm_hidden = norm_hidden * (1 + scale_mlp) + shift_mlp
        ff_output = self.ff(norm_hidden)
        ff_output = gate_mlp * ff_output
        hidden_states = hidden_states + ff_output

        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, N, control_dim)
            timestep: (B, 6*control_dim) from control_adaln_single
        """
        if self.training and self.gradient_checkpointing:
            return gradient_checkpoint(self._forward, hidden_states, timestep, use_reentrant=False)
        return self._forward(hidden_states, timestep)


# ---------------------------------------------------------------------------
# Adapter Model
# ---------------------------------------------------------------------------

class PixArtControlNetXSAdapter(ModelMixin, ConfigMixin):
    """PixArt ControlNet-XS 适配器。

    包含 N 个 XS blocks + 双向投射层 + 独立时间步/条件编码。
    所有 control→base 投射零初始化，确保训练初期不干扰 base 模型。
    """

    @register_to_config
    def __init__(
        self,
        num_layers: int = 14,
        size_ratio: float = 0.25,
        base_dim: int = 1152,
        base_num_attention_heads: int = 16,
        base_attention_head_dim: int = 72,
        patch_size: int = 2,
        sample_size: int = 128,
        in_channels: int = 4,
        conditioning_mode: str = "vae",
        connection_interval: int = 2,
        num_embeds_ada_norm: int = 1000,
        use_additional_conditions: bool = True,
        interpolation_scale: float = 2.0,
        pos_embed_max_size: int = 192,
        dropout: float = 0.0,
        attention_bias: bool = True,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.size_ratio = size_ratio
        self.base_dim = base_dim
        self.conditioning_mode = conditioning_mode
        self.connection_interval = connection_interval
        self.pos_embed_max_size = pos_embed_max_size

        control_dim = int(base_dim * size_ratio)
        self.control_dim = control_dim
        xs_heads = max(1, int(base_num_attention_heads * size_ratio))
        xs_head_dim = control_dim // xs_heads

        # --- Condition encoding ---
        self.condition_encoder = None
        self.control_patch_embed = None
        if conditioning_mode == "cnn_encoder":
            self.condition_encoder = PixArtControlNetXSConditionEncoder(
                in_channels=3, control_dim=control_dim, patch_size=patch_size,
            )
        else:
            self.control_patch_embed = nn.Conv2d(
                in_channels, control_dim,
                kernel_size=patch_size, stride=patch_size,
            )

        # --- Position embedding (2D sinusoidal, same approach as base PatchEmbed) ---
        base_size = sample_size // patch_size
        pos_embed = get_2d_sincos_pos_embed(
            control_dim,
            pos_embed_max_size,
            base_size=base_size,
            interpolation_scale=interpolation_scale,
            output_type="pt",
        )
        self.register_buffer(
            "control_pos_embed",
            pos_embed.float().unsqueeze(0),
            persistent=True,
        )

        # --- Time-step embedding (independent from base) ---
        self.control_adaln_single = AdaLayerNormSingle(
            control_dim, use_additional_conditions=use_additional_conditions,
        )

        # --- Bidirectional projections ---
        self.base_to_control_projs = nn.ModuleList()
        self.control_to_base_projs = nn.ModuleList()
        for _ in range(num_layers):
            self.base_to_control_projs.append(nn.Linear(base_dim, control_dim))

            c2b = nn.Linear(control_dim, base_dim)
            nn.init.zeros_(c2b.weight)
            nn.init.zeros_(c2b.bias)
            self.control_to_base_projs.append(c2b)

        # --- XS transformer blocks ---
        self.xs_blocks = nn.ModuleList([
            PixArtControlNetXSBlock(
                dim=control_dim,
                num_attention_heads=xs_heads,
                attention_head_dim=xs_head_dim,
                dropout=dropout,
                attention_bias=attention_bias,
                num_embeds_ada_norm=num_embeds_ada_norm,
                norm_elementwise_affine=norm_elementwise_affine,
                norm_eps=norm_eps,
            )
            for _ in range(num_layers)
        ])

    def cropped_pos_embed(self, height: int, width: int) -> torch.Tensor:
        """从预计算的 max-size 位置编码中心裁剪到目标 (height, width)。

        与 base PatchEmbed.cropped_pos_embed 保持一致。
        """
        if height > self.pos_embed_max_size or width > self.pos_embed_max_size:
            raise ValueError(
                f"Spatial size ({height}×{width}) exceeds pos_embed_max_size ({self.pos_embed_max_size})."
            )
        top = (self.pos_embed_max_size - height) // 2
        left = (self.pos_embed_max_size - width) // 2
        spatial = self.control_pos_embed.reshape(
            1, self.pos_embed_max_size, self.pos_embed_max_size, -1,
        )
        spatial = spatial[:, top : top + height, left : left + width, :]
        return spatial.reshape(1, -1, spatial.shape[-1])

    @classmethod
    def from_transformer(
        cls,
        transformer: PixArtTransformer2DModel,
        num_layers: int = 14,
        size_ratio: float = 0.25,
        conditioning_mode: str = "vae",
        connection_interval: int = 2,
    ) -> "PixArtControlNetXSAdapter":
        """从预训练 PixArt Transformer 提取配置并初始化 adapter。"""
        cfg = transformer.config
        pos_embed_max_size = 192 if cfg.sample_size > 64 else 96
        adapter = cls(
            num_layers=num_layers,
            size_ratio=size_ratio,
            base_dim=cfg.num_attention_heads * cfg.attention_head_dim,
            base_num_attention_heads=cfg.num_attention_heads,
            base_attention_head_dim=cfg.attention_head_dim,
            patch_size=cfg.patch_size,
            sample_size=cfg.sample_size,
            in_channels=cfg.in_channels,
            conditioning_mode=conditioning_mode,
            connection_interval=connection_interval,
            num_embeds_ada_norm=cfg.num_embeds_ada_norm,
            use_additional_conditions=transformer.use_additional_conditions,
            interpolation_scale=cfg.interpolation_scale,
            pos_embed_max_size=pos_embed_max_size,
            dropout=cfg.dropout,
            attention_bias=cfg.attention_bias,
            norm_elementwise_affine=cfg.norm_elementwise_affine,
            norm_eps=cfg.norm_eps,
        )

        n_params = sum(p.numel() for p in adapter.parameters())
        n_base = sum(p.numel() for p in transformer.parameters())
        logger.info(
            f"PixArt ControlNet-XS adapter initialized: "
            f"{num_layers} blocks, control_dim={adapter.control_dim}, "
            f"mode={conditioning_mode}, interval={connection_interval}, "
            f"params={n_params:,} ({n_params/n_base*100:.1f}% of base)"
        )
        return adapter

    def enable_gradient_checkpointing(self):
        for block in self.xs_blocks:
            block.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        for block in self.xs_blocks:
            block.gradient_checkpointing = False


# ---------------------------------------------------------------------------
# Joint Forward Model
# ---------------------------------------------------------------------------

class PixArtControlNetXSTransformerModel(ModelMixin, ConfigMixin):
    """联合前向传播模型 — frozen base transformer + trainable ControlNet-XS adapter。

    前向流程:
      1. Base PatchEmbed → base_hidden (B, N, base_dim)
      2. 条件编码 → ctrl_hidden (B, N, control_dim)
      3. 遍历 base blocks:
         在连接点 (i % interval == 0):
           ctrl = base_to_ctrl(base_hidden) + ctrl_hidden
           ctrl_hidden = xs_block(ctrl, timestep)
           base_hidden += ctrl_to_base(ctrl_hidden) * scale
         base_hidden = base_block(base_hidden, text, timestep)
      4. Output: norm + proj_out + unpatchify
    """

    def __init__(
        self,
        transformer: PixArtTransformer2DModel,
        controlnet: PixArtControlNetXSAdapter,
    ):
        super().__init__()
        self.register_to_config(**transformer.config)
        self.transformer = transformer
        self.controlnet = controlnet

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        controlnet_cond: Optional[torch.Tensor] = None,
        controlnet_conditioning_scale: float = 1.0,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        """联合前向传播 — 与 PixArtControlNetTransformerModel 签名一致。

        Args:
            hidden_states: noisy latent (B, C, H, W)
            controlnet_cond: 条件输入
                VAE 模式: (B, 4, H/8, W/8) latent
                CNN 模式: (B, 3, H, W) 像素图
            controlnet_conditioning_scale: 控制残差强度 (推理可调, 训练固定 1.0)
        """
        if self.transformer.use_additional_conditions and added_cond_kwargs is None:
            raise ValueError(
                "`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`."
            )

        model_dtype = next(self.transformer.parameters()).dtype
        hidden_states = hidden_states.to(dtype=model_dtype)
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.to(dtype=model_dtype)
        if controlnet_cond is not None:
            controlnet_cond = controlnet_cond.to(dtype=model_dtype)

        # --- Attention mask → bias ---
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        batch_size = hidden_states.shape[0]
        height, width = (
            hidden_states.shape[-2] // self.transformer.config.patch_size,
            hidden_states.shape[-1] // self.transformer.config.patch_size,
        )

        # ============================================================
        # Phase 1: Input embedding
        # ============================================================

        # Base path
        hidden_states = self.transformer.pos_embed(hidden_states)

        base_timestep_emb, base_embedded_timestep = self.transformer.adaln_single(
            timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype
        )

        if self.transformer.caption_projection is not None:
            encoder_hidden_states = self.transformer.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])

        # Control path — condition encoding + position embedding
        ctrl_hidden = None
        if controlnet_cond is not None:
            if self.controlnet.conditioning_mode == "cnn_encoder" and self.controlnet.condition_encoder is not None:
                ctrl_hidden = self.controlnet.condition_encoder(controlnet_cond)
            else:
                ctrl_hidden = self.controlnet.control_patch_embed(controlnet_cond)
                B, C, cH, cW = ctrl_hidden.shape
                ctrl_hidden = ctrl_hidden.flatten(2).transpose(1, 2)

            ctrl_pos_embed = self.controlnet.cropped_pos_embed(height, width)
            ctrl_hidden = ctrl_hidden + ctrl_pos_embed.to(ctrl_hidden.dtype)

        # Control timestep embedding
        ctrl_timestep_emb, _ = self.controlnet.control_adaln_single(
            timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype
        )

        # ============================================================
        # Phase 2: Transformer blocks + XS adapter
        # ============================================================

        interval = self.controlnet.connection_interval
        xs_idx = 0
        num_xs = self.controlnet.num_layers

        for block_index, block in enumerate(self.transformer.transformer_blocks):
            # --- Connection point: bidirectional exchange ---
            if (block_index % interval == 0
                    and xs_idx < num_xs
                    and ctrl_hidden is not None):

                # Base → Control injection
                ctrl_input = self.controlnet.base_to_control_projs[xs_idx](hidden_states) + ctrl_hidden

                # XS block forward (self-attn + FFN only)
                ctrl_hidden = self.controlnet.xs_blocks[xs_idx](ctrl_input, ctrl_timestep_emb)

                # Control → Base feedback (zero-init)
                base_residual = self.controlnet.control_to_base_projs[xs_idx](ctrl_hidden)
                hidden_states = hidden_states + base_residual * controlnet_conditioning_scale

                xs_idx += 1

            # --- Base block forward ---
            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=base_timestep_emb,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=None,
            )

        # ============================================================
        # Phase 3: Output
        # ============================================================

        shift, scale = (
            self.transformer.scale_shift_table[None]
            + base_embedded_timestep[:, None].to(self.transformer.scale_shift_table.device)
        ).chunk(2, dim=1)
        hidden_states = self.transformer.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale.to(hidden_states.device)) + shift.to(hidden_states.device)
        hidden_states = self.transformer.proj_out(hidden_states)
        hidden_states = hidden_states.squeeze(1)

        # unpatchify
        hidden_states = hidden_states.reshape(
            -1, height, width,
            self.transformer.config.patch_size,
            self.transformer.config.patch_size,
            self.transformer.out_channels,
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            -1,
            self.transformer.out_channels,
            height * self.transformer.config.patch_size,
            width * self.transformer.config.patch_size,
        )

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
