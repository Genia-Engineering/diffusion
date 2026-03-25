"""PixArt-Sigma ControlNet — 基于 Transformer 的条件控制适配器。

架构参考 diffusers research project (examples/research_projects/pixart/controlnet_pixart_alpha.py)，
适配到本项目训练框架并扩展双模式条件图编码。

核心组件:
  PixArtControlNetConditionEncoder — 轻量 CNN 条件图编码器（cnn_encoder 模式专用）
  PixArtControlNetAdapterBlock     — 单个 adapter 块（zero-init proj + transformer block）
  PixArtControlNetAdapterModel     — 包含 N 个 adapter 块，支持 save/load pretrained
  PixArtControlNetTransformerModel — 包装 frozen transformer + trainable adapter 的联合前向

条件图编码双模式 (conditioning_mode):
  "vae"         — 条件图通过 VAE encode 到 latent 空间（论文默认，可预缓存）
  "cnn_encoder" — 条件图通过轻量 CNN 映射到 latent 维度（保留精确边界，在线计算）
"""

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import PixArtTransformer2DModel
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin

logger = logging.getLogger(__name__)


class PixArtControlNetConditionEncoder(nn.Module):
    """轻量 CNN 条件图编码器 — 将像素空间条件图映射到 VAE latent 同维度。

    输入: (B, 3, H, W) [0, 1] 条件图像
    输出: (B, out_channels, H/8, W/8) 与 VAE latent 同维度

    3 层 stride-2 卷积实现 8x 空间下采样，输出卷积零初始化确保训练初期不干扰模型。
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.conv_out = nn.Conv2d(128, out_channels, kernel_size=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_out(self.encoder(x))


class PixArtControlNetAdapterBlock(nn.Module):
    """单个 ControlNet adapter 块。

    结构:
      [before_proj (zero-init, 仅 block_index=0)] → transformer_block → after_proj (zero-init)

    before_proj 仅在第一个块存在，用于将 transformer 的 hidden_states 混入 controlnet 分支。
    after_proj 输出残差，零初始化保证训练初期 adapter 不影响主 transformer。
    """

    def __init__(
        self,
        block_index: int,
        num_attention_heads: int = 16,
        attention_head_dim: int = 72,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = 1152,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm_single",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.block_index = block_index
        self.inner_dim = num_attention_heads * attention_head_dim

        if self.block_index == 0:
            self.before_proj = nn.Linear(self.inner_dim, self.inner_dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)

        self.transformer_block = BasicTransformerBlock(
            self.inner_dim,
            num_attention_heads,
            attention_head_dim,
            dropout=dropout,
            cross_attention_dim=cross_attention_dim,
            activation_fn=activation_fn,
            num_embeds_ada_norm=num_embeds_ada_norm,
            attention_bias=attention_bias,
            upcast_attention=upcast_attention,
            norm_type=norm_type,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
        )

        self.after_proj = nn.Linear(self.inner_dim, self.inner_dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        controlnet_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ):
        if self.block_index == 0:
            controlnet_states = self.before_proj(controlnet_states)
            controlnet_states = hidden_states + controlnet_states

        controlnet_states_down = self.transformer_block(
            hidden_states=controlnet_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            added_cond_kwargs=added_cond_kwargs,
            cross_attention_kwargs=cross_attention_kwargs,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            class_labels=None,
        )

        controlnet_states_left = self.after_proj(controlnet_states_down)
        return controlnet_states_left, controlnet_states_down


class PixArtControlNetAdapterModel(ModelMixin, ConfigMixin):
    """PixArt ControlNet 适配器 — 包含 N 个 adapter 块 + 可选 CNN 编码器。

    Args:
        num_layers: adapter 块数量（论文默认 13）
        conditioning_mode: "vae" 或 "cnn_encoder"
        conditioning_in_channels: CNN 编码器输入通道数（仅 cnn_encoder 模式）
        conditioning_out_channels: CNN 编码器输出通道数（需匹配 VAE latent channels）
    """

    @register_to_config
    def __init__(
        self,
        num_layers: int = 13,
        conditioning_mode: str = "vae",
        conditioning_in_channels: int = 3,
        conditioning_out_channels: int = 4,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.conditioning_mode = conditioning_mode

        self.controlnet_blocks = nn.ModuleList(
            [PixArtControlNetAdapterBlock(block_index=i) for i in range(num_layers)]
        )

        self.condition_encoder = None
        if conditioning_mode == "cnn_encoder":
            self.condition_encoder = PixArtControlNetConditionEncoder(
                in_channels=conditioning_in_channels,
                out_channels=conditioning_out_channels,
            )

    @classmethod
    def from_transformer(
        cls,
        transformer: PixArtTransformer2DModel,
        num_layers: int = 13,
        conditioning_mode: str = "vae",
    ) -> "PixArtControlNetAdapterModel":
        """从预训练 PixArt Transformer 初始化 adapter（复制前 N 层 transformer block 权重）。"""
        adapter = cls(num_layers=num_layers, conditioning_mode=conditioning_mode)

        for depth in range(min(num_layers, len(transformer.transformer_blocks))):
            adapter.controlnet_blocks[depth].transformer_block.load_state_dict(
                transformer.transformer_blocks[depth].state_dict()
            )

        n_params = sum(p.numel() for p in adapter.parameters())
        n_trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
        logger.info(
            f"PixArt ControlNet adapter initialized from transformer: "
            f"{num_layers} blocks, mode={conditioning_mode}, "
            f"params={n_params:,} (trainable={n_trainable:,})"
        )
        return adapter

    def enable_gradient_checkpointing(self):
        for block in self.controlnet_blocks:
            block.transformer_block.gradient_checkpointing = True


class PixArtControlNetTransformerModel(ModelMixin, ConfigMixin):
    """联合前向传播模型 — 包装 frozen transformer + trainable ControlNet adapter。

    训练时:
      - transformer 的参数全冻结（不参与梯度计算）
      - adapter 的参数可训练
      - 条件 latent 通过 pos_embed 后送入 adapter 块
      - adapter 产生的残差（zero-init）累加到 transformer 的 hidden states

    Args:
        transformer: 预训练 PixArtTransformer2DModel（冻结）
        controlnet: PixArtControlNetAdapterModel（可训练）
    """

    def __init__(
        self,
        transformer: PixArtTransformer2DModel,
        controlnet: PixArtControlNetAdapterModel,
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
        """联合前向传播。

        Args:
            hidden_states: 含噪 latent (B, C, H, W)
            controlnet_cond: 条件输入
                - VAE 模式: (B, 4, H/8, W/8) 已编码的 latent
                - CNN 模式: (B, 3, H, W) 像素空间条件图
            controlnet_conditioning_scale: 控制 adapter 残差强度（推理时可调，训练时固定 1.0）
            encoder_hidden_states: T5 文本嵌入 (B, seq_len, dim)
            timestep: 时间步 (B,)
        """
        if self.transformer.use_additional_conditions and added_cond_kwargs is None:
            raise ValueError(
                "`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`."
            )

        # attention mask → bias 转换
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

        # 1. 输入嵌入
        hidden_states = self.transformer.pos_embed(hidden_states)

        timestep_emb, embedded_timestep = self.transformer.adaln_single(
            timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype
        )

        if self.transformer.caption_projection is not None:
            encoder_hidden_states = self.transformer.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])

        # 2. 条件图编码
        controlnet_states_down = None
        if controlnet_cond is not None:
            if self.controlnet.conditioning_mode == "cnn_encoder" and self.controlnet.condition_encoder is not None:
                controlnet_cond = self.controlnet.condition_encoder(controlnet_cond)
            controlnet_states_down = self.transformer.pos_embed(controlnet_cond)

        # 3. Transformer blocks + ControlNet adapter
        num_adapter_blocks = self.controlnet.num_layers
        for block_index, block in enumerate(self.transformer.transformer_blocks):
            # adapter 块在 block 1~N 处插入残差
            if block_index > 0 and block_index <= num_adapter_blocks and controlnet_states_down is not None:
                controlnet_states_left, controlnet_states_down = self.controlnet.controlnet_blocks[
                    block_index - 1
                ](
                    hidden_states=hidden_states,
                    controlnet_states=controlnet_states_down,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep_emb,
                    added_cond_kwargs=added_cond_kwargs,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
                hidden_states = hidden_states + controlnet_states_left * controlnet_conditioning_scale

            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep_emb,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=None,
            )

        # 4. 输出
        shift, scale = (
            self.transformer.scale_shift_table[None]
            + embedded_timestep[:, None].to(self.transformer.scale_shift_table.device)
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
