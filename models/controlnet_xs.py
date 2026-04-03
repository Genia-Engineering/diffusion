"""ControlNet-XS 模型 — 基于 diffusers 的 ControlNetXSAdapter + UNetControlNetXSModel。

ControlNet-XS 核心特点:
  1. 仅使用 base model ~1% 的参数（SDXL: ~48M）
  2. 推理速度比标准 ControlNet 快 20-25%，显存减少 ~45%
  3. adapter 与 UNet 融合为 UNetControlNetXSModel 进行联合前向传播

本模块封装初始化逻辑，从已有 UNet 构建 ControlNet-XS。

可选 strip_ctrl_cross_attention():
  移除 ctrl 分支的 cross-attention，使其仅依赖图像信息。
  文本语义通过 base→ctrl 零卷积投射间接传递（与 PixArt-Sigma XS 设计一致）。
"""

import gc
import logging
from typing import Optional

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from diffusers.models.controlnets.controlnet_xs import (
    ControlNetXSAdapter,
    UNetControlNetXSModel,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 辅助类 & 函数 — 移除 ctrl 分支 cross-attention
# ---------------------------------------------------------------------------

class _ResnetOnlyMidBlock(nn.Module):
    """UNetMidBlock2DCrossAttn 的精简版 — 只保留 ResNet blocks，去掉 attention。

    替换 ctrl_midblock 后，ctrl 分支在 mid stage 只做卷积特征变换，
    不再直接处理文本嵌入。
    """

    def __init__(self, midblock: nn.Module):
        super().__init__()
        self.resnets = midblock.resnets

    def forward(self, hidden_states, temb=None, **kwargs):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
        return hidden_states


def strip_ctrl_cross_attention(unet_xs: UNetControlNetXSModel) -> int:
    """移除 UNetControlNetXSModel 中 ctrl 分支的所有 cross-attention。

    修改范围:
      - Down blocks: ctrl_attentions (ModuleList) → [None, ...]，参数释放
      - Mid block:   ctrl_midblock (UNetMidBlock2DCrossAttn) → _ResnetOnlyMidBlock

    修改后 ctrl 分支仅保留:
      - ResNet blocks (卷积 + GroupNorm + 时间步调制)
      - base_to_ctrl / ctrl_to_base 零卷积投射
      - Downsamplers

    Returns:
        被移除的参数总量。
    """
    removed = 0

    # --- Down blocks ---
    for block in unet_xs.down_blocks:
        ctrl_attns = getattr(block, "ctrl_attentions", None)
        if isinstance(ctrl_attns, nn.ModuleList):
            removed += sum(p.numel() for p in ctrl_attns.parameters())
            n = len(ctrl_attns)
            del block.ctrl_attentions
            block.ctrl_attentions = [None] * n

    # --- Mid block ---
    mid = unet_xs.mid_block
    if hasattr(mid, "ctrl_midblock"):
        old_mid = mid.ctrl_midblock
        if hasattr(old_mid, "attentions"):
            removed += sum(
                p.numel() for attn in old_mid.attentions for p in attn.parameters()
            )
        mid.ctrl_midblock = _ResnetOnlyMidBlock(old_mid)

    gc.collect()
    return removed


def create_controlnet_xs_sdxl(
    pretrained_model_name_or_path: str,
    conditioning_channels: int = 3,
    size_ratio: float | None = None,
    unet: Optional[UNet2DConditionModel] = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[ControlNetXSAdapter, UNetControlNetXSModel]:
    """从 SDXL UNet 初始化 ControlNet-XS。

    Args:
        pretrained_model_name_or_path: 预训练模型路径
        conditioning_channels: 条件图像通道数
        size_ratio: adapter 与 base model 的参数比例，None 使用 diffusers 默认值
        unet: 已加载的 UNet 实例（如融合 LoRA 后）；None 则从路径加载
        dtype: 加载 UNet 时使用的精度（仅在无 unet 传入时生效）

    Returns:
        (adapter, unet_xs):
          adapter — ControlNetXSAdapter，用于保存/加载 checkpoint
          unet_xs — UNetControlNetXSModel，融合后用于训练前向传播
    """
    should_cleanup = False
    if unet is None:
        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="unet", torch_dtype=dtype,
        )
        should_cleanup = True

    adapter_kwargs = {"conditioning_channels": conditioning_channels}
    adapter_kwargs["size_ratio"] = size_ratio if size_ratio is not None else 0.1

    adapter = ControlNetXSAdapter.from_unet(unet, **adapter_kwargs)
    unet_xs = UNetControlNetXSModel.from_unet(unet, controlnet=adapter)

    if should_cleanup:
        del unet

    n_adapter = sum(p.numel() for p in adapter.parameters())
    n_total = sum(p.numel() for p in unet_xs.parameters())
    logger.info(
        f"ControlNet-XS initialized: adapter={n_adapter/1e6:.1f}M, "
        f"total={n_total/1e6:.1f}M ({n_adapter/n_total*100:.1f}% adapter)"
    )

    return adapter, unet_xs


def create_controlnet_xs_sd15(
    pretrained_model_name_or_path: str,
    conditioning_channels: int = 3,
    size_ratio: float | None = None,
    unet: Optional[UNet2DConditionModel] = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[ControlNetXSAdapter, UNetControlNetXSModel]:
    """从 SD1.5 UNet 初始化 ControlNet-XS。"""
    should_cleanup = False
    if unet is None:
        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="unet", torch_dtype=dtype,
        )
        should_cleanup = True

    adapter_kwargs = {"conditioning_channels": conditioning_channels}
    adapter_kwargs["size_ratio"] = size_ratio if size_ratio is not None else 0.1

    adapter = ControlNetXSAdapter.from_unet(unet, **adapter_kwargs)
    unet_xs = UNetControlNetXSModel.from_unet(unet, controlnet=adapter)

    if should_cleanup:
        del unet

    n_adapter = sum(p.numel() for p in adapter.parameters())
    n_total = sum(p.numel() for p in unet_xs.parameters())
    logger.info(
        f"ControlNet-XS (SD1.5) initialized: adapter={n_adapter/1e6:.1f}M, "
        f"total={n_total/1e6:.1f}M ({n_adapter/n_total*100:.1f}% adapter)"
    )

    return adapter, unet_xs
