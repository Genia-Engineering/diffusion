"""ControlNet 模型 — 基于 diffusers 的 ControlNetModel。

ControlNet 架构核心:
  1. Trainable Copy: 复制 UNet 的 encoder + mid block 作为可训练分支
  2. Zero Convolution: 每个分支输出通过零初始化的 1x1 卷积连接到 UNet

本模块封装初始化逻辑，从已有 UNet 构建 ControlNet。
支持从预训练路径或已加载的 UNet 实例（如融合 LoRA 后的 UNet）创建。

conv_zero_init 选项:
  "zero"    — 原始论文做法，zero conv，训练初期完全无影响（默认）
  "normal"  — N(0, std) 初始化，训练开始即有微弱结构引导；std 由 conv_init_std 控制
  "xavier"  — Xavier uniform 初始化，适配 sigmoid/tanh 激活
  "kaiming" — Kaiming normal 初始化，适配 ReLU 类激活
"""

import logging
from typing import Literal, Optional

import torch
import torch.nn as nn
from diffusers import ControlNetModel, UNet2DConditionModel

logger = logging.getLogger(__name__)

ConvInitType = Literal["zero", "normal", "xavier", "kaiming"]


class ZeroConv(nn.Module):
    """零初始化卷积层 — ControlNet 的关键组件。

    权重与偏置均初始化为零，确保训练开始时 ControlNet 不影响 UNet 输出。
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _reinit_conv_layers(
    controlnet: ControlNetModel,
    init_type: ConvInitType,
    std: float = 0.02,
) -> None:
    """将 ControlNet 的 zero conv 层替换为指定初始化方式。

    目标层:
      - controlnet_down_blocks: 各下采样块到 UNet 的残差连接卷积
      - controlnet_mid_block:   瓶颈层到 UNet 的残差连接卷积
      - controlnet_cond_embedding.conv_out: 条件图像嵌入的输出卷积

    Args:
        controlnet: 已通过 from_unet() 创建的 ControlNetModel（此时为 zero init）
        init_type:  "normal" | "xavier" | "kaiming"（传入 "zero" 时本函数不应被调用）
        std:        仅 normal 模式使用的标准差（默认 0.02）
    """

    def _apply(conv_module: nn.Module) -> None:
        for m in conv_module.modules():
            if isinstance(m, nn.Conv2d):
                if init_type == "normal":
                    nn.init.normal_(m.weight, mean=0.0, std=std)
                elif init_type == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                elif init_type == "kaiming":
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # zero conv 连接层
    for block in controlnet.controlnet_down_blocks:
        _apply(block)
    _apply(controlnet.controlnet_mid_block)

    # 条件图像嵌入输出层
    cond_emb = getattr(controlnet, "controlnet_cond_embedding", None)
    if cond_emb is not None and hasattr(cond_emb, "conv_out"):
        _apply(cond_emb.conv_out)

    n_down = len(controlnet.controlnet_down_blocks)
    logger.info(
        f"ControlNet zero conv reinitialized: {init_type} "
        f"(down_blocks={n_down}, mid_block=1, cond_emb={'yes' if cond_emb else 'no'})"
    )


def _create_controlnet_from_unet(
    unet: UNet2DConditionModel,
    conditioning_channels: int = 3,
    conv_zero_init: ConvInitType = "zero",
    conv_init_std: float = 0.02,
) -> ControlNetModel:
    """从已加载的 UNet 实例创建 ControlNet（通用内部方法）。"""
    controlnet = ControlNetModel.from_unet(unet, conditioning_channels=conditioning_channels)
    if conv_zero_init != "zero":
        _reinit_conv_layers(controlnet, conv_zero_init, std=conv_init_std)
    return controlnet


def create_controlnet_sd15(
    pretrained_model_name_or_path: str,
    conditioning_channels: int = 3,
    unet: Optional[UNet2DConditionModel] = None,
    dtype: torch.dtype = torch.float32,
    conv_zero_init: ConvInitType = "zero",
    conv_init_std: float = 0.02,
) -> ControlNetModel:
    """从 SD1.5 UNet 初始化 ControlNet。

    Args:
        pretrained_model_name_or_path: 模型路径或 HuggingFace model id
        conditioning_channels: 条件图像通道数（默认 3）
        unet: 已加载的 UNet 实例（融合 LoRA 后传入）；为 None 时从路径加载
        dtype: 临时加载 UNet 的精度（仅在 unet=None 时生效）
        conv_zero_init: zero conv 初始化方式，"zero"|"normal"|"xavier"|"kaiming"
        conv_init_std: normal 模式下的标准差（其他模式忽略）
    """
    if unet is None:
        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="unet", torch_dtype=dtype,
        )
        controlnet = _create_controlnet_from_unet(unet, conditioning_channels, conv_zero_init, conv_init_std)
        del unet
    else:
        controlnet = _create_controlnet_from_unet(unet, conditioning_channels, conv_zero_init, conv_init_std)

    return controlnet


def create_controlnet_sdxl(
    pretrained_model_name_or_path: str,
    conditioning_channels: int = 3,
    unet: Optional[UNet2DConditionModel] = None,
    dtype: torch.dtype = torch.float32,
    conv_zero_init: ConvInitType = "zero",
    conv_init_std: float = 0.02,
) -> ControlNetModel:
    """从 SDXL UNet 初始化 ControlNet。

    Args:
        pretrained_model_name_or_path: 模型路径或 HuggingFace model id
        conditioning_channels: 条件图像通道数（默认 3）
        unet: 已加载的 UNet 实例（融合 LoRA 后传入）；为 None 时从路径加载
        dtype: 临时加载 UNet 的精度（仅在 unet=None 时生效）
        conv_zero_init: zero conv 初始化方式，"zero"|"normal"|"xavier"|"kaiming"
        conv_init_std: normal 模式下的标准差（其他模式忽略）
    """
    if unet is None:
        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="unet", torch_dtype=dtype,
        )
        controlnet = _create_controlnet_from_unet(unet, conditioning_channels, conv_zero_init, conv_init_std)
        del unet
    else:
        controlnet = _create_controlnet_from_unet(unet, conditioning_channels, conv_zero_init, conv_init_std)

    return controlnet


def load_controlnet(controlnet_path: str) -> ControlNetModel:
    """从本地路径加载已训练的 ControlNet。"""
    return ControlNetModel.from_pretrained(controlnet_path)
