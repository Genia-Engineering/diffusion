"""PixArt-Sigma ControlNet-XS 训练器 — 继承 PixArtControlNetTrainer。

与标准 PixArt ControlNet 的关键差异:
  1. 模型: PixArtControlNetXSAdapter (~4.7% base params) 替换全尺寸 adapter (~46%)
  2. 架构: 薄型 XS blocks (无 cross-attention) + 双向 base↔control 投射
  3. 其余复用: 训练循环、预缓存、数据集、验证 pipeline 均继承父类
"""

import logging

from omegaconf import DictConfig

from models.controlnet_xs_pixart import (
    PixArtControlNetXSAdapter,
    PixArtControlNetXSTransformerModel,
)
from .pixart_controlnet_trainer import PixArtControlNetTrainer

logger = logging.getLogger(__name__)


class PixArtControlNetXSTrainer(PixArtControlNetTrainer):
    """PixArt-Sigma ControlNet-XS 训练器。"""

    def __init__(self, config: DictConfig):
        super().__init__(config)

    def _create_controlnet(self):
        """创建 ControlNet-XS: 薄型 adapter + 联合前向模型。"""
        xs_cfg = self.config.get("controlnet_xs", {})
        num_layers = int(xs_cfg.get("num_layers", 14))
        size_ratio = float(xs_cfg.get("size_ratio", 0.25))
        connection_interval = int(xs_cfg.get("connection_interval", 2))

        self.controlnet = PixArtControlNetXSAdapter.from_transformer(
            self.transformer,
            num_layers=num_layers,
            size_ratio=size_ratio,
            conditioning_mode=self.conditioning_mode,
            connection_interval=connection_interval,
        )

        self.joint_model = PixArtControlNetXSTransformerModel(
            transformer=self.transformer,
            controlnet=self.controlnet,
        )

    def _freeze_parameters(self):
        """冻结 VAE + T5 + Transformer, 仅 XS adapter 可训练。"""
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.transformer.requires_grad_(False)
        self.controlnet.requires_grad_(True)

        n_trainable = sum(p.numel() for p in self.controlnet.parameters() if p.requires_grad)
        n_base = sum(p.numel() for p in self.transformer.parameters())
        logger.info(
            f"ControlNet-XS freeze: {n_trainable/1e6:.1f}M trainable "
            f"({n_trainable/n_base*100:.1f}% of base {n_base/1e6:.0f}M)"
        )
