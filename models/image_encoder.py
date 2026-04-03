"""图像条件编码器 — 将条件图像编码为 cross-attention 特征序列。

支持三种编码模式：
  - VAE 模式: 复用已有 VAE encoder，通过可训练 patchify 卷积投射到 caption_channels
  - DINOv2 模式: frozen DINOv2 ViT-L/14 + 可训练 MLP 投射层
  - CLIP 模式: frozen CLIP ViT-H/14 + 可训练 MLP 投射层

三种模式的 forward 统一返回 (encoder_hidden_states, encoder_attention_mask)，
可直接作为 PixArtTransformer2DModel 的 cross-attention 输入。

维度对齐：
  PixArt-Sigma 的 caption_channels = 4096，内部 caption_projection 将 4096 → 1152。
  三种编码器的投射层均输出 dim = 4096，复用 Transformer 已有的 caption_projection。
"""

import logging
import math

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class VAEImageEncoder(nn.Module):
    """VAE latent → cross-attention 特征序列。

    将 VAE encoder 产出的 latent (B, 4, H/8, W/8) 通过三层 Conv2d
    逐级提取特征并投射为 (B, num_patches, projection_dim) 的特征序列。

    三层 Conv(k=5, s=2, p=2) 总降采样 2³=8×，与原单层 Conv(k=8, s=8) token 数相同，
    但引入了：
      - 层间局部感受野重叠（k=5, s=2 → 相邻 token 共享 3 像素）
      - 逐级通道扩展（4→512→2048→4096）实现渐进式特征提取
      - 每层 GroupNorm + GELU 稳定激活值尺度并增加非线性表达能力
      - 有效感受野 29×29 latent = 232×232 原始像素（远大于原来的 8×8=64×64）

    1024px 图像: latent 128×128 → 64×64 → 32×32 → 16×16 = 256 tokens
    所有 PixArt 桶尺寸的 latent 维度均为 8 的倍数，三层 stride=2 恰好整除。
    """

    def __init__(
        self,
        in_channels: int = 4,
        projection_dim: int = 4096,
        hidden_channels: tuple[int, ...] = (512, 2048),
    ):
        super().__init__()
        self.projection_dim = projection_dim

        channels = [in_channels] + list(hidden_channels) + [projection_dim]

        layers: list[nn.Module] = []
        for i in range(len(channels) - 1):
            layers.append(
                nn.Conv2d(
                    channels[i], channels[i + 1],
                    kernel_size=5, stride=2, padding=2, bias=True,
                )
            )
            layers.append(nn.GroupNorm(32, channels[i + 1]))
            layers.append(nn.GELU())

        self.proj = nn.Sequential(*layers)
        self.output_norm = nn.LayerNorm(projection_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.proj.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                nn.init.zeros_(m.bias)

    def forward(
        self, latents: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            latents: VAE latent (B, C, H_lat, W_lat), 已乘 scaling_factor

        Returns:
            encoder_hidden_states: (B, num_patches, projection_dim)
            encoder_attention_mask: (B, num_patches), 全 1
        """
        x = self.proj(latents)                          # (B, D, H/8, W/8)
        x = x.flatten(2).transpose(1, 2)               # (B, H/8*W/8, D)
        x = self.output_norm(x)                         # 稳定输出尺度，防止 bf16 溢出
        mask = torch.ones(
            x.shape[:2], dtype=torch.long, device=x.device
        )
        return x, mask


class DINOv2ImageEncoder(nn.Module):
    """DINOv2 frozen backbone + 可训练 MLP 投射层。

    DINOv2 ViT-L/14:
      - hidden_dim = 1024, patch_size = 14
      - 518px 输入 → 37x37 = 1369 patch tokens (+ CLS)
      - 可通过 positional embedding 插值支持任意分辨率 (需为 14 的倍数)

    投射层将 DINOv2 特征 (B, N, 1024) 映射到 (B, N, projection_dim=4096)。
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-large",
        projection_dim: int = 4096,
        freeze_backbone: bool = True,
        resolution: int | None = 518,
    ):
        super().__init__()
        from transformers import Dinov2Model

        self.backbone = Dinov2Model.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size  # 1024 for ViT-L
        self.patch_size = self.backbone.config.patch_size  # 14

        self.projection = nn.Sequential(
            nn.Linear(hidden_size, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.output_norm = nn.LayerNorm(projection_dim)
        self._init_projection()

        self.resolution = resolution

        if freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()
            logger.info(
                f"DINOv2 backbone frozen ({model_name}, "
                f"hidden_size={hidden_size}, patch_size={self.patch_size})"
            )

    def _init_projection(self):
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def get_input_resolution(self, original_h: int, original_w: int) -> tuple[int, int]:
        """计算 DINOv2 实际输入分辨率（调整至 patch_size 的倍数）。"""
        ps = self.patch_size
        if self.resolution is not None:
            h = w = self.resolution
        else:
            h, w = original_h, original_w
        h = (h // ps) * ps
        w = (w // ps) * ps
        return h, w

    @torch.no_grad()
    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """仅编码 backbone 部分（frozen），返回 patch features 不含 CLS token。

        用于预缓存：结果保存到磁盘后不再需要 backbone。
        """
        outputs = self.backbone(pixel_values)
        features = outputs.last_hidden_state[:, 1:, :]  # 去掉 CLS token
        return features

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """投射 backbone 特征 → cross-attention hidden states。

        Args:
            features: (B, num_patches, hidden_size), 预缓存的 DINOv2 features

        Returns:
            encoder_hidden_states: (B, num_patches, projection_dim)
            encoder_attention_mask: (B, num_patches), 全 1
        """
        x = self.projection(features)
        x = self.output_norm(x)
        mask = torch.ones(
            x.shape[:2], dtype=torch.long, device=x.device
        )
        return x, mask


class CLIPImageEncoder(nn.Module):
    """CLIP Vision frozen backbone + 可训练 MLP 投射层。

    CLIP ViT-H/14 (laion/CLIP-ViT-H-14-laion2B-s32B-b79K):
      - hidden_size = 1280, patch_size = 14
      - 224px 输入 → 16x16 = 256 patch tokens (+ CLS)
      - 使用 hidden_states[-2]（倒数第二层）作为特征，去掉 CLS token

    投射层将 CLIP 特征 (B, N, 1280) 映射到 (B, N, projection_dim=4096)。
    """

    def __init__(
        self,
        model_name: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        projection_dim: int = 4096,
        freeze_backbone: bool = True,
        resolution: int | None = 224,
    ):
        super().__init__()
        from transformers import CLIPVisionModelWithProjection

        self.backbone = CLIPVisionModelWithProjection.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size  # 1280 for ViT-H
        self.patch_size = self.backbone.config.patch_size  # 14

        self.projection = nn.Sequential(
            nn.Linear(hidden_size, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.output_norm = nn.LayerNorm(projection_dim)
        self._init_projection()

        self.resolution = resolution

        if freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()
            logger.info(
                f"CLIP backbone frozen ({model_name}, "
                f"hidden_size={hidden_size}, patch_size={self.patch_size})"
            )

    def _init_projection(self):
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    @torch.no_grad()
    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """仅编码 backbone 部分（frozen），返回 patch features 不含 CLS token。

        使用 hidden_states[-2]（倒数第二层），与 IP-Adapter 一致。
        用于预缓存：结果保存到磁盘后不再需要 backbone。
        """
        outputs = self.backbone(pixel_values, output_hidden_states=True)
        features = outputs.hidden_states[-2][:, 1:, :]  # 去掉 CLS token
        return features

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """投射 backbone 特征 → cross-attention hidden states。

        Args:
            features: (B, num_patches, hidden_size), 预缓存的 CLIP features

        Returns:
            encoder_hidden_states: (B, num_patches, projection_dim)
            encoder_attention_mask: (B, num_patches), 全 1
        """
        x = self.projection(features)
        x = self.output_norm(x)
        mask = torch.ones(
            x.shape[:2], dtype=torch.long, device=x.device
        )
        return x, mask


def build_image_encoder(config) -> dict:
    """根据配置构建图像条件编码器。

    Args:
        config: image_encoder 子配置，包含 type, projection_dim 等

    Returns:
        {"encoder": nn.Module, "type": str}
    """
    enc_type = config.get("type", "dinov2")
    proj_dim = config.get("projection_dim", 4096)

    if enc_type == "vae":
        hidden_ch = config.get("vae_hidden_channels", [512, 2048])
        encoder = VAEImageEncoder(
            in_channels=4,
            projection_dim=proj_dim,
            hidden_channels=tuple(hidden_ch),
        )
        logger.info(
            f"Built VAE image encoder: channels=[4, {', '.join(map(str, hidden_ch))}, {proj_dim}], "
            f"3×Conv(k=5,s=2,p=2), total_stride=8"
        )
    elif enc_type == "dinov2":
        encoder = DINOv2ImageEncoder(
            model_name=config.get("dinov2_model", "facebook/dinov2-large"),
            projection_dim=proj_dim,
            freeze_backbone=True,
            resolution=config.get("dinov2_resolution", 518),
        )
        logger.info(
            f"Built DINOv2 image encoder: model={config.get('dinov2_model', 'facebook/dinov2-large')}, "
            f"resolution={config.get('dinov2_resolution', 518)}, projection_dim={proj_dim}"
        )
    elif enc_type == "clip":
        clip_model = config.get("clip_model", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        clip_res = config.get("clip_resolution", 224)
        encoder = CLIPImageEncoder(
            model_name=clip_model,
            projection_dim=proj_dim,
            freeze_backbone=True,
            resolution=clip_res,
        )
        logger.info(
            f"Built CLIP image encoder: model={clip_model}, "
            f"resolution={clip_res}, projection_dim={proj_dim}"
        )
    else:
        raise ValueError(f"Unknown image encoder type: {enc_type}")

    return {"encoder": encoder, "type": enc_type}
