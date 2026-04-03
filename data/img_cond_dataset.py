"""PixArt-Sigma 图像条件数据集 — 用预缓存条件特征替代文本嵌入。

数据配对:
  train_data_dir (size_1024/floor):           目标平面图
  conditioning_data_dir (size_1024_controlnet/floor): 条件 color 图
  → 通过 base_key 匹配（去掉已知后缀后的公共前缀相同）

两种条件缓存格式:
  - VAE 模式: 缓存 VAE latent (4, H/8, W/8)，训练时过可训练投射层
  - DINOv2 模式: 缓存 DINOv2 patch features (N, 1024)，训练时过可训练投射层
"""

import logging
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .dataset import BaseImageDataset, IMAGE_EXTENSIONS
from .controlnet_dataset import _build_cond_index, _strip_known_suffix, _KNOWN_SUFFIX_PAIRS

logger = logging.getLogger(__name__)


class PixArtImgCondCachedLatentDataset(BaseImageDataset):
    """目标图使用预缓存 VAE latent，条件图使用预缓存编码特征。

    __getitem__ 返回:
        latents:       目标 VAE latent (4, H/8, W/8)
        cond_features: 条件编码特征
            - VAE 模式: (4, H_cond/8, W_cond/8)  VAE latent
            - DINOv2 模式: (num_patches, hidden_size)  DINOv2 features
    """

    def __init__(
        self,
        data_dir: str,
        cache_dir: str,
        conditioning_data_dir: str,
        cond_feature_cache_dir: str,
        cond_encoder_type: str = "dinov2",
        resolution: int = 1024,
        random_flip: bool = True,
    ):
        super().__init__(
            data_dir, resolution=resolution,
            center_crop=False, random_flip=False,
        )
        self.cache_dir = Path(cache_dir)
        self.cond_feature_cache_dir = Path(cond_feature_cache_dir)
        self.cond_encoder_type = cond_encoder_type
        self.do_random_flip = random_flip

        cond_dir = Path(conditioning_data_dir)
        if cond_dir.exists():
            self._cond_index = _build_cond_index(cond_dir)
            logger.info(
                f"[ImgCond] conditioning index: {len(self._cond_index)} entries, "
                f"encoder={cond_encoder_type}"
            )
        else:
            self._cond_index = {}
            logger.warning(f"Conditioning dir not found: {conditioning_data_dir}")

    def _get_base_key(self, idx: int) -> str:
        fname = self.image_paths[idx].name
        for orig_suffix, _ in _KNOWN_SUFFIX_PAIRS:
            base_key = _strip_known_suffix(fname, orig_suffix)
            if base_key is not None:
                return base_key
        return self.image_paths[idx].stem

    def get_image_sizes(self) -> list[tuple[int, int]]:
        """从缓存文件读取 target_hw，避免重新打开原图。"""
        sizes = []
        for p in self.image_paths:
            data = torch.load(
                self.cache_dir / f"{p.stem}.pt",
                map_location="cpu", weights_only=True,
            )
            target_h, target_w = data["target_hw"].tolist()
            sizes.append((target_w, target_h))
        return sizes

    def __getitem__(self, idx: int) -> dict:
        stem = self.image_paths[idx].stem
        cache_file = self.cache_dir / f"{stem}.pt"
        data = torch.load(cache_file, map_location="cpu", weights_only=False)

        use_flip = self.do_random_flip and random.random() < 0.5
        latents = data["latent_flip"] if use_flip else data["latent"]

        base_key = self._get_base_key(idx)
        cond_cache_file = self.cond_feature_cache_dir / f"{base_key}.pt"
        cond_data = torch.load(cond_cache_file, map_location="cpu", weights_only=False)

        if self.cond_encoder_type == "vae":
            cond_features = cond_data["latent_flip"] if use_flip else cond_data["latent"]
        else:
            cond_features = cond_data["features_flip"] if use_flip else cond_data["features"]

        return {
            "latents": latents.float(),
            "cond_features": cond_features.float(),
        }
