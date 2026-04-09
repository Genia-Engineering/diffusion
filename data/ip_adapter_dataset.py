"""PixArt-Sigma IP-Adapter 数据集 — 同时提供目标 VAE latent 和 CLIP 条件特征。

数据配对:
  train_data_dir:        目标图（用于 VAE latent 缓存）
  conditioning_data_dir: 条件图（用于 CLIP 特征缓存）
  → 通过 base_key 匹配

文本嵌入由训练器统一预缓存为共享张量（固定 caption），不由数据集提供。
"""

import logging
import random
from pathlib import Path

import torch

from .dataset import BaseImageDataset
from .controlnet_dataset import _build_cond_index, _strip_known_suffix, _KNOWN_SUFFIX_PAIRS

logger = logging.getLogger(__name__)


class PixArtIPAdapterCachedDataset(BaseImageDataset):
    """目标图使用预缓存 VAE latent，条件图使用预缓存 CLIP 特征。

    __getitem__ 返回:
        latents:       目标 VAE latent (4, H/8, W/8)
        clip_features: CLIP 图像特征 (num_tokens, clip_embed_dim)
    """

    def __init__(
        self,
        data_dir: str,
        cache_dir: str,
        conditioning_data_dir: str,
        clip_feature_cache_dir: str,
        resolution: int = 1024,
        random_flip: bool = True,
        exclude_stems: set[str] | None = None,
    ):
        super().__init__(
            data_dir, resolution=resolution,
            center_crop=False, random_flip=False,
        )
        self.cache_dir = Path(cache_dir)
        self.clip_feature_cache_dir = Path(clip_feature_cache_dir)
        self.do_random_flip = random_flip

        cond_dir = Path(conditioning_data_dir)
        if cond_dir.exists():
            self._cond_index = _build_cond_index(cond_dir)
            logger.info(
                f"[IPAdapter] conditioning index: {len(self._cond_index)} entries"
            )
        else:
            self._cond_index = {}
            logger.warning(f"Conditioning dir not found: {conditioning_data_dir}")

        if self._cond_index:
            before = len(self.image_paths)
            self.image_paths = [
                p for p in self.image_paths if self._has_cond(p)
            ]
            dropped = before - len(self.image_paths)
            if dropped:
                logger.warning(
                    f"[IPAdapter] 过滤掉 {dropped} 张无条件图匹配的训练图 "
                    f"({before} → {len(self.image_paths)})"
                )

        if exclude_stems:
            import logging as _logging
            _logger = _logging.getLogger(__name__)
            before = len(self.image_paths)
            self.image_paths = [p for p in self.image_paths if p.stem not in exclude_stems]
            _logger.info(f"验证集排除: {before} → {len(self.image_paths)} (排除 {before - len(self.image_paths)} 张)")

    @staticmethod
    def _base_key_from_path(p: Path) -> str:
        for orig_suffix, _ in _KNOWN_SUFFIX_PAIRS:
            base_key = _strip_known_suffix(p.name, orig_suffix)
            if base_key is not None:
                return base_key
        return p.stem

    def _has_cond(self, p: Path) -> bool:
        return self._base_key_from_path(p) in self._cond_index

    def _get_base_key(self, idx: int) -> str:
        return self._base_key_from_path(self.image_paths[idx])

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
        clip_cache_file = self.clip_feature_cache_dir / f"{base_key}.pt"
        clip_data = torch.load(clip_cache_file, map_location="cpu", weights_only=False)

        clip_features = clip_data["features_flip"] if use_flip else clip_data["features"]

        return {
            "latents": latents.float(),
            "clip_features": clip_features.float(),
        }
