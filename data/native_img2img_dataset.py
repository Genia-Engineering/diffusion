"""Native Img2Img 数据集 — 加载预缓存的 (target_latent, reference_latent) 对。

两组 VAE latent 缓存:
  - target latent cache:    目标图 VAE latent (4, H/8, W/8)
  - reference latent cache: 条件色块图 VAE latent (4, H/8, W/8)

通过 base_key 匹配 target 和 reference 图片（去掉已知后缀后的公共前缀相同）。
两者的 latent 空间分辨率一致（按同一桶尺寸 resize 后 VAE encode）。
"""

import logging
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .dataset import BaseImageDataset
from .controlnet_dataset import _build_cond_index, _strip_known_suffix, _KNOWN_SUFFIX_PAIRS

logger = logging.getLogger(__name__)


class NativeImg2ImgCachedLatentDataset(BaseImageDataset):
    """目标图和条件色块图均使用预缓存 VAE latent。

    __getitem__ 返回:
        latents:     目标 VAE latent (4, H/8, W/8)
        ref_latents: 参考色块图 VAE latent (4, H/8, W/8)
    """

    def __init__(
        self,
        data_dir: str,
        cache_dir: str,
        conditioning_data_dir: str,
        ref_latent_cache_dir: str,
        resolution: int = 1024,
        random_flip: bool = True,
    ):
        super().__init__(
            data_dir, resolution=resolution,
            center_crop=False, random_flip=False,
        )
        self.cache_dir = Path(cache_dir)
        self.ref_latent_cache_dir = Path(ref_latent_cache_dir)
        self.do_random_flip = random_flip

        cond_dir = Path(conditioning_data_dir)
        if cond_dir.exists():
            self._cond_index = _build_cond_index(cond_dir)
            logger.info(
                f"[NativeImg2Img] conditioning index: {len(self._cond_index)} entries"
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

        weight_mask = data.get("weight_mask", None)
        if weight_mask is not None:
            weight_mask = torch.flip(weight_mask, dims=[-1]) if use_flip else weight_mask
            weight_mask = weight_mask.float()

        base_key = self._get_base_key(idx)
        ref_cache_file = self.ref_latent_cache_dir / f"{base_key}.pt"
        ref_data = torch.load(ref_cache_file, map_location="cpu", weights_only=False)
        ref_latents = ref_data["latent_flip"] if use_flip else ref_data["latent"]

        result = {
            "latents": latents.float(),
            "ref_latents": ref_latents.float(),
        }
        if weight_mask is not None:
            result["weight_mask"] = weight_mask
        return result
