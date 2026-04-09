"""图像数据集，支持 SD1.5（单编码器）和 SDXL（双编码器）。

caption 策略:
  - 优先使用 fixed_caption（从 config 传入），所有图像共用同一文本提示词。
  - 不再逐图读取 .txt 文件，fixed_caption 在 __init__ 中一次性完成 tokenization 并缓存。
  - 若未传 fixed_caption，则回退到空字符串 ""（等价于无条件训练）。
"""

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import apply_transforms, build_transforms

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class BaseImageDataset(Dataset):
    """图像数据集基类。"""

    def __init__(
        self,
        data_dir: str,
        resolution: int = 512,
        center_crop: bool = False,
        random_flip: bool = True,
        max_train_samples: int | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.transforms_dict = build_transforms(resolution, center_crop, random_flip)
        self.index_to_bucket: dict[int, tuple[int, int]] = {}

        self.image_paths = sorted(
            p for p in self.data_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {data_dir}")

        if max_train_samples is not None and max_train_samples < len(self.image_paths):
            self.image_paths = self.image_paths[:max_train_samples]

    def set_bucket_assignments(
        self, bucket_to_indices: dict[tuple[int, int], list[int]]
    ) -> None:
        """将桶分配结果传入 Dataset，使 __getitem__ 能按桶尺寸 resize。

        Args:
            bucket_to_indices: {(bucket_w, bucket_h): [image_index, ...]}
        """
        self.index_to_bucket = {
            idx: bucket
            for bucket, indices in bucket_to_indices.items()
            for idx in indices
        }

    def _get_target_size(self, idx: int) -> tuple[int, int]:
        """返回该图像对应的目标分辨率（桶尺寸优先，未分配时退化为正方形）。"""
        return self.index_to_bucket.get(idx, (self.resolution, self.resolution))

    def get_image_sizes(self) -> list[tuple[int, int]]:
        """返回所有图像的 (width, height)，用于 Bucket 分配。"""
        sizes = []
        for p in self.image_paths:
            with Image.open(p) as img:
                sizes.append(img.size)
        return sizes

    def get_pil_image(self, idx: int) -> Image.Image:
        """返回原始 PIL 图像（不做 tensor 转换），供 FID 真实图像采样使用。"""
        return Image.open(self.image_paths[idx]).convert("RGB")

    def __len__(self):
        return len(self.image_paths)


class SD15Dataset(BaseImageDataset):
    """SD 1.5 数据集 — 单 tokenizer，固定 caption。"""

    def __init__(
        self,
        data_dir: str,
        tokenizer,
        resolution: int = 512,
        center_crop: bool = False,
        random_flip: bool = True,
        fixed_caption: str = "",
        max_train_samples: int | None = None,
    ):
        super().__init__(data_dir, resolution, center_crop, random_flip, max_train_samples=max_train_samples)
        self.tokenizer = tokenizer

        # 提前 tokenize，所有样本共享同一份 input_ids
        self.cached_input_ids = tokenizer(
            fixed_caption,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)  # (seq_len,)

    def __getitem__(self, idx: int) -> dict:
        image = Image.open(self.image_paths[idx]).convert("RGB")

        target_size = self._get_target_size(idx)
        transformed = apply_transforms(image, target_size, self.transforms_dict)

        return {
            "pixel_values": transformed["pixel_values"],
            "input_ids": self.cached_input_ids,
        }


class SDXLDataset(BaseImageDataset):
    """SDXL 数据集 — 双 tokenizer (CLIP-L + CLIP-G)。

    支持两种文本模式:
      1. 固定 caption (caption_dir=None): 所有图像共用同一份 tokenized ids
      2. 逐图 caption (caption_dir 指定): 每张图像从对应 .txt 文件读取 caption 并实时 tokenize
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer_1,
        tokenizer_2,
        resolution: int = 1024,
        center_crop: bool = False,
        random_flip: bool = True,
        fixed_caption: str = "",
        max_train_samples: int | None = None,
        caption_dir: str | None = None,
        caption_stem_replace: dict | None = None,
        caption_fallback: str = "",
    ):
        super().__init__(data_dir, resolution, center_crop, random_flip, max_train_samples=max_train_samples)
        self.tokenizer_1 = tokenizer_1
        self.tokenizer_2 = tokenizer_2
        self.caption_dir = Path(caption_dir) if caption_dir else None
        stem_replace = caption_stem_replace or {}
        self._stem_from = stem_replace.get("from", "")
        self._stem_to = stem_replace.get("to", "")
        self._caption_fallback = caption_fallback or fixed_caption

        if self.caption_dir is None:
            self.cached_input_ids_1 = tokenizer_1(
                fixed_caption,
                padding="max_length",
                truncation=True,
                max_length=tokenizer_1.model_max_length,
                return_tensors="pt",
            ).input_ids.squeeze(0)

            self.cached_input_ids_2 = tokenizer_2(
                fixed_caption,
                padding="max_length",
                truncation=True,
                max_length=tokenizer_2.model_max_length,
                return_tensors="pt",
            ).input_ids.squeeze(0)

    def _tokenize_caption(self, caption: str):
        ids_1 = self.tokenizer_1(
            caption, padding="max_length", truncation=True,
            max_length=self.tokenizer_1.model_max_length, return_tensors="pt",
        ).input_ids.squeeze(0)
        ids_2 = self.tokenizer_2(
            caption, padding="max_length", truncation=True,
            max_length=self.tokenizer_2.model_max_length, return_tensors="pt",
        ).input_ids.squeeze(0)
        return ids_1, ids_2

    def _read_caption(self, idx: int):
        stem = self.image_paths[idx].stem
        caption_stem = stem.replace(self._stem_from, self._stem_to) if self._stem_from else stem
        caption_file = self.caption_dir / f"{caption_stem}.txt"
        if caption_file.exists():
            return caption_file.read_text(encoding="utf-8").strip()
        return self._caption_fallback

    def __getitem__(self, idx: int) -> dict:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        original_size = image.size  # PIL: (w, h)

        target_size = self._get_target_size(idx)
        transformed = apply_transforms(image, target_size, self.transforms_dict)

        orig_h, orig_w = original_size[1], original_size[0]
        tgt_h, tgt_w = target_size[1], target_size[0]

        if self.caption_dir is not None:
            input_ids_1, input_ids_2 = self._tokenize_caption(self._read_caption(idx))
        else:
            input_ids_1 = self.cached_input_ids_1
            input_ids_2 = self.cached_input_ids_2

        return {
            "pixel_values": transformed["pixel_values"],
            "input_ids_1": input_ids_1,
            "input_ids_2": input_ids_2,
            "original_size": torch.tensor([orig_h, orig_w], dtype=torch.long),
            "crop_top_left": torch.tensor([0, 0], dtype=torch.long),
            "target_size":   torch.tensor([tgt_h, tgt_w], dtype=torch.long),
        }


class SD15CachedLatentDataset(BaseImageDataset):
    """SD 1.5 使用预缓存 VAE latent 的数据集。

    训练时直接从磁盘读取预计算的 latent，跳过 VAE encode 步骤。
    每张图片保存原图与水平翻转两份 latent，训练时随机选其一实现数据增强。
    """

    def __init__(
        self,
        data_dir: str,
        cache_dir: str,
        tokenizer,
        resolution: int = 512,
        random_flip: bool = True,
        fixed_caption: str = "",
        max_train_samples: int | None = None,
    ):
        # 禁用父类的 random_flip（由本类自行处理）
        super().__init__(data_dir, resolution, center_crop=False, random_flip=False, max_train_samples=max_train_samples)
        self.cache_dir = Path(cache_dir)
        self.do_random_flip = random_flip

        self.cached_input_ids = tokenizer(
            fixed_caption,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)

    def get_image_sizes(self) -> list[tuple[int, int]]:
        """从缓存文件读取 target_hw，避免重新打开原图。"""
        sizes = []
        for p in self.image_paths:
            data = torch.load(self.cache_dir / f"{p.stem}.pt", map_location="cpu", weights_only=True)
            target_h, target_w = data["target_hw"].tolist()
            sizes.append((target_w, target_h))  # PIL 格式: (w, h)
        return sizes

    def __getitem__(self, idx: int) -> dict:
        cache_file = self.cache_dir / f"{self.image_paths[idx].stem}.pt"
        data = torch.load(cache_file, map_location="cpu", weights_only=False)

        use_flip = self.do_random_flip and random.random() < 0.5
        latents = data["latent_flip"] if use_flip else data["latent"]

        return {
            "latents": latents.float(),
            "input_ids": self.cached_input_ids,
        }


class SDXLCachedLatentDataset(BaseImageDataset):
    """SDXL 使用预缓存 VAE latent 的数据集。

    训练时直接从磁盘读取预计算的 latent，跳过 VAE encode 步骤。
    每张图片保存原图与水平翻转两份 latent，训练时随机选其一实现数据增强。
    size conditioning（original_size / target_size）从缓存文件中恢复。

    支持两种文本模式:
      1. 固定 caption (text_embed_cache_dir=None): 所有图像共用同一份 tokenized ids
      2. 逐图预编码 (text_embed_cache_dir 指定): 每张图像从磁盘加载预编码的 CLIP 嵌入
    """

    def __init__(
        self,
        data_dir: str,
        cache_dir: str,
        tokenizer_1=None,
        tokenizer_2=None,
        resolution: int = 1024,
        random_flip: bool = True,
        fixed_caption: str = "",
        max_train_samples: int | None = None,
        text_embed_cache_dir: str | None = None,
    ):
        super().__init__(data_dir, resolution, center_crop=False, random_flip=False, max_train_samples=max_train_samples)
        self.cache_dir = Path(cache_dir)
        self.do_random_flip = random_flip
        self.text_embed_cache_dir = Path(text_embed_cache_dir) if text_embed_cache_dir else None

        if self.text_embed_cache_dir is None:
            self.cached_input_ids_1 = tokenizer_1(
                fixed_caption,
                padding="max_length",
                truncation=True,
                max_length=tokenizer_1.model_max_length,
                return_tensors="pt",
            ).input_ids.squeeze(0)

            self.cached_input_ids_2 = tokenizer_2(
                fixed_caption,
                padding="max_length",
                truncation=True,
                max_length=tokenizer_2.model_max_length,
                return_tensors="pt",
            ).input_ids.squeeze(0)

    def get_image_sizes(self) -> list[tuple[int, int]]:
        """从缓存文件读取 target_hw，避免重新打开原图。"""
        sizes = []
        for p in self.image_paths:
            data = torch.load(self.cache_dir / f"{p.stem}.pt", map_location="cpu", weights_only=True)
            target_h, target_w = data["target_hw"].tolist()
            sizes.append((target_w, target_h))  # PIL 格式: (w, h)
        return sizes

    def __getitem__(self, idx: int) -> dict:
        cache_file = self.cache_dir / f"{self.image_paths[idx].stem}.pt"
        data = torch.load(cache_file, map_location="cpu", weights_only=False)

        use_flip = self.do_random_flip and random.random() < 0.5
        latents = data["latent_flip"] if use_flip else data["latent"]

        orig_h, orig_w = data["original_hw"].tolist()
        target_h, target_w = data["target_hw"].tolist()

        if self.text_embed_cache_dir is not None:
            text_file = self.text_embed_cache_dir / f"{self.image_paths[idx].stem}.pt"
            text_data = torch.load(text_file, map_location="cpu", weights_only=True)
            return {
                "latents": latents.float(),
                "prompt_embeds": text_data["prompt_embeds"].squeeze(0).float(),
                "pooled_prompt_embeds": text_data["pooled_prompt_embeds"].squeeze(0).float(),
                "original_size": torch.tensor([orig_h, orig_w], dtype=torch.long),
                "crop_top_left": torch.zeros(2, dtype=torch.long),
                "target_size":   torch.tensor([target_h, target_w], dtype=torch.long),
            }

        return {
            "latents": latents.float(),
            "input_ids_1": self.cached_input_ids_1,
            "input_ids_2": self.cached_input_ids_2,
            "original_size": torch.tensor([orig_h, orig_w], dtype=torch.long),
            "crop_top_left": torch.zeros(2, dtype=torch.long),
            "target_size":   torch.tensor([target_h, target_w], dtype=torch.long),
        }


class PixArtSigmaDataset(BaseImageDataset):
    """PixArt-Sigma 数据集 — T5 tokenizer（max_length=300），固定 caption。"""

    T5_MAX_LENGTH = 300

    def __init__(
        self,
        data_dir: str,
        tokenizer,
        resolution: int = 1024,
        center_crop: bool = False,
        random_flip: bool = True,
        fixed_caption: str = "",
        max_train_samples: int | None = None,
    ):
        super().__init__(data_dir, resolution, center_crop, random_flip, max_train_samples=max_train_samples)
        self.tokenizer = tokenizer

        tokenized = tokenizer(
            fixed_caption,
            padding="max_length",
            truncation=True,
            max_length=self.T5_MAX_LENGTH,
            return_tensors="pt",
        )
        self.cached_input_ids = tokenized.input_ids.squeeze(0)
        self.cached_attention_mask = tokenized.attention_mask.squeeze(0)

    def __getitem__(self, idx: int) -> dict:
        image = Image.open(self.image_paths[idx]).convert("RGB")

        target_size = self._get_target_size(idx)
        transformed = apply_transforms(image, target_size, self.transforms_dict)

        return {
            "pixel_values": transformed["pixel_values"],
            "input_ids": self.cached_input_ids,
            "attention_mask": self.cached_attention_mask,
        }


class PixArtSigmaCachedLatentDataset(BaseImageDataset):
    """PixArt-Sigma 使用预缓存 VAE latent 的数据集。

    与 SDXLCachedLatentDataset 类似，但使用 T5 tokenizer 和 attention mask。
    不需要 SDXL 的 size conditioning（original_size / target_size / crop_top_left）。

    支持两种文本模式:
      1. 固定 caption (text_embed_cache_dir=None): 所有图像共用同一份 tokenized ids
      2. 逐图预编码 (text_embed_cache_dir 指定): 每张图像从磁盘加载预编码的 T5 嵌入
    """

    T5_MAX_LENGTH = 300

    def __init__(
        self,
        data_dir: str,
        cache_dir: str,
        tokenizer=None,
        resolution: int = 1024,
        random_flip: bool = True,
        fixed_caption: str = "",
        max_train_samples: int | None = None,
        text_embed_cache_dir: str | None = None,
        exclude_stems: set[str] | None = None,
    ):
        super().__init__(data_dir, resolution, center_crop=False, random_flip=False, max_train_samples=max_train_samples)
        self.cache_dir = Path(cache_dir)
        self.do_random_flip = random_flip
        self.text_embed_cache_dir = Path(text_embed_cache_dir) if text_embed_cache_dir else None

        if exclude_stems:
            import logging as _logging
            _logger = _logging.getLogger(__name__)
            before = len(self.image_paths)
            self.image_paths = [
                p for p in self.image_paths if p.stem not in exclude_stems
            ]
            _logger.info(
                f"验证集排除: {before} → {len(self.image_paths)} "
                f"(排除 {before - len(self.image_paths)} 张)"
            )

        if self.text_embed_cache_dir is None:
            tokenized = tokenizer(
                fixed_caption,
                padding="max_length",
                truncation=True,
                max_length=self.T5_MAX_LENGTH,
                return_tensors="pt",
            )
            self.cached_input_ids = tokenized.input_ids.squeeze(0)
            self.cached_attention_mask = tokenized.attention_mask.squeeze(0)

    def get_image_sizes(self) -> list[tuple[int, int]]:
        """从缓存文件读取 target_hw，避免重新打开原图。"""
        sizes = []
        for p in self.image_paths:
            data = torch.load(self.cache_dir / f"{p.stem}.pt", map_location="cpu", weights_only=True)
            target_h, target_w = data["target_hw"].tolist()
            sizes.append((target_w, target_h))
        return sizes

    def __getitem__(self, idx: int) -> dict:
        cache_file = self.cache_dir / f"{self.image_paths[idx].stem}.pt"
        data = torch.load(cache_file, map_location="cpu", weights_only=False)

        use_flip = self.do_random_flip and random.random() < 0.5
        latents = data["latent_flip"] if use_flip else data["latent"]

        weight_mask = data.get("weight_mask", None)
        if weight_mask is not None:
            weight_mask = torch.flip(weight_mask, dims=[-1]) if use_flip else weight_mask
            weight_mask = weight_mask.float()

        padding_mask = data.get("padding_mask", None)
        if padding_mask is not None:
            padding_mask = torch.flip(padding_mask, dims=[-1]) if use_flip else padding_mask
            padding_mask = padding_mask.float()

        if self.text_embed_cache_dir is not None:
            text_file = self.text_embed_cache_dir / f"{self.image_paths[idx].stem}.pt"
            text_data = torch.load(text_file, map_location="cpu", weights_only=True)
            result = {
                "latents": latents.float(),
                "prompt_embeds": text_data["prompt_embeds"].squeeze(0).float(),
                "prompt_attention_mask": text_data["prompt_attention_mask"].squeeze(0),
            }
        else:
            result = {
                "latents": latents.float(),
                "input_ids": self.cached_input_ids,
                "attention_mask": self.cached_attention_mask,
            }

        if weight_mask is not None:
            result["weight_mask"] = weight_mask
        if padding_mask is not None:
            result["padding_mask"] = padding_mask
        return result


class SanaCachedLatentDataset(BaseImageDataset):
    """Sana 0.6B 使用预缓存 VAE latent 的数据集。

    与 PixArtSigmaCachedLatentDataset 结构相同（单文本编码器 + attention mask），
    但使用 Gemma2 tokenizer（max_length=300）和 AutoencoderDC 的 32 通道 latent。

    支持两种文本模式:
      1. 固定 caption (text_embed_cache_dir=None): 所有图像共用同一份 tokenized ids
      2. 逐图预编码 (text_embed_cache_dir 指定): 每张图像从磁盘加载预编码嵌入
    """

    MAX_SEQ_LENGTH = 300

    def __init__(
        self,
        data_dir: str,
        cache_dir: str,
        tokenizer=None,
        resolution: int = 1024,
        random_flip: bool = True,
        fixed_caption: str = "",
        max_train_samples: int | None = None,
        text_embed_cache_dir: str | None = None,
    ):
        super().__init__(data_dir, resolution, center_crop=False, random_flip=False, max_train_samples=max_train_samples)
        self.cache_dir = Path(cache_dir)
        self.do_random_flip = random_flip
        self.text_embed_cache_dir = Path(text_embed_cache_dir) if text_embed_cache_dir else None

        if self.text_embed_cache_dir is None:
            tokenized = tokenizer(
                fixed_caption,
                padding="max_length",
                truncation=True,
                max_length=self.MAX_SEQ_LENGTH,
                return_tensors="pt",
            )
            self.cached_input_ids = tokenized.input_ids.squeeze(0)
            self.cached_attention_mask = tokenized.attention_mask.squeeze(0)

    def get_image_sizes(self) -> list[tuple[int, int]]:
        sizes = []
        for p in self.image_paths:
            data = torch.load(self.cache_dir / f"{p.stem}.pt", map_location="cpu", weights_only=True)
            target_h, target_w = data["target_hw"].tolist()
            sizes.append((target_w, target_h))
        return sizes

    def __getitem__(self, idx: int) -> dict:
        cache_file = self.cache_dir / f"{self.image_paths[idx].stem}.pt"
        data = torch.load(cache_file, map_location="cpu", weights_only=False)

        use_flip = self.do_random_flip and random.random() < 0.5
        latents = data["latent_flip"] if use_flip else data["latent"]

        weight_mask = data.get("weight_mask", None)
        if weight_mask is not None:
            weight_mask = torch.flip(weight_mask, dims=[-1]) if use_flip else weight_mask
            weight_mask = weight_mask.float()

        padding_mask = data.get("padding_mask", None)
        if padding_mask is not None:
            padding_mask = torch.flip(padding_mask, dims=[-1]) if use_flip else padding_mask
            padding_mask = padding_mask.float()

        if self.text_embed_cache_dir is not None:
            text_file = self.text_embed_cache_dir / f"{self.image_paths[idx].stem}.pt"
            text_data = torch.load(text_file, map_location="cpu", weights_only=True)
            result = {
                "latents": latents.float(),
                "prompt_embeds": text_data["prompt_embeds"].squeeze(0).float(),
                "prompt_attention_mask": text_data["prompt_attention_mask"].squeeze(0),
            }
        else:
            result = {
                "latents": latents.float(),
                "input_ids": self.cached_input_ids,
                "attention_mask": self.cached_attention_mask,
            }

        if weight_mask is not None:
            result["weight_mask"] = weight_mask
        if padding_mask is not None:
            result["padding_mask"] = padding_mask
        return result
