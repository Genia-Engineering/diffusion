"""ControlNet 数据集 — 在图像-文本对基础上额外加载条件图像（Canny/Depth/Pose 等）。

数据目录结构（两种均支持）:
  方式 A（同名文件）:
    train_data_dir/        image_001.png
    conditioning_data_dir/ image_001.png   ← 与训练图像同 stem

  方式 B（本项目实际命名）:
    train_data_dir/        ..._f1___total__1024.png
    conditioning_data_dir/ ..._f1_controlnet_color_1024.png
    → 去掉各自已知后缀后 base_key 相同

  方式 C（在线提取）:
    conditioning_data_dir 不存在 / 未设置
    → 使用 conditioning_type="canny" 在线提取 Canny 边缘图
"""

import logging
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from .dataset import BaseImageDataset, IMAGE_EXTENSIONS
from .transforms import AspectRatioPad, AspectRatioResize, apply_transforms, build_transforms

logger = logging.getLogger(__name__)

# 已知的原始图像后缀与对应条件图像后缀（按优先级匹配）
_KNOWN_SUFFIX_PAIRS = [
    ("___total__1024.png", "_controlnet_color_1024.png"),
]


def make_canny(image: np.ndarray, low: int = 100, high: int = 200) -> np.ndarray:
    """从 RGB 图像在线提取 Canny 边缘图。"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low, high)
    return np.stack([edges] * 3, axis=-1)  # → (H, W, 3)


def _strip_known_suffix(name: str, suffix: str) -> str | None:
    """若 name 以 suffix 结尾则返回 base_key，否则返回 None。"""
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return None


def _build_cond_index(
    cond_dir: Path,
) -> dict[str, Path]:
    """扫描条件图像目录，构建 {base_key: path} 映射。

    base_key 提取规则：
      1. 尝试用 _KNOWN_SUFFIX_PAIRS 中的条件后缀剥离
      2. 若都不匹配，退回到 stem（同名方式 A）
    """
    index: dict[str, Path] = {}
    for f in cond_dir.iterdir():
        if not f.is_file() or f.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        base_key = None
        for _, cond_suffix in _KNOWN_SUFFIX_PAIRS:
            base_key = _strip_known_suffix(f.name, cond_suffix)
            if base_key is not None:
                break
        if base_key is None:
            base_key = f.stem
        index[base_key] = f
    return index


class ControlNetDataset(BaseImageDataset):
    """ControlNet 数据集，支持预提取条件图像或在线生成。

    Args:
        data_dir:             原始训练图像目录
        conditioning_data_dir:条件图像目录（None 则在线提取）
        tokenizer:            CLIP-L tokenizer
        conditioning_type:    在线提取类型，仅 "canny" 支持
        resolution:           基础分辨率（分桶时退化为正方形）
        center_crop:          是否中心裁剪
        random_flip:          是否随机水平翻转
        tokenizer_2:          SDXL 第二 tokenizer（OpenCLIP-G）
        fixed_caption:        所有样本共用的固定 caption（推荐），空字符串表示无条件
    """

    def __init__(
        self,
        data_dir: str,
        conditioning_data_dir: str,
        tokenizer,
        conditioning_type: str = "canny",
        resolution: int = 512,
        center_crop: bool = False,
        random_flip: bool = True,
        tokenizer_2=None,
        fixed_caption: str = "",
        max_train_samples: int | None = None,
    ):
        super().__init__(data_dir, resolution, center_crop, random_flip, max_train_samples=max_train_samples)
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.conditioning_type = conditioning_type

        # 预缓存 tokenize 结果（所有样本共享，避免逐帧 tokenize）
        self.cached_input_ids = tokenizer(
            fixed_caption,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        if tokenizer_2 is not None:
            self.cached_input_ids_2 = tokenizer_2(
                fixed_caption,
                padding="max_length",
                truncation=True,
                max_length=tokenizer_2.model_max_length,
                return_tensors="pt",
            ).input_ids.squeeze(0)
        else:
            self.cached_input_ids_2 = None

        # 构建条件图像索引
        self.conditioning_dir = Path(conditioning_data_dir) if conditioning_data_dir else None
        self.online_extract = self.conditioning_dir is None or not self.conditioning_dir.exists()

        if not self.online_extract:
            self._cond_index = _build_cond_index(self.conditioning_dir)
            logger.info(
                f"ControlNet conditioning index built: {len(self._cond_index)} entries "
                f"from {self.conditioning_dir}"
            )
        else:
            self._cond_index = {}
            logger.info(
                f"Conditioning dir not found or not set — will extract online "
                f"(type={conditioning_type})"
            )

    def _get_base_key(self, idx: int) -> str:
        """从原始图像文件名提取 base_key（与条件图像索引匹配）。"""
        fname = self.image_paths[idx].name
        for orig_suffix, _ in _KNOWN_SUFFIX_PAIRS:
            base_key = _strip_known_suffix(fname, orig_suffix)
            if base_key is not None:
                return base_key
        # 退回到 stem（方式 A 同名）
        return self.image_paths[idx].stem

    def _load_conditioning(self, idx: int, source_image: Image.Image) -> Image.Image:
        """加载或在线生成条件图像。"""
        if not self.online_extract:
            base_key = self._get_base_key(idx)
            cond_path = self._cond_index.get(base_key)
            if cond_path is not None:
                return Image.open(cond_path).convert("RGB")
            logger.warning(
                f"Conditioning image not found for base_key='{base_key}' "
                f"(original: {self.image_paths[idx].name}), falling back to online extraction"
            )

        if self.conditioning_type == "canny":
            arr = np.array(source_image)
            canny = make_canny(arr)
            return Image.fromarray(canny)

        raise ValueError(
            f"No conditioning image found for index {idx} and "
            f"online extraction not supported for type '{self.conditioning_type}'"
        )

    def __getitem__(self, idx: int) -> dict:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        conditioning_image = self._load_conditioning(idx, image)

        # 优先使用分桶分配的尺寸，未分配时退化为正方形
        target_size = self._get_target_size(idx)
        transformed = apply_transforms(
            image, target_size, self.transforms_dict, conditioning_image
        )

        result = {
            "pixel_values": transformed["pixel_values"],
            "conditioning_pixel_values": transformed["conditioning_pixel_values"],
            "input_ids": self.cached_input_ids,
        }

        if self.cached_input_ids_2 is not None:
            original_size = image.size  # PIL: (w, h)
            orig_h, orig_w = original_size[1], original_size[0]
            tgt_h, tgt_w = target_size[1], target_size[0]

            result["input_ids_2"] = self.cached_input_ids_2
            result["original_size"] = torch.tensor([orig_h, orig_w], dtype=torch.long)
            result["crop_top_left"] = torch.tensor([0, 0], dtype=torch.long)
            result["target_size"] = torch.tensor([tgt_h, tgt_w], dtype=torch.long)

        return result


class CachedLatentControlNetDataset(BaseImageDataset):
    """ControlNet 数据集：目标图使用预缓存 VAE latent，条件图仍在线加载。

    - pixel_values  → 从 .pt 文件读取预计算 latent（跳过 VAE encode）
    - conditioning_pixel_values → 每步从磁盘加载并 resize，flip 与 latent 保持同步
    """

    def __init__(
        self,
        data_dir: str,
        cache_dir: str,
        conditioning_data_dir: str,
        tokenizer,
        conditioning_type: str = "canny",
        center_crop: bool = False,
        random_flip: bool = True,
        tokenizer_2=None,
        fixed_caption: str = "",
        max_train_samples: int | None = None,
    ):
        super().__init__(data_dir, resolution=1024, center_crop=False, random_flip=False, max_train_samples=max_train_samples)
        self.cache_dir = Path(cache_dir)
        self.center_crop = center_crop
        self.do_random_flip = random_flip
        self.conditioning_type = conditioning_type

        self.conditioning_dir = Path(conditioning_data_dir) if conditioning_data_dir else None
        self.online_extract = self.conditioning_dir is None or not self.conditioning_dir.exists()

        if not self.online_extract:
            self._cond_index = _build_cond_index(self.conditioning_dir)
            logger.info(
                f"[CachedLatent] ControlNet conditioning index: {len(self._cond_index)} entries"
            )
        else:
            self._cond_index = {}

        self.cached_input_ids = tokenizer(
            fixed_caption,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        if tokenizer_2 is not None:
            self.cached_input_ids_2 = tokenizer_2(
                fixed_caption,
                padding="max_length",
                truncation=True,
                max_length=tokenizer_2.model_max_length,
                return_tensors="pt",
            ).input_ids.squeeze(0)
        else:
            self.cached_input_ids_2 = None

    def _get_base_key(self, idx: int) -> str:
        fname = self.image_paths[idx].name
        for orig_suffix, _ in _KNOWN_SUFFIX_PAIRS:
            base_key = _strip_known_suffix(fname, orig_suffix)
            if base_key is not None:
                return base_key
        return self.image_paths[idx].stem

    def _load_conditioning_image(self, idx: int) -> Image.Image:
        if not self.online_extract:
            base_key = self._get_base_key(idx)
            cond_path = self._cond_index.get(base_key)
            if cond_path is not None:
                return Image.open(cond_path).convert("RGB")
            logger.warning(
                f"Conditioning image not found for base_key='{base_key}', "
                "falling back to online extraction"
            )
        if self.conditioning_type == "canny":
            source = Image.open(self.image_paths[idx]).convert("RGB")
            return Image.fromarray(make_canny(np.array(source)))
        raise ValueError(
            f"No conditioning image for index {idx} and type '{self.conditioning_type}'"
        )

    def get_image_sizes(self) -> list[tuple[int, int]]:
        """从缓存文件读取 target_hw，避免重新打开原图。"""
        sizes = []
        for idx in range(len(self.image_paths)):
            stem = self.image_paths[idx].stem
            data = torch.load(
                self.cache_dir / f"{stem}.pt", map_location="cpu", weights_only=True
            )
            target_h, target_w = data["target_hw"].tolist()
            sizes.append((target_w, target_h))  # PIL 格式: (w, h)
        return sizes

    def __getitem__(self, idx: int) -> dict:
        stem = self.image_paths[idx].stem
        cache_file = self.cache_dir / f"{stem}.pt"
        data = torch.load(cache_file, map_location="cpu", weights_only=True)

        use_flip = self.do_random_flip and random.random() < 0.5
        latents = data["latent_flip"] if use_flip else data["latent"]

        orig_h, orig_w = data["original_hw"].tolist()
        target_h, target_w = data["target_hw"].tolist()
        target_size = (target_w, target_h)  # (w, h)

        # 加载条件图并 resize 到与 latent 对应的分辨率
        cond_image = self._load_conditioning_image(idx)
        resizer = AspectRatioResize(target_size, center_crop=self.center_crop)
        cond_image = resizer(cond_image)
        if use_flip:
            cond_image = TF.hflip(cond_image)
        cond_tensor = T.ToTensor()(cond_image)  # [0, 1]，条件图不做 [-1,1] 归一化

        result = {
            "latents": latents.float(),
            "conditioning_pixel_values": cond_tensor,
            "input_ids": self.cached_input_ids.clone(),
        }

        if "orig_max_channel" in data:
            omc = data["orig_max_channel"].float()  # (1, H/f, W/f)
            if use_flip:
                omc = omc.flip(-1)
            result["orig_max_channel"] = omc

        if self.cached_input_ids_2 is not None:
            result["input_ids_2"] = self.cached_input_ids_2.clone()
            result["original_size"] = torch.tensor([orig_h, orig_w], dtype=torch.long)
            result["crop_top_left"] = torch.zeros(2, dtype=torch.long)
            result["target_size"] = torch.tensor([target_h, target_w], dtype=torch.long)

        return result


class PixArtControlNetCachedLatentDataset(BaseImageDataset):
    """PixArt-Sigma ControlNet 数据集 — 支持双模式条件图编码。

    目标图使用预缓存 VAE latent（与 CachedLatentControlNetDataset 一致）。
    条件图根据 conditioning_mode 返回不同格式：
      - "vae" 模式: 从 conditioning_latent_cache 加载预缓存 latent
      - "cnn_encoder" 模式: 在线加载条件图为像素 [0, 1]

    使用 T5 tokenizer（含 attention_mask），而非 CLIP tokenizer。

    支持两种文本模式:
      1. 固定 caption (text_embed_cache_dir=None): 所有图像共用同一份 tokenized ids
      2. 逐图预编码 (text_embed_cache_dir 指定): 每张图像从磁盘加载预编码的 T5 嵌入
    """

    def __init__(
        self,
        data_dir: str,
        cache_dir: str,
        conditioning_data_dir: str,
        conditioning_latent_cache_dir: str | None,
        tokenizer,
        conditioning_mode: str = "vae",
        conditioning_type: str = "precomputed",
        center_crop: bool = False,
        random_flip: bool = True,
        fixed_caption: str = "",
        t5_max_length: int = 300,
        max_train_samples: int | None = None,
        text_embed_cache_dir: str | None = None,
        use_bucketing: bool = True,
        pad_color: tuple[int, ...] = (0, 0, 0),
    ):
        super().__init__(
            data_dir, resolution=1024, center_crop=False, random_flip=False,
            max_train_samples=max_train_samples,
        )
        self.cache_dir = Path(cache_dir)
        self.conditioning_mode = conditioning_mode
        self.center_crop = center_crop
        self.do_random_flip = random_flip
        self.conditioning_type = conditioning_type
        self.use_bucketing = use_bucketing
        self.pad_color = tuple(pad_color)

        # 条件图像索引（像素模式或 latent 回退时使用）
        self.conditioning_dir = Path(conditioning_data_dir) if conditioning_data_dir else None
        self.online_extract = self.conditioning_dir is None or not self.conditioning_dir.exists()
        if not self.online_extract:
            self._cond_index = _build_cond_index(self.conditioning_dir)
            logger.info(
                f"[PixArt-CN] conditioning index: {len(self._cond_index)} entries, mode={conditioning_mode}"
            )
            if conditioning_type == "precomputed":
                self._filter_unpaired_images()
        else:
            self._cond_index = {}

        # VAE 模式: 条件 latent 缓存目录
        self.conditioning_latent_cache_dir = (
            Path(conditioning_latent_cache_dir) if conditioning_latent_cache_dir else None
        )

        # 文本嵌入: 逐图预编码 vs 固定 caption
        self.text_embed_cache_dir = Path(text_embed_cache_dir) if text_embed_cache_dir else None

        if self.text_embed_cache_dir is None:
            tokenized = tokenizer(
                fixed_caption,
                padding="max_length",
                truncation=True,
                max_length=t5_max_length,
                return_tensors="pt",
            )
            self.cached_input_ids = tokenized.input_ids.squeeze(0)
            self.cached_attention_mask = tokenized.attention_mask.squeeze(0)

    def _get_base_key_from_path(self, path: Path) -> str:
        """从图像路径提取 base_key（不依赖 idx）。"""
        fname = path.name
        for orig_suffix, _ in _KNOWN_SUFFIX_PAIRS:
            base_key = _strip_known_suffix(fname, orig_suffix)
            if base_key is not None:
                return base_key
        return path.stem

    def _get_base_key(self, idx: int) -> str:
        return self._get_base_key_from_path(self.image_paths[idx])

    def _filter_unpaired_images(self):
        """过滤掉没有对应条件图的训练图（仅 precomputed 模式需要）。"""
        before = len(self.image_paths)
        self.image_paths = [
            p for p in self.image_paths
            if self._get_base_key_from_path(p) in self._cond_index
        ]
        after = len(self.image_paths)
        if before != after:
            logger.warning(
                f"[PixArt-CN] 过滤无配对条件图的训练图: {before} → {after} "
                f"(移除 {before - after} 张)"
            )

    def _load_conditioning_image(self, idx: int) -> Image.Image:
        if not self.online_extract:
            base_key = self._get_base_key(idx)
            cond_path = self._cond_index.get(base_key)
            if cond_path is not None:
                return Image.open(cond_path).convert("RGB")
            logger.warning(
                f"Conditioning image not found for base_key='{base_key}', "
                "falling back to online extraction"
            )
        if self.conditioning_type == "canny":
            source = Image.open(self.image_paths[idx]).convert("RGB")
            return Image.fromarray(make_canny(np.array(source)))
        raise ValueError(
            f"No conditioning image for index {idx} and type '{self.conditioning_type}'"
        )

    def __getitem__(self, idx: int) -> dict:
        # 目标 latent
        stem = self.image_paths[idx].stem
        cache_file = self.cache_dir / f"{stem}.pt"
        data = torch.load(cache_file, map_location="cpu", weights_only=True)

        use_flip = self.do_random_flip and random.random() < 0.5
        latents = data["latent_flip"] if use_flip else data["latent"]

        target_h, target_w = data["target_hw"].tolist()
        target_size = (target_w, target_h)

        # 文本嵌入: 逐图预编码 vs 固定 caption
        if self.text_embed_cache_dir is not None:
            text_file = self.text_embed_cache_dir / f"{stem}.pt"
            text_data = torch.load(text_file, map_location="cpu", weights_only=True)
            result = {
                "latents": latents.float(),
                "prompt_embeds": text_data["prompt_embeds"].squeeze(0).float(),
                "prompt_attention_mask": text_data["prompt_attention_mask"].squeeze(0),
            }
        else:
            result = {
                "latents": latents.float(),
                "input_ids": self.cached_input_ids.clone(),
                "attention_mask": self.cached_attention_mask.clone(),
            }

        # padding_mask: 非分桶模式下，目标 latent 预计算时已保存
        padding_mask = data.get("padding_mask", None)
        if padding_mask is not None:
            padding_mask = torch.flip(padding_mask, dims=[-1]) if use_flip else padding_mask
            result["padding_mask"] = padding_mask.float()

        # orig_max_channel: 辅助结构 loss 使用的预缓存二值 mask
        if "orig_max_channel" in data:
            omc = data["orig_max_channel"].float()
            if use_flip:
                omc = omc.flip(-1)
            result["orig_max_channel"] = omc

        # 条件数据
        if self.conditioning_mode == "vae" and self.conditioning_latent_cache_dir is not None:
            base_key = self._get_base_key(idx)
            cond_cache_file = self.conditioning_latent_cache_dir / f"{base_key}.pt"
            cond_data = torch.load(cond_cache_file, map_location="cpu", weights_only=True)
            cond_latent = cond_data["latent_flip"] if use_flip else cond_data["latent"]
            result["conditioning_latents"] = cond_latent.float()
        else:
            cond_image = self._load_conditioning_image(idx)
            if self.use_bucketing:
                resizer = AspectRatioResize(target_size, center_crop=self.center_crop)
                cond_image = resizer(cond_image)
            else:
                padder = AspectRatioPad(target_size, pad_color=self.pad_color)
                cond_image, _cond_mask = padder(cond_image)
            if use_flip:
                cond_image = TF.hflip(cond_image)
            cond_tensor = T.ToTensor()(cond_image)  # [0, 1]
            result["conditioning_pixel_values"] = cond_tensor

        return result
