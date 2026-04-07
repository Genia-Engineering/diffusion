"""Aspect Ratio Bucketing — 将不同宽高比的图像分桶，保证同 batch 内分辨率一致。"""

import math
import random
from collections import defaultdict

from PIL import Image
from torch.utils.data import Sampler


# 预定义的宽高比桶 (width, height)
SD15_BUCKETS = [
    (512, 512),
    (448, 576), (576, 448),
    (384, 640), (640, 384),
    (384, 704), (704, 384),
    (320, 768), (768, 320),
    (512, 768), (768, 512),
]

SDXL_BUCKETS = [
    (1024, 1024),
    (896, 1152), (1152, 896),
    (832, 1216), (1216, 832),
    (768, 1344), (1344, 768),
    (640, 1536), (1536, 640),
    (1024, 768), (768, 1024),
]

PIXART_SIGMA_BUCKETS = [
    (1024, 1024),
    (960, 1088), (1088, 960),
    (896, 1152), (1152, 896),
    (832, 1216), (1216, 832),
    (768, 1344), (1344, 768),
    (704, 1408), (1408, 704),
    (640, 1536), (1536, 640),
    (576, 1664), (1664, 576),
    (512, 1792), (1792, 512),
]


class BucketManager:
    """管理宽高比分桶，为每张图像分配最近的桶。"""

    def __init__(self, model_type: str = "sd15", custom_buckets: list = None):
        if custom_buckets:
            self.buckets = custom_buckets
        elif model_type in ("pixart_sigma", "sana"):
            self.buckets = PIXART_SIGMA_BUCKETS
        elif model_type == "sdxl":
            self.buckets = SDXL_BUCKETS
        else:
            self.buckets = SD15_BUCKETS

        self.bucket_ratios = [w / h for w, h in self.buckets]

    def get_bucket(self, width: int, height: int) -> tuple[int, int]:
        """返回与给定宽高比最接近的桶分辨率。"""
        ratio = width / height
        distances = [abs(ratio - br) for br in self.bucket_ratios]
        idx = distances.index(min(distances))
        return self.buckets[idx]

    def resize_to_bucket(
        self,
        image: Image.Image,
        target_w: int,
        target_h: int,
        ratio_thresh: float = 0.2,
        pad_color: tuple[int, int, int] = (0, 0, 0),
    ) -> Image.Image:
        """将图像缩放至桶的精确尺寸，根据宽高比与所有预定义桶的距离自动选择策略。

        判断逻辑：
          - 计算图像宽高比与**所有预定义桶**中最近一个的 log 差值。
          - 若最小差值 <= ratio_thresh：宽高比接近某个桶 → 中心裁剪（无黑边）。
          - 若最小差值 >  ratio_thresh：与所有桶差距均大 → 等比缩放 + 居中 padding。

        Args:
            image:        PIL 图像（RGB）
            target_w:     目标桶宽度
            target_h:     目标桶高度
            ratio_thresh: log 比值阈值（默认 0.2 ≈ 22% 差异），超过则使用 padding
            pad_color:    padding 填充颜色，默认黑色 (0, 0, 0)
        """
        src_w, src_h = image.size
        src_ratio = src_w / src_h

        # 与所有预定义桶比较，取最小 log 距离
        min_log_dist = min(abs(math.log(src_ratio / br)) for br in self.bucket_ratios)
        use_padding = min_log_dist > ratio_thresh

        if use_padding:
            # 等比缩小使图像完整 fit 进桶（长边对齐），居中后补 pad_color
            scale = min(target_w / src_w, target_h / src_h)
            new_w = round(src_w * scale)
            new_h = round(src_h * scale)
            image = image.resize((new_w, new_h), Image.LANCZOS)
            canvas = Image.new("RGB", (target_w, target_h), pad_color)
            paste_x = (target_w - new_w) // 2
            paste_y = (target_h - new_h) // 2
            canvas.paste(image, (paste_x, paste_y))
            return canvas
        else:
            # 等比放大覆盖目标尺寸，再中心裁剪
            scale = max(target_w / src_w, target_h / src_h)
            new_w = math.ceil(src_w * scale)
            new_h = math.ceil(src_h * scale)
            image = image.resize((new_w, new_h), Image.LANCZOS)
            left = (new_w - target_w) // 2
            top = (new_h - target_h) // 2
            return image.crop((left, top, left + target_w, top + target_h))

    def assign_buckets(
        self, image_sizes: list[tuple[int, int]]
    ) -> dict[tuple[int, int], list[int]]:
        """将所有图像索引按桶分组。

        Args:
            image_sizes: [(width, height), ...] 列表

        Returns:
            {(bucket_w, bucket_h): [image_index, ...]}
        """
        bucket_to_indices = defaultdict(list)
        for idx, (w, h) in enumerate(image_sizes):
            bucket = self.get_bucket(w, h)
            bucket_to_indices[bucket].append(idx)
        return dict(bucket_to_indices)


class BucketSampler(Sampler):
    """自定义 Sampler: 同一 batch 内的图像来自同一宽高比桶。

    每个 epoch 打乱桶内顺序和桶间顺序，不足一个 batch 的尾部丢弃。

    DDP 分片由 accelerator.prepare(dataloader) 中的 BatchSamplerShard 负责，
    本 Sampler 始终返回全量索引，不感知 rank/num_replicas。
    """

    def __init__(
        self,
        bucket_to_indices: dict[tuple[int, int], list[int]],
        batch_size: int,
        drop_last: bool = True,
        shuffle: bool = True,
    ):
        self.bucket_to_indices = bucket_to_indices
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    def __iter__(self):
        all_batches: list[list[int]] = []

        for indices in self.bucket_to_indices.values():
            indices = list(indices)
            if self.shuffle:
                random.shuffle(indices)

            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    all_batches.append(batch)

        if self.shuffle:
            random.shuffle(all_batches)

        for batch in all_batches:
            yield from batch

    def __len__(self):
        total = 0
        for indices in self.bucket_to_indices.values():
            n_batches = len(indices) // self.batch_size
            if not self.drop_last and len(indices) % self.batch_size > 0:
                n_batches += 1
            total += n_batches * self.batch_size
        return total
