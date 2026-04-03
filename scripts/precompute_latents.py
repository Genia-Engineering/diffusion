#!/usr/bin/env python3
"""VAE Latent 多卡并行预计算脚本。

每张 GPU 处理数据集的 1/N 分片，结果写入同一个目录，互不重叠，无需跨卡通信。

用法:
  # 4 卡并行（推荐）
  accelerate launch --num_processes 4 scripts/precompute_latents.py --config configs/controlnet_sdxl.yaml

  # 指定 GPU
  CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 \
      scripts/precompute_latents.py --config configs/controlnet_sdxl.yaml

  # 单卡调试
  python scripts/precompute_latents.py --config configs/lora_sdxl.yaml
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from pathlib import Path
from PIL import Image as PIL_Image
from tqdm import tqdm

from data.buckets import BucketManager
from data.dataset import BaseImageDataset
from data.transforms import AspectRatioPad, AspectRatioResize
from models.model_loader import load_sd15_components, load_sdxl_components

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="多卡并行预计算 VAE latents")
    parser.add_argument(
        "--config", type=str, required=True,
        help="训练配置文件路径（e.g. configs/lora_sdxl.yaml）",
    )
    parser.add_argument(
        "--base_config", type=str, default="configs/base.yaml",
        help="基础配置文件路径",
    )
    parser.add_argument(
        "--model_type", type=str, default=None,
        choices=["sd15", "sdxl"],
        help="模型类型，不指定时从配置文件读取",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── 加载配置 ────────────────────────────────────────────────
    base_cfg = OmegaConf.load(args.base_config)
    task_cfg = OmegaConf.load(args.config)
    config = OmegaConf.merge(base_cfg, task_cfg)

    model_type = args.model_type or config.model.get("model_type", "sdxl")
    data_cfg = config.data
    resolution = data_cfg.get("resolution", 512 if model_type == "sd15" else 1024)
    center_crop = data_cfg.get("center_crop", False)
    use_bucketing = data_cfg.get("use_aspect_ratio_bucketing", True)
    pad_color = tuple(data_cfg.get("pad_color", [0, 0, 0]))

    # ── 初始化 Accelerator（纯推理，无需优化器/调度器） ──────────
    accelerator = Accelerator(mixed_precision="bf16")
    set_seed(42)

    rank = accelerator.process_index
    world_size = accelerator.num_processes
    device = accelerator.device

    if accelerator.is_main_process:
        logger.info(f"启动多卡 VAE latent 预计算：{world_size} 卡并行")
        logger.info(f"模型类型: {model_type} | 分辨率: {resolution} | 配置: {args.config}")

    # ── 构建数据集 & 分配分桶 ────────────────────────────────────
    dataset = BaseImageDataset(
        data_dir=data_cfg.train_data_dir,
        resolution=resolution,
        center_crop=center_crop,
        random_flip=False,
    )

    if use_bucketing:
        bucket_manager = BucketManager(model_type=model_type)
        image_sizes = dataset.get_image_sizes()
        bucket_to_indices = bucket_manager.assign_buckets(image_sizes)
        dataset.set_bucket_assignments(bucket_to_indices)

    n_total = len(dataset)

    # ── 按 rank 切分索引（每卡处理不重叠的分片） ─────────────────
    # 简单均分，余数分给前几个 rank
    base_count = n_total // world_size
    remainder = n_total % world_size
    start = rank * base_count + min(rank, remainder)
    end = start + base_count + (1 if rank < remainder else 0)
    my_indices = list(range(start, end))

    # ── 缓存目录（优先读配置文件中的显式路径） ───────────────────
    training_cfg = config.get("training", {})
    explicit_cache_dir = training_cfg.get("latent_cache_dir", None)
    if explicit_cache_dir:
        cache_dir = Path(explicit_cache_dir)
    else:
        parent = Path(data_cfg.train_data_dir).parent
        suffix = "" if use_bucketing else "_pad"
        cache_dir = parent / f"latent_cache_{model_type}{suffix}"
    done_marker = cache_dir / ".precompute_done"

    if done_marker.exists():
        # 验证是否真的完整
        n_cached = sum(1 for i in range(n_total) if (cache_dir / f"{i:06d}.pt").exists())
        if n_cached == n_total:
            if accelerator.is_main_process:
                logger.info(f"缓存已完整（{n_total} 张），无需重新计算。删除 {done_marker} 可强制重算。")
            return

    if accelerator.is_main_process:
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"缓存目录: {cache_dir}")
        logger.info(f"总图片数: {n_total}，每卡约 {base_count} 张")

    accelerator.wait_for_everyone()

    # ── 加载 VAE（只取 vae 组件，其余丢弃） ─────────────────────
    model_name = config.model.get("pretrained_model_name_or_path")
    weights_dir = config.model.get("weights_dir", "./weights")

    logger.info(f"[Rank {rank}] 加载 VAE...")
    loader = load_sdxl_components if model_type == "sdxl" else load_sd15_components
    components = loader(model_name, weights_dir=weights_dir, dtype=torch.float32)
    vae = components["vae"].to(device=device)
    # 释放其余组件，节省内存
    for key in list(components.keys()):
        if key != "vae":
            del components[key]
    del components
    torch.cuda.empty_cache()
    vae.eval()
    # 预计算期间禁用 slicing/tiling，整图 encode 无分块伪影
    vae.disable_slicing()
    vae.disable_tiling()

    # ── 图像预处理工具 ────────────────────────────────────────────
    to_tensor = T.ToTensor()
    normalize = T.Normalize([0.5], [0.5])

    # ── 本 rank 的计算任务 ────────────────────────────────────────
    my_pending = [i for i in my_indices if not (cache_dir / f"{i:06d}.pt").exists()]

    logger.info(f"[Rank {rank}] 分配 {len(my_indices)} 张，待计算 {len(my_pending)} 张")

    if my_pending:
        for idx in tqdm(
            my_pending,
            desc=f"[Rank {rank}] 预计算 latents",
            position=rank,
            leave=True,
        ):
            out_path = cache_dir / f"{idx:06d}.pt"

            image = PIL_Image.open(dataset.image_paths[idx]).convert("RGB")
            orig_w, orig_h = image.size

            target_size = dataset._get_target_size(idx)   # (w, h)
            target_w, target_h = target_size

            pad_mask_tensor = None
            if use_bucketing:
                resizer = AspectRatioResize(target_size, center_crop=center_crop)
                image_resized = resizer(image)
            else:
                padder = AspectRatioPad(target_size, pad_color=pad_color)
                image_resized, pad_mask_pil = padder(image)
                pad_mask_tensor = to_tensor(pad_mask_pil)  # (1, H, W)

            img_t = normalize(to_tensor(image_resized)).unsqueeze(0).to(device)
            with torch.no_grad():
                latent = vae.encode(img_t).latent_dist.mode()
                latent = latent * vae.config.scaling_factor
            latent = latent.squeeze(0).to(dtype=torch.float16).cpu()

            img_t_flip = normalize(to_tensor(TF.hflip(image_resized))).unsqueeze(0).to(device)
            with torch.no_grad():
                latent_flip = vae.encode(img_t_flip).latent_dist.mode()
                latent_flip = latent_flip * vae.config.scaling_factor
            latent_flip = latent_flip.squeeze(0).to(dtype=torch.float16).cpu()

            save_dict = {
                "latent": latent,
                "latent_flip": latent_flip,
                "original_hw": torch.tensor([orig_h, orig_w], dtype=torch.long),
                "target_hw": torch.tensor([target_h, target_w], dtype=torch.long),
                "source_filename": dataset.image_paths[idx].name,
            }
            if pad_mask_tensor is not None:
                latent_h, latent_w = latent.shape[-2], latent.shape[-1]
                pm = torch.nn.functional.interpolate(
                    pad_mask_tensor.unsqueeze(0),
                    size=(latent_h, latent_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).to(torch.float16)
                save_dict["padding_mask"] = pm

            torch.save(save_dict, out_path)

    # ── 等待所有卡完成，由 rank 0 写完成标记 ─────────────────────
    # 使用文件轮询代替 NCCL barrier，避免 600s 超时
    rank_done_marker = cache_dir / f".rank_{rank}_done"
    rank_done_marker.touch()
    logger.info(f"[Rank {rank}] 本卡计算完成")

    if accelerator.is_main_process:
        import time
        logger.info("等待所有 rank 完成...")
        for r in range(world_size):
            marker = cache_dir / f".rank_{r}_done"
            while not marker.exists():
                time.sleep(2)

        # 验证完整性
        n_cached = sum(1 for i in range(n_total) if (cache_dir / f"{i:06d}.pt").exists())
        if n_cached == n_total:
            done_marker.touch()
            # 清理各 rank 临时标记
            for r in range(world_size):
                (cache_dir / f".rank_{r}_done").unlink(missing_ok=True)
            logger.info(f"全部完成！{n_total} 张图片 latent 已缓存至 {cache_dir}")
        else:
            logger.error(f"缓存不完整：期望 {n_total} 张，实际 {n_cached} 张，请检查错误日志")


if __name__ == "__main__":
    main()
