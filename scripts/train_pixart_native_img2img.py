#!/usr/bin/env python3
"""PixArt-Sigma Native Img2Img 训练入口脚本。

用法:
  # 多卡训练 (仅新增模块)
  accelerate launch scripts/train_pixart_native_img2img.py \
    --config configs/pixart_sigma_native_img2img.yaml

  # 恢复训练
  accelerate launch scripts/train_pixart_native_img2img.py \
    --config configs/pixart_sigma_native_img2img.yaml --resume latest

  # 全量微调模式
  accelerate launch scripts/train_pixart_native_img2img.py \
    --config configs/pixart_sigma_native_img2img.yaml \
    --override training.phase3_mode=full_finetune
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from omegaconf import OmegaConf

from trainers.pixart_native_img2img_trainer import PixArtNativeImg2ImgTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train PixArt-Sigma Native Img2Img")
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
    parser.add_argument(
        "--base_config", type=str, default="configs/base.yaml",
        help="Path to base config YAML for default values",
    )
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint or 'latest'")
    parser.add_argument("--override", nargs="*", default=[], help="Override config values")
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    base_config = OmegaConf.load(args.base_config) if os.path.exists(args.base_config) else OmegaConf.create()
    task_config = OmegaConf.load(args.config)
    cli_overrides = OmegaConf.from_dotlist(args.override)

    config = OmegaConf.merge(base_config, task_config, cli_overrides)

    if args.resume:
        config.training.resume_from_checkpoint = args.resume

    seed = config.training.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logging.getLogger(__name__).info(f"Config:\n{OmegaConf.to_yaml(config)}")

    os.makedirs(config.training.get("output_dir", "./outputs"), exist_ok=True)

    trainer = PixArtNativeImg2ImgTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
