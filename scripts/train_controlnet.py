#!/usr/bin/env python3
"""ControlNet 训练入口脚本。

用法:
  # 单卡训练
  python scripts/train_controlnet.py --config configs/controlnet_sd15.yaml

  # 多卡训练 (accelerate)
  accelerate launch scripts/train_controlnet.py --config configs/controlnet_sdxl.yaml

  # 联合 LoRA + ControlNet 训练（在 config 中配置 lora 段）
  accelerate launch scripts/train_controlnet.py \
    --config configs/controlnet_sd15.yaml \
    --override lora.rank=16 lora.alpha=16

  # 恢复训练
  accelerate launch scripts/train_controlnet.py \
    --config configs/controlnet_sd15.yaml \
    --resume latest
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from omegaconf import OmegaConf

from trainers.controlnet_trainer import ControlNetTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train ControlNet for Stable Diffusion")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML (e.g. configs/controlnet_sd15.yaml)",
    )
    parser.add_argument(
        "--base_config",
        type=str,
        default="configs/base.yaml",
        help="Path to base config YAML for default values",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint directory, or 'latest' for auto-detect",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Override config values, e.g. training.learning_rate=2e-5",
    )
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

    trainer = ControlNetTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
