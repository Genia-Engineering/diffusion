#!/usr/bin/env python3
"""Sana 0.6B LoRA 微调训练入口脚本。

用法:
  # 单卡训练
  python scripts/train_sana_lora.py --config configs/lora_sana_floorplan.yaml

  # 多卡训练 (accelerate)
  accelerate launch scripts/train_sana_lora.py --config configs/lora_sana_floorplan.yaml

  # 恢复训练
  accelerate launch scripts/train_sana_lora.py \
    --config configs/lora_sana_floorplan.yaml --resume latest
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from omegaconf import OmegaConf

from trainers.sana_lora_trainer import SanaLoRATrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Sana 0.6B LoRA")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML",
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
        help="Override config values, e.g. training.learning_rate=2e-4",
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

    trainer = SanaLoRATrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
