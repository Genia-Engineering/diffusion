#!/usr/bin/env python3
"""训练前权重预下载脚本 — 只拉取训练必需的子目录，跳过 safety_checker 等无关文件。

用法:
  # 从 config 文件自动读取模型信息（推荐）
  python scripts/download_weights.py --config configs/lora_sdxl_floorplan.yaml

  # 手动指定参数
  python scripts/download_weights.py \\
      --repo_id stabilityai/stable-diffusion-xl-base-1.0 \\
      --model_type sdxl \\
      --weights_dir /home/daiqing_tan/stable_diffusion_lora/weights

  # SD1.5
  python scripts/download_weights.py --config configs/lora_sd15_floorplan.yaml
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="预下载 SD/SDXL 训练权重")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--config",
        type=str,
        help="训练 config YAML 路径，自动读取 model.pretrained_model_name_or_path / "
             "model.model_type / model.weights_dir",
    )
    group.add_argument(
        "--repo_id",
        type=str,
        help="HuggingFace model ID，例如 stabilityai/stable-diffusion-xl-base-1.0",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["sd15", "sdxl"],
        default="sdxl",
        help="模型类型（仅 --repo_id 模式下需要指定，默认 sdxl）",
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        default=None,
        help="本地权重保存根目录（仅 --repo_id 模式下需要指定）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.config:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(args.config)
        repo_id     = cfg.model.pretrained_model_name_or_path
        model_type  = cfg.model.get("model_type", "sd15")
        weights_dir = cfg.model.get("weights_dir", None)
    else:
        repo_id     = args.repo_id
        model_type  = args.model_type
        weights_dir = args.weights_dir

    logger.info("=" * 60)
    logger.info(f"  repo_id     : {repo_id}")
    logger.info(f"  model_type  : {model_type}")
    logger.info(f"  weights_dir : {weights_dir or '(HuggingFace 默认缓存)'}")
    logger.info("=" * 60)

    from models.model_loader import resolve_model_path
    local_path = resolve_model_path(repo_id, weights_dir, model_type=model_type)

    logger.info(f"权重就绪，路径: {local_path}")


if __name__ == "__main__":
    main()
