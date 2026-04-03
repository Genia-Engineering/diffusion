#!/usr/bin/env python3
"""PixArt-Sigma LoRA 权重融合脚本 — 将训练好的 LoRA delta 合并回 Transformer 基础权重。

输出一个标准 diffusers 格式的 Transformer 目录，可直接用于推理或作为
Phase 3 架构手术的基础模型。

用法:
  python scripts/merge_pixart_lora.py \
    --base_model PixArt-alpha/PixArt-Sigma-XL-2-1024-MS \
    --lora_checkpoint ./outputs/pixart_sigma_lora_floorplan/checkpoints/step_002000 \
    --output_dir ./weights/pixart_lora_merged \
    --lora_rank 128 --lora_alpha 128 \
    --weights_dir ./weights
"""

import argparse
import logging
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from models.lora import LoRAInjector, load_lora_weights, merge_lora_to_base
from models.model_loader import resolve_model_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PIXART_DEFAULT_TARGET_MODULES = [
    "to_q", "to_k", "to_v", "to_out.0",
    "ff.net.0.proj", "ff.net.2",
]


def parse_args():
    p = argparse.ArgumentParser(description="Merge LoRA weights into PixArt-Sigma Transformer")
    p.add_argument("--base_model", type=str, required=True,
                   help="HuggingFace model ID or local path (e.g. PixArt-alpha/PixArt-Sigma-XL-2-1024-MS)")
    p.add_argument("--lora_checkpoint", type=str, required=True,
                   help="Path to LoRA checkpoint directory (containing lora_transformer.safetensors)")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output directory for merged model")
    p.add_argument("--lora_rank", type=int, default=128)
    p.add_argument("--lora_alpha", type=float, default=128.0)
    p.add_argument("--lora_target_modules", nargs="+", default=PIXART_DEFAULT_TARGET_MODULES)
    p.add_argument("--weights_dir", type=str, default=None,
                   help="Local weights cache directory")
    return p.parse_args()


def main():
    args = parse_args()

    base_path = resolve_model_path(args.base_model, args.weights_dir, "pixart_sigma")
    logger.info(f"Base model path: {base_path}")

    from diffusers import PixArtTransformer2DModel
    logger.info("Loading PixArt Transformer...")
    transformer = PixArtTransformer2DModel.from_pretrained(
        base_path, subfolder="transformer", torch_dtype=torch.float32,
    )

    logger.info(f"Injecting LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})...")
    injected = LoRAInjector.inject(
        transformer, args.lora_rank, args.lora_alpha, args.lora_target_modules,
    )
    logger.info(f"Injected {len(injected)} LoRA layers")

    lora_path = os.path.join(args.lora_checkpoint, "lora_transformer.safetensors")
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA Transformer weights not found: {lora_path}")
    logger.info(f"Loading LoRA weights from {lora_path}")
    load_lora_weights(transformer, lora_path)

    logger.info("Merging LoRA into base Transformer...")
    merge_lora_to_base(transformer)

    os.makedirs(args.output_dir, exist_ok=True)
    tf_dir = os.path.join(args.output_dir, "transformer")
    transformer.save_pretrained(tf_dir)
    logger.info(f"Merged Transformer saved to {tf_dir}")

    for subfolder in ["vae", "scheduler", "tokenizer", "model_index.json"]:
        src = os.path.join(base_path, subfolder)
        dst = os.path.join(args.output_dir, subfolder)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
            logger.info(f"Copied {subfolder}/")
        elif os.path.isfile(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            logger.info(f"Copied {subfolder}")

    logger.info(f"Merge complete! Output: {args.output_dir}")


if __name__ == "__main__":
    main()
