"""LoRA 权重融合脚本 — 将训练好的 LoRA delta 合并回基础模型。

输出一个标准 diffusers 格式的完整 pipeline 目录，可直接用于推理或作为
ControlNet 的 UNet 初始化来源。

用法:
  python scripts/merge_lora_weights.py \
    --base_model stabilityai/stable-diffusion-xl-base-1.0 \
    --lora_checkpoint ./outputs/lora_sdxl_floorplan/checkpoints/step_000810 \
    --output_dir ./weights/sdxl_lora_merged \
    --model_type sdxl \
    --lora_rank 32 --lora_alpha 32 \
    --lora_target_modules to_q to_k to_v to_out.0 ff.net.0.proj ff.net.2 \
    --weights_dir ./weights
"""

import argparse
import json
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


def parse_args():
    p = argparse.ArgumentParser(description="Merge LoRA weights into base model")
    p.add_argument("--base_model", type=str, required=True,
                   help="HuggingFace model ID or local path")
    p.add_argument("--lora_checkpoint", type=str, required=True,
                   help="Path to LoRA checkpoint directory (containing lora_unet.safetensors)")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output directory for merged model")
    p.add_argument("--model_type", type=str, default="sdxl", choices=["sd15", "sdxl"])
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=float, default=32.0)
    p.add_argument("--lora_target_modules", nargs="+",
                   default=["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"])
    p.add_argument("--weights_dir", type=str, default=None,
                   help="Local weights cache directory")
    p.add_argument("--merge_text_encoder", action="store_true",
                   help="Also merge LoRA into text encoder(s)")
    return p.parse_args()


def copy_subfolder(src_root: str, dst_root: str, subfolder: str):
    """复制子目录（如 vae/, tokenizer/ 等）。"""
    src = os.path.join(src_root, subfolder)
    dst = os.path.join(dst_root, subfolder)
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    elif os.path.isfile(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)


def main():
    args = parse_args()

    base_path = resolve_model_path(args.base_model, args.weights_dir, args.model_type)
    logger.info(f"Base model path: {base_path}")

    from diffusers import UNet2DConditionModel
    logger.info("Loading UNet...")
    unet = UNet2DConditionModel.from_pretrained(base_path, subfolder="unet", torch_dtype=torch.float32)

    logger.info(f"Injecting LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})...")
    injected = LoRAInjector.inject_unet(unet, args.lora_rank, args.lora_alpha, args.lora_target_modules)
    logger.info(f"Injected {len(injected)} LoRA layers")

    lora_unet_path = os.path.join(args.lora_checkpoint, "lora_unet.safetensors")
    if not os.path.exists(lora_unet_path):
        raise FileNotFoundError(f"LoRA UNet weights not found: {lora_unet_path}")
    logger.info(f"Loading LoRA weights from {lora_unet_path}")
    load_lora_weights(unet, lora_unet_path)

    logger.info("Merging LoRA into base UNet...")
    merge_lora_to_base(unet)

    os.makedirs(args.output_dir, exist_ok=True)
    unet_dir = os.path.join(args.output_dir, "unet")
    unet.save_pretrained(unet_dir)
    logger.info(f"Merged UNet saved to {unet_dir}")

    if args.merge_text_encoder:
        _merge_text_encoders(args, base_path)

    shared_subfolders = ["vae", "tokenizer", "scheduler"]
    if args.model_type == "sdxl":
        shared_subfolders.extend(["tokenizer_2"])
        if not args.merge_text_encoder:
            shared_subfolders.extend(["text_encoder", "text_encoder_2"])
    else:
        if not args.merge_text_encoder:
            shared_subfolders.append("text_encoder")

    for subfolder in shared_subfolders:
        logger.info(f"Copying {subfolder}/...")
        copy_subfolder(base_path, args.output_dir, subfolder)

    model_index_src = os.path.join(base_path, "model_index.json")
    if os.path.exists(model_index_src):
        shutil.copy2(model_index_src, os.path.join(args.output_dir, "model_index.json"))

    logger.info(f"Merge complete! Output: {args.output_dir}")


def _merge_text_encoders(args, base_path: str):
    """合并 text encoder 的 LoRA 权重（如果训练了的话）。"""
    from transformers import CLIPTextModel, CLIPTextModelWithProjection

    lora_te_path = os.path.join(args.lora_checkpoint, "lora_text_encoder.safetensors")
    if os.path.exists(lora_te_path):
        logger.info("Merging LoRA into TextEncoder...")
        te = CLIPTextModel.from_pretrained(base_path, subfolder="text_encoder", torch_dtype=torch.float32)
        LoRAInjector.inject_text_encoder(te, args.lora_rank, args.lora_alpha)
        load_lora_weights(te, lora_te_path)
        merge_lora_to_base(te)
        te_dir = os.path.join(args.output_dir, "text_encoder")
        te.save_pretrained(te_dir)
        logger.info(f"Merged TextEncoder saved to {te_dir}")

    if args.model_type == "sdxl":
        lora_te2_path = os.path.join(args.lora_checkpoint, "lora_text_encoder_2.safetensors")
        if os.path.exists(lora_te2_path):
            logger.info("Merging LoRA into TextEncoder2...")
            te2 = CLIPTextModelWithProjection.from_pretrained(
                base_path, subfolder="text_encoder_2", torch_dtype=torch.float32
            )
            LoRAInjector.inject_text_encoder(te2, args.lora_rank, args.lora_alpha)
            load_lora_weights(te2, lora_te2_path)
            merge_lora_to_base(te2)
            te2_dir = os.path.join(args.output_dir, "text_encoder_2")
            te2.save_pretrained(te2_dir)
            logger.info(f"Merged TextEncoder2 saved to {te2_dir}")


if __name__ == "__main__":
    main()
