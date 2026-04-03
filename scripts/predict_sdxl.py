"""SDXL 推理脚本 — LoRA 融合模型文生图，附带显存追踪。

调度器选择:
  DPMSolverSDEScheduler + Karras σ
  - SDE 随机采样带来更丰富的细节和纹理多样性
  - Karras σ 高噪密采低噪快收，25 步即可获得高质量结果
  - SDXL 社区广泛验证的最佳调度器之一

显存估算 (bf16, 1024×1024):
  UNet(bf16)    ~4.8 GB   (磁盘 fp32 9.6GB ÷ 2)
  TE2-OpenCLIP  ~1.3 GB   (磁盘 fp32 2.6GB ÷ 2)
  TE1-CLIP-L    ~0.24 GB  (磁盘 fp32 470MB ÷ 2)
  VAE(bf16)     ~0.16 GB  (磁盘 fp32 320MB ÷ 2)
  ─────────────────────────
  模型权重      ~6.5 GB
  推理激活值    ~2-3 GB   (batch=4, 1024×1024)
  PyTorch/CUDA  ~0.5 GB   (context + allocator)
  ─────────────────────────
  峰值总计      ~9-10 GB

用法:
  python scripts/predict_sdxl.py --gpu 3
  python scripts/predict_sdxl.py --gpu 3 --num_images 8 --seed 123
  python scripts/predict_sdxl.py --gpu 3 --prompt "modern house floor plan, clean lines"
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from PIL import Image, PngImagePlugin

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MERGED_MODEL = str(
    _PROJECT_ROOT / "outputs/lora_sdxl_floorplan/merged_step_000810"
)
_DEFAULT_PROMPT = "architectural floor plan, blueprint, technical drawing"
_DEFAULT_NEGATIVE = "blurry, noisy, low quality, photo, realistic, watermark"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SDXL 推理（LoRA merged 模型）+ 显存追踪",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--merged_model_path", type=str, default=_DEFAULT_MERGED_MODEL)
    p.add_argument("--prompt", nargs="+", default=[_DEFAULT_PROMPT])
    p.add_argument("--negative_prompt", type=str, default=_DEFAULT_NEGATIVE)
    p.add_argument("--num_inference_steps", type=int, default=25)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--num_images", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--gpu", type=int, default=3)
    p.add_argument(
        "--output_dir", type=str,
        default=str(_PROJECT_ROOT / "outputs/predict_sdxl"),
    )
    return p.parse_args()


def _dtype_map(s: str) -> torch.dtype:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[s]


def _prompt_tag(prompt: str, max_len: int = 32) -> str:
    slug = prompt[:max_len].strip().replace(" ", "_").replace(",", "").replace("/", "-")
    h = hashlib.md5(prompt.encode()).hexdigest()[:6]
    return f"{slug}__{h}"


def log_vram(label: str, device: torch.device) -> dict:
    """记录并打印当前 VRAM 使用情况，返回数值字典 (MB)。"""
    if device.type != "cuda":
        return {}
    torch.cuda.synchronize(device)
    allocated = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
    logger.info(
        f"[VRAM] {label}: "
        f"allocated={allocated:.0f}MB, reserved={reserved:.0f}MB, "
        f"peak={max_allocated:.0f}MB"
    )
    return {
        "label": label,
        "allocated_mb": round(allocated, 1),
        "reserved_mb": round(reserved, 1),
        "peak_mb": round(max_allocated, 1),
    }


def build_pipeline(args, dtype, device):
    """加载 SDXL pipeline，配置 DPMSolverSDEScheduler + Karras σ。"""
    from diffusers import DPMSolverSDEScheduler, StableDiffusionXLPipeline

    log_vram("加载前", device)

    logger.info(f"加载 SDXL 模型: {args.merged_model_path}")
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        args.merged_model_path, torch_dtype=dtype,
    )
    log_vram("模型加载后 (CPU→GPU 前)", device)

    pipeline = pipeline.to(device)
    vram_after_load = log_vram("模型上 GPU 后", device)

    pipeline.scheduler = DPMSolverSDEScheduler.from_config(
        pipeline.scheduler.config,
        use_karras_sigmas=True,
    )
    logger.info("调度器: DPMSolverSDEScheduler (Karras σ)")

    pipeline.set_progress_bar_config(desc="推理", leave=False)
    return pipeline, vram_after_load


def run(pipeline, args, device):
    """遍历每个 prompt 生成图像，追踪显存峰值。"""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.cuda.reset_peak_memory_stats(device)
    vram_snapshots = [log_vram("推理开始前", device)]
    all_saved: list[Path] = []

    for pi, prompt in enumerate(args.prompt):
        tag = _prompt_tag(prompt)
        logger.info(f"[Prompt {pi + 1}/{len(args.prompt)}] {prompt!r}")

        seed = args.seed if args.seed >= 0 else torch.randint(0, 2**31, (1,)).item()
        generator = torch.Generator(device=device).manual_seed(seed)

        with torch.inference_mode():
            output = pipeline(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                width=args.width,
                height=args.height,
                num_images_per_prompt=args.num_images,
                generator=generator,
            )

        vram_snapshots.append(log_vram(f"Prompt {pi + 1} 推理完成", device))

        for img_idx, img in enumerate(output.images):
            stem = f"p{pi:02d}_{tag}_s{seed}_i{img_idx:02d}"
            save_path = output_dir / f"{stem}.png"

            png_info = PngImagePlugin.PngInfo()
            png_info.add_text("prompt", prompt)
            png_info.add_text("negative_prompt", args.negative_prompt)
            png_info.add_text("seed", str(seed))
            png_info.add_text("steps", str(args.num_inference_steps))
            png_info.add_text("guidance_scale", str(args.guidance_scale))
            png_info.add_text("scheduler", "DPMSolverSDEScheduler_karras")
            png_info.add_text("model", args.merged_model_path)

            img.save(save_path, pnginfo=png_info)
            all_saved.append(save_path)
            logger.info(f"  保存: {save_path}")

    # 最终显存统计
    final_snap = log_vram("全部推理完成", device)
    vram_snapshots.append(final_snap)

    peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    reserved_mb = torch.cuda.memory_reserved(device) / 1024**2
    logger.info("=" * 60)
    logger.info(f"显存峰值 (allocated): {peak_mb:.0f} MB ({peak_mb / 1024:.2f} GB)")
    logger.info(f"显存保留 (reserved):  {reserved_mb:.0f} MB ({reserved_mb / 1024:.2f} GB)")
    logger.info("=" * 60)

    # 保存元数据 + 显存报告
    meta = {
        "merged_model_path": args.merged_model_path,
        "scheduler": "DPMSolverSDEScheduler (Karras σ)",
        "prompts": args.prompt,
        "negative_prompt": args.negative_prompt,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "resolution": f"{args.width}x{args.height}",
        "num_images": args.num_images,
        "seed": args.seed,
        "dtype": args.dtype,
        "device": str(device),
        "vram_peak_mb": round(peak_mb, 1),
        "vram_reserved_mb": round(reserved_mb, 1),
        "vram_snapshots": vram_snapshots,
    }
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info(f"元数据: {meta_path}")

    return all_saved


def main():
    args = parse_args()

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logger.warning("CUDA 不可用，降级为 CPU")

    dtype = _dtype_map(args.dtype)
    logger.info(f"设备: {device} | 精度: {args.dtype}")

    pipeline, vram_model = build_pipeline(args, dtype, device)
    saved = run(pipeline, args, device)

    logger.info(f"完成！共生成 {len(saved)} 张图像 → {args.output_dir}")


if __name__ == "__main__":
    main()
