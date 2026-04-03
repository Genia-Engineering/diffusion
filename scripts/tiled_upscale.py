"""SDXL ControlNet Tile — 分块超分辨率后处理脚本。

将低分辨率图像通过 Lanczos 预放大后，切分为重叠瓦片，逐块使用
SDXL + ControlNet Tile (TTPlanet) 进行 img2img 细节增强，最后
线性渐变加权融合消除接缝，输出高分辨率图像。

典型用法:
  # 2x 放大，strength=0.2（最大保留结构）
  python3 scripts/tiled_upscale.py \
    --input_image outputs/pixart_sigma_floorplan/samples/step_002000/cfg4.5_prompt_00_img_03.png \
    --gpu 3

  # strength=0.35（更多细节增强）
  python3 scripts/tiled_upscale.py \
    --input_image outputs/pixart_sigma_floorplan/samples/step_002000/cfg4.5_prompt_00_img_03.png \
    --strength 0.35 --gpu 3

  # 批量处理多张图片
  python scripts/tiled_upscale.py \
    --input_image img1.png img2.png --gpu 3

显存估算 (fp16, 1024x1024 tile):
  SDXL UNet          ~4.8 GB
  ControlNet Tile    ~2.5 GB
  VAE + TE1 + TE2    ~1.7 GB
  推理激活值          ~2-3 GB
  ────────────────────────
  峰值               ~12 GB
"""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_PROMPT = "architectural floor plan, blueprint, technical drawing, high quality, sharp lines"
_DEFAULT_NEGATIVE = "blurry, noisy, low quality, watermark, artifacts, pixelated"


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SDXL ControlNet Tile 分块超分辨率",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    mg = p.add_argument_group("模型")
    mg.add_argument(
        "--base_model_path", type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="SDXL base 模型 (HF ID 或本地路径)",
    )
    mg.add_argument(
        "--controlnet_path", type=str,
        default="OzzyGT/SDXL_Controlnet_Tile_Realistic",
        help="ControlNet Tile 模型 (HF ID 或本地路径)。"
             "默认使用 OzzyGT 的 diffusers 兼容重打包版 "
             "(与 TTPlanet/TTPLanet_SDXL_Controlnet_Tile_Realistic 同权重)",
    )
    mg.add_argument(
        "--weights_dir", type=str, default=str(_PROJECT_ROOT / "weights"),
        help="本地权重缓存根目录",
    )

    tg = p.add_argument_group("分块参数")
    tg.add_argument("--scale_factor", type=int, default=2, help="放大倍数")
    tg.add_argument("--tile_size", type=int, default=1024, help="瓦片边长 (px)")
    tg.add_argument("--min_overlap", type=int, default=128, help="相邻瓦片最小重叠 (px)")

    ig = p.add_argument_group("推理参数")
    ig.add_argument("--input_image", nargs="+", required=True, help="输入图片路径 (支持多张)")
    ig.add_argument("--prompt", type=str, default=_DEFAULT_PROMPT)
    ig.add_argument("--negative_prompt", type=str, default=_DEFAULT_NEGATIVE)
    ig.add_argument("--strength", type=float, default=0.2, help="img2img 去噪强度")
    ig.add_argument("--controlnet_scale", type=float, default=0.9, help="ControlNet conditioning scale")
    ig.add_argument("--guidance_scale", type=float, default=7.5)
    ig.add_argument("--num_inference_steps", type=int, default=30)
    ig.add_argument("--seed", type=int, default=42)

    sg = p.add_argument_group("系统")
    sg.add_argument("--dtype", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    sg.add_argument("--gpu", type=int, default=3)
    sg.add_argument(
        "--output_dir", type=str,
        default=str(_PROJECT_ROOT / "outputs/tiled_upscale"),
    )

    return p.parse_args()


# ── 工具函数 ─────────────────────────────────────────────────────────────────

def _dtype_map(s: str) -> torch.dtype:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[s]


def log_vram(label: str, device: torch.device) -> dict:
    if device.type != "cuda":
        return {}
    torch.cuda.synchronize(device)
    allocated = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    peak = torch.cuda.max_memory_allocated(device) / 1024**2
    logger.info(
        f"[VRAM] {label}: allocated={allocated:.0f}MB, "
        f"reserved={reserved:.0f}MB, peak={peak:.0f}MB"
    )
    return {"label": label, "allocated_mb": round(allocated, 1),
            "reserved_mb": round(reserved, 1), "peak_mb": round(peak, 1)}


def resolve_local_path(model_id: str, weights_dir: str) -> str:
    """如果本地 weights_dir 有缓存，返回本地路径；否则返回原始 model_id。"""
    if os.path.isabs(model_id) and os.path.isdir(model_id):
        return model_id
    slug = model_id.replace("/", "--")
    local = os.path.join(weights_dir, slug)
    if os.path.isdir(local):
        logger.info(f"使用本地缓存: {local}")
        return local
    return model_id


# ── 分块算法 ─────────────────────────────────────────────────────────────────

def compute_tile_positions(total_size: int, tile_size: int, min_overlap: int) -> list[int]:
    """计算瓦片起始位置，使之均匀分布覆盖 total_size，相邻重叠 >= min_overlap。"""
    if total_size <= tile_size:
        return [0]

    stride = tile_size - min_overlap
    n_tiles = math.ceil((total_size - tile_size) / stride) + 1

    if n_tiles <= 1:
        return [0]

    positions = []
    for i in range(n_tiles):
        pos = round(i * (total_size - tile_size) / (n_tiles - 1))
        positions.append(pos)
    return positions


def create_blend_mask(
    tile_h: int, tile_w: int,
    overlap_top: int, overlap_bottom: int,
    overlap_left: int, overlap_right: int,
) -> np.ndarray:
    """为单个 tile 创建线性渐变融合权重 (H, W)，中心=1.0，边缘在重叠区线性衰减。"""
    mask = np.ones((tile_h, tile_w), dtype=np.float32)

    if overlap_top > 0:
        ramp = np.linspace(0.0, 1.0, overlap_top, dtype=np.float32)
        mask[:overlap_top, :] *= ramp[:, np.newaxis]

    if overlap_bottom > 0:
        ramp = np.linspace(1.0, 0.0, overlap_bottom, dtype=np.float32)
        mask[-overlap_bottom:, :] *= ramp[:, np.newaxis]

    if overlap_left > 0:
        ramp = np.linspace(0.0, 1.0, overlap_left, dtype=np.float32)
        mask[:, :overlap_left] *= ramp[np.newaxis, :]

    if overlap_right > 0:
        ramp = np.linspace(1.0, 0.0, overlap_right, dtype=np.float32)
        mask[:, -overlap_right:] *= ramp[np.newaxis, :]

    return mask


# ── 核心：分块超分辨率 ──────────────────────────────────────────────────────

def tiled_upscale(
    pipeline,
    image: Image.Image,
    *,
    scale_factor: int,
    tile_size: int,
    min_overlap: int,
    prompt: str,
    negative_prompt: str,
    strength: float,
    controlnet_scale: float,
    guidance_scale: float,
    num_inference_steps: int,
    seed: int,
    device: torch.device,
) -> Image.Image:
    """对单张图像执行分块 ControlNet Tile 超分辨率。

    流程: Lanczos 预放大 -> 切分重叠瓦片 -> 逐块 img2img -> 加权融合
    """
    orig_w, orig_h = image.size
    target_w = (orig_w * scale_factor // 8) * 8
    target_h = (orig_h * scale_factor // 8) * 8

    logger.info(f"预放大: {orig_w}x{orig_h} -> {target_w}x{target_h} (Lanczos)")
    upscaled = image.resize((target_w, target_h), Image.LANCZOS)

    xs = compute_tile_positions(target_w, tile_size, min_overlap)
    ys = compute_tile_positions(target_h, tile_size, min_overlap)
    n_tiles = len(xs) * len(ys)
    logger.info(
        f"分块: {len(xs)}x{len(ys)} = {n_tiles} tiles, "
        f"tile={tile_size}px, positions x={xs} y={ys}"
    )

    result = np.zeros((target_h, target_w, 3), dtype=np.float64)
    weight_sum = np.zeros((target_h, target_w), dtype=np.float64)

    tile_idx = 0
    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            tile_idx += 1
            x_end = min(x + tile_size, target_w)
            y_end = min(y + tile_size, target_h)
            tw = x_end - x
            th = y_end - y

            tile_crop = upscaled.crop((x, y, x_end, y_end))

            needs_resize = (tw != tile_size or th != tile_size)
            if needs_resize:
                tile_crop = tile_crop.resize((tile_size, tile_size), Image.LANCZOS)

            overlap_top = max(tile_size - (ys[yi] - ys[yi - 1]), 0) if yi > 0 else 0
            overlap_bottom = max(tile_size - (ys[yi + 1] - ys[yi]), 0) if yi < len(ys) - 1 else 0
            overlap_left = max(tile_size - (xs[xi] - xs[xi - 1]), 0) if xi > 0 else 0
            overlap_right = max(tile_size - (xs[xi + 1] - xs[xi]), 0) if xi < len(xs) - 1 else 0

            logger.info(
                f"  Tile {tile_idx}/{n_tiles}: ({x},{y})->({x_end},{y_end}), "
                f"overlap TBLR=({overlap_top},{overlap_bottom},{overlap_left},{overlap_right})"
            )

            gen = torch.Generator(device=device).manual_seed(seed + tile_idx)
            with torch.inference_mode():
                out = pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=tile_crop,
                    control_image=tile_crop,
                    strength=strength,
                    controlnet_conditioning_scale=controlnet_scale,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=gen,
                    width=tile_size,
                    height=tile_size,
                ).images[0]

            out_np = np.array(out).astype(np.float64)

            if needs_resize:
                out_pil = Image.fromarray(out_np.astype(np.uint8))
                out_pil = out_pil.resize((tw, th), Image.LANCZOS)
                out_np = np.array(out_pil).astype(np.float64)
                scale_h, scale_w = th / tile_size, tw / tile_size
                mask = create_blend_mask(
                    th, tw,
                    max(int(overlap_top * scale_h), 0),
                    max(int(overlap_bottom * scale_h), 0),
                    max(int(overlap_left * scale_w), 0),
                    max(int(overlap_right * scale_w), 0),
                )
            else:
                mask = create_blend_mask(
                    tile_size, tile_size,
                    overlap_top, overlap_bottom,
                    overlap_left, overlap_right,
                )

            result[y:y_end, x:x_end] += out_np * mask[:, :, np.newaxis]
            weight_sum[y:y_end, x:x_end] += mask

    weight_sum = np.maximum(weight_sum, 1e-8)
    result = result / weight_sum[:, :, np.newaxis]
    result = np.clip(result, 0, 255).astype(np.uint8)

    return Image.fromarray(result)


# ── Pipeline 构建 ────────────────────────────────────────────────────────────

def build_pipeline(args, dtype, device):
    from diffusers import (
        ControlNetModel,
        DPMSolverMultistepScheduler,
        StableDiffusionXLControlNetImg2ImgPipeline,
    )

    log_vram("加载前", device)

    base_path = resolve_local_path(args.base_model_path, args.weights_dir)
    cn_path = resolve_local_path(args.controlnet_path, args.weights_dir)

    logger.info(f"加载 ControlNet Tile: {cn_path}")
    cn_kwargs = {"torch_dtype": dtype}
    if "OzzyGT" in cn_path:
        cn_kwargs["variant"] = "fp16"
    controlnet = ControlNetModel.from_pretrained(cn_path, **cn_kwargs)
    log_vram("ControlNet 加载后", device)

    logger.info(f"加载 SDXL base: {base_path}")
    pipeline = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        base_path,
        controlnet=controlnet,
        torch_dtype=dtype,
        use_safetensors=True,
    )

    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config,
        use_karras_sigmas=True,
    )
    logger.info("调度器: DPMSolverMultistepScheduler (Karras sigmas)")

    pipeline = pipeline.to(device)
    pipeline.enable_vae_tiling()
    pipeline.enable_vae_slicing()
    pipeline.set_progress_bar_config(desc="tile 推理", leave=False)

    log_vram("Pipeline 上 GPU 后", device)
    return pipeline


# ── 对比图 ───────────────────────────────────────────────────────────────────

def _make_comparison(
    original: Image.Image, upscaled: Image.Image, gap: int = 8
) -> Image.Image:
    """双栏对比图: [原图(放大到同尺寸)] | [超分结果]。"""
    w, h = upscaled.size
    orig_resized = original.resize((w, h), Image.LANCZOS)
    canvas = Image.new("RGB", (w * 2 + gap, h), color=(20, 20, 20))
    canvas.paste(orig_resized, (0, 0))
    canvas.paste(upscaled, (w + gap, 0))
    return canvas


# ── 入口 ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logger.warning("CUDA 不可用，降级为 CPU（速度极慢）")

    dtype = _dtype_map(args.dtype)
    logger.info(f"设备: {device} | 精度: {args.dtype}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = build_pipeline(args, dtype, device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for img_path in args.input_image:
        logger.info("=" * 60)
        logger.info(f"处理: {img_path}")

        src = Image.open(img_path).convert("RGB")
        logger.info(f"原始尺寸: {src.size[0]}x{src.size[1]}")

        result = tiled_upscale(
            pipeline, src,
            scale_factor=args.scale_factor,
            tile_size=args.tile_size,
            min_overlap=args.min_overlap,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            strength=args.strength,
            controlnet_scale=args.controlnet_scale,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
            device=device,
        )

        stem = Path(img_path).stem
        save_name = f"{stem}_x{args.scale_factor}_str{args.strength:.2f}.png"
        save_path = output_dir / save_name
        result.save(save_path)
        logger.info(f"保存: {save_path} ({result.size[0]}x{result.size[1]})")

        cmp = _make_comparison(src, result)
        cmp_path = output_dir / f"{stem}_x{args.scale_factor}_compare.png"
        cmp.save(cmp_path)
        logger.info(f"对比图: {cmp_path}")

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / 1024**2
        logger.info(f"显存峰值: {peak:.0f} MB ({peak / 1024:.2f} GB)")

    meta = {
        "base_model": args.base_model_path,
        "controlnet": args.controlnet_path,
        "scale_factor": args.scale_factor,
        "tile_size": args.tile_size,
        "min_overlap": args.min_overlap,
        "strength": args.strength,
        "controlnet_scale": args.controlnet_scale,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "seed": args.seed,
        "dtype": args.dtype,
        "input_images": args.input_image,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info("全部完成！")


if __name__ == "__main__":
    main()
