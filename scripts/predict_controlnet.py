"""ControlNet SDXL 推理脚本 — 输入控制图 → 生成预测 → 叠加对比图。

调度器选择:
  DPMSolverSDEScheduler + Karras σ 调度
  - 基于 SDE 求解器（随机采样），更丰富的细节纹理
  - Karras σ 在高噪阶段密集采样、低噪快速收敛，25-30 步即可高质量出图
  - 对 ControlNet 条件生成的结构保真度优于纯 ODE 求解器

用法:
  # 单张控制图推理
  python scripts/predict_controlnet.py \
    --control_images /path/to/control.png

  # 多张控制图 + 自定义参数
  python scripts/predict_controlnet.py \
    --control_images img1.png img2.png img3.png \
    --controlnet_scale 0.8 \
    --num_inference_steps 30 \
    --guidance_scale 7.5 \
    --num_images 2

  # 指定输出目录 + GPU
  python scripts/predict_controlnet.py \
    --control_images img1.png \
    --output_dir ./outputs/my_test \
    --gpu 1
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

# ── 项目默认路径 ────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

_DEFAULT_CONTROLNET = str(
    _PROJECT_ROOT / "outputs/controlnet_sdxl/checkpoints/step_017000/controlnet"
)
_DEFAULT_MERGED_MODEL = str(
    _PROJECT_ROOT / "outputs/lora_sdxl_floorplan/merged_step_000810"
)
_DEFAULT_PROMPT = "architectural floor plan, blueprint, technical drawing"
_DEFAULT_NEGATIVE = "blurry, noisy, low quality, photo, realistic, watermark"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ControlNet SDXL 推理：控制图 → 预测 → 叠加对比",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--control_images", nargs="+", required=True,
        help="控制图路径（一张或多张 PNG/JPG）",
    )
    p.add_argument(
        "--controlnet_path", type=str, default=_DEFAULT_CONTROLNET,
        help="ControlNet checkpoint 目录（diffusers 格式）",
    )
    p.add_argument(
        "--merged_model_path", type=str, default=_DEFAULT_MERGED_MODEL,
        help="LoRA 融合后的 SDXL 完整模型目录",
    )
    p.add_argument(
        "--controlnet_scale", type=float, default=1.0,
        help="ControlNet conditioning scale（0~2, 越大条件越强）",
    )
    p.add_argument(
        "--prompt", type=str, default=_DEFAULT_PROMPT,
        help="生成提示词",
    )
    p.add_argument(
        "--negative_prompt", type=str, default=_DEFAULT_NEGATIVE,
        help="负提示词",
    )
    p.add_argument(
        "--num_inference_steps", type=int, default=25,
        help="去噪步数（DPMSolverSDE + Karras 下 25 步已足够）",
    )
    p.add_argument(
        "--guidance_scale", type=float, default=7.5,
        help="Classifier-free guidance 强度",
    )
    p.add_argument("--width", type=int, default=1024, help="生成宽度")
    p.add_argument("--height", type=int, default=1024, help="生成高度")
    p.add_argument("--num_images", type=int, default=4, help="每张控制图生成的图像数")
    p.add_argument("--seed", type=int, default=42, help="随机种子（-1=随机）")
    p.add_argument("--overlay_alpha", type=float, default=0.45, help="叠加透明度")

    p.add_argument(
        "--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"],
        help="模型精度",
    )
    p.add_argument("--gpu", type=int, default=-1, help="指定 GPU 索引（-1=默认）")
    p.add_argument(
        "--output_dir", type=str,
        default=str(_PROJECT_ROOT / "outputs/predict_controlnet"),
        help="输出目录",
    )

    return p.parse_args()


def _dtype_map(s: str) -> torch.dtype:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[s]


def _prompt_tag(prompt: str, max_len: int = 32) -> str:
    slug = prompt[:max_len].strip().replace(" ", "_").replace(",", "").replace("/", "-")
    h = hashlib.md5(prompt.encode()).hexdigest()[:6]
    return f"{slug}__{h}"


def make_comparison_grid(
    ctrl_img: Image.Image,
    gen_img: Image.Image,
    alpha: float = 0.45,
    gap: int = 8,
    bg_color: tuple = (20, 20, 20),
) -> Image.Image:
    """三栏对比图：[控制图 | 半透明叠加 | 生成图]"""
    w, h = gen_img.size
    ctrl = ctrl_img.resize((w, h), Image.LANCZOS).convert("RGB")
    gen = gen_img.convert("RGB")
    blend = Image.blend(gen, ctrl, alpha=alpha)

    canvas = Image.new("RGB", (w * 3 + gap * 2, h), color=bg_color)
    canvas.paste(ctrl, (0, 0))
    canvas.paste(blend, (w + gap, 0))
    canvas.paste(gen, (w * 2 + gap * 2, 0))
    return canvas


def build_pipeline(args, dtype, device):
    """构建 SDXL ControlNet pipeline，配置 DPMSolverSDEScheduler + Karras σ。"""
    from diffusers import (
        ControlNetModel,
        DPMSolverSDEScheduler,
        StableDiffusionXLControlNetPipeline,
    )

    logger.info(f"加载 ControlNet: {args.controlnet_path}")
    controlnet = ControlNetModel.from_pretrained(
        args.controlnet_path, torch_dtype=dtype,
    )

    logger.info(f"加载 SDXL 基础模型: {args.merged_model_path}")
    pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
        args.merged_model_path,
        controlnet=controlnet,
        torch_dtype=dtype,
    )

    pipeline.scheduler = DPMSolverSDEScheduler.from_config(
        pipeline.scheduler.config,
        use_karras_sigmas=True,
    )
    logger.info("调度器: DPMSolverSDEScheduler (Karras σ)")

    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(desc="推理", leave=False)
    return pipeline


def run(pipeline, args, device):
    """遍历每张控制图，生成预测图 + 叠加对比图。"""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tag = _prompt_tag(args.prompt)
    all_saved: list[Path] = []

    for ci_idx, ctrl_path in enumerate(args.control_images):
        logger.info(f"[{ci_idx + 1}/{len(args.control_images)}] 控制图: {ctrl_path}")
        ctrl_img = Image.open(ctrl_path).convert("RGB")
        ctrl_img = ctrl_img.resize((args.width, args.height), Image.NEAREST)

        seed = args.seed if args.seed >= 0 else torch.randint(0, 2**31, (1,)).item()
        generator = torch.Generator(device=device).manual_seed(seed)

        with torch.inference_mode():
            output = pipeline(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                image=ctrl_img,
                controlnet_conditioning_scale=args.controlnet_scale,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                width=args.width,
                height=args.height,
                num_images_per_prompt=args.num_images,
                generator=generator,
            )

        ctrl_stem = Path(ctrl_path).stem[:60]
        for img_idx, gen_img in enumerate(output.images):
            stem = f"ctrl{ci_idx:02d}_{ctrl_stem}_s{seed}_i{img_idx:02d}"

            # 保存生成图
            png_info = PngImagePlugin.PngInfo()
            png_info.add_text("prompt", args.prompt)
            png_info.add_text("negative_prompt", args.negative_prompt)
            png_info.add_text("seed", str(seed))
            png_info.add_text("steps", str(args.num_inference_steps))
            png_info.add_text("guidance_scale", str(args.guidance_scale))
            png_info.add_text("controlnet_scale", str(args.controlnet_scale))
            png_info.add_text("controlnet_path", args.controlnet_path)
            png_info.add_text("scheduler", "DPMSolverSDEScheduler_karras")
            png_info.add_text("control_image", ctrl_path)

            gen_path = output_dir / f"{stem}.png"
            gen_img.save(gen_path, pnginfo=png_info)
            logger.info(f"  生成图: {gen_path}")

            # 保存三栏对比图（控制图 | 叠加 | 生成图）
            cmp_img = make_comparison_grid(ctrl_img, gen_img, alpha=args.overlay_alpha)
            cmp_path = output_dir / f"{stem}_cmp.png"
            cmp_img.save(cmp_path)
            logger.info(f"  对比图: {cmp_path}")

            all_saved.extend([gen_path, cmp_path])

    # 保存本次推理元数据
    meta = {
        "controlnet_path": args.controlnet_path,
        "merged_model_path": args.merged_model_path,
        "scheduler": "DPMSolverSDEScheduler (Karras σ)",
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "controlnet_scale": args.controlnet_scale,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "resolution": f"{args.width}x{args.height}",
        "num_images": args.num_images,
        "seed": args.seed,
        "dtype": args.dtype,
        "overlay_alpha": args.overlay_alpha,
        "control_images": args.control_images,
    }
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info(f"元数据: {meta_path}")

    return all_saved


def main():
    args = parse_args()

    # 设备
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logger.warning("CUDA 不可用，降级为 CPU（推理会非常慢）")

    dtype = _dtype_map(args.dtype)
    logger.info(f"设备: {device} | 精度: {args.dtype}")

    pipeline = build_pipeline(args, dtype, device)
    saved = run(pipeline, args, device)

    logger.info(f"完成！共保存 {len(saved)} 个文件 → {args.output_dir}")


if __name__ == "__main__":
    main()
