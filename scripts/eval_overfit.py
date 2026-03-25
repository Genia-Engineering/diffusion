"""过拟合测试评估脚本 — 使用训练完的 PixArt-Sigma 权重重新计算 FID，
并为全部 96 张训练图生成预测图及原图与预测图的对比叠加图。

FID 生成和预测均采用批处理以加速推理。

用法:
  python scripts/eval_overfit.py \
    --checkpoint ./outputs/overfit_test_pixart_sigma/checkpoints/step_000350/transformer \
    --train_data_dir ./data/overfit_test/train \
    --output_dir ./outputs/eval_overfit
"""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint", type=str,
                    default="./outputs/overfit_test_pixart_sigma/checkpoints/step_000350/transformer",
                    help="微调后的 transformer 权重目录")
    p.add_argument("--base_model", type=str,
                    default="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
                    help="基础模型名称")
    p.add_argument("--weights_dir", type=str, default="./weights",
                    help="本地权重根目录")
    p.add_argument("--train_data_dir", type=str,
                    default="./data/overfit_test/train",
                    help="训练数据目录")
    p.add_argument("--prompt", type=str,
                    default="architectural floor plan, blueprint, technical drawing",
                    help="生成用 prompt")
    p.add_argument("--negative_prompt", type=str, default="")
    p.add_argument("--num_inference_steps", type=int, default=25)
    p.add_argument("--guidance_scale", type=float, default=4.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=2,
                    help="批处理大小（每次 pipeline 调用生成的图像数，A10G 22GB 建议 ≤2）")
    p.add_argument("--output_dir", type=str, default="./outputs/eval_overfit")
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def load_pipeline(args, device, dtype):
    """加载 PixArt-Sigma pipeline，替换为微调 transformer + dpm_sde_karras 调度器。"""
    from diffusers import DPMSolverSDEScheduler, PixArtSigmaPipeline, PixArtTransformer2DModel
    from models.model_loader import load_pixart_sigma_components

    logger.info(f"加载基础模型组件: {args.base_model}")
    components = load_pixart_sigma_components(
        args.base_model, weights_dir=args.weights_dir, dtype=dtype,
    )

    logger.info(f"加载微调 transformer: {args.checkpoint}")
    transformer = PixArtTransformer2DModel.from_pretrained(
        args.checkpoint, torch_dtype=dtype,
    )

    inference_scheduler = DPMSolverSDEScheduler.from_config(
        components["noise_scheduler"].config, use_karras_sigmas=True,
    )
    logger.info("调度器: DPMSolverSDEScheduler (Karras sigmas)")

    pipeline = PixArtSigmaPipeline(
        vae=components["vae"],
        transformer=transformer,
        text_encoder=components["text_encoder"],
        tokenizer=components["tokenizer"],
        scheduler=inference_scheduler,
    )
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(desc="推理", leave=False)
    return pipeline


def load_train_images(train_data_dir: str, max_images: int = -1) -> list[tuple[str, Image.Image]]:
    """按文件名排序加载训练图像，返回 (filename, PIL.Image) 列表。"""
    data_dir = Path(train_data_dir)
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    paths = sorted([p for p in data_dir.iterdir() if p.suffix.lower() in exts])
    if max_images > 0:
        paths = paths[:max_images]
    results = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        results.append((p.name, img))
    logger.info(f"加载训练图像: {len(results)} 张 (from {data_dir})")
    return results


def make_side_by_side(
    original: Image.Image,
    predicted: Image.Image,
    overlay_alpha: float = 0.45,
    label_height: int = 36,
    gap: int = 6,
) -> Image.Image:
    """生成三栏对比图: [原图 | 原图+预测叠加 | 预测图]。"""
    w, h = 1024, 1024
    orig = original.resize((w, h), Image.LANCZOS).convert("RGB")
    pred = predicted.resize((w, h), Image.LANCZOS).convert("RGB")
    overlay = Image.blend(orig, pred, alpha=1 - overlay_alpha)

    total_w = w * 3 + gap * 2
    total_h = h + label_height
    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except (OSError, IOError):
        font = ImageFont.load_default()

    labels = ["Ground Truth", "Overlay (GT + Pred)", "Predicted"]
    panels = [orig, overlay, pred]
    for i, (label, img) in enumerate(zip(labels, panels)):
        x_off = i * (w + gap)
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        draw.text((x_off + (w - text_w) // 2, 6), label, fill=(0, 0, 0), font=font)
        canvas.paste(img, (x_off, label_height))

    return canvas


def precompute_embeds(pipeline, prompt, negative_prompt, device):
    """预编码 prompt，返回 (prompt_embeds, prompt_attention_mask,
    negative_prompt_embeds, negative_prompt_attention_mask)，
    之后可卸载 text_encoder 节省显存。"""
    (
        prompt_embeds,
        prompt_attention_mask,
        negative_prompt_embeds,
        negative_prompt_attention_mask,
    ) = pipeline.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=True,
        device=device,
        num_images_per_prompt=1,
    )
    return {
        "prompt_embeds": prompt_embeds,
        "prompt_attention_mask": prompt_attention_mask,
        "negative_prompt_embeds": negative_prompt_embeds,
        "negative_prompt_attention_mask": negative_prompt_attention_mask,
    }


def generate_batch(pipeline, embeds, num_inference_steps,
                    guidance_scale, seeds, device, batch_size):
    """使用预编码 embeddings 批量生成图像，返回 PIL Image 列表。"""
    all_images: list[Image.Image] = []
    for start in range(0, len(seeds), batch_size):
        batch_seeds = seeds[start:start + batch_size]
        bs = len(batch_seeds)
        generators = [
            torch.Generator(device=device).manual_seed(s) for s in batch_seeds
        ]
        batch_embeds = {k: v.expand(bs, -1, -1) for k, v in embeds.items()
                        if v.dim() == 3}
        batch_embeds.update({k: v.expand(bs, -1) for k, v in embeds.items()
                             if v.dim() == 2})
        output = pipeline(
            prompt=None,
            negative_prompt=None,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generators,
            num_images_per_prompt=1,
            **batch_embeds,
        )
        all_images.extend(output.images)
        logger.info(f"  已生成 {min(start + bs, len(seeds))}/{len(seeds)}")
    return all_images


def generate_and_evaluate(pipeline, embeds, train_images, args, device, output_dir: Path):
    """一次生成 96 张图像，同时用于 FID 计算和原图对比保存。"""
    from utils.fid import FIDCalculator

    n = len(train_images)
    compare_dir = output_dir / "comparisons"
    compare_dir.mkdir(parents=True, exist_ok=True)

    # ── 边生成边保存 ──
    logger.info("=" * 60)
    logger.info(f"批量生成 {n} 张图像并实时保存 (batch_size={args.batch_size})")
    logger.info("=" * 60)
    gen_images: list[Image.Image] = []
    img_idx = 0
    for start in range(0, n, args.batch_size):
        batch_seeds = [args.seed + i for i in range(start, min(start + args.batch_size, n))]
        bs = len(batch_seeds)
        generators = [
            torch.Generator(device=device).manual_seed(s) for s in batch_seeds
        ]
        batch_embeds = {k: v.expand(bs, -1, -1) for k, v in embeds.items()
                        if v.dim() == 3}
        batch_embeds.update({k: v.expand(bs, -1) for k, v in embeds.items()
                             if v.dim() == 2})
        output = pipeline(
            prompt=None, negative_prompt=None,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generators, num_images_per_prompt=1,
            **batch_embeds,
        )
        for pred_img in output.images:
            gen_images.append(pred_img)
            fname, orig_img = train_images[img_idx]

            pred_path = compare_dir / f"{img_idx:03d}_predicted.png"
            pred_img.save(pred_path)

            cmp_img = make_side_by_side(orig_img, pred_img)
            cmp_path = compare_dir / f"{img_idx:03d}_comparison.png"
            cmp_img.save(cmp_path)

            img_idx += 1
        logger.info(f"  已生成并保存 {img_idx}/{n}")
    logger.info(f"全部 {n} 张预测图和对比图已保存到 {compare_dir}")

    # ── FID 计算 ──
    logger.info("=" * 60)
    logger.info("FID 计算")
    logger.info("=" * 60)
    fid_calc = FIDCalculator(model_name="dinov2_vitb14", device=device)

    real_pil = [img for _, img in train_images]
    logger.info(f"提取 {len(real_pil)} 张真实图像特征...")
    fid_calc.update_real(real_pil)

    fid_score = fid_calc.compute(gen_images)
    logger.info(f"FID Score: {fid_score:.4f}")

    del fid_calc
    torch.cuda.empty_cache()

    return fid_score


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    pipeline = load_pipeline(args, device, dtype)

    all_train = load_train_images(args.train_data_dir)

    logger.info("预编码 prompt embeddings...")
    embeds = precompute_embeds(pipeline, args.prompt, args.negative_prompt, device)

    logger.info("卸载 text_encoder 释放显存...")
    pipeline.text_encoder = None
    pipeline.tokenizer = None
    torch.cuda.empty_cache()

    fid_score = generate_and_evaluate(pipeline, embeds, all_train, args, device, output_dir)

    summary = (
        f"评估完成\n"
        f"  Checkpoint:  {args.checkpoint}\n"
        f"  Scheduler:   DPMSolverSDEScheduler (Karras sigmas)\n"
        f"  Steps:       {args.num_inference_steps}\n"
        f"  Batch size:  {args.batch_size}\n"
        f"  FID Score:   {fid_score:.4f}\n"
        f"  对比图数量:  {len(all_train)}\n"
        f"  输出目录:    {output_dir / 'comparisons'}\n"
    )
    logger.info("\n" + summary)

    with open(output_dir / "eval_summary.txt", "w") as f:
        f.write(summary)


if __name__ == "__main__":
    main()
