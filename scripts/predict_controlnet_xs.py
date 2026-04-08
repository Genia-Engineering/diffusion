"""PixArt-Sigma ControlNet-XS 批量推理 — 使用所有检查点对训练集图片进行预测。

对 outputs/controlnet_xs_pixart_sigma_dual_lr/checkpoints 下的每个检查点:
  1. 加载微调后的 Transformer + ControlNet-XS 权重
  2. 从训练集中随机抽取 N 张图片对应的条件图
  3. 生成图像，并将 (条件图 | 生成图 | 真值图) 拼接保存

用法:
  python scripts/predict_controlnet_xs.py                         # 使用所有检查点
  python scripts/predict_controlnet_xs.py --ckpt best             # 仅使用 best 检查点
  python scripts/predict_controlnet_xs.py --num_samples 20        # 生成 20 张
  python scripts/predict_controlnet_xs.py --steps 30 --cfg 5.0    # 调整推理参数
  python scripts/predict_controlnet_xs.py --images_per_sample 3    # 每个输入生成 3 张
"""

import argparse
import hashlib
import logging
import random
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffusers import DPMSolverSDEScheduler, PixArtTransformer2DModel
from models.controlnet_xs_pixart import (
    PixArtControlNetXSAdapter,
    PixArtControlNetXSTransformerModel,
)
from models.model_loader import load_pixart_sigma_components
from pipelines.pixart_controlnet_pipeline import PixArtControlNetPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_ORIG_SUFFIX = "___total__1024.png"
_COND_SUFFIX = "_controlnet_color_1024.png"


def load_text_embed_cache(
    train_data_dir: str,
    caption: str,
    negative_prompt: str = "",
    model_type: str = "pixart_sigma",
    device: torch.device = torch.device("cpu"),
) -> dict[str, torch.Tensor] | None:
    """尝试从训练数据目录加载预缓存的文本嵌入。

    缓存路径与训练时一致: {train_data_dir}/text_embed_cache/embeds_{model_type}_{hash}.pt
    """
    cache_dir = Path(train_data_dir) / "text_embed_cache"
    embed_hash = hashlib.md5(f"{caption}||{negative_prompt}".encode()).hexdigest()[:8]
    cache_file = cache_dir / f"embeds_{model_type}_{embed_hash}.pt"

    if not cache_file.exists():
        logger.warning(f"未找到预缓存文本嵌入: {cache_file}")
        return None

    logger.info(f"加载预缓存文本嵌入: {cache_file}")
    cached = torch.load(cache_file, map_location=device, weights_only=True)
    return cached


def load_per_image_text_embed(
    train_data_dir: str,
    stem: str,
    model_type: str = "pixart_sigma",
    device: torch.device = torch.device("cpu"),
) -> dict[str, torch.Tensor] | None:
    """加载逐图片预缓存的文本嵌入。

    缓存路径: {train_data_dir}/../text_embed_per_image_cache_{model_type}/{stem}.pt
    """
    parent = Path(train_data_dir).parent
    cache_dir = parent / f"text_embed_per_image_cache_{model_type}"
    cache_file = cache_dir / f"{stem}.pt"

    if not cache_file.exists():
        return None

    return torch.load(cache_file, map_location=device, weights_only=True)


def strip_suffix(name: str, suffix: str) -> str | None:
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return None


def build_paired_list(
    train_dir: Path, cond_dir: Path
) -> list[tuple[Path, Path, str]]:
    """构建训练图与条件图的配对列表，返回 [(train_path, cond_path, base_key), ...]"""
    cond_index: dict[str, Path] = {}
    for f in cond_dir.iterdir():
        if not f.is_file() or f.suffix.lower() not in _IMG_EXTS:
            continue
        bk = strip_suffix(f.name, _COND_SUFFIX)
        if bk is None:
            bk = f.stem
        cond_index[bk] = f

    pairs = []
    for f in sorted(train_dir.iterdir()):
        if not f.is_file() or f.suffix.lower() not in _IMG_EXTS:
            continue
        bk = strip_suffix(f.name, _ORIG_SUFFIX)
        if bk is None:
            bk = f.stem
        if bk in cond_index:
            pairs.append((f, cond_index[bk], bk))
    return pairs


def load_caption(caption_dir: Path, base_key: str, fallback: str) -> str:
    """加载 base_key 对应的 caption 文本，不存在则返回 fallback。

    caption 文件名规则: base_key + "_controlnet_color_1024.txt"
    (与配置中的 caption_stem_replace 一致: ___total__1024 → _controlnet_color_1024)
    """
    caption_stem = base_key + "_controlnet_color_1024"
    for candidate in [
        caption_dir / f"{caption_stem}.txt",
        caption_dir / f"{base_key}.txt",
    ]:
        if candidate.exists():
            return candidate.read_text(encoding="utf-8").strip()
    return fallback


def blend_images(
    base: Image.Image, overlay: Image.Image, alpha: float = 0.5
) -> Image.Image:
    """将 overlay 以 alpha 透明度叠加到 base 上。"""
    base_r = base.resize(overlay.size, Image.LANCZOS).convert("RGBA")
    overlay_r = overlay.convert("RGBA")
    blended = Image.blend(base_r, overlay_r, alpha)
    return blended.convert("RGB")


def make_comparison_grid(
    cond_img: Image.Image,
    gen_imgs: list[Image.Image],
    gt_img: Image.Image,
    resolution: int = 1024,
    overlay_alpha: float = 0.5,
) -> Image.Image:
    """横向拼接: 条件图 | 生成图1 | 叠加图1 | ... | 生成图N | 叠加图N | 真值图"""
    cond_r = cond_img.resize((resolution, resolution), Image.LANCZOS)
    gt_r = gt_img.resize((resolution, resolution), Image.LANCZOS)

    n_gen = len(gen_imgs)
    total_cols = 1 + n_gen * 2 + 1  # cond + (gen + overlay) * N + gt
    grid = Image.new("RGB", (resolution * total_cols, resolution))

    grid.paste(cond_r, (0, 0))
    x_offset = resolution
    for gen_img in gen_imgs:
        gen_r = gen_img.resize((resolution, resolution), Image.LANCZOS)
        grid.paste(gen_r, (x_offset, 0))
        x_offset += resolution

        overlay = blend_images(cond_r, gen_r, alpha=overlay_alpha)
        grid.paste(overlay, (x_offset, 0))
        x_offset += resolution

    grid.paste(gt_r, (x_offset, 0))
    return grid


def run_inference_for_checkpoint(
    ckpt_dir: Path,
    base_model_path: str,
    weights_dir: str,
    pairs: list[tuple[Path, Path, str]],
    caption_dir: Path | None,
    fallback_caption: str,
    output_root: Path,
    num_samples: int = 10,
    images_per_sample: int = 1,
    num_inference_steps: int = 20,
    guidance_scale: float = 4.5,
    overlay_alpha: float = 0.5,
    seed: int = 42,
    device: torch.device = torch.device("cuda"),
    text_embed_cache: dict[str, torch.Tensor] | None = None,
    train_data_dir: str | None = None,
    prompt_mode: str = "fixed",
    finetuned_transformer_path: str | None = None,
):
    """对单个检查点执行推理。"""
    ckpt_name = ckpt_dir.name
    logger.info(f"=== 加载检查点: {ckpt_name} ===")

    transformer_dir = ckpt_dir / "transformer"
    controlnet_dir = ckpt_dir / "controlnet"

    if not controlnet_dir.exists():
        logger.warning(f"检查点 {ckpt_name} 缺少 controlnet 目录，跳过")
        return

    # adapter-only 模式: 检查点里没有 transformer，从外部路径加载
    if not transformer_dir.exists():
        if finetuned_transformer_path and Path(finetuned_transformer_path).exists():
            transformer_dir = Path(finetuned_transformer_path)
            logger.info(f"检查点无 transformer 目录，使用微调 transformer: {transformer_dir}")
        else:
            logger.warning(
                f"检查点 {ckpt_name} 缺少 transformer 目录，"
                f"且未提供有效的 --finetuned_transformer_path，跳过"
            )
            return

    use_cached_embeds = text_embed_cache is not None or (
        prompt_mode == "caption" and train_data_dir is not None
    )

    components = load_pixart_sigma_components(
        base_model_path,
        weights_dir=weights_dir,
        flow_matching=False,
        dtype=torch.bfloat16,
    )
    vae = components["vae"].to(device, dtype=torch.float32)
    noise_scheduler = components["noise_scheduler"]

    if use_cached_embeds:
        text_encoder = None
        tokenizer = None
        logger.info("使用预缓存文本嵌入，跳过加载 text_encoder")
    else:
        text_encoder = components["text_encoder"].to(device)
        tokenizer = components["tokenizer"]

    transformer = PixArtTransformer2DModel.from_pretrained(
        str(transformer_dir), torch_dtype=torch.bfloat16
    ).to(device)

    controlnet = PixArtControlNetXSAdapter.from_pretrained(
        str(controlnet_dir), torch_dtype=torch.bfloat16
    ).to(device)

    joint_model = PixArtControlNetXSTransformerModel(
        transformer=transformer, controlnet=controlnet
    )

    inference_scheduler = DPMSolverSDEScheduler.from_config(
        noise_scheduler.config,
        use_karras_sigmas=True,
    )

    pipeline = PixArtControlNetPipeline(
        vae=vae,
        transformer=joint_model,
        controlnet=None,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=inference_scheduler,
    )
    pipeline.set_progress_bar_config(disable=True)

    rng = random.Random(seed)
    selected = rng.sample(pairs, min(num_samples, len(pairs)))

    save_dir = output_root / ckpt_name
    save_dir.mkdir(parents=True, exist_ok=True)

    for i, (train_path, cond_path, base_key) in enumerate(tqdm(
        selected, desc=f"  [{ckpt_name}] 生成中"
    )):
        cond_img = Image.open(cond_path).convert("RGB").resize((1024, 1024), Image.LANCZOS)
        gt_img = Image.open(train_path).convert("RGB").resize((1024, 1024), Image.LANCZOS)

        # 构建 pipeline 参数: 优先使用预缓存嵌入
        if text_embed_cache is not None:
            pipeline_kwargs = {
                "prompt": None,
                "negative_prompt": None,
                "prompt_embeds": text_embed_cache["prompt_embeds"].to(device),
                "prompt_attention_mask": text_embed_cache["prompt_attention_mask"].to(device),
                "negative_prompt_embeds": text_embed_cache["negative_prompt_embeds"].to(device),
                "negative_prompt_attention_mask": text_embed_cache["negative_prompt_attention_mask"].to(device),
            }
        elif prompt_mode == "caption" and train_data_dir is not None:
            per_img = load_per_image_text_embed(train_data_dir, base_key, device=device)
            if per_img is not None:
                neg_cache = text_embed_cache or {}
                pipeline_kwargs = {
                    "prompt": None,
                    "negative_prompt": None,
                    "prompt_embeds": per_img["prompt_embeds"].to(device),
                    "prompt_attention_mask": per_img["prompt_attention_mask"].to(device),
                    "negative_prompt_embeds": neg_cache.get(
                        "negative_prompt_embeds",
                        torch.zeros_like(per_img["prompt_embeds"]),
                    ).to(device),
                    "negative_prompt_attention_mask": neg_cache.get(
                        "negative_prompt_attention_mask",
                        torch.ones(1, per_img["prompt_embeds"].shape[1], dtype=torch.long),
                    ).to(device),
                }
            else:
                prompt = load_caption(caption_dir, base_key, fallback_caption) if caption_dir else fallback_caption
                pipeline_kwargs = {"prompt": prompt, "negative_prompt": ""}
        else:
            prompt = load_caption(caption_dir, base_key, fallback_caption) if caption_dir else fallback_caption
            pipeline_kwargs = {"prompt": prompt, "negative_prompt": ""}

        gen_imgs = []
        for j in range(images_per_sample):
            generator = torch.Generator(device="cpu").manual_seed(seed + i * images_per_sample + j)
            result = pipeline(
                **pipeline_kwargs,
                image=cond_img,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=1024,
                width=1024,
                generator=generator,
            )
            gen_imgs.append(result.images[0])

        grid = make_comparison_grid(cond_img, gen_imgs, gt_img, overlay_alpha=overlay_alpha)
        grid.save(save_dir / f"{i:03d}_grid.png")

        logger.info(f"  [{i+1}/{len(selected)}] {images_per_sample} images saved: {base_key[:60]}...")

    del pipeline, joint_model, transformer, controlnet, vae
    if text_encoder is not None:
        del text_encoder
    torch.cuda.empty_cache()

    logger.info(f"=== 检查点 {ckpt_name} 完成，结果保存至: {save_dir} ===")


def main():
    parser = argparse.ArgumentParser(description="PixArt ControlNet-XS 批量推理")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/daiqing_tan/stable_diffusion_lora/outputs/controlnet_xs_pixart_sigma_adapter_only",
        help="训练输出目录",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        help="基础模型路径或 HuggingFace ID",
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        default="./weights",
        help="本地权重缓存目录",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default="/home/daiqing_tan/stable_diffusion_lora/data/data/size_1024/floor",
    )
    parser.add_argument(
        "--cond_data_dir",
        type=str,
        default="/home/daiqing_tan/stable_diffusion_lora/data/data/size_1024_controlnet/floor",
    )
    parser.add_argument(
        "--caption_dir",
        type=str,
        default="/home/daiqing_tan/stable_diffusion_lora/data/data/description",
    )
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--images_per_sample", type=int, default=4, help="每个输入条件图生成几张输出")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg", type=float, default=4.5, help="Guidance scale")
    parser.add_argument("--overlay_alpha", type=float, default=0.5, help="叠加图中生成图的不透明度 (0=全条件图, 1=全生成图)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="fixed",
        choices=["caption", "fixed", "empty"],
        help="提示词方案: caption=从文件加载每张图的描述, fixed=统一使用固定提示词, empty=空提示词",
    )
    parser.add_argument(
        "--prompt_text",
        type=str,
        default="architectural floor plan, blueprint, technical drawing",
        help="fixed 模式下使用的统一提示词",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="指定检查点名称 (如 best, step_003600)，默认遍历全部",
    )
    parser.add_argument(
        "--use_cached_embeds",
        action="store_true",
        default=True,
        help="使用训练时预缓存的文本嵌入（与训练验证一致），默认开启",
    )
    parser.add_argument(
        "--no_cached_embeds",
        action="store_true",
        help="禁用预缓存文本嵌入，使用实时 T5 编码",
    )
    parser.add_argument(
        "--finetuned_transformer_path",
        type=str,
        default="./outputs/pixart_sigma_floorplan/checkpoints/step_003600/transformer",
        help="微调后的 Transformer 权重路径 (adapter-only 模式下必需)",
    )
    args = parser.parse_args()

    if args.no_cached_embeds:
        args.use_cached_embeds = False

    output_dir = Path(args.output_dir)
    ckpt_root = output_dir / "checkpoints"
    predict_root = output_dir / "predictions"
    predict_root.mkdir(parents=True, exist_ok=True)

    train_dir = Path(args.train_data_dir)
    cond_dir = Path(args.cond_data_dir)
    caption_dir = Path(args.caption_dir) if args.caption_dir else None

    pairs = build_paired_list(train_dir, cond_dir)
    logger.info(f"共找到 {len(pairs)} 对训练图-条件图配对")

    if args.ckpt:
        ckpt_dirs = [ckpt_root / args.ckpt]
        if not ckpt_dirs[0].exists():
            logger.error(f"指定的检查点不存在: {ckpt_dirs[0]}")
            sys.exit(1)
    else:
        ckpt_dirs = sorted(
            [d for d in ckpt_root.iterdir() if d.is_dir()],
            key=lambda d: d.name,
        )

    logger.info(f"将对以下检查点进行推理: {[d.name for d in ckpt_dirs]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.prompt_mode == "caption":
        effective_caption_dir = caption_dir
        fallback_caption = args.prompt_text
    elif args.prompt_mode == "fixed":
        effective_caption_dir = None
        fallback_caption = args.prompt_text
    else:  # empty
        effective_caption_dir = None
        fallback_caption = ""

    # 加载预缓存文本嵌入
    text_embed_cache = None
    if args.use_cached_embeds:
        neg_prompt = ""
        if args.prompt_mode in ("fixed", "empty"):
            text_embed_cache = load_text_embed_cache(
                args.train_data_dir, fallback_caption, neg_prompt, device=device,
            )
        elif args.prompt_mode == "caption":
            text_embed_cache = load_text_embed_cache(
                args.train_data_dir, fallback_caption, neg_prompt, device=device,
            )
        if text_embed_cache is not None:
            logger.info("已加载预缓存文本嵌入，推理将与训练验证保持一致")
        else:
            logger.warning("未找到预缓存文本嵌入，将回退到实时 T5 编码")

    logger.info(f"提示词方案: {args.prompt_mode}" + (
        f" (固定: {fallback_caption!r})" if args.prompt_mode != "caption" else
        f" (caption_dir={effective_caption_dir}, fallback={fallback_caption!r})"
    ))

    for ckpt_dir in ckpt_dirs:
        run_inference_for_checkpoint(
            ckpt_dir=ckpt_dir,
            base_model_path=args.base_model,
            weights_dir=args.weights_dir,
            pairs=pairs,
            caption_dir=effective_caption_dir,
            fallback_caption=fallback_caption,
            output_root=predict_root,
            num_samples=args.num_samples,
            images_per_sample=args.images_per_sample,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            overlay_alpha=args.overlay_alpha,
            seed=args.seed,
            device=device,
            text_embed_cache=text_embed_cache,
            train_data_dir=args.train_data_dir,
            prompt_mode=args.prompt_mode,
            finetuned_transformer_path=args.finetuned_transformer_path,
        )

    logger.info(f"全部推理完成！结果保存在: {predict_root}")


if __name__ == "__main__":
    main()
