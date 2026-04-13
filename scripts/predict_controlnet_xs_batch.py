"""PixArt-Sigma ControlNet-XS 批量推理脚本 — 支持 batch_size 和每图多次生成。

功能:
  1. 对所有/指定检查点进行批量推理
  2. 支持 batch_size 控制 GPU 并行度
  3. 每张输入图生成 n 张预测图 (n 由 --num_generations 指定)
  4. 结果保存在两个子目录:
     - predictions/  : 纯预测结果图，命名为 {原图名}_{序号}.png
     - overlays/     : 预测图与控制图叠加图，命名为 {原图名}_{序号}.png

用法:
  python scripts/predict_controlnet_xs_batch.py
  python scripts/predict_controlnet_xs_batch.py --batch_size 4 --num_generations 3
  python scripts/predict_controlnet_xs_batch.py --ckpt step_008500 --num_samples 20
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

from diffusers import DPMSolverMultistepScheduler, PixArtTransformer2DModel
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


# ---------------------------------------------------------------------------
# 文本嵌入缓存工具
# ---------------------------------------------------------------------------


def load_text_embed_cache(
    train_data_dir: str,
    caption: str,
    negative_prompt: str = "",
    model_type: str = "pixart_sigma",
    device: torch.device = torch.device("cpu"),
) -> dict[str, torch.Tensor] | None:
    cache_dir = Path(train_data_dir) / "text_embed_cache"
    embed_hash = hashlib.md5(f"{caption}||{negative_prompt}".encode()).hexdigest()[:8]
    cache_file = cache_dir / f"embeds_{model_type}_{embed_hash}.pt"

    if not cache_file.exists():
        logger.warning(f"未找到预缓存文本嵌入: {cache_file}")
        return None

    logger.info(f"加载预缓存文本嵌入: {cache_file}")
    return torch.load(cache_file, map_location=device, weights_only=True)


def load_per_image_text_embed(
    train_data_dir: str,
    stem: str,
    model_type: str = "pixart_sigma",
    device: torch.device = torch.device("cpu"),
) -> dict[str, torch.Tensor] | None:
    parent = Path(train_data_dir).parent
    cache_dir = parent / f"text_embed_per_image_cache_{model_type}"
    cache_file = cache_dir / f"{stem}.pt"

    if not cache_file.exists():
        return None
    return torch.load(cache_file, map_location=device, weights_only=True)


# ---------------------------------------------------------------------------
# 文件配对与辅助函数
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 准备单个样本的 pipeline 参数（文本嵌入部分）
# ---------------------------------------------------------------------------


def build_prompt_kwargs(
    base_key: str,
    text_embed_cache: dict[str, torch.Tensor] | None,
    prompt_mode: str,
    train_data_dir: str | None,
    caption_dir: Path | None,
    fallback_caption: str,
    device: torch.device,
) -> dict:
    """为单个样本构建文本嵌入/prompt 参数字典。"""
    if text_embed_cache is not None:
        return {
            "prompt_embeds": text_embed_cache["prompt_embeds"].to(device),
            "prompt_attention_mask": text_embed_cache["prompt_attention_mask"].to(device),
            "negative_prompt_embeds": text_embed_cache["negative_prompt_embeds"].to(device),
            "negative_prompt_attention_mask": text_embed_cache["negative_prompt_attention_mask"].to(device),
        }

    if prompt_mode == "caption" and train_data_dir is not None:
        per_img = load_per_image_text_embed(train_data_dir, base_key, device=device)
        if per_img is not None:
            return {
                "prompt_embeds": per_img["prompt_embeds"].to(device),
                "prompt_attention_mask": per_img["prompt_attention_mask"].to(device),
                "negative_prompt_embeds": torch.zeros_like(per_img["prompt_embeds"]).to(device),
                "negative_prompt_attention_mask": torch.ones(
                    1, per_img["prompt_embeds"].shape[1], dtype=torch.long
                ).to(device),
            }

    prompt = load_caption(caption_dir, base_key, fallback_caption) if caption_dir else fallback_caption
    return {"prompt": prompt, "negative_prompt": ""}


def _stack_embed_kwargs(kwargs_list: list[dict], device: torch.device) -> dict:
    """将多个样本的 prompt_embeds 字典沿 batch 维度堆叠。

    如果列表中存在 "prompt" 键（文本模式），则返回 prompt 列表。
    """
    if "prompt" in kwargs_list[0]:
        return {
            "prompt": [kw["prompt"] for kw in kwargs_list],
            "negative_prompt": [kw.get("negative_prompt", "") for kw in kwargs_list],
        }

    return {
        "prompt": None,
        "negative_prompt": None,
        "prompt_embeds": torch.cat([kw["prompt_embeds"] for kw in kwargs_list], dim=0).to(device),
        "prompt_attention_mask": torch.cat(
            [kw["prompt_attention_mask"] for kw in kwargs_list], dim=0
        ).to(device),
        "negative_prompt_embeds": torch.cat(
            [kw["negative_prompt_embeds"] for kw in kwargs_list], dim=0
        ).to(device),
        "negative_prompt_attention_mask": torch.cat(
            [kw["negative_prompt_attention_mask"] for kw in kwargs_list], dim=0
        ).to(device),
    }


# ---------------------------------------------------------------------------
# 单检查点批量推理
# ---------------------------------------------------------------------------


def run_batch_inference(
    ckpt_dir: Path,
    base_model_path: str,
    weights_dir: str,
    pairs: list[tuple[Path, Path, str]],
    caption_dir: Path | None,
    fallback_caption: str,
    output_root: Path,
    num_samples: int = 10,
    num_generations: int = 1,
    batch_size: int = 1,
    num_inference_steps: int = 20,
    guidance_scale: float = 4.5,
    overlay_alpha: float = 0.5,
    seed: int = 42,
    device: torch.device = torch.device("cuda"),
    text_embed_cache: dict[str, torch.Tensor] | None = None,
    train_data_dir: str | None = None,
    prompt_mode: str = "fixed",
    finetuned_transformer_path: str | None = None,
    start_index: int = 0,
):
    """对单个检查点执行批量推理，保存到 predictions/、overlays/ 和 conditions/ 三个子目录。"""
    ckpt_name = ckpt_dir.name
    logger.info(f"=== 加载检查点: {ckpt_name} ===")

    transformer_dir = ckpt_dir / "transformer"
    controlnet_dir = ckpt_dir / "controlnet"

    if not controlnet_dir.exists():
        logger.warning(f"检查点 {ckpt_name} 缺少 controlnet 目录，跳过")
        return

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

    # ---- 加载模型 ----
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

    inference_scheduler = DPMSolverMultistepScheduler.from_config(
        noise_scheduler.config,
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

    # ---- 选择样本 & 创建输出目录 ----
    rng = random.Random(seed)
    if num_samples <= 0 or num_samples >= len(pairs):
        selected = pairs
    else:
        selected = rng.sample(pairs, num_samples)

    ckpt_out = output_root / ckpt_name
    pred_dir = ckpt_out / "predictions"
    overlay_dir = ckpt_out / "overlays"
    cond_dir_out = ckpt_out / "conditions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    cond_dir_out.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"检查点 {ckpt_name}: {len(selected)} 个样本, "
        f"每张生成 {num_generations} 张, batch_size={batch_size}"
    )

    # ---- 逐 generation 轮次，按 batch 推理 ----
    for gen_idx in range(num_generations):
        logger.info(f"  [Generation {gen_idx + 1}/{num_generations}]")

        for batch_start in tqdm(
            range(0, len(selected), batch_size),
            desc=f"  [{ckpt_name}] gen {gen_idx+1}/{num_generations}",
        ):
            batch_items = selected[batch_start : batch_start + batch_size]
            actual_bs = len(batch_items)

            cond_imgs = []
            base_keys = []

            for train_path, cond_path, base_key in batch_items:
                cond_img = Image.open(cond_path).convert("RGB").resize(
                    (1024, 1024), Image.NEAREST
                )
                cond_imgs.append(cond_img)
                base_keys.append(base_key)

            # 构建 batch prompt 参数
            prompt_kwargs_list = [
                build_prompt_kwargs(
                    bk, text_embed_cache, prompt_mode,
                    train_data_dir, caption_dir, fallback_caption, device,
                )
                for bk in base_keys
            ]
            batch_kwargs = _stack_embed_kwargs(prompt_kwargs_list, device)

            # 构建 batch generator
            generators = [
                torch.Generator(device="cpu").manual_seed(
                    seed + (batch_start + bi) * num_generations + gen_idx
                )
                for bi in range(actual_bs)
            ]

            result = pipeline(
                **batch_kwargs,
                image=cond_imgs,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=1024,
                width=1024,
                generator=generators,
            )

            for bi in range(actual_bs):
                gen_img = result.images[bi]
                bk = base_keys[bi]
                fname = f"{bk}_{start_index + gen_idx}.png"

                gen_img.save(pred_dir / fname)

                overlay = blend_images(cond_imgs[bi], gen_img, alpha=overlay_alpha)
                combined = Image.new("RGB", (gen_img.width + overlay.width, gen_img.height))
                combined.paste(gen_img, (0, 0))
                combined.paste(overlay, (gen_img.width, 0))
                combined.save(overlay_dir / fname)

                if gen_idx == 0:
                    cond_imgs[bi].save(cond_dir_out / f"{bk}.png")

            logger.info(
                f"    batch [{batch_start+1}..{batch_start+actual_bs}/"
                f"{len(selected)}] 已保存"
            )

    # ---- 清理 ----
    del pipeline, joint_model, transformer, controlnet, vae
    if text_encoder is not None:
        del text_encoder
    torch.cuda.empty_cache()

    logger.info(
        f"=== 检查点 {ckpt_name} 完成 ===\n"
        f"    预测图: {pred_dir}\n"
        f"    叠加图: {overlay_dir}\n"
        f"    控制图: {cond_dir_out}"
    )


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="PixArt ControlNet-XS 批量推理 (支持 batch_size & 多次生成)"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="/home/daiqing_tan/stable_diffusion_lora/outputs/controlnet_xs_pixart_sigma_adapter_only_train",
        help="训练输出目录 (包含 checkpoints 子目录)",
    )
    parser.add_argument(
        "--base_model", type=str,
        default="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        help="基础模型路径或 HuggingFace ID",
    )
    parser.add_argument("--weights_dir", type=str, default="./weights", help="本地权重缓存目录")
    parser.add_argument(
        "--train_data_dir", type=str,
        default="/home/daiqing_tan/stable_diffusion_lora/data/data/size_1024/floor",
    )
    parser.add_argument(
        "--cond_data_dir", type=str,
        default="/home/daiqing_tan/stable_diffusion_lora/data/data/size_1024_controlnet/floor",
    )
    parser.add_argument(
        "--caption_dir", type=str,
        default="/home/daiqing_tan/stable_diffusion_lora/data/data/description",
    )
    parser.add_argument("--num_samples", type=int, default=0, help="从数据集抽取的样本数 (0=使用全部)")
    parser.add_argument(
        "--num_generations", type=int, default=4,
        help="每张输入图生成的预测图数量",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="每次推理的 batch 大小 (根据 GPU 显存调整)",
    )
    parser.add_argument(
        "--start_index", type=int, default=0,
        help="生成图序号的起始数字 (默认从 0 开始)",
    )
    parser.add_argument("--steps", type=int, default=25, help="推理步数")
    parser.add_argument("--cfg", type=float, default=4.5, help="Guidance scale")
    parser.add_argument(
        "--overlay_alpha", type=float, default=0.5,
        help="叠加图中生成图的不透明度 (0=全条件图, 1=全生成图)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt_mode", type=str, default="fixed",
        choices=["caption", "fixed", "empty"],
        help="提示词方案: caption=逐图描述, fixed=统一提示词, empty=空提示词",
    )
    parser.add_argument(
        "--prompt_text", type=str,
        default="architectural floor plan, blueprint, technical drawing",
        help="fixed 模式下使用的统一提示词",
    )
    parser.add_argument(
        "--ckpt", type=str, default="step_005000",
        help="指定检查点名称 (如 best, step_003600)，留空遍历全部",
    )
    parser.add_argument(
        "--use_cached_embeds", action="store_true", default=True,
        help="使用训练时预缓存的文本嵌入，默认开启",
    )
    parser.add_argument(
        "--no_cached_embeds", action="store_true",
        help="禁用预缓存文本嵌入，使用实时 T5 编码",
    )
    parser.add_argument(
        "--finetuned_transformer_path", type=str,
        default="./outputs/pixart_sigma_floorplan/checkpoints/step_003600/transformer",
        help="微调后的 Transformer 权重路径 (adapter-only 模式下必需)",
    )
    args = parser.parse_args()

    if args.no_cached_embeds:
        args.use_cached_embeds = False

    output_dir = Path(args.output_dir)
    ckpt_root = output_dir / "checkpoints"
    predict_root = output_dir / "batch_predictions"
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
    else:
        effective_caption_dir = None
        fallback_caption = ""

    text_embed_cache = None
    if args.use_cached_embeds:
        neg_prompt = ""
        text_embed_cache = load_text_embed_cache(
            args.train_data_dir, fallback_caption, neg_prompt, device=device,
        )
        if text_embed_cache is not None:
            logger.info("已加载预缓存文本嵌入")
        else:
            logger.warning("未找到预缓存文本嵌入，将回退到实时 T5 编码")

    logger.info(
        f"提示词方案: {args.prompt_mode}"
        + (
            f" (固定: {fallback_caption!r})"
            if args.prompt_mode != "caption"
            else f" (caption_dir={effective_caption_dir}, fallback={fallback_caption!r})"
        )
    )
    logger.info(
        f"推理参数: steps={args.steps}, cfg={args.cfg}, "
        f"batch_size={args.batch_size}, num_generations={args.num_generations}, "
        f"start_index={args.start_index}"
    )

    for ckpt_dir in ckpt_dirs:
        run_batch_inference(
            ckpt_dir=ckpt_dir,
            base_model_path=args.base_model,
            weights_dir=args.weights_dir,
            pairs=pairs,
            caption_dir=effective_caption_dir,
            fallback_caption=fallback_caption,
            output_root=predict_root,
            num_samples=args.num_samples,
            num_generations=args.num_generations,
            batch_size=args.batch_size,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            overlay_alpha=args.overlay_alpha,
            seed=args.seed,
            device=device,
            text_embed_cache=text_embed_cache,
            train_data_dir=args.train_data_dir,
            prompt_mode=args.prompt_mode,
            finetuned_transformer_path=args.finetuned_transformer_path,
            start_index=args.start_index,
        )

    logger.info(f"全部推理完成！结果保存在: {predict_root}")


if __name__ == "__main__":
    main()
