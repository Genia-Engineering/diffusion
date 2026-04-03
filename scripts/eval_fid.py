"""FID 批量评估脚本 — 遍历 checkpoint 目录，逐权重生图并计算 FID 分数。

支持多 GPU 分布式推理（通过 accelerate launch 启动），与训练流程保持一致。

架构：
  每个 GPU (rank) 独立加载 pipeline，均分生成任务，本地提取 DINOv2 特征，
  通过 accelerator.gather() 汇聚到 rank 0 计算 FID。

支持模型类型：
  - PixArt-Sigma LoRA (lora_transformer.safetensors)
  - PixArt-Sigma 全量微调 (transformer/ 子目录)
  - PixArt-Sigma ControlNet (controlnet/ 子目录)
  - SDXL LoRA (lora_unet.safetensors)
  - SDXL ControlNet (controlnet/ 子目录)

用法示例：

  # 单卡
  python scripts/eval_fid.py \
    --config configs/lora_pixart_sigma_floorplan.yaml \
    --real_images_dir data/data/llm_1024/floor \
    --num_gen_images 200

  # 多卡（推荐）
  CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --config_file configs/accelerate_4gpu.yaml \
    scripts/eval_fid.py \
    --config configs/lora_pixart_sigma_floorplan.yaml \
    --real_images_dir data/data/llm_1024/floor \
    --num_gen_images 400 --batch_size 4

  # 指定评估特定 step
  CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file configs/accelerate_2gpu.yaml \
    scripts/eval_fid.py \
    --config configs/lora_pixart_sigma_floorplan.yaml \
    --real_images_dir data/data/llm_1024/floor \
    --steps 300 600 900 1200
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.fid import FIDCalculator

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


_DEFAULT_BASE_MODELS = {
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "sd15": "runwayml/stable-diffusion-v1-5",
    "pixart_sigma": "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
}
_DEFAULT_RESOLUTION = {"sdxl": 1024, "sd15": 512, "pixart_sigma": 1024}


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FID 批量评估：遍历 checkpoint，多 GPU 并行生图计算 FID",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, default="",
                    help="训练 YAML 配置路径，自动推导 model_type / output_dir / prompts 等")
    p.add_argument("--checkpoint_dir", type=str, default="",
                    help="checkpoint 目录（含 step_XXXXXX 子目录）；为空时从 config 推导")
    p.add_argument("--model_type", type=str, default="", choices=["", "sdxl", "sd15", "pixart_sigma"],
                    help="模型类型；为空时从 config 读取")
    p.add_argument("--base_model_path", type=str, default="",
                    help="基础模型路径（空时按 model_type 自动查找）")
    p.add_argument("--weights_dir", type=str, default="./weights",
                    help="本地权重根目录")

    fg = p.add_argument_group("FID 参数")
    fg.add_argument("--real_images_dir", type=str, required=True,
                    help="真实参考图目录")
    fg.add_argument("--real_features_cache", type=str, default="",
                    help=".npz 缓存路径（首次生成后复用）；为空时自动放在 checkpoint_dir")
    fg.add_argument("--num_gen_images", type=int, default=1024,
                    help="每个 checkpoint 总生成图数（各 rank 均分）")
    fg.add_argument("--batch_size", type=int, default=8,
                    help="每次 pipeline 调用生成的图像数")
    fg.add_argument("--fid_batch_size", type=int, default=32,
                    help="DINOv2 特征提取批大小")

    gg = p.add_argument_group("生成参数")
    gg.add_argument("--prompts", nargs="*", default=None,
                    help="生成提示词列表（默认从 config 读取）")
    gg.add_argument("--negative_prompt", type=str, default="",
                    help="负面提示词")
    gg.add_argument("--control_image_dir", type=str, default="",
                    help="ControlNet 条件图目录（检测到 controlnet/ 权重时必填）")
    gg.add_argument("--guidance_scale", type=float, default=0,
                    help="CFG scale（0 = 自动：PixArt 4.5, SDXL 7.5）")
    gg.add_argument("--noise_paradigm", type=str, default="",
                    choices=["", "flow_matching", "ddpm"],
                    help="PixArt 噪声范式（空时从 config 读取）")
    gg.add_argument("--num_inference_steps", type=int, default=30,
                    help="去噪步数")
    gg.add_argument("--seed", type=int, default=42,
                    help="基础随机种子（各 rank 自动加偏移）")

    sg = p.add_argument_group("Checkpoint 选择")
    sg.add_argument("--steps", nargs="*", type=int, default=None,
                    help="指定评估哪些 step（如 300 600 900）；留空则全部")

    og = p.add_argument_group("LoRA 参数")
    og.add_argument("--lora_rank", type=int, default=0,
                    help="LoRA 秩（0 = 从 config 读取）")
    og.add_argument("--lora_alpha", type=float, default=0,
                    help="LoRA alpha（0 = 从 config 读取）")
    og.add_argument("--lora_target_modules", nargs="*", default=None,
                    help="LoRA 注入模块（默认从 config 读取）")

    xg = p.add_argument_group("系统参数")
    xg.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    xg.add_argument("--output_file", type=str, default="",
                    help="结果输出路径（默认 {checkpoint_dir}/fid_results.json）")
    xg.add_argument("--plot", action="store_true",
                    help="保存 FID 曲线图")

    return p.parse_args()


def _dtype_from_str(s: str) -> torch.dtype:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[s]


# ── Config 读取 ──────────────────────────────────────────────────────────────


def load_config(config_path: str) -> dict:
    """从 YAML 加载配置，返回普通 dict。"""
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(config_path)
    return OmegaConf.to_container(cfg, resolve=True)


def fill_args_from_config(args: argparse.Namespace) -> None:
    """用 config 填充未显式指定的参数。"""
    if not args.config:
        return
    cfg = load_config(args.config)

    model_cfg = cfg.get("model", {})
    training_cfg = cfg.get("training", {})
    val_cfg = cfg.get("validation", {})
    lora_cfg = cfg.get("lora", {})

    if not args.model_type:
        args.model_type = model_cfg.get("model_type", "pixart_sigma")
    if not args.base_model_path:
        args.base_model_path = model_cfg.get("pretrained_model_name_or_path", "")
    if model_cfg.get("weights_dir"):
        args.weights_dir = model_cfg["weights_dir"]

    if not args.checkpoint_dir:
        output_dir = training_cfg.get("output_dir", "")
        if output_dir:
            args.checkpoint_dir = os.path.join(output_dir, "checkpoints")

    if not args.noise_paradigm:
        args.noise_paradigm = training_cfg.get("noise_paradigm", "flow_matching")

    if args.prompts is None:
        args.prompts = val_cfg.get("prompts", ["architectural floor plan, blueprint, technical drawing"])
    if not args.negative_prompt:
        args.negative_prompt = val_cfg.get("negative_prompt", "")

    if args.lora_rank == 0 and lora_cfg:
        args.lora_rank = lora_cfg.get("rank", 32)
    if args.lora_alpha == 0 and lora_cfg:
        args.lora_alpha = lora_cfg.get("alpha", 32)
    if args.lora_target_modules is None and lora_cfg:
        args.lora_target_modules = lora_cfg.get("target_modules", None)

    data_cfg = cfg.get("data", {})
    if not args.control_image_dir:
        args.control_image_dir = data_cfg.get("conditioning_data_dir", "")

    if args.guidance_scale == 0:
        raw_gs = val_cfg.get("guidance_scale", None)
        if isinstance(raw_gs, list):
            args.guidance_scale = raw_gs[0]
        elif raw_gs is not None:
            args.guidance_scale = float(raw_gs)
        else:
            args.guidance_scale = 4.5 if args.model_type == "pixart_sigma" else 7.5


# ── Checkpoint 扫描 ──────────────────────────────────────────────────────────


def detect_checkpoint_type(ckpt_dir: str) -> dict:
    """检测 checkpoint 目录中的权重类型。"""
    p = Path(ckpt_dir)
    info = {
        "has_lora_transformer": (p / "lora_transformer.safetensors").exists(),
        "has_lora_unet": (p / "lora_unet.safetensors").exists(),
        "has_transformer": (p / "transformer").is_dir(),
        "has_controlnet": (p / "controlnet").is_dir(),
    }
    return info


def scan_checkpoints(checkpoint_dir: str, selected_steps: list[int] | None = None) -> list[tuple[int, str, dict]]:
    """扫描 checkpoint 目录，返回 (step, path, type_info) 列表，按 step 升序排列。"""
    results = []
    ckpt_root = Path(checkpoint_dir)
    if not ckpt_root.exists():
        logger.warning(f"Checkpoint 目录不存在: {checkpoint_dir}")
        return results

    step_pattern = re.compile(r"^step_(\d+)$")
    for entry in sorted(ckpt_root.iterdir()):
        if not entry.is_dir():
            continue
        m = step_pattern.match(entry.name)
        if not m:
            continue
        step = int(m.group(1))
        if selected_steps is not None and step not in selected_steps:
            continue
        type_info = detect_checkpoint_type(str(entry))
        results.append((step, str(entry), type_info))

    results.sort(key=lambda x: x[0])
    return results


# ── 基础模型加载（所有 checkpoint 共用） ─────────────────────────────────────


def _load_base_components(
    model_type: str,
    base_model_path: str,
    weights_dir: str,
    noise_paradigm: str,
    dtype: torch.dtype,
) -> dict:
    """加载基础模型组件，所有 checkpoint 共用以避免重复加载。"""
    if model_type == "pixart_sigma":
        from models.model_loader import load_pixart_sigma_components
        use_fm = noise_paradigm == "flow_matching"
        return load_pixart_sigma_components(
            base_model_path, weights_dir=weights_dir, dtype=dtype, flow_matching=use_fm,
        )
    else:
        from models.model_loader import load_sdxl_components
        return load_sdxl_components(base_model_path, weights_dir=weights_dir, dtype=dtype)


# ── Pipeline 构建（复用 base_components，只加载 delta 权重）─────────────────


def build_pipeline_for_checkpoint(
    ckpt_path: str,
    type_info: dict,
    model_type: str,
    noise_paradigm: str,
    dtype: torch.dtype,
    lora_rank: int = 32,
    lora_alpha: float = 32.0,
    lora_target_modules: list[str] | None = None,
    base_components: dict | None = None,
):
    """根据 checkpoint 类型构建 pipeline。

    base_components 为预加载的基础模型组件，避免每个 checkpoint 重新加载整个模型。
    LoRA 场景下会 deepcopy backbone 后注入，确保不污染原始权重。

    返回 (pipeline, needs_control_image: bool)
    """
    is_pixart = model_type == "pixart_sigma"

    if is_pixart:
        return _build_pixart_pipeline(
            ckpt_path, type_info, noise_paradigm, dtype,
            lora_rank, lora_alpha, lora_target_modules, base_components,
        )
    else:
        return _build_sdxl_pipeline(
            ckpt_path, type_info, dtype,
            lora_rank, lora_alpha, lora_target_modules, base_components,
        )


def _build_pixart_pipeline(
    ckpt_path, type_info, noise_paradigm, dtype,
    lora_rank, lora_alpha, lora_target_modules, base_components,
):
    import copy

    from diffusers import PixArtSigmaPipeline
    from models.lora import LoRAInjector, load_lora_weights
    from models.model_loader import patch_fm_scheduler_for_pipeline

    use_fm = noise_paradigm == "flow_matching"
    transformer = base_components.get("transformer")

    if type_info["has_lora_transformer"]:
        transformer = copy.deepcopy(transformer)
        target_modules = lora_target_modules or [
            "to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2",
        ]
        LoRAInjector.inject(transformer, lora_rank, lora_alpha, target_modules)
        lora_path = os.path.join(ckpt_path, "lora_transformer.safetensors")
        load_lora_weights(transformer, lora_path)
        logger.info(f"PixArt LoRA 权重已加载: {lora_path}")

    elif type_info["has_transformer"]:
        from diffusers import PixArtTransformer2DModel
        transformer_dir = os.path.join(ckpt_path, "transformer")
        transformer = PixArtTransformer2DModel.from_pretrained(transformer_dir, torch_dtype=dtype)
        logger.info(f"PixArt 全量 transformer 权重已加载: {transformer_dir}")

    if use_fm:
        from diffusers import FlowMatchEulerDiscreteScheduler
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(base_components["noise_scheduler"].config)
        patch_fm_scheduler_for_pipeline(scheduler)
    else:
        from diffusers import DPMSolverMultistepScheduler
        scheduler = DPMSolverMultistepScheduler.from_config(
            base_components["noise_scheduler"].config,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True,
        )

    if type_info["has_controlnet"]:
        from models.controlnet_pixart import PixArtControlNetAdapterModel, PixArtControlNetTransformerModel
        from pipelines.pixart_controlnet_pipeline import PixArtControlNetPipeline

        controlnet = PixArtControlNetAdapterModel.from_pretrained(
            os.path.join(ckpt_path, "controlnet"), torch_dtype=dtype,
        )
        cn_transformer = PixArtControlNetTransformerModel(transformer, controlnet)

        pipeline = PixArtControlNetPipeline(
            tokenizer=base_components["tokenizer"],
            text_encoder=base_components["text_encoder"],
            vae=base_components["vae"],
            transformer=cn_transformer,
            controlnet=None,
            scheduler=scheduler,
        )
        return pipeline, True

    pipeline = PixArtSigmaPipeline(
        vae=base_components["vae"],
        transformer=transformer,
        text_encoder=base_components["text_encoder"],
        tokenizer=base_components["tokenizer"],
        scheduler=scheduler,
    )
    return pipeline, False


def _build_sdxl_pipeline(
    ckpt_path, type_info, dtype,
    lora_rank, lora_alpha, lora_target_modules, base_components,
):
    import copy

    from diffusers import (
        ControlNetModel,
        DPMSolverSDEScheduler,
        StableDiffusionXLControlNetPipeline,
        StableDiffusionXLPipeline,
    )
    from models.lora import LoRAInjector, load_lora_weights

    unet = base_components.get("unet")

    if type_info["has_lora_unet"]:
        unet = copy.deepcopy(unet)
        target_modules = lora_target_modules or ["to_q", "to_k", "to_v", "to_out.0"]
        LoRAInjector.inject_unet(unet, lora_rank, lora_alpha, target_modules)
        lora_path = os.path.join(ckpt_path, "lora_unet.safetensors")
        load_lora_weights(unet, lora_path)
        logger.info(f"SDXL LoRA 权重已加载: {lora_path}")

    scheduler = DPMSolverSDEScheduler.from_config(
        base_components["noise_scheduler"].config, use_karras_sigmas=True,
    )

    common = dict(
        vae=base_components["vae"],
        unet=unet,
        text_encoder=base_components["text_encoder"],
        text_encoder_2=base_components["text_encoder_2"],
        tokenizer=base_components["tokenizer"],
        tokenizer_2=base_components["tokenizer_2"],
        scheduler=scheduler,
    )

    if type_info["has_controlnet"]:
        controlnet = ControlNetModel.from_pretrained(
            os.path.join(ckpt_path, "controlnet"), torch_dtype=dtype,
        )
        pipeline = StableDiffusionXLControlNetPipeline(**common, controlnet=controlnet)
        return pipeline, True

    pipeline = StableDiffusionXLPipeline(**common)
    return pipeline, False


# ── 图像加载 ─────────────────────────────────────────────────────────────────


def load_images_from_dir(
    image_dir: str,
    max_n: int = 0,
    seed: int = 42,
    extensions: set[str] = (".png", ".jpg", ".jpeg", ".webp"),
) -> list[Image.Image]:
    """从目录加载图像，按文件名排序后可选随机采样。"""
    paths = sorted([
        p for p in Path(image_dir).rglob("*")
        if p.suffix.lower() in extensions and p.is_file()
    ])
    if not paths:
        raise FileNotFoundError(f"未找到图像文件: {image_dir}")

    if max_n > 0 and len(paths) > max_n:
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(paths), size=max_n, replace=False)
        paths = [paths[i] for i in sorted(indices)]

    images = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            images.append(img)
        except Exception as e:
            logger.warning(f"跳过损坏图像 {p}: {e}")
    return images


# ── 分布式生成 ────────────────────────────────────────────────────────────────


@torch.no_grad()
def generate_and_extract_features(
    pipeline,
    fid_calc: "FIDCalculator",
    num_images: int,
    prompts: list[str],
    negative_prompt: str,
    guidance_scale: float,
    num_inference_steps: int,
    batch_size: int,
    fid_batch_size: int,
    seed: int,
    seed_offset: int,
    device: torch.device,
    control_images: list[Image.Image] | None = None,
    controlnet_conditioning_scale: float = 1.0,
    is_pixart_controlnet: bool = False,
) -> np.ndarray:
    """生成图像并流式提取 DINOv2 特征，避免所有 PIL 图像同时驻留内存。

    每生成一个 batch 的图像后立即提取特征并丢弃 PIL 对象，
    峰值 CPU 内存从 N×3MB 降至 batch_size×3MB。
    DINOv2 ViT-B/14 仅 ~170MB，可与 pipeline 共存于 GPU。

    Returns:
        (N, D) numpy 特征矩阵
    """
    if num_images <= 0:
        return np.empty((0, 768), dtype=np.float32)

    all_features: list[np.ndarray] = []
    prompt_cycle = prompts * (num_images // len(prompts) + 1)
    control_cycle = control_images * (num_images // len(control_images) + 1) if control_images else None

    pending_images: list[Image.Image] = []

    def _flush_pending():
        nonlocal pending_images
        if not pending_images:
            return
        feats = fid_calc._extract_features(pending_images, batch_size=fid_batch_size)
        all_features.append(feats)
        pending_images = []

    if control_cycle is not None and is_pixart_controlnet:
        for idx in range(num_images):
            generator = torch.Generator(device="cpu").manual_seed(seed + seed_offset + idx + 10000)
            kwargs = {
                "prompt": prompt_cycle[idx % len(prompts)],
                "negative_prompt": negative_prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "generator": generator,
                "num_images_per_prompt": 1,
                "image": control_cycle[idx % len(control_images)],
                "controlnet_conditioning_scale": controlnet_conditioning_scale,
            }
            output = pipeline(**kwargs)
            pending_images.extend(output.images)
            if len(pending_images) >= fid_batch_size:
                _flush_pending()
        _flush_pending()
        return np.concatenate(all_features, axis=0)[:num_images]

    for batch_start in range(0, num_images, batch_size):
        batch_end = min(batch_start + batch_size, num_images)
        batch_indices = list(range(batch_start, batch_end))

        generators = [
            torch.Generator(device="cpu").manual_seed(seed + seed_offset + idx + 10000)
            for idx in batch_indices
        ]

        kwargs = {
            "prompt": [prompt_cycle[idx % len(prompts)] for idx in batch_indices],
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generators,
            "num_images_per_prompt": 1,
        }

        if control_cycle is not None:
            kwargs["image"] = [control_cycle[idx % len(control_images)] for idx in batch_indices]
            kwargs["controlnet_conditioning_scale"] = controlnet_conditioning_scale

        output = pipeline(**kwargs)
        pending_images.extend(output.images)
        if len(pending_images) >= fid_batch_size:
            _flush_pending()

    _flush_pending()
    return np.concatenate(all_features, axis=0)[:num_images]


# ── 结果保存 ──────────────────────────────────────────────────────────────────


def save_results(results: dict[int, float], output_path: str, plot: bool = False) -> None:
    """保存 FID 结果为 JSON + CSV，可选曲线图。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump({str(k): v for k, v in sorted(results.items())}, f, indent=2)
    logger.info(f"JSON 结果已保存: {output_path}")

    csv_path = output_path.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "fid"])
        for step, fid in sorted(results.items()):
            writer.writerow([step, f"{fid:.4f}"])
    logger.info(f"CSV 结果已保存: {csv_path}")

    if plot and len(results) >= 2:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            steps = sorted(results.keys())
            fids = [results[s] for s in steps]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(steps, fids, "o-", linewidth=2, markersize=6)
            ax.set_xlabel("Training Step", fontsize=13)
            ax.set_ylabel("FID (DINOv2)", fontsize=13)
            ax.set_title("FID Score vs Training Step", fontsize=15)
            ax.grid(True, alpha=0.3)

            for s, f in zip(steps, fids):
                ax.annotate(f"{f:.1f}", (s, f), textcoords="offset points",
                            xytext=(0, 10), ha="center", fontsize=8)

            fig.tight_layout()
            plot_path = output_path.with_suffix(".png")
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            logger.info(f"曲线图已保存: {plot_path}")
        except ImportError:
            logger.warning("matplotlib 不可用，跳过曲线图生成")


# ── 主函数 ────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    fill_args_from_config(args)

    if not args.model_type:
        args.model_type = "pixart_sigma"
    if not args.base_model_path:
        args.base_model_path = _DEFAULT_BASE_MODELS.get(args.model_type, "")
    if not args.checkpoint_dir:
        raise ValueError("必须指定 --checkpoint_dir 或通过 --config 自动推导")

    dtype = _dtype_from_str(args.dtype)
    resolution = _DEFAULT_RESOLUTION[args.model_type]

    if args.guidance_scale == 0:
        args.guidance_scale = 4.5 if args.model_type == "pixart_sigma" else 7.5

    # ── accelerate 初始化 ────────────────────────────────────────────────
    from accelerate import Accelerator
    accelerator = Accelerator()
    device = accelerator.device
    is_main = accelerator.is_main_process
    num_processes = accelerator.num_processes
    process_index = accelerator.process_index

    # 更新日志格式以包含 rank 信息
    for handler in logging.root.handlers:
        handler.setFormatter(logging.Formatter(
            f"%(asctime)s [%(levelname)s] [Rank {process_index}] %(message)s",
            datefmt="%H:%M:%S",
        ))

    logger.info(f"FID 评估启动: model_type={args.model_type}, num_processes={num_processes}")
    logger.info(f"checkpoint_dir={args.checkpoint_dir}")
    logger.info(f"num_gen_images={args.num_gen_images}, batch_size={args.batch_size}")

    # ── 扫描 checkpoints ─────────────────────────────────────────────────
    checkpoints = scan_checkpoints(args.checkpoint_dir, args.steps)
    if not checkpoints:
        logger.error(f"未找到任何 checkpoint: {args.checkpoint_dir}")
        return

    if is_main:
        for step, path, info in checkpoints:
            types = []
            if info["has_lora_transformer"]:
                types.append("LoRA-Transformer")
            if info["has_lora_unet"]:
                types.append("LoRA-UNet")
            if info["has_transformer"]:
                types.append("Full-Transformer")
            if info["has_controlnet"]:
                types.append("ControlNet")
            logger.info(f"  step {step:06d}: {', '.join(types) or 'unknown'}")

    # ── 加载真实图像特征（仅 rank 0 执行，其余等待）───────────────────────
    from utils.fid import FIDCalculator, _compute_fid

    cache_path = args.real_features_cache
    if not cache_path:
        cache_path = os.path.join(args.checkpoint_dir, "fid_real_features.npz")

    fid_calc = FIDCalculator(
        model_name="dinov2_vitb14",
        device=device,
        real_images_cache_path=cache_path,
    )

    if is_main:
        real_images = load_images_from_dir(
            args.real_images_dir, max_n=args.num_gen_images, seed=args.seed,
        )
        logger.info(f"加载了 {len(real_images)} 张真实图像用于 FID 计算")
        fid_calc.update_real(real_images, batch_size=args.fid_batch_size)
        del real_images
        gc.collect()

    accelerator.wait_for_everyone()

    # 所有 rank 需要能提取特征，触发模型加载
    if not is_main:
        fid_calc._get_model()

    # 主进程广播真实图像统计量到所有 rank（仅 rank 0 有 _real_mu/_real_sigma）
    # 其他 rank 不需要这些统计量，只需提取生成图特征
    accelerator.wait_for_everyone()

    # ── 加载 ControlNet 条件图（所有 rank 共用同一批图） ──────────────────
    control_images: list[Image.Image] | None = None
    first_needs_control = any(info["has_controlnet"] for _, _, info in checkpoints)
    if first_needs_control:
        if not args.control_image_dir:
            logger.error("检测到 ControlNet checkpoint 但未指定 --control_image_dir")
            return
        control_images = load_images_from_dir(
            args.control_image_dir, max_n=args.num_gen_images, seed=args.seed,
        )
        for i in range(len(control_images)):
            control_images[i] = control_images[i].resize((resolution, resolution), Image.LANCZOS)
        logger.info(f"加载了 {len(control_images)} 张条件图像")

    # ── 主评估循环 ───────────────────────────────────────────────────────
    results: dict[int, float] = {}

    if args.num_gen_images < num_processes:
        logger.warning(
            f"num_gen_images ({args.num_gen_images}) < num_processes ({num_processes})，"
            f"调整为 {num_processes}"
        )
        args.num_gen_images = num_processes

    num_this_rank = args.num_gen_images // num_processes
    remainder = args.num_gen_images % num_processes
    if process_index < remainder:
        num_this_rank += 1
    my_seed_offset = process_index * (args.num_gen_images // num_processes) + min(process_index, remainder)

    logger.info(
        f"本 rank 生成 {num_this_rank} 张 (总计 {args.num_gen_images} 张, {num_processes} ranks), "
        f"seed_offset={my_seed_offset}"
    )

    # 预加载 base components（所有 checkpoint 共用，避免重复加载）
    base_components = _load_base_components(
        args.model_type, args.base_model_path, args.weights_dir,
        args.noise_paradigm, dtype,
    )
    logger.info("基础模型组件加载完成（所有 checkpoint 共用）")

    # full-transformer checkpoint 会加载独立权重，base transformer/unet 不被使用，
    # 提前释放以节省 ~2.5GB 内存
    all_full_weight = all(
        info["has_transformer"] and not info["has_lora_transformer"] and not info["has_lora_unet"]
        for _, _, info in checkpoints
    )
    if all_full_weight:
        backbone_key = "transformer" if args.model_type == "pixart_sigma" else "unet"
        if backbone_key in base_components:
            del base_components[backbone_key]
            gc.collect()
            logger.info(f"所有 checkpoint 均为 full-weight，已释放未使用的 base {backbone_key}")

    for ckpt_idx, (step, ckpt_path, type_info) in enumerate(checkpoints):
        logger.info(f"[{ckpt_idx+1}/{len(checkpoints)}] 评估 step {step:06d} ...")

        # 构建 pipeline（复用 base_components，只加载 delta 权重）
        try:
            pipeline, needs_control = build_pipeline_for_checkpoint(
                ckpt_path, type_info,
                model_type=args.model_type,
                noise_paradigm=args.noise_paradigm,
                dtype=dtype,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_target_modules=args.lora_target_modules,
                base_components=base_components,
            )
            pipeline = pipeline.to(device)
            pipeline.set_progress_bar_config(disable=not is_main)
        except Exception as e:
            logger.error(f"step {step} pipeline 加载失败: {e}")
            accelerator.wait_for_everyone()
            continue

        is_pixart_cn = args.model_type == "pixart_sigma" and needs_control
        ctl_imgs = control_images if needs_control else None
        features_np = generate_and_extract_features(
            pipeline=pipeline,
            fid_calc=fid_calc,
            num_images=num_this_rank,
            prompts=args.prompts,
            negative_prompt=args.negative_prompt,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            batch_size=args.batch_size,
            fid_batch_size=args.fid_batch_size,
            seed=args.seed,
            seed_offset=my_seed_offset,
            device=device,
            control_images=ctl_imgs,
            is_pixart_controlnet=is_pixart_cn,
        )
        logger.info(f"本 rank 已生成并提取 {features_np.shape[0]} 张图像特征")

        del pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        features_tensor = torch.from_numpy(features_np).to(device)
        del features_np

        # 跨卡汇聚特征
        # gather 要求所有 rank tensor shape 一致，不均分时需 pad
        if num_processes > 1:
            max_per_rank = (args.num_gen_images + num_processes - 1) // num_processes
            actual_n = features_tensor.shape[0]
            if actual_n < max_per_rank:
                pad = torch.zeros(max_per_rank - actual_n, features_tensor.shape[1],
                                  device=device, dtype=features_tensor.dtype)
                features_tensor = torch.cat([features_tensor, pad], dim=0)
            gathered = accelerator.gather(features_tensor)
            # 去掉 padding：只保留前 num_gen_images 行
            gathered = gathered[:args.num_gen_images]
        else:
            gathered = features_tensor

        # rank 0 计算 FID
        if is_main and fid_calc.is_ready():
            try:
                gathered_np = gathered.cpu().float().numpy()
                gen_mu, gen_sigma = FIDCalculator._calc_stats(gathered_np)
                fid_score = _compute_fid(
                    fid_calc._real_mu, fid_calc._real_sigma,
                    gen_mu, gen_sigma,
                )
                results[step] = fid_score
                logger.info(
                    f"step {step:06d} — FID: {fid_score:.4f} "
                    f"(n_gen={gathered_np.shape[0]}, {num_processes} ranks)"
                )
            except Exception as e:
                logger.warning(f"step {step} FID 计算失败: {e}")

        del features_tensor, gathered
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        accelerator.wait_for_everyone()

    # ── 保存结果 ─────────────────────────────────────────────────────────
    if is_main and results:
        output_path = args.output_file
        if not output_path:
            output_path = os.path.join(args.checkpoint_dir, "fid_results.json")
        save_results(results, output_path, plot=args.plot)

        logger.info("=" * 60)
        logger.info("FID 评估汇总:")
        logger.info("-" * 40)
        best_step = min(results, key=results.get)
        for step in sorted(results):
            marker = " ★ best" if step == best_step else ""
            logger.info(f"  step {step:06d}: FID = {results[step]:.4f}{marker}")
        logger.info("=" * 60)

    accelerator.wait_for_everyone()
    logger.info("FID 评估完成")


if __name__ == "__main__":
    main()
