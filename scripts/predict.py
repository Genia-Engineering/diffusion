"""推理脚本 — 支持 SD1.5 / SDXL / PixArt-Sigma，可选 LoRA / ControlNet / img2img。

用法示例:

  # SDXL + merged 模型（纯文生图）
  python scripts/predict.py \
    --model_type sdxl \
    --merged_model_path ./outputs/lora_sdxl_floorplan/merged_step_000810

  # SDXL + LoRA delta + ControlNet（逐 prompt 条件图模式）
  python scripts/predict.py \
    --model_type sdxl \
    --lora_checkpoint ./outputs/lora_sdxl_floorplan/checkpoints/step_000810 \
    --controlnet_paths ./outputs/controlnet_sdxl/checkpoints/step_016000/controlnet \
    --control_images img1.png img2.png --controlnet_scales 0.7 --num_images 1

  # SD1.5 + LoRA delta
  python scripts/predict.py \
    --model_type sd15 \
    --lora_checkpoint ./outputs/lora_sd15_floorplan/checkpoints/step_002130

  # PixArt-Sigma t2i（纯文生图）
  python scripts/predict.py \
    --model_type pixart_sigma \
    --merged_model_path ./outputs/pixart_sigma_floorplan/checkpoints/step_005000

  # PixArt-Sigma img2img（参考图引导生成，strength 控制偏离程度）
  python scripts/predict.py \
    --model_type pixart_sigma \
    --merged_model_path ./outputs/pixart_sigma_floorplan/checkpoints/step_005000 \
    --input_images data/data/size_1024/image001.png \
    --strength 0.5

  # img2img 多张参考图 × 多个 prompt（一一配对）
  python scripts/predict.py \
    --model_type pixart_sigma \
    --merged_model_path ./outputs/pixart_sigma_floorplan/checkpoints/step_005000 \
    --input_images img1.png img2.png \
    --prompt "prompt for img1" "prompt for img2" \
    --strength 0.3 --num_images 2

  # SDXL img2img 同样支持
  python scripts/predict.py \
    --model_type sdxl \
    --merged_model_path ./outputs/lora_sdxl_floorplan/merged_step_000810 \
    --input_images data/data/size_1024/image001.png \
    --strength 0.5
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
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── 默认路径（按 model_type 自动选择）──────────────────────────────────────
_DEFAULT_BASE_MODELS = {
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "sd15":  "runwayml/stable-diffusion-v1-5",
    "pixart_sigma": "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
}
_DEFAULT_RESOLUTION = {"sdxl": 1024, "sd15": 512, "pixart_sigma": 1024}


# ── CLI 参数解析 ────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stable Diffusion 推理脚本（SD1.5 / SDXL + LoRA + ControlNet）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 模型配置
    mg = p.add_argument_group("模型配置")
    mg.add_argument("--model_type", type=str, default="sdxl", choices=["sd15", "sdxl", "pixart_sigma"],
                    help="基础模型类型")
    mg.add_argument("--base_model_path", type=str, default="",
                    help="原始基础模型路径（空时自动按 model_type 从 weights_dir 查找）；"
                         "仅在走 --lora_checkpoint 路径时使用，与 --merged_model_path 无关")
    mg.add_argument("--weights_dir", type=str, default="./weights",
                    help="本地权重根目录")
    mg.add_argument("--merged_model_path", type=str, default=r"/home/daiqing_tan/stable_diffusion_lora/outputs/lora_sdxl_floorplan/merged_step_000810",
                    help="已 merged 的完整模型目录（含 unet/vae/tokenizer 等子目录）。"
                         "与 --lora_checkpoint 二选一；优先级更高")
    mg.add_argument("--lora_checkpoint", type=str, default="",
                    help="LoRA checkpoint 目录（含 lora_unet.safetensors）")
    mg.add_argument("--lora_rank", type=int, default=32,
                    help="LoRA 秩（需与训练时一致）")
    mg.add_argument("--lora_alpha", type=float, default=32.0,
                    help="LoRA alpha（需与训练时一致）")
    mg.add_argument("--lora_target_modules", nargs="+",
                    default=["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"],
                    help="LoRA 注入的目标模块名称")

    # ControlNet
    cg = p.add_argument_group("ControlNet（可选，支持多个）")
    cg.add_argument("--controlnet_paths", nargs="*", default=[],
                    help="ControlNet checkpoint 目录列表（diffusers 格式，可多个）")
    cg.add_argument("--control_images", nargs="*", default=[],
                    help="条件图像路径列表，数量须与 --controlnet_paths 对应")
    cg.add_argument("--controlnet_scales", nargs="*", type=float, default=[],
                    help="每个 ControlNet 的 conditioning_scale，默认全部 1.0")

    # 推理参数
    ig = p.add_argument_group("推理参数")
    ig.add_argument("--prompt", nargs="+",
                    default=["architectural floor plan, blueprint, technical drawing"],
                    help="生成提示词（可多个，每个独立生成一批图像）")
    ig.add_argument("--negative_prompt", type=str,
                    default="blurry, noisy, low quality, photo, realistic, watermark",
                    help="负提示词")
    ig.add_argument("--num_inference_steps", type=int, default=50,
                    help="去噪步数")
    ig.add_argument("--guidance_scale", type=float, default=7.5,
                    help="无分类器引导强度")
    ig.add_argument("--width", type=int, default=0,
                    help="生成宽度（0 = 按 model_type 自动：sdxl→1024，sd15→512）")
    ig.add_argument("--height", type=int, default=0,
                    help="生成高度（0 = 按 model_type 自动：sdxl→1024，sd15→512）")
    ig.add_argument("--num_images", type=int, default=4,
                    help="每个 prompt 生成的图像数量")
    ig.add_argument("--seed", type=int, default=40,
                    help="随机种子（-1 = 随机）")

    # img2img 参数
    i2i = p.add_argument_group("img2img（可选）")
    i2i.add_argument("--input_images", nargs="*", default=[],
                     help="img2img 参考图像路径列表；提供后进入 img2img 模式")
    i2i.add_argument("--strength", type=float, default=0.7,
                     help="img2img 噪声强度：0 附近=保留原图结构，1.0=接近纯文生图")

    # 噪声范式（仅 PixArt-Sigma 有效）
    i2i.add_argument("--noise_paradigm", type=str, default="flow_matching",
                     choices=["flow_matching", "ddpm"],
                     help="PixArt-Sigma 训练时使用的噪声范式：flow_matching 或 ddpm。"
                          "决定推理 scheduler 和 img2img 加噪方式")

    # 系统参数
    sg = p.add_argument_group("系统参数")
    sg.add_argument("--scheduler", type=str, default="dpm_sde_karras",
                    choices=["euler", "dpm", "dpm++_sde_karras", "ddpm", "dpm_sde_karras", "flow_match_euler"],
                    help="推理调度器：euler=EulerDiscreteScheduler，"
                         "dpm=DPMSolverMultistepScheduler(dpmsolver++)，"
                         "dpm++_sde_karras=DPMSolverMultistepScheduler(sde-dpmsolver++, karras)，"
                         "ddpm=DDPMScheduler，"
                         "dpm_sde_karras=DPMSolverSDEScheduler(karras, 仅 SD/SDXL)，"
                         "flow_match_euler=FlowMatchEulerDiscreteScheduler(PixArt-Sigma FM)")
    sg.add_argument("--dtype", type=str, default="bf16",
                    choices=["fp32", "fp16", "bf16"],
                    help="模型精度（所有组件统一精度，推理用 bf16 即可）")
    sg.add_argument("--gpu", type=int, default=-1,
                    help="指定使用哪块 GPU 的索引（如 --gpu 1 表示 cuda:1）；"
                         "-1 表示使用 CUDA_VISIBLE_DEVICES 或默认 cuda:0")
    sg.add_argument("--device", type=str, default="cuda",
                    help="推理设备（cuda / cpu）；当 --gpu 有效时自动覆盖为 cuda:<id>")
    sg.add_argument("--output_dir", type=str, default="./outputs/predict/dpm_sde_karras",
                    help="输出目录")

    return p.parse_args()


# ── 工具函数 ────────────────────────────────────────────────────────────────

def _dtype_from_str(s: str) -> torch.dtype:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[s]


def _prompt_tag(prompt: str, max_len: int = 32) -> str:
    """从 prompt 生成简短标签（截断 + 哈希后缀）。"""
    slug = prompt[:max_len].strip().replace(" ", "_").replace(",", "").replace("/", "-")
    h = hashlib.md5(prompt.encode()).hexdigest()[:6]
    return f"{slug}__{h}"


def _load_control_image(path: str, width: int, height: int) -> Image.Image:
    """加载并 resize 条件图像到目标分辨率。"""
    img = Image.open(path).convert("RGB")
    img = img.resize((width, height), Image.LANCZOS)
    return img


def _encode_image_to_latent(
    vae, image: Image.Image, device: torch.device,
) -> torch.Tensor:
    """PIL Image → VAE clean latent (1, C, H/f, W/f), float32, 已乘 scaling_factor。

    调用方需确保 image 已 resize 到目标分辨率。
    """
    import torchvision.transforms.functional as TF

    img_tensor = TF.to_tensor(image.convert("RGB")).unsqueeze(0)
    img_tensor = img_tensor.to(device=device, dtype=vae.dtype) * 2.0 - 1.0
    with torch.no_grad():
        z = vae.encode(img_tensor).latent_dist.sample()
    return (z * vae.config.scaling_factor).float()


def _make_comparison_grid(
    ctrl_img: Image.Image,
    gen_img: Image.Image,
    alpha: float = 0.45,
    gap: int = 8,
    bg_color: tuple = (20, 20, 20),
) -> Image.Image:
    """生成三栏对比图：[控制图 ｜ 半透明叠加 ｜ 生成图]。

    Args:
        ctrl_img: 条件控制图（会自动 resize 到与 gen_img 相同尺寸）
        gen_img:  模型生成图
        alpha:    叠加透明度，0=纯生成图，1=纯控制图（默认 0.45）
        gap:      栏间隙像素（默认 8）
        bg_color: 背景色 RGB（默认深灰）
    """
    w, h = gen_img.size
    ctrl = ctrl_img.resize((w, h), Image.LANCZOS).convert("RGB")
    gen = gen_img.convert("RGB")
    blend = Image.blend(gen, ctrl, alpha=alpha)

    canvas = Image.new("RGB", (w * 3 + gap * 2, h), color=bg_color)
    canvas.paste(ctrl,  (0,              0))
    canvas.paste(blend, (w + gap,        0))
    canvas.paste(gen,   (w * 2 + gap * 2, 0))
    return canvas


def _save_metadata(output_dir: Path, args: argparse.Namespace, run_info: dict) -> None:
    """将本次推理的参数保存为 metadata.json。"""
    meta = {
        "model_type": args.model_type,
        "merged_model_path": args.merged_model_path,
        "lora_checkpoint": args.lora_checkpoint,
        "controlnet_paths": args.controlnet_paths,
        "controlnet_scales": args.controlnet_scales,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "width": run_info["width"],
        "height": run_info["height"],
        "num_images": args.num_images,
        "seed": args.seed,
        "scheduler": args.scheduler,
        "dtype": args.dtype,
        "input_images": getattr(args, "input_images", []),
        "strength": getattr(args, "strength", None),
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info(f"Metadata saved to {output_dir / 'metadata.json'}")


# ── 模型加载 ────────────────────────────────────────────────────────────────

def load_pipeline(args: argparse.Namespace, dtype: torch.dtype, controlnets: list):
    """根据参数构建并返回 diffusers pipeline。

    两条加载路径（优先级：merged > lora_checkpoint > 原始基础模型）：
      A. merged_model_path 非空 → StableDiffusion(XL)Pipeline.from_pretrained(merged_path)
      B. lora_checkpoint 非空  → 加载基础模型 + 注入 LoRA + 加载 delta 权重
      C. 两者均空              → 加载原始基础模型（无 LoRA）
    """
    from diffusers import (
        ControlNetModel,
        StableDiffusionControlNetPipeline,
        StableDiffusionPipeline,
        StableDiffusionXLControlNetPipeline,
        StableDiffusionXLPipeline,
    )
    from models.lora import LoRAInjector, load_lora_weights
    from models.model_loader import load_sd15_components, load_sdxl_components, resolve_model_path

    is_sdxl = args.model_type == "sdxl"
    is_pixart = args.model_type == "pixart_sigma"
    has_controlnet = len(controlnets) > 0
    cn_arg = controlnets[0] if len(controlnets) == 1 else (controlnets if controlnets else None)

    # ── PixArt-Sigma 路径 ─────────────────────────────────────────────────
    if is_pixart:
        return _load_pixart_sigma_pipeline(args, dtype)

    # ── 路径 A：merged 完整模型 ──────────────────────────────────────────────
    if args.merged_model_path:
        merged_path = args.merged_model_path
        logger.info(f"[路径A] 从 merged 模型加载: {merged_path}")

        if is_sdxl:
            PipelineCls = StableDiffusionXLControlNetPipeline if has_controlnet else StableDiffusionXLPipeline
        else:
            PipelineCls = StableDiffusionControlNetPipeline if has_controlnet else StableDiffusionPipeline

        extra = {}
        if has_controlnet:
            extra["controlnet"] = cn_arg
        if not is_sdxl:
            extra.update({"safety_checker": None, "feature_extractor": None})

        pipeline = PipelineCls.from_pretrained(merged_path, torch_dtype=dtype, **extra)
        return pipeline

    # ── 路径 B / C：组件式加载 ──────────────────────────────────────────────
    base_path = args.base_model_path or _DEFAULT_BASE_MODELS[args.model_type]
    logger.info(f"[路径B/C] 从基础模型加载: {base_path}")

    if is_sdxl:
        components = load_sdxl_components(base_path, weights_dir=args.weights_dir or None, dtype=dtype)
    else:
        components = load_sd15_components(base_path, weights_dir=args.weights_dir or None, dtype=dtype)

    unet = components["unet"]

    # 注入 LoRA 并加载 delta 权重
    if args.lora_checkpoint:
        lora_unet_path = os.path.join(args.lora_checkpoint, "lora_unet.safetensors")
        if not os.path.exists(lora_unet_path):
            raise FileNotFoundError(f"LoRA UNet 权重不存在: {lora_unet_path}")

        logger.info(f"注入 LoRA（rank={args.lora_rank}, alpha={args.lora_alpha}）...")
        injected = LoRAInjector.inject_unet(unet, args.lora_rank, args.lora_alpha, args.lora_target_modules)
        logger.info(f"  注入层数: {len(injected)}")

        logger.info(f"加载 LoRA delta 权重: {lora_unet_path}")
        load_lora_weights(unet, lora_unet_path)
    else:
        logger.info("未指定 LoRA checkpoint，使用原始基础模型权重")

    # 构建 pipeline
    if is_sdxl:
        if has_controlnet:
            pipeline = StableDiffusionXLControlNetPipeline(
                vae=components["vae"],
                unet=unet,
                text_encoder=components["text_encoder"],
                text_encoder_2=components["text_encoder_2"],
                tokenizer=components["tokenizer"],
                tokenizer_2=components["tokenizer_2"],
                scheduler=components["noise_scheduler"],
                controlnet=cn_arg,
            )
        else:
            pipeline = StableDiffusionXLPipeline(
                vae=components["vae"],
                unet=unet,
                text_encoder=components["text_encoder"],
                text_encoder_2=components["text_encoder_2"],
                tokenizer=components["tokenizer"],
                tokenizer_2=components["tokenizer_2"],
                scheduler=components["noise_scheduler"],
            )
    else:
        if has_controlnet:
            pipeline = StableDiffusionControlNetPipeline(
                vae=components["vae"],
                unet=unet,
                text_encoder=components["text_encoder"],
                tokenizer=components["tokenizer"],
                scheduler=components["noise_scheduler"],
                controlnet=cn_arg,
                safety_checker=None,
                feature_extractor=None,
            )
        else:
            pipeline = StableDiffusionPipeline(
                vae=components["vae"],
                unet=unet,
                text_encoder=components["text_encoder"],
                tokenizer=components["tokenizer"],
                scheduler=components["noise_scheduler"],
                safety_checker=None,
                feature_extractor=None,
            )

    return pipeline


def _load_pixart_sigma_pipeline(args, dtype):
    """加载 PixArt-Sigma pipeline（merged 模型或基础模型 + 微调 transformer）。

    根据 args.noise_paradigm 选择推理 scheduler:
      - flow_matching: FlowMatchEulerDiscreteScheduler (Euler ODE)
      - ddpm: DPMSolverMultistepScheduler (dpmsolver++)
    """
    from diffusers import PixArtSigmaPipeline
    from models.model_loader import load_pixart_sigma_components

    use_flow_matching = getattr(args, "noise_paradigm", "flow_matching") == "flow_matching"

    components = load_pixart_sigma_components(
        _DEFAULT_BASE_MODELS["pixart_sigma"],
        weights_dir=args.weights_dir or None,
        dtype=dtype,
        flow_matching=use_flow_matching,
    )

    if use_flow_matching:
        from diffusers import FlowMatchEulerDiscreteScheduler
        inference_scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            components["noise_scheduler"].config,
        )
        from models.model_loader import patch_fm_scheduler_for_pipeline
        patch_fm_scheduler_for_pipeline(inference_scheduler)
        logger.info("[PixArt-Sigma] 噪声范式: Flow Matching → FlowMatchEulerDiscreteScheduler")
    else:
        from diffusers import DPMSolverMultistepScheduler
        inference_scheduler = DPMSolverMultistepScheduler.from_config(
            components["noise_scheduler"].config,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True,
        )
        logger.info("[PixArt-Sigma] 噪声范式: DDPM → DPMSolverMultistepScheduler (sde-dpmsolver++, karras)")

    if args.merged_model_path:
        logger.info(f"[PixArt-Sigma] 从 merged/finetuned 模型加载: {args.merged_model_path}")
        merged_path = args.merged_model_path

        from diffusers import PixArtTransformer2DModel
        transformer_path = os.path.join(merged_path, "transformer")
        if os.path.isdir(transformer_path):
            transformer = PixArtTransformer2DModel.from_pretrained(transformer_path, torch_dtype=dtype)
        else:
            transformer = PixArtTransformer2DModel.from_pretrained(merged_path, torch_dtype=dtype)

        pipeline = PixArtSigmaPipeline(
            vae=components["vae"],
            transformer=transformer,
            text_encoder=components["text_encoder"],
            tokenizer=components["tokenizer"],
            scheduler=inference_scheduler,
        )
        return pipeline

    logger.info("[PixArt-Sigma] 从基础模型加载")
    pipeline = PixArtSigmaPipeline(
        vae=components["vae"],
        transformer=components["transformer"],
        text_encoder=components["text_encoder"],
        tokenizer=components["tokenizer"],
        scheduler=inference_scheduler,
    )
    return pipeline


def load_controlnets(paths: list[str], dtype: torch.dtype) -> list:
    """加载一个或多个 ControlNet 模型。"""
    from diffusers import ControlNetModel

    controlnets = []
    for path in paths:
        logger.info(f"加载 ControlNet: {path}")
        cn = ControlNetModel.from_pretrained(path, torch_dtype=dtype)
        controlnets.append(cn)
    return controlnets


def swap_scheduler(pipeline, scheduler_name: str) -> None:
    """将 pipeline 的调度器替换为推理用调度器。"""
    from diffusers import (
        DDPMScheduler, DPMSolverMultistepScheduler, DPMSolverSDEScheduler,
        EulerDiscreteScheduler, FlowMatchEulerDiscreteScheduler,
    )

    cfg = pipeline.scheduler.config
    if scheduler_name == "flow_match_euler":
        from models.model_loader import patch_fm_scheduler_for_pipeline
        fm_scheduler = FlowMatchEulerDiscreteScheduler.from_config(cfg)
        patch_fm_scheduler_for_pipeline(fm_scheduler)
        pipeline.scheduler = fm_scheduler
        logger.info("调度器: FlowMatchEulerDiscreteScheduler (Flow Matching Euler ODE)")
    elif scheduler_name == "euler":
        pipeline.scheduler = EulerDiscreteScheduler.from_config(cfg)
        logger.info("调度器: EulerDiscreteScheduler")
    elif scheduler_name == "dpm":
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(cfg)
        logger.info("调度器: DPMSolverMultistepScheduler (dpmsolver++)")
    elif scheduler_name == "dpm++_sde_karras":
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            cfg, algorithm_type="sde-dpmsolver++", use_karras_sigmas=True,
        )
        logger.info("调度器: DPMSolverMultistepScheduler (sde-dpmsolver++, Karras sigmas)")
    elif scheduler_name == "dpm_sde_karras":
        pipeline.scheduler = DPMSolverSDEScheduler.from_config(
            cfg, use_karras_sigmas=True
        )
        logger.info("调度器: DPMSolverSDEScheduler (Karras sigmas)")
    else:
        logger.info("调度器: DDPMScheduler（训练调度器，推理较慢）")


# ── 推理主循环 ──────────────────────────────────────────────────────────────

def run_inference(
    pipeline,
    args: argparse.Namespace,
    width: int,
    height: int,
    control_images: list[Image.Image],
    output_dir: Path,
    device: torch.device,
) -> list[Path]:
    """推理主循环：遍历所有 prompt，生成图像并保存。

    当使用单个 ControlNet 且条件图数量与 prompt 数量相同时，自动进入
    「逐 prompt 条件图模式」：每个 prompt 使用对应索引的条件图，并生成
    三栏对比图（控制图 ｜ 叠加 ｜ 生成图）。

    Returns:
        已保存的图像路径列表
    """
    has_controlnet = len(control_images) > 0
    scales = args.controlnet_scales or [1.0] * len(control_images)
    while len(scales) < len(control_images):
        scales.append(1.0)

    # 逐 prompt 条件图模式：1 个 ControlNet + N 张条件图对应 N 个 prompt
    per_prompt_mode = (
        has_controlnet
        and len(getattr(args, "controlnet_paths", [])) == 1
        and len(control_images) == len(args.prompt)
        and len(control_images) > 1
    )
    if per_prompt_mode:
        logger.info("检测到逐 prompt 条件图模式：每个 prompt 使用对应的条件图，并保存对比图")

    saved_paths: list[Path] = []

    for prompt_idx, prompt in enumerate(args.prompt):
        tag = _prompt_tag(prompt)
        logger.info(f"[Prompt {prompt_idx + 1}/{len(args.prompt)}] {prompt!r}")

        seed = args.seed if args.seed >= 0 else torch.randint(0, 2**31, (1,)).item()
        generator = torch.Generator(device=device).manual_seed(seed)

        call_kwargs = dict(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            width=width,
            height=height,
            num_images_per_prompt=args.num_images,
            generator=generator,
        )

        cond_for_compare: "Image.Image | None" = None
        if has_controlnet:
            if per_prompt_mode:
                cond_for_compare = control_images[prompt_idx]
                call_kwargs["image"] = cond_for_compare
                call_kwargs["controlnet_conditioning_scale"] = scales[0]
            elif len(control_images) == 1:
                cond_for_compare = control_images[0]
                call_kwargs["image"] = cond_for_compare
                call_kwargs["controlnet_conditioning_scale"] = scales[0]
            else:
                # 多 ControlNet 分支
                call_kwargs["image"] = control_images
                call_kwargs["controlnet_conditioning_scale"] = scales

        with torch.inference_mode():
            output = pipeline(**call_kwargs)

        for img_idx, img in enumerate(output.images):
            stem = f"p{prompt_idx:02d}_{tag}_s{seed}_i{img_idx:02d}"
            save_path = output_dir / f"{stem}.png"
            pnginfo = _build_png_metadata(prompt, args.negative_prompt, seed, args)
            img.save(save_path, pnginfo=pnginfo)
            saved_paths.append(save_path)
            logger.info(f"  保存: {save_path}")

            # 生成三栏对比图（有单张对应条件图时）
            if cond_for_compare is not None:
                cmp_path = output_dir / f"{stem}_cmp.png"
                cmp_img = _make_comparison_grid(cond_for_compare, img)
                cmp_img.save(cmp_path)
                logger.info(f"  对比图: {cmp_path}")

    return saved_paths


def run_img2img(
    pipeline,
    args: argparse.Namespace,
    width: int,
    height: int,
    input_images: list[Image.Image],
    output_dir: Path,
    device: torch.device,
) -> list[Path]:
    """img2img 推理：对参考图加噪后去噪重建。

    支持两种范式：
      - DDPM (SD/SDXL): DDPMScheduler.add_noise → DPMSolver 推理
      - Flow Matching (PixArt-Sigma): 线性插值加噪 → FlowMatchEuler 推理

    配对逻辑:
      - N 张图 × N 个 prompt → 一一配对
      - 1 张图 × N 个 prompt → 同一图配所有 prompt
      - N 张图 × 1 个 prompt → 同一 prompt 配所有图

    Returns:
        已保存的图像路径列表
    """
    from diffusers import FlowMatchEulerDiscreteScheduler

    strength = args.strength
    vae = pipeline.vae

    saved_scheduler = pipeline.scheduler
    is_flow_matching = isinstance(pipeline.scheduler, FlowMatchEulerDiscreteScheduler)
    inference_scheduler = pipeline.scheduler

    if is_flow_matching:
        # Flow Matching: 使用线性插值加噪
        inference_scheduler.set_timesteps(args.num_inference_steps, device=device)
        all_timesteps = inference_scheduler.timesteps
        start_idx = max(int(args.num_inference_steps * (1.0 - strength)), 1)
        start_idx = min(start_idx, len(all_timesteps) - 1)
        truncated_timesteps = all_timesteps[start_idx:].tolist()
        t_start_continuous = 1.0 - strength

        logger.info(
            f"img2img (Flow Matching): 去噪步数 {len(truncated_timesteps)}/{args.num_inference_steps}, "
            f"t_start={t_start_continuous:.2f}"
        )
    else:
        # DDPM: 使用 DDPMScheduler.add_noise
        from diffusers import DDPMScheduler, DPMSolverMultistepScheduler
        inference_scheduler = DPMSolverMultistepScheduler.from_config(saved_scheduler.config)
        pipeline.scheduler = inference_scheduler
        logger.info(f"img2img 推理调度器: DPMSolverMultistepScheduler (strength={strength})")

        noise_scheduler = DDPMScheduler.from_config(saved_scheduler.config)
        inference_scheduler.set_timesteps(args.num_inference_steps, device=device)
        all_timesteps = inference_scheduler.timesteps

        start_idx = max(int(args.num_inference_steps * (1.0 - strength)), 1)
        start_idx = min(start_idx, len(all_timesteps) - 1)
        truncated_timesteps = all_timesteps[start_idx:].tolist()
        t_start = all_timesteps[start_idx]

        inference_scheduler.set_timesteps(timesteps=truncated_timesteps, device=device)
        init_sigma = getattr(inference_scheduler, "init_noise_sigma", 1.0)
        if hasattr(init_sigma, "item"):
            init_sigma = float(init_sigma.item())

        logger.info(
            f"img2img: 去噪步数 {len(truncated_timesteps)}/{args.num_inference_steps}, "
            f"起始 t={int(t_start.item())}, init_sigma={init_sigma:.4f}"
        )

    # ── 配对逻辑 ────────────────────────────────────────────────────────────
    n_img = len(input_images)
    n_prompt = len(args.prompt)
    if n_img == n_prompt:
        pairs = list(zip(input_images, args.prompt))
    elif n_img == 1:
        pairs = [(input_images[0], p) for p in args.prompt]
    elif n_prompt == 1:
        pairs = [(img, args.prompt[0]) for img in input_images]
    else:
        raise ValueError(
            f"--input_images ({n_img}) 与 --prompt ({n_prompt}) 数量不匹配；"
            f"需一一对应，或其中一方为 1"
        )

    model_dtype = _dtype_from_str(args.dtype)
    saved_paths: list[Path] = []

    for pair_idx, (src_image, prompt) in enumerate(pairs):
        tag = _prompt_tag(prompt)
        logger.info(f"[img2img {pair_idx + 1}/{len(pairs)}] {prompt!r}")

        z_clean = _encode_image_to_latent(vae, src_image, device)

        for n in range(args.num_images):
            seed = (args.seed + n) if args.seed >= 0 else torch.randint(0, 2**31, (1,)).item()
            generator = torch.Generator(device=device).manual_seed(seed)

            noise = torch.randn(z_clean.shape, generator=generator, device=device, dtype=z_clean.dtype)

            if is_flow_matching:
                z_noisy = t_start_continuous * z_clean + (1.0 - t_start_continuous) * noise
                z_input = z_noisy.to(dtype=model_dtype)
            else:
                z_noisy = noise_scheduler.add_noise(z_clean, noise, t_start.unsqueeze(0))
                z_input = (z_noisy / init_sigma if init_sigma != 1.0 else z_noisy).to(dtype=model_dtype)

            generator = torch.Generator(device=device).manual_seed(seed)
            call_kwargs = dict(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                guidance_scale=args.guidance_scale,
                width=width,
                height=height,
                num_images_per_prompt=1,
                generator=generator,
                latents=z_input,
                timesteps=truncated_timesteps,
            )

            with torch.inference_mode():
                output = pipeline(**call_kwargs)

            img = output.images[0]
            stem = f"img2img_p{pair_idx:02d}_{tag}_s{seed}_n{n:02d}_str{strength:.2f}"
            save_path = output_dir / f"{stem}.png"
            pnginfo = _build_png_metadata(prompt, args.negative_prompt, seed, args)
            img.save(save_path, pnginfo=pnginfo)
            saved_paths.append(save_path)
            logger.info(f"  保存: {save_path}")

            cmp_path = output_dir / f"{stem}_cmp.png"
            cmp_img = _make_comparison_grid(src_image, img)
            cmp_img.save(cmp_path)
            logger.info(f"  对比图: {cmp_path}")

    pipeline.scheduler = saved_scheduler
    return saved_paths


def _build_png_metadata(
    prompt: str,
    negative_prompt: str,
    seed: int,
    args: argparse.Namespace,
):
    """构建 PNG tEXt metadata，方便后续查看生成参数。"""
    from PIL import PngImagePlugin

    info = PngImagePlugin.PngInfo()
    info.add_text("prompt", prompt)
    info.add_text("negative_prompt", negative_prompt)
    info.add_text("seed", str(seed))
    info.add_text("steps", str(args.num_inference_steps))
    info.add_text("guidance_scale", str(args.guidance_scale))
    info.add_text("scheduler", args.scheduler)
    info.add_text("model_type", args.model_type)
    if args.lora_checkpoint:
        info.add_text("lora_checkpoint", args.lora_checkpoint)
    if args.merged_model_path:
        info.add_text("merged_model_path", args.merged_model_path)
    if args.controlnet_paths:
        info.add_text("controlnet_paths", "|".join(args.controlnet_paths))
    if getattr(args, "input_images", None):
        info.add_text("mode", "img2img")
        info.add_text("strength", str(args.strength))
    return info


# ── 入口 ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # 分辨率：0 = 按 model_type 自动填充
    resolution = _DEFAULT_RESOLUTION[args.model_type]
    width  = args.width  if args.width  > 0 else resolution
    height = args.height if args.height > 0 else resolution
    logger.info(f"目标分辨率: {width}×{height}")

    # PixArt-Sigma 参数自动适配
    if args.model_type == "pixart_sigma":
        if args.guidance_scale == 7.5:
            args.guidance_scale = 4.5
            logger.info("PixArt-Sigma: guidance_scale 自动调整为 4.5（--guidance_scale 可覆盖）")
        if args.noise_paradigm == "flow_matching":
            if args.scheduler in ("dpm_sde_karras", "dpm++_sde_karras", "dpm", "ddpm"):
                args.scheduler = "flow_match_euler"
                logger.info("PixArt-Sigma (Flow Matching): scheduler 自动调整为 flow_match_euler")
        else:
            if args.scheduler in ("flow_match_euler", "dpm_sde_karras"):
                args.scheduler = "dpm++_sde_karras"
                logger.info("PixArt-Sigma (DDPM): scheduler 自动调整为 dpm++_sde_karras (sde-dpmsolver++, karras)")

    is_img2img = bool(args.input_images)

    # 校验 img2img 参数
    if is_img2img:
        if args.controlnet_paths:
            raise ValueError("img2img 模式与 ControlNet 不能同时使用")
        if not (0.0 < args.strength <= 1.0):
            raise ValueError(f"--strength 需在 (0, 1] 范围内，当前值: {args.strength}")

    # 校验 ControlNet 参数
    n_cn = len(args.controlnet_paths)
    n_ci = len(args.control_images)
    if args.controlnet_paths and not args.control_images:
        raise ValueError("指定了 --controlnet_paths 但未提供 --control_images")
    if args.control_images and n_cn != n_ci:
        if not (n_cn == 1 and n_ci > 1):
            raise ValueError(
                f"--controlnet_paths ({n_cn}) 与 --control_images ({n_ci}) 数量不一致；"
                f"仅当使用单个 ControlNet 时可提供多张条件图（逐 prompt 模式）"
            )

    # GPU 选择：--gpu 优先于 --device
    if args.gpu >= 0:
        if torch.cuda.is_available():
            device_str = f"cuda:{args.gpu}"
            logger.info(f"指定 GPU: cuda:{args.gpu}")
        else:
            logger.warning("CUDA 不可用，忽略 --gpu，降级为 CPU")
            device_str = "cpu"
    elif not torch.cuda.is_available():
        logger.warning("CUDA 不可用，降级为 CPU")
        device_str = "cpu"
    else:
        device_str = args.device
    device = torch.device(device_str)
    dtype = _dtype_from_str(args.dtype)
    logger.info(f"设备: {device}，模型精度: {args.dtype}")

    # 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载 ControlNet 模型
    controlnets = load_controlnets(args.controlnet_paths, dtype) if args.controlnet_paths else []

    # 加载主 pipeline
    logger.info(f"加载模型（model_type={args.model_type}）...")
    pipeline = load_pipeline(args, dtype, controlnets)
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(desc="推理", leave=False)

    # 替换推理调度器（img2img 模式下 run_img2img 内部会再切换为 DPMSolverMultistep）
    swap_scheduler(pipeline, args.scheduler)

    # 保存 metadata
    _save_metadata(output_dir, args, {"width": width, "height": height})

    if is_img2img:
        # ── img2img 模式 ─────────────────────────────────────────────────
        input_images: list[Image.Image] = []
        for img_path in args.input_images:
            img = _load_control_image(img_path, width, height)
            input_images.append(img)
            logger.info(f"参考图像: {img_path} → resize to {width}×{height}")

        logger.info(
            f"开始 img2img: {len(input_images)} 张参考图 × "
            f"{len(args.prompt)} 个 prompt × {args.num_images} 张/组, "
            f"strength={args.strength}"
        )
        saved = run_img2img(pipeline, args, width, height, input_images, output_dir, device)
    else:
        # ── t2i / ControlNet 模式 ────────────────────────────────────────
        control_images: list[Image.Image] = []
        for img_path in args.control_images:
            ci = _load_control_image(img_path, width, height)
            control_images.append(ci)
            logger.info(f"条件图像: {img_path} → resize to {width}×{height}")

        logger.info(f"开始推理: {len(args.prompt)} 个 prompt × {args.num_images} 张/prompt")
        saved = run_inference(pipeline, args, width, height, control_images, output_dir, device)

    logger.info(f"完成！共生成 {len(saved)} 张图像，保存至: {output_dir}")


if __name__ == "__main__":
    main()
