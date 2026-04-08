"""补跑验证 — 对已有检查点加载 EMA 权重并生成验证图像，回写 TensorBoard。

用法:
    python scripts/retroactive_validation.py \
        --config configs/controlnet_xs_pixart_sigma_dual_lr.yaml \
        --steps 500 1000
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.controlnet_xs_pixart import (
    PixArtControlNetXSAdapter,
    PixArtControlNetXSTransformerModel,
)
from models.model_loader import load_pixart_sigma_components
from utils.ema import EMAModel
from utils.logger import TensorBoardLogger
from utils.validation import ValidationLoop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def load_conditioning_images(config, n: int, seed: int = 42):
    """从条件图目录随机采样 n 张图并 resize 到目标分辨率。"""
    cond_dir = Path(config.data.get("conditioning_data_dir", ""))
    resolution = config.data.get("resolution", 1024)
    if not cond_dir.exists():
        logger.warning(f"条件图目录不存在: {cond_dir}")
        return None, None

    files = sorted(f for f in cond_dir.iterdir() if f.is_file() and f.suffix.lower() in _IMG_EXTS)
    if not files:
        return None, None

    rng = random.Random(seed)
    selected = rng.sample(files, min(n, len(files)))

    cond_images = [Image.open(f).convert("RGB").resize((resolution, resolution)) for f in selected]

    train_dir = Path(config.data.get("train_data_dir", ""))
    gt_images = None
    if train_dir.exists():
        from data.controlnet_dataset import _KNOWN_SUFFIX_PAIRS, _strip_known_suffix, _build_cond_index

        train_index = {}
        for f in train_dir.iterdir():
            if not f.is_file() or f.suffix.lower() not in _IMG_EXTS:
                continue
            bk = None
            for orig_suffix, _ in _KNOWN_SUFFIX_PAIRS:
                bk = _strip_known_suffix(f.name, orig_suffix)
                if bk is not None:
                    break
            if bk is None:
                bk = f.stem
            train_index[bk] = f

        gt_list = []
        for cf in selected:
            bk = None
            for _, cond_suffix in _KNOWN_SUFFIX_PAIRS:
                bk = _strip_known_suffix(cf.name, cond_suffix)
                if bk is not None:
                    break
            if bk is None:
                bk = cf.stem
            gt_path = train_index.get(bk)
            if gt_path is not None:
                gt_list.append(Image.open(gt_path).convert("RGB").resize((resolution, resolution)))
            else:
                gt_list.append(None)

        if any(img is not None for img in gt_list):
            placeholder = Image.new("RGB", (resolution, resolution), (128, 128, 128))
            gt_images = [img if img is not None else placeholder for img in gt_list]

    return cond_images, gt_images


def run_validation_for_step(config, ckpt_dir: Path, step: int, device: torch.device):
    """加载单个检查点并运行验证。"""
    logger.info(f"=== 补跑验证: step {step} ({ckpt_dir}) ===")

    model_path = config.model.pretrained_model_name_or_path
    weights_dir = config.model.get("weights_dir", None)
    components = load_pixart_sigma_components(model_path, weights_dir=weights_dir, flow_matching=False)
    vae = components["vae"]
    transformer = components["transformer"]
    tokenizer = components["tokenizer"]
    text_encoder = components["text_encoder"]
    noise_scheduler = components["noise_scheduler"]

    ft_path = config.model.get("finetuned_transformer_path", None)
    if ft_path and os.path.isdir(ft_path):
        from diffusers import PixArtTransformer2DModel
        transformer = PixArtTransformer2DModel.from_pretrained(ft_path, torch_dtype=transformer.dtype)
        logger.info(f"Fine-tuned transformer loaded from: {ft_path}")

    # ControlNet-XS
    xs_cfg = config.get("controlnet_xs", {})
    controlnet = PixArtControlNetXSAdapter.from_transformer(
        transformer,
        num_layers=int(xs_cfg.get("num_layers", 14)),
        size_ratio=float(xs_cfg.get("size_ratio", 0.25)),
        conditioning_mode=str(config.controlnet.get("conditioning_mode", "cnn_encoder")),
        connection_interval=int(xs_cfg.get("connection_interval", 2)),
    )

    # 加载 checkpoint 权重
    tf_dir = ckpt_dir / "transformer"
    cn_dir = ckpt_dir / "controlnet"
    if tf_dir.exists():
        from diffusers import PixArtTransformer2DModel
        loaded_tf = PixArtTransformer2DModel.from_pretrained(str(tf_dir))
        transformer.load_state_dict(loaded_tf.state_dict())
        del loaded_tf
        logger.info(f"Transformer loaded from {tf_dir}")
    if cn_dir.exists():
        loaded_cn = PixArtControlNetXSAdapter.from_pretrained(str(cn_dir))
        controlnet.load_state_dict(loaded_cn.state_dict())
        del loaded_cn
        logger.info(f"ControlNet loaded from {cn_dir}")

    # 加载 EMA 权重并 swap
    ema_tf_path = ckpt_dir / "ema_transformer.pt"
    ema_cn_path = ckpt_dir / "ema_controlnet.pt"
    if ema_tf_path.exists():
        ema_tf = EMAModel(transformer.parameters())
        ema_tf.load_state_dict(torch.load(ema_tf_path, map_location="cpu", weights_only=True))
        ema_tf.copy_to(transformer.parameters())
        del ema_tf
        logger.info("EMA transformer weights applied")
    if ema_cn_path.exists():
        ema_cn = EMAModel(controlnet.parameters())
        ema_cn.load_state_dict(torch.load(ema_cn_path, map_location="cpu", weights_only=True))
        ema_cn.copy_to(controlnet.parameters())
        del ema_cn
        logger.info("EMA controlnet weights applied")

    joint_model = PixArtControlNetXSTransformerModel(
        transformer=transformer,
        controlnet=controlnet,
    )

    vae.to(device, dtype=torch.float32)
    joint_model.to(device, dtype=torch.bfloat16)
    joint_model.eval()

    # 编码文本
    text_encoder.to(device)
    val_cfg = config.get("validation", {})
    prompts = list(val_cfg.get("prompts", ["architectural floor plan, blueprint, technical drawing"]))
    num_val_samples = val_cfg.get("num_val_samples", len(prompts))
    if len(prompts) < num_val_samples:
        prompts = (prompts * ((num_val_samples // len(prompts)) + 1))[:num_val_samples]

    t5_max_len = config.data.get("t5_max_length", 300)

    prompt_embeds_list = []
    prompt_masks_list = []
    for p in prompts:
        tok = tokenizer(p, padding="max_length", truncation=True, max_length=t5_max_len, return_tensors="pt")
        ids = tok.input_ids.to(device)
        mask = tok.attention_mask.to(device)
        with torch.no_grad():
            embeds = text_encoder(ids, attention_mask=mask)[0]
        prompt_embeds_list.append(embeds)
        prompt_masks_list.append(mask)

    neg_tok = tokenizer("", padding="max_length", truncation=True, max_length=t5_max_len, return_tensors="pt")
    neg_ids = neg_tok.input_ids.to(device)
    neg_mask = neg_tok.attention_mask.to(device)
    with torch.no_grad():
        neg_embeds = text_encoder(neg_ids, attention_mask=neg_mask)[0]

    text_encoder.to("cpu")
    torch.cuda.empty_cache()

    pipeline_kwargs_override = [
        {
            "prompt_embeds": prompt_embeds_list[i],
            "prompt_attention_mask": prompt_masks_list[i],
            "negative_prompt_embeds": neg_embeds,
            "negative_prompt_attention_mask": neg_mask,
        }
        for i in range(len(prompts))
    ]

    # 构建 pipeline
    from diffusers import DPMSolverSDEScheduler
    from pipelines.pixart_controlnet_pipeline import PixArtControlNetPipeline

    inference_scheduler = DPMSolverSDEScheduler.from_config(
        noise_scheduler.config, use_karras_sigmas=True,
    )
    pipeline = PixArtControlNetPipeline(
        vae=vae,
        transformer=joint_model,
        controlnet=None,
        text_encoder=None,
        tokenizer=None,
        scheduler=inference_scheduler,
    )
    pipeline.set_progress_bar_config(disable=True)

    # 条件图
    cond_images, gt_images = load_conditioning_images(config, num_val_samples, seed=val_cfg.get("seed", 42))

    # TensorBoard logger
    output_dir = config.training.get("output_dir", "./outputs")
    tb_log_dir = os.path.join(output_dir, "tensorboard")
    tb_logger = TensorBoardLogger(log_dir=tb_log_dir, is_main_process=True)

    # ValidationLoop
    val_loop = ValidationLoop(
        prompts=prompts,
        negative_prompt=val_cfg.get("negative_prompt", ""),
        num_inference_steps=val_cfg.get("num_inference_steps", 20),
        guidance_scale=val_cfg.get("guidance_scale", 4.5),
        seed=val_cfg.get("seed", 42),
        num_images_per_prompt=val_cfg.get("num_images_per_prompt", 1),
        save_dir=os.path.join(output_dir, "samples"),
    )

    val_loop.run(
        pipeline,
        step,
        tb_logger,
        device=device,
        conditioning_images=cond_images,
        pipeline_kwargs_override=pipeline_kwargs_override,
        ground_truth_images=gt_images,
    )

    tb_logger.close()
    del pipeline, joint_model, vae, transformer, controlnet
    torch.cuda.empty_cache()
    logger.info(f"=== step {step} 验证完成 ===\n")


def main():
    parser = argparse.ArgumentParser(description="补跑验证：加载已有检查点并生成验证图像")
    parser.add_argument("--config", type=str, required=True, help="训练配置文件路径")
    parser.add_argument("--steps", type=int, nargs="+", required=True, help="要补跑的步数列表")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    device = torch.device(args.device)
    ckpt_base = Path(config.training.get("output_dir", "./outputs")) / "checkpoints"

    for step in args.steps:
        ckpt_dir = ckpt_base / f"step_{step:06d}"
        if not ckpt_dir.exists():
            logger.warning(f"检查点不存在，跳过: {ckpt_dir}")
            continue
        run_validation_for_step(config, ckpt_dir, step, device)

    logger.info("所有补跑验证完成!")


if __name__ == "__main__":
    main()
