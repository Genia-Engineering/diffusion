"""ControlNet 训练器 — 支持 SD1.5/SDXL 的条件控制网络训练。

权重冻结策略:
  模式 B (纯 ControlNet):
    冻结 → VAE, UNet 全部, TextEncoder(s)
    可训练 → ControlNet (encoder copy + zero conv)

  模式 C (ControlNet + LoRA 联合训练):
    冻结 → VAE, UNet base 权重, TextEncoder(s) base 权重
    可训练 → ControlNet 全部 + UNet/TE LoRA 参数

ControlNet 灵活冻结（可组合，互不冲突）:
  freeze_cross_attention: 冻结 ControlNet 内所有交叉注意力模块
  freeze_down_blocks:     冻结指定下采样块（0-3 中任选）
  freeze_mid_block:       冻结瓶颈层
"""

import gc
import logging
import math
import os

import torch
import torch.nn.functional as F
from diffusers import UNet2DConditionModel
from omegaconf import DictConfig, ListConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.buckets import BucketManager, BucketSampler
from data.controlnet_dataset import CachedLatentControlNetDataset, ControlNetDataset
from models.controlnet import create_controlnet_sd15, create_controlnet_sdxl
from models.lora import LoRAInjector, get_lora_params
from models.model_loader import load_sd15_components, load_sdxl_components, resolve_model_path
from utils.memory import apply_memory_optimizations, compute_grad_norm
from utils.validation import ValidationLoop
from .base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


def _safe_collate(batch):
    """PyTorch 2.10+ 兼容 collate: 避免 shared storage 不可 resize 问题。"""
    elem = batch[0]
    if isinstance(elem, dict):
        return {key: _safe_collate([d[key] for d in batch]) for key in elem}
    if isinstance(elem, torch.Tensor):
        return torch.stack([x.clone().contiguous() for x in batch], dim=0)
    from torch.utils.data._utils.collate import default_collate
    return default_collate(batch)


class ControlNetTrainer(BaseTrainer):
    """ControlNet 训练器。"""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.model_type = config.model.model_type
        self.cn_cfg = config.controlnet
        self.joint_lora = "lora" in config and config.get("lora", {}).get("rank", 0) > 0

        self.noise_offset: float = float(self.training_cfg.get("noise_offset", 0.0))
        self.min_snr_gamma: float = float(self.training_cfg.get("min_snr_gamma", 0.0))
        self.conditioning_scale: float = float(self.cn_cfg.get("conditioning_scale", 1.0))

        self.semantic_loss_weight: float = float(self.cn_cfg.get("semantic_loss_weight", 1.0))
        self.semantic_bg_threshold: float = float(self.cn_cfg.get("semantic_bg_threshold", 0.1))
        self._last_fg_ratio: float | None = None

        self._load_models()
        self._create_controlnet()

        if self.joint_lora:
            self._inject_lora()

        self._freeze_parameters()

        self._snr_cache: torch.Tensor | None = None
        if self.min_snr_gamma > 0:
            alphas_cumprod = self.noise_scheduler.alphas_cumprod
            self._snr_cache = alphas_cumprod / (1.0 - alphas_cumprod)

        apply_memory_optimizations(
            unet=self.unet,
            vae=self.vae,
            text_encoder=self.text_encoder,
            text_encoder_2=getattr(self, "text_encoder_2", None),
            controlnet=self.controlnet,
            enable_gradient_checkpointing=self.training_cfg.get("gradient_checkpointing", True),
            attention_backend=self.training_cfg.get("attention_backend", "sdpa"),
            enable_channels_last=self.training_cfg.get("enable_channels_last", True),
        )

    # ── 模型加载 ─────────────────────────────────────────────────────────

    def _load_models(self):
        model_path = self.config.model.pretrained_model_name_or_path
        weights_dir = self.config.model.get("weights_dir", None)
        merged_unet_path = self.config.model.get("merged_unet_path", None)

        # 根据 mixed_precision 配置选择加载 dtype，避免 float32 撑爆显存
        mixed_precision = self.training_cfg.get("mixed_precision", "no")
        if mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        elif mixed_precision == "fp16":
            weight_dtype = torch.float16
        else:
            weight_dtype = torch.float32

        # merged_unet_path 存在时跳过默认 UNet 加载，避免同时驻留多份 UNet (~5GB each)
        has_merged = merged_unet_path and os.path.isdir(merged_unet_path)

        if self.model_type == "sdxl":
            components = load_sdxl_components(
                model_path, weights_dir=weights_dir, dtype=weight_dtype,
                skip_unet=has_merged,
            )
            self.text_encoder_2 = components["text_encoder_2"]
            self.tokenizer_2 = components["tokenizer_2"]
        else:
            components = load_sd15_components(
                model_path, weights_dir=weights_dir, dtype=weight_dtype,
                skip_unet=has_merged,
            )
            self.text_encoder_2 = None
            self.tokenizer_2 = None

        self.vae = components["vae"]
        self.text_encoder = components["text_encoder"]
        self.tokenizer = components["tokenizer"]
        self.noise_scheduler = components["noise_scheduler"]

        if has_merged:
            logger.info(f"Loading merged UNet from {merged_unet_path}")
            self.unet = UNet2DConditionModel.from_pretrained(
                merged_unet_path, subfolder="unet", torch_dtype=weight_dtype,
            )
            self._merged_unet_for_cn = UNet2DConditionModel.from_pretrained(
                merged_unet_path, subfolder="unet", torch_dtype=weight_dtype,
            )
        else:
            self.unet = components["unet"]
            self._merged_unet_for_cn = None

    def _create_controlnet(self):
        conditioning_channels = self.cn_cfg.get("conditioning_channels", 3)
        model_path = self.config.model.pretrained_model_name_or_path
        weights_dir = self.config.model.get("weights_dir", None)
        conv_zero_init = str(self.cn_cfg.get("conv_zero_init", "zero"))
        conv_init_std = float(self.cn_cfg.get("conv_init_std", 0.02))

        source_unet = self._merged_unet_for_cn if self._merged_unet_for_cn is not None else None

        mixed_precision = self.training_cfg.get("mixed_precision", "no")
        if mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        elif mixed_precision == "fp16":
            weight_dtype = torch.float16
        else:
            weight_dtype = torch.float32

        if self.model_type == "sdxl":
            resolved = resolve_model_path(model_path, weights_dir, "sdxl")
            self.controlnet = create_controlnet_sdxl(
                resolved, conditioning_channels, unet=source_unet, dtype=weight_dtype,
                conv_zero_init=conv_zero_init, conv_init_std=conv_init_std,
            )
        else:
            resolved = resolve_model_path(model_path, weights_dir, "sd15")
            self.controlnet = create_controlnet_sd15(
                resolved, conditioning_channels, unet=source_unet, dtype=weight_dtype,
                conv_zero_init=conv_zero_init, conv_init_std=conv_init_std,
            )

        if self._merged_unet_for_cn is not None:
            del self._merged_unet_for_cn
            self._merged_unet_for_cn = None
            logger.info("ControlNet initialized from merged LoRA UNet")
        else:
            logger.info(f"ControlNet initialized from {model_path}")

    def _inject_lora(self):
        """模式 C: 联合训练时在 UNet 上注入 LoRA。"""
        lora_cfg = self.config.lora
        rank = lora_cfg.get("rank", 16)
        alpha = lora_cfg.get("alpha", 16.0)
        target_modules = list(lora_cfg.get("target_modules", LoRAInjector.DEFAULT_TARGET_MODULES))

        injected = LoRAInjector.inject_unet(self.unet, rank, alpha, target_modules)
        logger.info(f"[Joint mode] LoRA injected into UNet: {len(injected)} layers")

    # ── 权重冻结 ─────────────────────────────────────────────────────────

    def _freeze_parameters(self):
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        if self.text_encoder_2 is not None:
            self.text_encoder_2.requires_grad_(False)

        if self.joint_lora:
            for param in self.unet.parameters():
                param.requires_grad = False
            for module in self.unet.modules():
                from models.lora import LoRALinear
                if isinstance(module, LoRALinear):
                    module.lora_A.requires_grad_(True)
                    module.lora_B.requires_grad_(True)
        else:
            self.unet.requires_grad_(False)

        self.controlnet.requires_grad_(True)
        self._apply_controlnet_freeze()

        self.print_trainable_params(self.unet, self.controlnet, self.text_encoder)

    def _apply_controlnet_freeze(self):
        """根据配置灵活冻结 ControlNet 的指定部分。

        配置项（均可组合，互不冲突）:
          freeze_cross_attention: bool  — 冻结所有交叉注意力
          freeze_down_blocks: list[int] — 冻结指定下采样块索引 (0-3)
          freeze_mid_block: bool        — 冻结瓶颈层
        """
        freeze_cross_attn = self.cn_cfg.get("freeze_cross_attention", False)
        freeze_down_blocks = self.cn_cfg.get("freeze_down_blocks", [])
        freeze_mid = self.cn_cfg.get("freeze_mid_block", False)

        if isinstance(freeze_down_blocks, (list, ListConfig)):
            freeze_down_blocks = [int(i) for i in freeze_down_blocks]
        else:
            freeze_down_blocks = []

        frozen_parts = []

        if freeze_cross_attn:
            self._freeze_cross_attention_modules(self.controlnet)
            frozen_parts.append("cross_attention")

        if freeze_down_blocks:
            num_blocks = len(self.controlnet.down_blocks)
            for idx in freeze_down_blocks:
                if 0 <= idx < num_blocks:
                    self.controlnet.down_blocks[idx].requires_grad_(False)
                    frozen_parts.append(f"down_block[{idx}]")
                else:
                    logger.warning(f"freeze_down_blocks: index {idx} out of range [0, {num_blocks - 1}]")

        if freeze_mid:
            if hasattr(self.controlnet, "mid_block") and self.controlnet.mid_block is not None:
                self.controlnet.mid_block.requires_grad_(False)
                frozen_parts.append("mid_block")

        if frozen_parts:
            logger.info(f"ControlNet frozen parts: {', '.join(frozen_parts)}")
        else:
            logger.info("ControlNet: all parameters trainable (no freeze)")

    @staticmethod
    def _freeze_cross_attention_modules(model):
        """冻结模型中所有 Transformer block 的交叉注意力参数。"""
        from diffusers.models.attention import BasicTransformerBlock
        for module in model.modules():
            if isinstance(module, BasicTransformerBlock):
                if hasattr(module, "attn2") and module.attn2 is not None:
                    module.attn2.requires_grad_(False)
                if hasattr(module, "norm2") and module.norm2 is not None:
                    module.norm2.requires_grad_(False)

    # ── 数据加载 ─────────────────────────────────────────────────────────

    def _log_bucket_stats(self, bucket_to_indices: dict) -> None:
        total = sum(len(v) for v in bucket_to_indices.values())
        lines = [f"Aspect ratio bucket distribution (total={total}):"]
        for (w, h), indices in sorted(bucket_to_indices.items()):
            lines.append(f"  {w}×{h}: {len(indices)} images ({len(indices)/total*100:.1f}%)")
        logger.info("\n".join(lines))

    def _build_dataloader(self) -> DataLoader:
        data_cfg = self.config.data
        resolution = data_cfg.get("resolution", 512 if self.model_type == "sd15" else 1024)
        batch_size = self.training_cfg.get("train_batch_size", 4)
        fixed_caption = data_cfg.get("caption", "")
        cache_latents = self.training_cfg.get("cache_latents", False)
        max_train_samples = data_cfg.get("max_train_samples", None)

        if cache_latents:
            dataset = CachedLatentControlNetDataset(
                data_dir=data_cfg.train_data_dir,
                cache_dir=self._get_latent_cache_dir(),
                conditioning_data_dir=data_cfg.get("conditioning_data_dir", None),
                tokenizer=self.tokenizer,
                tokenizer_2=self.tokenizer_2 if self.model_type == "sdxl" else None,
                conditioning_type=self.cn_cfg.get("conditioning_type", "canny"),
                center_crop=data_cfg.get("center_crop", False),
                random_flip=data_cfg.get("random_flip", True),
                fixed_caption=fixed_caption,
                max_train_samples=max_train_samples,
            )
        else:
            dataset = ControlNetDataset(
                data_dir=data_cfg.train_data_dir,
                conditioning_data_dir=data_cfg.get("conditioning_data_dir", None),
                tokenizer=self.tokenizer,
                tokenizer_2=self.tokenizer_2 if self.model_type == "sdxl" else None,
                conditioning_type=self.cn_cfg.get("conditioning_type", "canny"),
                resolution=resolution,
                center_crop=data_cfg.get("center_crop", False),
                random_flip=data_cfg.get("random_flip", True),
                fixed_caption=fixed_caption,
                max_train_samples=max_train_samples,
            )

        if cache_latents or data_cfg.get("use_aspect_ratio_bucketing", True):
            bucket_manager = BucketManager(model_type=self.model_type)
            image_sizes = dataset.get_image_sizes()
            bucket_to_indices = bucket_manager.assign_buckets(image_sizes)
            dataset.set_bucket_assignments(bucket_to_indices)
            self._log_bucket_stats(bucket_to_indices)
            sampler = BucketSampler(bucket_to_indices, batch_size, drop_last=True, shuffle=True)
            dataloader = DataLoader(
                dataset, batch_size=batch_size, sampler=sampler,
                num_workers=self.training_cfg.get("dataloader_num_workers", 4),
                pin_memory=True, drop_last=True,
                collate_fn=_safe_collate,
            )
        else:
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True,
                num_workers=self.training_cfg.get("dataloader_num_workers", 4),
                pin_memory=True, drop_last=True,
                collate_fn=_safe_collate,
            )

        return dataloader

    # ── 编码辅助 ─────────────────────────────────────────────────────────

    def _encode_prompt_sd15(self, input_ids):
        with torch.no_grad():
            return self.text_encoder(input_ids, return_dict=False)[0]

    def _encode_prompt_sdxl(self, input_ids_1, input_ids_2):
        with torch.no_grad():
            encoder_output_1 = self.text_encoder(input_ids_1, output_hidden_states=True)
            hidden_states_1 = encoder_output_1.hidden_states[-2]

            encoder_output_2 = self.text_encoder_2(input_ids_2, output_hidden_states=True)
            hidden_states_2 = encoder_output_2.hidden_states[-2]
            pooled_prompt_embeds = encoder_output_2[0]

        prompt_embeds = torch.cat([hidden_states_1, hidden_states_2], dim=-1)
        return prompt_embeds, pooled_prompt_embeds

    def _compute_sdxl_time_ids(self, batch, bsz, device, dtype):
        """计算 SDXL add_time_ids (B, 6): [orig_h, orig_w, crop_top, crop_left, tgt_h, tgt_w]。
        bsz 由调用方显式传入，兼容 pixel_values 和 cached latents 两种 batch 格式。
        """
        bs = bsz
        default_hw = torch.tensor([[1024, 1024]], dtype=torch.long).expand(bs, -1)
        default_crop = torch.zeros(bs, 2, dtype=torch.long)

        original_size = batch.get("original_size", default_hw)
        crop_top_left = batch.get("crop_top_left", default_crop)
        target_size = batch.get("target_size", default_hw)

        add_time_ids = torch.cat([original_size, crop_top_left, target_size], dim=1)
        return add_time_ids.to(device=device, dtype=dtype)

    # ── 训练主循环 ───────────────────────────────────────────────────────

    def train(self):
        """ControlNet 主训练循环。"""
        cache_latents = self.training_cfg.get("cache_latents", False)

        self.vae.to(self.accelerator.device, dtype=torch.float32)

        # 分布式安全预计算：文件信号代替 NCCL barrier，避免 10 分钟超时崩溃
        if cache_latents:
            self._precompute_latents_distributed(self._get_latent_cache_dir())

        dataloader = self._build_dataloader()
        num_train_steps = self.training_cfg.get("num_train_steps", 10000)
        max_grad_norm = self.training_cfg.get("max_grad_norm", 1.0)
        validation_steps = self.training_cfg.get("validation_steps", 500)
        save_steps = self.training_cfg.get("save_steps", 500)

        cn_params = [p for p in self.controlnet.parameters() if p.requires_grad]
        lora_params = get_lora_params(self.unet) if self.joint_lora else []
        all_trainable = cn_params + lora_params

        optimizer = self.setup_optimizer(trainable_params=all_trainable)
        lr_scheduler = self.setup_lr_scheduler(optimizer, num_train_steps)

        self.controlnet, self.unet, optimizer, dataloader = (
            self.accelerator.prepare(
                self.controlnet, self.unet, optimizer, dataloader
            )
        )

        # VAE 已在 train() 开头移到 device，此处只需移动文本编码器
        self.text_encoder.to(self.accelerator.device)
        if self.text_encoder_2 is not None:
            self.text_encoder_2.to(self.accelerator.device)

        if self.training_cfg.get("cache_text_embeddings", False):
            self._precompute_text_embeddings()

        # 恢复训练
        resume_dir = self.training_cfg.get("resume_from_checkpoint", None)
        if resume_dir == "latest":
            resume_dir = self.ckpt_manager.get_latest_checkpoint()
        if resume_dir:
            state = self.ckpt_manager.load(
                resume_dir,
                unet=self.unet if self.joint_lora else None,
                controlnet=self.accelerator.unwrap_model(self.controlnet),
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                is_lora=self.joint_lora,
            )
            self.global_step = state["step"]
            self.global_epoch = state["epoch"]
            if "ema_loss" in state:
                self._ema_loss = state["ema_loss"]

        val_cfg = self.config.get("validation", {})
        val_loop = ValidationLoop(
            prompts=list(val_cfg.get("prompts", ["a test image"])),
            negative_prompt=val_cfg.get("negative_prompt", ""),
            num_inference_steps=val_cfg.get("num_inference_steps", 25),
            guidance_scale=val_cfg.get("guidance_scale", 7.5),
            seed=val_cfg.get("seed", 42),
            save_dir=os.path.join(self.training_cfg.get("output_dir", "./outputs"), "samples"),
            controlnet_conditioning_scale=float(
                val_cfg.get("controlnet_conditioning_scale", self.conditioning_scale)
            ),
        )

        gradient_accumulation_steps = self.training_cfg.get("gradient_accumulation_steps", 1)
        steps_per_epoch = math.ceil(len(dataloader) / gradient_accumulation_steps)
        num_epochs = math.ceil(num_train_steps / max(steps_per_epoch, 1))
        progress_bar = tqdm(
            total=num_train_steps,
            initial=self.global_step,
            desc="Training ControlNet",
            disable=not self.accelerator.is_main_process,
        )

        self.controlnet.train()
        if self.joint_lora:
            self.unet.train()

        for epoch in range(num_epochs):
            self.global_epoch = epoch
            for batch in dataloader:
                if self.global_step >= num_train_steps:
                    break

                with self.accelerator.accumulate(self.controlnet):
                    loss = self._training_step(batch)
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        grad_norm = self.accelerator.clip_grad_norm_(all_trainable, max_grad_norm)
                        grad_norm = grad_norm.item() if hasattr(grad_norm, 'item') else float(grad_norm)

                    optimizer.step()
                    optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    lr_scheduler.step()
                    self.global_step += 1
                    current_lr = lr_scheduler.get_last_lr()[0]
                    self.log_step(loss.item(), current_lr, grad_norm)

                    if self._last_fg_ratio is not None:
                        self.tb_logger.log_scalar(
                            "train/semantic_fg_ratio", self._last_fg_ratio, self.global_step,
                        )

                    progress_bar.update(1)
                    progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}")

                    if self.global_step % validation_steps == 0:
                        self._run_validation(val_loop)
                        self.accelerator.wait_for_everyone()

                    if self.global_step % save_steps == 0:
                        self._save_checkpoint(optimizer, lr_scheduler)
                        self.accelerator.wait_for_everyone()

            if self.global_step >= num_train_steps:
                break

        self._save_checkpoint(optimizer, lr_scheduler)
        self.tb_logger.close()
        self.accelerator.end_training()
        logger.info("ControlNet training complete!")

    # ── 语义区域加权 ─────────────────────────────────────────────────────

    def _build_semantic_weights(
        self,
        conditioning_pixel_values: torch.Tensor,
        latent_hw: tuple[int, int],
    ) -> torch.Tensor | None:
        """从条件图构建空间加权图，语义色块区域获得更高权重。

        Args:
            conditioning_pixel_values: (B, 3, H, W) [0, 1] 条件图像
            latent_hw: latent 空间的 (H, W)

        Returns:
            (B, 1, latent_H, latent_W) 权重图，或 None（功能未启用时）
        """
        if self.semantic_loss_weight <= 1.0:
            self._last_fg_ratio = None
            return None

        with torch.no_grad():
            fg_mask = (
                conditioning_pixel_values.sum(dim=1, keepdim=True) > self.semantic_bg_threshold
            ).float()

            self._last_fg_ratio = fg_mask.mean().item()

            mask_down = F.interpolate(
                fg_mask, size=latent_hw, mode="area",
            )

            weights = 1.0 + mask_down * (self.semantic_loss_weight - 1.0)

        return weights

    # ── 单步训练 ─────────────────────────────────────────────────────────

    def _training_step(self, batch) -> torch.Tensor:
        """单步训练: 加噪 → ControlNet → UNet → loss (Huber + MinSNR + noise_offset)。

        支持两种 batch 格式:
          - 含 "pixel_values"：在线 VAE encode（cache_latents=false）
          - 含 "latents"：使用预缓存 latent，跳过 VAE encode（cache_latents=true）
        """
        conditioning_pixel_values = batch["conditioning_pixel_values"].to(
            dtype=self.accelerator.unwrap_model(self.controlnet).dtype
        )

        if "latents" in batch:
            latents = batch["latents"].to(self.accelerator.device)
        else:
            pixel_values = batch["pixel_values"].to(dtype=self.vae.dtype)
            with torch.no_grad():
                latents = self.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)

        if self.noise_offset > 0:
            noise = noise + self.noise_offset * torch.randn(
                latents.shape[0], latents.shape[1], 1, 1,
                device=latents.device, dtype=latents.dtype,
            )

        bsz = latents.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,),
            device=latents.device,
        ).long()

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        if self.model_type == "sdxl":
            if hasattr(self, "_cached_prompt_embeds"):
                prompt_embeds = self._cached_prompt_embeds.expand(bsz, -1, -1).to(latents.device)
                pooled_prompt_embeds = self._cached_pooled_prompt_embeds.expand(bsz, -1).to(latents.device)
            else:
                prompt_embeds, pooled_prompt_embeds = self._encode_prompt_sdxl(
                    batch["input_ids"], batch["input_ids_2"]
                )
            add_time_ids = self._compute_sdxl_time_ids(batch, bsz, latents.device, prompt_embeds.dtype)
            added_cond_kwargs = {
                "text_embeds": pooled_prompt_embeds,
                "time_ids": add_time_ids,
            }
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                noisy_latents, timesteps, prompt_embeds,
                controlnet_cond=conditioning_pixel_values,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )
            if self.conditioning_scale != 1.0:
                down_block_res_samples = [r * self.conditioning_scale for r in down_block_res_samples]
                mid_block_res_sample = mid_block_res_sample * self.conditioning_scale
            noise_pred = self.unet(
                noisy_latents, timesteps, prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            ).sample
        else:
            if hasattr(self, "_cached_prompt_embeds"):
                encoder_hidden_states = self._cached_prompt_embeds.expand(bsz, -1, -1).to(latents.device)
            else:
                encoder_hidden_states = self._encode_prompt_sd15(batch["input_ids"])

            down_block_res_samples, mid_block_res_sample = self.controlnet(
                noisy_latents, timesteps, encoder_hidden_states,
                controlnet_cond=conditioning_pixel_values,
                return_dict=False,
            )
            if self.conditioning_scale != 1.0:
                down_block_res_samples = [r * self.conditioning_scale for r in down_block_res_samples]
                mid_block_res_sample = mid_block_res_sample * self.conditioning_scale
            noise_pred = self.unet(
                noisy_latents, timesteps, encoder_hidden_states,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            ).sample

        if self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            target = noise

        spatial_weights = self._build_semantic_weights(
            conditioning_pixel_values, latents.shape[2:]
        )

        loss = self.compute_loss(
            noise_pred, target, timesteps,
            snr_cache=self._snr_cache,
            min_snr_gamma=self.min_snr_gamma,
            prediction_type=self.noise_scheduler.config.prediction_type,
            spatial_weights=spatial_weights,
        )
        return loss

    # ── 验证 ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _run_validation(self, val_loop: ValidationLoop):
        if not self.accelerator.is_main_process:
            return

        from diffusers import DPMSolverMultistepScheduler, StableDiffusionControlNetPipeline
        from diffusers import StableDiffusionXLControlNetPipeline

        self.controlnet.eval()
        unwrapped_cn = self.accelerator.unwrap_model(self.controlnet)
        unwrapped_unet = self.accelerator.unwrap_model(self.unet)

        # 文本编码器可能已被 cache_text_embeddings 卸载（为 None）
        unwrapped_te = (
            self.accelerator.unwrap_model(self.text_encoder)
            if self.text_encoder is not None else None
        )

        inference_scheduler = DPMSolverMultistepScheduler.from_config(
            self.noise_scheduler.config
        )

        if self.model_type == "sdxl":
            unwrapped_te2 = (
                self.accelerator.unwrap_model(self.text_encoder_2)
                if self.text_encoder_2 is not None else None
            )
            pipeline = StableDiffusionXLControlNetPipeline(
                vae=self.vae, unet=unwrapped_unet,
                text_encoder=unwrapped_te, text_encoder_2=unwrapped_te2,
                tokenizer=self.tokenizer, tokenizer_2=self.tokenizer_2,
                scheduler=inference_scheduler, controlnet=unwrapped_cn,
            )
        else:
            pipeline = StableDiffusionControlNetPipeline(
                vae=self.vae, unet=unwrapped_unet,
                text_encoder=unwrapped_te, tokenizer=self.tokenizer,
                scheduler=inference_scheduler, controlnet=unwrapped_cn,
                safety_checker=None, feature_extractor=None,
            )

        # 构建 pipeline_kwargs_override：文本编码器已卸载时用预缓存嵌入
        pipeline_kwargs_override = None
        if self.text_encoder is None and hasattr(self, "_cached_prompt_embeds"):
            dev = self.accelerator.device
            pipeline_kwargs_override = {
                "prompt_embeds": self._cached_prompt_embeds.to(dev),
                "negative_prompt_embeds": self._cached_negative_prompt_embeds.to(dev),
            }
            if self.model_type == "sdxl" and hasattr(self, "_cached_pooled_prompt_embeds"):
                pipeline_kwargs_override["pooled_prompt_embeds"] = (
                    self._cached_pooled_prompt_embeds.to(dev)
                )
                pipeline_kwargs_override["negative_pooled_prompt_embeds"] = (
                    self._cached_negative_pooled_prompt_embeds.to(dev)
                )

        conditioning_images, ground_truth_images = self._load_val_images()

        pipeline.set_progress_bar_config(disable=True)
        val_loop.run(
            pipeline, self.global_step, self.tb_logger,
            device=self.accelerator.device,
            conditioning_images=conditioning_images,
            pipeline_kwargs_override=pipeline_kwargs_override,
            ground_truth_images=ground_truth_images,
        )
        del pipeline

        self.controlnet.train()
        if self.joint_lora:
            self.unet.train()

    def _load_val_images(self):
        """采样验证用条件图像和对应的原始训练图像（ground truth）。

        条件图来源（按优先级）：
          1. validation.val_conditioning_images 显式路径列表
          2. conditioning_data_dir 目录中固定 seed 采样
        原图来源：
          根据条件图通过后缀映射关系在 train_data_dir 中查找对应原图。

        Returns:
            (conditioning_images, ground_truth_images)
            conditioning_images: PIL Image 列表或 None
            ground_truth_images: PIL Image 列表或 None
        """
        from PIL import Image as PIL_Image
        from pathlib import Path
        from data.controlnet_dataset import _KNOWN_SUFFIX_PAIRS, _strip_known_suffix

        val_cfg = self.config.get("validation", {})
        n = val_cfg.get("num_val_samples", len(list(val_cfg.get("prompts", [""]))))
        resolution = self.config.data.get("resolution", 1024)
        _IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

        train_dir = Path(self.config.data.get("train_data_dir", ""))
        cond_dir_str = self.config.data.get("conditioning_data_dir", None)

        sampled_cond_files: list[Path] = []

        val_cond_paths = list(val_cfg.get("val_conditioning_images", []))
        if val_cond_paths:
            sampled_cond_files = [Path(p) for p in val_cond_paths[:n]]
            while len(sampled_cond_files) < n:
                sampled_cond_files.append(sampled_cond_files[len(sampled_cond_files) % len(val_cond_paths)])
        elif cond_dir_str:
            cond_path = Path(cond_dir_str)
            if cond_path.exists():
                files = sorted(
                    f for f in cond_path.iterdir()
                    if f.is_file() and f.suffix.lower() in _IMG_EXTS
                )
                if files:
                    import random
                    rng = random.Random(42)
                    sampled_cond_files = rng.sample(files, min(n, len(files)))
                    logger.info(f"验证条件图采样：{[f.name for f in sampled_cond_files]}")

        if not sampled_cond_files:
            logger.warning("验证时跳过条件图：无可用条件图像")
            return None, None

        conditioning_images = [
            PIL_Image.open(f).convert("RGB").resize((resolution, resolution))
            for f in sampled_cond_files
        ]

        # 通过后缀映射查找对应的原始训练图像
        ground_truth_images = None
        if train_dir.exists():
            gt_images = []
            train_index: dict[str, Path] = {}
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

            for cf in sampled_cond_files:
                bk = None
                for _, cond_suffix in _KNOWN_SUFFIX_PAIRS:
                    bk = _strip_known_suffix(cf.name, cond_suffix)
                    if bk is not None:
                        break
                if bk is None:
                    bk = cf.stem
                gt_path = train_index.get(bk)
                if gt_path is not None:
                    gt_images.append(
                        PIL_Image.open(gt_path).convert("RGB").resize((resolution, resolution))
                    )
                else:
                    gt_images.append(None)

            if any(img is not None for img in gt_images):
                placeholder = PIL_Image.new("RGB", (resolution, resolution), (128, 128, 128))
                ground_truth_images = [
                    img if img is not None else placeholder for img in gt_images
                ]
                logger.info(f"验证原图加载：{sum(1 for img in gt_images if img is not None)}/{n} 张匹配成功")

        return conditioning_images, ground_truth_images

    # ── 检查点 ───────────────────────────────────────────────────────────

    def _save_checkpoint(self, optimizer, lr_scheduler):
        if not self.accelerator.is_main_process:
            return

        self.ckpt_manager.save(
            step=self.global_step,
            global_epoch=self.global_epoch,
            unet=self.accelerator.unwrap_model(self.unet) if self.joint_lora else None,
            controlnet=self.accelerator.unwrap_model(self.controlnet),
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            seed=self.training_cfg.get("seed", 42),
            is_lora=self.joint_lora,
            ema_loss=self._ema_loss,
        )
