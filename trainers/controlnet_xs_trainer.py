"""ControlNet-XS 训练器 — 基于 ControlNet-XS 的轻量级条件控制网络训练。

与标准 ControlNet 的关键差异：
  1. 模型结构：ControlNetXSAdapter + UNetControlNetXSModel 融合为单一模型
  2. 前向传播：unet_xs(sample, t, embeds, controlnet_cond=cond).sample
     （标准 ControlNet 需要两次前向：controlnet → unet）
  3. 冻结策略：unet_xs.freeze_unet_params() 一次性冻结 base UNet 参数
  4. 推理管线：StableDiffusionXLControlNetXSPipeline
"""

import logging
import math
import os

import torch
from diffusers import UNet2DConditionModel
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.controlnet_xs import create_controlnet_xs_sd15, create_controlnet_xs_sdxl
from models.model_loader import resolve_model_path
from utils.memory import apply_memory_optimizations
from utils.validation import ValidationLoop
from .controlnet_trainer import ControlNetTrainer

logger = logging.getLogger(__name__)


class ControlNetXSTrainer(ControlNetTrainer):
    """ControlNet-XS 训练器，继承 ControlNetTrainer 并覆盖 XS 相关方法。"""

    def __init__(self, config: DictConfig):
        # adapter: ControlNetXSAdapter（用于保存 checkpoint 和构建推理 pipeline）
        # unet_xs: UNetControlNetXSModel（用于训练前向传播）
        self.adapter = None
        self.unet_xs = None
        super().__init__(config)

        # unet_xs 也需要梯度检查点和注意力后端优化
        if self.unet_xs is not None:
            apply_memory_optimizations(
                unet=self.unet_xs,
                enable_gradient_checkpointing=self.training_cfg.get("gradient_checkpointing", True),
                attention_backend=self.training_cfg.get("attention_backend", "sdpa"),
                enable_channels_last=self.training_cfg.get("enable_channels_last", True),
            )

    def _create_controlnet(self):
        """创建 ControlNet-XS：adapter + unet_xs 融合模型。"""
        conditioning_channels = self.cn_cfg.get("conditioning_channels", 3)
        model_path = self.config.model.pretrained_model_name_or_path
        weights_dir = self.config.model.get("weights_dir", None)

        xs_cfg = self.config.get("controlnet_xs", {})
        size_ratio = xs_cfg.get("size_ratio", None)

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
            self.adapter, self.unet_xs = create_controlnet_xs_sdxl(
                resolved, conditioning_channels, size_ratio=size_ratio,
                unet=source_unet, dtype=weight_dtype,
            )
        else:
            resolved = resolve_model_path(model_path, weights_dir, "sd15")
            self.adapter, self.unet_xs = create_controlnet_xs_sd15(
                resolved, conditioning_channels, size_ratio=size_ratio,
                unet=source_unet, dtype=weight_dtype,
            )

        # ControlNetTrainer 的其他方法会引用 self.controlnet，
        # 这里设为 adapter 以兼容 checkpoint 保存逻辑
        self.controlnet = self.adapter

        if self._merged_unet_for_cn is not None:
            del self._merged_unet_for_cn
            self._merged_unet_for_cn = None
            logger.info("ControlNet-XS initialized from merged LoRA UNet")
        else:
            logger.info(f"ControlNet-XS initialized from {model_path}")

    def _freeze_parameters(self):
        """冻结 base UNet 参数，仅 adapter 参数可训练。"""
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        if self.text_encoder_2 is not None:
            self.text_encoder_2.requires_grad_(False)

        # UNetControlNetXSModel.freeze_unet_params() 冻结 base UNet 部分，
        # 保留 adapter（ctrl_*）参数可训练
        self.unet_xs.freeze_unet_params()

        # 标准 ControlNet Trainer 用的 self.unet 不再参与训练
        self.unet.requires_grad_(False)

        n_trainable = sum(p.numel() for p in self.unet_xs.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.unet_xs.parameters())
        logger.info(
            f"ControlNet-XS freeze: {n_trainable/1e6:.1f}M trainable / "
            f"{n_total/1e6:.1f}M total ({n_trainable/n_total*100:.1f}%)"
        )

    def train(self):
        """ControlNet-XS 主训练循环。"""
        cache_latents = self.training_cfg.get("cache_latents", False)

        self.vae.to(self.accelerator.device, dtype=torch.float32)

        if cache_latents:
            self._precompute_latents_distributed(self._get_latent_cache_dir())

        dataloader = self._build_dataloader()
        num_train_steps = self.training_cfg.get("num_train_steps", 10000)
        max_grad_norm = self.training_cfg.get("max_grad_norm", 1.0)
        validation_steps = self.training_cfg.get("validation_steps", 500)
        save_steps = self.training_cfg.get("save_steps", 500)

        all_trainable = [p for p in self.unet_xs.parameters() if p.requires_grad]

        optimizer = self.setup_optimizer(trainable_params=all_trainable)
        lr_scheduler = self.setup_lr_scheduler(optimizer, num_train_steps)

        # prepare unet_xs（融合模型），标准 unet 不参与训练
        self.unet_xs, optimizer, dataloader, lr_scheduler = (
            self.accelerator.prepare(
                self.unet_xs, optimizer, dataloader, lr_scheduler
            )
        )

        self.text_encoder.to(self.accelerator.device)
        if self.text_encoder_2 is not None:
            self.text_encoder_2.to(self.accelerator.device)

        if self.training_cfg.get("cache_text_embeddings", False):
            self._precompute_text_embeddings()

        resume_dir = self.training_cfg.get("resume_from_checkpoint", None)
        if resume_dir == "latest":
            resume_dir = self.ckpt_manager.get_latest_checkpoint()
        if resume_dir:
            state = self.ckpt_manager.load(
                resume_dir,
                controlnet=self.accelerator.unwrap_model(self.adapter),
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
            )
            self.global_step = state["step"]
            self.global_epoch = state["epoch"]

        val_cfg = self.config.get("validation", {})
        val_loop = ValidationLoop(
            prompts=list(val_cfg.get("prompts", ["a test image"])),
            negative_prompt=val_cfg.get("negative_prompt", ""),
            num_inference_steps=val_cfg.get("num_inference_steps", 25),
            guidance_scale=val_cfg.get("guidance_scale", 7.5),
            seed=val_cfg.get("seed", 42),
            save_dir=os.path.join(self.training_cfg.get("output_dir", "./outputs"), "samples"),
        )

        gradient_accumulation_steps = self.training_cfg.get("gradient_accumulation_steps", 1)
        steps_per_epoch = math.ceil(len(dataloader) / gradient_accumulation_steps)
        num_epochs = math.ceil(num_train_steps / max(steps_per_epoch, 1))
        progress_bar = tqdm(
            total=num_train_steps,
            initial=self.global_step,
            desc="Training ControlNet-XS",
            disable=not self.accelerator.is_main_process,
        )

        self.unet_xs.train()

        for epoch in range(num_epochs):
            self.global_epoch = epoch
            for batch in dataloader:
                if self.global_step >= num_train_steps:
                    break

                with self.accelerator.accumulate(self.unet_xs):
                    loss = self._training_step(batch)
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        grad_norm = self.accelerator.clip_grad_norm_(all_trainable, max_grad_norm)
                        grad_norm = grad_norm.item() if hasattr(grad_norm, 'item') else float(grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    self.global_step += 1
                    current_lr = lr_scheduler.get_last_lr()[0]
                    self.log_step(loss.item(), current_lr, grad_norm)

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
        logger.info("ControlNet-XS training complete!")

    def _training_step(self, batch) -> torch.Tensor:
        """单步训练: 加噪 → UNetControlNetXSModel 联合前向 → loss。

        与标准 ControlNet 的区别：单次前向传播，无需分别调用 controlnet 和 unet。
        """
        conditioning_pixel_values = batch["conditioning_pixel_values"].to(
            dtype=self.accelerator.unwrap_model(self.unet_xs).dtype
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
            noise_pred = self.unet_xs(
                sample=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                controlnet_cond=conditioning_pixel_values,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=True,
            ).sample
        else:
            if hasattr(self, "_cached_prompt_embeds"):
                encoder_hidden_states = self._cached_prompt_embeds.expand(bsz, -1, -1).to(latents.device)
            else:
                encoder_hidden_states = self._encode_prompt_sd15(batch["input_ids"])

            noise_pred = self.unet_xs(
                sample=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=conditioning_pixel_values,
                return_dict=True,
            ).sample

        if self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            target = noise

        loss = self.compute_loss(
            noise_pred, target, timesteps,
            snr_cache=self._snr_cache,
            min_snr_gamma=self.min_snr_gamma,
            prediction_type=self.noise_scheduler.config.prediction_type,
        )
        return loss

    @torch.no_grad()
    def _run_validation(self, val_loop: ValidationLoop):
        if not self.accelerator.is_main_process:
            return

        from diffusers import (
            DPMSolverMultistepScheduler,
            StableDiffusionControlNetXSPipeline,
            StableDiffusionXLControlNetXSPipeline,
        )

        self.unet_xs.eval()
        unwrapped_unet_xs = self.accelerator.unwrap_model(self.unet_xs)

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
            pipeline = StableDiffusionXLControlNetXSPipeline(
                vae=self.vae,
                unet=unwrapped_unet_xs,
                controlnet=self.adapter,
                text_encoder=unwrapped_te,
                text_encoder_2=unwrapped_te2,
                tokenizer=self.tokenizer,
                tokenizer_2=self.tokenizer_2,
                scheduler=inference_scheduler,
            )
        else:
            pipeline = StableDiffusionControlNetXSPipeline(
                vae=self.vae,
                unet=unwrapped_unet_xs,
                controlnet=self.adapter,
                text_encoder=unwrapped_te,
                tokenizer=self.tokenizer,
                scheduler=inference_scheduler,
                safety_checker=None,
                feature_extractor=None,
            )

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

        self.unet_xs.train()

    def _save_checkpoint(self, optimizer, lr_scheduler):
        if not self.accelerator.is_main_process:
            return

        self.ckpt_manager.save(
            step=self.global_step,
            global_epoch=self.global_epoch,
            controlnet=self.adapter,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            seed=self.training_cfg.get("seed", 42),
        )
