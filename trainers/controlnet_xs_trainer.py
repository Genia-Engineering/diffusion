"""ControlNet-XS 训练器 — 基于 ControlNet-XS 的轻量级条件控制网络训练。

与标准 ControlNet 的关键差异：
  1. 模型结构：ControlNetXSAdapter + UNetControlNetXSModel 融合为单一模型
  2. 前向传播：unet_xs(sample, t, embeds, controlnet_cond=cond).sample
     （标准 ControlNet 需要两次前向：controlnet → unet）
  3. 冻结策略：unet_xs.freeze_unet_params() 一次性冻结 base UNet 参数
  4. 推理管线：StableDiffusionXLControlNetXSPipeline
"""

import gc
import logging
import math
import os

import torch
import torch.nn.functional as F
from diffusers import UNet2DConditionModel
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.controlnet_xs import (
    create_controlnet_xs_sd15,
    create_controlnet_xs_sdxl,
    strip_ctrl_cross_attention,
)
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

        self._init_auxiliary_loss(config)

        # unet_xs 也需要梯度检查点和注意力后端优化
        if self.unet_xs is not None:
            apply_memory_optimizations(
                unet=self.unet_xs,
                enable_gradient_checkpointing=self.training_cfg.get("gradient_checkpointing", True),
                attention_backend=self.training_cfg.get("attention_backend", "sdpa"),
                enable_channels_last=self.training_cfg.get("enable_channels_last", True),
            )

        # XS 训练仅使用 unet_xs（融合模型），self.unet 从未参与前向传播，
        # 释放其 CPU 内存 (~5GB for SDXL bf16)
        if self.unet is not None:
            del self.unet
            self.unet = None
            gc.collect()
            logger.info("已释放冗余 self.unet（XS 训练仅使用 unet_xs）")

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

        # 可选：移除 ctrl 分支的 cross-attention（与 PixArt XS 一致）
        if xs_cfg.get("no_ctrl_cross_attention", False):
            removed = strip_ctrl_cross_attention(self.unet_xs)
            logger.info(
                f"Stripped ctrl cross-attention: removed {removed/1e6:.1f}M params, "
                f"ctrl branch now uses image info only"
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

    # ── 辅助结构 loss ─────────────────────────────────────────────────────

    def _init_auxiliary_loss(self, config: DictConfig):
        """解析 auxiliary_loss 配置，预构建颜色查找表和权重张量。"""
        aux_cfg = config.get("auxiliary_loss", {})
        self._aux_enabled = bool(aux_cfg.get("enabled", False))
        if not self._aux_enabled:
            return

        self._aux_weight = float(aux_cfg.get("weight", 0.1))
        self._aux_t_threshold = int(aux_cfg.get("timestep_threshold", 250))
        self._aux_fg_threshold = float(aux_cfg.get("fg_threshold", 0.1))
        self._aux_temperature = float(aux_cfg.get("temperature", 20.0))
        self._aux_bg_weight = float(aux_cfg.get("bg_weight", 0.0))
        self._aux_max_decode = int(aux_cfg.get("max_decode_samples", 2))
        self._last_aux_loss: float | None = None
        self._last_diffusion_loss: float | None = None

        labels = list(aux_cfg.get("labels", []))
        if not labels:
            logger.warning("auxiliary_loss.labels 为空，辅助 loss 将被禁用")
            self._aux_enabled = False
            return

        colors = []
        weights = []
        for lab in labels:
            colors.append(lab["rgb"])
            weights.append(float(lab["weight"]))

        self._aux_label_colors = torch.tensor(colors, dtype=torch.float32) / 255.0  # (N, 3)
        self._aux_label_weights = torch.tensor(weights, dtype=torch.float32)          # (N,)

        logger.info(
            f"Auxiliary structure loss: weight={self._aux_weight}, "
            f"t_threshold={self._aux_t_threshold}, fg_threshold={self._aux_fg_threshold}, "
            f"temperature={self._aux_temperature}, max_decode={self._aux_max_decode}, "
            f"{len(labels)} labels"
        )

    @torch.no_grad()
    def _build_label_weight_mask(
        self,
        conditioning_pixel_values: torch.Tensor,
        target_hw: tuple[int, int],
    ) -> torch.Tensor:
        """从控制图构建 per-label 权重 mask。

        Args:
            conditioning_pixel_values: (B, 3, H, W) [0, 1] 控制图
            target_hw: 目标 (H, W)，与 VAE 解码输出对齐

        Returns:
            (B, H, W) 权重 mask，bg=bg_weight，每个 label 区域=对应 weight
        """
        B, _, H, W = conditioning_pixel_values.shape
        device = conditioning_pixel_values.device

        label_colors = self._aux_label_colors.to(device)  # (N, 3)
        label_weights = self._aux_label_weights.to(device)  # (N,)

        pixels = conditioning_pixel_values.permute(0, 2, 3, 1)  # (B, H, W, 3)

        # 计算每个像素到每个 label 颜色的 L2 距离
        # pixels: (B, H, W, 1, 3), label_colors: (1, 1, 1, N, 3)
        dists = (pixels.unsqueeze(3) - label_colors.view(1, 1, 1, -1, 3)).pow(2).sum(dim=-1)  # (B, H, W, N)
        min_dist, min_idx = dists.min(dim=-1)  # (B, H, W)

        # 容差：距离太大的像素视为背景
        color_tol = (10.0 / 255.0) ** 2 * 3  # 容差：每通道 ±10/255
        is_matched = min_dist < color_tol

        weight_map = torch.full((B, H, W), self._aux_bg_weight, device=device)
        weight_map[is_matched] = label_weights[min_idx[is_matched]]

        # 下采样到目标尺寸
        if (H, W) != target_hw:
            weight_map = F.interpolate(
                weight_map.unsqueeze(1), size=target_hw, mode="nearest",
            ).squeeze(1)

        return weight_map

    def _predict_x0(
        self,
        noisy_latents: torch.Tensor,
        noise_pred: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """从噪声预测计算 predicted x0（可微分）。"""
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(
            device=timesteps.device, dtype=noise_pred.dtype,
        )
        alpha_bar = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_alpha_bar = alpha_bar.sqrt()
        sqrt_one_minus_alpha_bar = (1.0 - alpha_bar).sqrt()

        if self.noise_scheduler.config.prediction_type == "v_prediction":
            x0 = sqrt_alpha_bar * noisy_latents - sqrt_one_minus_alpha_bar * noise_pred
        else:
            x0 = (noisy_latents - sqrt_one_minus_alpha_bar * noise_pred) / sqrt_alpha_bar.clamp(min=1e-8)
        return x0

    def _compute_auxiliary_loss(
        self,
        noise_pred: torch.Tensor,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        conditioning_pixel_values: torch.Tensor,
        batch: dict,
    ) -> torch.Tensor | None:
        """计算辅助结构 loss。

        流程：预测 x0 → VAE decode → 归一化 → max-channel → 软阈值 →
             × mask → 与原图二值 max-channel × mask 做 MSE（前景归一化）

        仅对 t < threshold 的样本计算；若 batch 中无符合条件的样本则返回 None。
        """
        low_t_mask = timesteps < self._aux_t_threshold
        if not low_t_mask.any():
            return None

        # 筛选低 timestep 样本，限制最大 decode 数量以控制显存
        idx = low_t_mask.nonzero(as_tuple=True)[0]
        if len(idx) > self._aux_max_decode:
            perm = torch.randperm(len(idx), device=idx.device)[: self._aux_max_decode]
            idx = idx[perm]
        sel_noise_pred = noise_pred[idx]
        sel_noisy_latents = noisy_latents[idx]
        sel_timesteps = timesteps[idx]
        sel_cond = conditioning_pixel_values[idx]

        # 1. 预测 x0（有梯度）
        x0_pred = self._predict_x0(sel_noisy_latents, sel_noise_pred, sel_timesteps)

        # 2. VAE decode（权重冻结但保留计算图，梯度穿过 VAE 回传到 UNet-XS）
        scaling_factor = self.vae.config.scaling_factor
        x0_for_decode = x0_pred / scaling_factor
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            decoded = self.vae.decode(x0_for_decode, return_dict=True).sample  # (B', 3, H, W)

        # 3. 归一化 [-1,1] → [0,1] → max-channel → 软阈值
        normalized = (decoded + 1.0) / 2.0
        pred_max = normalized.max(dim=1).values  # (B', H, W)
        pred_soft = torch.sigmoid(
            (pred_max - self._aux_fg_threshold) * self._aux_temperature
        )

        # 4. 原图二值 max-channel（预缓存，无梯度）
        if "orig_max_channel" in batch:
            orig_binary = batch["orig_max_channel"][idx].to(pred_soft.device)  # (B', 1, H/f, W/f)
            orig_binary = orig_binary.squeeze(1)  # (B', H/f, W/f)
            if orig_binary.shape[-2:] != pred_soft.shape[-2:]:
                orig_binary = F.interpolate(
                    orig_binary.unsqueeze(1), size=pred_soft.shape[-2:],
                    mode="nearest",
                ).squeeze(1)
        else:
            pixel_values = batch["pixel_values"][idx].to(pred_soft.device)
            with torch.no_grad():
                orig_01 = (pixel_values + 1.0) / 2.0
                orig_max = orig_01.max(dim=1).values
                orig_binary = (orig_max > self._aux_fg_threshold).float()

        orig_binary = orig_binary.detach()

        # 5. 控制图 per-label 权重 mask
        weight_mask = self._build_label_weight_mask(sel_cond, pred_soft.shape[-2:])

        # 6. 按 timestep 衰减的质量权重（t 越小 x0_pred 越可靠，权重越高）
        t_quality = 1.0 - sel_timesteps.float() / self._aux_t_threshold  # (B',)

        # 7. Weighted MSE：先算 MSE 再乘权重（线性加权，避免 w² 放大）
        diff_sq = (pred_soft - orig_binary).pow(2) * weight_mask * t_quality[:, None, None]
        fg_count = (weight_mask > 0).sum().clamp(min=1)
        aux_loss = diff_sq.sum() / fg_count

        self._last_aux_loss = aux_loss.item()
        return aux_loss

    def train(self):
        """ControlNet-XS 主训练循环。"""
        cache_latents = self.training_cfg.get("cache_latents", False)

        self.vae.to(self.accelerator.device, dtype=torch.float32)

        if cache_latents:
            self._precompute_latents_distributed(self._get_latent_cache_dir())
            if self._aux_enabled:
                self._delete_vae_encoder()
            else:
                self.vae.to("cpu")
            torch.cuda.empty_cache()

        self.text_encoder.to(self.accelerator.device)
        if self.text_encoder_2 is not None:
            self.text_encoder_2.to(self.accelerator.device)

        if self.training_cfg.get("cache_text_embeddings", False):
            self._precompute_text_embeddings()

        # fork 前将 unet_xs 移到 GPU 并回收堆碎片：
        #   1. unet_xs (.to GPU) 释放 ~5.2GB CPU tensor 存储
        #   2. gc + malloc_trim 强制 glibc 归还模型加载期间积累的堆碎片
        # 这样 DataLoader fork 的子进程不会继承这些无用的 CPU 内存页
        self.unet_xs.to(self.accelerator.device)
        gc.collect()
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        torch.cuda.empty_cache()

        dataloader = self._build_dataloader()
        num_train_steps = self.training_cfg.get("num_train_steps", 10000)
        max_grad_norm = self.training_cfg.get("max_grad_norm", 1.0)
        validation_steps = self.training_cfg.get("validation_steps", 500)
        save_steps = self.training_cfg.get("save_steps", 500)

        all_trainable = [p for p in self.unet_xs.parameters() if p.requires_grad]

        optimizer = self.setup_optimizer(trainable_params=all_trainable)
        lr_scheduler = self.setup_lr_scheduler(optimizer, num_train_steps)

        self.unet_xs, optimizer, dataloader = (
            self.accelerator.prepare(
                self.unet_xs, optimizer, dataloader
            )
        )

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
        prompt_source = val_cfg.get("prompt_source", "config")
        if prompt_source == "data":
            if hasattr(self, "_per_image_val_captions") and self._per_image_val_captions:
                val_prompts = self._per_image_val_captions
            else:
                val_prompts = [self.config.data.get("caption", "a test image")]
        else:
            val_prompts = list(val_cfg.get("prompts", ["a test image"]))

        num_val_samples = val_cfg.get("num_val_samples", None)
        if num_val_samples is not None and len(val_prompts) < num_val_samples:
            val_prompts = (val_prompts * ((num_val_samples // len(val_prompts)) + 1))[:num_val_samples]

        val_loop = ValidationLoop(
            prompts=val_prompts,
            negative_prompt=val_cfg.get("negative_prompt", ""),
            num_inference_steps=val_cfg.get("num_inference_steps", 25),
            guidance_scale=val_cfg.get("guidance_scale", 7.5),
            seed=val_cfg.get("seed", 42),
            num_images_per_prompt=val_cfg.get("num_images_per_prompt", 1),
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
                    optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    lr_scheduler.step()
                    self.global_step += 1
                    current_lr = lr_scheduler.get_last_lr()[0]
                    total_loss_val = loss.item()
                    self.log_step(total_loss_val, current_lr, grad_norm)

                    if self._aux_enabled:
                        if self._last_diffusion_loss is not None:
                            self.tb_logger.log_scalar(
                                "train/diffusion_loss", self._last_diffusion_loss, self.global_step,
                            )
                        if self._last_aux_loss is not None:
                            aux_w = self._aux_weight * self._last_aux_loss
                            self.tb_logger.log_scalar(
                                "train/aux_loss_weighted", aux_w, self.global_step,
                            )
                            if self._last_diffusion_loss > 0:
                                self.tb_logger.log_scalar(
                                    "train/aux_div_diffusion", aux_w / self._last_diffusion_loss, self.global_step,
                                )

                    progress_bar.update(1)
                    progress_bar.set_postfix(loss=f"{total_loss_val:.4f}", lr=f"{current_lr:.2e}")

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

        diffusion_loss = self.compute_loss(
            noise_pred, target, timesteps,
            snr_cache=self._snr_cache,
            min_snr_gamma=self.min_snr_gamma,
            prediction_type=self.noise_scheduler.config.prediction_type,
        )
        self._last_diffusion_loss = diffusion_loss.item()

        loss = diffusion_loss
        if self._aux_enabled:
            aux_loss = self._compute_auxiliary_loss(
                noise_pred, noisy_latents, timesteps,
                conditioning_pixel_values, batch,
            )
            if aux_loss is not None:
                loss = diffusion_loss + self._aux_weight * aux_loss

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
        dev = self.accelerator.device
        vae_was_on_cpu = next(self.vae.parameters()).device.type == "cpu"
        if vae_was_on_cpu:
            self.vae.to(dev)

        try:
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

            pipeline.to(dev)

            pipeline_kwargs_override = None
            if hasattr(self, "_cached_val_prompt_embeds_list") and self._cached_val_prompt_embeds_list:
                pipeline_kwargs_override = [
                    {k: v.to(dev) for k, v in d.items()}
                    for d in self._cached_val_prompt_embeds_list
                ]
            elif self.text_encoder is None and hasattr(self, "_cached_prompt_embeds"):
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
                device=dev,
                conditioning_images=conditioning_images,
                pipeline_kwargs_override=pipeline_kwargs_override,
                ground_truth_images=ground_truth_images,
            )
            del pipeline
        except Exception:
            logger.exception("Validation failed at step %d, skipping", self.global_step)
        finally:
            if vae_was_on_cpu:
                self.vae.to("cpu")
                torch.cuda.empty_cache()
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
