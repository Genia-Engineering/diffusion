"""PixArt-Sigma 全参数微调训练器 — 支持 Flow Matching 和 DDPM 双噪声范式。

通过配置项 noise_paradigm 切换训练范式:

1. Flow Matching (noise_paradigm="flow_matching"):
   sigma 参数化, 与 FlowMatchEulerDiscreteScheduler 对齐
   - 前向插值: x_t = t*x_1 + (1-t)*x_0, sigma = 1-t
   - 预测目标: velocity = x_0 - x_1 (noise - clean)
   - 时间步采样: t ~ Logit-Normal 或 Uniform(0,1), timestep = (1-t)*1000

2. DDPM (noise_paradigm="ddpm"):
   epsilon/v-prediction, 与 DDPMScheduler 对齐
   - 前向加噪: scheduler.add_noise(latents, noise, timesteps)
   - 预测目标: epsilon (noise) 或 v-prediction
   - 时间步采样: t ~ Uniform{0..num_train_timesteps-1}

权重冻结策略:
  - VAE: 全冻结，预缓存 latent 后卸载 encoder
  - T5 Text Encoder: 全冻结，预缓存嵌入后完全卸载
  - PixArtTransformer2DModel: 全部参数可训练（~611M）
"""

import logging
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.buckets import BucketManager, BucketSampler
from data.dataset import (
    PixArtSigmaCachedLatentDataset,
    PixArtSigmaDataset,
)
from models.model_loader import load_pixart_sigma_components
from utils.fid import FIDCalculator
from utils.memory import apply_memory_optimizations
from utils.validation import ValidationLoop
from .base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class PixArtSigmaTrainer(BaseTrainer):
    """PixArt-Sigma 全参数微调训练器。"""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.model_type = config.model.model_type  # "pixart_sigma"

        # 噪声范式: "flow_matching" 或 "ddpm"
        self.noise_paradigm: str = self.training_cfg.get("noise_paradigm", "flow_matching")

        # Flow Matching 参数
        self.timestep_sampling: str = self.training_cfg.get("timestep_sampling", "logit_normal")
        self.logit_mean: float = float(self.training_cfg.get("logit_mean", 0.0))
        self.logit_std: float = float(self.training_cfg.get("logit_std", 1.0))
        self.caption_dropout_rate: float = float(self.training_cfg.get("caption_dropout_rate", 0.1))

        # DDPM 参数
        self.noise_offset: float = float(self.training_cfg.get("noise_offset", 0.0))
        self.min_snr_gamma: float = float(self.training_cfg.get("min_snr_gamma", 0.0))

        # 空间加权 + 通道加权 (DDPM)
        self.spatial_weight_scale: float = float(self.training_cfg.get("spatial_weight_scale", 0.0))
        self.spatial_weight_threshold: float = float(self.training_cfg.get("spatial_weight_threshold", 0.1))
        self.channel_0_weight: float = float(self.training_cfg.get("channel_0_weight", 1.0))

        self._load_models()
        self._freeze_parameters()

        # DDPM 模式下预计算 SNR 查找表（Min-SNR 加权需要）
        self._snr_cache: torch.Tensor | None = None
        if self.noise_paradigm == "ddpm" and self.min_snr_gamma > 0:
            alphas_cumprod = self.noise_scheduler.alphas_cumprod
            self._snr_cache = alphas_cumprod / (1.0 - alphas_cumprod)

        apply_memory_optimizations(
            transformer=self.transformer,
            vae=self.vae,
            text_encoder=self.text_encoder,
            enable_gradient_checkpointing=self.training_cfg.get("gradient_checkpointing", True),
            attention_backend=self.training_cfg.get("attention_backend", "sdpa"),
            enable_channels_last=False,
        )

    def _load_models(self):
        model_path = self.config.model.pretrained_model_name_or_path
        weights_dir = self.config.model.get("weights_dir", None)
        scheduler_shift = float(self.training_cfg.get("scheduler_shift", 1.0))
        use_flow_matching = self.noise_paradigm == "flow_matching"

        components = load_pixart_sigma_components(
            model_path, weights_dir=weights_dir, scheduler_shift=scheduler_shift,
            flow_matching=use_flow_matching,
        )
        self.vae = components["vae"]
        self.transformer = components["transformer"]
        self.text_encoder = components["text_encoder"]
        self.tokenizer = components["tokenizer"]
        self.noise_scheduler = components["noise_scheduler"]

        # DDPM 模式：允许配置覆盖 prediction_type
        if not use_flow_matching:
            override_pred_type = self.training_cfg.get("prediction_type", None)
            if override_pred_type and override_pred_type != self.noise_scheduler.config.prediction_type:
                self.noise_scheduler.register_to_config(prediction_type=override_pred_type)
                logger.info(f"DDPM prediction_type overridden to '{override_pred_type}'")
            logger.info(
                f"噪声范式: DDPM (prediction_type={self.noise_scheduler.config.prediction_type})"
            )
        else:
            logger.info("噪声范式: Flow Matching (Rectified Flow, sigma velocity)")

    def _freeze_parameters(self):
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.transformer.requires_grad_(True)
        self.print_trainable_params(self.transformer)

    def _text_embed_cache_exists(self) -> bool:
        """检查文本嵌入缓存是否已存在，避免不必要地将 T5 移至 GPU。"""
        import hashlib
        caption = self.config.data.get("caption", "")
        neg_prompt = self.config.get("validation", {}).get("negative_prompt", "")
        data_dir = self.config.data.get("train_data_dir", "./data")
        embed_hash = hashlib.md5(f"{caption}||{neg_prompt}".encode()).hexdigest()[:8]
        cache_file = Path(data_dir) / "text_embed_cache" / f"embeds_{self.model_type}_{embed_hash}.pt"
        return cache_file.exists()

    @torch.no_grad()
    def _encode_null_prompt(self):
        """预编码空提示词（用于 caption dropout），在 text_encoder 卸载前调用。"""
        device = self.text_encoder.device
        t5_max_len = 300
        tokenized = self.tokenizer(
            "", padding="max_length", truncation=True,
            max_length=t5_max_len, return_tensors="pt",
        )
        ids = tokenized.input_ids.to(device)
        attn_mask = tokenized.attention_mask.to(device)
        self._cached_negative_prompt_embeds = self.text_encoder(ids, attention_mask=attn_mask)[0].cpu()
        self._cached_negative_prompt_attention_mask = attn_mask.cpu()
        logger.info(f"Null prompt encoded for caption dropout (rate={self.caption_dropout_rate})")

    def _build_dataloader(self) -> DataLoader:
        data_cfg = self.config.data
        resolution = data_cfg.get("resolution", 1024)
        batch_size = self.training_cfg.get("train_batch_size", 2)
        fixed_caption = data_cfg.get("caption", data_cfg.get("caption_fallback", ""))
        cache_latents = self.training_cfg.get("cache_latents", False)

        if cache_latents:
            cache_dir = self._get_latent_cache_dir()
            text_embed_cache_dir = (
                self._get_text_embed_per_image_cache_dir()
                if getattr(self, "_per_image_caption", False)
                and self.training_cfg.get("cache_text_embeddings", False)
                else None
            )
            dataset = PixArtSigmaCachedLatentDataset(
                data_dir=data_cfg.train_data_dir,
                cache_dir=cache_dir,
                tokenizer=self.tokenizer,
                resolution=resolution,
                random_flip=data_cfg.get("random_flip", True),
                fixed_caption=fixed_caption,
                text_embed_cache_dir=text_embed_cache_dir,
            )
        else:
            dataset = PixArtSigmaDataset(
                data_dir=data_cfg.train_data_dir,
                tokenizer=self.tokenizer,
                resolution=resolution,
                center_crop=data_cfg.get("center_crop", False),
                random_flip=data_cfg.get("random_flip", True),
                fixed_caption=fixed_caption,
            )

        use_bucketing = data_cfg.get("use_aspect_ratio_bucketing", True)
        if use_bucketing:
            bucket_manager = BucketManager(model_type=self.model_type)
            image_sizes = dataset.get_image_sizes()
            bucket_to_indices = bucket_manager.assign_buckets(image_sizes)
            dataset.set_bucket_assignments(bucket_to_indices)
            self._log_bucket_stats(bucket_to_indices)
            sampler = BucketSampler(bucket_to_indices, batch_size, drop_last=True, shuffle=True)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=self.training_cfg.get("dataloader_num_workers", 4),
                pin_memory=True,
                drop_last=True,
            )
        else:
            if cache_latents:
                logger.info(
                    f"分桶已关闭，所有图片 pad 至 {resolution}×{resolution}，"
                    f"使用标准 shuffle DataLoader"
                )
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=self.training_cfg.get("dataloader_num_workers", 4),
                pin_memory=True,
                drop_last=True,
            )

        return dataloader

    def _log_bucket_stats(self, bucket_to_indices: dict) -> None:
        total = sum(len(v) for v in bucket_to_indices.values())
        lines = [f"Aspect ratio bucket distribution (total={total}):"]
        for (w, h), indices in sorted(bucket_to_indices.items()):
            lines.append(f"  {w}×{h}: {len(indices)} images ({len(indices)/total*100:.1f}%)")
        logger.info("\n".join(lines))

    def _sample_real_images(self, n: int) -> list:
        """从训练数据集随机采样 n 张原始 PIL 图像，供 FID 计算使用。"""
        import random
        from data.dataset import BaseImageDataset

        data_cfg = self.config.data
        resolution = data_cfg.get("resolution", 1024)

        dataset = BaseImageDataset(
            data_dir=data_cfg.train_data_dir,
            resolution=resolution,
            center_crop=data_cfg.get("center_crop", False),
            random_flip=False,
        )

        indices = random.sample(range(len(dataset)), min(n, len(dataset)))
        images = [dataset.get_pil_image(i) for i in indices]
        return images

    def _load_val_ground_truth_images(self, n: int) -> list:
        """从训练数据集按顺序取前 n 张图作为验证对比用的 ground truth。

        使用固定顺序（非随机）确保每次验证对比的是同一批图像，
        便于直观追踪过拟合进度。
        """
        from data.dataset import BaseImageDataset

        data_cfg = self.config.data
        resolution = data_cfg.get("resolution", 1024)

        dataset = BaseImageDataset(
            data_dir=data_cfg.train_data_dir,
            resolution=resolution,
            center_crop=data_cfg.get("center_crop", False),
            random_flip=False,
        )

        indices = list(range(min(n, len(dataset))))
        images = [dataset.get_pil_image(i) for i in indices]
        return images

    def train(self):
        """PixArt-Sigma 全参数微调主训练循环。"""
        cache_latents = self.training_cfg.get("cache_latents", False)
        cache_text_embeddings = self.training_cfg.get("cache_text_embeddings", False)

        # ── 阶段1：VAE latent 预计算（完成后卸载 encoder 释放显存）────────────
        self.vae.to(self.accelerator.device, dtype=torch.float32)
        if cache_latents:
            self._precompute_latents_distributed(self._get_latent_cache_dir())

        # ── 阶段2：T5 文本嵌入预计算（必须在 accelerator.prepare 之前完成）─────
        # T5-XXL 在 bf16 下约 11 GB，accelerator.prepare 后 DiT+optimizer 约 14 GB，
        # 两者同时在 GPU 会 OOM。先做文本预计算再卸载 T5，再走 prepare。
        # 仅在缓存不存在时才将 T5 移至 GPU 进行编码；缓存命中则直接从磁盘加载，跳过 GPU 占用。
        caption_mode = self.config.data.get("caption_mode", "fixed")
        self._per_image_caption = (caption_mode == "per_image") and bool(self.config.data.get("caption_dir", None))
        if caption_mode == "per_image" and not self.config.data.get("caption_dir", None):
            logger.warning("caption_mode='per_image' but caption_dir not set, falling back to fixed caption")

        if cache_text_embeddings:
            if self._per_image_caption:
                text_embed_cache_dir = self._get_text_embed_per_image_cache_dir()
                need_encode = not self._per_image_text_embed_cache_exists(text_embed_cache_dir)
                if need_encode:
                    self.text_encoder.to(self.accelerator.device)
                self._precompute_per_image_text_embeddings(text_embed_cache_dir)
            else:
                need_encode = not self._text_embed_cache_exists()
                if need_encode:
                    self.text_encoder.to(self.accelerator.device)
                self._precompute_text_embeddings()
            # _precompute_text_embeddings 内部会 del self.text_encoder 并 empty_cache
        else:
            self.text_encoder.to(self.accelerator.device)

        if self.caption_dropout_rate > 0 and not hasattr(self, "_cached_negative_prompt_embeds"):
            self._encode_null_prompt()

        # ── 阶段3：准备训练组件（DiT + optimizer 上 GPU）────────────────────────
        dataloader = self._build_dataloader()
        num_train_steps = self.training_cfg.get("num_train_steps", 5000)
        max_grad_norm = self.training_cfg.get("max_grad_norm", 1.0)
        validation_steps = self.training_cfg.get("validation_steps", 500)
        save_steps = self.training_cfg.get("save_steps", 500)

        trainable_params = [p for p in self.transformer.parameters() if p.requires_grad]
        optimizer = self.setup_optimizer(trainable_params=trainable_params)
        lr_scheduler = self.setup_lr_scheduler(optimizer, num_train_steps)

        use_deepspeed = os.environ.get("ACCELERATE_USE_DEEPSPEED", "false").lower() == "true"
        if self.training_cfg.get("compile_model", False):
            if use_deepspeed:
                logger.warning(
                    "检测到 DeepSpeed（ZeRO），跳过 torch.compile。"
                    "两者同时使用会导致双重 wrapping，引发 fp32 强制转换和显存峰值激增。"
                )
            else:
                logger.info("Compiling transformer with torch.compile (mode=max-autotune-no-cudagraphs)...")
                self.transformer = torch.compile(
                    self.transformer, mode="max-autotune-no-cudagraphs"
                )

        self.transformer, optimizer, dataloader = self.accelerator.prepare(
            self.transformer, optimizer, dataloader
        )

        # cache_text_embeddings=False 时 text_encoder 仍需移到 GPU（实时编码路径）
        if not cache_text_embeddings and self.text_encoder is not None:
            self.text_encoder.to(self.accelerator.device)

        # 恢复训练
        resume_dir = self.training_cfg.get("resume_from_checkpoint", None)
        if resume_dir == "latest":
            resume_dir = self.ckpt_manager.get_latest_checkpoint()
        if resume_dir:
            state = self.ckpt_manager.load(
                resume_dir,
                transformer=self.accelerator.unwrap_model(self.transformer),
                optimizer=optimizer,
                lr_scheduler=None,
                is_lora=False,
            )
            self.global_step = state["step"]
            self.global_epoch = state["epoch"]
            for _ in range(self.global_step):
                lr_scheduler.step()
            logger.info(f"Resumed from step {self.global_step}, lr_scheduler re-synced to step {self.global_step}")

        # FID 计算器初始化
        val_cfg = self.config.get("validation", {})
        fid_calculator = None
        if val_cfg.get("compute_fid", False):
            fid_cfg = val_cfg.get("fid", {})
            cache_path = fid_cfg.get(
                "real_features_cache",
                os.path.join(self.training_cfg.get("output_dir", "./outputs"), "fid_real_features.npz"),
            )
            fid_calculator = FIDCalculator(
                model_name=fid_cfg.get("model_name", "dinov2_vitb14"),
                device=self.accelerator.device,
                real_images_cache_path=cache_path if self.accelerator.is_main_process else None,
            )
            if self.accelerator.is_main_process:
                fid_num_real = fid_cfg.get("num_real_images", 512)
                real_images = self._sample_real_images(fid_num_real)
                fid_calculator.update_real(real_images)
                logger.info(f"FID 计算器就绪，真实图像数: {len(real_images)}")
            self.accelerator.wait_for_everyone()

        fid_num_gen = val_cfg.get("fid", {}).get("num_gen_images", 256)
        prompt_source = val_cfg.get("prompt_source", "config")
        if prompt_source == "data":
            if hasattr(self, "_per_image_val_captions") and self._per_image_val_captions:
                val_prompts = self._per_image_val_captions
            else:
                val_prompts = [self.config.data.get("caption", "a test image")]
        else:
            val_prompts = list(val_cfg.get("prompts", ["a test image"]))
        raw_strengths = val_cfg.get("img2img_strength", None)
        if raw_strengths is None:
            img2img_strengths = []
        elif isinstance(raw_strengths, (int, float)):
            img2img_strengths = [float(raw_strengths)] if float(raw_strengths) > 0 else []
        else:
            img2img_strengths = [float(s) for s in raw_strengths if float(s) > 0]

        val_loop = ValidationLoop(
            prompts=val_prompts,
            negative_prompt=val_cfg.get("negative_prompt", ""),
            num_inference_steps=val_cfg.get("num_inference_steps", 20),
            guidance_scale=val_cfg.get("guidance_scale", 4.5),
            seed=val_cfg.get("seed", 42),
            num_images_per_prompt=val_cfg.get("num_images_per_prompt", 1),
            save_dir=os.path.join(self.training_cfg.get("output_dir", "./outputs"), "samples"),
            fid_calculator=fid_calculator,
            fid_num_gen_images=fid_num_gen,
            fid_batch_size=val_cfg.get("fid", {}).get("batch_size", 4),
            img2img_strengths=img2img_strengths,
        )

        # 加载验证用 ground truth：固定取训练集前 N 张（N = prompt 数量），
        # 每次验证都与这些图对比，直观追踪过拟合进度。
        val_gt_images = None
        if self.accelerator.is_main_process:
            val_gt_images = self._load_val_ground_truth_images(len(val_prompts))
            logger.info(f"验证 ground truth 加载完成：{len(val_gt_images)} 张训练图")

        # best loss 追踪：EMA 平滑逐步检查，遇到新低即保存
        self._ema_loss = None
        self._ema_decay = 0.99  # ~100 步窗口
        self._best_ema_loss = float("inf")
        self._last_best_save_step = 0
        self._min_best_save_interval = max(save_steps // 4, 50)  # 最小保存间隔

        # 训练主循环
        gradient_accumulation_steps = self.training_cfg.get("gradient_accumulation_steps", 1)
        steps_per_epoch = math.ceil(len(dataloader) / gradient_accumulation_steps)
        num_epochs = math.ceil(num_train_steps / max(steps_per_epoch, 1))
        progress_bar = tqdm(
            total=num_train_steps,
            initial=self.global_step,
            desc="Training",
            disable=not self.accelerator.is_main_process,
        )

        self.transformer.train()

        for epoch in range(num_epochs):
            self.global_epoch = epoch
            for batch in dataloader:
                if self.global_step >= num_train_steps:
                    break

                with self.accelerator.accumulate(self.transformer):
                    loss = self._training_step(batch)
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        grad_norm = self.accelerator.clip_grad_norm_(trainable_params, max_grad_norm)
                        grad_norm = grad_norm.item() if hasattr(grad_norm, 'item') else float(grad_norm)

                    optimizer.step()
                    optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    lr_scheduler.step()
                    self.global_step += 1
                    current_loss = loss.item()
                    current_lr = lr_scheduler.get_last_lr()[0]
                    self.log_step(current_loss, current_lr, grad_norm)
                    self._maybe_save_best()

                    progress_bar.update(1)
                    progress_bar.set_postfix(loss=f"{current_loss:.4f}", lr=f"{current_lr:.2e}")

                    if self.global_step % save_steps == 0:
                        self._save_checkpoint(optimizer, lr_scheduler)
                        self.accelerator.wait_for_everyone()

                    if self.global_step % validation_steps == 0:
                        self._run_validation(val_loop, val_gt_images)
                        self.accelerator.wait_for_everyone()

            if self.global_step >= num_train_steps:
                break

        self._save_checkpoint(optimizer, lr_scheduler)
        self.tb_logger.close()
        self.accelerator.end_training()
        logger.info("Training complete!")

    def _training_step(self, batch) -> torch.Tensor:
        """单步训练，根据 noise_paradigm 分发到 Flow Matching 或 DDPM 路径。"""
        # ── 获取 clean latent ──
        if "latents" in batch:
            latents = batch["latents"].to(self.accelerator.device)
        else:
            pixel_values = batch["pixel_values"].to(dtype=self.vae.dtype)
            with torch.no_grad():
                latents = self.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor

        bsz = latents.shape[0]
        noise = torch.randn_like(latents)

        # ── 获取文本条件 ──
        if "prompt_embeds" in batch:
            prompt_embeds = batch["prompt_embeds"].to(latents.device)
            attention_mask = batch["prompt_attention_mask"].to(latents.device)
        elif hasattr(self, "_cached_prompt_embeds"):
            prompt_embeds = self._cached_prompt_embeds.expand(bsz, -1, -1).to(latents.device)
            attention_mask = self._cached_prompt_attention_mask.expand(bsz, -1).to(latents.device)
        else:
            input_ids = batch["input_ids"].to(self.accelerator.device)
            attn_mask = batch["attention_mask"].to(self.accelerator.device)
            with torch.no_grad():
                prompt_embeds = self.text_encoder(input_ids, attention_mask=attn_mask)[0]
            attention_mask = attn_mask

        # ── caption dropout ──
        if self.caption_dropout_rate > 0:
            drop_mask = torch.rand(bsz, device=latents.device) < self.caption_dropout_rate
            if drop_mask.any():
                null_embeds = self._cached_negative_prompt_embeds.to(latents.device)
                null_mask = self._cached_negative_prompt_attention_mask.to(latents.device)
                prompt_embeds = prompt_embeds.clone()
                attention_mask = attention_mask.clone()
                prompt_embeds[drop_mask] = null_embeds.expand(drop_mask.sum(), -1, -1)
                attention_mask[drop_mask] = null_mask.expand(drop_mask.sum(), -1)

        padding_mask = batch.get("padding_mask", None)
        if padding_mask is not None:
            padding_mask = padding_mask.to(latents.device)

        if self.noise_paradigm == "flow_matching":
            return self._training_step_fm(latents, noise, bsz, prompt_embeds, attention_mask, padding_mask)
        else:
            weight_mask = self._get_weight_mask(batch, latents)
            return self._training_step_ddpm(latents, noise, bsz, prompt_embeds, attention_mask, weight_mask, padding_mask)

    def _training_step_fm(self, latents, noise, bsz, prompt_embeds, attention_mask, padding_mask=None) -> torch.Tensor:
        """Flow Matching 训练步: 线性插值 → velocity prediction → MSE loss。

        Rectified Flow (sigma 参数化):
          x_t = t*clean + (1-t)*noise, sigma = 1-t
          target = noise - clean (sigma velocity)
          timestep = sigma * 1000 = (1-t) * 1000
        """
        if self.timestep_sampling == "logit_normal":
            t = torch.sigmoid(
                self.logit_mean + self.logit_std * torch.randn(bsz, device=latents.device)
            )
        else:
            t = torch.rand(bsz, device=latents.device)

        t_expanded = t.view(-1, 1, 1, 1)
        noisy_latents = t_expanded * latents + (1.0 - t_expanded) * noise
        timesteps_scaled = (1.0 - t) * 1000.0

        model_output = self.transformer(
            hidden_states=noisy_latents,
            timestep=timesteps_scaled,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=attention_mask,
        ).sample

        if model_output.shape[1] != latents.shape[1]:
            model_output, _ = model_output.chunk(2, dim=1)

        target = noise - latents

        if padding_mask is not None:
            per_pixel = F.mse_loss(model_output.float(), target.float(), reduction="none")
            n_content = padding_mask.sum() * per_pixel.shape[1]
            return (per_pixel * padding_mask).sum() / n_content.clamp(min=1)
        return F.mse_loss(model_output.float(), target.float())

    def _get_weight_mask(self, batch, latents) -> torch.Tensor | None:
        """从 batch 获取或计算亮度权重掩码 (B, 1, H, W)，与 latent 同分辨率。"""
        if self.spatial_weight_scale <= 0:
            return None

        if "weight_mask" in batch:
            return batch["weight_mask"].to(latents.device)

        if "pixel_values" in batch:
            pv = batch["pixel_values"].to(latents.device).float()
            pv_01 = (pv + 1.0) / 2.0
            lum = 0.299 * pv_01[:, 0] + 0.587 * pv_01[:, 1] + 0.114 * pv_01[:, 2]
            return F.interpolate(
                lum.unsqueeze(1), size=latents.shape[-2:],
                mode="bilinear", align_corners=False,
            )
        return None

    def _build_loss_weights(self, latents, weight_mask, padding_mask=None) -> torch.Tensor | None:
        """组合空间权重、通道权重和 padding 掩码为 (B, C, H, W) 张量。

        padding_mask: (B, 1, H, W)，内容区域=1 / padding=0。
        当存在 padding_mask 时，padding 区域的权重被置零，loss 不计算黑边部分。
        """
        has_spatial = self.spatial_weight_scale > 0 and weight_mask is not None
        has_channel = self.channel_0_weight != 1.0
        has_padding = padding_mask is not None

        if not has_spatial and not has_channel and not has_padding:
            return None

        device = latents.device
        bsz, C, H, W = latents.shape

        if has_spatial:
            binary_mask = (weight_mask > self.spatial_weight_threshold).float()
            spatial_w = 1.0 + self.spatial_weight_scale * binary_mask
        else:
            spatial_w = torch.ones(1, 1, 1, 1, device=device)

        if has_channel:
            ch_w = torch.ones(1, C, 1, 1, device=device)
            ch_w[:, 0] = self.channel_0_weight
        else:
            ch_w = torch.ones(1, 1, 1, 1, device=device)

        combined = spatial_w * ch_w
        if has_padding:
            combined = combined * padding_mask
        return combined

    def _training_step_ddpm(self, latents, noise, bsz, prompt_embeds, attention_mask, weight_mask=None, padding_mask=None) -> torch.Tensor:
        """DDPM 训练步: scheduler.add_noise → epsilon/v-prediction → weighted MSE loss。"""
        if self.noise_offset > 0:
            noise = noise + self.noise_offset * torch.randn(
                latents.shape[0], latents.shape[1], 1, 1,
                device=latents.device, dtype=latents.dtype,
            )

        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,),
            device=latents.device,
        ).long()

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        model_output = self.transformer(
            hidden_states=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=attention_mask,
        ).sample

        if model_output.shape[1] != latents.shape[1]:
            model_output, _ = model_output.chunk(2, dim=1)

        if self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            target = noise

        combined_weights = self._build_loss_weights(latents, weight_mask, padding_mask)

        return self.compute_loss(
            model_output, target, timesteps,
            snr_cache=self._snr_cache,
            min_snr_gamma=self.min_snr_gamma,
            prediction_type=self.noise_scheduler.config.prediction_type,
            spatial_weights=combined_weights,
        )

    @torch.no_grad()
    def _run_validation(self, val_loop: ValidationLoop, val_gt_images=None):
        """执行验证生成 + 分布式 FID 计算。

        根据 noise_paradigm 选择推理 scheduler:
          - flow_matching: FlowMatchEulerDiscreteScheduler
          - ddpm: DPMSolverMultistepScheduler

        Args:
            val_gt_images: 训练集 ground truth 图像列表（与 prompts 一一对应），
                           传入后每次验证自动生成 [生成图 | 训练原图] 对比图。
        """
        from diffusers import PixArtSigmaPipeline

        self.transformer.eval()

        unwrapped_transformer = self.accelerator.unwrap_model(self.transformer)
        if hasattr(unwrapped_transformer, "_orig_mod"):
            unwrapped_transformer = unwrapped_transformer._orig_mod

        if self.noise_paradigm == "flow_matching":
            from diffusers import FlowMatchEulerDiscreteScheduler
            inference_scheduler = FlowMatchEulerDiscreteScheduler.from_config(
                self.noise_scheduler.config
            )
            from models.model_loader import patch_fm_scheduler_for_pipeline
            patch_fm_scheduler_for_pipeline(inference_scheduler)
        else:
            from diffusers import DPMSolverMultistepScheduler
            inference_scheduler = DPMSolverMultistepScheduler.from_config(
                self.noise_scheduler.config,
                algorithm_type="sde-dpmsolver++",
                use_karras_sigmas=True,
            )

        pipeline = PixArtSigmaPipeline(
            vae=self.vae,
            transformer=unwrapped_transformer,
            text_encoder=None,
            tokenizer=None,
            scheduler=inference_scheduler,
        )

        dev = self.accelerator.device
        pipeline_kwargs_override = None
        if hasattr(self, "_cached_val_prompt_embeds_list"):
            pipeline_kwargs_override = [
                {k: v.to(dev) for k, v in d.items()}
                for d in self._cached_val_prompt_embeds_list
            ]
        elif hasattr(self, "_cached_prompt_embeds"):
            pipeline_kwargs_override = {
                "prompt_embeds": self._cached_prompt_embeds.to(dev),
                "negative_prompt_embeds": self._cached_negative_prompt_embeds.to(dev),
                "prompt_attention_mask": self._cached_prompt_attention_mask.to(dev),
                "negative_prompt_attention_mask": self._cached_negative_prompt_attention_mask.to(dev),
            }

        pipeline.set_progress_bar_config(disable=True)

        # img2img: 需要支持自定义 timesteps 的 scheduler
        # DPMSolverMultistep(karras) 不支持 timesteps 参数，img2img 改用 EulerDiscrete
        img2img_data = None
        img2img_sched = None
        if val_loop.img2img_strengths and val_gt_images:
            if self.noise_paradigm == "flow_matching":
                from diffusers import FlowMatchEulerDiscreteScheduler
                img2img_sched = FlowMatchEulerDiscreteScheduler.from_config(
                    self.noise_scheduler.config
                )
            else:
                from diffusers import EulerDiscreteScheduler
                img2img_sched = EulerDiscreteScheduler.from_config(
                    self.noise_scheduler.config
                )

            img2img_data = []
            for strength in val_loop.img2img_strengths:
                latents, timesteps = self._prepare_img2img_latents(
                    scheduler=img2img_sched,
                    images=val_gt_images,
                    strength=strength,
                    num_inference_steps=val_loop.num_inference_steps,
                    device=self.accelerator.device,
                    seed=val_loop.seed,
                )
                img2img_data.append((strength, latents, timesteps))
            logger.info(f"img2img 数据已准备：{[s for s,_,_ in img2img_data]} strengths × {len(val_gt_images)} 图")

        val_loop.run(
            pipeline,
            self.global_step,
            self.tb_logger,
            device=self.accelerator.device,
            accelerator=self.accelerator,
            pipeline_kwargs_override=pipeline_kwargs_override,
            ground_truth_images=val_gt_images,
            img2img_data=img2img_data,
            img2img_scheduler=img2img_sched,
        )
        del pipeline

        self.transformer.train()

    @torch.no_grad()
    def _prepare_img2img_latents(
        self,
        scheduler,
        images: list,
        strength: float,
        num_inference_steps: int,
        device: torch.device,
        seed: int,
    ):
        """将训练图的 clean latent 加噪，供 img2img 验证使用。

        根据 noise_paradigm 选择加噪方式:
          - flow_matching: 线性插值 z_noisy = t_start * z_clean + (1-t_start) * noise
          - ddpm: scheduler.add_noise(z_clean, noise, t_start_timestep)
        """
        scheduler.set_timesteps(num_inference_steps, device=device)
        all_timesteps = scheduler.timesteps

        start_idx = max(int(num_inference_steps * (1.0 - strength)), 1)
        start_idx = min(start_idx, len(all_timesteps) - 1)
        img2img_timesteps = all_timesteps[start_idx:].tolist()

        cache_dir = Path(self._get_latent_cache_dir())
        cache_files = sorted(cache_dir.glob("*.pt")) if cache_dir.exists() else []

        latents_list = []
        for i in range(len(images)):
            if i < len(cache_files):
                cached = torch.load(cache_files[i], map_location="cpu", weights_only=False)
                z_clean = cached["latent"].unsqueeze(0).to(device, dtype=torch.float32)
            else:
                import torchvision.transforms.functional as TF
                img_tensor = TF.to_tensor(images[i]).unsqueeze(0).to(device, dtype=self.vae.dtype)
                img_tensor = img_tensor * 2.0 - 1.0
                z_clean = self.vae.encode(img_tensor).latent_dist.sample()
                z_clean = z_clean * self.vae.config.scaling_factor

            generator = torch.Generator(device=device).manual_seed(seed + i)
            noise = torch.randn_like(z_clean, generator=generator)

            if self.noise_paradigm == "flow_matching":
                t_start = 1.0 - strength
                z_noisy = t_start * z_clean + (1.0 - t_start) * noise
            else:
                t_start_ts = all_timesteps[start_idx]
                z_noisy = self.noise_scheduler.add_noise(
                    z_clean, noise, torch.tensor([t_start_ts], device=device).long(),
                )

            latents_list.append(z_noisy.to(dtype=torch.bfloat16))

        return latents_list, img2img_timesteps

    def _maybe_save_best(self):
        """若 EMA loss 为历史最低且距上次保存已过最小间隔，保存 best checkpoint。"""
        if self._ema_loss is None:
            return
        steps_since_last = self.global_step - self._last_best_save_step
        if steps_since_last < self._min_best_save_interval:
            return
        if self._ema_loss < self._best_ema_loss:
            prev_best = self._best_ema_loss
            self._best_ema_loss = self._ema_loss
            self._last_best_save_step = self.global_step
            if self.accelerator.is_main_process:
                best_dir = Path(self.training_cfg.get("output_dir", "./outputs")) / "best_checkpoint"
                if best_dir.exists():
                    import shutil
                    shutil.rmtree(best_dir)
                best_dir.mkdir(parents=True, exist_ok=True)

                unwrapped = self.accelerator.unwrap_model(self.transformer)
                state_dict = self.accelerator.get_state_dict(self.transformer)
                unwrapped.save_pretrained(str(best_dir / "transformer"), state_dict=state_dict)

                import json
                with open(best_dir / "training_state.json", "w") as f:
                    json.dump({
                        "step": self.global_step,
                        "epoch": self.global_epoch,
                        "ema_loss": self._ema_loss,
                    }, f, indent=2)

                logger.info(
                    f"Best checkpoint saved: step={self.global_step}, "
                    f"ema_loss={self._ema_loss:.6f} (prev best={prev_best:.6f})"
                )

    def _save_checkpoint(self, optimizer, lr_scheduler):
        """保存完整 transformer 权重检查点（仅主进程）。"""
        if not self.accelerator.is_main_process:
            return

        self.ckpt_manager.save(
            step=self.global_step,
            global_epoch=self.global_epoch,
            transformer=self.transformer,
            accelerator=self.accelerator,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            seed=self.training_cfg.get("seed", 42),
            is_lora=False,
        )
