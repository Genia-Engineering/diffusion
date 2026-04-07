"""Sana 0.6B 全参数微调训练器 — 支持 Flow Matching 和 DDPM 双噪声范式。

通过配置项 noise_paradigm 切换训练范式:

1. Flow Matching (noise_paradigm="flow_matching"):
   sigma 参数化, 与 DPMSolverMultistepScheduler (flow_prediction) 对齐
   - 前向插值: x_t = t*x_1 + (1-t)*x_0, sigma = 1-t
   - 预测目标: velocity = x_0 - x_1 (noise - clean)
   - 时间步采样: t ~ Logit-Normal 或 Uniform(0,1), timestep = (1-t)*1000

2. DDPM (noise_paradigm="ddpm"):
   epsilon/v-prediction, 与 DDPMScheduler 对齐

权重冻结策略:
  - VAE (AutoencoderDC): 全冻结，预缓存 latent 后卸载 encoder
  - Gemma2 Text Encoder: 全冻结，预缓存嵌入后完全卸载
  - SanaTransformer2DModel: 全部参数可训练
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
from data.dataset import SanaCachedLatentDataset
from models.model_loader import load_sana_components
from utils.fid import FIDCalculator
from utils.memory import apply_memory_optimizations
from utils.validation import ValidationLoop
from .base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class SanaTrainer(BaseTrainer):
    """Sana 0.6B 全参数微调训练器。"""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.model_type = config.model.model_type  # "sana"

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

        self.scheduler_shift: float = float(self.training_cfg.get("scheduler_shift", 3.0))

        self._load_models()
        self._freeze_parameters()

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
        flow_shift = float(self.training_cfg.get("scheduler_shift", 3.0))

        components = load_sana_components(
            model_path,
            weights_dir=weights_dir,
            flow_shift=flow_shift,
        )
        self.vae = components["vae"]
        self.transformer = components["transformer"]
        self.text_encoder = components["text_encoder"]
        self.tokenizer = components["tokenizer"]
        self.noise_scheduler = components["noise_scheduler"]

        if self.noise_paradigm == "flow_matching":
            logger.info("噪声范式: Flow Matching (Rectified Flow, DPMSolver flow_prediction)")
        else:
            override_pred_type = self.training_cfg.get("prediction_type", None)
            if override_pred_type and override_pred_type != self.noise_scheduler.config.prediction_type:
                self.noise_scheduler.register_to_config(prediction_type=override_pred_type)
                logger.info(f"DDPM prediction_type overridden to '{override_pred_type}'")
            logger.info(
                f"噪声范式: DDPM (prediction_type={self.noise_scheduler.config.prediction_type})"
            )

    def _freeze_parameters(self):
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.transformer.requires_grad_(True)
        self.print_trainable_params(self.transformer)

    def _text_embed_cache_exists(self) -> bool:
        import hashlib
        caption = self.config.data.get("caption", "")
        neg_prompt = self.config.get("validation", {}).get("negative_prompt", "")
        data_dir = self.config.data.get("train_data_dir", "./data")
        embed_hash = hashlib.md5(f"{caption}||{neg_prompt}".encode()).hexdigest()[:8]
        cache_file = Path(data_dir) / "text_embed_cache" / f"embeds_{self.model_type}_{embed_hash}.pt"
        return cache_file.exists()

    @torch.no_grad()
    def _encode_null_prompt(self):
        device = self.text_encoder.device
        max_seq_len = 300
        tokenized = self.tokenizer(
            "", padding="max_length", truncation=True,
            max_length=max_seq_len, return_tensors="pt",
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
            dataset = SanaCachedLatentDataset(
                data_dir=data_cfg.train_data_dir,
                cache_dir=cache_dir,
                tokenizer=self.tokenizer,
                resolution=resolution,
                random_flip=data_cfg.get("random_flip", True),
                fixed_caption=fixed_caption,
                text_embed_cache_dir=text_embed_cache_dir,
            )
        else:
            from data.dataset import PixArtSigmaDataset
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
        return [dataset.get_pil_image(i) for i in indices]

    def _load_val_ground_truth_images(self, n: int) -> list:
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
        return [dataset.get_pil_image(i) for i in indices]

    def train(self):
        """Sana 0.6B 全参数微调主训练循环。"""
        cache_latents = self.training_cfg.get("cache_latents", False)
        cache_text_embeddings = self.training_cfg.get("cache_text_embeddings", False)

        # ── 阶段1：VAE latent 预计算 ────────────────────────────────────────
        self.vae.to(self.accelerator.device, dtype=torch.float32)
        if cache_latents:
            self._precompute_latents_distributed(self._get_latent_cache_dir())

        # ── 阶段2：Gemma2 文本嵌入预计算 ────────────────────────────────────
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
        else:
            self.text_encoder.to(self.accelerator.device)

        if self.caption_dropout_rate > 0 and not hasattr(self, "_cached_negative_prompt_embeds"):
            self._encode_null_prompt()

        # ── 阶段3：准备训练组件 ──────────────────────────────────────────────
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
                )
            else:
                logger.info("Compiling transformer with torch.compile (mode=max-autotune-no-cudagraphs)...")
                self.transformer = torch.compile(
                    self.transformer, mode="max-autotune-no-cudagraphs"
                )

        self.transformer, optimizer, dataloader = self.accelerator.prepare(
            self.transformer, optimizer, dataloader
        )

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
            logger.info(f"Resumed from step {self.global_step}")

        # FID
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

        val_gt_images = None
        if self.accelerator.is_main_process:
            val_gt_images = self._load_val_ground_truth_images(len(val_prompts))
            logger.info(f"验证 ground truth 加载完成：{len(val_gt_images)} 张训练图")

        self._ema_loss = None
        self._ema_decay = 0.99
        self._best_ema_loss = float("inf")
        self._last_best_save_step = 0
        self._min_best_save_interval = max(save_steps // 4, 50)

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

                    self._update_ema_loss(current_loss)
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
        if "latents" in batch:
            latents = batch["latents"].to(self.accelerator.device)
        else:
            pixel_values = batch["pixel_values"].to(dtype=self.vae.dtype)
            with torch.no_grad():
                enc_out = self.vae.encode(pixel_values)
                if hasattr(enc_out, "latent_dist"):
                    latents = enc_out.latent_dist.sample()
                else:
                    latents = enc_out.latent
                latents = latents * self.vae.config.scaling_factor

        bsz = latents.shape[0]
        noise = torch.randn_like(latents)

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
        """Flow Matching 训练步: Rectified Flow velocity prediction。

        Sana 使用 DPMSolverMultistepScheduler (flow_shift=3.0) 的 shifted sigma schedule:
          sigma = shift * u / (1 + (shift - 1) * u)
        训练时必须使用相同的 shift 以保持 timestep-to-noise-level 映射一致。

        SanaTransformer2DModel 输出 32 通道（与输入 latent 通道一致），
        不需要像 PixArt 那样 chunk(2, dim=1) 拆分。
        """
        if self.timestep_sampling == "logit_normal":
            u = torch.sigmoid(
                self.logit_mean + self.logit_std * torch.randn(bsz, device=latents.device)
            )
        else:
            u = torch.rand(bsz, device=latents.device)

        shift = self.scheduler_shift
        sigma = shift * u / (1.0 + (shift - 1.0) * u)

        sigma_expanded = sigma.view(-1, 1, 1, 1)
        noisy_latents = (1.0 - sigma_expanded) * latents + sigma_expanded * noise
        timesteps_scaled = sigma * 1000.0

        model_output = self.transformer(
            hidden_states=noisy_latents,
            timestep=timesteps_scaled,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=attention_mask,
        ).sample

        target = noise - latents

        if padding_mask is not None:
            per_pixel = F.mse_loss(model_output.float(), target.float(), reduction="none")
            n_content = padding_mask.sum() * per_pixel.shape[1]
            return (per_pixel * padding_mask).sum() / n_content.clamp(min=1)
        return F.mse_loss(model_output.float(), target.float())

    def _get_weight_mask(self, batch, latents) -> torch.Tensor | None:
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
        from diffusers import SanaPipeline

        self.transformer.eval()

        unwrapped_transformer = self.accelerator.unwrap_model(self.transformer)
        if hasattr(unwrapped_transformer, "_orig_mod"):
            unwrapped_transformer = unwrapped_transformer._orig_mod

        from diffusers import DPMSolverMultistepScheduler
        inference_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            prediction_type="flow_prediction",
            use_flow_sigmas=True,
            flow_shift=float(self.training_cfg.get("scheduler_shift", 3.0)),
            algorithm_type="dpmsolver++",
            solver_order=2,
            solver_type="midpoint",
            lower_order_final=True,
            final_sigmas_type="zero",
        )

        pipeline = SanaPipeline(
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

        img2img_data = None
        img2img_sched = None

        val_loop.run(
            pipeline,
            self.global_step,
            self.tb_logger,
            device=dev,
            accelerator=self.accelerator,
            pipeline_kwargs_override=pipeline_kwargs_override,
            ground_truth_images=val_gt_images,
            img2img_data=img2img_data,
            img2img_scheduler=img2img_sched,
        )
        del pipeline

        self.transformer.train()

    def _update_ema_loss(self, loss: float):
        if self._ema_loss is None:
            self._ema_loss = loss
        else:
            self._ema_loss = self._ema_decay * self._ema_loss + (1 - self._ema_decay) * loss

    def _maybe_save_best(self):
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
