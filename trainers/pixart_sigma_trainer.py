"""PixArt-Sigma 全参数微调训练器 — Rectified Flow (Flow Matching) 范式。

训练范式: Flow Matching (sigma 参数化, 与 FlowMatchEulerDiscreteScheduler 对齐)
  - 前向插值: x_t = (1-sigma)*x_1 + sigma*x_0 ≡ t*x_1 + (1-t)*x_0
    其中 x_1 为 clean latent，x_0 ~ N(0, I)，sigma = 1-t
  - 预测目标: dx/d(sigma) = x_0 - x_1 (sigma 参数化 velocity)
  - 损失函数: MSE(model_output, x_0 - x_1)
  - 时间步采样: t ~ Logit-Normal(mu, std) 或 Uniform(0, 1)
  - 模型输入 timestep: sigma * 1000 = (1-t) * 1000

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

        # Flow Matching 时间步采样参数
        self.timestep_sampling: str = self.training_cfg.get("timestep_sampling", "logit_normal")
        self.logit_mean: float = float(self.training_cfg.get("logit_mean", 0.0))
        self.logit_std: float = float(self.training_cfg.get("logit_std", 1.0))

        self._load_models()
        self._freeze_parameters()

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

        components = load_pixart_sigma_components(
            model_path, weights_dir=weights_dir, scheduler_shift=scheduler_shift,
        )
        self.vae = components["vae"]
        self.transformer = components["transformer"]
        self.text_encoder = components["text_encoder"]
        self.tokenizer = components["tokenizer"]
        self.noise_scheduler = components["noise_scheduler"]

    def _freeze_parameters(self):
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.transformer.requires_grad_(True)
        self.print_trainable_params(self.transformer)

    def _build_dataloader(self) -> DataLoader:
        data_cfg = self.config.data
        resolution = data_cfg.get("resolution", 1024)
        batch_size = self.training_cfg.get("train_batch_size", 2)
        fixed_caption = data_cfg.get("caption", "")
        cache_latents = self.training_cfg.get("cache_latents", False)

        if cache_latents:
            cache_dir = self._get_latent_cache_dir()
            dataset = PixArtSigmaCachedLatentDataset(
                data_dir=data_cfg.train_data_dir,
                cache_dir=cache_dir,
                tokenizer=self.tokenizer,
                resolution=resolution,
                random_flip=data_cfg.get("random_flip", True),
                fixed_caption=fixed_caption,
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

        if cache_latents or data_cfg.get("use_aspect_ratio_bucketing", True):
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
        self.text_encoder.to(self.accelerator.device)
        if cache_text_embeddings:
            self._precompute_text_embeddings()
            # _precompute_text_embeddings 内部会 del self.text_encoder 并 empty_cache

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
                    current_lr = lr_scheduler.get_last_lr()[0]
                    self.log_step(loss.item(), current_lr, grad_norm)

                    progress_bar.update(1)
                    progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}")

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
        """单步 Flow Matching 训练: 采样 t → 线性插值 x_t → 预测 velocity → MSE loss。

        数学公式 (Rectified Flow, sigma 参数化以匹配 FlowMatchEulerDiscreteScheduler):
          x_1 = latent             clean latent (已含 VAE scaling_factor)
          x_0 ~ N(0, I)           噪声
          t ~ Logit-Normal        采样时刻, t=0 → 噪声, t=1 → clean
          sigma = 1 - t           scheduler 内部参数化, sigma=1 → 噪声, sigma=0 → clean
          x_sigma = (1-sigma)*x_1 + sigma*x_0   等价于 t*x_1 + (1-t)*x_0
          target = x_0 - x_1      sigma 参数化下的 velocity = dx/d(sigma)

        注意: FlowMatchEulerDiscreteScheduler 的 step 公式为
          prev = sample + model_output * (sigma_next - sigma)
        推理时 sigma 递减 (sigma_next < sigma)，要求 model_output = x_0 - x_1
        才能让 sample 向 x_1 (clean) 方向移动。
        """
        # ── 1. 获取 clean latent x_1 ──
        if "latents" in batch:
            latents = batch["latents"].to(self.accelerator.device)
        else:
            pixel_values = batch["pixel_values"].to(dtype=self.vae.dtype)
            with torch.no_grad():
                latents = self.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor

        bsz = latents.shape[0]

        # ── 2. 采样噪声 x_0 ~ N(0, I) ──
        noise = torch.randn_like(latents)

        # ── 3. 采样连续时间步 t ∈ (0, 1)，t=0 → 噪声, t=1 → clean ──
        if self.timestep_sampling == "logit_normal":
            t = torch.sigmoid(
                self.logit_mean + self.logit_std * torch.randn(bsz, device=latents.device)
            )
        else:
            t = torch.rand(bsz, device=latents.device)

        # ── 4. 线性插值: x_t = t * x_1 + (1-t) * x_0 ──
        t_expanded = t.view(-1, 1, 1, 1)
        noisy_latents = t_expanded * latents + (1.0 - t_expanded) * noise

        # ── 5. 获取文本条件 ──
        if hasattr(self, "_cached_prompt_embeds"):
            prompt_embeds = self._cached_prompt_embeds.expand(bsz, -1, -1).to(latents.device)
            attention_mask = self._cached_prompt_attention_mask.expand(bsz, -1).to(latents.device)
        else:
            input_ids = batch["input_ids"].to(self.accelerator.device)
            attn_mask = batch["attention_mask"].to(self.accelerator.device)
            with torch.no_grad():
                prompt_embeds = self.text_encoder(input_ids, attention_mask=attn_mask)[0]
            attention_mask = attn_mask

        # ── 6. Transformer 前向传播 ──
        # 转换为 scheduler 的 sigma 参数化: sigma = 1 - t, timestep = sigma * 1000
        # 推理时 scheduler 给纯噪声传 timestep≈1000，给干净数据传 timestep≈0
        timesteps_scaled = (1.0 - t) * 1000.0

        model_output = self.transformer(
            hidden_states=noisy_latents,
            timestep=timesteps_scaled,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=attention_mask,
        ).sample

        # PixArt-Sigma 输出 out_channels=2*in_channels（learned variance），
        # 只取前半部分作为 velocity 预测，丢弃方差部分。
        if model_output.shape[1] != latents.shape[1]:
            model_output, _ = model_output.chunk(2, dim=1)

        # ── 7. 计算目标向量场和 MSE 损失 ──
        # sigma 参数化下的 velocity: dx/d(sigma) = x_0 - x_1 = noise - latents
        target = noise - latents

        loss = F.mse_loss(model_output.float(), target.float())
        return loss

    @torch.no_grad()
    def _run_validation(self, val_loop: ValidationLoop, val_gt_images=None):
        """执行验证生成 + 分布式 FID 计算（Flow Matching Euler ODE 推理）。

        Args:
            val_gt_images: 训练集 ground truth 图像列表（与 prompts 一一对应），
                           传入后每次验证自动生成 [生成图 | 训练原图] 对比图。
        """
        from diffusers import FlowMatchEulerDiscreteScheduler, PixArtSigmaPipeline

        self.transformer.eval()

        unwrapped_transformer = self.accelerator.unwrap_model(self.transformer)
        if hasattr(unwrapped_transformer, "_orig_mod"):
            unwrapped_transformer = unwrapped_transformer._orig_mod

        inference_scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            self.noise_scheduler.config
        )
        from models.model_loader import patch_fm_scheduler_for_pipeline
        patch_fm_scheduler_for_pipeline(inference_scheduler)

        pipeline = PixArtSigmaPipeline(
            vae=self.vae,
            transformer=unwrapped_transformer,
            text_encoder=None,
            tokenizer=None,
            scheduler=inference_scheduler,
        )

        pipeline_kwargs_override = None
        if hasattr(self, "_cached_prompt_embeds"):
            dev = self.accelerator.device
            pipeline_kwargs_override = {
                "prompt_embeds": self._cached_prompt_embeds.to(dev),
                "negative_prompt_embeds": self._cached_negative_prompt_embeds.to(dev),
                "prompt_attention_mask": self._cached_prompt_attention_mask.to(dev),
                "negative_prompt_attention_mask": self._cached_negative_prompt_attention_mask.to(dev),
            }

        pipeline.set_progress_bar_config(disable=True)

        # img2img: Flow Matching 使用线性插值加噪，scheduler 统一为 FlowMatchEuler
        img2img_data = None
        if val_loop.img2img_strengths and val_gt_images:
            img2img_data = []
            for strength in val_loop.img2img_strengths:
                latents, timesteps = self._prepare_img2img_latents(
                    scheduler=inference_scheduler,
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
            img2img_scheduler=None,
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
        """将训练图的 clean latent 用 Flow Matching 线性插值加噪，供 img2img 验证使用。

        Flow Matching img2img:
          t_start = 1.0 - strength  (strength=0.3 → t_start=0.7, 保留 70% 原图结构)
          z_noisy = t_start * z_clean + (1 - t_start) * noise

        推理从 t_start 开始，沿 ODE 求解到 t=1.0（clean）。
        """
        scheduler.set_timesteps(num_inference_steps, device=device)
        all_timesteps = scheduler.timesteps

        # 根据 strength 截断 timestep 序列：保留后 strength 比例的步数
        start_idx = max(int(num_inference_steps * (1.0 - strength)), 1)
        start_idx = min(start_idx, len(all_timesteps) - 1)
        img2img_timesteps = all_timesteps[start_idx:].tolist()

        # t_start: 连续时间，对应 scheduler 的 sigma
        t_start = 1.0 - strength

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

            # Flow Matching 线性插值加噪
            generator = torch.Generator(device=device).manual_seed(seed + i)
            noise = torch.randn_like(z_clean, generator=generator)
            z_noisy = t_start * z_clean + (1.0 - t_start) * noise

            latents_list.append(z_noisy.to(dtype=torch.bfloat16))

        return latents_list, img2img_timesteps

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
