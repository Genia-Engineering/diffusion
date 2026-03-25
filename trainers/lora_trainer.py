"""LoRA 训练器 — 完整训练循环，支持 SD1.5 和 SDXL。

权重冻结策略:
  - VAE: 全冻结
  - UNet 原始权重: 全冻结
  - Text Encoder(s) 原始权重: 全冻结
  - LoRA 参数 (lora_A, lora_B): 可训练
"""

import contextlib
import logging
import math
import os
from pathlib import Path

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.buckets import BucketManager, BucketSampler
from data.dataset import (
    SD15CachedLatentDataset,
    SD15Dataset,
    SDXLCachedLatentDataset,
    SDXLDataset,
)
from models.lora import LoRAInjector, get_lora_params
from models.model_loader import load_sd15_components, load_sdxl_components
from utils.fid import FIDCalculator
from utils.memory import apply_memory_optimizations, compute_grad_norm
from utils.validation import ValidationLoop
from .base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class LoRATrainer(BaseTrainer):
    """LoRA 微调训练器。"""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.model_type = config.model.model_type
        self.lora_cfg = config.lora
        self.use_qlora = self.lora_cfg.get("use_qlora", False)

        # 损失函数增强超参（在 _load_models 之后才能拿到 noise_scheduler，故先存原始值）
        self.noise_offset: float = float(self.training_cfg.get("noise_offset", 0.0))
        self.min_snr_gamma: float = float(self.training_cfg.get("min_snr_gamma", 0.0))

        self._load_models()
        self._inject_lora()
        self._freeze_parameters()

        # 预计算 SNR 查找表（仅当 min_snr_gamma > 0 时才需要）
        self._snr_cache: torch.Tensor | None = None
        if self.min_snr_gamma > 0:
            alphas_cumprod = self.noise_scheduler.alphas_cumprod  # (T,) float32
            self._snr_cache = alphas_cumprod / (1.0 - alphas_cumprod)  # SNR_t = ᾱ_t / (1 - ᾱ_t)

        # QLoRA 量化权重与 channels_last 内存格式不兼容，需要禁用
        enable_channels_last = self.training_cfg.get("enable_channels_last", True)
        if self.use_qlora and enable_channels_last:
            logger.info("QLoRA 模式：跳过 UNet channels_last（与 NF4 量化不兼容）")
            enable_channels_last = False

        apply_memory_optimizations(
            unet=self.unet,
            vae=self.vae,
            text_encoder=self.text_encoder,
            text_encoder_2=getattr(self, "text_encoder_2", None),
            enable_gradient_checkpointing=self.training_cfg.get("gradient_checkpointing", True),
            attention_backend=self.training_cfg.get("attention_backend", "sdpa"),
            enable_channels_last=enable_channels_last,
        )

    def _build_quantization_config(self):
        """构建 bitsandbytes NF4 量化配置（QLoRA 模式）。"""
        compute_dtype_str = self.lora_cfg.get("qlora_compute_dtype", "bf16")
        compute_dtype = torch.bfloat16 if compute_dtype_str == "bf16" else torch.float16

        from diffusers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        logger.info(
            f"QLoRA 量化配置: NF4 + double_quant, compute_dtype={compute_dtype}"
        )
        return quantization_config

    def _load_models(self):
        model_path = self.config.model.pretrained_model_name_or_path
        weights_dir = self.config.model.get("weights_dir", None)

        quantization_config = None
        if self.use_qlora:
            quantization_config = self._build_quantization_config()

        if self.model_type == "sdxl":
            components = load_sdxl_components(
                model_path, weights_dir=weights_dir,
                unet_quantization_config=quantization_config,
            )
            self.text_encoder_2 = components["text_encoder_2"]
            self.tokenizer_2 = components["tokenizer_2"]
        else:
            components = load_sd15_components(
                model_path, weights_dir=weights_dir,
                unet_quantization_config=quantization_config,
            )
            self.text_encoder_2 = None
            self.tokenizer_2 = None

        self.vae = components["vae"]
        self.unet = components["unet"]
        self.text_encoder = components["text_encoder"]
        self.tokenizer = components["tokenizer"]
        self.noise_scheduler = components["noise_scheduler"]

    def _inject_lora(self):
        rank = self.lora_cfg.get("rank", 16)
        alpha = self.lora_cfg.get("alpha", 16.0)
        target_modules = list(self.lora_cfg.get("target_modules", LoRAInjector.DEFAULT_TARGET_MODULES))

        injected_unet = LoRAInjector.inject_unet(self.unet, rank, alpha, target_modules)
        logger.info(f"LoRA injected into UNet: {len(injected_unet)} layers")

        self.train_te = self.lora_cfg.get("train_text_encoder", False)
        if self.train_te:
            injected_te = LoRAInjector.inject_text_encoder(self.text_encoder, rank, alpha)
            logger.info(f"LoRA injected into TextEncoder: {len(injected_te)} layers")

        self.train_te2 = self.lora_cfg.get("train_text_encoder_2", False)
        if self.train_te2 and self.text_encoder_2 is not None:
            injected_te2 = LoRAInjector.inject_text_encoder(self.text_encoder_2, rank, alpha)
            logger.info(f"LoRA injected into TextEncoder2: {len(injected_te2)} layers")

    def _freeze_parameters(self):
        self.vae.requires_grad_(False)

        for param in self.unet.parameters():
            param.requires_grad = False
        for module in self.unet.modules():
            from models.lora import LoRALinear
            if isinstance(module, LoRALinear):
                module.lora_A.requires_grad_(True)
                module.lora_B.requires_grad_(True)

        for param in self.text_encoder.parameters():
            param.requires_grad = False
        if self.train_te:
            for module in self.text_encoder.modules():
                from models.lora import LoRALinear
                if isinstance(module, LoRALinear):
                    module.lora_A.requires_grad_(True)
                    module.lora_B.requires_grad_(True)

        if self.text_encoder_2 is not None:
            for param in self.text_encoder_2.parameters():
                param.requires_grad = False
            if self.train_te2:
                for module in self.text_encoder_2.modules():
                    from models.lora import LoRALinear
                    if isinstance(module, LoRALinear):
                        module.lora_A.requires_grad_(True)
                        module.lora_B.requires_grad_(True)

        self.print_trainable_params(self.unet, self.text_encoder, self.text_encoder_2)

    def _log_bucket_stats(self, bucket_to_indices: dict) -> None:
        """打印各宽高比桶的图像数量分布。"""
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

        if cache_latents:
            cache_dir = self._get_latent_cache_dir()
            if self.model_type == "sdxl":
                dataset = SDXLCachedLatentDataset(
                    data_dir=data_cfg.train_data_dir,
                    cache_dir=cache_dir,
                    tokenizer_1=self.tokenizer,
                    tokenizer_2=self.tokenizer_2,
                    resolution=resolution,
                    random_flip=data_cfg.get("random_flip", True),
                    fixed_caption=fixed_caption,
                )
            else:
                dataset = SD15CachedLatentDataset(
                    data_dir=data_cfg.train_data_dir,
                    cache_dir=cache_dir,
                    tokenizer=self.tokenizer,
                    resolution=resolution,
                    random_flip=data_cfg.get("random_flip", True),
                    fixed_caption=fixed_caption,
                )
        elif self.model_type == "sdxl":
            dataset = SDXLDataset(
                data_dir=data_cfg.train_data_dir,
                tokenizer_1=self.tokenizer,
                tokenizer_2=self.tokenizer_2,
                resolution=resolution,
                center_crop=data_cfg.get("center_crop", False),
                random_flip=data_cfg.get("random_flip", True),
                fixed_caption=fixed_caption,
            )
        else:
            dataset = SD15Dataset(
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

    def _sample_real_images(self, n: int) -> list:
        """从训练数据集随机采样 n 张原始 PIL 图像，供 FID 计算使用。"""
        import random

        data_cfg = self.config.data
        resolution = data_cfg.get("resolution", 512 if self.model_type == "sd15" else 1024)
        fixed_caption = data_cfg.get("caption", "")

        if self.model_type == "sdxl":
            dataset = SDXLDataset(
                data_dir=data_cfg.train_data_dir,
                tokenizer_1=self.tokenizer,
                tokenizer_2=self.tokenizer_2,
                resolution=resolution,
                center_crop=data_cfg.get("center_crop", False),
                random_flip=False,
                fixed_caption=fixed_caption,
            )
        else:
            dataset = SD15Dataset(
                data_dir=data_cfg.train_data_dir,
                tokenizer=self.tokenizer,
                resolution=resolution,
                center_crop=data_cfg.get("center_crop", False),
                random_flip=False,
                fixed_caption=fixed_caption,
            )

        indices = random.sample(range(len(dataset)), min(n, len(dataset)))
        images = [dataset.get_pil_image(i) for i in indices]
        return images

    def _encode_prompt_sd15(self, input_ids):
        """SD1.5 文本编码。"""
        encoder_output = self.text_encoder(input_ids, return_dict=False)
        return encoder_output[0]

    def _encode_prompt_sdxl(self, input_ids_1, input_ids_2):
        """SDXL 双文本编码器。"""
        encoder_output_1 = self.text_encoder(input_ids_1, output_hidden_states=True)
        hidden_states_1 = encoder_output_1.hidden_states[-2]

        encoder_output_2 = self.text_encoder_2(input_ids_2, output_hidden_states=True)
        hidden_states_2 = encoder_output_2.hidden_states[-2]
        pooled_prompt_embeds = encoder_output_2[0]

        prompt_embeds = torch.cat([hidden_states_1, hidden_states_2], dim=-1)
        return prompt_embeds, pooled_prompt_embeds

    def _compute_sdxl_time_ids(self, batch, bsz, device, dtype):
        """计算 SDXL 的 add_time_ids (B, 6): [orig_h, orig_w, crop_top, crop_left, tgt_h, tgt_w]。

        Dataset 已将三个字段存为 (B, 2) LongTensor（格式 [h, w]），直接拼接即可。
        bsz 由调用方显式传入，兼容 pixel_values 和 cached latents 两种 batch 格式。
        """
        bs = bsz
        default_hw = torch.tensor([[1024, 1024]], dtype=torch.long).expand(bs, -1)
        default_crop = torch.zeros(bs, 2, dtype=torch.long)

        original_size = batch.get("original_size", default_hw)   # (B, 2): [h, w]
        crop_top_left = batch.get("crop_top_left", default_crop)  # (B, 2): [top, left]
        target_size   = batch.get("target_size",   default_hw)    # (B, 2): [h, w]

        add_time_ids = torch.cat(
            [original_size, crop_top_left, target_size], dim=1
        )  # (B, 6)
        return add_time_ids.to(device=device, dtype=dtype)

    def train(self):
        """LoRA 主训练循环。"""
        cache_latents = self.training_cfg.get("cache_latents", False)

        # VAE 提前上 GPU：latent 预计算（如需）在 accelerator.prepare 之前完成
        self.vae.to(self.accelerator.device, dtype=torch.float32)

        # 分布式安全预计算：文件信号代替 NCCL barrier，避免 10 分钟超时崩溃
        if cache_latents:
            self._precompute_latents_distributed(self._get_latent_cache_dir())

        dataloader = self._build_dataloader()
        num_train_steps = self.training_cfg.get("num_train_steps", 5000)
        max_grad_norm = self.training_cfg.get("max_grad_norm", 1.0)
        validation_steps = self.training_cfg.get("validation_steps", 500)
        save_steps = self.training_cfg.get("save_steps", 500)

        unet_lora_params = get_lora_params(self.unet)
        te_lora_params = get_lora_params(self.text_encoder) if self.train_te else []
        if self.train_te2 and self.text_encoder_2 is not None:
            te_lora_params += get_lora_params(self.text_encoder_2)

        optimizer = self.setup_optimizer(
            trainable_params=unet_lora_params,
            text_encoder_params=te_lora_params if te_lora_params else None,
        )
        lr_scheduler = self.setup_lr_scheduler(optimizer, num_train_steps)

        # accelerate 封装
        models_to_prepare = [self.unet]
        if self.train_te:
            models_to_prepare.append(self.text_encoder)
        if self.train_te2 and self.text_encoder_2 is not None:
            models_to_prepare.append(self.text_encoder_2)

        prepared = self.accelerator.prepare(
            *models_to_prepare, optimizer, dataloader, lr_scheduler
        )
        idx = 0
        self.unet = prepared[idx]; idx += 1
        if self.train_te:
            self.text_encoder = prepared[idx]; idx += 1
        if self.train_te2 and self.text_encoder_2 is not None:
            self.text_encoder_2 = prepared[idx]; idx += 1
        optimizer = prepared[idx]; idx += 1
        dataloader = prepared[idx]; idx += 1
        lr_scheduler = prepared[idx]; idx += 1

        # VAE 已在 train() 开头移到 device，此处无需重复（保留文本编码器移动）
        if not self.train_te:
            self.text_encoder.to(self.accelerator.device)
        if self.text_encoder_2 is not None and not self.train_te2:
            self.text_encoder_2.to(self.accelerator.device)

        if (
            self.training_cfg.get("cache_text_embeddings", False)
            and not self.train_te
            and not self.train_te2
        ):
            self._precompute_text_embeddings()

        # 恢复训练
        resume_dir = self.training_cfg.get("resume_from_checkpoint", None)
        if resume_dir == "latest":
            resume_dir = self.ckpt_manager.get_latest_checkpoint()
        if resume_dir:
            state = self.ckpt_manager.load(
                resume_dir, self.unet, self.text_encoder,
                self.text_encoder_2 if self.model_type == "sdxl" else None,
                optimizer=optimizer, lr_scheduler=lr_scheduler,
            )
            self.global_step = state["step"]
            self.global_epoch = state["epoch"]
            logger.info(f"Resumed from step {self.global_step}")

        # ── FID 计算器初始化（所有 Rank 均创建，用于本地特征提取） ──────────────
        # 分布式 FID 策略：所有卡均匀生成图像并提取 DINOv2 特征，Rank 0 汇聚后计算。
        # - 所有 Rank 创建 FIDCalculator（用于调用 _extract_features）
        # - 仅 Rank 0 调用 update_real() 加载/缓存真实图像统计量（_real_mu / _real_sigma）
        # - 非主进程的 fid_calculator.is_ready() 返回 False，不会执行最终 FID 计算
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
                # 非主进程不需要读写缓存文件
                real_images_cache_path=cache_path if self.accelerator.is_main_process else None,
            )
            if self.accelerator.is_main_process:
                fid_num_real = fid_cfg.get("num_real_images", 512)
                real_images = self._sample_real_images(fid_num_real)
                fid_calculator.update_real(real_images)
                logger.info(f"FID 计算器就绪，真实图像数: {len(real_images)}")
            # 等待 Rank 0 完成真实图像特征加载/缓存，再开始训练
            self.accelerator.wait_for_everyone()
        # ────────────────────────────────────────────────────────────────────

        # Validation Loop 初始化
        fid_num_gen = val_cfg.get("fid", {}).get("num_gen_images", 256)
        val_loop = ValidationLoop(
            prompts=list(val_cfg.get("prompts", ["a test image"])),
            negative_prompt=val_cfg.get("negative_prompt", ""),
            num_inference_steps=val_cfg.get("num_inference_steps", 25),
            guidance_scale=val_cfg.get("guidance_scale", 7.5),
            seed=val_cfg.get("seed", 42),
            num_images_per_prompt=val_cfg.get("num_images_per_prompt", 1),
            save_dir=os.path.join(self.training_cfg.get("output_dir", "./outputs"), "samples"),
            fid_calculator=fid_calculator,
            fid_num_gen_images=fid_num_gen,
        )

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

        self.unet.train()
        if self.train_te:
            self.text_encoder.train()
        if self.train_te2 and self.text_encoder_2 is not None:
            self.text_encoder_2.train()

        for epoch in range(num_epochs):
            self.global_epoch = epoch
            for batch in dataloader:
                if self.global_step >= num_train_steps:
                    break

                with self.accelerator.accumulate(self.unet):
                    loss = self._training_step(batch)
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        all_params = unet_lora_params + te_lora_params
                        grad_norm = self.accelerator.clip_grad_norm_(all_params, max_grad_norm)
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

                    # 验证 + 保存图像
                    if self.global_step % validation_steps == 0:
                        self._run_validation(val_loop)
                        self.accelerator.wait_for_everyone()  # 等待主进程完成验证/FID再继续训练

                    # 保存权重检查点
                    if self.global_step % save_steps == 0:
                        self._save_checkpoint(optimizer, lr_scheduler)
                        self.accelerator.wait_for_everyone()

            if self.global_step >= num_train_steps:
                break

        # 训练结束: 最终保存
        self._save_checkpoint(optimizer, lr_scheduler)
        self.tb_logger.close()
        self.accelerator.end_training()
        logger.info("Training complete!")

    def _training_step(self, batch) -> torch.Tensor:
        """单步训练: 加噪 → 编码 → UNet 前向 → 计算 loss。

        可选增强:
          noise_offset  — 向噪声中叠加一个通道级随机偏移，改善极亮/极暗图像的生成质量。
          min_snr_gamma — Min-SNR 时间步加权，抑制高噪声步的过度优化，稳定训练。

        支持两种 batch 格式:
          - 含 "pixel_values"：在线 VAE encode（cache_latents=false）
          - 含 "latents"：使用预缓存 latent，跳过 VAE encode（cache_latents=true）
        """
        if "latents" in batch:
            latents = batch["latents"].to(self.accelerator.device)
        else:
            pixel_values = batch["pixel_values"].to(dtype=self.vae.dtype)
            with torch.no_grad():
                latents = self.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)

        # ── noise offset ────────────────────────────────────────────────────
        # 为每个样本在每个潜变量通道上叠加一个相同的随机偏移标量
        # offset shape: (B, C, 1, 1)，广播到 (B, C, H, W)
        if self.noise_offset > 0:
            noise = noise + self.noise_offset * torch.randn(
                latents.shape[0], latents.shape[1], 1, 1,
                device=latents.device, dtype=latents.dtype,
            )
        # ────────────────────────────────────────────────────────────────────

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
                te_ctx = (
                    torch.no_grad()
                    if (not self.train_te and not self.train_te2)
                    else contextlib.nullcontext()
                )
                with te_ctx:
                    prompt_embeds, pooled_prompt_embeds = self._encode_prompt_sdxl(
                        batch["input_ids_1"], batch["input_ids_2"]
                    )
            add_time_ids = self._compute_sdxl_time_ids(
                batch, bsz, latents.device, prompt_embeds.dtype
            )
            added_cond_kwargs = {
                "text_embeds": pooled_prompt_embeds,
                "time_ids": add_time_ids,
            }
            noise_pred = self.unet(
                noisy_latents, timesteps, prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
            ).sample
        else:
            if hasattr(self, "_cached_prompt_embeds"):
                encoder_hidden_states = self._cached_prompt_embeds.expand(bsz, -1, -1).to(latents.device)
            else:
                te_ctx = torch.no_grad() if not self.train_te else contextlib.nullcontext()
                with te_ctx:
                    encoder_hidden_states = self._encode_prompt_sd15(batch["input_ids"])
            noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

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
        """执行验证生成（所有 Rank 均参与 FID 图像生成，仅 Rank 0 生成样本图和计算 FID）。

        所有 Rank 各自创建 pipeline（使用 unwrapped 模型，不触发 DDP AllReduce），
        均匀分配 FID 图像生成任务，通过 accelerator.gather 汇聚特征后由 Rank 0 计算 FID。
        """
        from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline

        self.unet.eval()
        if self.train_te and self.text_encoder is not None:
            self.text_encoder.eval()
        if self.train_te2 and self.text_encoder_2 is not None:
            self.text_encoder_2.eval()

        unwrapped_unet = self.accelerator.unwrap_model(self.unet)
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
            pipeline = StableDiffusionXLPipeline(
                vae=self.vae, unet=unwrapped_unet,
                text_encoder=unwrapped_te, text_encoder_2=unwrapped_te2,
                tokenizer=self.tokenizer, tokenizer_2=self.tokenizer_2,
                scheduler=inference_scheduler,
            )
        else:
            pipeline = StableDiffusionPipeline(
                vae=self.vae, unet=unwrapped_unet,
                text_encoder=unwrapped_te, tokenizer=self.tokenizer,
                scheduler=inference_scheduler,
                safety_checker=None, feature_extractor=None,
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

        pipeline.set_progress_bar_config(disable=True)
        val_loop.run(
            pipeline,
            self.global_step,
            self.tb_logger,
            device=self.accelerator.device,
            accelerator=self.accelerator,
            pipeline_kwargs_override=pipeline_kwargs_override,
        )
        del pipeline

        self.unet.train()
        if self.train_te and self.text_encoder is not None:
            self.text_encoder.train()
        if self.train_te2 and self.text_encoder_2 is not None:
            self.text_encoder_2.train()

    def _save_checkpoint(self, optimizer, lr_scheduler):
        """保存权重检查点（仅主进程）。"""
        if not self.accelerator.is_main_process:
            return

        self.ckpt_manager.save(
            step=self.global_step,
            global_epoch=self.global_epoch,
            unet=self.accelerator.unwrap_model(self.unet),
            text_encoder=self.accelerator.unwrap_model(self.text_encoder) if self.train_te else None,
            text_encoder_2=(
                self.accelerator.unwrap_model(self.text_encoder_2)
                if self.train_te2 and self.text_encoder_2 is not None else None
            ),
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            seed=self.training_cfg.get("seed", 42),
            is_lora=True,
        )
