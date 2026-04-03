"""PixArt-Sigma ControlNet 训练器 — Transformer 冻结，只训练 ControlNet adapter。

权重冻结策略:
  冻结 → VAE, T5 Text Encoder, PixArtTransformer2DModel
  可训练 → PixArtControlNetAdapterModel (adapter blocks + 可选 CNN encoder)

条件图编码双模式 (conditioning_mode):
  "vae"         — 条件图通过 VAE encode 到 latent 空间（可预缓存）
  "cnn_encoder" — 条件图通过轻量 CNN 映射到 latent 维度（在线计算）
"""

import logging
import math
import os
from pathlib import Path

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.buckets import BucketManager, BucketSampler
from data.controlnet_dataset import PixArtControlNetCachedLatentDataset
from models.controlnet_pixart import (
    PixArtControlNetAdapterModel,
    PixArtControlNetTransformerModel,
)
from models.model_loader import load_pixart_sigma_components
from utils.memory import apply_memory_optimizations
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


class PixArtControlNetTrainer(BaseTrainer):
    """PixArt-Sigma ControlNet 训练器。"""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.model_type = "pixart_sigma"
        self.cn_cfg = config.controlnet

        self.noise_offset: float = float(self.training_cfg.get("noise_offset", 0.0))
        self.min_snr_gamma: float = float(self.training_cfg.get("min_snr_gamma", 0.0))
        self.conditioning_mode: str = str(self.cn_cfg.get("conditioning_mode", "vae"))

        self._load_models()
        self._create_controlnet()
        self._freeze_parameters()

        self._snr_cache: torch.Tensor | None = None
        if self.min_snr_gamma > 0:
            alphas_cumprod = self.noise_scheduler.alphas_cumprod
            self._snr_cache = alphas_cumprod / (1.0 - alphas_cumprod)

        apply_memory_optimizations(
            transformer=self.transformer,
            controlnet=self.controlnet,
            vae=self.vae,
            text_encoder=self.text_encoder,
            enable_gradient_checkpointing=self.training_cfg.get("gradient_checkpointing", True),
            attention_backend=self.training_cfg.get("attention_backend", "sdpa"),
            enable_channels_last=False,
        )

    def _load_models(self):
        model_path = self.config.model.pretrained_model_name_or_path
        weights_dir = self.config.model.get("weights_dir", None)

        components = load_pixart_sigma_components(model_path, weights_dir=weights_dir, flow_matching=False)
        self.vae = components["vae"]
        self.transformer = components["transformer"]
        self.text_encoder = components["text_encoder"]
        self.tokenizer = components["tokenizer"]
        self.noise_scheduler = components["noise_scheduler"]

    def _create_controlnet(self):
        num_layers = int(self.cn_cfg.get("num_layers", 13))

        self.controlnet = PixArtControlNetAdapterModel.from_transformer(
            self.transformer,
            num_layers=num_layers,
            conditioning_mode=self.conditioning_mode,
        )

        self.joint_model = PixArtControlNetTransformerModel(
            transformer=self.transformer,
            controlnet=self.controlnet,
        )

    def _freeze_parameters(self):
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.transformer.requires_grad_(False)
        self.controlnet.requires_grad_(True)
        self.print_trainable_params(self.controlnet)

    def _build_dataloader(self) -> DataLoader:
        data_cfg = self.config.data
        batch_size = self.training_cfg.get("train_batch_size", 2)
        fixed_caption = data_cfg.get("caption", "")

        cache_dir = self._get_latent_cache_dir()
        conditioning_latent_cache_dir = self._get_conditioning_latent_cache_dir()

        text_embed_cache_dir = (
            self._get_text_embed_per_image_cache_dir()
            if getattr(self, "_per_image_caption", False)
            and self.training_cfg.get("cache_text_embeddings", False)
            else None
        )

        dataset = PixArtControlNetCachedLatentDataset(
            data_dir=data_cfg.train_data_dir,
            cache_dir=cache_dir,
            conditioning_data_dir=data_cfg.get("conditioning_data_dir", None),
            conditioning_latent_cache_dir=conditioning_latent_cache_dir,
            tokenizer=self.tokenizer,
            conditioning_mode=self.conditioning_mode,
            conditioning_type=self.cn_cfg.get("conditioning_type", "precomputed"),
            center_crop=data_cfg.get("center_crop", False),
            random_flip=data_cfg.get("random_flip", True),
            fixed_caption=fixed_caption,
            t5_max_length=int(data_cfg.get("t5_max_length", 300)),
            max_train_samples=data_cfg.get("max_train_samples", None),
            text_embed_cache_dir=text_embed_cache_dir,
        )

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
            collate_fn=_safe_collate,
        )

        return dataloader

    def _get_conditioning_latent_cache_dir(self) -> str | None:
        """返回条件图 latent 缓存目录路径（仅 VAE 模式使用）。"""
        if self.conditioning_mode != "vae":
            return None
        explicit = self.cn_cfg.get("conditioning_latent_cache_dir", None)
        if explicit:
            return explicit
        data_dir = self.config.data.get("train_data_dir", "./data")
        parent = os.path.dirname(os.path.normpath(data_dir))
        return os.path.join(parent, "conditioning_latent_cache")

    def _log_bucket_stats(self, bucket_to_indices: dict) -> None:
        total = sum(len(v) for v in bucket_to_indices.values())
        lines = [f"Aspect ratio bucket distribution (total={total}):"]
        for (w, h), indices in sorted(bucket_to_indices.items()):
            lines.append(f"  {w}×{h}: {len(indices)} images ({len(indices)/total*100:.1f}%)")
        logger.info("\n".join(lines))

    # ── 条件图 latent 预缓存 ────────────────────────────────────────────

    def _precompute_conditioning_latents(self, cache_dir: str) -> None:
        """预计算条件图像的 VAE latent（仅 vae 模式使用）。

        与 _precompute_latents 保持一致：每张条件图按其对应目标图的 bucket 尺寸
        resize 后送入 VAE，相同尺寸的图片打包为 batch 批量 encode。
        """
        import torchvision.transforms as T
        import torchvision.transforms.functional as TF
        from collections import defaultdict
        from PIL import Image as PIL_Image

        from data.buckets import BucketManager
        from data.controlnet_dataset import _build_cond_index, _strip_known_suffix, _KNOWN_SUFFIX_PAIRS
        from data.dataset import BaseImageDataset
        from data.transforms import AspectRatioResize

        cache_dir_path = Path(cache_dir)
        cache_dir_path.mkdir(parents=True, exist_ok=True)

        data_cfg = self.config.data
        center_crop = data_cfg.get("center_crop", False)
        vae_batch_size = self.training_cfg.get("latent_cache_batch_size", 4)

        num_processes = self.accelerator.num_processes
        process_index = self.accelerator.process_index
        device = self.accelerator.device

        cond_dir_str = data_cfg.get("conditioning_data_dir", None)
        if cond_dir_str is None:
            logger.warning("No conditioning_data_dir set, skipping conditioning latent precompute")
            return

        cond_dir = Path(cond_dir_str)
        if not cond_dir.exists():
            logger.warning(f"conditioning_data_dir {cond_dir} does not exist, skipping")
            return

        cond_index = _build_cond_index(cond_dir)
        cond_keys = sorted(cond_index.keys())
        cond_paths = [cond_index[k] for k in cond_keys]

        resolution = data_cfg.get("resolution", 1024)
        default_size = (resolution, resolution)

        temp_dataset = BaseImageDataset(
            data_dir=data_cfg.train_data_dir,
            resolution=resolution,
            center_crop=center_crop,
            random_flip=False,
        )
        if data_cfg.get("use_aspect_ratio_bucketing", True):
            bucket_manager = BucketManager(model_type=self.model_type)
            image_sizes = temp_dataset.get_image_sizes()
            bucket_to_indices = bucket_manager.assign_buckets(image_sizes)
            temp_dataset.set_bucket_assignments(bucket_to_indices)

        target_key_to_idx: dict[str, int] = {}
        for idx, img_path in enumerate(temp_dataset.image_paths):
            fname = img_path.name
            base_key = None
            for orig_suffix, _ in _KNOWN_SUFFIX_PAIRS:
                base_key = _strip_known_suffix(fname, orig_suffix)
                if base_key is not None:
                    break
            if base_key is None:
                base_key = img_path.stem
            target_key_to_idx[base_key] = idx

        my_indices = list(range(process_index, len(cond_keys), num_processes))
        todo_indices = [
            i for i in my_indices
            if not (cache_dir_path / f"{cond_keys[i]}.pt").exists()
        ]

        if not todo_indices:
            logger.info(
                f"[Rank {process_index}/{num_processes}] "
                f"条件 latent 缓存已完整（负责 {len(my_indices)} 张），跳过预计算"
            )
        else:
            logger.info(
                f"[Rank {process_index}/{num_processes}] "
                f"预计算 {len(todo_indices)}/{len(cond_keys)} 张条件图的 VAE latents → {cache_dir}"
            )

            self.vae.eval()
            to_tensor = T.ToTensor()
            normalize = T.Normalize([0.5], [0.5])

            size_groups: dict[tuple, list[int]] = defaultdict(list)
            for ci in todo_indices:
                target_idx = target_key_to_idx.get(cond_keys[ci])
                if target_idx is not None:
                    target_size = temp_dataset._get_target_size(target_idx)
                else:
                    target_size = default_size
                size_groups[target_size].append(ci)

            with tqdm(
                total=len(todo_indices),
                desc=f"[Rank {process_index}] 预计算条件 latents",
                disable=not self.accelerator.is_main_process,
            ) as pbar:
                for target_size, indices in size_groups.items():
                    resizer = AspectRatioResize(target_size, center_crop=center_crop)

                    for batch_start in range(0, len(indices), vae_batch_size):
                        batch_indices = indices[batch_start: batch_start + vae_batch_size]

                        imgs_normal, imgs_flip = [], []
                        for ci in batch_indices:
                            cond_img = PIL_Image.open(cond_paths[ci]).convert("RGB")
                            cond_img = resizer(cond_img)
                            imgs_normal.append(normalize(to_tensor(cond_img)))
                            imgs_flip.append(normalize(to_tensor(TF.hflip(cond_img))))

                        batch_t = torch.stack(imgs_normal).to(device)
                        batch_t_flip = torch.stack(imgs_flip).to(device)

                        with torch.no_grad():
                            latents = self.vae.encode(batch_t).latent_dist.mode()
                            latents = (latents * self.vae.config.scaling_factor).to(torch.float16).cpu()
                            latents_flip = self.vae.encode(batch_t_flip).latent_dist.mode()
                            latents_flip = (latents_flip * self.vae.config.scaling_factor).to(torch.float16).cpu()

                        for j, ci in enumerate(batch_indices):
                            torch.save(
                                {"latent": latents[j], "latent_flip": latents_flip[j]},
                                cache_dir_path / f"{cond_keys[ci]}.pt",
                            )
                        pbar.update(len(batch_indices))

            logger.info(f"[Rank {process_index}] 条件 latent 预计算完成")

        self.accelerator.wait_for_everyone()

    # ── VAE encoder 保留 ─────────────────────────────────────────────────

    def _precompute_latents_distributed(self, cache_dir: str) -> None:
        """保留 VAE encoder：ControlNet 验证时需要编码条件图像。

        基类在预计算完成后删除 VAE encoder 释放显存，但 ControlNet 验证时
        pipeline._prepare_control_latents() 需要 self.vae.encode() 来编码
        条件图像（VAE 模式），因此这里保存引用并在基类删除后恢复。
        """
        vae_encoder = getattr(self.vae, "encoder", None)
        vae_quant_conv = getattr(self.vae, "quant_conv", None)

        super()._precompute_latents_distributed(cache_dir)

        if vae_encoder is not None and not hasattr(self.vae, "encoder"):
            self.vae.encoder = vae_encoder
        if vae_quant_conv is not None and not hasattr(self.vae, "quant_conv"):
            self.vae.quant_conv = vae_quant_conv

    # ── 训练主循环 ───────────────────────────────────────────────────────

    def train(self):
        """PixArt-Sigma ControlNet 主训练循环。"""
        self.vae.to(self.accelerator.device, dtype=torch.float32)

        # 阶段1: 条件图 latent 预计算（仅 VAE 模式）
        # 必须在目标图预计算之前执行，因为后者结束时会删除 VAE encoder
        if self.conditioning_mode == "vae":
            cond_cache_dir = self._get_conditioning_latent_cache_dir()
            if cond_cache_dir:
                self._precompute_conditioning_latents(cond_cache_dir)

        # 阶段2: 目标图 VAE latent 预计算
        self._precompute_latents_distributed(self._get_latent_cache_dir())

        # 阶段3: T5 文本嵌入预计算
        # 先卸载 VAE 释放显存 — latent 预计算已完成，VAE 直到验证阶段才需要
        self.vae.to("cpu")
        torch.cuda.empty_cache()

        cache_text_embeddings = self.training_cfg.get("cache_text_embeddings", True)

        caption_mode = self.config.data.get("caption_mode", "fixed")
        self._per_image_caption = (caption_mode == "per_image") and bool(self.config.data.get("caption_dir", None))
        if caption_mode == "per_image" and not self.config.data.get("caption_dir", None):
            logger.warning("caption_mode='per_image' but caption_dir not set, falling back to fixed caption")

        if cache_text_embeddings:
            if self._per_image_caption:
                text_embed_cache_dir = self._get_text_embed_per_image_cache_dir()
                need_encode = not self._per_image_text_embed_cache_exists(text_embed_cache_dir)
                # 验证嵌入缓存也可能需要重新编码（如从 fixed 切换到 per_image 模式）
                val_cache_file = Path(text_embed_cache_dir) / "__val_and_neg_prompts__.pt"
                need_val_encode = not val_cache_file.exists()
                if val_cache_file.exists():
                    _tmp = torch.load(val_cache_file, map_location="cpu", weights_only=True)
                    if "val_prompt_embeds_list" not in _tmp:
                        need_val_encode = True
                    del _tmp
                if need_encode or need_val_encode:
                    self.text_encoder.to(self.accelerator.device)
                self._precompute_per_image_text_embeddings(text_embed_cache_dir)
            else:
                self.text_encoder.to(self.accelerator.device)
                self._precompute_text_embeddings()
        else:
            self.text_encoder.to(self.accelerator.device)

        # 阶段4: 准备训练组件
        dataloader = self._build_dataloader()
        num_train_steps = self.training_cfg.get("num_train_steps", 5000)
        max_grad_norm = self.training_cfg.get("max_grad_norm", 1.0)
        validation_steps = self.training_cfg.get("validation_steps", 500)
        save_steps = self.training_cfg.get("save_steps", 500)

        trainable_params = [p for p in self.controlnet.parameters() if p.requires_grad]
        optimizer = self.setup_optimizer(trainable_params=trainable_params)
        lr_scheduler = self.setup_lr_scheduler(optimizer, num_train_steps)

        self.joint_model, optimizer, dataloader = self.accelerator.prepare(
            self.joint_model, optimizer, dataloader,
        )

        if not cache_text_embeddings and self.text_encoder is not None:
            self.text_encoder.to(self.accelerator.device)

        # 恢复训练
        save_best = self.training_cfg.get("save_best", False)
        best_ema_decay = float(self.training_cfg.get("best_loss_ema_decay", 0.99))
        self._ema_loss: float | None = None
        self._best_loss = self.ckpt_manager.load_best_metric() if save_best else float("inf")

        resume_dir = self.training_cfg.get("resume_from_checkpoint", None)
        if resume_dir == "latest":
            resume_dir = self.ckpt_manager.get_latest_checkpoint()
        if resume_dir:
            state = self.ckpt_manager.load(
                resume_dir,
                controlnet=self.controlnet,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                is_lora=False,
            )
            self.global_step = state["step"]
            self.global_epoch = state["epoch"]
            if save_best and "ema_loss" in state:
                self._ema_loss = state["ema_loss"]
                logger.info(
                    f"Resumed from step {self.global_step}, "
                    f"ema_loss={self._ema_loss:.6f}, best_loss={self._best_loss:.6f}"
                )
            else:
                logger.info(f"Resumed from step {self.global_step}")
        elif save_best and self._best_loss < float("inf"):
            logger.info(f"Loaded best EMA loss from previous run: {self._best_loss:.6f}")

        val_cfg = self.config.get("validation", {})
        prompt_source = val_cfg.get("prompt_source", "config")
        if prompt_source == "data":
            if hasattr(self, "_per_image_val_captions") and self._per_image_val_captions:
                val_prompts = self._per_image_val_captions
            else:
                val_prompts = [self.config.data.get("caption", "a test image")]
        else:
            val_prompts = list(val_cfg.get("prompts", ["a test image"]))
        val_loop = ValidationLoop(
            prompts=val_prompts,
            negative_prompt=val_cfg.get("negative_prompt", ""),
            num_inference_steps=val_cfg.get("num_inference_steps", 20),
            guidance_scale=val_cfg.get("guidance_scale", 4.5),
            seed=val_cfg.get("seed", 42),
            num_images_per_prompt=val_cfg.get("num_images_per_prompt", 1),
            save_dir=os.path.join(self.training_cfg.get("output_dir", "./outputs"), "samples"),
        )

        # 训练主循环
        gradient_accumulation_steps = self.training_cfg.get("gradient_accumulation_steps", 1)
        steps_per_epoch = math.ceil(len(dataloader) / gradient_accumulation_steps)
        num_epochs = math.ceil(num_train_steps / max(steps_per_epoch, 1))
        progress_bar = tqdm(
            total=num_train_steps,
            initial=self.global_step,
            desc="Training PixArt ControlNet",
            disable=not self.accelerator.is_main_process,
        )

        self.joint_model.train()
        # 确保 transformer 子模块冻结时处于 eval 模式
        unwrapped = self.accelerator.unwrap_model(self.joint_model)
        unwrapped.transformer.eval()

        for epoch in range(num_epochs):
            self.global_epoch = epoch
            for batch in dataloader:
                if self.global_step >= num_train_steps:
                    break

                with self.accelerator.accumulate(self.joint_model):
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
                    step_loss = loss.item()
                    current_lr = lr_scheduler.get_last_lr()[0]
                    self.log_step(step_loss, current_lr, grad_norm)

                    progress_bar.update(1)
                    progress_bar.set_postfix(loss=f"{step_loss:.4f}", lr=f"{current_lr:.2e}")

                    if save_best:
                        if self._ema_loss is None:
                            self._ema_loss = step_loss
                        else:
                            self._ema_loss = best_ema_decay * self._ema_loss + (1 - best_ema_decay) * step_loss
                        self.tb_logger.log_ema_loss(self._ema_loss, self.global_step)

                        if self._ema_loss < self._best_loss:
                            self._best_loss = self._ema_loss
                            self.tb_logger.log_best_loss(self._best_loss, self.global_step)
                            self._save_best_checkpoint(optimizer, lr_scheduler, self._best_loss)

                    if self.global_step % save_steps == 0:
                        self._save_checkpoint(optimizer, lr_scheduler)
                        self.accelerator.wait_for_everyone()

                    if self.global_step % validation_steps == 0:
                        self._run_validation(val_loop)
                        self.accelerator.wait_for_everyone()

            if self.global_step >= num_train_steps:
                break

        self._save_checkpoint(optimizer, lr_scheduler)
        self.tb_logger.close()
        self.accelerator.end_training()
        logger.info("PixArt ControlNet training complete!")

    # ── 单步训练 ─────────────────────────────────────────────────────────

    def _training_step(self, batch) -> torch.Tensor:
        """单步训练: 加噪 → ControlNet + Transformer → loss。"""
        latents = batch["latents"].to(self.accelerator.device)
        bsz = latents.shape[0]

        noise = torch.randn_like(latents)
        if self.noise_offset > 0:
            noise = noise + self.noise_offset * torch.randn(
                bsz, latents.shape[1], 1, 1,
                device=latents.device, dtype=latents.dtype,
            )

        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,),
            device=latents.device,
        ).long()

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # 文本嵌入: 逐图预编码 > 固定缓存 > 实时编码
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

        # 条件输入
        if "conditioning_latents" in batch:
            controlnet_cond = batch["conditioning_latents"].to(self.accelerator.device)
        elif "conditioning_pixel_values" in batch:
            controlnet_cond = batch["conditioning_pixel_values"].to(self.accelerator.device)
        else:
            controlnet_cond = None

        noise_pred = self.joint_model(
            hidden_states=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=attention_mask,
            controlnet_cond=controlnet_cond,
        ).sample

        # learned variance: 只取前半部分
        if noise_pred.shape[1] != latents.shape[1]:
            noise_pred, _ = noise_pred.chunk(2, dim=1)

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

    # ── 验证 ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _run_validation(self, val_loop: ValidationLoop):
        if not self.accelerator.is_main_process:
            return

        unwrapped = self.accelerator.unwrap_model(self.joint_model)
        unwrapped.controlnet.eval()
        dev = self.accelerator.device
        self.vae.to(dev)

        try:
            from diffusers import DPMSolverSDEScheduler
            from pipelines.pixart_controlnet_pipeline import PixArtControlNetPipeline

            inference_scheduler = DPMSolverSDEScheduler.from_config(
                self.noise_scheduler.config, use_karras_sigmas=True,
            )

            pipeline = PixArtControlNetPipeline(
                vae=self.vae,
                transformer=unwrapped,
                controlnet=None,
                text_encoder=None,
                tokenizer=None,
                scheduler=inference_scheduler,
            )

            conditioning_images, ground_truth_images = self._load_val_images()

            pipeline_kwargs_override = None
            val_stems = getattr(self, "_val_matched_train_stems", [])
            if val_stems and getattr(self, "_per_image_caption", False):
                text_embed_cache_dir = Path(self._get_text_embed_per_image_cache_dir())
                neg_embeds = self._cached_negative_prompt_embeds.to(dev)
                neg_mask = self._cached_negative_prompt_attention_mask.to(dev)
                pipeline_kwargs_override = []
                for stem in val_stems:
                    text_file = text_embed_cache_dir / f"{stem}.pt"
                    text_data = torch.load(text_file, map_location=dev, weights_only=True)
                    pipeline_kwargs_override.append({
                        "prompt_embeds": text_data["prompt_embeds"].float().to(dev),
                        "prompt_attention_mask": text_data["prompt_attention_mask"].to(dev),
                        "negative_prompt_embeds": neg_embeds,
                        "negative_prompt_attention_mask": neg_mask,
                    })

                matched_captions = self._load_captions_for_stems(val_stems)
                val_loop.prompts = matched_captions
            elif hasattr(self, "_cached_val_prompt_embeds_list"):
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
            val_loop.run(
                pipeline,
                self.global_step,
                self.tb_logger,
                device=dev,
                pipeline_kwargs_override=pipeline_kwargs_override,
                conditioning_images=conditioning_images,
                ground_truth_images=ground_truth_images,
            )
            del pipeline
        except Exception:
            logger.exception(f"Validation failed at step {self.global_step}, skipping")
        finally:
            self.vae.to("cpu")
            torch.cuda.empty_cache()
            unwrapped.controlnet.train()

    def _load_captions_for_stems(self, stems: list[str]) -> list[str]:
        """根据训练图 stem 列表加载对应的 caption 文本。"""
        data_cfg = self.config.data
        caption_dir = Path(data_cfg.caption_dir)
        stem_replace_cfg = data_cfg.get("caption_stem_replace", {})
        stem_from = stem_replace_cfg.get("from", "")
        stem_to = stem_replace_cfg.get("to", "")
        fallback = data_cfg.get("caption_fallback", data_cfg.get("caption", ""))

        captions = []
        for stem in stems:
            caption_stem = stem.replace(stem_from, stem_to) if stem_from else stem
            caption_file = caption_dir / f"{caption_stem}.txt"
            if caption_file.exists():
                captions.append(caption_file.read_text(encoding="utf-8").strip())
            else:
                captions.append(fallback)
        return captions

    def _load_val_images(self):
        """加载验证用条件图像和对应训练原图。

        当 per_image caption 模式时，按训练图排序顺序选取对应条件图，
        确保 conditioning_images[i], val_prompts[i], ground_truth[i] 三者配对。
        """
        from PIL import Image as PIL_Image
        from data.controlnet_dataset import (
            _KNOWN_SUFFIX_PAIRS, _strip_known_suffix, _build_cond_index,
        )

        val_cfg = self.config.get("validation", {})
        n = val_cfg.get("num_val_samples", len(list(val_cfg.get("prompts", [""]))))
        resolution = self.config.data.get("resolution", 1024)
        _IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

        train_dir = Path(self.config.data.get("train_data_dir", ""))
        cond_dir_str = self.config.data.get("conditioning_data_dir", None)

        sampled_cond_files: list[Path] = []
        matched_train_files: list[Path | None] = []

        val_cond_paths = list(val_cfg.get("val_conditioning_images", []))
        if val_cond_paths:
            # 显式指定的验证条件图（最高优先级）
            sampled_cond_files = [Path(p) for p in val_cond_paths[:n]]
        elif getattr(self, "_per_image_caption", False) and cond_dir_str:
            # 逐图 caption 模式: 随机采样 n 对配对的 (训练图, 条件图)
            import random
            cond_dir = Path(cond_dir_str)
            if cond_dir.exists() and train_dir.exists():
                cond_index = _build_cond_index(cond_dir)
                train_images = sorted(
                    p for p in train_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in _IMG_EXTS
                )
                # 筛选有对应条件图的训练图，再随机抽样
                paired: list[tuple[Path, Path]] = []
                for img_path in train_images:
                    fname = img_path.name
                    base_key = None
                    for orig_suffix, _ in _KNOWN_SUFFIX_PAIRS:
                        base_key = _strip_known_suffix(fname, orig_suffix)
                        if base_key is not None:
                            break
                    if base_key is None:
                        base_key = img_path.stem
                    cond_path = cond_index.get(base_key)
                    if cond_path is not None:
                        paired.append((img_path, cond_path))

                rng = random.Random(val_cfg.get("seed", 42))
                selected = rng.sample(paired, min(n, len(paired)))
                for train_path, cond_path in selected:
                    matched_train_files.append(train_path)
                    sampled_cond_files.append(cond_path)
                logger.info(
                    f"验证条件图（随机配对采样, seed={val_cfg.get('seed', 42)}）：{[f.name for f in sampled_cond_files]}"
                )
        elif cond_dir_str:
            # 固定 caption 模式: 随机采样（条件图之间无需与特定 caption 配对）
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

        # 记录匹配到的训练图 stem，供 _run_validation 加载对应嵌入
        self._val_matched_train_stems = [
            f.stem if f is not None else None for f in matched_train_files
        ] if matched_train_files else []

        # 加载 ground truth（训练原图）
        ground_truth_images = None
        if matched_train_files:
            # 逐图模式: 已有精确配对
            gt_images = [
                PIL_Image.open(f).convert("RGB").resize((resolution, resolution))
                if f is not None else None
                for f in matched_train_files
            ]
        elif train_dir.exists():
            # 固定 caption / 显式条件图: 通过 base_key 反查训练图
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
        else:
            gt_images = []

        if gt_images and any(img is not None for img in gt_images):
            placeholder = PIL_Image.new("RGB", (resolution, resolution), (128, 128, 128))
            ground_truth_images = [
                img if img is not None else placeholder for img in gt_images
            ]
            logger.info(
                f"验证原图加载：{sum(1 for img in gt_images if img is not None)}/{len(gt_images)} 张匹配成功"
            )

        return conditioning_images, ground_truth_images

    # ── 检查点 ───────────────────────────────────────────────────────────

    def _save_checkpoint(self, optimizer, lr_scheduler):
        if not self.accelerator.is_main_process:
            return

        self.ckpt_manager.save(
            step=self.global_step,
            global_epoch=self.global_epoch,
            controlnet=self.controlnet,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            seed=self.training_cfg.get("seed", 42),
            is_lora=False,
            ema_loss=self._ema_loss,
        )

    def _save_best_checkpoint(self, optimizer, lr_scheduler, best_loss: float):
        if not self.accelerator.is_main_process:
            return

        self.ckpt_manager.save_best(
            step=self.global_step,
            global_epoch=self.global_epoch,
            best_metric_value=best_loss,
            controlnet=self.controlnet,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            seed=self.training_cfg.get("seed", 42),
            is_lora=False,
        )
