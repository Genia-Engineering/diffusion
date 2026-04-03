"""PixArt-Sigma Native Img2Img 训练器 — 8-channel concatenation + 多层残差注入。

训练范式: 支持 Flow Matching 和 DDPM 双噪声范式 (通过 noise_paradigm 切换)
  1. Flow Matching (noise_paradigm="flow_matching"):
     Rectified Flow velocity prediction, 线性插值加噪
  2. DDPM (noise_paradigm="ddpm"):
     epsilon/v-prediction, scheduler.add_noise 加噪, 可选 Min-SNR 加权

输入架构: 8-channel = 4-ch noisy latent + 4-ch clean reference latent
条件注入: PatchEmbed 通道拼接 + 每 N 层残差注入参考特征
文本条件: 彻底弃用 T5, cross-attention 接收可学习 null embedding

训练阶段:
  1. 目标图 VAE latent 预缓存 (复用 BaseTrainer._precompute_latents_distributed)
  2. 条件色块图 VAE latent 预缓存 (_precompute_ref_latents)
  3. 卸载 VAE encoder, 释放显存
  4. 训练主循环: 加载双 latent → 8-ch concat → model → loss

冻结策略 (phase3_mode):
  - "new_modules_only": 冻结 Transformer base weights, 仅训练新增模块
      (PatchEmbed.proj, ref_proj, ref_norm, injection_gates, null_embed)
  - "full_finetune": Transformer + 新增模块全量微调
"""

import logging
import math
import os
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.buckets import BucketManager, BucketSampler
from data.dataset import BaseImageDataset
from data.native_img2img_dataset import NativeImg2ImgCachedLatentDataset
from data.transforms import AspectRatioResize
from models.model_loader import load_pixart_sigma_components
from models.native_img2img import build_native_img2img_model, NativeImg2ImgPixArtWrapper
from utils.ema import EMAModel
from utils.fid import FIDCalculator
from utils.memory import apply_memory_optimizations
from utils.validation import ValidationLoop
from .base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class PixArtNativeImg2ImgTrainer(BaseTrainer):
    """PixArt-Sigma Native Img2Img 训练器。"""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.model_type = config.model.model_type

        self.noise_paradigm: str = self.training_cfg.get("noise_paradigm", "flow_matching")

        # Flow Matching 参数
        self.timestep_sampling: str = self.training_cfg.get("timestep_sampling", "logit_normal")
        self.logit_mean: float = float(self.training_cfg.get("logit_mean", 0.0))
        self.logit_std: float = float(self.training_cfg.get("logit_std", 1.0))

        # DDPM 参数
        self.noise_offset: float = float(self.training_cfg.get("noise_offset", 0.0))
        self.min_snr_gamma: float = float(self.training_cfg.get("min_snr_gamma", 0.0))

        # 损失权重矩阵 (前景/背景 × 结构/色彩)
        self.loss_weight_enabled: bool = bool(self.training_cfg.get("loss_weight_enabled", False))
        self.loss_weight_threshold: float = float(self.training_cfg.get("loss_weight_threshold", 0.1))
        self.w_fg_struct: float = float(self.training_cfg.get("loss_weight_fg_structure", 1.0))
        self.w_bg_struct: float = float(self.training_cfg.get("loss_weight_bg_structure", 1.0))
        self.w_fg_color: float = float(self.training_cfg.get("loss_weight_fg_color", 1.0))
        self.w_bg_color: float = float(self.training_cfg.get("loss_weight_bg_color", 1.0))

        self.cond_dropout_prob: float = float(self.training_cfg.get("cond_dropout_prob", 0.1))

        # Refine 模式 (Stage 2: 低噪声步结构精修)
        self.refine_enabled: bool = bool(self.training_cfg.get("refine_enabled", False))
        self.refine_max_timestep: int = int(self.training_cfg.get("refine_max_timestep", 400))
        self.refine_struct_lambda: float = float(self.training_cfg.get("refine_struct_lambda", 0.5))
        self.refine_struct_sharpness: float = float(self.training_cfg.get("refine_struct_sharpness", 20.0))

        injection_interval = config.model.get("injection_interval", 4)

        self._load_models(injection_interval)
        self._freeze_parameters()

        # DDPM Min-SNR 加权需要 SNR 查找表
        self._snr_cache: torch.Tensor | None = None
        if self.noise_paradigm == "ddpm" and self.min_snr_gamma > 0:
            alphas_cumprod = self.noise_scheduler.alphas_cumprod
            self._snr_cache = alphas_cumprod / (1.0 - alphas_cumprod)

        if self.refine_enabled:
            logger.info(
                f"[Refine] Stage 2 enabled: t ∈ [0, {self.refine_max_timestep}], "
                f"struct_lambda={self.refine_struct_lambda}, sharpness={self.refine_struct_sharpness}"
            )

        apply_memory_optimizations(
            transformer=self.transformer,
            vae=self.vae,
            enable_gradient_checkpointing=self.training_cfg.get("gradient_checkpointing", True),
            attention_backend=self.training_cfg.get("attention_backend", "sdpa"),
            enable_channels_last=False,
        )

    # ── 模型加载与架构手术 ─────────────────────────────────────────

    def _load_models(self, injection_interval: int):
        model_path = self.config.model.pretrained_model_name_or_path
        weights_dir = self.config.model.get("weights_dir", None)
        scheduler_shift = float(self.training_cfg.get("scheduler_shift", 1.0))
        use_flow_matching = self.noise_paradigm == "flow_matching"

        components = load_pixart_sigma_components(
            model_path, weights_dir=weights_dir, scheduler_shift=scheduler_shift,
            load_text_encoder=False,
            flow_matching=use_flow_matching,
        )
        self.vae = components["vae"]
        self.transformer = components["transformer"]
        self.noise_scheduler = components["noise_scheduler"]
        self.text_encoder = None
        self.tokenizer = None

        # 从全参微调 checkpoint 加载 transformer 权重（覆盖 pretrained 权重）
        ft_path = self.config.model.get("finetuned_transformer_path", None)
        if ft_path and os.path.isdir(ft_path):
            from diffusers import PixArtTransformer2DModel
            self.transformer = PixArtTransformer2DModel.from_pretrained(
                ft_path, torch_dtype=self.transformer.dtype,
            )
            logger.info(f"Fine-tuned transformer loaded from: {ft_path}")

        # DDPM 模式: 允许配置覆盖 prediction_type
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

        self.model = build_native_img2img_model(
            self.transformer, injection_interval=injection_interval,
        )

    def _freeze_parameters(self):
        self.vae.requires_grad_(False)

        phase3_mode = self.training_cfg.get("phase3_mode", "new_modules_only")

        if phase3_mode == "new_modules_only":
            self.transformer.requires_grad_(False)
            self.transformer.pos_embed.proj.requires_grad_(True)
            self.model.ref_proj.requires_grad_(True)
            self.model.ref_norm.requires_grad_(True)
            for gate in self.model.injection_gates:
                gate.requires_grad_(True)
            self.model.null_embed.requires_grad_(True)
            logger.info(
                "Phase3 mode: new_modules_only — Transformer frozen, "
                "PatchEmbed.proj + ref_proj + injection_gates + null_embed trainable"
            )
        else:
            self.model.requires_grad_(True)
            self.vae.requires_grad_(False)
            logger.info("Phase3 mode: full_finetune — all model parameters trainable")

        self.print_trainable_params(self.model)

    # ── 条件 latent 预缓存 ─────────────────────────────────────────

    def _get_ref_latent_cache_dir(self) -> str:
        explicit = self.training_cfg.get("ref_latent_cache_dir", None)
        if explicit:
            return explicit
        data_dir = self.config.data.get("conditioning_data_dir", "")
        parent = os.path.dirname(os.path.normpath(data_dir))
        return os.path.join(parent, "ref_latent_cache")

    def _precompute_ref_latents(self, bucket_to_indices: dict) -> None:
        """预缓存条件色块图的 VAE latent（多卡并行，按桶分辨率对齐）。"""
        from data.controlnet_dataset import _build_cond_index, _KNOWN_SUFFIX_PAIRS, _strip_known_suffix

        cache_dir = Path(self._get_ref_latent_cache_dir())
        cache_dir.mkdir(parents=True, exist_ok=True)

        data_cfg = self.config.data
        cond_data_dir = Path(data_cfg.conditioning_data_dir)
        train_data_dir = Path(data_cfg.train_data_dir)
        center_crop = data_cfg.get("center_crop", False)
        vae_batch_size = self.training_cfg.get("latent_cache_batch_size", 4)

        num_processes = self.accelerator.num_processes
        process_index = self.accelerator.process_index
        device = self.accelerator.device

        cond_index = _build_cond_index(cond_data_dir)

        train_images = sorted(
            p for p in train_data_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        )

        index_to_bucket = {}
        for bucket, indices in bucket_to_indices.items():
            for idx in indices:
                index_to_bucket[idx] = bucket

        def _get_base_key_from_name(fname: str) -> str:
            for orig_suffix, _ in _KNOWN_SUFFIX_PAIRS:
                bk = _strip_known_suffix(fname, orig_suffix)
                if bk is not None:
                    return bk
            return Path(fname).stem

        my_indices = list(range(process_index, len(train_images), num_processes))
        todo_indices = [
            i for i in my_indices
            if not (cache_dir / f"{_get_base_key_from_name(train_images[i].name)}.pt").exists()
        ]

        if not todo_indices:
            logger.info(f"[Rank {process_index}] 条件 latent 缓存已完整，跳过")
        else:
            logger.info(
                f"[Rank {process_index}] 预缓存 {len(todo_indices)} 张条件图的 VAE latent → {cache_dir}"
            )

            use_slicing = getattr(self.vae, "use_slicing", False)
            use_tiling = getattr(self.vae, "use_tiling", False)
            if use_slicing:
                self.vae.disable_slicing()
            if use_tiling:
                self.vae.disable_tiling()
            self.vae.eval()

            to_tensor = T.ToTensor()
            normalize = T.Normalize([0.5], [0.5])

            size_groups: dict[tuple, list[int]] = defaultdict(list)
            for idx in todo_indices:
                bucket = index_to_bucket.get(idx, (1024, 1024))
                size_groups[bucket].append(idx)

            try:
                with tqdm(
                    total=len(todo_indices),
                    desc=f"[Rank {process_index}] ref VAE latent cache",
                    disable=not self.accelerator.is_main_process,
                ) as pbar:
                    for target_size, indices in size_groups.items():
                        resizer = AspectRatioResize(target_size, center_crop=center_crop)

                        for batch_start in range(0, len(indices), vae_batch_size):
                            batch_indices = indices[batch_start:batch_start + vae_batch_size]

                            imgs_normal, imgs_flip, base_keys = [], [], []
                            for idx in batch_indices:
                                fname = train_images[idx].name
                                bk = _get_base_key_from_name(fname)
                                base_keys.append(bk)

                                cond_path = cond_index.get(bk)
                                if cond_path is None:
                                    raise FileNotFoundError(
                                        f"条件图未找到: base_key='{bk}' (target: {fname})"
                                    )

                                cond_img = Image.open(cond_path).convert("RGB")
                                cond_resized = resizer(cond_img)
                                imgs_normal.append(normalize(to_tensor(cond_resized)))
                                imgs_flip.append(normalize(to_tensor(TF.hflip(cond_resized))))

                            batch_t = torch.stack(imgs_normal).to(device)
                            batch_t_flip = torch.stack(imgs_flip).to(device)

                            with torch.no_grad():
                                latents = self.vae.encode(batch_t).latent_dist.mode()
                                latents = (latents * self.vae.config.scaling_factor).float().cpu()
                                latents_flip = self.vae.encode(batch_t_flip).latent_dist.mode()
                                latents_flip = (latents_flip * self.vae.config.scaling_factor).float().cpu()

                            for j, bk in enumerate(base_keys):
                                torch.save(
                                    {"latent": latents[j], "latent_flip": latents_flip[j]},
                                    cache_dir / f"{bk}.pt",
                                )
                            pbar.update(len(batch_indices))
            finally:
                if use_slicing:
                    self.vae.enable_slicing()
                if use_tiling:
                    self.vae.enable_tiling()

        self.accelerator.wait_for_everyone()
        logger.info(f"[Rank {process_index}] 条件 latent 预缓存完成")

    # ── DataLoader ─────────────────────────────────────────────────

    def _build_dataloader(self, bucket_to_indices: dict) -> DataLoader:
        data_cfg = self.config.data
        resolution = data_cfg.get("resolution", 1024)
        batch_size = self.training_cfg.get("train_batch_size", 2)

        dataset = NativeImg2ImgCachedLatentDataset(
            data_dir=data_cfg.train_data_dir,
            cache_dir=self._get_latent_cache_dir(),
            conditioning_data_dir=data_cfg.conditioning_data_dir,
            ref_latent_cache_dir=self._get_ref_latent_cache_dir(),
            resolution=resolution,
            random_flip=data_cfg.get("random_flip", True),
        )

        dataset.set_bucket_assignments(bucket_to_indices)
        sampler = BucketSampler(bucket_to_indices, batch_size, drop_last=True, shuffle=True)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=self.training_cfg.get("dataloader_num_workers", 4),
            pin_memory=True,
            drop_last=True,
        )
        return dataloader

    def _log_bucket_stats(self, bucket_to_indices: dict) -> None:
        total = sum(len(v) for v in bucket_to_indices.values())
        lines = [f"Aspect ratio bucket distribution (total={total}):"]
        for (w, h), indices in sorted(bucket_to_indices.items()):
            lines.append(f"  {w}x{h}: {len(indices)} images ({len(indices)/total*100:.1f}%)")
        logger.info("\n".join(lines))

    # ── 辅助函数 ─────────────────────────────────────────────────

    def _sample_real_images(self, n: int) -> list:
        import random as rng

        data_cfg = self.config.data
        resolution = data_cfg.get("resolution", 1024)
        dataset = BaseImageDataset(
            data_dir=data_cfg.train_data_dir,
            resolution=resolution,
            center_crop=data_cfg.get("center_crop", False),
            random_flip=False,
        )
        indices = rng.sample(range(len(dataset)), min(n, len(dataset)))
        return [dataset.get_pil_image(i) for i in indices]

    def _load_val_ground_truth_images(self, n: int) -> list:
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

    def _load_val_conditioning_images(self) -> list[Image.Image]:
        """加载验证用条件图像。"""
        val_cfg = self.config.get("validation", {})
        cond_paths = val_cfg.get("conditioning_images", [])

        if cond_paths:
            return [Image.open(p).convert("RGB") for p in cond_paths]

        from data.controlnet_dataset import _build_cond_index, _KNOWN_SUFFIX_PAIRS, _strip_known_suffix
        data_cfg = self.config.data
        cond_dir = Path(data_cfg.conditioning_data_dir)
        train_dir = Path(data_cfg.train_data_dir)

        cond_index = _build_cond_index(cond_dir)
        train_images = sorted(
            p for p in train_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        )

        n = val_cfg.get("num_val_images", 4)
        images = []
        for i in range(min(n, len(train_images))):
            fname = train_images[i].name
            bk = None
            for orig_suffix, _ in _KNOWN_SUFFIX_PAIRS:
                bk = _strip_known_suffix(fname, orig_suffix)
                if bk is not None:
                    break
            if bk is None:
                bk = train_images[i].stem
            cond_path = cond_index.get(bk)
            if cond_path:
                images.append(Image.open(cond_path).convert("RGB"))
        return images

    @torch.no_grad()
    def _encode_val_ref_latents(self, cond_images: list[Image.Image]) -> list[torch.Tensor]:
        """在 VAE encoder 卸载前，预编码验证条件图为 VAE latent。"""
        device = self.accelerator.device
        to_tensor = T.ToTensor()
        normalize = T.Normalize([0.5], [0.5])
        bucket_manager = BucketManager(model_type=self.model_type)

        use_slicing = getattr(self.vae, "use_slicing", False)
        use_tiling = getattr(self.vae, "use_tiling", False)
        if use_slicing:
            self.vae.disable_slicing()
        if use_tiling:
            self.vae.disable_tiling()

        latents_list = []
        for img in cond_images:
            w, h = img.size
            bucket = bucket_manager.get_bucket(w, h)
            resizer = AspectRatioResize(bucket, center_crop=False)
            img_resized = resizer(img)
            img_t = normalize(to_tensor(img_resized)).unsqueeze(0).to(device)
            latent = self.vae.encode(img_t).latent_dist.mode()
            latent = latent * self.vae.config.scaling_factor
            latents_list.append(latent.cpu())

        if use_slicing:
            self.vae.enable_slicing()
        if use_tiling:
            self.vae.enable_tiling()

        return latents_list

    def _broadcast_tensor(self, tensor: torch.Tensor | None) -> torch.Tensor | None:
        """从主进程广播 tensor 到所有 rank。"""
        import torch.distributed as dist
        if self.accelerator.num_processes <= 1:
            return tensor

        has_tensor = torch.tensor(
            [1 if tensor is not None else 0],
            device=self.accelerator.device, dtype=torch.long,
        )
        dist.broadcast(has_tensor, src=0)
        if has_tensor.item() == 0:
            return None

        if self.accelerator.is_main_process:
            shape_tensor = torch.tensor(
                list(tensor.shape), device=self.accelerator.device, dtype=torch.long,
            )
            ndim = torch.tensor([len(tensor.shape)], device=self.accelerator.device, dtype=torch.long)
        else:
            ndim = torch.tensor([0], device=self.accelerator.device, dtype=torch.long)

        dist.broadcast(ndim, src=0)
        if not self.accelerator.is_main_process:
            shape_tensor = torch.zeros(ndim.item(), device=self.accelerator.device, dtype=torch.long)
        dist.broadcast(shape_tensor, src=0)

        shape = tuple(shape_tensor.tolist())
        if self.accelerator.is_main_process:
            data = tensor.to(self.accelerator.device).contiguous()
        else:
            data = torch.zeros(shape, device=self.accelerator.device, dtype=torch.float32)
        dist.broadcast(data, src=0)
        return data.cpu()

    def _broadcast_tensor_list(
        self, tensor_list: list[torch.Tensor] | None
    ) -> list[torch.Tensor] | None:
        import torch.distributed as dist
        if self.accelerator.num_processes <= 1:
            return tensor_list

        count = torch.tensor(
            [len(tensor_list) if tensor_list is not None else 0],
            device=self.accelerator.device, dtype=torch.long,
        )
        dist.broadcast(count, src=0)
        n = count.item()
        if n == 0:
            return None

        result = []
        for i in range(n):
            t = tensor_list[i] if tensor_list is not None else None
            result.append(self._broadcast_tensor(t))
        return result

    # ── 主训练循环 ─────────────────────────────────────────────────

    def train(self):
        """Native Img2Img PixArt-Sigma 训练主循环。"""
        # ── 阶段1: 目标图 VAE latent 预缓存 ─────────────────────
        self.vae.to(self.accelerator.device, dtype=torch.float32)
        if self.training_cfg.get("cache_latents", True):
            self._precompute_latents_distributed(
                self._get_latent_cache_dir(), delete_encoder=False,
            )

        # ── 计算桶分配 ──────────────────────────────────────────
        data_cfg = self.config.data
        resolution = data_cfg.get("resolution", 1024)
        temp_dataset = BaseImageDataset(
            data_dir=data_cfg.train_data_dir,
            resolution=resolution,
            center_crop=data_cfg.get("center_crop", False),
            random_flip=False,
        )
        bucket_manager = BucketManager(model_type=self.model_type)
        image_sizes = temp_dataset.get_image_sizes()
        bucket_to_indices = bucket_manager.assign_buckets(image_sizes)
        self._log_bucket_stats(bucket_to_indices)
        del temp_dataset

        # ── 阶段2: 条件色块图 VAE latent 预缓存 ─────────────────
        self._precompute_ref_latents(bucket_to_indices)

        # 预编码验证条件图 latent（在卸载 VAE encoder 之前）
        val_cond_images = None
        val_ref_latents = None
        if self.accelerator.is_main_process:
            val_cond_images = self._load_val_conditioning_images()
            if val_cond_images:
                val_ref_latents = self._encode_val_ref_latents(val_cond_images)
                logger.info(f"验证条件 latent 已预编码: {len(val_ref_latents)} 张")
        val_ref_latents = self._broadcast_tensor_list(val_ref_latents)

        # 卸载 VAE encoder
        if hasattr(self.vae, "encoder"):
            del self.vae.encoder
        if hasattr(self.vae, "quant_conv"):
            del self.vae.quant_conv
        torch.cuda.empty_cache()
        logger.info("VAE encoder 已卸载")

        # ── 阶段3: 准备训练组件 ────────────────────────────────
        dataloader = self._build_dataloader(bucket_to_indices)
        num_train_steps = self.training_cfg.get("num_train_steps", 5000)
        max_grad_norm = self.training_cfg.get("max_grad_norm", 1.0)
        validation_steps = self.training_cfg.get("validation_steps", 500)
        save_steps = self.training_cfg.get("save_steps", 500)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = self.setup_optimizer(trainable_params=trainable_params)
        lr_scheduler = self.setup_lr_scheduler(optimizer, num_train_steps)

        self.model, optimizer, dataloader = self.accelerator.prepare(
            self.model, optimizer, dataloader
        )

        # EMA
        use_ema = self.training_cfg.get("use_ema", False)
        ema_model = None
        if use_ema:
            ema_decay = float(self.training_cfg.get("ema_decay", 0.9999))
            ema_update_after = int(self.training_cfg.get("ema_update_after_step", 0))
            unwrapped = self.accelerator.unwrap_model(self.model)
            ema_model = EMAModel(
                unwrapped.parameters(), decay=ema_decay,
                update_after_step=ema_update_after,
            )
            logger.info(f"EMA enabled: decay={ema_decay}, update_after_step={ema_update_after}")

        # 恢复训练
        resume_dir = self.training_cfg.get("resume_from_checkpoint", None)
        if resume_dir == "latest":
            resume_dir = self.ckpt_manager.get_latest_checkpoint()
        if resume_dir:
            # Load training state + optimizer (bypass transformer load since
            # ckpt_manager.load uses from_pretrained which fails with 8ch PatchEmbed)
            state = self.ckpt_manager.load(
                resume_dir,
                optimizer=optimizer,
                lr_scheduler=None,
                is_lora=False,
            )
            self.global_step = state["step"]
            self.global_epoch = state["epoch"]
            for _ in range(self.global_step):
                lr_scheduler.step()

            unwrapped = self.accelerator.unwrap_model(self.model)

            # Load transformer state dict directly (compatible with 8ch PatchEmbed)
            tf_sd_path = os.path.join(resume_dir, "transformer_state_dict.pt")
            if os.path.exists(tf_sd_path):
                tf_sd = torch.load(tf_sd_path, map_location="cpu", weights_only=True)
                unwrapped.transformer.load_state_dict(tf_sd)
                logger.info(f"Transformer state_dict loaded from {tf_sd_path}")

            wrapper_ckpt = os.path.join(resume_dir, "native_img2img_modules.pt")
            if os.path.exists(wrapper_ckpt):
                unwrapped.load_new_module_state_dict(
                    torch.load(wrapper_ckpt, map_location="cpu", weights_only=True)
                )
                logger.info(f"Native Img2Img modules loaded from {wrapper_ckpt}")

            logger.info(f"Resumed from step {self.global_step}")

            if use_ema:
                ema_path = os.path.join(resume_dir, "ema_model.pt")
                if os.path.exists(ema_path):
                    ema_model.load_state_dict(
                        torch.load(ema_path, map_location="cpu", weights_only=True)
                    )
                    logger.info(f"EMA loaded from {ema_path}")

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
            self.accelerator.wait_for_everyone()

        num_val_images = val_cfg.get("num_val_images", 4)
        val_prompts = [f"cond_{i}" for i in range(num_val_images)]

        val_loop = ValidationLoop(
            prompts=val_prompts,
            negative_prompt="",
            num_inference_steps=val_cfg.get("num_inference_steps", 50),
            guidance_scale=val_cfg.get("guidance_scale", 4.5),
            seed=val_cfg.get("seed", 42),
            num_images_per_prompt=1,
            save_dir=os.path.join(self.training_cfg.get("output_dir", "./outputs"), "samples"),
            fid_calculator=fid_calculator,
            fid_num_gen_images=val_cfg.get("fid", {}).get("num_gen_images", 256),
            fid_batch_size=val_cfg.get("fid", {}).get("batch_size", 4),
        )

        val_gt_images = None
        if self.accelerator.is_main_process:
            val_gt_images = self._load_val_ground_truth_images(num_val_images)
            n_cond = len(val_cond_images) if val_cond_images else 0
            logger.info(f"验证 GT: {len(val_gt_images)} 张, 条件图: {n_cond} 张")

        # ── 训练主循环 ────────────────────────────────────────────
        gradient_accumulation_steps = self.training_cfg.get("gradient_accumulation_steps", 1)
        steps_per_epoch = math.ceil(len(dataloader) / gradient_accumulation_steps)
        num_epochs = math.ceil(num_train_steps / max(steps_per_epoch, 1))
        progress_bar = tqdm(
            total=num_train_steps,
            initial=self.global_step,
            desc="Training",
            disable=not self.accelerator.is_main_process,
        )

        self.model.train()
        grad_norm = 0.0

        epoch_offset = self.global_epoch
        for epoch in range(num_epochs):
            self.global_epoch = epoch_offset + epoch
            for batch in dataloader:
                if self.global_step >= num_train_steps:
                    break

                with self.accelerator.accumulate(self.model):
                    loss = self._training_step(batch)
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        grad_norm = self.accelerator.clip_grad_norm_(trainable_params, max_grad_norm)
                        grad_norm = grad_norm.item() if hasattr(grad_norm, 'item') else float(grad_norm)

                    optimizer.step()
                    optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    if use_ema:
                        ema_model.step(self.accelerator.unwrap_model(self.model).parameters())

                    lr_scheduler.step()
                    self.global_step += 1

                    current_lr = lr_scheduler.get_last_lr()[0]
                    self.log_step(loss.item(), current_lr, grad_norm)

                    progress_bar.update(1)
                    progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}")

                    if self.global_step % save_steps == 0:
                        self._save_checkpoint(optimizer, lr_scheduler, ema_model=ema_model)
                        self.accelerator.wait_for_everyone()

                    if self.global_step % validation_steps == 0:
                        if use_ema:
                            unwrapped = self.accelerator.unwrap_model(self.model)
                            ema_model.store(unwrapped.parameters())
                            ema_model.copy_to(unwrapped.parameters())

                        self._run_validation(
                            val_loop, val_gt_images, val_cond_images, val_ref_latents,
                        )

                        if use_ema:
                            ema_model.restore(unwrapped.parameters())

                        self.accelerator.wait_for_everyone()

            if self.global_step >= num_train_steps:
                break

        self._save_checkpoint(optimizer, lr_scheduler, ema_model=ema_model)
        self.tb_logger.close()
        self.accelerator.end_training()
        logger.info("Training complete!")

    # ── 训练步 ─────────────────────────────────────────────────────

    def _training_step(self, batch) -> torch.Tensor:
        """单步训练，根据 noise_paradigm 分发到 Flow Matching 或 DDPM 路径。"""
        target_latents = batch["latents"].to(self.accelerator.device)
        ref_latents = batch["ref_latents"].to(self.accelerator.device)
        bsz = target_latents.shape[0]

        noise = torch.randn_like(target_latents)

        # CFG dropout: 以概率 p 将 ref_latents 置零
        if self.model.training and self.cond_dropout_prob > 0:
            drop_mask = torch.rand(bsz, device=target_latents.device) < self.cond_dropout_prob
            if drop_mask.any():
                ref_latents = ref_latents.clone()
                ref_latents[drop_mask] = 0.0

        if self.noise_paradigm == "flow_matching":
            return self._training_step_fm(target_latents, ref_latents, noise, bsz)
        else:
            weight_mask = self._get_weight_mask(batch, target_latents)
            return self._training_step_ddpm(target_latents, ref_latents, noise, bsz, weight_mask)

    def _build_added_cond_kwargs(self, latents: torch.Tensor, bsz: int) -> dict:
        _, _, h, w = latents.shape
        return {
            "resolution": torch.tensor([h * 8, w * 8], device=latents.device).repeat(bsz, 1),
            "aspect_ratio": torch.tensor([h * 8 / (w * 8)], device=latents.device).repeat(bsz, 1),
        }

    def _training_step_fm(
        self, target_latents, ref_latents, noise, bsz,
    ) -> torch.Tensor:
        """Flow Matching 训练步: 线性插值 → velocity prediction → MSE loss。"""
        if self.timestep_sampling == "logit_normal":
            t = torch.sigmoid(
                self.logit_mean + self.logit_std * torch.randn(bsz, device=target_latents.device)
            )
        else:
            t = torch.rand(bsz, device=target_latents.device)

        t_expanded = t.view(-1, 1, 1, 1)
        noisy_latents = t_expanded * target_latents + (1.0 - t_expanded) * noise
        timesteps_scaled = (1.0 - t) * 1000.0

        added_cond_kwargs = self._build_added_cond_kwargs(target_latents, bsz)

        model_output = self.model(
            noisy_latent=noisy_latents,
            reference_latent=ref_latents,
            timestep=timesteps_scaled,
            added_cond_kwargs=added_cond_kwargs,
        ).sample

        if model_output.shape[1] != target_latents.shape[1]:
            model_output, _ = model_output.chunk(2, dim=1)

        target = noise - target_latents
        loss = F.mse_loss(model_output.float(), target.float())

        if torch.isnan(loss):
            logger.warning(
                f"[Step {self.global_step}] NaN loss detected! "
                f"model_output: min={model_output.min().item():.4f}, max={model_output.max().item():.4f}"
            )
        return loss

    def _get_weight_mask(self, batch, latents) -> torch.Tensor | None:
        """从 batch 获取预缓存的亮度权重掩码 (B, 1, H, W)。"""
        if not self.loss_weight_enabled:
            return None
        if "weight_mask" in batch:
            return batch["weight_mask"].to(latents.device)
        return None

    def _build_loss_weights(self, latents, weight_mask) -> torch.Tensor | None:
        """构建 4 格权重矩阵 (B, C, H, W)。

        权重矩阵:
                      ch0 (结构)          ch1-3 (色彩)
          前景(线条)  w_fg_struct         w_fg_color
          背景(空白)  w_bg_struct         w_bg_color

        保证: 结构误差(前景↔背景混淆) >> 色彩误差(颜色偏差)
        """
        if not self.loss_weight_enabled or weight_mask is None:
            return None

        device = latents.device
        bsz, C, H, W = latents.shape

        fg_mask = (weight_mask > self.loss_weight_threshold).float()  # (B, 1, H, W)
        bg_mask = 1.0 - fg_mask

        weights = torch.empty(bsz, C, H, W, device=device)
        weights[:, 0:1] = fg_mask * self.w_fg_struct + bg_mask * self.w_bg_struct
        weights[:, 1:]  = fg_mask * self.w_fg_color  + bg_mask * self.w_bg_color

        return weights

    def _compute_struct_loss(
        self,
        model_output: torch.Tensor,
        noisy_latents: torch.Tensor,
        target_latents: torch.Tensor,
        timesteps: torch.Tensor,
        weight_mask: torch.Tensor,
    ) -> torch.Tensor:
        """x₀ 空间非线性结构 BCE loss。

        1. 从 ε 预测反推 x₀_hat
        2. 用自适应阈值将 x₀_hat ch0 软二值化为前景概率
        3. 与 GT 前景掩码做 BCE → 直接惩罚前景↔背景混淆
        """
        device = model_output.device

        alpha_t = self.noise_scheduler.alphas_cumprod.to(device)[timesteps]
        alpha_t = alpha_t.float().view(-1, 1, 1, 1)
        x0_hat = (noisy_latents.float() - (1.0 - alpha_t).sqrt() * model_output.float()) / (alpha_t.sqrt() + 1e-8)

        gt_fg = (weight_mask > self.loss_weight_threshold).float()  # (B, 1, H, W)

        target_ch0 = target_latents[:, 0:1].float()
        fg_px = gt_fg.sum(dim=[-1, -2], keepdim=True).clamp(min=1)
        bg_px = (1.0 - gt_fg).sum(dim=[-1, -2], keepdim=True).clamp(min=1)
        fg_mean = (target_ch0 * gt_fg).sum(dim=[-1, -2], keepdim=True) / fg_px
        bg_mean = (target_ch0 * (1.0 - gt_fg)).sum(dim=[-1, -2], keepdim=True) / bg_px
        latent_thresh = (fg_mean + bg_mean) * 0.5  # (B, 1, 1, 1)

        x0_ch0 = x0_hat[:, 0:1]
        pred_fg = torch.sigmoid(self.refine_struct_sharpness * (x0_ch0 - latent_thresh))
        pred_fg = pred_fg.clamp(1e-6, 1.0 - 1e-6)

        return F.binary_cross_entropy(pred_fg, gt_fg)

    def _training_step_ddpm(
        self, target_latents, ref_latents, noise, bsz, weight_mask=None,
    ) -> torch.Tensor:
        """DDPM 训练步: scheduler.add_noise → epsilon/v-prediction → weighted MSE loss。

        Refine 模式下:
        - 时间步采样范围限制为 [0, refine_max_timestep)
        - 额外叠加非线性结构 BCE loss (x₀ 空间)
        """
        if self.noise_offset > 0:
            noise = noise + self.noise_offset * torch.randn(
                target_latents.shape[0], target_latents.shape[1], 1, 1,
                device=target_latents.device, dtype=target_latents.dtype,
            )

        max_t = self.refine_max_timestep if self.refine_enabled else self.noise_scheduler.config.num_train_timesteps
        timesteps = torch.randint(0, max_t, (bsz,), device=target_latents.device).long()

        noisy_latents = self.noise_scheduler.add_noise(target_latents, noise, timesteps)

        added_cond_kwargs = self._build_added_cond_kwargs(target_latents, bsz)

        model_output = self.model(
            noisy_latent=noisy_latents,
            reference_latent=ref_latents,
            timestep=timesteps,
            added_cond_kwargs=added_cond_kwargs,
        ).sample

        if model_output.shape[1] != target_latents.shape[1]:
            model_output, _ = model_output.chunk(2, dim=1)

        if self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(target_latents, noise, timesteps)
        else:
            target = noise

        combined_weights = self._build_loss_weights(target_latents, weight_mask)

        noise_loss = self.compute_loss(
            model_output, target, timesteps,
            snr_cache=self._snr_cache,
            min_snr_gamma=self.min_snr_gamma,
            prediction_type=self.noise_scheduler.config.prediction_type,
            spatial_weights=combined_weights,
        )

        loss = noise_loss

        if self.refine_enabled and self.refine_struct_lambda > 0 and weight_mask is not None:
            struct_loss = self._compute_struct_loss(
                model_output, noisy_latents, target_latents, timesteps, weight_mask,
            )
            loss = loss + self.refine_struct_lambda * struct_loss

            if self.global_step % 50 == 0:
                logger.info(
                    f"[Step {self.global_step}] noise_loss={noise_loss.item():.4f}, "
                    f"struct_loss={struct_loss.item():.4f}, "
                    f"total={loss.item():.4f}"
                )

        if torch.isnan(loss):
            logger.warning(
                f"[Step {self.global_step}] NaN loss detected! "
                f"model_output: min={model_output.min().item():.4f}, max={model_output.max().item():.4f}"
            )
        return loss

    # ── 验证 ─────────────────────────────────────────────────────

    @torch.no_grad()
    def _run_validation(
        self, val_loop, val_gt_images, val_cond_images, val_ref_latents,
    ):
        """Native Img2Img 验证。"""
        from pipelines.pixart_native_img2img_pipeline import PixArtNativeImg2ImgPipeline

        self.model.eval()

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        if self.noise_paradigm == "flow_matching":
            from diffusers import FlowMatchEulerDiscreteScheduler
            from models.model_loader import patch_fm_scheduler_for_pipeline
            inference_scheduler = FlowMatchEulerDiscreteScheduler.from_config(
                self.noise_scheduler.config
            )
            patch_fm_scheduler_for_pipeline(inference_scheduler)
        else:
            from diffusers import DPMSolverMultistepScheduler
            inference_scheduler = DPMSolverMultistepScheduler.from_config(
                self.noise_scheduler.config,
                algorithm_type="sde-dpmsolver++",
                use_karras_sigmas=True,
            )

        pipeline = PixArtNativeImg2ImgPipeline(
            model=unwrapped_model,
            vae=self.vae,
            scheduler=inference_scheduler,
            device=self.accelerator.device,
        )

        pipeline_kwargs_override = None
        if val_ref_latents is not None:
            device = self.accelerator.device
            per_image_overrides = []
            for ref_lat in val_ref_latents:
                ref_lat_dev = ref_lat.to(device)
                override = {
                    "reference_latent": ref_lat_dev,
                    "height": ref_lat.shape[2] * 8,
                    "width": ref_lat.shape[3] * 8,
                }
                per_image_overrides.append(override)
            pipeline_kwargs_override = per_image_overrides

        val_loop.run(
            pipeline,
            self.global_step,
            self.tb_logger,
            device=self.accelerator.device,
            accelerator=self.accelerator,
            pipeline_kwargs_override=pipeline_kwargs_override,
            ground_truth_images=val_gt_images,
            conditioning_images=val_cond_images,
        )
        del pipeline

        self.model.train()

    # ── Checkpoint ───────────────────────────────────────────────

    def _save_checkpoint(
        self, optimizer, lr_scheduler,
        ema_model: EMAModel | None = None,
    ):
        if not self.accelerator.is_main_process:
            return

        unwrapped = self.accelerator.unwrap_model(self.model)

        # Bypass ckpt_manager._save_transformer (which uses save_pretrained)
        # because the surgically modified 8ch PatchEmbed breaks from_pretrained on load.
        # Instead: save optimizer/scheduler/training_state via ckpt_manager,
        # and save transformer + wrapper state dicts directly as .pt files.
        self.ckpt_manager.save(
            step=self.global_step,
            global_epoch=self.global_epoch,
            accelerator=self.accelerator,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            seed=self.training_cfg.get("seed", 42),
            is_lora=False,
        )

        ckpt_dir = self.ckpt_manager.get_latest_checkpoint()
        if ckpt_dir:
            tf_sd = unwrapped.transformer.state_dict()
            torch.save(tf_sd, os.path.join(ckpt_dir, "transformer_state_dict.pt"))
            logger.info(f"Transformer state_dict saved to {ckpt_dir}")

            modules_path = os.path.join(ckpt_dir, "native_img2img_modules.pt")
            torch.save(unwrapped.get_new_module_state_dict(), modules_path)
            logger.info(f"Native Img2Img modules saved to {modules_path}")

            if ema_model is not None:
                torch.save(ema_model.state_dict(), os.path.join(ckpt_dir, "ema_model.pt"))
                logger.info(f"EMA weights saved to {ckpt_dir}")
