"""PixArt-Sigma 图像条件训练器 — 用图像编码器替代 T5 文本编码器。

训练范式: 支持 Flow Matching 和 DDPM 双噪声范式 (通过 noise_paradigm 切换)
  1. Flow Matching (noise_paradigm="flow_matching"):
     Rectified Flow velocity prediction, 线性插值加噪
  2. DDPM (noise_paradigm="ddpm"):
     epsilon/v-prediction, scheduler.add_noise 加噪, 可选 Min-SNR 加权

cross-attention 的条件信号来源：
  - 原版: T5-XXL 文本嵌入 (B, 300, 4096)
  - 本版: 图像编码器特征 (B, N, 4096)
    - VAE 模式: VAE latent → patchify Conv → (B, 256, 4096)
    - DINOv2 模式: DINOv2 features → MLP → (B, 1369, 4096)
    - CLIP 模式: CLIP features → MLP → (B, N, 4096)

训练阶段:
  1. 目标图 VAE latent 预缓存 (复用基类 _precompute_latents_distributed)
  2. 条件图特征预缓存 (_precompute_cond_features)，完成后卸载 frozen 编码器
  3. 训练主循环: 加载缓存 → 投射层 → Transformer → loss

CFG (Classifier-Free Guidance):
  训练时以 cond_dropout_prob 概率将条件特征置零 (对应无条件生成)。
  推理时: output = uncond + guidance_scale * (cond - uncond)
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
from data.img_cond_dataset import PixArtImgCondCachedLatentDataset
from data.transforms import AspectRatioResize
from models.image_encoder import build_image_encoder, VAEImageEncoder, DINOv2ImageEncoder, CLIPImageEncoder
from models.lora import LoRAInjector, LoRALinear, get_lora_params
from models.model_loader import load_pixart_sigma_components
from utils.ema import EMAModel
from utils.fid import FIDCalculator
from utils.memory import apply_memory_optimizations
from utils.validation import ValidationLoop
from .base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class PixArtImgCondTrainer(BaseTrainer):
    """PixArt-Sigma 图像条件全参数微调训练器。"""

    PIXART_DEFAULT_TARGET_MODULES = [
        "to_q", "to_k", "to_v", "to_out.0",
        "ff.net.0.proj", "ff.net.2",
    ]

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

        self.cond_dropout_prob: float = float(self.training_cfg.get("cond_dropout_prob", 0.1))

        self.img_enc_cfg = config.model.get("image_encoder", {})
        self.img_enc_type: str = self.img_enc_cfg.get("type", "dinov2")

        self.lora_cfg = config.get("lora", None)
        self.use_lora: bool = self.lora_cfg is not None and self.lora_cfg.get("enabled", True)

        self._load_models()
        if self.use_lora:
            self._inject_lora()
        self._freeze_parameters()
        self._patch_caption_projection_layernorm()

        # DDPM Min-SNR 加权需要 SNR 查找表
        self._snr_cache: torch.Tensor | None = None
        if self.noise_paradigm == "ddpm" and self.min_snr_gamma > 0:
            alphas_cumprod = self.noise_scheduler.alphas_cumprod
            self._snr_cache = alphas_cumprod / (1.0 - alphas_cumprod)

        apply_memory_optimizations(
            transformer=self.transformer,
            vae=self.vae,
            enable_gradient_checkpointing=self.training_cfg.get("gradient_checkpointing", True),
            attention_backend=self.training_cfg.get("attention_backend", "sdpa"),
            enable_channels_last=False,
        )

    # ── 模型加载与冻结 ──────────────────────────────────────────────

    def _load_models(self):
        model_path = self.config.model.pretrained_model_name_or_path
        weights_dir = self.config.model.get("weights_dir", None)
        scheduler_shift = float(self.training_cfg.get("scheduler_shift", 1.0))
        init_dit_randomly = self.training_cfg.get("init_dit_randomly", False)
        use_flow_matching = self.noise_paradigm == "flow_matching"

        components = load_pixart_sigma_components(
            model_path, weights_dir=weights_dir, scheduler_shift=scheduler_shift,
            load_text_encoder=False,
            init_transformer_randomly=init_dit_randomly,
            flow_matching=use_flow_matching,
        )
        self.vae = components["vae"]
        self.transformer = components["transformer"]
        self.noise_scheduler = components["noise_scheduler"]
        self.text_encoder = None
        self.tokenizer = None

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

        enc_result = build_image_encoder(self.img_enc_cfg)
        self.image_encoder: nn.Module = enc_result["encoder"]

    def _inject_lora(self):
        """向 Transformer 注入 LoRA 层。"""
        rank = self.lora_cfg.get("rank", 64)
        alpha = self.lora_cfg.get("alpha", 64.0)
        target_modules = list(
            self.lora_cfg.get("target_modules", self.PIXART_DEFAULT_TARGET_MODULES)
        )
        injected = LoRAInjector.inject(self.transformer, rank, alpha, target_modules)
        logger.info(
            f"LoRA injected into PixArt Transformer: {len(injected)} layers, "
            f"rank={rank}, alpha={alpha}"
        )

    def _freeze_parameters(self):
        self.vae.requires_grad_(False)

        if self.use_lora:
            for param in self.transformer.parameters():
                param.requires_grad = False
            for module in self.transformer.modules():
                if isinstance(module, LoRALinear):
                    module.lora_A.requires_grad_(True)
                    module.lora_B.requires_grad_(True)
            if hasattr(self.transformer, "caption_projection"):
                self.transformer.caption_projection.requires_grad_(True)
            logger.info("Transformer: frozen (LoRA + caption_projection trainable)")
        else:
            self.transformer.requires_grad_(True)
            logger.info("Transformer: full fine-tuning")

        if isinstance(self.image_encoder, VAEImageEncoder):
            self.image_encoder.requires_grad_(True)
            logger.info("VAE image encoder projector: trainable")
        elif isinstance(self.image_encoder, DINOv2ImageEncoder):
            self.image_encoder.backbone.requires_grad_(False)
            self.image_encoder.projection.requires_grad_(True)
            logger.info("DINOv2 backbone: frozen, projection: trainable")
        elif isinstance(self.image_encoder, CLIPImageEncoder):
            self.image_encoder.backbone.requires_grad_(False)
            self.image_encoder.projection.requires_grad_(True)
            logger.info("CLIP backbone: frozen, projection: trainable")

        self.print_trainable_params(self.transformer, self.image_encoder)

    def _patch_caption_projection_layernorm(self):
        """Inject LayerNorm into caption_projection to prevent bf16 overflow."""
        from utils.memory import patch_caption_projection_layernorm
        if patch_caption_projection_layernorm(self.transformer):
            self.transformer.caption_projection.norm.requires_grad_(True)

    # ── 条件特征预缓存 ──────────────────────────────────────────────

    def _get_cond_feature_cache_dir(self) -> str:
        explicit = self.training_cfg.get("cond_feature_cache_dir", None)
        if explicit:
            return explicit
        data_dir = self.config.data.get("conditioning_data_dir", "")
        parent = os.path.dirname(os.path.normpath(data_dir))
        return os.path.join(parent, f"cond_feature_cache_{self.img_enc_type}")

    def _precompute_cond_features(self, bucket_to_indices: dict) -> None:
        """预缓存条件图像的编码特征（多卡并行，按桶分辨率对齐）。

        VAE 模式: 条件图 → VAE encode → latent (4, H/8, W/8)
                  需要按目标桶尺寸 resize 以保持空间对齐
        DINOv2 模式: 条件图 → resize to fixed res → DINOv2 → features (N, 1024)
        CLIP 模式: 条件图 → resize to fixed res → CLIP → hidden_states[-2] (N, 1280)
        """
        from data.controlnet_dataset import _build_cond_index, _KNOWN_SUFFIX_PAIRS, _strip_known_suffix

        cache_dir = Path(self._get_cond_feature_cache_dir())
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
            logger.info(
                f"[Rank {process_index}] 条件特征缓存已完整，跳过"
            )
        else:
            logger.info(
                f"[Rank {process_index}] 预缓存 {len(todo_indices)} 张条件图的"
                f" {self.img_enc_type} 特征 → {cache_dir}"
            )

            if self.img_enc_type == "vae":
                self._cache_vae_cond_features(
                    todo_indices, train_images, index_to_bucket, cond_index,
                    cache_dir, center_crop, vae_batch_size, device,
                )
            elif self.img_enc_type == "clip":
                self._cache_clip_cond_features(
                    todo_indices, train_images, cond_index,
                    cache_dir, device, vae_batch_size,
                )
            else:
                self._cache_dinov2_cond_features(
                    todo_indices, train_images, cond_index,
                    cache_dir, device, vae_batch_size,
                )

        self.accelerator.wait_for_everyone()
        logger.info(f"[Rank {process_index}] 条件特征预缓存完成")

    def _cache_vae_cond_features(
        self, todo_indices, train_images, index_to_bucket, cond_index,
        cache_dir, center_crop, batch_size, device,
    ):
        """VAE 模式：将条件图 resize 到目标桶尺寸后 VAE encode。"""
        from data.controlnet_dataset import _strip_known_suffix, _KNOWN_SUFFIX_PAIRS

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
                desc=f"[Rank {self.accelerator.process_index}] VAE cond cache",
                disable=not self.accelerator.is_main_process,
            ) as pbar:
                for target_size, indices in size_groups.items():
                    target_w, target_h = target_size
                    resizer = AspectRatioResize(target_size, center_crop=center_crop)

                    for batch_start in range(0, len(indices), batch_size):
                        batch_indices = indices[batch_start:batch_start + batch_size]

                        imgs_normal, imgs_flip, base_keys = [], [], []
                        for idx in batch_indices:
                            fname = train_images[idx].name
                            bk = None
                            for orig_suffix, _ in _KNOWN_SUFFIX_PAIRS:
                                bk = _strip_known_suffix(fname, orig_suffix)
                                if bk is not None:
                                    break
                            if bk is None:
                                bk = Path(fname).stem
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

    def _cache_dinov2_cond_features(
        self, todo_indices, train_images, cond_index,
        cache_dir, device, batch_size,
    ):
        """DINOv2 模式：编码条件图为 patch features。"""
        from data.controlnet_dataset import _strip_known_suffix, _KNOWN_SUFFIX_PAIRS
        from transformers import AutoImageProcessor

        assert isinstance(self.image_encoder, DINOv2ImageEncoder)
        encoder = self.image_encoder.backbone.to(device)
        encoder.eval()

        processor = AutoImageProcessor.from_pretrained(
            self.img_enc_cfg.get("dinov2_model", "facebook/dinov2-large")
        )
        resolution = self.img_enc_cfg.get("dinov2_resolution", 518)

        with tqdm(
            total=len(todo_indices),
            desc=f"[Rank {self.accelerator.process_index}] DINOv2 cond cache",
            disable=not self.accelerator.is_main_process,
        ) as pbar:
            for batch_start in range(0, len(todo_indices), batch_size):
                batch_indices = todo_indices[batch_start:batch_start + batch_size]
                images_pil, images_pil_flip, base_keys = [], [], []

                for idx in batch_indices:
                    fname = train_images[idx].name
                    bk = None
                    for orig_suffix, _ in _KNOWN_SUFFIX_PAIRS:
                        bk = _strip_known_suffix(fname, orig_suffix)
                        if bk is not None:
                            break
                    if bk is None:
                        bk = Path(fname).stem
                    base_keys.append(bk)

                    cond_path = cond_index.get(bk)
                    if cond_path is None:
                        raise FileNotFoundError(
                            f"条件图未找到: base_key='{bk}' (target: {fname})"
                        )

                    cond_img = Image.open(cond_path).convert("RGB")
                    if resolution is not None:
                        cond_img = cond_img.resize(
                            (resolution, resolution), Image.LANCZOS
                        )
                    images_pil.append(cond_img)
                    images_pil_flip.append(TF.hflip(cond_img))

                inputs = processor(
                    images=images_pil, return_tensors="pt"
                ).pixel_values.to(device)
                inputs_flip = processor(
                    images=images_pil_flip, return_tensors="pt"
                ).pixel_values.to(device)

                with torch.no_grad():
                    features = encoder(inputs).last_hidden_state[:, 1:, :]
                    features = features.to(torch.float16).cpu()
                    features_flip = encoder(inputs_flip).last_hidden_state[:, 1:, :]
                    features_flip = features_flip.to(torch.float16).cpu()

                for j, bk in enumerate(base_keys):
                    torch.save(
                        {"features": features[j], "features_flip": features_flip[j]},
                        cache_dir / f"{bk}.pt",
                    )
                pbar.update(len(batch_indices))

        encoder.cpu()
        torch.cuda.empty_cache()

    def _cache_clip_cond_features(
        self, todo_indices, train_images, cond_index,
        cache_dir, device, batch_size,
    ):
        """CLIP 模式：编码条件图为 patch features (hidden_states[-2])。"""
        from data.controlnet_dataset import _strip_known_suffix, _KNOWN_SUFFIX_PAIRS
        from transformers import CLIPImageProcessor

        assert isinstance(self.image_encoder, CLIPImageEncoder)
        encoder = self.image_encoder.backbone.to(device)
        encoder.eval()

        clip_model_name = self.img_enc_cfg.get(
            "clip_model", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        )
        processor = CLIPImageProcessor.from_pretrained(clip_model_name)
        resolution = self.img_enc_cfg.get("clip_resolution", 224)

        with tqdm(
            total=len(todo_indices),
            desc=f"[Rank {self.accelerator.process_index}] CLIP cond cache",
            disable=not self.accelerator.is_main_process,
        ) as pbar:
            for batch_start in range(0, len(todo_indices), batch_size):
                batch_indices = todo_indices[batch_start:batch_start + batch_size]
                images_pil, images_pil_flip, base_keys = [], [], []

                for idx in batch_indices:
                    fname = train_images[idx].name
                    bk = None
                    for orig_suffix, _ in _KNOWN_SUFFIX_PAIRS:
                        bk = _strip_known_suffix(fname, orig_suffix)
                        if bk is not None:
                            break
                    if bk is None:
                        bk = Path(fname).stem
                    base_keys.append(bk)

                    cond_path = cond_index.get(bk)
                    if cond_path is None:
                        raise FileNotFoundError(
                            f"条件图未找到: base_key='{bk}' (target: {fname})"
                        )

                    cond_img = Image.open(cond_path).convert("RGB")
                    if resolution is not None:
                        cond_img = cond_img.resize(
                            (resolution, resolution), Image.LANCZOS
                        )
                    images_pil.append(cond_img)
                    images_pil_flip.append(TF.hflip(cond_img))

                inputs = processor(
                    images=images_pil, return_tensors="pt"
                ).pixel_values.to(device)
                inputs_flip = processor(
                    images=images_pil_flip, return_tensors="pt"
                ).pixel_values.to(device)

                with torch.no_grad():
                    out = encoder(inputs, output_hidden_states=True)
                    features = out.hidden_states[-2][:, 1:, :]
                    features = features.to(torch.float16).cpu()
                    out_flip = encoder(inputs_flip, output_hidden_states=True)
                    features_flip = out_flip.hidden_states[-2][:, 1:, :]
                    features_flip = features_flip.to(torch.float16).cpu()

                for j, bk in enumerate(base_keys):
                    torch.save(
                        {"features": features[j], "features_flip": features_flip[j]},
                        cache_dir / f"{bk}.pt",
                    )
                pbar.update(len(batch_indices))

        encoder.cpu()
        torch.cuda.empty_cache()

    # ── DataLoader 构建 ──────────────────────────────────────────────

    def _build_dataloader(self, bucket_to_indices: dict) -> DataLoader:
        data_cfg = self.config.data
        resolution = data_cfg.get("resolution", 1024)
        batch_size = self.training_cfg.get("train_batch_size", 2)

        dataset = PixArtImgCondCachedLatentDataset(
            data_dir=data_cfg.train_data_dir,
            cache_dir=self._get_latent_cache_dir(),
            conditioning_data_dir=data_cfg.conditioning_data_dir,
            cond_feature_cache_dir=self._get_cond_feature_cache_dir(),
            cond_encoder_type=self.img_enc_type,
            resolution=resolution,
            random_flip=data_cfg.get("random_flip", True),
            exclude_stems=getattr(self, "_val_exclude_stems", None),
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
            lines.append(f"  {w}×{h}: {len(indices)} images ({len(indices)/total*100:.1f}%)")
        logger.info("\n".join(lines))

    # ── 辅助函数 ─────────────────────────────────────────────────────

    def _broadcast_tensor(self, tensor: torch.Tensor | None) -> torch.Tensor | None:
        """从主进程广播 tensor 到所有 rank；主进程传入实际 tensor，其他传 None。"""
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
        """从主进程广播 tensor 列表到所有 rank（支持各元素 shape 不同）。"""
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

        split_dir = Path(self.training_cfg.output_dir) / "val_split"
        manifest_path = split_dir / "manifest.json"
        if manifest_path.exists():
            import json
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            train_split_dir = split_dir / "train"
            images = []
            for name in manifest["train_files"]:
                p = train_split_dir / name
                if p.exists():
                    images.append(Image.open(p).convert("RGB").resize((resolution, resolution)))
            logger.info(f"验证 ground truth 从 val_split/ 加载: {len(images)} 张")
            return images

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
            images = []
            for p in cond_paths:
                images.append(Image.open(p).convert("RGB"))
            return images

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

    # ── 主训练循环 ───────────────────────────────────────────────────

    def train(self):
        """图像条件 PixArt-Sigma 训练主循环。"""
        self._prepare_validation_split()

        # ── 阶段1: 目标图 VAE latent 预缓存 ──────────────────────────
        self.vae.to(self.accelerator.device, dtype=torch.float32)
        if self.training_cfg.get("cache_latents", True):
            self._precompute_latents_distributed(
                self._get_latent_cache_dir(), delete_encoder=False,
            )

        # ── 计算桶分配 (条件缓存和数据集都需要) ───────────────────────
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

        # ── 阶段2: 条件图特征预缓存 ─────────────────────────────────
        if self.img_enc_type == "dinov2":
            assert isinstance(self.image_encoder, DINOv2ImageEncoder)
            self.image_encoder.backbone.to(self.accelerator.device)
        elif self.img_enc_type == "clip":
            assert isinstance(self.image_encoder, CLIPImageEncoder)
            self.image_encoder.backbone.to(self.accelerator.device)

        self._precompute_cond_features(bucket_to_indices)

        # 在卸载编码器之前，预编码验证条件图的原始特征（主进程编码后广播给所有 rank）
        val_cond_images = None
        val_cond_raw_features = None
        if self.accelerator.is_main_process:
            val_cond_images = self._load_val_conditioning_images()
            if val_cond_images:
                val_cond_raw_features = self._pre_encode_val_conditioning(val_cond_images)
                shapes = [f.shape for f in val_cond_raw_features]
                logger.info(f"验证条件特征已预编码: {len(shapes)} 张, shapes={shapes}")
        val_cond_raw_features = self._broadcast_tensor_list(val_cond_raw_features)

        # 卸载 frozen 编码器
        if self.img_enc_type in ("dinov2", "clip"):
            if hasattr(self.image_encoder, "backbone") and self.image_encoder.backbone is not None:
                del self.image_encoder.backbone
                self.image_encoder.backbone = None
            torch.cuda.empty_cache()
            logger.info(f"{self.img_enc_type.upper()} backbone 已卸载")

        if hasattr(self.vae, "encoder"):
            del self.vae.encoder
        if hasattr(self.vae, "quant_conv"):
            del self.vae.quant_conv
        torch.cuda.empty_cache()
        logger.info("VAE encoder 已卸载")

        # ── 阶段3: 准备训练组件 ──────────────────────────────────────
        dataloader = self._build_dataloader(bucket_to_indices)
        num_train_steps = self.training_cfg.get("num_train_steps", 5000)
        projector_warmup_steps = int(self.training_cfg.get("projector_warmup_steps", 0))
        projector_warmup_lr = float(self.training_cfg.get("projector_warmup_lr", 0))
        max_grad_norm = self.training_cfg.get("max_grad_norm", 1.0)
        validation_steps = self.training_cfg.get("validation_steps", 500)
        save_steps = self.training_cfg.get("save_steps", 500)

        # projector 统一使用 self.image_encoder（forward 均返回 (embeds, mask) 元组）
        # VAEImageEncoder.forward(latent_4d) → (B, N, 4096), (B, N)
        # DINOv2ImageEncoder.forward(features_3d) → (B, N, 4096), (B, N)
        # CLIPImageEncoder.forward(features_3d) → (B, N, 4096), (B, N)
        #   backbone 已卸载 (=None)，forward 只走 self.projection，不需要 backbone
        projector = self.image_encoder

        caption_proj_ids = set()
        if hasattr(self.transformer, "caption_projection"):
            caption_proj_ids = {id(p) for p in self.transformer.caption_projection.parameters()}

        if self.use_lora:
            transformer_params = get_lora_params(self.transformer)
        else:
            transformer_params = [
                p for p in self.transformer.parameters()
                if p.requires_grad and id(p) not in caption_proj_ids
            ]

        caption_proj_params = [
            p for p in self.transformer.caption_projection.parameters()
            if p.requires_grad
        ] if caption_proj_ids else []

        if isinstance(self.image_encoder, (DINOv2ImageEncoder, CLIPImageEncoder)):
            projector_params = [p for p in self.image_encoder.projection.parameters() if p.requires_grad]
            if hasattr(self.image_encoder, "output_norm"):
                projector_params += [p for p in self.image_encoder.output_norm.parameters() if p.requires_grad]
        else:
            projector_params = [p for p in self.image_encoder.parameters() if p.requires_grad]
        projector_params = projector_params + caption_proj_params
        trainable_params = transformer_params + projector_params
        optimizer = self.setup_optimizer(
            trainable_params=transformer_params,
            text_encoder_params=projector_params,
        )
        lr_scheduler = self.setup_lr_scheduler(optimizer, num_train_steps)

        self.transformer, projector, optimizer, dataloader = self.accelerator.prepare(
            self.transformer, projector, optimizer, dataloader
        )

        if projector_warmup_steps > 0 and projector_warmup_lr > 0 and self.global_step < projector_warmup_steps:
            optimizer.param_groups[1]['lr'] = projector_warmup_lr

        # EMA
        use_ema = self.training_cfg.get("use_ema", False)
        ema_transformer = None
        ema_projector = None
        if use_ema:
            ema_decay = float(self.training_cfg.get("ema_decay", 0.9999))
            ema_update_after = int(self.training_cfg.get("ema_update_after_step", 0))
            unwrapped_tf = self.accelerator.unwrap_model(self.transformer)
            ema_transformer = EMAModel(
                unwrapped_tf.parameters(), decay=ema_decay,
                update_after_step=ema_update_after,
            )
            ema_projector = EMAModel(
                projector.parameters(), decay=ema_decay,
                update_after_step=ema_update_after,
            )
            logger.info(f"EMA enabled: decay={ema_decay}, update_after_step={ema_update_after}")

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
                is_lora=self.use_lora,
            )
            self.global_step = state["step"]
            self.global_epoch = state["epoch"]
            for _ in range(self.global_step):
                lr_scheduler.step()
            logger.info(f"Resumed from step {self.global_step}")

            projector_ckpt = os.path.join(resume_dir, "projector.pt")
            if os.path.exists(projector_ckpt):
                self.accelerator.unwrap_model(projector).load_state_dict(
                    torch.load(projector_ckpt, map_location=self.accelerator.device, weights_only=True)
                )
                logger.info(f"Projector weights loaded from {projector_ckpt}")

            cp_ckpt = os.path.join(resume_dir, "caption_projection.pt")
            if os.path.exists(cp_ckpt):
                unwrapped_tf = self.accelerator.unwrap_model(self.transformer)
                if hasattr(unwrapped_tf, "caption_projection"):
                    result = unwrapped_tf.caption_projection.load_state_dict(
                        torch.load(cp_ckpt, map_location=self.accelerator.device, weights_only=True),
                        strict=False,
                    )
                    logger.info(f"caption_projection weights loaded from {cp_ckpt}")
                    if result.missing_keys:
                        logger.info(f"  missing keys (will use init): {result.missing_keys}")

            if use_ema:
                ema_tf_path = os.path.join(resume_dir, "ema_transformer.pt")
                ema_proj_path = os.path.join(resume_dir, "ema_projector.pt")
                if os.path.exists(ema_tf_path):
                    ema_transformer.load_state_dict(
                        torch.load(ema_tf_path, map_location="cpu", weights_only=True)
                    )
                    logger.info(f"EMA transformer loaded from {ema_tf_path}")
                if os.path.exists(ema_proj_path):
                    ema_projector.load_state_dict(
                        torch.load(ema_proj_path, map_location="cpu", weights_only=True)
                    )
                    logger.info(f"EMA projector loaded from {ema_proj_path}")

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

        if getattr(self, "_val_exclude_stems", None):
            num_val = len(self._val_exclude_stems)
            if len(val_prompts) < num_val:
                val_prompts = (val_prompts * ((num_val // len(val_prompts)) + 1))[:num_val]

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
            logger.info(
                f"验证 GT: {len(val_gt_images)} 张, 条件图: {n_cond} 张"
            )

        # ── 训练主循环 ───────────────────────────────────────────────
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
        projector.train()

        if projector_warmup_steps > 0:
            warmup_lr_msg = f", projector LR={projector_warmup_lr:.1e}" if projector_warmup_lr > 0 else ""
            logger.info(
                f"Two-stage training: projector-only warmup for "
                f"{projector_warmup_steps} steps{warmup_lr_msg}, then joint fine-tuning"
            )

        epoch_offset = self.global_epoch
        grad_norm = None
        tf_grad_norm_val = None
        proj_grad_norm_val = None
        for epoch in range(num_epochs):
            self.global_epoch = epoch_offset + epoch
            for batch in dataloader:
                if self.global_step >= num_train_steps:
                    break

                with self.accelerator.accumulate(self.transformer, projector):
                    loss = self._training_step(batch, projector)

                    nan_detected = torch.isnan(loss) or torch.isinf(loss)
                    if nan_detected:
                        optimizer.zero_grad()
                        continue

                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        if self.global_step < projector_warmup_steps:
                            for p in transformer_params:
                                if p.grad is not None:
                                    p.grad.zero_()

                        tf_grad_norm = torch.nn.utils.clip_grad_norm_(transformer_params, max_grad_norm)
                        proj_grad_norm = torch.nn.utils.clip_grad_norm_(projector_params, max_grad_norm)
                        grad_norm = (tf_grad_norm ** 2 + proj_grad_norm ** 2) ** 0.5
                        tf_grad_norm_val = tf_grad_norm.item() if hasattr(tf_grad_norm, 'item') else float(tf_grad_norm)
                        proj_grad_norm_val = proj_grad_norm.item() if hasattr(proj_grad_norm, 'item') else float(proj_grad_norm)
                        grad_norm = grad_norm.item() if hasattr(grad_norm, 'item') else float(grad_norm)

                    optimizer.step()
                    optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    if use_ema:
                        ema_transformer.step(self.accelerator.unwrap_model(self.transformer).parameters())
                        ema_projector.step(projector.parameters())

                    lr_scheduler.step()
                    self.global_step += 1

                    if self.global_step < projector_warmup_steps and projector_warmup_lr > 0:
                        optimizer.param_groups[1]['lr'] = projector_warmup_lr

                    current_lr = lr_scheduler.get_last_lr()[0]
                    current_proj_lr = optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else None
                    self.log_step(
                        loss.item(), current_lr, grad_norm,
                        projector_lr=current_proj_lr,
                        tf_grad_norm=tf_grad_norm_val,
                        proj_grad_norm=proj_grad_norm_val,
                    )

                    if self.global_step == projector_warmup_steps and projector_warmup_steps > 0:
                        joint_proj_lr = optimizer.param_groups[1]['lr']
                        logger.info(
                            f"Projector warmup complete (step {self.global_step}). "
                            f"Joint training begins. Projector LR: "
                            f"{projector_warmup_lr:.1e} → {joint_proj_lr:.1e}"
                        )

                    progress_bar.update(1)
                    progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}")

                    if self.global_step % save_steps == 0:
                        self._save_checkpoint(
                            optimizer, lr_scheduler, projector,
                            ema_transformer=ema_transformer, ema_projector=ema_projector,
                        )
                        self.accelerator.wait_for_everyone()

                    if self.global_step % validation_steps == 0:
                        if use_ema:
                            unwrapped_tf = self.accelerator.unwrap_model(self.transformer)
                            ema_transformer.store(unwrapped_tf.parameters())
                            ema_transformer.copy_to(unwrapped_tf.parameters())
                            ema_projector.store(projector.parameters())
                            ema_projector.copy_to(projector.parameters())

                        self._run_validation(
                            val_loop, projector,
                            val_gt_images, val_cond_images,
                            val_cond_raw_features,
                        )

                        if use_ema:
                            ema_transformer.restore(unwrapped_tf.parameters())
                            ema_projector.restore(projector.parameters())

                        self.accelerator.wait_for_everyone()

            if self.global_step >= num_train_steps:
                break

        self._save_checkpoint(
            optimizer, lr_scheduler, projector,
            ema_transformer=ema_transformer, ema_projector=ema_projector,
        )
        self.tb_logger.close()
        self.accelerator.end_training()
        logger.info("Training complete!")

    # ── 训练步 ───────────────────────────────────────────────────────

    def _training_step(self, batch, projector: nn.Module) -> torch.Tensor:
        """单步训练，根据 noise_paradigm 分发到 Flow Matching 或 DDPM 路径。"""
        latents = batch["latents"].to(self.accelerator.device)
        cond_features = batch["cond_features"].to(self.accelerator.device)
        bsz = latents.shape[0]

        noise = torch.randn_like(latents)

        # 投射层: cond_features → encoder_hidden_states (B, N, 4096)
        prompt_embeds, attention_mask = projector(cond_features)

        # CFG dropout: 在投射层之后将条件特征置零，确保训练/推理一致
        # （投射层含 Conv2d bias，零输入会产生非零输出，故 dropout 须在投射后执行）
        if self.transformer.training and self.cond_dropout_prob > 0:
            drop_mask = torch.rand(bsz, device=latents.device) < self.cond_dropout_prob
            if drop_mask.any():
                prompt_embeds = prompt_embeds.clone()
                prompt_embeds[drop_mask] = 0.0

        unwrapped_tf = self.accelerator.unwrap_model(self.transformer)
        if hasattr(unwrapped_tf, "caption_projection"):
            caption_proj_out = unwrapped_tf.caption_projection(prompt_embeds)
            if torch.isnan(caption_proj_out).any() or torch.isinf(caption_proj_out).any():
                logger.warning(
                    f"[Step {self.global_step}] caption_projection overflow! "
                    f"input(prompt_embeds): min={prompt_embeds.min().item():.4f}, "
                    f"max={prompt_embeds.max().item():.4f}, "
                    f"output: min={caption_proj_out.min().item():.4f}, "
                    f"max={caption_proj_out.max().item():.4f}, "
                    f"has_nan={torch.isnan(caption_proj_out).any().item()}, "
                    f"has_inf={torch.isinf(caption_proj_out).any().item()}"
                )

        if self.noise_paradigm == "flow_matching":
            return self._training_step_fm(latents, noise, bsz, prompt_embeds, attention_mask)
        else:
            return self._training_step_ddpm(latents, noise, bsz, prompt_embeds, attention_mask)

    def _training_step_fm(
        self, latents, noise, bsz, prompt_embeds, attention_mask,
    ) -> torch.Tensor:
        """Flow Matching 训练步: 线性插值 → velocity prediction → MSE loss。"""
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
        loss = F.mse_loss(model_output.float(), target.float())

        if torch.isnan(loss):
            logger.warning(
                f"[Step {self.global_step}] NaN loss detected! "
                f"model_output: min={model_output.min().item():.4f}, max={model_output.max().item():.4f}, "
                f"has_nan={torch.isnan(model_output).any().item()}, "
                f"prompt_embeds: min={prompt_embeds.min().item():.4f}, max={prompt_embeds.max().item():.4f}, "
                f"has_nan={torch.isnan(prompt_embeds).any().item()}"
            )

        return loss

    def _training_step_ddpm(
        self, latents, noise, bsz, prompt_embeds, attention_mask,
    ) -> torch.Tensor:
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

        loss = self.compute_loss(
            model_output, target, timesteps,
            snr_cache=self._snr_cache,
            min_snr_gamma=self.min_snr_gamma,
            prediction_type=self.noise_scheduler.config.prediction_type,
        )

        if torch.isnan(loss):
            logger.warning(
                f"[Step {self.global_step}] NaN loss detected! "
                f"model_output: min={model_output.min().item():.4f}, max={model_output.max().item():.4f}, "
                f"has_nan={torch.isnan(model_output).any().item()}, "
                f"prompt_embeds: min={prompt_embeds.min().item():.4f}, max={prompt_embeds.max().item():.4f}, "
                f"has_nan={torch.isnan(prompt_embeds).any().item()}"
            )

        return loss

    # ── 验证 ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def _pre_encode_val_conditioning(
        self, cond_images: list[Image.Image],
    ) -> torch.Tensor:
        """在编码器卸载前，预编码验证条件图为原始特征。

        返回:
            VAE 模式: (N, 4, H/8, W/8) — VAE latent (尚未经 projector)
            DINOv2 模式: (N, seq_len, 1024) — backbone features (尚未经 projection)
            CLIP 模式: (N, seq_len, 1280) — backbone features (尚未经 projection)
        """
        device = self.accelerator.device
        all_features = []

        if self.img_enc_type in ("dinov2", "clip"):
            self.image_encoder.backbone.to(device)

        use_slicing = getattr(self.vae, "use_slicing", False)
        use_tiling = getattr(self.vae, "use_tiling", False)
        if use_slicing:
            self.vae.disable_slicing()
        if use_tiling:
            self.vae.disable_tiling()

        to_tensor = T.ToTensor()
        normalize = T.Normalize([0.5], [0.5])
        bucket_manager = BucketManager(model_type=self.model_type)

        processor = None
        resolution = None
        if self.img_enc_type == "clip":
            from transformers import CLIPImageProcessor
            clip_model_name = self.img_enc_cfg.get(
                "clip_model", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
            )
            processor = CLIPImageProcessor.from_pretrained(clip_model_name)
            resolution = self.img_enc_cfg.get("clip_resolution", 224)
        elif self.img_enc_type == "dinov2":
            from transformers import AutoImageProcessor
            processor = AutoImageProcessor.from_pretrained(
                self.img_enc_cfg.get("dinov2_model", "facebook/dinov2-large")
            )
            resolution = self.img_enc_cfg.get("dinov2_resolution", 518)

        for img in cond_images:
            if self.img_enc_type == "vae":
                w, h = img.size
                bucket = bucket_manager.get_bucket(w, h)
                resizer = AspectRatioResize(bucket, center_crop=False)
                img_resized = resizer(img)
                img_t = normalize(to_tensor(img_resized))
                img_t = img_t.unsqueeze(0).to(device)
                latent = self.vae.encode(img_t).latent_dist.mode()
                latent = latent * self.vae.config.scaling_factor
                all_features.append(latent.cpu())
            else:
                img_resized = img.resize((resolution, resolution), Image.LANCZOS)
                inputs = processor(
                    images=[img_resized], return_tensors="pt"
                ).pixel_values.to(device)
                features = self.image_encoder.encode_images(inputs)
                all_features.append(features.cpu())

        if use_slicing:
            self.vae.enable_slicing()
        if use_tiling:
            self.vae.enable_tiling()

        if self.img_enc_type in ("dinov2", "clip"):
            self.image_encoder.backbone.cpu()
            torch.cuda.empty_cache()

        return all_features

    @torch.no_grad()
    def _run_validation(
        self, val_loop, projector, val_gt_images, val_cond_images,
        val_cond_raw_features,
    ):
        """图像条件验证：用预编码的原始特征 → projector → pipeline。"""
        from diffusers import PixArtSigmaPipeline

        self.transformer.eval()
        projector.eval()

        unwrapped_transformer = self.accelerator.unwrap_model(self.transformer)
        if hasattr(unwrapped_transformer, "_orig_mod"):
            unwrapped_transformer = unwrapped_transformer._orig_mod

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

        pipeline = PixArtSigmaPipeline(
            vae=self.vae,
            transformer=unwrapped_transformer,
            text_encoder=None,
            tokenizer=None,
            scheduler=inference_scheduler,
        )
        pipeline.set_progress_bar_config(disable=True)

        pipeline_kwargs_override = None
        if val_cond_raw_features is not None:
            device = self.accelerator.device
            per_image_overrides = []
            for feat in val_cond_raw_features:
                pe, am = projector(feat.to(device))
                override = {
                    "prompt_embeds": pe,
                    "negative_prompt_embeds": torch.zeros_like(pe),
                    "prompt_attention_mask": am,
                    "negative_prompt_attention_mask": torch.ones_like(am),
                }
                if self.img_enc_type == "vae":
                    override["height"] = feat.shape[2] * 8
                    override["width"] = feat.shape[3] * 8
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

        self.transformer.train()
        projector.train()

    # ── Checkpoint ───────────────────────────────────────────────────

    def _save_checkpoint(
        self, optimizer, lr_scheduler, projector: nn.Module,
        ema_transformer: EMAModel | None = None,
        ema_projector: EMAModel | None = None,
    ):
        if not self.accelerator.is_main_process:
            return

        self.ckpt_manager.save(
            step=self.global_step,
            global_epoch=self.global_epoch,
            transformer=self.accelerator.unwrap_model(self.transformer),
            accelerator=self.accelerator,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            seed=self.training_cfg.get("seed", 42),
            is_lora=self.use_lora,
        )

        ckpt_dir = self.ckpt_manager.get_latest_checkpoint()
        if ckpt_dir:
            projector_path = os.path.join(ckpt_dir, "projector.pt")
            unwrapped_proj = self.accelerator.unwrap_model(projector)
            torch.save(unwrapped_proj.state_dict(), projector_path)
            logger.info(f"Projector weights saved to {projector_path}")

            unwrapped_tf = self.accelerator.unwrap_model(self.transformer)
            if hasattr(unwrapped_tf, "caption_projection"):
                cp_path = os.path.join(ckpt_dir, "caption_projection.pt")
                torch.save(unwrapped_tf.caption_projection.state_dict(), cp_path)
                logger.info(f"caption_projection weights saved to {cp_path}")

            if ema_transformer is not None:
                torch.save(ema_transformer.state_dict(), os.path.join(ckpt_dir, "ema_transformer.pt"))
            if ema_projector is not None:
                torch.save(ema_projector.state_dict(), os.path.join(ckpt_dir, "ema_projector.pt"))
            if ema_transformer is not None:
                logger.info(f"EMA weights saved to {ckpt_dir}")
