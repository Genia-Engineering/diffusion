"""PixArt-Sigma IP-Adapter 训练器 — T5 文本条件 + CLIP 图像条件 (decoupled cross-attention)。

训练范式: Rectified Flow (Flow Matching), 与 PixArtSigmaTrainer 完全一致。
条件注入方式:
  - T5 文本嵌入 → caption_projection → 原有 cross-attention (to_k, to_v)
  - CLIP 图像特征 → ImageProjection (Resampler) → 新增 cross-attention (to_k_ip, to_v_ip)
  - 最终输出: text_attn_output + ip_scale * ip_attn_output

训练阶段:
  1. 目标图 VAE latent 预缓存 (复用基类)
  2. T5 文本嵌入预缓存 (复用基类)
  3. CLIP 图像特征预缓存 (_precompute_clip_features)
  4. 训练主循环: 加载缓存 → IP-Adapter forward → Flow Matching loss

CFG (Classifier-Free Guidance):
  文本和图像条件独立 dropout:
    - text_dropout_prob: 置零文本嵌入 → 无文本条件分支
    - cond_dropout_prob: 置零 CLIP 特征 → 无图像条件分支
  推理时可独立控制 text_guidance 和 ip_guidance。

冻结策略 (通过 freeze_transformer 配置切换):
  - true:  冻结 Transformer 全部参数，只训练 IP-Adapter (ImageProjection + to_k_ip/to_v_ip)
  - false: Transformer 也一起微调 (全量模式)
"""

import logging
import math
import os
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
from data.ip_adapter_dataset import PixArtIPAdapterCachedDataset
from data.controlnet_dataset import _build_cond_index, _KNOWN_SUFFIX_PAIRS, _strip_known_suffix
from models.ip_adapter import build_ip_adapter, IPAdapterWrapper
from models.lora import LoRAInjector, LoRALinear, get_lora_params, save_lora_weights, load_lora_weights
from models.model_loader import load_pixart_sigma_components, load_clip_vision_model
from utils.ema import EMAModel
from utils.fid import FIDCalculator
from utils.memory import apply_memory_optimizations
from utils.validation import ValidationLoop
from .base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class _IPAdapterPipelineProxy:
    """Transparent proxy that sets ip_hidden_states on IP-Adapter processors
    before each pipeline call and clears them afterward.

    When ValidationLoop iterates over per-image overrides, each override dict
    contains a ``_ip_token_idx`` key.  The proxy pops this key, looks up the
    corresponding pre-projected ip_tokens, and sets them on all processors
    so the correct conditioning is used for that specific generation.
    """

    def __init__(self, pipeline, ip_processors, ip_tokens_list):
        self._pipeline = pipeline
        self._ip_processors = ip_processors
        self._ip_tokens_list = ip_tokens_list

    def __call__(self, *args, **kwargs):
        idx = kwargs.pop("_ip_token_idx", None)
        if idx is not None and idx < len(self._ip_tokens_list):
            for proc in self._ip_processors.values():
                proc._ip_hidden_states = self._ip_tokens_list[idx]

        try:
            return self._pipeline(*args, **kwargs)
        finally:
            for proc in self._ip_processors.values():
                proc._ip_hidden_states = None

    def __getattr__(self, name):
        return getattr(self._pipeline, name)


PIXART_DEFAULT_TARGET_MODULES = [
    "to_q", "to_k", "to_v", "to_out.0",
    "ff.net.0.proj", "ff.net.2",
]


class PixArtIPAdapterTrainer(BaseTrainer):
    """PixArt-Sigma IP-Adapter 训练器 — 文本 + 图像双条件。"""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.model_type = config.model.model_type

        self.timestep_sampling: str = self.training_cfg.get("timestep_sampling", "logit_normal")
        self.logit_mean: float = float(self.training_cfg.get("logit_mean", 0.0))
        self.logit_std: float = float(self.training_cfg.get("logit_std", 1.0))
        self.cond_dropout_prob: float = float(self.training_cfg.get("cond_dropout_prob", 0.05))
        self.text_dropout_prob: float = float(self.training_cfg.get("text_dropout_prob", 0.05))

        self.ipa_cfg = config.model.get("ip_adapter", {})
        self.lora_cfg = config.get("lora", None)
        self.use_lora: bool = self.lora_cfg is not None and self.lora_cfg.get("enabled", True)

        self._load_models()
        if self.use_lora:
            self._inject_lora()
        self._freeze_parameters()

        apply_memory_optimizations(
            transformer=self.transformer,
            vae=self.vae,
            text_encoder=self.text_encoder,
            enable_gradient_checkpointing=self.training_cfg.get("gradient_checkpointing", True),
            attention_backend=self.training_cfg.get("attention_backend", "sdpa"),
            enable_channels_last=False,
        )

        self._restore_ip_adapter_processors()

    # ── 模型加载与冻结 ──────────────────────────────────────────────

    def _load_models(self):
        model_path = self.config.model.pretrained_model_name_or_path
        weights_dir = self.config.model.get("weights_dir", None)
        scheduler_shift = float(self.training_cfg.get("scheduler_shift", 1.0))

        components = load_pixart_sigma_components(
            model_path, weights_dir=weights_dir, scheduler_shift=scheduler_shift,
            load_text_encoder=True,
        )
        self.vae = components["vae"]
        self.transformer = components["transformer"]
        self.text_encoder = components["text_encoder"]
        self.tokenizer = components["tokenizer"]
        self.noise_scheduler = components["noise_scheduler"]

        clip_result = load_clip_vision_model(
            model_name_or_path=self.ipa_cfg.get("clip_model", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"),
            weights_dir=weights_dir,
        )
        self.clip_model = clip_result["model"]
        self.clip_processor = clip_result["processor"]
        self.clip_embed_dim = clip_result["embed_dim"]

        self.ip_adapter_wrapper: IPAdapterWrapper = build_ip_adapter(
            transformer=self.transformer,
            clip_embed_dim=self.clip_embed_dim,
            num_tokens=self.ipa_cfg.get("num_tokens", 16),
            resampler_depth=self.ipa_cfg.get("resampler_depth", 4),
            ip_scale=float(self.ipa_cfg.get("ip_scale", 1.0)),
        )

    def _inject_lora(self):
        rank = self.lora_cfg.get("rank", 64)
        alpha = self.lora_cfg.get("alpha", 64.0)
        target_modules = list(
            self.lora_cfg.get("target_modules", PIXART_DEFAULT_TARGET_MODULES)
        )

        injected = LoRAInjector.inject(self.transformer, rank, alpha, target_modules)
        logger.info(
            f"LoRA injected into PixArt Transformer: {len(injected)} layers, "
            f"rank={rank}, alpha={alpha}"
        )

    def _restore_ip_adapter_processors(self):
        """Restore IP-Adapter processors after apply_memory_optimizations.

        _enable_sdpa calls set_attn_processor(AttnProcessor2_0()) which
        replaces ALL processors including our IP-Adapter ones.  Re-apply
        them so that decoupled cross-attention is active during training.
        """
        attn_procs = dict(self.transformer.attn_processors)
        for name, proc in self.ip_adapter_wrapper._ip_processors.items():
            attn_procs[name] = proc
        self.transformer.set_attn_processor(attn_procs)
        logger.info(
            f"Restored {len(self.ip_adapter_wrapper._ip_processors)} "
            f"IP-Adapter processors after memory optimizations"
        )

    def _freeze_parameters(self):
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.clip_model.requires_grad_(False)

        if self.use_lora:
            for param in self.transformer.parameters():
                param.requires_grad = False
            for module in self.transformer.modules():
                if isinstance(module, LoRALinear):
                    module.lora_A.requires_grad_(True)
                    module.lora_B.requires_grad_(True)
            logger.info("Transformer: frozen + LoRA trainable (IP-Adapter + LoRA mode)")
        else:
            freeze_transformer = self.training_cfg.get("freeze_transformer", True)
            if freeze_transformer:
                self.transformer.requires_grad_(False)
                logger.info("Transformer: frozen (IP-Adapter only mode)")
            else:
                self.transformer.requires_grad_(True)
                logger.info("Transformer: trainable (full fine-tune mode)")

        self.ip_adapter_wrapper.image_proj.requires_grad_(True)
        for proc in self.ip_adapter_wrapper._ip_processors.values():
            proc.requires_grad_(True)

        self.print_trainable_params(
            self.transformer, self.ip_adapter_wrapper.image_proj,
        )

    def _load_text_encoder_if_needed(self):
        """Only move text_encoder to GPU when the embedding cache does not exist."""
        import hashlib
        caption = self.config.data.get("caption", "")
        neg_prompt = self.config.get("validation", {}).get("negative_prompt", "")
        data_dir = self.config.data.get("train_data_dir", "./data")
        embed_hash = hashlib.md5(f"{caption}||{neg_prompt}".encode()).hexdigest()[:8]
        cache_file = Path(data_dir) / "text_embed_cache" / f"embeds_{self.model_type}_{embed_hash}.pt"
        if not cache_file.exists() and self.text_encoder is not None:
            self.text_encoder.to(self.accelerator.device)

    # ── CLIP 特征预缓存 ─────────────────────────────────────────────

    def _get_clip_feature_cache_dir(self) -> str:
        explicit = self.training_cfg.get("clip_feature_cache_dir", None)
        if explicit:
            return explicit
        data_dir = self.config.data.get("conditioning_data_dir", "")
        parent = os.path.dirname(os.path.normpath(data_dir))
        return os.path.join(parent, "clip_feature_cache")

    def _precompute_clip_features(self) -> None:
        """预缓存条件图像的 CLIP 特征（多卡并行）。

        每张条件图编码为:
          features:      (num_tokens+1, hidden_size) — 含 CLS + patch tokens
          features_flip: 同上，水平翻转版本
        """
        cache_dir = Path(self._get_clip_feature_cache_dir())
        cache_dir.mkdir(parents=True, exist_ok=True)

        data_cfg = self.config.data
        cond_data_dir = Path(data_cfg.conditioning_data_dir)
        train_data_dir = Path(data_cfg.train_data_dir)
        batch_size = self.training_cfg.get("latent_cache_batch_size", 4)

        num_processes = self.accelerator.num_processes
        process_index = self.accelerator.process_index
        device = self.accelerator.device

        cond_index = _build_cond_index(cond_data_dir)

        train_images = sorted(
            p for p in train_data_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        )

        def _get_base_key(fname: str) -> str:
            for orig_suffix, _ in _KNOWN_SUFFIX_PAIRS:
                bk = _strip_known_suffix(fname, orig_suffix)
                if bk is not None:
                    return bk
            return Path(fname).stem

        my_indices = list(range(process_index, len(train_images), num_processes))
        todo_indices = [
            i for i in my_indices
            if not (cache_dir / f"{_get_base_key(train_images[i].name)}.pt").exists()
        ]

        if not todo_indices:
            logger.info(f"[Rank {process_index}] CLIP 特征缓存已完整，跳过")
        else:
            logger.info(
                f"[Rank {process_index}] 预缓存 {len(todo_indices)} 张条件图的"
                f" CLIP 特征 → {cache_dir}"
            )
            self.clip_model.to(device)

            with tqdm(
                total=len(todo_indices),
                desc=f"[Rank {process_index}] CLIP feature cache",
                disable=not self.accelerator.is_main_process,
            ) as pbar:
                for batch_start in range(0, len(todo_indices), batch_size):
                    batch_indices = todo_indices[batch_start:batch_start + batch_size]
                    images_pil, images_pil_flip, base_keys = [], [], []

                    for idx in batch_indices:
                        fname = train_images[idx].name
                        bk = _get_base_key(fname)

                        cond_path = cond_index.get(bk)
                        if cond_path is None:
                            logger.warning(
                                f"条件图未找到，跳过: base_key='{bk}' (target: {fname})"
                            )
                            continue

                        base_keys.append(bk)
                        img = Image.open(cond_path).convert("RGB")
                        img_flip = TF.hflip(img)
                        images_pil.append(img)
                        images_pil_flip.append(img_flip)

                    if not images_pil:
                        pbar.update(len(batch_indices))
                        continue

                    inputs = self.clip_processor(
                        images=images_pil, return_tensors="pt"
                    ).pixel_values.to(device)
                    inputs_flip = self.clip_processor(
                        images=images_pil_flip, return_tensors="pt"
                    ).pixel_values.to(device)

                    with torch.no_grad():
                        out = self.clip_model(inputs, output_hidden_states=True)
                        features = out.hidden_states[-2]
                        features = features.to(torch.float16).cpu()

                        out_flip = self.clip_model(inputs_flip, output_hidden_states=True)
                        features_flip = out_flip.hidden_states[-2]
                        features_flip = features_flip.to(torch.float16).cpu()

                    for j, bk in enumerate(base_keys):
                        torch.save(
                            {"features": features[j], "features_flip": features_flip[j]},
                            cache_dir / f"{bk}.pt",
                        )
                    pbar.update(len(batch_indices))

            self.clip_model.cpu()
            torch.cuda.empty_cache()

        self.accelerator.wait_for_everyone()
        logger.info(f"[Rank {process_index}] CLIP 特征预缓存完成")

    # ── DataLoader 构建 ──────────────────────────────────────────────

    def _build_dataloader(self, bucket_to_indices: dict) -> DataLoader:
        data_cfg = self.config.data
        resolution = data_cfg.get("resolution", 1024)
        batch_size = self.training_cfg.get("train_batch_size", 2)

        dataset = PixArtIPAdapterCachedDataset(
            data_dir=data_cfg.train_data_dir,
            cache_dir=self._get_latent_cache_dir(),
            conditioning_data_dir=data_cfg.conditioning_data_dir,
            clip_feature_cache_dir=self._get_clip_feature_cache_dir(),
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

    _DTYPE_MAP = {0: torch.float32, 1: torch.float16, 2: torch.bfloat16, 3: torch.float64}
    _DTYPE_INV = {v: k for k, v in _DTYPE_MAP.items()}

    def _broadcast_tensor(self, tensor: torch.Tensor | None) -> torch.Tensor | None:
        """从主进程广播 tensor 到所有 rank（自动同步 shape 和 dtype）。"""
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
            dtype_code = self._DTYPE_INV.get(tensor.dtype, 0)
            shape_tensor = torch.tensor(
                list(tensor.shape), device=self.accelerator.device, dtype=torch.long,
            )
            meta = torch.tensor(
                [len(tensor.shape), dtype_code],
                device=self.accelerator.device, dtype=torch.long,
            )
        else:
            meta = torch.tensor([0, 0], device=self.accelerator.device, dtype=torch.long)

        dist.broadcast(meta, src=0)
        ndim, dtype_code = meta[0].item(), meta[1].item()
        dtype = self._DTYPE_MAP.get(dtype_code, torch.float32)

        if not self.accelerator.is_main_process:
            shape_tensor = torch.zeros(ndim, device=self.accelerator.device, dtype=torch.long)
        dist.broadcast(shape_tensor, src=0)

        shape = tuple(shape_tensor.tolist())
        if self.accelerator.is_main_process:
            data = tensor.to(device=self.accelerator.device, dtype=dtype).contiguous()
        else:
            data = torch.zeros(shape, device=self.accelerator.device, dtype=dtype)
        dist.broadcast(data, src=0)
        return data.cpu()

    def _broadcast_tensor_list(
        self, tensor_list: list[torch.Tensor] | None
    ) -> list[torch.Tensor] | None:
        """从主进程广播 tensor 列表到所有 rank。"""
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
            return [Image.open(p).convert("RGB") for p in cond_paths]

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
    def _pre_encode_val_clip_features(
        self, cond_images: list[Image.Image],
    ) -> list[torch.Tensor]:
        """在 CLIP 模型卸载前，预编码验证条件图的 CLIP 特征。"""
        device = self.accelerator.device
        self.clip_model.to(device)
        all_features = []

        for img in cond_images:
            inputs = self.clip_processor(
                images=[img], return_tensors="pt"
            ).pixel_values.to(device)

            out = self.clip_model(inputs, output_hidden_states=True)
            features = out.hidden_states[-2]
            all_features.append(features.cpu())

        self.clip_model.cpu()
        torch.cuda.empty_cache()
        return all_features

    # ── 主训练循环 ───────────────────────────────────────────────────

    def train(self):
        """IP-Adapter PixArt-Sigma 训练主循环。"""
        self._prepare_validation_split()

        # ── 阶段1: 目标图 VAE latent 预缓存 ──────────────────────────
        self.vae.to(self.accelerator.device, dtype=torch.float32)
        if self.training_cfg.get("cache_latents", True):
            self._precompute_latents_distributed(
                self._get_latent_cache_dir(), delete_encoder=False,
            )
        self.vae.cpu()
        torch.cuda.empty_cache()

        # ── 阶段2: T5 文本嵌入预缓存 ────────────────────────────────
        if self.training_cfg.get("cache_text_embeddings", True):
            self._load_text_encoder_if_needed()
            self._precompute_text_embeddings()

        # ── 计算桶分配 ──────────────────────────────────────────────
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

        # ── 阶段3: CLIP 图像特征预缓存 ──────────────────────────────
        if self.training_cfg.get("cache_clip_features", True):
            self._precompute_clip_features()

        # 预编码验证条件图的 CLIP 特征（在卸载 CLIP 模型之前）
        val_cond_images = None
        val_cond_clip_features = None
        if self.accelerator.is_main_process:
            val_cond_images = self._load_val_conditioning_images()
            if val_cond_images:
                val_cond_clip_features = self._pre_encode_val_clip_features(val_cond_images)
                shapes = [f.shape for f in val_cond_clip_features]
                logger.info(f"验证 CLIP 特征已预编码: {len(shapes)} 张, shapes={shapes}")
        val_cond_clip_features = self._broadcast_tensor_list(val_cond_clip_features)

        # 卸载 CLIP 模型和 VAE encoder
        del self.clip_model
        self.clip_model = None
        del self.clip_processor
        self.clip_processor = None

        if hasattr(self.vae, "encoder"):
            del self.vae.encoder
        if hasattr(self.vae, "quant_conv"):
            del self.vae.quant_conv
        self.vae.to(self.accelerator.device)
        torch.cuda.empty_cache()
        logger.info("CLIP 模型和 VAE encoder 已卸载")

        # ── 阶段4: 准备训练组件 ──────────────────────────────────────
        dataloader = self._build_dataloader(bucket_to_indices)
        num_train_steps = self.training_cfg.get("num_train_steps", 5000)
        max_grad_norm = self.training_cfg.get("max_grad_norm", 1.0)
        validation_steps = self.training_cfg.get("validation_steps", 500)
        save_steps = self.training_cfg.get("save_steps", 500)

        freeze_transformer = self.training_cfg.get("freeze_transformer", True)

        ip_adapter_params = (
            list(self.ip_adapter_wrapper.image_proj.parameters())
            + [
                p for proc in self.ip_adapter_wrapper._ip_processors.values()
                for p in proc.parameters()
            ]
        )

        ip_adapter_lr = float(self.training_cfg.get("ip_adapter_lr",
                              self.training_cfg.get("projector_lr",
                              self.training_cfg.learning_rate)))

        if self.use_lora:
            lora_params = get_lora_params(self.transformer)
            lora_lr = float(self.training_cfg.get("learning_rate", 1e-5))
            optimizer = self.setup_optimizer(
                trainable_params=lora_params,
                text_encoder_params=ip_adapter_params,
            )
            optimizer.param_groups[0]["lr"] = lora_lr
            optimizer.param_groups[1]["lr"] = ip_adapter_lr
            trainable_params = lora_params + ip_adapter_params
            logger.info(
                f"Optimizer: LoRA params ({len(lora_params)}) lr={lora_lr}, "
                f"IP-Adapter params ({len(ip_adapter_params)}) lr={ip_adapter_lr}"
            )
        elif freeze_transformer:
            optimizer = self.setup_optimizer(trainable_params=ip_adapter_params)
            optimizer.param_groups[0]["lr"] = ip_adapter_lr
            trainable_params = ip_adapter_params
        else:
            ip_param_ids = {id(p) for p in ip_adapter_params}
            transformer_params = [
                p for p in self.transformer.parameters()
                if p.requires_grad and id(p) not in ip_param_ids
            ]
            optimizer = self.setup_optimizer(
                trainable_params=transformer_params,
                text_encoder_params=ip_adapter_params,
            )
            trainable_params = transformer_params + ip_adapter_params

        lr_scheduler = self.setup_lr_scheduler(optimizer, num_train_steps)

        self.ip_adapter_wrapper, optimizer, dataloader = self.accelerator.prepare(
            self.ip_adapter_wrapper, optimizer, dataloader
        )

        # 恢复训练
        resume_dir = self.training_cfg.get("resume_from_checkpoint", None)
        if resume_dir == "latest":
            resume_dir = self.ckpt_manager.get_latest_checkpoint()
        if resume_dir:
            unwrapped_tf = self.accelerator.unwrap_model(self.ip_adapter_wrapper).transformer
            state = self.ckpt_manager.load(
                resume_dir,
                transformer=unwrapped_tf,
                optimizer=optimizer,
                lr_scheduler=None,
                is_lora=self.use_lora,
            )
            ipa_ckpt = os.path.join(resume_dir, "ip_adapter.pt")
            if os.path.exists(ipa_ckpt):
                ipa_state = torch.load(ipa_ckpt, map_location="cpu")
                unwrapped_wrapper = self.accelerator.unwrap_model(self.ip_adapter_wrapper)
                unwrapped_wrapper.load_ip_adapter_state_dict(ipa_state)
                logger.info(f"IP-Adapter weights restored from {ipa_ckpt}")
            self.global_step = state["step"]
            self.global_epoch = state["epoch"]
            for _ in range(self.global_step):
                lr_scheduler.step()
            logger.info(f"Resumed from step {self.global_step}")

        # EMA
        use_ema = self.training_cfg.get("use_ema", False)
        ema_ip_adapter = None
        if use_ema:
            ema_decay = float(self.training_cfg.get("ema_decay", 0.9999))
            ema_update_after = int(self.training_cfg.get("ema_update_after_step", 0))
            ema_ip_adapter = EMAModel(
                ip_adapter_params,
                decay=ema_decay,
                update_after_step=ema_update_after,
            )
            logger.info(f"EMA enabled for IP-Adapter params (decay={ema_decay})")

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

        self.ip_adapter_wrapper.train()

        epoch_offset = self.global_epoch
        for epoch in range(num_epochs):
            self.global_epoch = epoch_offset + epoch
            for batch in dataloader:
                if self.global_step >= num_train_steps:
                    break

                with self.accelerator.accumulate(self.ip_adapter_wrapper):
                    loss = self._training_step(batch)
                    self.accelerator.backward(loss)
                    self.accelerator.unwrap_model(self.ip_adapter_wrapper).clear_ip_hidden_states()

                    if self.accelerator.sync_gradients:
                        grad_norm = self.accelerator.clip_grad_norm_(
                            trainable_params, max_grad_norm
                        )
                        grad_norm = grad_norm.item() if hasattr(grad_norm, 'item') else float(grad_norm)

                    optimizer.step()
                    optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    if use_ema:
                        ema_ip_adapter.step(ip_adapter_params)

                    lr_scheduler.step()
                    self.global_step += 1

                    current_lr = lr_scheduler.get_last_lr()[0]
                    self.log_step(loss.item(), current_lr, grad_norm)

                    progress_bar.update(1)
                    progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}")

                    if self.global_step % save_steps == 0:
                        self._save_checkpoint(
                            optimizer, lr_scheduler,
                            ema_ip_adapter=ema_ip_adapter,
                        )
                        self.accelerator.wait_for_everyone()

                    if self.global_step % validation_steps == 0:
                        if use_ema and ema_ip_adapter is not None:
                            ema_ip_adapter.store(ip_adapter_params)
                            ema_ip_adapter.copy_to(ip_adapter_params)

                        self._run_validation(
                            val_loop, val_gt_images, val_cond_images,
                            val_cond_clip_features,
                        )

                        if use_ema and ema_ip_adapter is not None:
                            ema_ip_adapter.restore(ip_adapter_params)

                        self.accelerator.wait_for_everyone()

            if self.global_step >= num_train_steps:
                break

        self._save_checkpoint(
            optimizer, lr_scheduler,
            ema_ip_adapter=ema_ip_adapter,
        )
        self.tb_logger.close()
        self.accelerator.end_training()
        logger.info("Training complete!")

    # ── 训练步 ───────────────────────────────────────────────────────

    def _training_step(self, batch) -> torch.Tensor:
        """单步 Flow Matching 训练（IP-Adapter 版）。"""
        latents = batch["latents"].to(self.accelerator.device)
        clip_features = batch["clip_features"].to(self.accelerator.device)
        bsz = latents.shape[0]

        noise = torch.randn_like(latents)

        if self.timestep_sampling == "logit_normal":
            t = torch.sigmoid(
                self.logit_mean + self.logit_std * torch.randn(bsz, device=latents.device)
            )
        else:
            t = torch.rand(bsz, device=latents.device)

        t_expanded = t.view(-1, 1, 1, 1)
        noisy_latents = t_expanded * latents + (1.0 - t_expanded) * noise

        # gradient checkpointing (use_reentrant=True) requires at least one
        # input tensor with requires_grad=True to properly build the
        # computation graph through IP-Adapter parameters.
        if self.training_cfg.get("gradient_checkpointing", False):
            noisy_latents = noisy_latents.detach().requires_grad_(True)

        # 文本条件
        prompt_embeds = self._cached_prompt_embeds.expand(bsz, -1, -1).to(latents.device)
        attention_mask = self._cached_prompt_attention_mask.expand(bsz, -1).to(latents.device)

        # 独立 CFG dropout: 文本
        if self.ip_adapter_wrapper.training and self.text_dropout_prob > 0:
            text_drop_mask = torch.rand(bsz, device=latents.device) < self.text_dropout_prob
            if text_drop_mask.any():
                prompt_embeds = prompt_embeds.clone()
                prompt_embeds[text_drop_mask] = 0.0

        # 独立 CFG dropout: 图像
        if self.ip_adapter_wrapper.training and self.cond_dropout_prob > 0:
            img_drop_mask = torch.rand(bsz, device=latents.device) < self.cond_dropout_prob
            if img_drop_mask.any():
                clip_features = clip_features.clone()
                clip_features[img_drop_mask] = 0.0

        timesteps_scaled = (1.0 - t) * 1000.0

        model_output = self.ip_adapter_wrapper(
            hidden_states=noisy_latents,
            timestep=timesteps_scaled,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=attention_mask,
            ip_hidden_states=clip_features,
        ).sample

        if model_output.shape[1] != latents.shape[1]:
            model_output, _ = model_output.chunk(2, dim=1)

        target = noise - latents
        loss = F.mse_loss(model_output.float(), target.float())

        if torch.isnan(loss):
            logger.warning(
                f"[Step {self.global_step}] NaN loss detected! "
                f"model_output range: [{model_output.min().item():.4f}, {model_output.max().item():.4f}]"
            )

        return loss

    # ── 验证 ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def _run_validation(
        self, val_loop, val_gt_images, val_cond_images,
        val_cond_clip_features,
    ):
        """IP-Adapter 验证：用预编码的 CLIP 特征 + 文本嵌入生成图像。

        Uses _IPAdapterPipelineProxy to wrap the pipeline so that each
        validation image is generated with the correct per-image ip_hidden_states.
        """
        from diffusers import FlowMatchEulerDiscreteScheduler, PixArtSigmaPipeline
        from models.model_loader import patch_fm_scheduler_for_pipeline

        self.ip_adapter_wrapper.eval()

        unwrapped_wrapper = self.accelerator.unwrap_model(self.ip_adapter_wrapper)
        unwrapped_transformer = unwrapped_wrapper.transformer
        if hasattr(unwrapped_transformer, "_orig_mod"):
            unwrapped_transformer = unwrapped_transformer._orig_mod

        inference_scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            self.noise_scheduler.config
        )
        patch_fm_scheduler_for_pipeline(inference_scheduler)

        pipeline = PixArtSigmaPipeline(
            vae=self.vae,
            transformer=unwrapped_transformer,
            text_encoder=None,
            tokenizer=None,
            scheduler=inference_scheduler,
        )
        pipeline.set_progress_bar_config(disable=True)

        pipeline_kwargs_override = None
        actual_pipeline = pipeline
        if val_cond_clip_features is not None:
            device = self.accelerator.device
            image_proj = unwrapped_wrapper.image_proj.to(device)

            all_ip_tokens = []
            per_image_overrides = []

            pe = self._cached_prompt_embeds.to(device)
            am = self._cached_prompt_attention_mask.to(device)
            neg_pe = self._cached_negative_prompt_embeds.to(device)
            neg_am = self._cached_negative_prompt_attention_mask.to(device)

            for i, feat in enumerate(val_cond_clip_features):
                ip_tokens = image_proj(feat.to(device))
                all_ip_tokens.append(ip_tokens)

                override = {
                    "prompt_embeds": pe,
                    "negative_prompt_embeds": neg_pe,
                    "prompt_attention_mask": am,
                    "negative_prompt_attention_mask": neg_am,
                    "_ip_token_idx": i,
                }
                per_image_overrides.append(override)

            pipeline_kwargs_override = per_image_overrides
            actual_pipeline = _IPAdapterPipelineProxy(
                pipeline, unwrapped_wrapper._ip_processors, all_ip_tokens,
            )

        val_loop.run(
            actual_pipeline,
            self.global_step,
            self.tb_logger,
            device=self.accelerator.device,
            accelerator=self.accelerator,
            pipeline_kwargs_override=pipeline_kwargs_override,
            ground_truth_images=val_gt_images,
            conditioning_images=val_cond_images,
        )

        for proc in unwrapped_wrapper._ip_processors.values():
            proc._ip_hidden_states = None
        del pipeline

        self.ip_adapter_wrapper.train()

    # ── Checkpoint ───────────────────────────────────────────────────

    def _save_checkpoint(
        self, optimizer, lr_scheduler,
        ema_ip_adapter: EMAModel | None = None,
    ):
        if not self.accelerator.is_main_process:
            return

        unwrapped_wrapper = self.accelerator.unwrap_model(self.ip_adapter_wrapper)

        self.ckpt_manager.save(
            step=self.global_step,
            global_epoch=self.global_epoch,
            transformer=unwrapped_wrapper.transformer,
            accelerator=self.accelerator,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            seed=self.training_cfg.get("seed", 42),
            is_lora=self.use_lora,
        )

        ckpt_dir = self.ckpt_manager.get_latest_checkpoint()
        if ckpt_dir:
            ipa_state = unwrapped_wrapper.get_ip_adapter_state_dict()
            ipa_path = os.path.join(ckpt_dir, "ip_adapter.pt")
            torch.save(ipa_state, ipa_path)
            logger.info(f"IP-Adapter weights saved to {ipa_path}")

            if ema_ip_adapter is not None:
                ema_path = os.path.join(ckpt_dir, "ema_ip_adapter.pt")
                torch.save(ema_ip_adapter.state_dict(), ema_path)
                logger.info(f"EMA IP-Adapter weights saved to {ema_path}")
