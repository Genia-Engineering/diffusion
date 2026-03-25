"""训练器基类 — accelerate 初始化、优化器配置、显存优化、损失计算、抽象训练接口。"""

import logging
import math
import os
from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from diffusers.optimization import get_scheduler
from tqdm import tqdm

from utils.logger import TensorBoardLogger
from utils.checkpoint import CheckpointManager
from utils.memory import apply_memory_optimizations, compute_grad_norm, enable_tf32

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """训练器抽象基类。"""

    def __init__(self, config: DictConfig):
        self.config = config
        self.training_cfg = config.training
        self.logging_cfg = config.get("logging", {})

        if self.training_cfg.get("allow_tf32", True):
            enable_tf32()

        # DeepSpeed with deepspeed_config_file 禁止在 Accelerator() 中显式传
        # mixed_precision / gradient_accumulation_steps（由 deepspeed JSON 的 auto 字段接管）
        import os
        use_deepspeed = os.environ.get("ACCELERATE_USE_DEEPSPEED", "false").lower() == "true"
        if use_deepspeed:
            self.accelerator = Accelerator()
        else:
            self.accelerator = Accelerator(
                mixed_precision=self.training_cfg.get("mixed_precision", "bf16"),
                gradient_accumulation_steps=self.training_cfg.get("gradient_accumulation_steps", 1),
            )

        output_dir = self.training_cfg.get("output_dir", "./outputs")
        default_log_dir = os.path.join(output_dir, "tensorboard")
        self.tb_logger = TensorBoardLogger(
            log_dir=self.logging_cfg.get("log_dir", default_log_dir),
            is_main_process=self.accelerator.is_main_process,
            log_every_n_steps=self.logging_cfg.get("log_every_n_steps", 10),
        )

        self.ckpt_manager = CheckpointManager(
            save_dir=os.path.join(self.training_cfg.get("output_dir", "./outputs"), "checkpoints"),
            keep_last_n=self.training_cfg.get("keep_last_n_checkpoints", 3),
        )

        self.global_step = 0
        self.global_epoch = 0

        # 损失函数配置
        self.loss_type: str = self.training_cfg.get("loss_type", "mse")
        self.huber_delta: float = float(self.training_cfg.get("huber_delta", 0.1))

    def setup_optimizer(self, trainable_params, text_encoder_params=None) -> torch.optim.Optimizer:
        """配置优化器 — 支持 8-bit AdamW 和分组学习率。"""
        use_8bit = self.training_cfg.get("use_8bit_adam", True)
        lr = float(self.training_cfg.learning_rate)
        te_lr = float(
            self.training_cfg.get("projector_lr",
                self.config.get("lora", {}).get("text_encoder_lr", lr))
        )

        param_groups = [{"params": trainable_params, "lr": lr}]
        if text_encoder_params:
            param_groups.append({"params": text_encoder_params, "lr": te_lr})

        is_deepspeed = (
            hasattr(self.accelerator.state, "deepspeed_plugin")
            and self.accelerator.state.deepspeed_plugin is not None
        )

        if use_8bit and not is_deepspeed:
            try:
                import bitsandbytes as bnb
                optimizer = bnb.optim.AdamW8bit(
                    param_groups,
                    betas=(
                        self.training_cfg.get("adam_beta1", 0.9),
                        self.training_cfg.get("adam_beta2", 0.999),
                    ),
                    weight_decay=self.training_cfg.get("adam_weight_decay", 0.01),
                    eps=float(self.training_cfg.get("adam_epsilon", 1e-8)),
                )
                logger.info("Using 8-bit AdamW optimizer (bitsandbytes)")
                return optimizer
            except ImportError:
                logger.warning("bitsandbytes not available, falling back to standard AdamW")
        elif use_8bit and is_deepspeed:
            logger.info("DeepSpeed active — skipping 8-bit Adam (incompatible), using standard AdamW")

        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(
                self.training_cfg.get("adam_beta1", 0.9),
                self.training_cfg.get("adam_beta2", 0.999),
            ),
            weight_decay=self.training_cfg.get("adam_weight_decay", 0.01),
            eps=float(self.training_cfg.get("adam_epsilon", 1e-8)),
        )
        return optimizer

    def setup_lr_scheduler(self, optimizer, num_training_steps: int):
        """配置学习率调度器 — 默认 cosine with warmup。

        注意：BucketSampler 已按 rank 分片，lr_scheduler.step() 每个 global step
        只调用 1 次（不随 num_processes 倍增），因此不乘以 num_processes。
        """
        warmup_steps = self.training_cfg.get("lr_warmup_steps", 100)
        scheduler_type = self.training_cfg.get("lr_scheduler", "cosine")

        lr_scheduler = get_scheduler(
            scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
        return lr_scheduler

    def log_step(self, loss: float, lr: float, grad_norm: float = None):
        """记录当前步的训练指标到 TensorBoard。"""
        self.tb_logger.log_loss(loss, self.global_step)
        self.tb_logger.log_lr(lr, self.global_step)
        if grad_norm is not None:
            self.tb_logger.log_grad_norm(grad_norm, self.global_step)

    @abstractmethod
    def _freeze_parameters(self):
        """冻结/解冻模型参数 — 子类实现。"""
        ...

    @abstractmethod
    def _build_dataloader(self) -> DataLoader:
        """构建训练 DataLoader — 子类实现。"""
        ...

    @abstractmethod
    def train(self):
        """主训练循环 — 子类实现。"""
        ...

    def print_trainable_params(self, *models: nn.Module):
        """打印可训练参数量统计。"""
        total_params = 0
        trainable_params = 0
        for model in models:
            if model is None:
                continue
            for p in model.parameters():
                total_params += p.numel()
                if p.requires_grad:
                    trainable_params += p.numel()

        pct = 100 * trainable_params / total_params if total_params > 0 else 0
        logger.info(
            f"Trainable params: {trainable_params:,} / {total_params:,} ({pct:.2f}%)"
        )

    # ── 共用损失计算 ─────────────────────────────────────────────────────

    def compute_loss(
        self,
        noise_pred: torch.Tensor,
        target: torch.Tensor,
        timesteps: torch.Tensor = None,
        snr_cache: torch.Tensor = None,
        min_snr_gamma: float = 0.0,
        prediction_type: str = "epsilon",
        spatial_weights: torch.Tensor = None,
    ) -> torch.Tensor:
        """统一损失计算 — 支持 MSE / Huber，可选 Min-SNR 加权和空间加权。

        Args:
            noise_pred: 模型预测 (B, C, H, W)
            target: 目标噪声或 v-prediction (B, C, H, W)
            timesteps: 当前时间步 (B,)，Min-SNR 加权时必需
            snr_cache: 预计算的 SNR 查找表 (T,)
            min_snr_gamma: Min-SNR gamma 值，0 表示不使用
            prediction_type: "epsilon" 或 "v_prediction"
            spatial_weights: 空间加权图 (B, 1, H, W)，与 latent 同分辨率；
                             None 时退化为均匀权重（行为不变）
        """
        pred_f = noise_pred.float()
        tgt_f = target.float()

        if min_snr_gamma > 0 and snr_cache is not None and timesteps is not None:
            per_sample = self._per_sample_loss(pred_f, tgt_f, spatial_weights)
            snr = snr_cache.to(timesteps.device)[timesteps]
            if prediction_type == "v_prediction":
                weights = torch.clamp(snr, max=min_snr_gamma) / (snr + 1.0)
            else:
                weights = torch.clamp(snr, max=min_snr_gamma) / snr
            return (per_sample * weights).mean()

        if spatial_weights is not None:
            per_pixel = self._per_pixel_loss(pred_f, tgt_f)
            return (per_pixel * spatial_weights).mean()

        if self.loss_type == "huber":
            return F.huber_loss(pred_f, tgt_f, reduction="mean", delta=self.huber_delta)
        return F.mse_loss(pred_f, tgt_f, reduction="mean")

    def _per_pixel_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """逐像素损失 (B, C, H, W)，用于空间加权。"""
        if self.loss_type == "huber":
            return F.huber_loss(pred, target, reduction="none", delta=self.huber_delta)
        return F.mse_loss(pred, target, reduction="none")

    def _per_sample_loss(
        self, pred: torch.Tensor, target: torch.Tensor,
        spatial_weights: torch.Tensor = None,
    ) -> torch.Tensor:
        """逐样本损失 (B,)，用于 Min-SNR 加权。可叠加空间加权。"""
        per_pixel = self._per_pixel_loss(pred, target)
        if spatial_weights is not None:
            per_pixel = per_pixel * spatial_weights
        return per_pixel.mean(dim=list(range(1, pred.ndim)))

    # ── 文本嵌入预缓存 ────────────────────────────────────────────────────

    def _precompute_text_embeddings(self) -> None:
        """预计算固定 caption 及验证负面提示的文本嵌入，持久化到数据目录，卸载文本编码器释放显存。

        缓存路径：{train_data_dir}/text_embed_cache/embeds_{model_type}_{hash}.pt
        hash 基于 caption + negative_prompt，任一变化自动使用新缓存文件。

        缓存内容：
          SD1.5:        prompt_embeds, negative_prompt_embeds
          SDXL:         prompt_embeds, pooled_prompt_embeds,
                        negative_prompt_embeds, negative_pooled_prompt_embeds
          PixArt-Sigma: prompt_embeds, prompt_attention_mask,
                        negative_prompt_embeds, negative_prompt_attention_mask

        依赖子类属性: self.model_type, self.tokenizer, self.tokenizer_2,
                      self.text_encoder, self.text_encoder_2,
                      self._encode_prompt_sd15(), self._encode_prompt_sdxl()
        """
        import hashlib

        caption = self.config.data.get("caption", "")
        neg_prompt = self.config.get("validation", {}).get("negative_prompt", "")
        device = self.accelerator.device

        data_dir = self.config.data.get("train_data_dir", "./data")
        cache_dir = Path(data_dir) / "text_embed_cache"
        # hash 同时包含正负提示，任一变化时使用新缓存文件
        embed_hash = hashlib.md5(f"{caption}||{neg_prompt}".encode()).hexdigest()[:8]
        cache_file = cache_dir / f"embeds_{self.model_type}_{embed_hash}.pt"

        if cache_file.exists():
            logger.info(f"从磁盘加载文本嵌入缓存：{cache_file}")
            cached = torch.load(cache_file, map_location=device, weights_only=True)
            self._cached_prompt_embeds = cached["prompt_embeds"]
            self._cached_negative_prompt_embeds = cached["negative_prompt_embeds"]
            if "pooled_prompt_embeds" in cached:
                self._cached_pooled_prompt_embeds = cached["pooled_prompt_embeds"]
                self._cached_negative_pooled_prompt_embeds = cached["negative_pooled_prompt_embeds"]
            if "prompt_attention_mask" in cached:
                self._cached_prompt_attention_mask = cached["prompt_attention_mask"]
                self._cached_negative_prompt_attention_mask = cached["negative_prompt_attention_mask"]
        else:
            if self.accelerator.is_main_process:
                cache_dir.mkdir(parents=True, exist_ok=True)

            if self.model_type == "pixart_sigma":
                t5_max_len = 300

                def _encode_pixart(text):
                    tokenized = self.tokenizer(
                        text, padding="max_length", truncation=True,
                        max_length=t5_max_len, return_tensors="pt",
                    )
                    ids = tokenized.input_ids.to(device)
                    attn_mask = tokenized.attention_mask.to(device)
                    with torch.no_grad():
                        embeds = self.text_encoder(ids, attention_mask=attn_mask)[0]
                    return embeds, attn_mask

                embeds, attn_mask = _encode_pixart(caption)
                neg_embeds, neg_attn_mask = _encode_pixart(neg_prompt)

                self._cached_prompt_embeds = embeds
                self._cached_prompt_attention_mask = attn_mask
                self._cached_negative_prompt_embeds = neg_embeds
                self._cached_negative_prompt_attention_mask = neg_attn_mask

                if self.accelerator.is_main_process:
                    torch.save(
                        {
                            "prompt_embeds": embeds.cpu(),
                            "prompt_attention_mask": attn_mask.cpu(),
                            "negative_prompt_embeds": neg_embeds.cpu(),
                            "negative_prompt_attention_mask": neg_attn_mask.cpu(),
                        },
                        cache_file,
                    )

            elif self.model_type == "sdxl":
                def _encode_sdxl(text):
                    ids_1 = self.tokenizer(
                        text, padding="max_length", truncation=True,
                        max_length=self.tokenizer.model_max_length,
                        return_tensors="pt",
                    ).input_ids.to(device)
                    ids_2 = self.tokenizer_2(
                        text, padding="max_length", truncation=True,
                        max_length=self.tokenizer_2.model_max_length,
                        return_tensors="pt",
                    ).input_ids.to(device)
                    with torch.no_grad():
                        return self._encode_prompt_sdxl(ids_1, ids_2)

                embeds, pooled = _encode_sdxl(caption)
                neg_embeds, neg_pooled = _encode_sdxl(neg_prompt)

                self._cached_prompt_embeds = embeds                    # (1, seq, 2048)
                self._cached_pooled_prompt_embeds = pooled             # (1, 1280)
                self._cached_negative_prompt_embeds = neg_embeds
                self._cached_negative_pooled_prompt_embeds = neg_pooled

                if self.accelerator.is_main_process:
                    torch.save(
                        {
                            "prompt_embeds": embeds.cpu(),
                            "pooled_prompt_embeds": pooled.cpu(),
                            "negative_prompt_embeds": neg_embeds.cpu(),
                            "negative_pooled_prompt_embeds": neg_pooled.cpu(),
                        },
                        cache_file,
                    )
            else:
                def _encode_sd15(text):
                    ids = self.tokenizer(
                        text, padding="max_length", truncation=True,
                        max_length=self.tokenizer.model_max_length,
                        return_tensors="pt",
                    ).input_ids.to(device)
                    with torch.no_grad():
                        return self._encode_prompt_sd15(ids)   # (1, seq, 768)

                embeds = _encode_sd15(caption)
                neg_embeds = _encode_sd15(neg_prompt)

                self._cached_prompt_embeds = embeds
                self._cached_negative_prompt_embeds = neg_embeds

                if self.accelerator.is_main_process:
                    torch.save(
                        {
                            "prompt_embeds": embeds.cpu(),
                            "negative_prompt_embeds": neg_embeds.cpu(),
                        },
                        cache_file,
                    )

            logger.info(f"文本嵌入（正/负）已保存至：{cache_file}")

        del self.text_encoder
        self.text_encoder = None
        if self.model_type == "sdxl":
            del self.text_encoder_2
            self.text_encoder_2 = None
        # tokenizer 本身内存极小（词表映射，无 GPU 参数），不删除；
        # 后续 _build_dataloader() 中 PixArtSigmaCachedLatentDataset 仍需要用它做 caption tokenize。
        torch.cuda.empty_cache()
        logger.info("文本编码器已卸载以释放显存")

    # ── VAE Latent 预缓存 ─────────────────────────────────────────────────

    def _get_latent_cache_dir(self) -> str:
        """返回 latent 缓存目录路径。

        优先级：training.latent_cache_dir（子配置显式指定）
               → {train_data_dir}/../latent_cache（train_data_dir 同级目录）
        例如 train_data_dir = /data/overfit_test/train → /data/overfit_test/latent_cache
        """
        explicit = self.training_cfg.get("latent_cache_dir", None)
        if explicit:
            return explicit
        data_dir = self.config.data.get("train_data_dir", "./data")
        parent = os.path.dirname(os.path.normpath(data_dir))
        return os.path.join(parent, "latent_cache")

    def _precompute_latents(self, cache_dir: str, *, delete_encoder: bool = True) -> None:
        """预计算所有训练图片的 VAE latent — 多卡并行 + 批量 encode + 按图片名缓存。

        设计要点：
        - 所有 Rank 均匀分担：Rank i 处理索引 [i, i+P, i+2P, ...]（步长 = 进程数）
        - 同一 bucket（相同分辨率）的图片打包为 batch 批量送入 VAE
        - 缓存文件名 = {image_stem}.pt，与原图文件名保持一致，便于人工排查
        - 完成后所有 Rank 通过 wait_for_everyone() 同步，再统一卸载 VAE encoder
        """
        import torchvision.transforms as T
        import torchvision.transforms.functional as TF
        from collections import defaultdict
        from PIL import Image as PIL_Image

        from data.buckets import BucketManager
        from data.dataset import BaseImageDataset
        from data.transforms import AspectRatioResize

        cache_dir_path = Path(cache_dir)
        cache_dir_path.mkdir(parents=True, exist_ok=True)

        data_cfg = self.config.data
        resolution = data_cfg.get("resolution", 512 if self.model_type == "sd15" else 1024)
        center_crop = data_cfg.get("center_crop", False)
        vae_batch_size = self.training_cfg.get("latent_cache_batch_size", 4)

        num_processes = self.accelerator.num_processes
        process_index = self.accelerator.process_index
        device = self.accelerator.device

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

        n_total = len(temp_dataset)

        # 步长分配：Rank i 处理索引 [i, i+P, i+2P, ...]
        my_indices = list(range(process_index, n_total, num_processes))
        todo_indices = [
            i for i in my_indices
            if not (cache_dir_path / f"{temp_dataset.image_paths[i].stem}.pt").exists()
        ]

        if not todo_indices:
            logger.info(
                f"[Rank {process_index}/{num_processes}] "
                f"VAE latent 缓存已完整（负责 {len(my_indices)} 张），跳过预计算"
            )
        else:
            logger.info(
                f"[Rank {process_index}/{num_processes}] "
                f"预计算 {len(todo_indices)}/{n_total} 张图片的 VAE latents "
                f"(batch_size={vae_batch_size}) → {cache_dir}"
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

            # 按 target_size 分组，保证同一 batch 内分辨率一致
            size_groups: dict[tuple, list[int]] = defaultdict(list)
            for idx in todo_indices:
                size_groups[temp_dataset._get_target_size(idx)].append(idx)

            try:
                with tqdm(
                    total=len(todo_indices),
                    desc=f"[Rank {process_index}] 预计算 latents",
                    disable=not self.accelerator.is_main_process,
                ) as pbar:
                    for target_size, indices in size_groups.items():
                        target_w, target_h = target_size
                        resizer = AspectRatioResize(target_size, center_crop=center_crop)

                        for batch_start in range(0, len(indices), vae_batch_size):
                            batch_indices = indices[batch_start: batch_start + vae_batch_size]

                            imgs_normal, imgs_flip, orig_sizes = [], [], []
                            for idx in batch_indices:
                                image = PIL_Image.open(temp_dataset.image_paths[idx]).convert("RGB")
                                orig_w, orig_h = image.size
                                orig_sizes.append((orig_h, orig_w))
                                image_resized = resizer(image)
                                imgs_normal.append(normalize(to_tensor(image_resized)))
                                imgs_flip.append(normalize(to_tensor(TF.hflip(image_resized))))

                            batch_t = torch.stack(imgs_normal).to(device)
                            batch_t_flip = torch.stack(imgs_flip).to(device)

                            with torch.no_grad():
                                latents = self.vae.encode(batch_t).latent_dist.mode()
                                latents = (latents * self.vae.config.scaling_factor).to(torch.float16).cpu()
                                latents_flip = self.vae.encode(batch_t_flip).latent_dist.mode()
                                latents_flip = (latents_flip * self.vae.config.scaling_factor).to(torch.float16).cpu()

                            for j, idx in enumerate(batch_indices):
                                orig_h, orig_w = orig_sizes[j]
                                torch.save(
                                    {
                                        "latent": latents[j],
                                        "latent_flip": latents_flip[j],
                                        "original_hw": torch.tensor([orig_h, orig_w], dtype=torch.long),
                                        "target_hw": torch.tensor([target_h, target_w], dtype=torch.long),
                                    },
                                    cache_dir_path / f"{temp_dataset.image_paths[idx].stem}.pt",
                                )
                            pbar.update(len(batch_indices))
            finally:
                if use_slicing:
                    self.vae.enable_slicing()
                if use_tiling:
                    self.vae.enable_tiling()

            logger.info(f"[Rank {process_index}] VAE latents 预计算完成")

        # 所有 Rank 完成后同步，再统一卸载 encoder
        self.accelerator.wait_for_everyone()
        if delete_encoder:
            self._delete_vae_encoder()

    def _delete_vae_encoder(self) -> None:
        """卸载 VAE encoder 子模块以释放显存（保留 decoder 供 validation 使用）。"""
        process_index = self.accelerator.process_index
        if hasattr(self.vae, "encoder"):
            del self.vae.encoder
        if hasattr(self.vae, "quant_conv"):
            del self.vae.quant_conv
        torch.cuda.empty_cache()
        logger.info(f"[Rank {process_index}] VAE encoder 已卸载（保留 decoder 供 validation 使用）")

    def _precompute_latents_distributed(self, cache_dir: str, *, delete_encoder: bool = True) -> None:
        """分布式 latent 预计算入口（所有 Rank 并行，内部已包含同步逻辑）。"""
        self._precompute_latents(cache_dir, delete_encoder=delete_encoder)
