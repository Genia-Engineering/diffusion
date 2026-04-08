"""训练器基类 — accelerate 初始化、优化器配置、显存优化、损失计算、抽象训练接口。"""

import logging
import math
import os
from abc import ABC, abstractmethod
from datetime import timedelta
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs
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
        nccl_timeout_min = int(self.training_cfg.get("nccl_timeout_minutes", 60))
        init_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=nccl_timeout_min))
        if use_deepspeed:
            self.accelerator = Accelerator(kwargs_handlers=[init_kwargs])
        else:
            self.accelerator = Accelerator(
                mixed_precision=self.training_cfg.get("mixed_precision", "bf16"),
                gradient_accumulation_steps=self.training_cfg.get("gradient_accumulation_steps", 1),
                kwargs_handlers=[init_kwargs],
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

        # EMA loss 追踪
        self._ema_loss: float | None = None
        self._ema_decay: float = float(self.training_cfg.get("ema_loss_decay", 0.99))

        # 损失函数配置
        self.loss_type: str = self.training_cfg.get("loss_type", "mse")
        self.huber_delta: float = float(self.training_cfg.get("huber_delta", 0.1))

    def setup_optimizer(
        self,
        trainable_params=None,
        text_encoder_params=None,
        param_groups=None,
    ) -> torch.optim.Optimizer:
        """配置优化器 — 支持 8-bit AdamW 和分组学习率。

        Args:
            trainable_params: 可训练参数列表（与 param_groups 互斥）
            text_encoder_params: 文本编码器参数（可选第二组，使用 text_encoder_lr）
            param_groups: 自定义参数组 [{"params": [...], "lr": float}, ...]，
                          传入时忽略 trainable_params / text_encoder_params
        """
        use_8bit = self.training_cfg.get("use_8bit_adam", True)
        lr = float(self.training_cfg.learning_rate)

        if param_groups is None:
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

        支持通过 lr_min_ratio 设置 cosine 衰减的最小学习率下限:
          lr_min_ratio: 0.0 ~ 1.0，最小 lr = peak_lr × lr_min_ratio
          默认 0.0 即衰减到 0（与原行为一致）

        注意：BucketSampler 已按 rank 分片，lr_scheduler.step() 每个 global step
        只调用 1 次（不随 num_processes 倍增），因此不乘以 num_processes。
        """
        warmup_steps = self.training_cfg.get("lr_warmup_steps", 100)
        scheduler_type = self.training_cfg.get("lr_scheduler", "cosine")
        min_lr_ratio = float(self.training_cfg.get("lr_min_ratio", 0.0))

        if min_lr_ratio > 0 and scheduler_type == "cosine":
            return self._cosine_with_min_lr(
                optimizer, warmup_steps, num_training_steps, min_lr_ratio,
            )

        lr_scheduler = get_scheduler(
            scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
        return lr_scheduler

    @staticmethod
    def _cosine_with_min_lr(
        optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float,
    ):
        """Cosine annealing with linear warmup and non-zero floor lr.

        每个 param_group 独立计算: min_lr = group['lr'] × min_lr_ratio，
        因此双学习率场景下各组保持正确的衰减比例。
        """
        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return current_step / max(warmup_steps, 1)
            decay_steps = total_steps - warmup_steps
            progress = min((current_step - warmup_steps) / max(decay_steps, 1), 1.0)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

        return LambdaLR(optimizer, lr_lambda)

    def _update_ema_loss(self, loss: float):
        """更新 loss 的指数移动平均并记录到 TensorBoard。"""
        if self._ema_loss is None:
            self._ema_loss = loss
        else:
            self._ema_loss = self._ema_decay * self._ema_loss + (1 - self._ema_decay) * loss
        self.tb_logger.log_ema_loss(self._ema_loss, self.global_step)

    def log_step(self, loss: float, lr: float, grad_norm: float = None,
                 projector_lr: float = None,
                 tf_grad_norm: float = None, proj_grad_norm: float = None):
        """记录当前步的训练指标到 TensorBoard。"""
        self.tb_logger.log_loss(loss, self.global_step)
        self.tb_logger.log_lr(lr, self.global_step)
        if projector_lr is not None:
            self.tb_logger.log_projector_lr(projector_lr, self.global_step)
        if grad_norm is not None:
            self.tb_logger.log_grad_norm(grad_norm, self.global_step)
        if tf_grad_norm is not None and proj_grad_norm is not None:
            self.tb_logger.log_grad_norm_group(tf_grad_norm, proj_grad_norm, self.global_step)

        self._update_ema_loss(loss)

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

            if self.model_type in ("pixart_sigma", "sana"):
                max_seq_len = 300

                def _encode_seq2seq(text):
                    tokenized = self.tokenizer(
                        text, padding="max_length", truncation=True,
                        max_length=max_seq_len, return_tensors="pt",
                    )
                    ids = tokenized.input_ids.to(device)
                    attn_mask = tokenized.attention_mask.to(device)
                    with torch.no_grad():
                        embeds = self.text_encoder(ids, attention_mask=attn_mask)[0]
                    return embeds, attn_mask

                embeds, attn_mask = _encode_seq2seq(caption)
                neg_embeds, neg_attn_mask = _encode_seq2seq(neg_prompt)

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

        # ── 预缓存验证 prompt 嵌入（多 prompt 验证支持）──────────────────────
        # 验证 prompt 可能与训练 caption 不同，需在编码器卸载前完成编码。
        # prompt_source="data" 时使用训练 caption，无需额外编码。
        self._cache_validation_prompt_embeds(cache_dir, device)

        del self.text_encoder
        self.text_encoder = None
        if self.model_type == "sdxl":
            del self.text_encoder_2
            self.text_encoder_2 = None
        # tokenizer 本身内存极小（词表映射，无 GPU 参数），不删除；
        # 后续 _build_dataloader() 中 PixArtSigmaCachedLatentDataset 仍需要用它做 caption tokenize。
        torch.cuda.empty_cache()
        logger.info("文本编码器已卸载以释放显存")

    @torch.no_grad()
    def _cache_validation_prompt_embeds(self, cache_dir: Path, device: torch.device) -> None:
        """预缓存所有验证 prompt 的文本嵌入（支持多 prompt 验证对比）。

        当 validation.prompt_source="config" 且 prompts 列表中有多个不同提示词时，
        逐一编码并缓存。验证推理时通过 _cached_val_prompt_embeds_list 传递给 pipeline，
        每个 prompt 使用独立的嵌入。

        缓存文件: {cache_dir}/val_embeds_{model_type}_{hash}.pt
        hash 基于所有验证 prompt + negative_prompt，任一变化自动使用新缓存。
        """
        import hashlib

        val_cfg = self.config.get("validation", {})
        prompt_source = val_cfg.get("prompt_source", "config")
        if prompt_source == "data":
            return

        caption = self.config.data.get("caption", "")
        val_prompts = list(val_cfg.get("prompts", [caption]))
        neg_prompt = val_cfg.get("negative_prompt", "")

        if len(val_prompts) <= 1 and (not val_prompts or val_prompts[0] == caption):
            return

        embed_hash = hashlib.md5(
            ("||".join(val_prompts) + "||" + neg_prompt).encode()
        ).hexdigest()[:8]
        val_cache_file = cache_dir / f"val_embeds_{self.model_type}_{embed_hash}.pt"

        if val_cache_file.exists():
            logger.info(f"从磁盘加载验证 prompt 嵌入缓存：{val_cache_file}")
            cached = torch.load(val_cache_file, map_location="cpu", weights_only=True)
            self._cached_val_prompt_embeds_list = cached["val_prompt_embeds_list"]
            return

        if self.text_encoder is None:
            logger.warning("文本编码器已卸载，无法编码验证 prompt；将使用训练 caption 嵌入替代")
            return

        if next(self.text_encoder.parameters()).device != device:
            self.text_encoder.to(device)

        neg_embeds = self._cached_negative_prompt_embeds
        neg_mask = getattr(self, "_cached_negative_prompt_attention_mask", None)

        val_embeds_list = []

        if self.model_type in ("pixart_sigma", "sana"):
            max_seq_len = 300
            for vp in val_prompts:
                tokenized = self.tokenizer(
                    vp, padding="max_length", truncation=True,
                    max_length=max_seq_len, return_tensors="pt",
                )
                ids = tokenized.input_ids.to(device)
                attn_mask = tokenized.attention_mask.to(device)
                embeds = self.text_encoder(ids, attention_mask=attn_mask)[0]
                d = {
                    "prompt_embeds": embeds.cpu(),
                    "prompt_attention_mask": attn_mask.cpu(),
                    "negative_prompt_embeds": neg_embeds.cpu() if torch.is_tensor(neg_embeds) else neg_embeds,
                    "negative_prompt_attention_mask": neg_mask.cpu() if torch.is_tensor(neg_mask) else neg_mask,
                }
                val_embeds_list.append(d)

        elif self.model_type == "sdxl":
            for vp in val_prompts:
                ids_1 = self.tokenizer(
                    vp, padding="max_length", truncation=True,
                    max_length=self.tokenizer.model_max_length, return_tensors="pt",
                ).input_ids.to(device)
                ids_2 = self.tokenizer_2(
                    vp, padding="max_length", truncation=True,
                    max_length=self.tokenizer_2.model_max_length, return_tensors="pt",
                ).input_ids.to(device)
                embeds, pooled = self._encode_prompt_sdxl(ids_1, ids_2)
                d = {
                    "prompt_embeds": embeds.cpu(),
                    "pooled_prompt_embeds": pooled.cpu(),
                    "negative_prompt_embeds": self._cached_negative_prompt_embeds.cpu()
                        if torch.is_tensor(self._cached_negative_prompt_embeds)
                        else self._cached_negative_prompt_embeds,
                    "negative_pooled_prompt_embeds": self._cached_negative_pooled_prompt_embeds.cpu()
                        if torch.is_tensor(self._cached_negative_pooled_prompt_embeds)
                        else self._cached_negative_pooled_prompt_embeds,
                }
                val_embeds_list.append(d)

        if val_embeds_list:
            self._cached_val_prompt_embeds_list = val_embeds_list
            if self.accelerator.is_main_process:
                torch.save({"val_prompt_embeds_list": val_embeds_list}, val_cache_file)
            logger.info(
                f"验证 prompt 嵌入已缓存：{len(val_embeds_list)} 个 prompt → {val_cache_file}"
            )

    # ── 逐图片文本嵌入预缓存 ──────────────────────────────────────────────

    def _get_text_embed_per_image_cache_dir(self) -> str:
        """返回逐图片文本嵌入缓存目录。

        优先级: training.text_embed_cache_dir
               → {train_data_dir}/../text_embed_per_image_cache_{model_type}

        目录名包含 model_type 以避免不同文本编码器（如 T5-XXL 4096d vs Gemma2 2304d）
        产生的嵌入维度不同导致缓存冲突。
        """
        explicit = self.training_cfg.get("text_embed_cache_dir", None)
        if explicit:
            return explicit
        data_dir = self.config.data.get("train_data_dir", "./data")
        parent = os.path.dirname(os.path.normpath(data_dir))
        model_type = getattr(self, "model_type", "default")
        return os.path.join(parent, f"text_embed_per_image_cache_{model_type}")

    def _per_image_text_embed_cache_exists(self, cache_dir: str) -> bool:
        return (Path(cache_dir) / ".precompute_done").exists()

    def _load_per_image_val_captions(self, n: int, shuffle_seed: int | None = None) -> list[str]:
        """从训练图对应的描述文件中读取 n 条 caption，用作逐图验证提示词。

        shuffle_seed 不为 None 时随机采样（使用固定 seed 保证可复现），否则取排序后前 n 条。
        采样的图片索引同时写入 self._val_sample_image_indices，供 gt 图片加载对齐使用。
        """
        import random
        from data.dataset import IMAGE_EXTENSIONS

        data_cfg = self.config.data
        caption_dir = Path(data_cfg.caption_dir)
        stem_replace_cfg = data_cfg.get("caption_stem_replace", {})
        stem_from = stem_replace_cfg.get("from", "")
        stem_to = stem_replace_cfg.get("to", "")
        fallback = data_cfg.get("caption_fallback", data_cfg.get("caption", ""))

        image_dir = Path(data_cfg.train_data_dir)
        all_image_paths = sorted(
            p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )

        total = len(all_image_paths)
        n = min(n, total)
        if shuffle_seed is not None:
            selected_indices = sorted(random.Random(shuffle_seed).sample(range(total), n))
            logger.info(f"随机采样 {n} 张训练图用于验证 (seed={shuffle_seed}, 共 {total} 张)")
        else:
            selected_indices = list(range(n))

        self._val_sample_image_indices = selected_indices

        captions = []
        for idx in selected_indices:
            stem = all_image_paths[idx].stem
            caption_stem = stem.replace(stem_from, stem_to) if stem_from else stem
            caption_file = caption_dir / f"{caption_stem}.txt"
            if caption_file.exists():
                captions.append(caption_file.read_text(encoding="utf-8").strip())
            else:
                captions.append(fallback)
        return captions

    @torch.no_grad()
    def _precompute_per_image_text_embeddings(self, cache_dir: str) -> None:
        """逐图片预编码文本嵌入 — 多卡并行 + 按图片名缓存。

        支持两种模型架构:
          - PixArt-Sigma (T5): 缓存 prompt_embeds + prompt_attention_mask
          - SDXL (双 CLIP):    缓存 prompt_embeds + pooled_prompt_embeds

        同时编码验证用的 prompt 和 negative prompt，完成后卸载文本编码器释放显存。
        缓存文件名 = {image_stem}.pt，与训练图片一一对应。
        """
        from data.dataset import IMAGE_EXTENSIONS

        cache_dir_path = Path(cache_dir)
        done_marker = cache_dir_path / ".precompute_done"
        is_sdxl = self.model_type == "sdxl"

        data_cfg = self.config.data
        caption_dir = Path(data_cfg.caption_dir)
        stem_replace_cfg = data_cfg.get("caption_stem_replace", {})
        stem_from = stem_replace_cfg.get("from", "")
        stem_to = stem_replace_cfg.get("to", "")
        fallback_caption = data_cfg.get("caption_fallback", data_cfg.get("caption", ""))

        image_dir = Path(data_cfg.train_data_dir)
        image_paths = sorted(
            p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        n_total = len(image_paths)

        if done_marker.exists():
            logger.info(f"逐图片文本嵌入缓存已完整（{n_total} 张），跳过编码")
        else:
            if self.accelerator.is_main_process:
                cache_dir_path.mkdir(parents=True, exist_ok=True)
            self.accelerator.wait_for_everyone()

            num_processes = self.accelerator.num_processes
            process_index = self.accelerator.process_index
            device = self.accelerator.device

            my_indices = list(range(process_index, n_total, num_processes))
            todo_indices = [
                i for i in my_indices
                if not (cache_dir_path / f"{image_paths[i].stem}.pt").exists()
            ]

            if not todo_indices:
                logger.info(
                    f"[Rank {process_index}/{num_processes}] "
                    f"逐图片文本嵌入缓存已完整（负责 {len(my_indices)} 张），跳过编码"
                )
            else:
                logger.info(
                    f"[Rank {process_index}/{num_processes}] "
                    f"预编码 {len(todo_indices)}/{n_total} 张图片的文本嵌入 → {cache_dir}"
                )
                for idx in tqdm(
                    todo_indices,
                    desc=f"[Rank {process_index}] 编码文本嵌入",
                    disable=not self.accelerator.is_main_process,
                ):
                    image_stem = image_paths[idx].stem
                    caption_stem = (
                        image_stem.replace(stem_from, stem_to) if stem_from else image_stem
                    )
                    caption_file = caption_dir / f"{caption_stem}.txt"

                    if caption_file.exists():
                        caption_text = caption_file.read_text(encoding="utf-8").strip()
                    else:
                        caption_text = fallback_caption
                        logger.warning(
                            f"描述文件缺失 {caption_file.name}，使用 fallback caption"
                        )

                    if is_sdxl:
                        ids_1 = self.tokenizer(
                            caption_text, padding="max_length", truncation=True,
                            max_length=self.tokenizer.model_max_length, return_tensors="pt",
                        ).input_ids.to(device)
                        ids_2 = self.tokenizer_2(
                            caption_text, padding="max_length", truncation=True,
                            max_length=self.tokenizer_2.model_max_length, return_tensors="pt",
                        ).input_ids.to(device)
                        embeds, pooled = self._encode_prompt_sdxl(ids_1, ids_2)
                        torch.save(
                            {
                                "prompt_embeds": embeds.cpu().to(torch.float16),
                                "pooled_prompt_embeds": pooled.cpu().to(torch.float16),
                            },
                            cache_dir_path / f"{image_stem}.pt",
                        )
                    else:
                        t5_max_len = 300
                        tokenized = self.tokenizer(
                            caption_text, padding="max_length", truncation=True,
                            max_length=t5_max_len, return_tensors="pt",
                        )
                        ids = tokenized.input_ids.to(device)
                        attn_mask = tokenized.attention_mask.to(device)
                        embeds = self.text_encoder(ids, attention_mask=attn_mask)[0]
                        torch.save(
                            {
                                "prompt_embeds": embeds.cpu().to(torch.float16),
                                "prompt_attention_mask": attn_mask.cpu(),
                            },
                            cache_dir_path / f"{image_stem}.pt",
                        )

                logger.info(f"[Rank {process_index}] 逐图片文本嵌入编码完成")

            self.accelerator.wait_for_everyone()

            if self.accelerator.is_main_process:
                n_cached = sum(
                    1 for p in image_paths if (cache_dir_path / f"{p.stem}.pt").exists()
                )
                if n_cached >= n_total:
                    done_marker.touch()
                    logger.info(f"全部完成！{n_total} 张图片文本嵌入已缓存至 {cache_dir}")
                else:
                    logger.warning(
                        f"文本嵌入缓存不完整：期望 {n_total} 张，实际 {n_cached} 张"
                    )

        # ── 编码验证 prompt 及 negative prompt（编码器卸载后验证推理仍需使用）──
        val_cfg = self.config.get("validation", {})
        neg_prompt = val_cfg.get("negative_prompt", "")
        val_prompts = list(val_cfg.get("prompts", []))
        val_prompt = val_prompts[0] if val_prompts else fallback_caption
        prompt_source = val_cfg.get("prompt_source", "config")

        per_image_val = getattr(self, "_per_image_caption", False) and not is_sdxl

        device = self.accelerator.device
        misc_cache_file = cache_dir_path / "__val_and_neg_prompts__.pt"

        # 如果缓存是旧格式（无逐图验证嵌入），但当前模式需要逐图验证，则删除旧缓存重新编码
        if misc_cache_file.exists() and per_image_val:
            _tmp = torch.load(misc_cache_file, map_location="cpu", weights_only=True)
            needs_recode = "val_prompt_embeds_list" not in _tmp
            # prompt_source="data" 时需要 val_image_indices 以保证 gt 图对齐；旧格式缺此字段则重新编码
            if not needs_recode and prompt_source == "data" and "val_image_indices" not in _tmp:
                needs_recode = True
            if needs_recode:
                logger.info("验证嵌入缓存格式不含逐图数据，将重新编码")
                if self.accelerator.is_main_process:
                    misc_cache_file.unlink()
                self.accelerator.wait_for_everyone()
            del _tmp

        # ── 验证/负面 prompt 编码：仅 Rank 0 执行，其余 Rank 等待后统一从文件加载 ──
        # 这样避免所有 Rank 同时把大型编码器（T5-XXL ~18GB）加载到 GPU 导致 OOM。
        if not misc_cache_file.exists():
            if self.accelerator.is_main_process:
                if is_sdxl and self.text_encoder is not None:
                    if next(self.text_encoder.parameters()).device != device:
                        self.text_encoder.to(device)
                    if self.text_encoder_2 is not None and next(self.text_encoder_2.parameters()).device != device:
                        self.text_encoder_2.to(device)

                    def _encode_sdxl(text):
                        _ids_1 = self.tokenizer(
                            text, padding="max_length", truncation=True,
                            max_length=self.tokenizer.model_max_length, return_tensors="pt",
                        ).input_ids.to(device)
                        _ids_2 = self.tokenizer_2(
                            text, padding="max_length", truncation=True,
                            max_length=self.tokenizer_2.model_max_length, return_tensors="pt",
                        ).input_ids.to(device)
                        _emb, _pooled = self._encode_prompt_sdxl(_ids_1, _ids_2)
                        return _emb.cpu(), _pooled.cpu()

                    val_embeds, val_pooled = _encode_sdxl(val_prompt)
                    neg_embeds, neg_pooled = _encode_sdxl(neg_prompt)
                    torch.save(
                        {
                            "val_prompt_embeds": val_embeds,
                            "val_pooled_prompt_embeds": val_pooled,
                            "negative_prompt_embeds": neg_embeds,
                            "negative_pooled_prompt_embeds": neg_pooled,
                        },
                        misc_cache_file,
                    )

                elif self.text_encoder is not None:
                    if next(self.text_encoder.parameters()).device != device:
                        self.text_encoder.to(device)

                    t5_max_len = 300

                    def _encode_t5(text):
                        tok = self.tokenizer(
                            text, padding="max_length", truncation=True,
                            max_length=t5_max_len, return_tensors="pt",
                        )
                        _ids = tok.input_ids.to(device)
                        _mask = tok.attention_mask.to(device)
                        _emb = self.text_encoder(_ids, attention_mask=_mask)[0]
                        return _emb.cpu(), _mask.cpu()

                    neg_embeds, neg_mask = _encode_t5(neg_prompt)

                    if per_image_val:
                        # prompt_source="data" 时从 num_val_prompts 取数量并随机采样；否则取前 N 条
                        if prompt_source == "data":
                            num_val = val_cfg.get("num_val_prompts", len(val_prompts) if val_prompts else 3)
                            shuffle_seed = val_cfg.get("seed", 42)
                        else:
                            num_val = len(val_prompts) if val_prompts else 3
                            shuffle_seed = None
                        val_captions = self._load_per_image_val_captions(num_val, shuffle_seed=shuffle_seed)
                        val_embeds_list = []
                        for vc in val_captions:
                            v_emb, v_mask = _encode_t5(vc)
                            val_embeds_list.append({
                                "prompt_embeds": v_emb,
                                "prompt_attention_mask": v_mask,
                                "negative_prompt_embeds": neg_embeds,
                                "negative_prompt_attention_mask": neg_mask,
                            })
                        logger.info(
                            f"编码了 {len(val_captions)} 条逐图验证 caption"
                            + (f"（随机采样 seed={shuffle_seed}）" if shuffle_seed is not None else "（顺序前 N 条）")
                        )
                        torch.save(
                            {
                                "val_prompt_embeds_list": val_embeds_list,
                                "negative_prompt_embeds": neg_embeds,
                                "negative_prompt_attention_mask": neg_mask,
                                "val_captions": val_captions,
                                "val_image_indices": self._val_sample_image_indices,
                            },
                            misc_cache_file,
                        )
                    else:
                        val_embeds, val_mask = _encode_t5(val_prompt)
                        torch.save(
                            {
                                "negative_prompt_embeds": neg_embeds,
                                "negative_prompt_attention_mask": neg_mask,
                                "val_prompt_embeds": val_embeds,
                                "val_prompt_attention_mask": val_mask,
                            },
                            misc_cache_file,
                        )

                else:
                    raise RuntimeError(
                        "无法编码验证/负面 prompt：text_encoder 已卸载且缓存文件不存在 "
                        f"({misc_cache_file})。请删除 {done_marker} 并重新运行。"
                    )

            # 等待 Rank 0 完成编码和写盘，再让所有 Rank 从文件加载
            self.accelerator.wait_for_everyone()

        # ── 所有 Rank（含 Rank 0）统一从缓存文件加载 ──
        if misc_cache_file.exists():
            logger.info(f"从磁盘加载验证/负面 prompt 嵌入缓存：{misc_cache_file}")
            cached = torch.load(misc_cache_file, map_location="cpu", weights_only=True)
            if is_sdxl:
                self._cached_prompt_embeds = cached["val_prompt_embeds"]
                self._cached_pooled_prompt_embeds = cached["val_pooled_prompt_embeds"]
                self._cached_negative_prompt_embeds = cached["negative_prompt_embeds"]
                self._cached_negative_pooled_prompt_embeds = cached["negative_pooled_prompt_embeds"]
            elif "val_prompt_embeds_list" in cached:
                self._cached_val_prompt_embeds_list = cached["val_prompt_embeds_list"]
                self._cached_negative_prompt_embeds = cached["negative_prompt_embeds"]
                self._cached_negative_prompt_attention_mask = cached["negative_prompt_attention_mask"]
                num_val = len(cached["val_prompt_embeds_list"])
                if "val_captions" in cached:
                    # 直接从缓存恢复 caption 文本及对应图片索引，无需重新读取文件
                    self._per_image_val_captions = cached["val_captions"]
                    self._val_sample_image_indices = cached.get(
                        "val_image_indices", list(range(num_val))
                    )
                    logger.info(
                        f"从缓存恢复 {num_val} 条逐图验证 caption "
                        f"(indices={self._val_sample_image_indices[:4]}...)"
                    )
                else:
                    self._per_image_val_captions = self._load_per_image_val_captions(num_val)
            else:
                self._cached_negative_prompt_embeds = cached["negative_prompt_embeds"]
                self._cached_negative_prompt_attention_mask = cached["negative_prompt_attention_mask"]
                self._cached_prompt_embeds = cached["val_prompt_embeds"]
                self._cached_prompt_attention_mask = cached["val_prompt_attention_mask"]
        else:
            raise RuntimeError(
                "验证/负面 prompt 嵌入缓存写入失败，Rank 0 未能正常完成编码。"
                f"请检查磁盘空间并重试（预期路径：{misc_cache_file}）。"
            )

        # 卸载文本编码器释放显存
        if self.text_encoder is not None:
            del self.text_encoder
            self.text_encoder = None
        if is_sdxl and hasattr(self, "text_encoder_2") and self.text_encoder_2 is not None:
            del self.text_encoder_2
            self.text_encoder_2 = None
        torch.cuda.empty_cache()
        logger.info("文本编码器已卸载以释放显存（逐图片文本嵌入已缓存）")

    # ── VAE Latent 预缓存 ─────────────────────────────────────────────────

    def _get_latent_cache_dir(self) -> str:
        """返回 latent 缓存目录路径。

        优先级：training.latent_cache_dir（子配置显式指定）
               → {train_data_dir}/../latent_cache_{model_type}（train_data_dir 同级目录）
        例如 train_data_dir = /data/overfit_test/train, model_type = sdxl
              → /data/overfit_test/latent_cache_sdxl

        目录名包含 model_type 以避免不同模型（如 PixArt vs SDXL）因 bucket 尺寸差异
        导致缓存冲突：同一张图在不同模型下被 resize 到不同 bucket 分辨率，
        产生不同形状的 latent，混用会导致 DataLoader collate 时 shape mismatch。
        """
        explicit = self.training_cfg.get("latent_cache_dir", None)
        if explicit:
            return explicit
        data_dir = self.config.data.get("train_data_dir", "./data")
        parent = os.path.dirname(os.path.normpath(data_dir))
        model_type = getattr(self, "model_type", "default")
        use_bucketing = self.config.data.get("use_aspect_ratio_bucketing", True)
        suffix = "" if use_bucketing else "_pad"
        return os.path.join(parent, f"latent_cache_{model_type}{suffix}")

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
        from data.transforms import AspectRatioPad, AspectRatioResize

        cache_dir_path = Path(cache_dir)
        cache_dir_path.mkdir(parents=True, exist_ok=True)

        data_cfg = self.config.data
        resolution = data_cfg.get("resolution", 512 if self.model_type == "sd15" else 1024)
        center_crop = data_cfg.get("center_crop", False)
        vae_batch_size = self.training_cfg.get("latent_cache_batch_size", 4)
        use_bucketing = data_cfg.get("use_aspect_ratio_bucketing", True)
        pad_color = tuple(data_cfg.get("pad_color", [0, 0, 0]))

        num_processes = self.accelerator.num_processes
        process_index = self.accelerator.process_index
        device = self.accelerator.device

        temp_dataset = BaseImageDataset(
            data_dir=data_cfg.train_data_dir,
            resolution=resolution,
            center_crop=center_crop,
            random_flip=False,
        )

        if use_bucketing:
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

                        if use_bucketing:
                            resizer = AspectRatioResize(target_size, center_crop=center_crop)
                        else:
                            padder = AspectRatioPad(target_size, pad_color=pad_color)

                        for batch_start in range(0, len(indices), vae_batch_size):
                            batch_indices = indices[batch_start: batch_start + vae_batch_size]

                            imgs_normal, imgs_flip, orig_sizes = [], [], []
                            pad_masks = []
                            for idx in batch_indices:
                                image = PIL_Image.open(temp_dataset.image_paths[idx]).convert("RGB")
                                orig_w, orig_h = image.size
                                orig_sizes.append((orig_h, orig_w))

                                if use_bucketing:
                                    image_resized = resizer(image)
                                else:
                                    image_resized, pad_mask_pil = padder(image)
                                    pad_masks.append(to_tensor(pad_mask_pil))  # (1, H, W), 0~1

                                imgs_normal.append(normalize(to_tensor(image_resized)))
                                imgs_flip.append(normalize(to_tensor(TF.hflip(image_resized))))

                            batch_t = torch.stack(imgs_normal).to(device)
                            batch_t_flip = torch.stack(imgs_flip).to(device)

                            with torch.no_grad():
                                enc_out = self.vae.encode(batch_t)
                                if hasattr(enc_out, "latent_dist"):
                                    latents = enc_out.latent_dist.mode()
                                else:
                                    latents = enc_out.latent
                                latents = (latents * self.vae.config.scaling_factor).to(torch.float16).cpu()

                                enc_out_flip = self.vae.encode(batch_t_flip)
                                if hasattr(enc_out_flip, "latent_dist"):
                                    latents_flip = enc_out_flip.latent_dist.mode()
                                else:
                                    latents_flip = enc_out_flip.latent
                                latents_flip = (latents_flip * self.vae.config.scaling_factor).to(torch.float16).cpu()

                            latent_h, latent_w = latents.shape[-2], latents.shape[-1]
                            weight_masks = []
                            orig_max_channels = []
                            padding_masks = []
                            aux_fg_thresh = float(
                                self.config.get("auxiliary_loss", {}).get("fg_threshold", 0.1)
                            )
                            for j_idx in range(len(batch_indices)):
                                img_t = imgs_normal[j_idx]  # (3, H, W), normalized [-1, 1]
                                img_01 = (img_t + 1.0) / 2.0

                                lum = 0.299 * img_01[0] + 0.587 * img_01[1] + 0.114 * img_01[2]
                                mask = F.interpolate(
                                    lum.unsqueeze(0).unsqueeze(0),
                                    size=(latent_h, latent_w),
                                    mode="bilinear",
                                    align_corners=False,
                                ).squeeze(0).to(torch.float16).cpu()  # (1, H/f, W/f)
                                weight_masks.append(mask)

                                max_ch = img_01.max(dim=0).values  # (H, W)
                                binary_max = (max_ch > aux_fg_thresh).float()
                                binary_max_down = F.interpolate(
                                    binary_max.unsqueeze(0).unsqueeze(0),
                                    size=(latent_h, latent_w),
                                    mode="nearest",
                                ).squeeze(0).to(torch.float16).cpu()  # (1, H/f, W/f)
                                orig_max_channels.append(binary_max_down)

                                if not use_bucketing:
                                    pm = F.interpolate(
                                        pad_masks[j_idx].unsqueeze(0),
                                        size=(latent_h, latent_w),
                                        mode="bilinear",
                                        align_corners=False,
                                    ).squeeze(0).to(torch.float16).cpu()  # (1, H/f, W/f)
                                    padding_masks.append(pm)

                            for j, idx in enumerate(batch_indices):
                                orig_h, orig_w = orig_sizes[j]
                                save_dict = {
                                    "latent": latents[j],
                                    "latent_flip": latents_flip[j],
                                    "weight_mask": weight_masks[j],
                                    "orig_max_channel": orig_max_channels[j],
                                    "original_hw": torch.tensor([orig_h, orig_w], dtype=torch.long),
                                    "target_hw": torch.tensor([target_h, target_w], dtype=torch.long),
                                }
                                if not use_bucketing:
                                    save_dict["padding_mask"] = padding_masks[j]
                                torch.save(
                                    save_dict,
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
        if hasattr(self.vae, "encoder_conv_in"):
            del self.vae.encoder_conv_in
        if hasattr(self.vae, "encoder_stages"):
            del self.vae.encoder_stages
        if hasattr(self.vae, "encoder_conv_out"):
            del self.vae.encoder_conv_out
        torch.cuda.empty_cache()
        logger.info(f"[Rank {process_index}] VAE encoder 已卸载（保留 decoder 供 validation 使用）")

    def _precompute_latents_distributed(self, cache_dir: str, *, delete_encoder: bool = True) -> None:
        """分布式 latent 预计算入口（所有 Rank 并行，内部已包含同步逻辑）。"""
        self._precompute_latents(cache_dir, delete_encoder=delete_encoder)
