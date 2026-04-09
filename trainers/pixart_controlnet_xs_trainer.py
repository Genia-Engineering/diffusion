"""PixArt-Sigma ControlNet-XS 训练器 — 继承 PixArtControlNetTrainer。

与标准 PixArt ControlNet 的关键差异:
  1. 模型: PixArtControlNetXSAdapter (~4.7% base params) 替换全尺寸 adapter (~46%)
  2. 架构: 薄型 XS blocks (无 cross-attention) + 双向 base↔control 投射
  3. 其余复用: 训练循环、预缓存、数据集、验证 pipeline 均继承父类

辅助结构 loss (auxiliary_loss):
  从 SDXL ControlNet-XS 迁移的辅助 loss，让模型更好地学习控制图中各颜色块
  对应区域的结构信息。通过 VAE decode 预测结果到像素空间，与原图做结构对比。
  详见 configs/controlnet_xs_pixart_sigma.yaml 中 auxiliary_loss 注释。
"""

import logging

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from omegaconf import DictConfig

from models.controlnet_xs_pixart import (
    PixArtControlNetXSAdapter,
    PixArtControlNetXSTransformerModel,
)
from .pixart_controlnet_trainer import PixArtControlNetTrainer

logger = logging.getLogger(__name__)


class PixArtControlNetXSTrainer(PixArtControlNetTrainer):
    """PixArt-Sigma ControlNet-XS 训练器。"""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self._init_auxiliary_loss(config)

    def _create_controlnet(self):
        """创建 ControlNet-XS: 薄型 adapter + 联合前向模型。"""
        xs_cfg = self.config.get("controlnet_xs", {})
        num_layers = int(xs_cfg.get("num_layers", 14))
        size_ratio = float(xs_cfg.get("size_ratio", 0.25))
        connection_interval = int(xs_cfg.get("connection_interval", 2))

        self.controlnet = PixArtControlNetXSAdapter.from_transformer(
            self.transformer,
            num_layers=num_layers,
            size_ratio=size_ratio,
            conditioning_mode=self.conditioning_mode,
            connection_interval=connection_interval,
        )

        self.joint_model = PixArtControlNetXSTransformerModel(
            transformer=self.transformer,
            controlnet=self.controlnet,
        )

    def _freeze_parameters(self):
        """根据 train_transformer 配置决定冻结策略。

        train_transformer=False: 冻结 VAE + T5 + Transformer, 仅 XS adapter 可训练
        train_transformer=True:  冻结 VAE + T5, Transformer + XS adapter 全部可训练
        """
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        if self.train_transformer:
            self.transformer.requires_grad_(True)
            self.controlnet.requires_grad_(True)
            n_tf = sum(p.numel() for p in self.transformer.parameters())
            n_xs = sum(p.numel() for p in self.controlnet.parameters())
            logger.info(
                f"全参微调模式: Transformer({n_tf/1e6:.1f}M) + XS Adapter({n_xs/1e6:.1f}M) "
                f"= {(n_tf+n_xs)/1e6:.1f}M trainable"
            )
        else:
            self.transformer.requires_grad_(False)
            self.controlnet.requires_grad_(True)
            n_trainable = sum(p.numel() for p in self.controlnet.parameters() if p.requires_grad)
            n_base = sum(p.numel() for p in self.transformer.parameters())
            logger.info(
                f"ControlNet-XS freeze: {n_trainable/1e6:.1f}M trainable "
                f"({n_trainable/n_base*100:.1f}% of base {n_base/1e6:.0f}M)"
            )

    # ── 辅助结构 loss ─────────────────────────────────────────────────────

    def _init_auxiliary_loss(self, config: DictConfig):
        """解析 auxiliary_loss 配置，预构建颜色查找表和权重张量。"""
        aux_cfg = config.get("auxiliary_loss", {})
        self._aux_enabled = bool(aux_cfg.get("enabled", False))
        if not self._aux_enabled:
            return

        self._aux_weight = float(aux_cfg.get("weight", 0.1))
        self._aux_t_threshold = int(aux_cfg.get("timestep_threshold", 250))
        self._aux_fg_threshold = float(aux_cfg.get("fg_threshold", 0.1))
        self._aux_temperature = float(aux_cfg.get("temperature", 20.0))
        self._aux_bg_weight = float(aux_cfg.get("bg_weight", 0.0))
        self._aux_max_decode = int(aux_cfg.get("max_decode_samples", 2))
        self._aux_sequential_decode = bool(aux_cfg.get("sequential_decode", False))
        self._last_aux_loss: float | None = None
        self._last_diffusion_loss: float | None = None

        if hasattr(self.vae, "enable_gradient_checkpointing"):
            self.vae.enable_gradient_checkpointing()
            logger.info("VAE gradient checkpointing enabled for auxiliary loss")

        labels = list(aux_cfg.get("labels", []))
        if not labels:
            logger.warning("auxiliary_loss.labels 为空，辅助 loss 将被禁用")
            self._aux_enabled = False
            return

        colors = []
        weights = []
        for lab in labels:
            colors.append(lab["rgb"])
            weights.append(float(lab["weight"]))

        self._aux_label_colors = torch.tensor(colors, dtype=torch.float32) / 255.0  # (N, 3)
        self._aux_label_weights = torch.tensor(weights, dtype=torch.float32)          # (N,)

        logger.info(
            f"Auxiliary structure loss: weight={self._aux_weight}, "
            f"t_threshold={self._aux_t_threshold}, fg_threshold={self._aux_fg_threshold}, "
            f"temperature={self._aux_temperature}, max_decode={self._aux_max_decode}, "
            f"sequential_decode={self._aux_sequential_decode}, {len(labels)} labels"
        )

    @torch.no_grad()
    def _build_label_weight_mask(
        self,
        conditioning_pixel_values: torch.Tensor,
        target_hw: tuple[int, int],
    ) -> torch.Tensor:
        """从控制图构建 per-label 权重 mask。

        Args:
            conditioning_pixel_values: (B, 3, H, W) [0, 1] 控制图
            target_hw: 目标 (H, W)，与 VAE 解码输出对齐

        Returns:
            (B, H, W) 权重 mask，bg=bg_weight，每个 label 区域=对应 weight
        """
        B, _, H, W = conditioning_pixel_values.shape
        device = conditioning_pixel_values.device

        label_colors = self._aux_label_colors.to(device)  # (N, 3)
        label_weights = self._aux_label_weights.to(device)  # (N,)

        pixels = conditioning_pixel_values.permute(0, 2, 3, 1)  # (B, H, W, 3)

        dists = (pixels.unsqueeze(3) - label_colors.view(1, 1, 1, -1, 3)).pow(2).sum(dim=-1)  # (B, H, W, N)
        min_dist, min_idx = dists.min(dim=-1)  # (B, H, W)

        color_tol = (10.0 / 255.0) ** 2 * 3
        is_matched = min_dist < color_tol

        weight_map = torch.full((B, H, W), self._aux_bg_weight, device=device)
        weight_map[is_matched] = label_weights[min_idx[is_matched]]

        if (H, W) != target_hw:
            weight_map = F.interpolate(
                weight_map.unsqueeze(1), size=target_hw, mode="nearest",
            ).squeeze(1)

        return weight_map

    def _vae_decode_fn(self, latents: torch.Tensor) -> torch.Tensor:
        """VAE decode 的薄包装，供 torch_checkpoint 调用。"""
        return self.vae.decode(latents, return_dict=True).sample

    def _predict_x0(
        self,
        noisy_latents: torch.Tensor,
        noise_pred: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """从噪声预测计算 predicted x0（可微分）。"""
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(
            device=timesteps.device, dtype=noise_pred.dtype,
        )
        alpha_bar = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_alpha_bar = alpha_bar.sqrt()
        sqrt_one_minus_alpha_bar = (1.0 - alpha_bar).sqrt()

        if self.noise_scheduler.config.prediction_type == "v_prediction":
            x0 = sqrt_alpha_bar * noisy_latents - sqrt_one_minus_alpha_bar * noise_pred
        else:
            x0 = (noisy_latents - sqrt_one_minus_alpha_bar * noise_pred) / sqrt_alpha_bar.clamp(min=1e-8)
        return x0

    def _decode_single_sample(self, x0_single: torch.Tensor) -> torch.Tensor:
        """对单个样本做 VAE decode → 软阈值结构图，逐样本处理控制显存峰值。"""
        scaling_factor = self.vae.config.scaling_factor
        latent = x0_single.unsqueeze(0) / scaling_factor
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            decoded = torch_checkpoint(
                self._vae_decode_fn, latent, use_reentrant=False,
            )
        rgb = (decoded.squeeze(0) + 1.0) / 2.0
        pred_max = rgb.max(dim=0).values
        return torch.sigmoid(
            (pred_max - self._aux_fg_threshold) * self._aux_temperature
        )

    def _compute_auxiliary_loss(
        self,
        noise_pred: torch.Tensor,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        conditioning_pixel_values: torch.Tensor,
        batch: dict,
    ) -> torch.Tensor | None:
        """计算辅助结构 loss。

        仅对 t < threshold 的样本计算；若 batch 中无符合条件的样本则返回 None。
        sequential_decode=True 时逐样本 VAE decode 以降低显存峰值。
        """
        low_t_mask = timesteps < self._aux_t_threshold
        if not low_t_mask.any():
            return None

        idx = low_t_mask.nonzero(as_tuple=True)[0]
        if len(idx) > self._aux_max_decode:
            perm = torch.randperm(len(idx), device=idx.device)[: self._aux_max_decode]
            idx = idx[perm]

        if "orig_max_channel" not in batch:
            logger.warning(
                "batch 中缺少 orig_max_channel，请重新生成 latent 缓存。跳过本步辅助 loss。"
            )
            return None

        sel_noise_pred = noise_pred[idx]
        sel_noisy_latents = noisy_latents[idx]
        sel_timesteps = timesteps[idx]
        sel_cond = conditioning_pixel_values[idx]

        x0_pred = self._predict_x0(sel_noisy_latents, sel_noise_pred, sel_timesteps)

        if self._aux_sequential_decode:
            return self._aux_loss_sequential(x0_pred, sel_timesteps, sel_cond, idx, batch)
        return self._aux_loss_batched(x0_pred, sel_timesteps, sel_cond, idx, batch)

    def _aux_loss_batched(self, x0_pred, sel_timesteps, sel_cond, idx, batch):
        """批量 VAE decode（默认模式，速度快但显存占用高）。"""
        scaling_factor = self.vae.config.scaling_factor
        x0_for_decode = x0_pred / scaling_factor
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            decoded = torch_checkpoint(
                self._vae_decode_fn, x0_for_decode, use_reentrant=False,
            )

        normalized = (decoded + 1.0) / 2.0
        pred_max = normalized.max(dim=1).values
        pred_soft = torch.sigmoid(
            (pred_max - self._aux_fg_threshold) * self._aux_temperature
        )

        orig_binary = batch["orig_max_channel"][idx].to(pred_soft.device).squeeze(1)
        if orig_binary.shape[-2:] != pred_soft.shape[-2:]:
            orig_binary = F.interpolate(
                orig_binary.unsqueeze(1), size=pred_soft.shape[-2:], mode="nearest",
            ).squeeze(1)
        orig_binary = orig_binary.detach()

        weight_mask = self._build_label_weight_mask(sel_cond, pred_soft.shape[-2:])
        t_quality = 1.0 - sel_timesteps.float() / self._aux_t_threshold

        diff_sq = (pred_soft - orig_binary).pow(2) * weight_mask * t_quality[:, None, None]
        fg_count = (weight_mask > 0).sum().clamp(min=1)
        aux_loss = diff_sq.sum() / fg_count

        self._last_aux_loss = aux_loss.item()
        return aux_loss

    def _aux_loss_sequential(self, x0_pred, sel_timesteps, sel_cond, idx, batch):
        """逐样本 VAE decode（sequential_decode=True 时使用，显存友好）。"""
        total_weighted_sq = torch.tensor(0.0, device=x0_pred.device)
        total_fg_count = 0

        for i in range(len(idx)):
            pred_soft_i = self._decode_single_sample(x0_pred[i])
            hw = pred_soft_i.shape[-2:]

            orig_i = batch["orig_max_channel"][idx[i]].to(pred_soft_i.device).squeeze(0)
            if orig_i.shape[-2:] != hw:
                orig_i = F.interpolate(
                    orig_i.unsqueeze(0).unsqueeze(0), size=hw, mode="nearest",
                ).squeeze(0).squeeze(0)
            orig_i = orig_i.detach()

            wm_i = self._build_label_weight_mask(sel_cond[i:i + 1], hw).squeeze(0)
            tq_i = 1.0 - sel_timesteps[i].float() / self._aux_t_threshold

            diff_sq_i = (pred_soft_i - orig_i).pow(2) * wm_i * tq_i
            total_weighted_sq = total_weighted_sq + diff_sq_i.sum()
            total_fg_count += (wm_i > 0).sum().item()

        fg_count = max(total_fg_count, 1)
        aux_loss = total_weighted_sq / fg_count

        self._last_aux_loss = aux_loss.item()
        return aux_loss

    # ── 训练步骤（覆盖父类以集成辅助 loss）───────────────────────────────

    def _training_step(self, batch) -> torch.Tensor:
        """单步训练: 加噪 → ControlNet-XS + Transformer → loss + 可选辅助 loss。"""
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

        # 文本嵌入
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

        # caption dropout
        if self.caption_dropout_rate > 0 and hasattr(self, "_cached_negative_prompt_embeds"):
            drop_mask = torch.rand(bsz, device=latents.device) < self.caption_dropout_rate
            if drop_mask.any():
                null_embeds = self._cached_negative_prompt_embeds.to(latents.device)
                null_mask = self._cached_negative_prompt_attention_mask.to(latents.device)
                prompt_embeds = prompt_embeds.clone()
                attention_mask = attention_mask.clone()
                prompt_embeds[drop_mask] = null_embeds.expand(drop_mask.sum(), -1, -1)
                attention_mask[drop_mask] = null_mask.expand(drop_mask.sum(), -1)

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

        if noise_pred.shape[1] != latents.shape[1]:
            noise_pred, _ = noise_pred.chunk(2, dim=1)

        if self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            target = noise

        # padding_mask: 非分桶模式下，padding 区域不计算 loss
        padding_mask = batch.get("padding_mask", None)
        if padding_mask is not None:
            padding_mask = padding_mask.to(latents.device)
            per_pixel = F.mse_loss(noise_pred.float(), target.float(), reduction="none")

            if self.min_snr_gamma > 0 and self._snr_cache is not None:
                snr = self._snr_cache.to(timesteps.device)[timesteps]
                if self.noise_scheduler.config.prediction_type == "v_prediction":
                    weights = torch.clamp(snr, max=self.min_snr_gamma) / (snr + 1.0)
                else:
                    weights = torch.clamp(snr, max=self.min_snr_gamma) / snr
                per_sample = (per_pixel * padding_mask).sum(dim=[1, 2, 3]) / padding_mask.sum(dim=[1, 2, 3]).clamp(min=1) / per_pixel.shape[1]
                diffusion_loss = (per_sample * weights).mean()
            else:
                n_content = padding_mask.sum() * per_pixel.shape[1]
                diffusion_loss = (per_pixel * padding_mask).sum() / n_content.clamp(min=1)
        else:
            diffusion_loss = self.compute_loss(
                noise_pred, target, timesteps,
                snr_cache=self._snr_cache,
                min_snr_gamma=self.min_snr_gamma,
                prediction_type=self.noise_scheduler.config.prediction_type,
            )

        if not self._aux_enabled:
            return diffusion_loss

        self._last_diffusion_loss = diffusion_loss.item()

        conditioning_pixel_values = batch.get("conditioning_pixel_values", None)
        if conditioning_pixel_values is None:
            return diffusion_loss

        conditioning_pixel_values = conditioning_pixel_values.to(self.accelerator.device)
        aux_loss = self._compute_auxiliary_loss(
            noise_pred, noisy_latents, timesteps,
            conditioning_pixel_values, batch,
        )
        if aux_loss is not None:
            aux_weighted = self._aux_weight * aux_loss
            max_aux = 0.5 * diffusion_loss.detach()
            aux_weighted = torch.clamp(aux_weighted, max=max_aux)
            return diffusion_loss + aux_weighted
        return diffusion_loss
