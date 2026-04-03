"""TensorBoard 日志封装 — 记录标量指标和生成的验证图像。"""

import os
from typing import Optional

import numpy as np
import torch
from PIL import Image


class TensorBoardLogger:
    """TensorBoard 日志记录器，仅在主进程写入。"""

    def __init__(
        self,
        log_dir: str = "./outputs/tensorboard",
        is_main_process: bool = True,
        log_every_n_steps: int = 10,
    ):
        self.log_dir = log_dir
        self.is_main_process = is_main_process
        self.log_every_n_steps = log_every_n_steps
        self._writer = None

        if self.is_main_process:
            os.makedirs(log_dir, exist_ok=True)
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=log_dir)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """记录标量指标（受 log_every_n_steps 控制）。"""
        if not self.is_main_process or self._writer is None:
            return
        if step % self.log_every_n_steps == 0:
            self._writer.add_scalar(tag, value, step)

    def log_scalar_force(self, tag: str, value: float, step: int) -> None:
        """强制记录标量（无间隔限制），用于重要的非频繁指标。"""
        if not self.is_main_process or self._writer is None:
            return
        self._writer.add_scalar(tag, value, step)

    def log_images(
        self,
        tag: str,
        images: list[Image.Image],
        step: int,
        max_images: int = 8,
    ) -> None:
        """将 PIL 图像列表记录到 TensorBoard（网格排列）。"""
        if not self.is_main_process or self._writer is None:
            return

        images = images[:max_images]
        np_images = [np.array(img) for img in images]

        for i, img_np in enumerate(np_images):
            # HWC → CHW
            if img_np.ndim == 3:
                img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
            else:
                img_tensor = torch.from_numpy(img_np).unsqueeze(0).float() / 255.0
            self._writer.add_image(f"{tag}/{i}", img_tensor, step)

    def log_lr(self, lr: float, step: int) -> None:
        self.log_scalar("train/learning_rate", lr, step)

    def log_projector_lr(self, lr: float, step: int) -> None:
        self.log_scalar("train/projector_lr", lr, step)

    def log_loss(self, loss: float, step: int) -> None:
        self.log_scalar("train/loss", loss, step)

    def log_grad_norm(self, grad_norm: float, step: int) -> None:
        self.log_scalar("train/grad_norm", grad_norm, step)

    def log_grad_norm_group(self, tf_norm: float, proj_norm: float, step: int) -> None:
        self.log_scalar("train/grad_norm_transformer", tf_norm, step)
        self.log_scalar("train/grad_norm_projector", proj_norm, step)

    def log_ema_loss(self, ema_loss: float, step: int) -> None:
        self.log_scalar("train/ema_loss", ema_loss, step)

    def log_best_loss(self, best_loss: float, step: int) -> None:
        """记录当前最佳 EMA loss（强制写入）。"""
        self.log_scalar_force("train/best_ema_loss", best_loss, step)

    def log_fid(self, fid_score: float, step: int) -> None:
        """记录 FID 分数（强制写入，不受 log_every_n_steps 约束）。"""
        self.log_scalar_force("validation/fid", fid_score, step)

    def flush(self) -> None:
        if self._writer is not None:
            self._writer.flush()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
