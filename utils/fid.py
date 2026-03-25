"""FID（Fréchet Inception Distance）计算模块 —— 使用 DINOv2 特征提取器。

依赖: scipy (pip install scipy)，torch.hub 加载 DINOv2 (需网络或本地缓存)。

工作流程:
  1. 训练开始前, 用 FIDCalculator.update_real() 从训练集抽样加载真实图像特征 (一次性)。
  2. 每次验证时, 用 FIDCalculator.compute(generated_images) 计算 FID 分数。

DINOv2 模型选项 (model_name):
  - "dinov2_vits14"  : ViT-S/14, 特征维度 384
  - "dinov2_vitb14"  : ViT-B/14, 特征维度 768  (默认)
  - "dinov2_vitl14"  : ViT-L/14, 特征维度 1024
  - "dinov2_vitg14"  : ViT-G/14, 特征维度 1536

注意:
  - DINOv2 特征提取在 GPU 上运行 (如可用)，仅主进程执行。
  - 真实图像特征在首次调用 update_real() 时缓存, 后续复用。
  - 输入图像需 resize 到 224x224 并用 ImageNet 均值/方差归一化。
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

# DINOv2 输入分辨率 (14 的倍数)
_DINOV2_INPUT_SIZE = 224

# ImageNet 归一化参数
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _pil_to_float_tensor(images: list[Image.Image], size: int = _DINOV2_INPUT_SIZE) -> torch.Tensor:
    """将 PIL 图像列表转为归一化后的 (N, 3, H, W) float32 Tensor，供 DINOv2 使用。"""
    mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    tensors = []
    for img in images:
        img = img.convert("RGB").resize((size, size), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0          # [0, 1]
        t = torch.from_numpy(arr).permute(2, 0, 1)             # (3, H, W)
        t = (t - mean) / std
        tensors.append(t)
    return torch.stack(tensors)  # (N, 3, H, W)


def _compute_fid(mu1: np.ndarray, sigma1: np.ndarray,
                 mu2: np.ndarray, sigma2: np.ndarray,
                 eps: float = 1e-6) -> float:
    """根据两组高斯分布的均值/协方差计算 FID。

    FID = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2·(Σ₁Σ₂)^(1/2))
    """
    try:
        from scipy import linalg
    except ImportError as e:
        raise ImportError("FID 计算需要 scipy，请运行: pip install scipy") from e

    diff = mu1 - mu2
    # 数值稳定：对协方差矩阵加小量对角扰动
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

    # 消除数值误差产生的虚部
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            logger.warning("矩阵平方根存在较大虚部，FID 结果可能不准确")
        covmean = covmean.real

    fid = float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))
    return fid


class FIDCalculator:
    """使用 DINOv2 CLS token 特征计算 Fréchet Inception Distance。

    Args:
        model_name: DINOv2 模型名称，默认 "dinov2_vitb14"（ViT-B/14，特征维度 768）。
        device: 运行 DINOv2 的设备。
        real_images_cache_path: 若提供，将真实图像统计量缓存到磁盘，下次直接加载。
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        device: Optional[torch.device] = None,
        real_images_cache_path: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.cache_path = Path(real_images_cache_path) if real_images_cache_path else None
        self._model = None
        self._real_mu: Optional[np.ndarray] = None
        self._real_sigma: Optional[np.ndarray] = None

    def _get_model(self) -> torch.nn.Module:
        """惰性加载 DINOv2 模型（避免在不需要 FID 时占用显存）。"""
        if self._model is None:
            logger.info(f"正在加载 DINOv2 模型: {self.model_name} ...")
            try:
                model = torch.hub.load("facebookresearch/dinov2", self.model_name)
            except Exception as e:
                raise RuntimeError(
                    f"无法加载 DINOv2 模型 '{self.model_name}'，"
                    "请确认网络可访问 GitHub/PyTorch Hub，或已设置本地缓存。\n"
                    f"原始错误: {e}"
                ) from e
            model.eval().to(self.device)
            self._model = model
            logger.info(f"DINOv2 ({self.model_name}) 加载完成，运行于 {self.device}")
        return self._model

    @torch.no_grad()
    def _extract_features(self, images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        """提取图像的 DINOv2 CLS token 特征，返回 (N, D) numpy 数组。"""
        model = self._get_model()
        all_features = []
        for start in range(0, len(images), batch_size):
            batch = images[start: start + batch_size]
            tensor = _pil_to_float_tensor(batch).to(self.device)
            features = model(tensor)                  # (B, D)，DINOv2 默认返回 CLS token
            all_features.append(features.cpu().float().numpy())
        return np.concatenate(all_features, axis=0)  # (N, D)

    @staticmethod
    def _calc_stats(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """计算特征的均值向量和协方差矩阵。"""
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    def update_real(self, images: list[Image.Image], batch_size: int = 32) -> None:
        """提取并缓存真实图像的 DINOv2 特征统计量（只需调用一次）。

        Args:
            images: 真实训练图像列表（PIL.Image），建议 256~2048 张。
            batch_size: 送入 DINOv2 的批大小，防止 OOM。
        """
        if self._real_mu is not None:
            logger.debug("真实图像统计量已缓存，跳过 update_real()")
            return

        # 尝试从磁盘加载缓存
        if self.cache_path and self.cache_path.exists():
            logger.info(f"从磁盘加载真实图像统计量缓存: {self.cache_path}")
            state = np.load(self.cache_path, allow_pickle=False)
            self._real_mu = state["mu"]
            self._real_sigma = state["sigma"]
            logger.info(f"缓存加载完成（特征维度: {self._real_mu.shape[0]}）")
            return

        logger.info(f"开始提取 {len(images)} 张真实图像的 DINOv2 特征...")
        features = self._extract_features(images, batch_size)
        self._real_mu, self._real_sigma = self._calc_stats(features)
        logger.info(f"真实图像特征提取完成（shape: {features.shape}）")

        # 写入磁盘缓存
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                self.cache_path,
                mu=self._real_mu,
                sigma=self._real_sigma,
            )
            logger.info(f"真实图像统计量已缓存至: {self.cache_path}")

    def compute(self, generated_images: list[Image.Image], batch_size: int = 32) -> float:
        """计算当前生成图像的 FID 分数。

        Args:
            generated_images: 模型生成的 PIL 图像列表（建议 ≥ 50 张，越多越准确）。
            batch_size: DINOv2 推理批大小。

        Returns:
            FID 标量（越低越好，0 表示完全一致）。
        """
        if self._real_mu is None:
            raise RuntimeError("请先调用 update_real() 提取真实图像特征")

        if len(generated_images) < 2:
            logger.warning(f"生成图像数量过少 ({len(generated_images)})，FID 结果不可靠")

        logger.info(f"开始提取 {len(generated_images)} 张生成图像的 DINOv2 特征...")
        gen_features = self._extract_features(generated_images, batch_size)
        gen_mu, gen_sigma = self._calc_stats(gen_features)

        fid_score = _compute_fid(self._real_mu, self._real_sigma, gen_mu, gen_sigma)
        logger.info(f"FID score (DINOv2/{self.model_name}): {fid_score:.4f} (n_gen={len(generated_images)})")
        return fid_score

    def is_ready(self) -> bool:
        """真实图像统计量是否已准备好。"""
        return self._real_mu is not None
