"""VAE 图像重建测试脚本

功能:
  - 仅加载 VAE 组件（不加载 UNet / 文本编码器），节省显存
  - 对数据集中随机采样的图像做 encode → decode 重建
  - 计算 PSNR / SSIM / MSE / MAE 等量化指标
  - 将 原图 | 重建图 | 差异热力图 拼合保存，便于目视检查
  - 支持 SD1.5（512×512）、SDXL（1024×1024）和 PixArt-Sigma（1024×1024）
  - 支持 --image_list 保证两次测试使用完全相同的图片集合

用法示例:
  # ── 基础测试（各自随机抽图）──
  python scripts/test_vae_reconstruction.py --model_type sd15 --num_images 50
  python scripts/test_vae_reconstruction.py --model_type sdxl --num_images 50
  python scripts/test_vae_reconstruction.py --model_type pixart_sigma --num_images 50

  # ── 同时测试 SDXL 与 PixArt-Sigma（结果各存独立子目录）──
  # 第一步：测试 SDXL，保存采样列表
  python scripts/test_vae_reconstruction.py \
      --model_type sdxl --num_images 50 \
      --image_list ./outputs/shared_list_1024.json

  # 第二步：测试 PixArt-Sigma，复用相同图片列表
  python scripts/test_vae_reconstruction.py \
      --model_type pixart_sigma \
      --image_list ./outputs/shared_list_1024.json

  # ── 保证测试集相同（推荐对比两个模型时使用）──
  # 第一步：测试 SD1.5，同时把采样列表保存到 shared_list.json
  python scripts/test_vae_reconstruction.py \
      --model_type sd15 --num_images 50 \
      --image_list ./outputs/shared_list.json \
      --output_dir ./outputs/vae_test_sd15

  # 第二步：测试 SDXL，自动从同一个 json 中读取，找对应 1024 分辨率版本
  python scripts/test_vae_reconstruction.py \
      --model_type sdxl \
      --image_list ./outputs/shared_list.json \
      --output_dir ./outputs/vae_test_sdxl

  # ── 自定义 VAE 路径 ──
  python scripts/test_vae_reconstruction.py \
      --vae_path /path/to/my/vae \
      --data_dir /path/to/images \
      --resolution 512
"""

import argparse
import json
import logging
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# 默认路径常量
# ──────────────────────────────────────────────────────────────
WEIGHTS_ROOT = Path(__file__).resolve().parents[1] / "weights"
DATA_ROOT    = Path(__file__).resolve().parents[1] / "data" / "data"

MODEL_DEFAULTS = {
    "sd15": {
        "slug": "runwayml--stable-diffusion-v1-5",
        "resolution": 512,
        "data_subdir": "size_512",
    },
    "sdxl": {
        "slug": "stabilityai--stable-diffusion-xl-base-1.0",
        "resolution": 1024,
        "data_subdir": "size_1024",
    },
    "pixart_sigma": {
        "slug": "PixArt-alpha--PixArt-Sigma-XL-2-1024-MS",
        "resolution": 1024,
        "data_subdir": "size_1024",
    },
}


# ──────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────

_RES_SUFFIX_RE = re.compile(r"___total__\d+$")

def _stem_key(path: Path) -> str:
    """提取图片 stem 中去掉 '___total__{N}' 后缀的公共 key。
    例: '20250206_xxx_t3_per_floor_per_layer_f1___total__512' → '20250206_xxx_t3_per_floor_per_layer_f1'
    """
    return _RES_SUFFIX_RE.sub("", path.stem)


def _build_stem_index(data_dir: Path) -> dict[str, Path]:
    """递归扫描 data_dir，返回 {stem_key: 文件路径} 映射。"""
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return {
        _stem_key(p): p
        for p in data_dir.rglob("*")
        if p.suffix.lower() in exts
    }


def collect_images(
    data_dir: Path,
    num_images: int,
    seed: int,
    image_list_file: Path | None = None,
    other_data_dirs: list[Path] | None = None,
) -> list[Path]:
    """收集测试图片路径，支持与其他数据集取交集以保证测试集一致。

    逻辑:
      1. 若 image_list_file 存在 → 从 JSON 读取 stem 列表，在 data_dir 中查找对应文件。
      2. 若 image_list_file 不存在（或未指定）→ 在 data_dir（与 other_data_dirs 交集）中
         随机采样 num_images 张，并将 stem 列表写入 image_list_file（如果指定了路径）。
    """
    # ── 情况 1: 已有 image_list 文件，直接加载 ──
    if image_list_file is not None and image_list_file.exists():
        with open(image_list_file) as f:
            stems = json.load(f)
        logger.info(f"从 {image_list_file} 加载图片列表（{len(stems)} 个 stem）")

        index = _build_stem_index(data_dir)
        matched: list[Path] = []
        missing = []
        for stem in stems:
            if stem in index:
                matched.append(index[stem])
            else:
                missing.append(stem)

        if missing:
            logger.warning(f"  {len(missing)} 个 stem 在 {data_dir} 中未找到对应文件，已跳过")
        if not matched:
            logger.error(f"image_list 中的所有 stem 在 {data_dir} 中均无匹配，请检查路径")
            sys.exit(1)

        logger.info(f"成功匹配 {len(matched)} 张图像（来自 {data_dir}）")
        return matched

    # ── 情况 2: 首次运行，采样并（可选）保存列表 ──
    index = _build_stem_index(data_dir)

    if other_data_dirs:
        # 取各目录 stem 的交集，保证跨模型可复用
        common_stems = set(index.keys())
        for other_dir in other_data_dirs:
            if other_dir.exists():
                common_stems &= set(_build_stem_index(other_dir).keys())
        candidate_stems = sorted(common_stems)
        logger.info(
            f"与其他数据集取交集后，公共图片数量: {len(candidate_stems)}，"
            f"本次抽取 {min(num_images, len(candidate_stems))} 张"
        )
    else:
        candidate_stems = sorted(index.keys())
        logger.info(f"找到 {len(candidate_stems)} 张图像，随机抽取 {min(num_images, len(candidate_stems))} 张")

    if not candidate_stems:
        logger.error(f"在 {data_dir} 下未找到任何图像文件")
        sys.exit(1)

    rng = random.Random(seed)
    sampled_stems = rng.sample(candidate_stems, min(num_images, len(candidate_stems)))

    if image_list_file is not None:
        image_list_file.parent.mkdir(parents=True, exist_ok=True)
        with open(image_list_file, "w") as f:
            json.dump(sampled_stems, f, indent=2, ensure_ascii=False)
        logger.info(f"图片列表已保存至 {image_list_file}（可供其他模型复用）")

    return [index[s] for s in sampled_stems]


def load_image_as_tensor(path: Path, resolution: int) -> torch.Tensor:
    """加载并中心裁剪到 resolution×resolution，返回 (1,3,H,W) in [-1,1]。"""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    min_side = min(w, h)
    left = (w - min_side) // 2
    top  = (h - min_side) // 2
    img  = img.crop((left, top, left + min_side, top + min_side))
    img  = img.resize((resolution, resolution), Image.LANCZOS)
    arr  = np.array(img, dtype=np.float32) / 127.5 - 1.0          # [0,255] → [-1,1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    return tensor


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """(1,3,H,W) [-1,1] 张量 → PIL Image。"""
    arr = tensor.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
    arr = np.clip((arr + 1.0) * 127.5, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def diff_heatmap(orig: torch.Tensor, recon: torch.Tensor) -> Image.Image:
    """生成绝对差值热力图（灰度，越亮误差越大）。"""
    diff = (orig - recon).abs().mean(dim=1, keepdim=True)          # (1,1,H,W)
    diff = diff.squeeze().float().cpu().numpy()
    diff = (diff / (diff.max() + 1e-8) * 255).astype(np.uint8)
    return Image.fromarray(diff).convert("RGB")


# ──────────────────────────────────────────────────────────────
# 指标计算
# ──────────────────────────────────────────────────────────────

def compute_mse(orig: torch.Tensor, recon: torch.Tensor) -> float:
    return ((orig - recon) ** 2).mean().item()


def compute_mae(orig: torch.Tensor, recon: torch.Tensor) -> float:
    return (orig - recon).abs().mean().item()


def compute_psnr(mse: float, max_val: float = 2.0) -> float:
    """PSNR（信号范围 [-1,1] → max_val=2.0）。"""
    if mse < 1e-10:
        return float("inf")
    return 10.0 * math.log10(max_val ** 2 / mse)


def compute_ssim(orig: torch.Tensor, recon: torch.Tensor) -> float:
    """简化版 SSIM（单一窗口，通道平均）。在 CPU numpy 上计算。"""
    x = orig.squeeze(0).float().cpu().numpy()   # (3,H,W)
    y = recon.squeeze(0).float().cpu().numpy()

    C1 = (0.01 * 2) ** 2
    C2 = (0.03 * 2) ** 2

    ssim_vals = []
    for c in range(x.shape[0]):
        mu_x = x[c].mean()
        mu_y = y[c].mean()
        sig_x  = x[c].var()
        sig_y  = y[c].var()
        sig_xy = ((x[c] - mu_x) * (y[c] - mu_y)).mean()
        num = (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)
        den = (mu_x ** 2 + mu_y ** 2 + C1) * (sig_x + sig_y + C2)
        ssim_vals.append(num / (den + 1e-8))
    return float(np.mean(ssim_vals))


# ──────────────────────────────────────────────────────────────
# 主测试流程
# ──────────────────────────────────────────────────────────────

def run_test(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    dtype  = torch.float16 if (args.dtype == "fp16") else \
             torch.bfloat16 if (args.dtype == "bf16") else \
             torch.float32

    # ── 解析 VAE 路径 ──
    if args.vae_path:
        vae_path = Path(args.vae_path)
        resolution = args.resolution
        data_dir   = Path(args.data_dir) if args.data_dir else None
        other_data_dirs: list[Path] = []
    else:
        cfg       = MODEL_DEFAULTS[args.model_type]
        vae_path  = WEIGHTS_ROOT / cfg["slug"] / "vae"
        resolution = cfg["resolution"]
        data_dir   = Path(args.data_dir) if args.data_dir else DATA_ROOT / cfg["data_subdir"]
        # 预构建对侧数据目录列表，用于取 stem 交集
        other_data_dirs = [
            DATA_ROOT / v["data_subdir"]
            for k, v in MODEL_DEFAULTS.items()
            if k != args.model_type
        ]

    image_list_file = Path(args.image_list) if args.image_list else None

    if not vae_path.exists():
        logger.error(
            f"VAE 路径不存在: {vae_path}\n"
            "请先运行: python scripts/download_weights.py"
        )
        sys.exit(1)

    if data_dir is None or not data_dir.exists():
        logger.error(f"数据目录不存在: {data_dir}")
        sys.exit(1)

    # ── 准备输出目录 ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"结果将保存到: {output_dir}")

    # ── 显存工具函数（仅 CUDA 有效）──
    def vram_mb() -> str:
        if device.type != "cuda":
            return "N/A (CPU)"
        alloc   = torch.cuda.memory_allocated(device) / 1024 ** 2
        reserved = torch.cuda.memory_reserved(device) / 1024 ** 2
        return f"alloc={alloc:.0f}MB  reserved={reserved:.0f}MB"

    def vram_peak_mb() -> float:
        if device.type != "cuda":
            return 0.0
        return torch.cuda.max_memory_allocated(device) / 1024 ** 2

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # ── 加载 VAE ──
    logger.info(f"加载 VAE: {vae_path}  dtype={dtype}")
    t0  = time.time()
    vae = AutoencoderKL.from_pretrained(str(vae_path), torch_dtype=dtype)
    vae = vae.to(device).eval()
    load_sec = time.time() - t0
    param_mb = sum(p.numel() * p.element_size() for p in vae.parameters()) / 1024 ** 2
    logger.info(
        f"VAE 加载完毕（{load_sec:.1f}s）  "
        f"参数量: {sum(p.numel() for p in vae.parameters()) / 1e6:.1f}M  "
        f"权重显存: {param_mb:.0f}MB  "
        f"当前显存: [{vram_mb()}]"
    )

    # ── 收集测试图像 ──
    image_paths = collect_images(
        data_dir=data_dir,
        num_images=args.num_images,
        seed=args.seed,
        image_list_file=image_list_file,
        other_data_dirs=other_data_dirs if not args.vae_path else [],
    )

    # ── 逐张推理 ──
    all_mse, all_mae, all_psnr, all_ssim = [], [], [], []
    latent_stds = []

    for idx, img_path in enumerate(image_paths, 1):
        logger.info(f"[{idx}/{len(image_paths)}] {img_path.name[:60]}…")

        # 加载并预处理
        orig_tensor = load_image_as_tensor(img_path, resolution).to(device, dtype)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        with torch.no_grad():
            # Encode
            posterior = vae.encode(orig_tensor).latent_dist
            latents   = posterior.sample()        # (1, C_lat, H/8, W/8)
            latents_scaled = latents * vae.config.scaling_factor

            latent_stds.append(latents_scaled.float().std().item())

            # Decode
            recon_tensor = vae.decode(latents_scaled / vae.config.scaling_factor).sample

        peak_mb = vram_peak_mb()

        # 限幅（编码 → 解码后可能略超 [-1,1]）
        recon_tensor = recon_tensor.clamp(-1.0, 1.0)
        orig_f       = orig_tensor.float()
        recon_f      = recon_tensor.float()

        # 指标
        mse  = compute_mse(orig_f, recon_f)
        mae  = compute_mae(orig_f, recon_f)
        psnr = compute_psnr(mse)
        ssim = compute_ssim(orig_f, recon_f)
        all_mse.append(mse);  all_mae.append(mae)
        all_psnr.append(psnr); all_ssim.append(ssim)
        vram_info = f"  峰值显存={peak_mb:.0f}MB" if device.type == "cuda" else ""
        logger.info(f"  MSE={mse:.5f}  MAE={mae:.4f}  PSNR={psnr:.2f}dB  SSIM={ssim:.4f}{vram_info}")

        # 保存对比图
        pil_orig  = tensor_to_pil(orig_f)
        pil_recon = tensor_to_pil(recon_f)
        pil_diff  = diff_heatmap(orig_f, recon_f)

        # 拼合: 原图 | 重建图 | 差异图
        W, H = pil_orig.size
        canvas = Image.new("RGB", (W * 3, H))
        canvas.paste(pil_orig,  (0,     0))
        canvas.paste(pil_recon, (W,     0))
        canvas.paste(pil_diff,  (W * 2, 0))

        # 添加标注文字（若 Pillow 版本支持）
        try:
            from PIL import ImageDraw
            draw = ImageDraw.Draw(canvas)
            labels = ["Original", "Reconstructed", "Diff (abs)"]
            for i, label in enumerate(labels):
                draw.text((i * W + 4, 4), label, fill=(255, 255, 0))
        except Exception:
            pass

        save_name = f"vae_recon_{idx:03d}_psnr{psnr:.1f}dB.png"
        canvas.save(output_dir / save_name)

    # ── 汇总报告 ──
    n = len(all_mse)
    print("\n" + "=" * 60)
    print(f"VAE 重建指标汇总  ({n} 张图像, resolution={resolution})")
    print("=" * 60)
    print(f"  MSE   : avg={np.mean(all_mse):.5f}  std={np.std(all_mse):.5f}")
    print(f"  MAE   : avg={np.mean(all_mae):.4f}  std={np.std(all_mae):.4f}")
    print(f"  PSNR  : avg={np.mean(all_psnr):.2f}dB  min={np.min(all_psnr):.2f}  max={np.max(all_psnr):.2f}")
    print(f"  SSIM  : avg={np.mean(all_ssim):.4f}  min={np.min(all_ssim):.4f}  max={np.max(all_ssim):.4f}")
    print(f"  Latent std (scaled): avg={np.mean(latent_stds):.3f}")
    print(f"  Latent 空间维度    : {vae.config.latent_channels} ch @ {resolution//8}×{resolution//8}")
    if device.type == "cuda":
        total_peak = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        param_mb_val = sum(p.numel() * p.element_size() for p in vae.parameters()) / 1024 ** 2
        print(f"  VAE 权重显存      : {param_mb_val:.0f} MB")
        print(f"  推理峰值显存(累计): {total_peak:.0f} MB  （含模型权重 + 激活）")
    else:
        param_mb_val = sum(p.numel() * p.element_size() for p in vae.parameters()) / 1024 ** 2
        print(f"  VAE 权重内存(CPU) : {param_mb_val:.0f} MB  （CPU 模式，无 GPU 显存统计）")
    print("=" * 60)
    print(f"对比图已保存至: {output_dir}")

    # 保存指标到 txt
    report_path = output_dir / "metrics_report.txt"
    with open(report_path, "w") as f:
        f.write(f"VAE 重建测试报告\n")
        f.write(f"VAE 路径: {vae_path}\n")
        f.write(f"图像数量: {n}\n")
        f.write(f"分辨率: {resolution}×{resolution}\n")
        f.write(f"设备: {device}  dtype: {dtype}\n\n")
        f.write(f"MSE   avg={np.mean(all_mse):.6f}  std={np.std(all_mse):.6f}\n")
        f.write(f"MAE   avg={np.mean(all_mae):.5f}  std={np.std(all_mae):.5f}\n")
        f.write(f"PSNR  avg={np.mean(all_psnr):.3f}dB  min={np.min(all_psnr):.3f}  max={np.max(all_psnr):.3f}\n")
        f.write(f"SSIM  avg={np.mean(all_ssim):.5f}  min={np.min(all_ssim):.5f}  max={np.max(all_ssim):.5f}\n")
        f.write(f"Latent std (scaled) avg={np.mean(latent_stds):.4f}\n")
        f.write("\n逐图明细:\n")
        for i, (path, mse, mae, psnr, ssim) in enumerate(
            zip(image_paths, all_mse, all_mae, all_psnr, all_ssim), 1
        ):
            f.write(f"  [{i:03d}] mse={mse:.5f} mae={mae:.4f} psnr={psnr:.2f}dB ssim={ssim:.4f}  {path.name[:80]}\n")
    logger.info(f"指标报告已写入: {report_path}")


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="仅加载 VAE 测试图像重建能力",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # 模型选择
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--model_type", choices=["sd15", "sdxl", "pixart_sigma"], default="sdxl",
        help="使用预设的 SD1.5 / SDXL / PixArt-Sigma VAE（默认: sdxl）",
    )
    group.add_argument(
        "--vae_path", default=None,
        help="自定义 VAE 目录路径（内含 config.json + *.safetensors），与 --model_type 互斥",
    )
    # 数据
    parser.add_argument(
        "--data_dir", default=None,
        help="测试图像目录（递归搜索 PNG/JPG）；不填则使用项目默认数据集",
    )
    parser.add_argument(
        "--resolution", type=int, default=None,
        help="目标分辨率，使用 --vae_path 时必须指定；使用 --model_type 时自动推断",
    )
    parser.add_argument(
        "--num_images", type=int, default=50,
        help="从数据集随机抽取多少张图进行测试（默认: 50）",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机采样种子（默认: 42）",
    )
    parser.add_argument(
        "--image_list", default=None,
        metavar="PATH",
        help=(
            "图片列表 JSON 文件路径，用于保证两次测试使用相同图片集合。\n"
            "  - 文件不存在：自动采样后保存到该路径（首次运行 SD1.5 时使用）\n"
            "  - 文件已存在：直接加载列表，在当前数据目录中查找对应文件（运行 SDXL 时使用）\n"
            "示例: --image_list ./outputs/shared_list.json"
        ),
    )
    # 硬件
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="推理设备，例如 cuda / cpu / cuda:1（默认: 自动检测）",
    )
    parser.add_argument(
        "--dtype", choices=["fp32", "fp16", "bf16"], default="fp32",
        help="VAE 权重精度（默认: fp32；GPU 推荐 bf16/fp16）",
    )
    # 输出
    parser.add_argument(
        "--output_dir", default=None,
        help=(
            "对比图和报告的保存目录。\n"
            "默认按模型类型自动命名: ./outputs/vae_test_{model_type}\n"
            "例如 sdxl → ./outputs/vae_test_sdxl，pixart_sigma → ./outputs/vae_test_pixart_sigma"
        ),
    )

    args = parser.parse_args()

    # 当使用 --vae_path 时，resolution 必须显式指定
    if args.vae_path and args.resolution is None:
        parser.error("使用 --vae_path 时必须同时指定 --resolution")

    # 自动推断默认输出目录
    if args.output_dir is None:
        if args.vae_path:
            args.output_dir = "./outputs/vae_test_custom"
        else:
            args.output_dir = f"./outputs/vae_test_{args.model_type}"

    return args


if __name__ == "__main__":
    args = parse_args()

    logger.info("=" * 60)
    logger.info("VAE 重建测试")
    if args.vae_path:
        logger.info(f"  VAE 路径 : {args.vae_path}")
        logger.info(f"  分辨率   : {args.resolution}")
    else:
        display_names = {"sd15": "SD1.5", "sdxl": "SDXL", "pixart_sigma": "PixArt-Sigma"}
        logger.info(f"  模型类型 : {display_names.get(args.model_type, args.model_type.upper())}")
        cfg = MODEL_DEFAULTS[args.model_type]
        logger.info(f"  权重路径 : weights/{cfg['slug']}/vae")
    logger.info(f"  测试张数 : {args.num_images}")
    logger.info(f"  输出目录 : {args.output_dir}")
    logger.info(f"  设备     : {args.device}  dtype={args.dtype}")
    if args.image_list:
        list_path = Path(args.image_list)
        status = "加载已有列表" if list_path.exists() else "首次采样并保存"
        logger.info(f"  图片列表 : {args.image_list}  [{status}]")
    logger.info("=" * 60)

    run_test(args)
