#!/usr/bin/env python3
"""准备过拟合测试数据 — 从完整训练集随机采样 N 张图片对（原图+条件图）到独立目录。

用法:
  python scripts/prepare_overfit_data.py                      # 默认 4 张
  python scripts/prepare_overfit_data.py --num_samples 8      # 8 张
  python scripts/prepare_overfit_data.py --seed 123           # 指定随机种子

输出目录结构:
  data/overfit_test/
  ├── train/                      # 原始训练图
  │   ├── 000_<原始文件名>.png
  │   ├── 001_<原始文件名>.png
  │   └── ...
  └── conditioning/               # 对应条件图
      ├── 000_<原始文件名>.png
      ├── 001_<原始文件名>.png
      └── ...

文件名前缀编号确保排序稳定，后跟原始文件名方便追溯。
"""

import argparse
import logging
import random
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

_KNOWN_SUFFIX_PAIRS = [
    ("___total__1024.png", "_controlnet_color_1024.png"),
]


def _strip_known_suffix(name: str, suffix: str) -> str | None:
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return None


def _get_base_key(fname: str, suffixes: list[str]) -> str:
    for suffix in suffixes:
        bk = _strip_known_suffix(fname, suffix)
        if bk is not None:
            return bk
    return Path(fname).stem


def main():
    parser = argparse.ArgumentParser(description="准备过拟合测试数据")
    parser.add_argument(
        "--train_dir",
        type=str,
        default="/home/daiqing_tan/stable_diffusion_lora/data/data/size_1024/floor",
    )
    parser.add_argument(
        "--cond_dir",
        type=str,
        default="/home/daiqing_tan/stable_diffusion_lora/data/data/size_1024_controlnet/floor",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/daiqing_tan/stable_diffusion_lora/data/overfit_test",
    )
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    cond_dir = Path(args.cond_dir)
    output_dir = Path(args.output_dir)

    if not train_dir.exists():
        logger.error(f"训练数据目录不存在: {train_dir}")
        sys.exit(1)
    if not cond_dir.exists():
        logger.error(f"条件图目录不存在: {cond_dir}")
        sys.exit(1)

    orig_suffixes = [pair[0] for pair in _KNOWN_SUFFIX_PAIRS]
    cond_suffixes = [pair[1] for pair in _KNOWN_SUFFIX_PAIRS]

    # 构建条件图索引: {base_key: Path}
    cond_index: dict[str, Path] = {}
    for f in cond_dir.iterdir():
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            bk = _get_base_key(f.name, cond_suffixes)
            cond_index[bk] = f

    # 筛选有配对的训练图
    paired: list[tuple[Path, Path]] = []
    for f in sorted(train_dir.iterdir()):
        if not f.is_file() or f.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        bk = _get_base_key(f.name, orig_suffixes)
        if bk in cond_index:
            paired.append((f, cond_index[bk]))

    if len(paired) < args.num_samples:
        logger.error(
            f"配对图片只有 {len(paired)} 对，不足 {args.num_samples} 张"
        )
        sys.exit(1)

    logger.info(f"找到 {len(paired)} 对配对图片，随机采样 {args.num_samples} 张")

    rng = random.Random(args.seed)
    selected = rng.sample(paired, args.num_samples)

    out_train = output_dir / "train"
    out_cond = output_dir / "conditioning"

    if output_dir.exists():
        shutil.rmtree(output_dir)
        logger.info(f"已清理旧目录: {output_dir}")

    out_train.mkdir(parents=True)
    out_cond.mkdir(parents=True)

    logger.info(f"输出目录: {output_dir}")
    logger.info("-" * 60)

    for i, (train_path, cond_path) in enumerate(selected):
        dst_train = out_train / f"{i:03d}_{train_path.name}"
        dst_cond = out_cond / f"{i:03d}_{cond_path.name}"

        shutil.copy2(train_path, dst_train)
        shutil.copy2(cond_path, dst_cond)

        logger.info(f"  [{i}] 训练图: {train_path.name}")
        logger.info(f"      条件图: {cond_path.name}")

    logger.info("-" * 60)
    logger.info(f"完成！{args.num_samples} 对图片已复制到 {output_dir}")
    logger.info(f"  训练图: {out_train}")
    logger.info(f"  条件图: {out_cond}")


if __name__ == "__main__":
    main()
