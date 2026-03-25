"""
检测 size_1024_controlnet 与 size_1024 目录下的文件是否一一对应，
符合 ControlNet 训练数据集要求。

文件命名规律：
  size_1024          : ..._<key>___total__1024.png
  size_1024_controlnet: ..._<key>_controlnet_color_1024.png

两者去掉各自后缀后，剩余的 base_key 应完全一致。
"""

import os
import sys
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path("/home/daiqing_tan/stable_diffusion_lora/data/data")
DIR_ORIGINAL = BASE_DIR / "size_1024"
DIR_CONTROLNET = BASE_DIR / "size_1024_controlnet"

SUFFIX_ORIGINAL = "___total__1024.png"
SUFFIX_CONTROLNET = "_controlnet_color_1024.png"


def extract_base_key(filename: str, suffix: str) -> str | None:
    """去掉指定后缀，返回 base_key；若不匹配则返回 None。"""
    if filename.endswith(suffix):
        return filename[: -len(suffix)]
    return None


def collect_keys(directory: Path, suffix: str) -> dict[str, list[str]]:
    """
    遍历目录下所有子目录，收集 {subdir -> [base_key, ...]}。
    跳过无法识别后缀的文件并给出警告。
    """
    result: dict[str, list[str]] = defaultdict(list)
    if not directory.exists():
        print(f"[错误] 目录不存在: {directory}")
        sys.exit(1)

    for subdir in sorted(directory.iterdir()):
        if not subdir.is_dir():
            continue
        for f in sorted(subdir.iterdir()):
            if not f.is_file():
                continue
            key = extract_base_key(f.name, suffix)
            if key is None:
                print(f"  [警告] 无法识别的文件（后缀不匹配）: {f.relative_to(BASE_DIR)}")
            else:
                result[subdir.name].append(key)
    return result


def check_correspondence() -> bool:
    print("=" * 70)
    print("ControlNet 训练数据集文件对应性检测")
    print(f"  原始图像目录  : {DIR_ORIGINAL}")
    print(f"  ControlNet 目录: {DIR_CONTROLNET}")
    print("=" * 70)

    keys_original = collect_keys(DIR_ORIGINAL, SUFFIX_ORIGINAL)
    keys_controlnet = collect_keys(DIR_CONTROLNET, SUFFIX_CONTROLNET)

    all_subdirs = sorted(set(keys_original) | set(keys_controlnet))
    overall_ok = True

    for subdir in all_subdirs:
        print(f"\n[子目录: {subdir}]")
        orig_set = set(keys_original.get(subdir, []))
        ctrl_set = set(keys_controlnet.get(subdir, []))

        print(f"  size_1024           文件数: {len(orig_set)}")
        print(f"  size_1024_controlnet 文件数: {len(ctrl_set)}")

        missing_in_controlnet = orig_set - ctrl_set
        missing_in_original = ctrl_set - orig_set

        if missing_in_controlnet:
            overall_ok = False
            print(f"  [缺失] 以下文件在 size_1024 中存在，但 controlnet 目录中缺少对应文件（共 {len(missing_in_controlnet)} 个）：")
            for k in sorted(missing_in_controlnet):
                print(f"    {k}{SUFFIX_ORIGINAL}")

        if missing_in_original:
            overall_ok = False
            print(f"  [多余] 以下文件在 controlnet 目录中存在，但 size_1024 中缺少对应文件（共 {len(missing_in_original)} 个）：")
            for k in sorted(missing_in_original):
                print(f"    {k}{SUFFIX_CONTROLNET}")

        if not missing_in_controlnet and not missing_in_original:
            print(f"  [OK] 文件完全对应，共 {len(orig_set)} 对。")

    print("\n" + "=" * 70)
    if overall_ok:
        total = sum(len(v) for v in keys_original.values())
        print(f"[全部通过] 所有子目录文件一一对应，总计 {total} 对训练样本。")
    else:
        print("[检测失败] 存在不对应的文件，请查看上方详细信息。")
    print("=" * 70)

    return overall_ok


if __name__ == "__main__":
    ok = check_correspondence()
    sys.exit(0 if ok else 1)
