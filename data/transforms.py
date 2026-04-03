"""数据增强流水线，支持 SD1.5 和 SDXL 的不同分辨率。"""

import random

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


class PairedRandomHorizontalFlip:
    """对图像和条件图像同步进行随机水平翻转。"""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: Image.Image, conditioning_image: Image.Image = None):
        if random.random() < self.p:
            image = TF.hflip(image)
            if conditioning_image is not None:
                conditioning_image = TF.hflip(conditioning_image)
        return image, conditioning_image


class AspectRatioResize:
    """根据目标桶分辨率 resize 图像，保持宽高比后中心裁剪或直接 resize。"""

    def __init__(self, target_size: tuple[int, int], center_crop: bool = True):
        self.target_w, self.target_h = target_size
        self.center_crop = center_crop

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.center_crop:
            w, h = image.size
            scale = max(self.target_w / w, self.target_h / h)
            new_w, new_h = int(w * scale + 0.5), int(h * scale + 0.5)
            image = image.resize((new_w, new_h), Image.LANCZOS)
            image = TF.center_crop(image, (self.target_h, self.target_w))
        else:
            image = image.resize((self.target_w, self.target_h), Image.LANCZOS)
        return image


class AspectRatioPad:
    """等比缩放至目标尺寸内（fit），不足部分居中 padding。

    同时生成 padding_mask（内容区域=1, padding 区域=0），
    供 loss 掩码使用，避免模型学习 padding 噪声。
    """

    def __init__(
        self,
        target_size: tuple[int, int],
        pad_color: tuple[int, ...] = (0, 0, 0),
    ):
        self.target_w, self.target_h = target_size
        self.pad_color = tuple(pad_color)

    def __call__(self, image: Image.Image) -> tuple[Image.Image, Image.Image]:
        """返回 (padded_image, padding_mask)。mask 为 L 模式 PIL 图像，255=内容/0=padding。"""
        w, h = image.size
        scale = min(self.target_w / w, self.target_h / h)
        new_w = round(w * scale)
        new_h = round(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)

        canvas = Image.new("RGB", (self.target_w, self.target_h), self.pad_color)
        paste_x = (self.target_w - new_w) // 2
        paste_y = (self.target_h - new_h) // 2
        canvas.paste(image, (paste_x, paste_y))

        mask = Image.new("L", (self.target_w, self.target_h), 0)
        mask.paste(Image.new("L", (new_w, new_h), 255), (paste_x, paste_y))

        return canvas, mask


def build_transforms(
    resolution: int,
    center_crop: bool = False,
    random_flip: bool = True,
) -> dict:
    """构建训练用的 transform 组件字典。

    返回字典而非 Compose，因为 Aspect Ratio Bucketing 需要动态设置目标分辨率。
    """
    normalize = T.Normalize([0.5], [0.5])  # 缩放到 [-1, 1]

    return {
        "resize": AspectRatioResize((resolution, resolution), center_crop),
        "flip": PairedRandomHorizontalFlip(0.5) if random_flip else None,
        "to_tensor": T.ToTensor(),
        "normalize": normalize,
    }


def apply_transforms(
    image: Image.Image,
    target_size: tuple[int, int],
    transforms_dict: dict,
    conditioning_image: Image.Image = None,
) -> dict:
    """对单张图像（及可选的条件图像）应用完整 transform 流水线。"""
    resize_fn = transforms_dict.get("resize")
    center_crop = resize_fn.center_crop if resize_fn is not None else False
    resizer = AspectRatioResize(target_size, center_crop=center_crop)
    image = resizer(image)
    if conditioning_image is not None:
        conditioning_image = resizer(conditioning_image)

    flip_fn = transforms_dict.get("flip")
    if flip_fn is not None:
        image, conditioning_image = flip_fn(image, conditioning_image)

    to_tensor = transforms_dict["to_tensor"]
    normalize = transforms_dict["normalize"]

    image_tensor = normalize(to_tensor(image))

    result = {"pixel_values": image_tensor}
    if conditioning_image is not None:
        result["conditioning_pixel_values"] = to_tensor(conditioning_image)

    return result
