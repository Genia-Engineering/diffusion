"""统一模型加载入口 — 支持 SD1.5 和 SDXL 的组件加载。

下载策略:
  - 使用 allow_patterns 白名单，只拉取训练必需的子目录
  - 忽略 safety_checker / feature_extractor / fp16 副本 / onnx / flax / tf 等无关文件
  - SD1.5 必需: unet, vae, text_encoder, tokenizer, scheduler
  - SDXL  必需: unet, vae, text_encoder, text_encoder_2, tokenizer, tokenizer_2, scheduler
"""

import logging
import os

import torch
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection, T5EncoderModel

logger = logging.getLogger(__name__)

# 各模型类型训练所需的子目录白名单
_ALLOW_PATTERNS_SD15 = [
    "model_index.json",
    "unet/config.json",
    "unet/*.safetensors",
    "vae/config.json",
    "vae/*.safetensors",
    "text_encoder/config.json",
    "text_encoder/*.safetensors",
    "text_encoder/*.index.json",    # 分片索引文件（sharded model）
    "tokenizer/**",
    "scheduler/**",
]

_ALLOW_PATTERNS_SDXL = [
    "model_index.json",
    "unet/config.json",
    "unet/*.safetensors",
    "vae/config.json",
    "vae/*.safetensors",
    "text_encoder/config.json",
    "text_encoder/*.safetensors",
    "text_encoder/*.index.json",    # 分片索引文件
    "text_encoder_2/config.json",
    "text_encoder_2/*.safetensors",
    "text_encoder_2/*.index.json",  # 分片索引文件
    "tokenizer/**",
    "tokenizer_2/**",
    "scheduler/**",
]

_ALLOW_PATTERNS_PIXART_SIGMA = [
    "model_index.json",
    "transformer/config.json",
    "transformer/*.safetensors",
    "vae/config.json",
    "vae/*.safetensors",
    "text_encoder/config.json",
    "text_encoder/*.safetensors",
    "text_encoder/*.index.json",    # 分片索引文件（T5 sharded model）
    "text_encoder/spiece.model",
    "tokenizer/**",
    "scheduler/**",
]

# 无论哪种模型都排除的文件
_IGNORE_PATTERNS_COMMON = [
    "*.fp16.safetensors",      # fp16 副本，训练用 bf16/fp32 不需要
    "*.msgpack",               # Flax 权重
    "*.h5",                    # Keras/TF 权重
    "flax_model*",
    "tf_model*",
    "rust_model*",
    "*.onnx",
    "*.pb",
    "*.bin",                   # 旧版 PyTorch bin 格式，优先用 safetensors
    "safety_checker/**",
    "feature_extractor/**",
    "image_encoder/**",
]


def resolve_model_path(
    pretrained_model_name_or_path: str,
    weights_dir: str = None,
    model_type: str = "sd15",
) -> str:
    """解析模型路径：优先从本地 weights_dir 加载，不存在则按模型类型下载最小必要权重。

    Args:
        pretrained_model_name_or_path: HuggingFace model ID 或本地绝对路径。
        weights_dir: 本地权重根目录。为 None 时使用 HuggingFace 默认缓存。
        model_type: "sd15" 或 "sdxl"，决定下载哪些子目录。

    Returns:
        实际加载时使用的模型路径。
    """
    if weights_dir is None:
        return pretrained_model_name_or_path

    if os.path.isabs(pretrained_model_name_or_path) and os.path.isdir(pretrained_model_name_or_path):
        return pretrained_model_name_or_path

    model_slug = pretrained_model_name_or_path.replace("/", "--")
    local_path = os.path.join(weights_dir, model_slug)

    if os.path.isdir(local_path) and os.path.exists(os.path.join(local_path, "model_index.json")):
        logger.info(f"找到本地权重，从缓存加载: {local_path}")
        return local_path

    if model_type == "pixart_sigma":
        allow_patterns = _ALLOW_PATTERNS_PIXART_SIGMA
    elif model_type == "sdxl":
        allow_patterns = _ALLOW_PATTERNS_SDXL
    else:
        allow_patterns = _ALLOW_PATTERNS_SD15
    logger.info(
        f"本地权重不存在，开始下载 [{model_type}]: {pretrained_model_name_or_path} → {local_path}\n"
        f"  只下载: {allow_patterns}\n"
        f"  忽略:   {_IGNORE_PATTERNS_COMMON}"
    )
    os.makedirs(local_path, exist_ok=True)

    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=pretrained_model_name_or_path,
        local_dir=local_path,
        allow_patterns=allow_patterns,
        ignore_patterns=_IGNORE_PATTERNS_COMMON,
    )
    logger.info(f"下载完成: {local_path}")
    return local_path


def load_sd15_components(
    pretrained_model_name_or_path: str,
    weights_dir: str = None,
    dtype: torch.dtype = torch.float32,
    unet_quantization_config=None,
) -> dict:
    """加载 SD 1.5 训练所需全部组件。

    Args:
        unet_quantization_config: 可选 BitsAndBytesConfig，用于 QLoRA 4-bit 量化加载 UNet。

    Returns:
        {
            "vae": AutoencoderKL,
            "unet": UNet2DConditionModel,
            "text_encoder": CLIPTextModel,
            "tokenizer": AutoTokenizer,
            "noise_scheduler": DDPMScheduler,
        }
    """
    path = resolve_model_path(pretrained_model_name_or_path, weights_dir, model_type="sd15")

    tokenizer = AutoTokenizer.from_pretrained(path, subfolder="tokenizer", use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(path, subfolder="text_encoder", torch_dtype=dtype)
    vae = AutoencoderKL.from_pretrained(path, subfolder="vae", torch_dtype=dtype)

    if unet_quantization_config is not None:
        unet = UNet2DConditionModel.from_pretrained(
            path, subfolder="unet", quantization_config=unet_quantization_config,
        )
        logger.info("UNet loaded with quantization (QLoRA mode)")
    else:
        unet = UNet2DConditionModel.from_pretrained(path, subfolder="unet", torch_dtype=dtype)

    noise_scheduler = DDPMScheduler.from_pretrained(path, subfolder="scheduler")

    return {
        "vae": vae,
        "unet": unet,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "noise_scheduler": noise_scheduler,
    }


def load_sdxl_components(
    pretrained_model_name_or_path: str,
    weights_dir: str = None,
    dtype: torch.dtype = torch.float32,
    unet_quantization_config=None,
) -> dict:
    """加载 SDXL 训练所需全部组件（含双文本编码器）。

    Args:
        unet_quantization_config: 可选 BitsAndBytesConfig，用于 QLoRA 4-bit 量化加载 UNet。

    Returns:
        {
            "vae": AutoencoderKL,
            "unet": UNet2DConditionModel,
            "text_encoder": CLIPTextModel,
            "text_encoder_2": CLIPTextModelWithProjection,
            "tokenizer": AutoTokenizer,
            "tokenizer_2": AutoTokenizer,
            "noise_scheduler": DDPMScheduler,
        }
    """
    path = resolve_model_path(pretrained_model_name_or_path, weights_dir, model_type="sdxl")

    tokenizer   = AutoTokenizer.from_pretrained(path, subfolder="tokenizer",   use_fast=False)
    tokenizer_2 = AutoTokenizer.from_pretrained(path, subfolder="tokenizer_2", use_fast=False)
    text_encoder   = CLIPTextModel.from_pretrained(path, subfolder="text_encoder",   torch_dtype=dtype)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(path, subfolder="text_encoder_2", torch_dtype=dtype)
    vae  = AutoencoderKL.from_pretrained(path, subfolder="vae",  torch_dtype=dtype)

    if unet_quantization_config is not None:
        unet = UNet2DConditionModel.from_pretrained(
            path, subfolder="unet", quantization_config=unet_quantization_config,
        )
        logger.info("UNet loaded with quantization (QLoRA mode)")
    else:
        unet = UNet2DConditionModel.from_pretrained(path, subfolder="unet", torch_dtype=dtype)

    noise_scheduler = DDPMScheduler.from_pretrained(path, subfolder="scheduler")

    return {
        "vae": vae,
        "unet": unet,
        "text_encoder": text_encoder,
        "text_encoder_2": text_encoder_2,
        "tokenizer": tokenizer,
        "tokenizer_2": tokenizer_2,
        "noise_scheduler": noise_scheduler,
    }


def patch_fm_scheduler_for_pipeline(scheduler) -> None:
    """为 FlowMatchEulerDiscreteScheduler 补上 PixArtSigmaPipeline 所需的缺失属性。

    diffusers >=0.37 的 FlowMatchEulerDiscreteScheduler 移除了部分旧接口，
    但 PixArtSigmaPipeline 仍然依赖它们：
      - init_noise_sigma: prepare_latents() 用来缩放初始噪声，FM 下为 1.0（恒等）
      - scale_model_input: 去噪循环中缩放模型输入，FM 下为恒等函数
    """
    if not hasattr(scheduler, "init_noise_sigma"):
        scheduler.init_noise_sigma = 1.0
    if not hasattr(scheduler, "scale_model_input"):
        scheduler.scale_model_input = lambda sample, *args, **kwargs: sample


def load_pixart_sigma_components(
    pretrained_model_name_or_path: str,
    weights_dir: str = None,
    dtype: torch.dtype = torch.float32,
    flow_matching: bool = True,
    scheduler_shift: float = 1.0,
    load_text_encoder: bool = True,
    init_transformer_randomly: bool = False,
) -> dict:
    """加载 PixArt-Sigma 训练所需全部组件。

    PixArt-Sigma 使用 DiT (PixArtTransformer2DModel) 替代 UNet，
    T5-XXL 替代 CLIP 文本编码器，VAE 与 SDXL 相同。

    Args:
        flow_matching: True 时使用 FlowMatchEulerDiscreteScheduler (Rectified Flow)，
            False 时使用 DDPMScheduler (传统扩散，供 ControlNet 等未迁移的训练器使用)。
        scheduler_shift: FlowMatchEuler 的 sigma shift 参数（仅 flow_matching=True 时生效）。
            1024px 图像推荐 1.0-3.0（SD3 用 3.0）。
        load_text_encoder: False 时跳过 T5-XXL 和 tokenizer 的加载（节省 ~10GB 内存），
            返回值中 text_encoder 和 tokenizer 为 None。
        init_transformer_randomly: True 时只加载 Transformer 架构配置，权重随机初始化。
            用于验证条件编码器是否能驱动从零训练。

    Returns:
        {
            "vae": AutoencoderKL,
            "transformer": PixArtTransformer2DModel,
            "text_encoder": T5EncoderModel | None,
            "tokenizer": AutoTokenizer | None,
            "noise_scheduler": FlowMatchEulerDiscreteScheduler | DDPMScheduler,
        }
    """
    from diffusers import PixArtTransformer2DModel

    path = resolve_model_path(pretrained_model_name_or_path, weights_dir, model_type="pixart_sigma")

    if load_text_encoder:
        tokenizer = AutoTokenizer.from_pretrained(path, subfolder="tokenizer")
        text_encoder = T5EncoderModel.from_pretrained(path, subfolder="text_encoder", torch_dtype=dtype)
    else:
        tokenizer = None
        text_encoder = None

    vae = AutoencoderKL.from_pretrained(path, subfolder="vae", torch_dtype=dtype)

    if init_transformer_randomly:
        config = PixArtTransformer2DModel.load_config(path, subfolder="transformer")
        transformer = PixArtTransformer2DModel.from_config(config).to(dtype)
        logger.info("Transformer 随机初始化（未加载预训练权重）")
    else:
        transformer = PixArtTransformer2DModel.from_pretrained(path, subfolder="transformer", torch_dtype=dtype)

    if flow_matching:
        from diffusers import FlowMatchEulerDiscreteScheduler
        noise_scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=scheduler_shift,
        )
        patch_fm_scheduler_for_pipeline(noise_scheduler)
    else:
        noise_scheduler = DDPMScheduler.from_pretrained(path, subfolder="scheduler")

    return {
        "vae": vae,
        "transformer": transformer,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "noise_scheduler": noise_scheduler,
    }
