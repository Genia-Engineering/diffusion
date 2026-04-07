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

_ALLOW_PATTERNS_SANA = [
    "model_index.json",
    "transformer/config.json",
    "transformer/*.safetensors",
    "vae/config.json",
    "vae/*.safetensors",
    "text_encoder/config.json",
    "text_encoder/*.safetensors",
    "text_encoder/*.index.json",
    "tokenizer/**",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
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
    elif model_type == "sana":
        allow_patterns = _ALLOW_PATTERNS_SANA
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
    skip_unet: bool = False,
) -> dict:
    """加载 SD 1.5 训练所需全部组件。

    Args:
        unet_quantization_config: 可选 BitsAndBytesConfig，用于 QLoRA 4-bit 量化加载 UNet。
        skip_unet: True 时跳过 UNet 加载（调用方将自行加载，如 merged_unet_path 场景）。

    Returns:
        dict，其中 "unet" 在 skip_unet=True 时为 None。
    """
    path = resolve_model_path(pretrained_model_name_or_path, weights_dir, model_type="sd15")

    tokenizer = AutoTokenizer.from_pretrained(path, subfolder="tokenizer", use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(path, subfolder="text_encoder", torch_dtype=dtype)
    vae = AutoencoderKL.from_pretrained(path, subfolder="vae", torch_dtype=dtype)

    unet = None
    if not skip_unet:
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
    skip_unet: bool = False,
) -> dict:
    """加载 SDXL 训练所需全部组件（含双文本编码器）。

    Args:
        unet_quantization_config: 可选 BitsAndBytesConfig，用于 QLoRA 4-bit 量化加载 UNet。
        skip_unet: True 时跳过 UNet 加载（调用方将自行加载，如 merged_unet_path 场景）。

    Returns:
        dict，其中 "unet" 在 skip_unet=True 时为 None。
    """
    path = resolve_model_path(pretrained_model_name_or_path, weights_dir, model_type="sdxl")

    tokenizer   = AutoTokenizer.from_pretrained(path, subfolder="tokenizer",   use_fast=False)
    tokenizer_2 = AutoTokenizer.from_pretrained(path, subfolder="tokenizer_2", use_fast=False)
    text_encoder   = CLIPTextModel.from_pretrained(path, subfolder="text_encoder",   torch_dtype=dtype)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(path, subfolder="text_encoder_2", torch_dtype=dtype)
    vae  = AutoencoderKL.from_pretrained(path, subfolder="vae",  torch_dtype=dtype)

    unet = None
    if not skip_unet:
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


def load_sana_components(
    pretrained_model_name_or_path: str,
    weights_dir: str = None,
    dtype: torch.dtype = torch.float32,
    flow_shift: float = 3.0,
    load_text_encoder: bool = True,
) -> dict:
    """加载 Sana 0.6B 训练所需全部组件。

    Sana 使用 SanaTransformer2DModel (Linear DiT) 替代 UNet，
    Gemma-2-2B-IT 替代 T5/CLIP 文本编码器，AutoencoderDC (32x 压缩) 替代标准 VAE。

    Args:
        flow_shift: DPMSolverMultistepScheduler 的 flow_shift 参数（默认 3.0）。
        load_text_encoder: False 时跳过 Gemma2 和 tokenizer 的加载。

    Returns:
        {
            "vae": AutoencoderDC,
            "transformer": SanaTransformer2DModel,
            "text_encoder": Gemma2Model | None,
            "tokenizer": GemmaTokenizerFast | None,
            "noise_scheduler": DPMSolverMultistepScheduler,
        }
    """
    from diffusers import AutoencoderDC, SanaTransformer2DModel, DPMSolverMultistepScheduler
    from transformers import AutoTokenizer, AutoModel

    path = resolve_model_path(pretrained_model_name_or_path, weights_dir, model_type="sana")

    if load_text_encoder:
        tokenizer = AutoTokenizer.from_pretrained(path, subfolder="tokenizer")
        text_encoder = AutoModel.from_pretrained(path, subfolder="text_encoder", torch_dtype=dtype)
    else:
        tokenizer = None
        text_encoder = None

    vae = AutoencoderDC.from_pretrained(path, subfolder="vae", torch_dtype=dtype)
    transformer = SanaTransformer2DModel.from_pretrained(path, subfolder="transformer", torch_dtype=dtype)

    noise_scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
        flow_shift=flow_shift,
        algorithm_type="dpmsolver++",
        solver_order=2,
        solver_type="midpoint",
        lower_order_final=True,
        final_sigmas_type="zero",
    )

    return {
        "vae": vae,
        "transformer": transformer,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "noise_scheduler": noise_scheduler,
    }


def load_clip_vision_model(
    model_name_or_path: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    weights_dir: str = None,
    dtype: torch.dtype = torch.float32,
) -> dict:
    """加载 CLIP Vision 模型（用于 IP-Adapter 图像编码）。

    支持 OpenCLIP 和 HuggingFace CLIP 模型。返回 frozen 模型和预处理器。

    Args:
        model_name_or_path: HuggingFace model ID 或本地路径。
        weights_dir: 本地权重根目录，非 None 时检查本地缓存。
        dtype: 模型权重数据类型。

    Returns:
        {
            "model": CLIPVisionModelWithProjection (frozen),
            "processor": CLIPImageProcessor,
            "embed_dim": int (hidden_size of vision model),
        }
    """
    from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

    path = model_name_or_path
    if weights_dir is not None and not os.path.isabs(model_name_or_path):
        model_slug = model_name_or_path.replace("/", "--")
        local_path = os.path.join(weights_dir, model_slug)
        if os.path.isdir(local_path):
            logger.info(f"从本地加载 CLIP Vision: {local_path}")
            path = local_path
        else:
            logger.info(f"本地 CLIP 不存在，从 HuggingFace 下载: {model_name_or_path}")

    model = CLIPVisionModelWithProjection.from_pretrained(path, torch_dtype=dtype)
    processor = CLIPImageProcessor.from_pretrained(path)

    model.requires_grad_(False)
    model.eval()

    embed_dim = model.config.hidden_size
    logger.info(
        f"CLIP Vision loaded: {model_name_or_path}, "
        f"hidden_size={embed_dim}, frozen"
    )

    return {
        "model": model,
        "processor": processor,
        "embed_dim": embed_dim,
    }
