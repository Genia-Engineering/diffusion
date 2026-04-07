"""显存优化工具 — 注意力后端、梯度检查点、内存格式、精度控制。"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def apply_memory_optimizations(
    unet: nn.Module = None,
    vae: nn.Module = None,
    text_encoder: nn.Module = None,
    text_encoder_2: nn.Module = None,
    controlnet: nn.Module = None,
    transformer: nn.Module = None,
    enable_gradient_checkpointing: bool = True,
    attention_backend: str = "sdpa",
    enable_channels_last: bool = True,
) -> None:
    """统一应用显存优化策略。

    Args:
        unet: UNet2DConditionModel（SD/SDXL），与 transformer 互斥
        transformer: PixArtTransformer2DModel（PixArt-Sigma），与 unet 互斥
        attention_backend: "sdpa" (PyTorch 原生, 默认) | "xformers" (需额外安装)
        enable_channels_last: 启用 channels_last 内存格式加速 CNN 层
    """
    denoise_model = unet or transformer
    denoise_name = "UNet" if unet is not None else "Transformer"

    if enable_gradient_checkpointing:
        if denoise_model is not None:
            _enable_gradient_checkpointing(denoise_model, denoise_name)
        if controlnet is not None:
            _enable_gradient_checkpointing(controlnet, "ControlNet")
        if text_encoder is not None and hasattr(text_encoder, "gradient_checkpointing_enable"):
            text_encoder.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled for TextEncoder")
        if text_encoder_2 is not None and hasattr(text_encoder_2, "gradient_checkpointing_enable"):
            text_encoder_2.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled for TextEncoder2")

    if denoise_model is not None:
        _apply_attention_backend(denoise_model, denoise_name, attention_backend)
    if vae is not None:
        _apply_attention_backend(vae, "VAE", attention_backend)
    if controlnet is not None:
        _apply_attention_backend(controlnet, "ControlNet", attention_backend)

    if enable_channels_last:
        if unet is not None:
            _apply_channels_last(unet, "UNet")
        if controlnet is not None:
            _apply_channels_last(controlnet, "ControlNet")

    if vae is not None:
        vae.enable_slicing()
        logger.info("VAE slicing enabled")
        vae.enable_tiling()
        logger.info("VAE tiling enabled")


def enable_tf32():
    """启用 TF32 精度 — Ampere+ GPU (A10G/A100/H100) float32 运算加速约 3x。"""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    logger.info("TF32 precision enabled for matmul and cudnn")


def _enable_gradient_checkpointing(model: nn.Module, name: str) -> None:
    if not hasattr(model, "enable_gradient_checkpointing"):
        return
    if hasattr(model, "_supports_gradient_checkpointing") and not model._supports_gradient_checkpointing:
        logger.info(f"{name} does not support gradient checkpointing, skipping")
        return
    try:
        model.enable_gradient_checkpointing()
        logger.info(f"Gradient checkpointing enabled for {name}")
    except ValueError:
        logger.info(f"{name} does not support gradient checkpointing, skipping")


def _apply_attention_backend(model: nn.Module, name: str, backend: str) -> None:
    """根据配置选择注意力计算后端。优先 SDPA，xformers 作为备选。"""
    if backend == "sdpa":
        _enable_sdpa(model, name)
    elif backend == "xformers":
        _enable_xformers(model, name)
    else:
        logger.warning(f"Unknown attention backend '{backend}', falling back to SDPA")
        _enable_sdpa(model, name)


def _enable_sdpa(model: nn.Module, name: str) -> None:
    """启用 PyTorch 原生 Scaled Dot-Product Attention (FlashAttention/MemEfficient/Math)。

    对于已使用专用注意力处理器的模型（如 Sana 的 SanaLinearAttnProcessor2_0），
    跳过替换以避免破坏模型特定的注意力机制。
    """
    try:
        if not hasattr(model, "set_attn_processor"):
            logger.info(f"{name} does not support set_attn_processor, using default attention")
            return

        from diffusers.models.attention_processor import AttnProcessor2_0

        current_procs = model.attn_processors
        proc_types = {type(p).__name__ for p in current_procs.values()}
        generic_types = {"AttnProcessor", "AttnProcessor2_0", "XFormersAttnProcessor"}
        if proc_types and not proc_types.issubset(generic_types):
            logger.info(
                f"{name} uses specialized attention processors ({proc_types}), "
                f"skipping AttnProcessor2_0 override to preserve model-specific attention"
            )
            return

        model.set_attn_processor(AttnProcessor2_0())
        logger.info(f"SDPA (AttnProcessor2_0) enabled for {name}")
    except Exception as e:
        logger.warning(f"Could not enable SDPA for {name}: {e}")


def _enable_xformers(model: nn.Module, name: str) -> None:
    try:
        if hasattr(model, "enable_xformers_memory_efficient_attention"):
            model.enable_xformers_memory_efficient_attention()
            logger.info(f"xformers memory-efficient attention enabled for {name}")
    except ImportError:
        logger.warning(
            f"xformers not available for {name}, falling back to SDPA."
        )
        _enable_sdpa(model, name)
    except Exception as e:
        logger.warning(f"Could not enable xformers for {name}: {e}, falling back to SDPA")
        _enable_sdpa(model, name)


def _apply_channels_last(model: nn.Module, name: str) -> None:
    """转换模型为 channels_last 内存格式，加速 CNN 卷积层 5-15%。"""
    try:
        model.to(memory_format=torch.channels_last)
        logger.info(f"channels_last memory format applied to {name}")
    except Exception as e:
        logger.warning(f"Could not apply channels_last to {name}: {e}")


def compute_grad_norm(parameters) -> float:
    """计算参数梯度的 L2 范数。"""
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            total_norm += p.grad.data.float().norm(2).item() ** 2
    return total_norm ** 0.5


def log_gpu_memory(prefix: str = "") -> None:
    """打印当前 GPU 显存使用量。"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"{prefix} GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def patch_caption_projection_layernorm(transformer: nn.Module) -> bool:
    """Inject LayerNorm into PixArt caption_projection to prevent bf16 overflow.

    Original: Linear(4096→1152) → GELU → Linear(1152→1152)
    Patched:  Linear(4096→1152) → LayerNorm(1152) → GELU → Linear(1152→1152)

    Safe to call multiple times; skips if already patched.

    Returns:
        True if patched, False if skipped.
    """
    if not hasattr(transformer, "caption_projection"):
        return False

    cp = transformer.caption_projection
    if not hasattr(cp, "linear_1"):
        return False
    if hasattr(cp, "norm"):
        return False

    hidden_size = cp.linear_1.out_features
    cp.norm = nn.LayerNorm(hidden_size).to(
        device=cp.linear_1.weight.device,
        dtype=cp.linear_1.weight.dtype,
    )

    def _ln_forward(caption):
        hidden_states = cp.linear_1(caption)
        hidden_states = cp.norm(hidden_states)
        hidden_states = cp.act_1(hidden_states)
        hidden_states = cp.linear_2(hidden_states)
        return hidden_states

    cp.forward = _ln_forward
    logger.info(
        f"caption_projection patched: injected LayerNorm({hidden_size}) "
        f"after linear_1 to prevent bf16 overflow"
    )
    return True
