"""LoRA (Low-Rank Adaptation) 实现 — 原生实现 + PEFT 双模式。

原生模式: 手动替换目标线性层为 LoRALinear，适合精细控制。
PEFT 模式: 使用 HuggingFace PEFT 库快速注入，适合快速原型。
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file


def _is_linear_layer(module: nn.Module) -> bool:
    """判断模块是否为线性层（支持原生 nn.Linear 和 bitsandbytes 4/8-bit 量化线性层）。"""
    if isinstance(module, nn.Linear):
        return True
    try:
        import bitsandbytes as bnb
        if isinstance(module, (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)):
            return True
    except ImportError:
        pass
    return False


def _get_linear_features(module: nn.Module) -> tuple[int, int]:
    """从线性层提取 (in_features, out_features)，兼容原生和量化线性层。"""
    return module.in_features, module.out_features


class LoRALinear(nn.Module):
    """LoRA 低秩适配层: W' = W + (alpha/r) * B @ A

    A 使用 Kaiming 初始化, B 初始化为零, 确保训练开始时 LoRA delta 为零。
    兼容 nn.Linear 和 bitsandbytes 量化线性层（QLoRA）。
    """

    def __init__(
        self,
        original_linear: nn.Module,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original_linear = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features, out_features = _get_linear_features(original_linear)

        # 获取原始层所在的设备和 dtype，确保 LoRA 适配器与之一致
        # （QLoRA 场景下 bnb 量化层已在 CUDA 上，LoRA 必须放同一设备避免 DDP 报错）
        orig_device = next(original_linear.parameters()).device
        orig_dtype = next(original_linear.parameters()).dtype
        # bnb 4-bit 量化层的 dtype 是 uint8，LoRA 适配器应使用计算精度
        if orig_dtype == torch.uint8:
            orig_dtype = torch.bfloat16

        self.lora_A = nn.Linear(in_features, rank, bias=False, device=orig_device, dtype=orig_dtype)
        self.lora_B = nn.Linear(rank, out_features, bias=False, device=orig_device, dtype=orig_dtype)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        self.original_linear.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.original_linear(x)
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return base_out + lora_out

    @property
    def weight(self):
        return self.original_linear.weight


class LoRAInjector:
    """LoRA 注入器 — 扫描模型并将目标线性层替换为 LoRALinear。"""

    DEFAULT_TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]

    @staticmethod
    def inject(
        model: nn.Module,
        rank: int = 16,
        alpha: float = 16.0,
        target_modules: list[str] = None,
        dropout: float = 0.0,
    ) -> list[str]:
        """向模型注入 LoRA 层。

        Returns:
            被注入的模块名称列表
        """
        if target_modules is None:
            target_modules = LoRAInjector.DEFAULT_TARGET_MODULES

        injected = []
        for name, module in list(model.named_modules()):
            for target in target_modules:
                if name.endswith(target) and _is_linear_layer(module):
                    parent_name = ".".join(name.split(".")[:-1])
                    child_name = name.split(".")[-1]
                    parent = model.get_submodule(parent_name) if parent_name else model

                    lora_layer = LoRALinear(module, rank, alpha, dropout)
                    setattr(parent, child_name, lora_layer)
                    injected.append(name)
                    break

        return injected

    @staticmethod
    def inject_unet(
        unet: nn.Module,
        rank: int = 16,
        alpha: float = 16.0,
        target_modules: list[str] = None,
    ) -> list[str]:
        return LoRAInjector.inject(unet, rank, alpha, target_modules)

    @staticmethod
    def inject_text_encoder(
        text_encoder: nn.Module,
        rank: int = 16,
        alpha: float = 16.0,
        target_modules: list[str] = None,
    ) -> list[str]:
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
        return LoRAInjector.inject(text_encoder, rank, alpha, target_modules)


def get_lora_params(model: nn.Module) -> list[nn.Parameter]:
    """提取模型中所有 LoRA 可训练参数。"""
    params = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            params.extend(module.lora_A.parameters())
            params.extend(module.lora_B.parameters())
    return params


def save_lora_weights(model: nn.Module, save_path: str) -> None:
    """仅保存 LoRA delta 权重（safetensors 格式）。"""
    lora_state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state[f"{name}.lora_A.weight"] = module.lora_A.weight.data.cpu()
            lora_state[f"{name}.lora_B.weight"] = module.lora_B.weight.data.cpu()
    save_file(lora_state, save_path)


def load_lora_weights(model: nn.Module, load_path: str) -> None:
    """加载 LoRA delta 权重到已注入 LoRA 的模型。"""
    state = load_file(load_path)
    model_state = dict(model.named_parameters())
    for key, value in state.items():
        if key in model_state:
            model_state[key].data.copy_(value)


def merge_lora_to_base(model: nn.Module) -> nn.Module:
    """将所有 LoRALinear 层合并回标准 nn.Linear，就地修改模型。

    合并公式: W_merged = W_original + (alpha/rank) * B @ A
    合并后模型不再包含任何 LoRA 结构，可直接以 diffusers 格式保存。

    对于 QLoRA（量化基础层），会先反量化为 fp16/bf16 再合并。
    """
    import logging as _logging
    _logger = _logging.getLogger(__name__)

    merge_count = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, LoRALinear):
            continue

        orig = module.original_linear
        is_quantized = not isinstance(orig, nn.Linear) and _is_linear_layer(orig)

        if is_quantized:
            _logger.info(f"Dequantizing {name} before merge (QLoRA → fp16)")
            in_f, out_f = _get_linear_features(orig)
            dequant_weight = orig.weight.data.dequantize().to(torch.float16)
            new_linear = nn.Linear(in_f, out_f, bias=orig.bias is not None).to(dequant_weight.device)
            new_linear.weight.data.copy_(dequant_weight)
            if orig.bias is not None:
                new_linear.bias.data.copy_(orig.bias.data.to(torch.float16))
            module.original_linear = new_linear
            orig = new_linear

        with torch.no_grad():
            delta = module.lora_B.weight @ module.lora_A.weight  # (out, in)
            orig.weight.add_(delta.to(orig.weight.dtype) * module.scaling)

        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, orig)
        merge_count += 1

    return model


# ======================== PEFT 模式 ========================

def inject_lora_peft(
    model: nn.Module,
    rank: int = 16,
    alpha: float = 16.0,
    target_modules: list[str] = None,
    task_type: str = None,
):
    """使用 HuggingFace PEFT 库注入 LoRA（快速原型模式）。"""
    from peft import LoraConfig, get_peft_model

    if target_modules is None:
        target_modules = LoRAInjector.DEFAULT_TARGET_MODULES

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
    )
    return get_peft_model(model, config)
