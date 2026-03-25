"""检查点管理 — 保存/恢复模型权重、优化器状态、训练进度。

保存策略:
  - 每 save_steps 步保存一次
  - 训练结束时强制保存
  - 保留最近 keep_last_n 份，自动删除旧检查点

保存内容:
  outputs/checkpoints/step_XXXXXX/
    ├── lora_unet.safetensors          # LoRA UNet delta 权重
    ├── lora_text_encoder.safetensors  # LoRA TE delta 权重（如果有）
    ├── controlnet/                    # ControlNet 完整权重（diffusers 格式）
    ├── optimizer.pt                   # 优化器状态（含动量、lr 等）
    ├── lr_scheduler.pt                # 学习率调度器状态
    └── training_state.json            # 训练进度（step/epoch/seed）
"""

import json
import logging
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file, load_file

logger = logging.getLogger(__name__)


class CheckpointManager:
    """训练检查点管理器。"""

    def __init__(
        self,
        save_dir: str = "./outputs/checkpoints",
        keep_last_n: int = 3,
    ):
        self.save_dir = Path(save_dir)
        self.keep_last_n = keep_last_n
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        step: int,
        global_epoch: int,
        unet=None,
        text_encoder=None,
        text_encoder_2=None,
        controlnet=None,
        transformer=None,
        accelerator=None,
        optimizer=None,
        lr_scheduler=None,
        seed: int = 42,
        is_lora: bool = True,
    ) -> Path:
        """保存完整检查点。"""
        ckpt_dir = self.save_dir / f"step_{step:06d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if is_lora:
            self._save_lora_weights(unet, text_encoder, text_encoder_2, ckpt_dir)
        if controlnet is not None:
            self._save_controlnet(controlnet, ckpt_dir)
        if transformer is not None:
            self._save_transformer(transformer, accelerator, ckpt_dir)

        if optimizer is not None:
            torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
        if lr_scheduler is not None:
            torch.save(lr_scheduler.state_dict(), ckpt_dir / "lr_scheduler.pt")

        training_state = {
            "step": step,
            "epoch": global_epoch,
            "seed": seed,
        }
        with open(ckpt_dir / "training_state.json", "w") as f:
            json.dump(training_state, f, indent=2)

        logger.info(f"Checkpoint saved: {ckpt_dir}")

        self._cleanup_old_checkpoints()
        return ckpt_dir

    def _save_lora_weights(self, unet, text_encoder, text_encoder_2, ckpt_dir: Path):
        from models.lora import save_lora_weights

        if unet is not None:
            save_lora_weights(unet, str(ckpt_dir / "lora_unet.safetensors"))
        if text_encoder is not None:
            save_lora_weights(text_encoder, str(ckpt_dir / "lora_text_encoder.safetensors"))
        if text_encoder_2 is not None:
            save_lora_weights(text_encoder_2, str(ckpt_dir / "lora_text_encoder_2.safetensors"))

    def _save_controlnet(self, controlnet, ckpt_dir: Path):
        cn_dir = ckpt_dir / "controlnet"
        controlnet.save_pretrained(str(cn_dir))

    def _save_transformer(self, transformer, accelerator, ckpt_dir: Path):
        """保存完整 transformer 权重（diffusers 格式），兼容 DeepSpeed ZeRO。"""
        tf_dir = ckpt_dir / "transformer"
        if accelerator is not None:
            unwrapped = accelerator.unwrap_model(transformer)
            state_dict = accelerator.get_state_dict(transformer)
            unwrapped.save_pretrained(str(tf_dir), state_dict=state_dict)
        else:
            transformer.save_pretrained(str(tf_dir))

    def _cleanup_old_checkpoints(self):
        """保留最近 keep_last_n 个检查点，删除旧的。"""
        if self.keep_last_n <= 0:
            return

        ckpt_dirs = sorted(
            [d for d in self.save_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
            key=lambda d: int(d.name.split("_")[1]),
        )
        while len(ckpt_dirs) > self.keep_last_n:
            old_dir = ckpt_dirs.pop(0)
            shutil.rmtree(old_dir)
            logger.info(f"Removed old checkpoint: {old_dir}")

    def load(
        self,
        checkpoint_dir: str,
        unet=None,
        text_encoder=None,
        text_encoder_2=None,
        controlnet=None,
        transformer=None,
        optimizer=None,
        lr_scheduler=None,
        is_lora: bool = True,
    ) -> dict:
        """加载检查点，恢复训练状态。

        Returns:
            training_state dict: {"step": int, "epoch": int, "seed": int}
        """
        ckpt_dir = Path(checkpoint_dir)

        if is_lora:
            self._load_lora_weights(unet, text_encoder, text_encoder_2, ckpt_dir)

        if controlnet is not None:
            cn_dir = ckpt_dir / "controlnet"
            if cn_dir.exists():
                controlnet = self._load_controlnet_weights(controlnet, cn_dir)

        if transformer is not None:
            tf_dir = ckpt_dir / "transformer"
            if tf_dir.exists():
                from diffusers import PixArtTransformer2DModel
                loaded_tf = PixArtTransformer2DModel.from_pretrained(str(tf_dir))
                src_sd = loaded_tf.state_dict()
                del loaded_tf
                # torch.compile wraps the model as OptimizedModule, adding "_orig_mod." prefix to all keys.
                # Remap checkpoint keys to match the current model's key space.
                model_keys = set(transformer.state_dict().keys())
                sample_key = next(iter(model_keys))
                if sample_key.startswith("_orig_mod.") and not next(iter(src_sd.keys())).startswith("_orig_mod."):
                    src_sd = {"_orig_mod." + k: v for k, v in src_sd.items()}
                elif not sample_key.startswith("_orig_mod.") and next(iter(src_sd.keys())).startswith("_orig_mod."):
                    src_sd = {k[len("_orig_mod."):]: v for k, v in src_sd.items()}
                transformer.load_state_dict(src_sd)

        if optimizer is not None:
            opt_path = ckpt_dir / "optimizer.pt"
            if opt_path.exists():
                optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))

        if lr_scheduler is not None:
            sched_path = ckpt_dir / "lr_scheduler.pt"
            if sched_path.exists():
                lr_scheduler.load_state_dict(torch.load(sched_path, map_location="cpu"))

        state_path = ckpt_dir / "training_state.json"
        if state_path.exists():
            with open(state_path) as f:
                training_state = json.load(f)
        else:
            training_state = {"step": 0, "epoch": 0, "seed": 42}

        logger.info(f"Checkpoint loaded from {ckpt_dir}, resuming from step {training_state['step']}")
        return training_state

    @staticmethod
    def _load_controlnet_weights(controlnet, cn_dir: Path):
        """加载 ControlNet 权重，自动检测 UNet-based 还是 PixArt adapter 格式。"""
        from models.controlnet_pixart import PixArtControlNetAdapterModel

        if isinstance(controlnet, PixArtControlNetAdapterModel):
            loaded_cn = PixArtControlNetAdapterModel.from_pretrained(str(cn_dir))
            controlnet.load_state_dict(loaded_cn.state_dict())
            del loaded_cn
        else:
            from diffusers import ControlNetModel
            loaded_cn = ControlNetModel.from_pretrained(str(cn_dir))
            controlnet.load_state_dict(loaded_cn.state_dict())
            del loaded_cn
        return controlnet

    def _load_lora_weights(self, unet, text_encoder, text_encoder_2, ckpt_dir: Path):
        from models.lora import load_lora_weights

        lora_unet_path = ckpt_dir / "lora_unet.safetensors"
        if unet is not None and lora_unet_path.exists():
            load_lora_weights(unet, str(lora_unet_path))

        lora_te_path = ckpt_dir / "lora_text_encoder.safetensors"
        if text_encoder is not None and lora_te_path.exists():
            load_lora_weights(text_encoder, str(lora_te_path))

        lora_te2_path = ckpt_dir / "lora_text_encoder_2.safetensors"
        if text_encoder_2 is not None and lora_te2_path.exists():
            load_lora_weights(text_encoder_2, str(lora_te2_path))

    def get_latest_checkpoint(self) -> str | None:
        """返回最新检查点路径，如果没有则返回 None。"""
        ckpt_dirs = sorted(
            [d for d in self.save_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
            key=lambda d: int(d.name.split("_")[1]),
        )
        if ckpt_dirs:
            return str(ckpt_dirs[-1])
        return None
