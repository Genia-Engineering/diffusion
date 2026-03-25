"""PixArt-Sigma 图像条件推理 Pipeline — 封装图像编码 + PixArtSigmaPipeline。

独立推理时使用，训练验证阶段直接在 trainer 内完成编码。

用法:
    pipeline = PixArtImgCondPipeline.from_pretrained(
        transformer_path="outputs/pixart_img_cond_floorplan/checkpoints/step_001800",
        vae_path="weights/PixArt-alpha--PixArt-Sigma-XL-2-1024-MS",
        projector_path="outputs/pixart_img_cond_floorplan/checkpoints/step_001800/projector.pt",
        image_encoder_config={...},
    )
    images = pipeline(conditioning_image=pil_image, guidance_scale=4.5)
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torchvision.transforms as T
from diffusers import FlowMatchEulerDiscreteScheduler, PixArtSigmaPipeline
from PIL import Image

from models.image_encoder import VAEImageEncoder, DINOv2ImageEncoder, build_image_encoder
from models.model_loader import patch_fm_scheduler_for_pipeline

logger = logging.getLogger(__name__)


class PixArtImgCondPipeline:
    """图像条件 PixArt-Sigma 推理封装。

    内部使用标准 PixArtSigmaPipeline，通过预先编码条件图像
    生成 prompt_embeds 来绕过文本编码器。
    """

    def __init__(
        self,
        pipeline: PixArtSigmaPipeline,
        projector: nn.Module,
        image_encoder_type: str,
        vae_for_encoding: Optional[nn.Module] = None,
        dinov2_backbone: Optional[nn.Module] = None,
        dinov2_processor=None,
        dinov2_resolution: int = 518,
        device: torch.device = torch.device("cuda"),
    ):
        self.pipeline = pipeline
        self.projector = projector.to(device).eval()
        self.image_encoder_type = image_encoder_type
        self.vae_for_encoding = vae_for_encoding
        self.dinov2_backbone = dinov2_backbone
        self.dinov2_processor = dinov2_processor
        self.dinov2_resolution = dinov2_resolution
        self.device = device

    @torch.no_grad()
    def encode_conditioning(
        self, image: Image.Image,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """将条件图像编码为 (prompt_embeds, attention_mask)。"""
        if self.image_encoder_type == "vae":
            to_tensor = T.ToTensor()
            normalize = T.Normalize([0.5], [0.5])
            img_t = normalize(to_tensor(image.resize((1024, 1024), Image.LANCZOS)))
            img_t = img_t.unsqueeze(0).to(self.device)
            latent = self.vae_for_encoding.encode(img_t).latent_dist.mode()
            latent = latent * self.vae_for_encoding.config.scaling_factor
            return self.projector(latent)
        else:
            img_resized = image.resize(
                (self.dinov2_resolution, self.dinov2_resolution), Image.LANCZOS
            )
            inputs = self.dinov2_processor(
                images=[img_resized], return_tensors="pt"
            ).pixel_values.to(self.device)
            features = self.dinov2_backbone(inputs).last_hidden_state[:, 1:, :]
            return self.projector(features)

    @torch.no_grad()
    def __call__(
        self,
        conditioning_image: Image.Image,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.5,
        seed: int = 42,
        num_images_per_prompt: int = 1,
        height: int = 1024,
        width: int = 1024,
    ):
        prompt_embeds, prompt_mask = self.encode_conditioning(conditioning_image)
        seq_len = prompt_embeds.shape[1]
        neg_embeds = torch.zeros_like(prompt_embeds)
        neg_mask = torch.ones(1, seq_len, dtype=torch.long, device=self.device)

        generator = torch.Generator(device=self.device).manual_seed(seed)

        output = self.pipeline(
            prompt=None,
            negative_prompt=None,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=neg_embeds,
            prompt_attention_mask=prompt_mask,
            negative_prompt_attention_mask=neg_mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            num_images_per_prompt=num_images_per_prompt,
            height=height,
            width=width,
        )
        return output
