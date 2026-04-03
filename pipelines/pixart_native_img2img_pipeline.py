"""PixArt-Sigma Native Img2Img 推理 Pipeline — 色块图条件生成。

通过 8-channel 通道拼接 + 多层残差注入实现结构化条件控制。
支持 Classifier-Free Guidance: 有条件分支 (reference_latent) + 无条件分支 (零 latent)。

用法:
    pipeline = PixArtNativeImg2ImgPipeline(
        model=wrapper_model,
        vae=vae,
        scheduler=scheduler,
        device=torch.device("cuda"),
    )
    images = pipeline(
        conditioning_image=pil_image,
        guidance_scale=4.5,
        num_inference_steps=50,
    )
"""

import logging
from typing import Optional

import torch
import torchvision.transforms as T
from PIL import Image

logger = logging.getLogger(__name__)


class PixArtNativeImg2ImgPipeline:
    """Native Img2Img PixArt-Sigma 推理封装。

    不继承 DiffusionPipeline，直接操作 NativeImg2ImgPixArtWrapper
    实现自定义的 CFG 去噪循环。
    支持 FlowMatchEulerDiscreteScheduler (FM) 和 DPMSolverMultistepScheduler (DDPM) 等调度器。
    """

    def __init__(
        self,
        model,
        vae,
        scheduler,
        device: torch.device = torch.device("cuda"),
    ):
        self.model = model
        self.vae = vae
        self.scheduler = scheduler
        self.device = device
        self._progress_bar_config = {}

    def set_progress_bar_config(self, **kwargs):
        self._progress_bar_config = kwargs

    @torch.no_grad()
    def encode_conditioning(
        self, image: Image.Image, height: int = 1024, width: int = 1024,
    ) -> torch.Tensor:
        """Encode a conditioning image into VAE latent space."""
        to_tensor = T.ToTensor()
        normalize = T.Normalize([0.5], [0.5])

        img_resized = image.resize((width, height), Image.LANCZOS)
        img_t = normalize(to_tensor(img_resized)).unsqueeze(0).to(self.device)

        latent = self.vae.encode(img_t).latent_dist.mode()
        latent = latent * self.vae.config.scaling_factor
        return latent

    @torch.no_grad()
    def __call__(
        self,
        conditioning_image: Optional[Image.Image] = None,
        reference_latent: Optional[torch.Tensor] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.5,
        seed: int = 42,
        height: int = 1024,
        width: int = 1024,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ):
        """Run denoising loop with CFG.

        Provide either `conditioning_image` (PIL) or `reference_latent` (pre-encoded tensor).
        """
        if reference_latent is None:
            if conditioning_image is None:
                raise ValueError("Must provide either conditioning_image or reference_latent")
            reference_latent = self.encode_conditioning(conditioning_image, height, width)

        reference_latent = reference_latent.to(self.device, dtype=self.model.dtype)

        lat_h = height // 8
        lat_w = width // 8

        if generator is None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        latents = torch.randn(1, 4, lat_h, lat_w, generator=generator, device=self.device, dtype=self.model.dtype)

        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        # DDPM 类 scheduler (如 DPMSolver) 的 init_noise_sigma 可达 ~14.5,
        # 不缩放会导致首步噪声量级错误; FM scheduler 为 1.0 (no-op)
        latents = latents * getattr(self.scheduler, "init_noise_sigma", 1.0)

        do_cfg = guidance_scale > 1.0

        added_cond_kwargs = {
            "resolution": torch.tensor([height, width], device=self.device).unsqueeze(0),
            "aspect_ratio": torch.tensor([height / width], device=self.device).unsqueeze(0),
        }

        if do_cfg:
            added_cond_kwargs_cfg = {
                "resolution": added_cond_kwargs["resolution"].repeat(2, 1),
                "aspect_ratio": added_cond_kwargs["aspect_ratio"].repeat(2, 1),
            }

        for t in timesteps:
            timestep = t.unsqueeze(0)

            if do_cfg:
                latent_model_input = torch.cat([latents, latents], dim=0)
                ref_input = torch.cat([reference_latent, torch.zeros_like(reference_latent)], dim=0)
                timestep_input = timestep.expand(2)

                output = self.model(
                    noisy_latent=latent_model_input,
                    reference_latent=ref_input,
                    timestep=timestep_input,
                    added_cond_kwargs=added_cond_kwargs_cfg,
                    return_dict=False,
                )[0]

                noise_cond, noise_uncond = output.chunk(2, dim=0)
                noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            else:
                output = self.model(
                    noisy_latent=latents,
                    reference_latent=reference_latent,
                    timestep=timestep,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = output

            if noise_pred.shape[1] != 4:
                noise_pred, _ = noise_pred.chunk(2, dim=1)

            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        latents = latents / self.vae.config.scaling_factor
        images = self.vae.decode(latents.to(self.vae.dtype)).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()

        from diffusers.image_processor import VaeImageProcessor
        image_processor = VaeImageProcessor()
        pil_images = image_processor.numpy_to_pil(images)

        return type("PipelineOutput", (), {"images": pil_images})()
