"""PixArt-Sigma ControlNet 推理 Pipeline — 用于训练验证和推理生成。

适配自 diffusers research project (pipeline_pixart_alpha_controlnet.py)，简化为本项目所需：
  - 支持 prompt_embeds + prompt_attention_mask（预缓存嵌入模式，文本编码器已卸载）
  - 条件图像 → VAE encode 或 CNN encode → latent → PixArtControlNetTransformerModel
  - 标准去噪循环 + learned sigma 处理
"""

import inspect
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
from transformers import T5EncoderModel, T5Tokenizer

from diffusers.image_processor import PixArtImageProcessor
from diffusers.models import AutoencoderKL, PixArtTransformer2DModel
from diffusers.pipelines import DiffusionPipeline, ImagePipelineOutput
from diffusers.schedulers import DPMSolverMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor

from models.controlnet_pixart import PixArtControlNetAdapterModel, PixArtControlNetTransformerModel


def _retrieve_timesteps(scheduler, num_inference_steps=None, device=None, timesteps=None, sigmas=None):
    if timesteps is not None:
        scheduler.set_timesteps(timesteps=timesteps, device=device)
        return scheduler.timesteps, len(scheduler.timesteps)
    elif sigmas is not None:
        scheduler.set_timesteps(sigmas=sigmas, device=device)
        return scheduler.timesteps, len(scheduler.timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device)
        return scheduler.timesteps, num_inference_steps


class PixArtControlNetPipeline(DiffusionPipeline):
    """PixArt-Sigma ControlNet Pipeline — 支持条件控制的文生图。

    使用 PixArtControlNetTransformerModel（联合 transformer + adapter）进行去噪，
    条件图像通过 VAE 或 CNN 编码到 latent 空间后注入 adapter。
    """

    _optional_components = ["tokenizer", "text_encoder", "controlnet"]
    model_cpu_offload_seq = "text_encoder->transformer->vae"

    def __init__(
        self,
        tokenizer: Optional[T5Tokenizer],
        text_encoder: Optional[T5EncoderModel],
        vae: AutoencoderKL,
        transformer: Union[PixArtControlNetTransformerModel, PixArtTransformer2DModel],
        controlnet: Optional[PixArtControlNetAdapterModel],
        scheduler: DPMSolverMultistepScheduler,
    ):
        super().__init__()

        if not isinstance(transformer, PixArtControlNetTransformerModel) and controlnet is not None:
            transformer = PixArtControlNetTransformerModel(
                transformer=transformer, controlnet=controlnet,
            )

        self.register_modules(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            controlnet=controlnet,
            scheduler=scheduler,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = PixArtImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def _encode_prompt(
        self,
        prompt,
        negative_prompt="",
        num_images_per_prompt=1,
        device=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
        max_sequence_length=300,
    ):
        if device is None:
            device = self._execution_device

        if prompt_embeds is not None:
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
            if prompt_attention_mask is not None:
                prompt_attention_mask = prompt_attention_mask.view(bs_embed, -1)
                prompt_attention_mask = prompt_attention_mask.repeat(num_images_per_prompt, 1)

            if negative_prompt_embeds is not None:
                seq_len_neg = negative_prompt_embeds.shape[1]
                negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
                negative_prompt_embeds = negative_prompt_embeds.view(
                    bs_embed * num_images_per_prompt, seq_len_neg, -1
                )
                if negative_prompt_attention_mask is not None:
                    negative_prompt_attention_mask = negative_prompt_attention_mask.view(bs_embed, -1)
                    negative_prompt_attention_mask = negative_prompt_attention_mask.repeat(num_images_per_prompt, 1)

            return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask

        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` must be str or list, got {type(prompt)}")

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        prompt_attention_mask = text_inputs.attention_mask.to(device)

        prompt_embeds = self.text_encoder(text_input_ids, attention_mask=prompt_attention_mask)[0]

        dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.controlnet.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        prompt_attention_mask = prompt_attention_mask.view(bs_embed, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_images_per_prompt, 1)

        uncond_input = self.tokenizer(
            [negative_prompt] * batch_size,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        negative_prompt_attention_mask = uncond_input.attention_mask.to(device)
        negative_prompt_embeds = self.text_encoder(
            uncond_input.input_ids.to(device), attention_mask=negative_prompt_attention_mask
        )[0]
        negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)

        seq_len_neg = negative_prompt_embeds.shape[1]
        negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len_neg, -1)
        negative_prompt_attention_mask = negative_prompt_attention_mask.view(bs_embed, -1)
        negative_prompt_attention_mask = negative_prompt_attention_mask.repeat(num_images_per_prompt, 1)

        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask

    def _prepare_control_latents(self, image, height, width, batch_size, num_images_per_prompt, device, dtype):
        """条件图像 → VAE latent 或 CNN 编码器输入格式。"""
        image = self.image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)

        repeat_by = batch_size * num_images_per_prompt if image.shape[0] == 1 else num_images_per_prompt
        image = image.repeat_interleave(repeat_by, dim=0)
        image = image.to(device=device, dtype=dtype)

        controlnet = getattr(self.transformer, "controlnet", None) or getattr(self, "controlnet", None)
        if controlnet is not None and controlnet.conditioning_mode == "cnn_encoder":
            return (image + 1.0) / 2.0

        image_latents = self.vae.encode(image).latent_dist.sample()
        image_latents = image_latents * self.vae.config.scaling_factor
        return image_latents

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, int(height) // self.vae_scale_factor, int(width) // self.vae_scale_factor)
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Optional[PIL.Image.Image] = None,
        negative_prompt: str = "",
        num_inference_steps: int = 20,
        timesteps: List[int] = None,
        guidance_scale: float = 4.5,
        num_images_per_prompt: int = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        controlnet_conditioning_scale: float = 1.0,
        output_type: str = "pil",
        return_dict: bool = True,
        max_sequence_length: int = 300,
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        height = height or self.transformer.config.sample_size * self.vae_scale_factor
        width = width or self.transformer.config.sample_size * self.vae_scale_factor

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        elif prompt_embeds is not None:
            batch_size = prompt_embeds.shape[0]
        else:
            batch_size = 1

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self._encode_prompt(
            prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            max_sequence_length=max_sequence_length,
        )

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        timesteps_tensor, num_inference_steps = _retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps,
        )

        # 条件图像编码
        image_latents = None
        if image is not None:
            controlnet = getattr(self.transformer, "controlnet", None) or getattr(self, "controlnet", None)
            cond_dtype = controlnet.dtype if controlnet is not None else prompt_embeds.dtype
            image_latents = self._prepare_control_latents(
                image, height, width,
                batch_size, num_images_per_prompt,
                device, cond_dtype,
            )
            if do_classifier_free_guidance:
                image_latents = torch.cat([image_latents] * 2)

        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            latent_channels, height, width,
            prompt_embeds.dtype, device, generator, latents,
        )

        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        if self.transformer.config.sample_size == 128:
            resolution = torch.tensor([height, width]).repeat(batch_size * num_images_per_prompt, 1)
            aspect_ratio = torch.tensor([float(height / width)]).repeat(batch_size * num_images_per_prompt, 1)
            resolution = resolution.to(dtype=prompt_embeds.dtype, device=device)
            aspect_ratio = aspect_ratio.to(dtype=prompt_embeds.dtype, device=device)
            if do_classifier_free_guidance:
                resolution = torch.cat([resolution, resolution], dim=0)
                aspect_ratio = torch.cat([aspect_ratio, aspect_ratio], dim=0)
            added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}

        extra_step_kwargs = {}
        if "generator" in set(inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["generator"] = generator

        for i, t in enumerate(timesteps_tensor):
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            current_timestep = t
            if not torch.is_tensor(current_timestep):
                dtype = torch.int64
                current_timestep = torch.tensor([current_timestep], dtype=dtype, device=latent_model_input.device)
            elif len(current_timestep.shape) == 0:
                current_timestep = current_timestep[None].to(latent_model_input.device)
            current_timestep = current_timestep.expand(latent_model_input.shape[0])

            noise_pred = self.transformer(
                latent_model_input,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attention_mask,
                timestep=current_timestep,
                controlnet_cond=image_latents,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            if self.transformer.config.out_channels // 2 == latent_channels:
                noise_pred = noise_pred.chunk(2, dim=1)[0]

            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

        if output_type != "latent":
            image_output = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image_output = self.image_processor.postprocess(image_output, output_type=output_type)
        else:
            image_output = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image_output,)
        return ImagePipelineOutput(images=image_output)
