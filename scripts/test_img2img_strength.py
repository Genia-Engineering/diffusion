"""测试不同 strength 下的 img2img 结果（Flow Matching PixArt-Sigma）。

对比图为三栏：[原图 | 半透明叠加 | 生成图]

用法:
  # 单张参考图，多个 strength
  python scripts/test_img2img_strength.py \
    --ckpt outputs/pixart_sigma_floorplan/checkpoints/step_001800/transformer \
    --input_image data/data/llm_1024/floor/some_image.png \
    --strengths 0.3 0.5 0.7 0.9 1.0

  # 指定输出目录、guidance_scale、步数
  python scripts/test_img2img_strength.py \
    --ckpt outputs/pixart_sigma_floorplan/checkpoints/step_001800/transformer \
    --input_image data/data/llm_1024/floor/some_image.png \
    --strengths 0.9 1.0 \
    --guidance_scale 4.5 \
    --num_inference_steps 50 \
    --output_dir ./outputs/test_img2img
"""

import argparse
import os

import torch
import torchvision.transforms.functional as TF
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    PixArtSigmaPipeline,
    PixArtTransformer2DModel,
)
from PIL import Image
from transformers import T5EncoderModel, T5Tokenizer


def parse_args():
    p = argparse.ArgumentParser(
        description="Test PixArt-Sigma img2img with varying strength (Flow Matching)",
    )
    p.add_argument("--ckpt", type=str, required=True,
                    help="Path to finetuned transformer directory")
    p.add_argument("--input_image", type=str, required=True,
                    help="Reference image path for img2img")
    p.add_argument("--base_model", type=str,
                    default="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS")
    p.add_argument("--weights_dir", type=str,
                    default="/home/daiqing_tan/stable_diffusion_lora/weights")
    p.add_argument("--output_dir", type=str, default="./outputs/test_img2img")
    p.add_argument("--prompt", type=str,
                    default="architectural floor plan, blueprint, technical drawing")
    p.add_argument("--negative_prompt", type=str, default="")
    p.add_argument("--strengths", type=float, nargs="+",
                    default=[0.3, 0.5, 0.7, 0.9, 1.0])
    p.add_argument("--guidance_scale", type=float, default=4.5)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_images", type=int, default=2)
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


def encode_image_to_latent(vae, image: Image.Image, device: torch.device) -> torch.Tensor:
    """PIL Image -> VAE latent (1, C, H/f, W/f), float32, scaled."""
    img_tensor = TF.to_tensor(image.convert("RGB")).unsqueeze(0)
    img_tensor = img_tensor.to(device=device, dtype=vae.dtype) * 2.0 - 1.0
    with torch.no_grad():
        z = vae.encode(img_tensor).latent_dist.sample()
    return (z * vae.config.scaling_factor).float()


def make_comparison(src: Image.Image, gen: Image.Image,
                    alpha: float = 0.45, gap: int = 8) -> Image.Image:
    """三栏对比图: [原图 | 叠加 | 生成图]"""
    w, h = gen.size
    ctrl = src.resize((w, h), Image.LANCZOS).convert("RGB")
    gen_rgb = gen.convert("RGB")
    blend = Image.blend(gen_rgb, ctrl, alpha=alpha)

    canvas = Image.new("RGB", (w * 3 + gap * 2, h), color=(20, 20, 20))
    canvas.paste(ctrl, (0, 0))
    canvas.paste(blend, (w + gap, 0))
    canvas.paste(gen_rgb, (w * 2 + gap * 2, 0))
    return canvas


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)
    dtype = torch.bfloat16
    res = args.resolution

    # ── 加载模型 ──
    print(f"Loading VAE from {args.base_model}...")
    vae = AutoencoderKL.from_pretrained(
        args.base_model, subfolder="vae",
        cache_dir=args.weights_dir, torch_dtype=dtype,
    ).to(device)

    print(f"Loading T5 from {args.base_model}...")
    tokenizer = T5Tokenizer.from_pretrained(
        args.base_model, subfolder="tokenizer", cache_dir=args.weights_dir,
    )
    text_encoder = T5EncoderModel.from_pretrained(
        args.base_model, subfolder="text_encoder",
        cache_dir=args.weights_dir, torch_dtype=dtype,
    ).to(device)

    print(f"Loading finetuned transformer from {args.ckpt}...")
    transformer = PixArtTransformer2DModel.from_pretrained(
        args.ckpt, torch_dtype=dtype,
    ).to(device)

    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000, shift=1.0,
    )
    if not hasattr(scheduler, "init_noise_sigma"):
        scheduler.init_noise_sigma = 1.0
    if not hasattr(scheduler, "scale_model_input"):
        scheduler.scale_model_input = lambda sample, *a, **kw: sample

    pipe = PixArtSigmaPipeline(
        vae=vae, transformer=transformer,
        text_encoder=text_encoder, tokenizer=tokenizer,
        scheduler=scheduler,
    )
    pipe.set_progress_bar_config(disable=False)

    # ── 加载参考图 ──
    src_image = Image.open(args.input_image).convert("RGB")
    src_image = src_image.resize((res, res), Image.LANCZOS)
    print(f"Reference image: {args.input_image} -> {res}x{res}")

    z_clean = encode_image_to_latent(vae, src_image, device)

    # ── 遍历 strength ──
    for strength in args.strengths:
        print(f"\n{'='*60}")
        print(f"  strength = {strength:.2f}")
        print(f"{'='*60}")

        scheduler.set_timesteps(args.num_inference_steps, device=device)
        all_timesteps = scheduler.timesteps

        start_idx = max(int(args.num_inference_steps * (1.0 - strength)), 1)
        start_idx = min(start_idx, len(all_timesteps) - 1)
        truncated_timesteps = all_timesteps[start_idx:].tolist()
        t_start = 1.0 - strength

        print(f"  t_start={t_start:.2f}, denoise steps={len(truncated_timesteps)}/{args.num_inference_steps}")

        for n in range(args.num_images):
            seed = args.seed + n
            gen = torch.Generator(device=device).manual_seed(seed)
            noise = torch.randn(z_clean.shape, generator=gen, device=device, dtype=z_clean.dtype)

            z_noisy = t_start * z_clean + (1.0 - t_start) * noise
            z_input = z_noisy.to(dtype=dtype)

            gen = torch.Generator(device=device).manual_seed(seed)
            output = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                guidance_scale=args.guidance_scale,
                num_images_per_prompt=1,
                generator=gen,
                latents=z_input,
                timesteps=truncated_timesteps,
            )

            img = output.images[0]
            tag = f"s{strength:.1f}_seed{seed}"
            img_path = os.path.join(args.output_dir, f"{tag}.png")
            img.save(img_path)
            print(f"  Saved: {img_path}")

            cmp_path = os.path.join(args.output_dir, f"{tag}_cmp.png")
            cmp = make_comparison(src_image, img)
            cmp.save(cmp_path)
            print(f"  Compare: {cmp_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
