"""快速测试不同 guidance_scale 下的 t2i 结果（含无 CFG 对比）。

用法:
  python scripts/test_no_cfg.py \
    --ckpt outputs/pixart_sigma_floorplan/checkpoints/step_001800/transformer \
    --output_dir /tmp/test_no_cfg_results \
    --guidance_scales 1.0 2.0 4.5 \
    --prompt "architectural floor plan, blueprint, technical drawing"
"""

import argparse
import os

import torch
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    PixArtSigmaPipeline,
    PixArtTransformer2DModel,
)
from transformers import T5EncoderModel, T5Tokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Test PixArt-Sigma t2i with varying guidance_scale")
    p.add_argument("--ckpt", type=str, required=True,
                    help="Path to finetuned transformer directory")
    p.add_argument("--base_model", type=str, default="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS")
    p.add_argument("--weights_dir", type=str, default="/home/daiqing_tan/stable_diffusion_lora/weights")
    p.add_argument("--output_dir", type=str, default="./outputs/test_no_cfg")
    p.add_argument("--prompt", type=str, default="architectural floor plan, blueprint, technical drawing")
    p.add_argument("--negative_prompt", type=str, default="")
    p.add_argument("--guidance_scales", type=float, nargs="+", default=[1.0, 2.0, 4.5])
    p.add_argument("--num_inference_steps", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_images", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)
    dtype = torch.bfloat16

    print(f"Loading VAE from {args.base_model}...")
    vae = AutoencoderKL.from_pretrained(
        args.base_model, subfolder="vae", cache_dir=args.weights_dir, torch_dtype=dtype,
    ).to(device)

    print(f"Loading T5 from {args.base_model}...")
    tokenizer = T5Tokenizer.from_pretrained(
        args.base_model, subfolder="tokenizer", cache_dir=args.weights_dir,
    )
    text_encoder = T5EncoderModel.from_pretrained(
        args.base_model, subfolder="text_encoder", cache_dir=args.weights_dir, torch_dtype=dtype,
    ).to(device)

    print(f"Loading finetuned transformer from {args.ckpt}...")
    transformer = PixArtTransformer2DModel.from_pretrained(args.ckpt, torch_dtype=dtype).to(device)

    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0)
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

    for gs in args.guidance_scales:
        print(f"\n{'='*50}")
        print(f"guidance_scale = {gs}")
        print(f"{'='*50}")
        gen = torch.Generator(device=device).manual_seed(args.seed)
        images = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=gs,
            generator=gen,
            num_images_per_prompt=args.num_images,
        ).images
        for i, img in enumerate(images):
            fname = os.path.join(args.output_dir, f"gs_{gs:.1f}_img_{i:02d}.png")
            img.save(fname)
            print(f"  Saved: {fname}")

    print("\nDone!")


if __name__ == "__main__":
    main()
