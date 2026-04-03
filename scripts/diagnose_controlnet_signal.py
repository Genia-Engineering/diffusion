"""诊断 SDXL ControlNet 主分支 vs 侧分支信号强度。

在 UNet forward 中 ControlNet 残差与 UNet skip-connection 相加之前，
捕获两个分支的 max / min / mean / std / norm，
用于判断 ControlNet 侧分支信号是否被 UNet 主分支淹没。

用法:
  conda activate sd_lora
  python scripts/diagnose_controlnet_signal.py \
      --config configs/controlnet_sdxl.yaml \
      [--checkpoint outputs/controlnet_sdxl/checkpoints/step_017000] \
      [--num_timesteps 5]

也支持 ControlNet-XS（自动检测配置）。
"""

import argparse
import os
import sys
from functools import wraps
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.model_loader import load_sdxl_components, resolve_model_path


def fmt(t: torch.Tensor) -> str:
    """格式化 tensor 统计量。"""
    return (
        f"shape={list(t.shape):>20s}  "
        f"min={t.min().item():+.6f}  max={t.max().item():+.6f}  "
        f"mean={t.mean().item():+.6f}  std={t.std().item():.6f}  "
        f"norm={t.norm().item():.4f}  absmax={t.abs().max().item():.6f}"
    )


# ── 标准 ControlNet 诊断 ──────────────────────────────────────────────


def diagnose_standard_controlnet(config, args):
    """标准 ControlNet: 两次前向 (ControlNet → UNet)，残差通过参数传入。"""
    from diffusers import ControlNetModel, UNet2DConditionModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    model_cfg = config.model
    weights_dir = model_cfg.get("weights_dir", None)
    merged_unet_path = model_cfg.get("merged_unet_path", None)
    resolved = resolve_model_path(model_cfg.pretrained_model_name_or_path, weights_dir, "sdxl")

    # 加载 UNet
    if merged_unet_path and os.path.isdir(merged_unet_path):
        print(f"[INFO] 从 merged UNet 加载: {merged_unet_path}")
        unet = UNet2DConditionModel.from_pretrained(merged_unet_path, subfolder="unet", torch_dtype=dtype)
    else:
        print(f"[INFO] 从预训练模型加载 UNet: {resolved}")
        unet = UNet2DConditionModel.from_pretrained(resolved, subfolder="unet", torch_dtype=dtype)

    # 加载 ControlNet
    ckpt_dir = args.checkpoint
    if ckpt_dir and os.path.isdir(ckpt_dir):
        cn_path = os.path.join(ckpt_dir, "controlnet")
        if os.path.isdir(cn_path):
            print(f"[INFO] 从 checkpoint 加载 ControlNet: {cn_path}")
            controlnet = ControlNetModel.from_pretrained(cn_path, torch_dtype=dtype)
        else:
            print(f"[WARN] checkpoint 中未找到 controlnet/ 子目录，从 UNet 初始化")
            controlnet = ControlNetModel.from_unet(unet, conditioning_channels=3)
    else:
        print("[INFO] 未指定 checkpoint，从 UNet 初始化 ControlNet（零初始化状态）")
        controlnet = ControlNetModel.from_unet(unet, conditioning_channels=3)

    unet.to(device, dtype=dtype).eval()
    controlnet.to(device, dtype=dtype).eval()

    # 加载文本编码器获取 prompt embeds
    components = load_sdxl_components(resolved, weights_dir=weights_dir, dtype=dtype, skip_unet=True)
    tokenizer = components["tokenizer"]
    tokenizer_2 = components["tokenizer_2"]
    text_encoder = components["text_encoder"].to(device, dtype=dtype)
    text_encoder_2 = components["text_encoder_2"].to(device, dtype=dtype)
    noise_scheduler = components["noise_scheduler"]

    caption = config.data.get("caption", "architectural floor plan, blueprint, technical drawing")

    with torch.no_grad():
        tokens_1 = tokenizer(caption, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        tokens_2 = tokenizer_2(caption, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        enc_out_1 = text_encoder(tokens_1.input_ids.to(device), output_hidden_states=True)
        enc_out_2 = text_encoder_2(tokens_2.input_ids.to(device), output_hidden_states=True)
        prompt_embeds = torch.cat([enc_out_1.hidden_states[-2], enc_out_2.hidden_states[-2]], dim=-1)
        pooled = enc_out_2[0]

    del text_encoder, text_encoder_2, tokenizer, tokenizer_2
    torch.cuda.empty_cache()

    resolution = config.data.get("resolution", 1024)
    latent_h, latent_w = resolution // 8, resolution // 8

    add_time_ids = torch.tensor([[resolution, resolution, 0, 0, resolution, resolution]], device=device, dtype=dtype)
    added_cond_kwargs = {"text_embeds": pooled, "time_ids": add_time_ids}

    # 加载一张真实的条件图（如果有），否则用随机图
    cond_dir = config.data.get("conditioning_data_dir", "")
    cond_img = None
    if cond_dir and os.path.isdir(cond_dir):
        from PIL import Image
        from torchvision import transforms
        imgs = [f for f in os.listdir(cond_dir) if f.endswith(".png")]
        if imgs:
            img_path = os.path.join(cond_dir, imgs[0])
            print(f"[INFO] 使用真实条件图: {imgs[0]}")
            img = Image.open(img_path).convert("RGB").resize((resolution, resolution))
            cond_img = transforms.ToTensor()(img).unsqueeze(0).to(device, dtype=dtype)

    if cond_img is None:
        print("[INFO] 使用随机条件图")
        cond_img = torch.randn(1, 3, resolution, resolution, device=device, dtype=dtype)

    # 在不同 timestep 下测试
    timestep_list = args.timesteps
    if not timestep_list:
        T = noise_scheduler.config.num_train_timesteps
        timestep_list = [int(T * r) for r in [0.05, 0.2, 0.5, 0.8, 0.95]]

    print(f"\n{'='*100}")
    print(f"  SDXL 标准 ControlNet 主分支 vs 侧分支信号诊断")
    print(f"  Resolution: {resolution}, Latent: {latent_h}x{latent_w}")
    print(f"  Checkpoint: {args.checkpoint or '(零初始化)'}")
    print(f"{'='*100}\n")

    for t_val in timestep_list:
        t = torch.tensor([t_val], device=device, dtype=torch.long)
        latents = torch.randn(1, 4, latent_h, latent_w, device=device, dtype=dtype)
        noise = torch.randn_like(latents)
        noisy_latents = noise_scheduler.add_noise(latents, noise, t)

        with torch.no_grad():
            # ControlNet 前向 → 得到侧分支残差
            cn_down, cn_mid = controlnet(
                noisy_latents, t, prompt_embeds,
                controlnet_cond=cond_img,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )

        # Patch UNet forward 以捕获主分支 skip-connections
        captured_main = {}
        captured_side = {}

        original_forward = unet.forward

        @wraps(original_forward)
        def patched_forward(*a, **kw):
            down_additional = kw.get("down_block_additional_residuals", None)
            mid_additional = kw.get("mid_block_additional_residual", None)

            # 暂时移除 ControlNet 残差，让 UNet 正常跑，手动在内部捕获
            kw["down_block_additional_residuals"] = None
            kw["mid_block_additional_residual"] = None

            # 直接跑原始 forward 获取无 ControlNet 的输出不可行（UNet内部已融合）
            # 改为：重新跑一遍 UNet 的 down blocks 和 mid block 来获取主分支值

            # 手动执行 UNet 前半段以获取 skip connections
            sample = a[0] if len(a) > 0 else kw.get("sample")
            timestep = a[1] if len(a) > 1 else kw.get("timestep")
            encoder_hidden_states = a[2] if len(a) > 2 else kw.get("encoder_hidden_states")
            acond = kw.get("added_cond_kwargs", None)

            # 恢复正常参数调用原始 forward（带 ControlNet 残差）
            kw["down_block_additional_residuals"] = down_additional
            kw["mid_block_additional_residual"] = mid_additional
            return original_forward(*a, **kw)

        # 更简单的方法：直接对比 ControlNet 输出和 UNet skip-connections
        # 通过 hook 捕获 UNet down blocks 输出
        hooks = []
        main_outputs = []

        def make_hook(idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden, res_samples = output
                    for j, r in enumerate(res_samples):
                        main_outputs.append((f"down_{idx}_res_{j}", r.detach().clone()))
                else:
                    main_outputs.append((f"down_{idx}", output.detach().clone()))
            return hook_fn

        for i, block in enumerate(unet.down_blocks):
            hooks.append(block.register_forward_hook(make_hook(i)))

        mid_output = []
        def mid_hook(module, input, output):
            mid_output.append(("mid_block", output.detach().clone()))
        hooks.append(unet.mid_block.register_forward_hook(mid_hook))

        with torch.no_grad():
            # 不带 ControlNet 残差跑 UNet，获取纯主分支值
            main_outputs.clear()
            mid_output.clear()
            _ = unet(
                noisy_latents, t, prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                down_block_additional_residuals=None,
                mid_block_additional_residual=None,
            )

        for h in hooks:
            h.remove()

        # 打印统计
        print(f"\n{'─'*100}")
        print(f"  Timestep t={t_val}  (t/T = {t_val/noise_scheduler.config.num_train_timesteps:.2f})")
        print(f"{'─'*100}")

        # Down blocks: 主分支 skip connections vs ControlNet 侧分支残差
        # UNet down blocks 输出是 (hidden_state, (res1, res2, ...))
        # 收集所有 skip connections
        all_main_res = []
        for name, tensor in main_outputs:
            all_main_res.append((name, tensor))

        print(f"\n  {'位置':<25s} | {'主分支(UNet skip)':<75s} | {'侧分支(ControlNet)':<75s} | 比值(侧/主)")
        print(f"  {'─'*25} | {'─'*75} | {'─'*75} | {'─'*12}")

        # ControlNet down_block_res_samples 与 UNet skip connections 一一对应
        cn_idx = 0
        for name, main_t in all_main_res:
            if cn_idx < len(cn_down):
                side_t = cn_down[cn_idx]
                if main_t.shape == side_t.shape:
                    main_norm = main_t.norm().item()
                    side_norm = side_t.norm().item()
                    ratio = side_norm / max(main_norm, 1e-10)
                    print(
                        f"  {name:<25s} | "
                        f"mean={main_t.mean().item():+.5f} std={main_t.std().item():.5f} absmax={main_t.abs().max().item():.5f} norm={main_norm:.3f} | "
                        f"mean={side_t.mean().item():+.5f} std={side_t.std().item():.5f} absmax={side_t.abs().max().item():.5f} norm={side_norm:.3f} | "
                        f"{ratio:.6f}"
                    )
                    cn_idx += 1
                else:
                    cn_idx += 1

        # Mid block
        if mid_output and cn_mid is not None:
            main_m = mid_output[0][1]
            main_norm = main_m.norm().item()
            side_norm = cn_mid.norm().item()
            ratio = side_norm / max(main_norm, 1e-10)
            print(
                f"  {'mid_block':<25s} | "
                f"mean={main_m.mean().item():+.5f} std={main_m.std().item():.5f} absmax={main_m.abs().max().item():.5f} norm={main_norm:.3f} | "
                f"mean={cn_mid.mean().item():+.5f} std={cn_mid.std().item():.5f} absmax={cn_mid.abs().max().item():.5f} norm={side_norm:.3f} | "
                f"{ratio:.6f}"
            )

    print(f"\n{'='*100}")
    print("  诊断说明：")
    print("  - 比值(侧/主) ≈ 0：ControlNet 信号几乎为零（零初始化未学到东西）")
    print("  - 比值(侧/主) < 0.01：ControlNet 信号极弱，被 UNet 主分支淹没")
    print("  - 比值(侧/主) ∈ [0.01, 0.1]：合理的早期训练状态")
    print("  - 比值(侧/主) ∈ [0.1, 1.0]：ControlNet 有效控制范围")
    print("  - 比值(侧/主) > 1.0：ControlNet 信号过强，可能导致伪影")
    print(f"{'='*100}\n")


# ── ControlNet-XS 诊断 ────────────────────────────────────────────────


def diagnose_controlnet_xs(config, args):
    """ControlNet-XS: 融合模型，通过 hook 捕获 base↔ctrl 投射前后的信号。"""
    from diffusers import UNet2DConditionModel
    from models.controlnet_xs import create_controlnet_xs_sdxl, strip_ctrl_cross_attention

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    model_cfg = config.model
    weights_dir = model_cfg.get("weights_dir", None)
    merged_unet_path = model_cfg.get("merged_unet_path", None)
    resolved = resolve_model_path(model_cfg.pretrained_model_name_or_path, weights_dir, "sdxl")
    xs_cfg = config.get("controlnet_xs", {})

    # 加载 merged UNet 作为初始化源
    source_unet = None
    if merged_unet_path and os.path.isdir(merged_unet_path):
        print(f"[INFO] 从 merged UNet 初始化: {merged_unet_path}")
        source_unet = UNet2DConditionModel.from_pretrained(merged_unet_path, subfolder="unet", torch_dtype=dtype)

    adapter, unet_xs = create_controlnet_xs_sdxl(
        resolved, conditioning_channels=3,
        size_ratio=xs_cfg.get("size_ratio", 0.1),
        unet=source_unet, dtype=dtype,
    )

    if xs_cfg.get("no_ctrl_cross_attention", False):
        removed = strip_ctrl_cross_attention(unet_xs)
        print(f"[INFO] 移除 ctrl cross-attention: {removed/1e6:.1f}M params")

    # 加载 checkpoint
    ckpt_dir = args.checkpoint
    if ckpt_dir and os.path.isdir(ckpt_dir):
        cn_path = os.path.join(ckpt_dir, "controlnet")
        if os.path.isdir(cn_path):
            from diffusers.models.controlnets.controlnet_xs import ControlNetXSAdapter
            print(f"[INFO] 从 checkpoint 加载 adapter: {cn_path}")
            loaded = ControlNetXSAdapter.from_pretrained(cn_path, torch_dtype=dtype)
            adapter.load_state_dict(loaded.state_dict(), strict=False)
            from diffusers.models.controlnets.controlnet_xs import UNetControlNetXSModel
            unet_xs = UNetControlNetXSModel.from_unet(source_unet or UNet2DConditionModel.from_pretrained(
                resolved, subfolder="unet", torch_dtype=dtype,
            ), controlnet=adapter)
            if xs_cfg.get("no_ctrl_cross_attention", False):
                strip_ctrl_cross_attention(unet_xs)

    unet_xs.to(device, dtype=dtype).eval()

    # 文本编码器
    components = load_sdxl_components(resolved, weights_dir=weights_dir, dtype=dtype, skip_unet=True)
    tokenizer = components["tokenizer"]
    tokenizer_2 = components["tokenizer_2"]
    text_encoder = components["text_encoder"].to(device, dtype=dtype)
    text_encoder_2 = components["text_encoder_2"].to(device, dtype=dtype)
    noise_scheduler = components["noise_scheduler"]

    caption = config.data.get("caption", "architectural floor plan, blueprint, technical drawing")
    with torch.no_grad():
        t1 = tokenizer(caption, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        t2 = tokenizer_2(caption, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        e1 = text_encoder(t1.input_ids.to(device), output_hidden_states=True)
        e2 = text_encoder_2(t2.input_ids.to(device), output_hidden_states=True)
        prompt_embeds = torch.cat([e1.hidden_states[-2], e2.hidden_states[-2]], dim=-1)
        pooled = e2[0]

    del text_encoder, text_encoder_2
    torch.cuda.empty_cache()

    resolution = config.data.get("resolution", 1024)
    latent_h, latent_w = resolution // 8, resolution // 8
    add_time_ids = torch.tensor([[resolution, resolution, 0, 0, resolution, resolution]], device=device, dtype=dtype)
    added_cond_kwargs = {"text_embeds": pooled, "time_ids": add_time_ids}

    cond_dir = config.data.get("conditioning_data_dir", "")
    cond_img = None
    if cond_dir and os.path.isdir(cond_dir):
        from PIL import Image
        from torchvision import transforms
        imgs = [f for f in os.listdir(cond_dir) if f.endswith(".png")]
        if imgs:
            img_path = os.path.join(cond_dir, imgs[0])
            print(f"[INFO] 使用真实条件图: {imgs[0]}")
            img = Image.open(img_path).convert("RGB").resize((resolution, resolution))
            cond_img = transforms.ToTensor()(img).unsqueeze(0).to(device, dtype=dtype)
    if cond_img is None:
        cond_img = torch.randn(1, 3, resolution, resolution, device=device, dtype=dtype)

    timestep_list = args.timesteps or [50, 200, 500, 800, 950]

    # Hook 到 down_blocks 的 base_to_ctrl 和 ctrl_to_base 投射
    print(f"\n{'='*100}")
    print(f"  SDXL ControlNet-XS 主分支(base) vs 侧分支(ctrl) 信号诊断")
    print(f"  Checkpoint: {args.checkpoint or '(零初始化)'}")
    print(f"{'='*100}")

    hook_data = []
    hooks = []

    for i, block in enumerate(unet_xs.down_blocks):
        # 捕获 ctrl_to_base 零卷积输出（= 侧分支对主分支的注入量）
        c2b_convs = getattr(block, "ctrl_to_base", None)
        if c2b_convs is not None:
            if hasattr(c2b_convs, '__iter__'):
                for j, conv in enumerate(c2b_convs):
                    if conv is not None:
                        def make_c2b_hook(bi, ci):
                            def fn(m, inp, out):
                                hook_data.append({
                                    "loc": f"down_{bi}_c2b_{ci}",
                                    "input": inp[0].detach().clone() if isinstance(inp, tuple) else inp.detach().clone(),
                                    "output": out.detach().clone(),
                                })
                            return fn
                        hooks.append(conv.register_forward_hook(make_c2b_hook(i, j)))
            elif isinstance(c2b_convs, torch.nn.Module):
                def make_c2b_hook_single(bi):
                    def fn(m, inp, out):
                        hook_data.append({
                            "loc": f"down_{bi}_c2b",
                            "input": inp[0].detach().clone() if isinstance(inp, tuple) else inp.detach().clone(),
                            "output": out.detach().clone(),
                        })
                    return fn
                hooks.append(c2b_convs.register_forward_hook(make_c2b_hook_single(i)))

        # 捕获 base_to_ctrl 投射
        b2c_convs = getattr(block, "base_to_ctrl", None)
        if b2c_convs is not None:
            if hasattr(b2c_convs, '__iter__'):
                for j, conv in enumerate(b2c_convs):
                    if conv is not None:
                        def make_b2c_hook(bi, ci):
                            def fn(m, inp, out):
                                hook_data.append({
                                    "loc": f"down_{bi}_b2c_{ci}",
                                    "input": inp[0].detach().clone() if isinstance(inp, tuple) else inp.detach().clone(),
                                    "output": out.detach().clone(),
                                })
                            return fn
                        hooks.append(conv.register_forward_hook(make_b2c_hook(i, j)))

    for t_val in timestep_list:
        t = torch.tensor([t_val], device=device, dtype=torch.long)
        latents = torch.randn(1, 4, latent_h, latent_w, device=device, dtype=dtype)
        noise = torch.randn_like(latents)
        noisy_latents = noise_scheduler.add_noise(latents, noise, t)

        hook_data.clear()
        with torch.no_grad():
            _ = unet_xs(
                sample=noisy_latents, timestep=t,
                encoder_hidden_states=prompt_embeds,
                controlnet_cond=cond_img,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=True,
            )

        print(f"\n{'─'*100}")
        print(f"  Timestep t={t_val}")
        print(f"{'─'*100}")
        print(f"  {'位置':<25s} | {'输入(ctrl/base hidden)':<60s} | {'输出(投射后)':<60s} | 衰减比")
        print(f"  {'─'*25} | {'─'*60} | {'─'*60} | {'─'*10}")

        for d in hook_data:
            inp = d["input"]
            out = d["output"]
            inp_norm = inp.norm().item()
            out_norm = out.norm().item()
            ratio = out_norm / max(inp_norm, 1e-10)
            print(
                f"  {d['loc']:<25s} | "
                f"mean={inp.mean().item():+.5f} std={inp.std().item():.5f} norm={inp_norm:.3f} | "
                f"mean={out.mean().item():+.5f} std={out.std().item():.5f} norm={out_norm:.3f} | "
                f"{ratio:.6f}"
            )

    for h in hooks:
        h.remove()

    print(f"\n{'='*100}")
    print("  诊断说明 (ControlNet-XS)：")
    print("  - ctrl_to_base (c2b): 零初始化卷积，投射 ctrl 分支特征到 base 分支")
    print("    衰减比 ≈ 0 表示零卷积未学到任何东西；> 0.01 表示开始有效注入")
    print("  - base_to_ctrl (b2c): base 分支信息注入 ctrl 分支的通道")
    print("    这个方向的信号通常较强（非零初始化）")
    print(f"{'='*100}\n")


# ── 入口 ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="诊断 ControlNet 主/侧分支信号强度")
    parser.add_argument("--config", required=True, help="训练配置文件路径")
    parser.add_argument("--checkpoint", default=None, help="ControlNet checkpoint 目录")
    parser.add_argument("--timesteps", type=int, nargs="+", default=None,
                        help="要测试的 timestep 列表 (默认: 50 200 500 800 950)")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    is_xs = "controlnet_xs" in config
    if is_xs:
        print("[INFO] 检测到 controlnet_xs 配置，使用 ControlNet-XS 诊断模式")
        diagnose_controlnet_xs(config, args)
    else:
        print("[INFO] 使用标准 ControlNet 诊断模式")
        diagnose_standard_controlnet(config, args)


if __name__ == "__main__":
    main()
