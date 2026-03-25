"""Validation Loop — 定期生成验证图像，保存到磁盘并记录到 TensorBoard。

FID 计算采用分布式策略：
  - 样本图（少量，用于可视化）：仅 Rank 0 生成并保存
  - FID 图（大量）：所有 Rank 均匀分配生成，各自提取 DINOv2 特征后
    通过 accelerator.gather 汇聚到 Rank 0 完成 FID 计算
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .fid import FIDCalculator, _compute_fid
from .logger import TensorBoardLogger

if TYPE_CHECKING:
    from accelerate import Accelerator

logger = logging.getLogger(__name__)


def _make_comparison_grid(
    predicted: Image.Image,
    conditioning: Image.Image | None = None,
    ground_truth: Image.Image | None = None,
    overlay_alpha: float = 0.4,
    label_height: int = 32,
) -> Image.Image:
    """将预测图、控制图、原图拼成对比图。

    布局: [控制图 | 预测+控制叠加 | 预测图 | 原图(可选)]
    - 预测+控制叠加：将控制图以 overlay_alpha 透明度叠加到预测图上，方便比对结构对齐
    """
    h, w = predicted.size[1], predicted.size[0]
    panels: list[tuple[str, Image.Image]] = []

    if conditioning is not None:
        cond_resized = conditioning.resize((w, h), Image.LANCZOS)
        panels.append(("Control", cond_resized))

        overlay = Image.blend(predicted, cond_resized.convert("RGB"), overlay_alpha)
        panels.append(("Pred+Ctrl", overlay))

    panels.append(("Predicted", predicted))

    if ground_truth is not None:
        gt_resized = ground_truth.resize((w, h), Image.LANCZOS)
        panels.append(("Ground Truth", gt_resized))

    total_w = w * len(panels)
    total_h = h + label_height
    grid = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for i, (label, img) in enumerate(panels):
        x_off = i * w
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        draw.text((x_off + (w - text_w) // 2, 4), label, fill=(0, 0, 0), font=font)
        grid.paste(img, (x_off, label_height))

    return grid


class ValidationLoop:
    """每隔 N 步生成验证图像，可选计算分布式 FID 分数。

    - 使用固定 seed 确保不同 step 的图像可横向对比
    - 样本图保存到磁盘和 TensorBoard（仅 Rank 0）
    - FID 图像生成分散到所有 Rank，特征 gather 后由 Rank 0 计算 FID
    - 推理完成后清理显存
    """

    def __init__(
        self,
        prompts: list[str],
        negative_prompt: str = "",
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: int = 42,
        num_images_per_prompt: int = 1,
        save_dir: str = "./outputs/samples",
        fid_calculator: Optional[FIDCalculator] = None,
        fid_num_gen_images: int = 256,
        fid_batch_size: int = 4,
        controlnet_conditioning_scale: float = 1.0,
        img2img_strengths: list[float] | None = None,
    ):
        self.prompts = prompts
        self.negative_prompt = negative_prompt
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.seed = seed
        self.num_images_per_prompt = num_images_per_prompt
        self.save_dir = Path(save_dir)
        self.fid_calculator = fid_calculator
        self.fid_num_gen_images = fid_num_gen_images
        self.fid_batch_size = fid_batch_size
        self.controlnet_conditioning_scale = controlnet_conditioning_scale
        # img2img_strengths: 列表中每个值生成一组结果，方便一次看到不同噪声程度下的表现
        #   None / []  = 不做 img2img（纯文生图）
        #   [0.3]      = 只生成轻度重建
        #   [0.3, 0.5, 0.7] = 三个 strength 各生成一组，保存为 s0.3_* / s0.5_* / s0.7_*
        self.img2img_strengths: list[float] = [s for s in (img2img_strengths or []) if s > 0]

    @torch.no_grad()
    def run(
        self,
        pipeline,
        step: int,
        tb_logger: TensorBoardLogger,
        device: torch.device = None,
        conditioning_images: list[Image.Image] = None,
        accelerator: Optional["Accelerator"] = None,
        pipeline_kwargs_override: Optional[dict] = None,
        ground_truth_images: list[Image.Image] = None,
        img2img_data: Optional[list[tuple[float, list, list]]] = None,
        img2img_scheduler=None,
    ) -> list[Image.Image]:
        """执行一次验证生成，并可选计算分布式 FID 分数。

        Args:
            pipeline: diffusers Pipeline（已加载权重）
            step: 当前训练步数
            tb_logger: TensorBoard 日志记录器
            device: 推理设备
            conditioning_images: ControlNet 条件图像（可选）
            accelerator: Accelerator 实例，用于多卡 gather；为 None 时退化为单卡模式
            pipeline_kwargs_override: 额外传给 pipeline 的参数字典（可选）。
                包含 "prompt_embeds" 时自动省略 prompt/negative_prompt 字符串，
                适用于文本编码器已卸载的情况。
            ground_truth_images: 验证用原始训练图像（可选），用于与生成结果对比
            img2img_data: img2img 模式数据列表，每个元素为 (strength, latents, timesteps)。
                每个 strength 独立生成一组图像，保存为 s{strength}_prompt_XX.png。

        Returns:
            生成的 PIL 样本图像列表（仅 Rank 0 有内容，其他 Rank 返回空列表）
        """
        is_main = accelerator is None or accelerator.is_main_process

        # ── 样本图：仅 Rank 0 生成，用于可视化 ────────────────────────────────
        all_images: list[Image.Image] = []
        comparison_grids: list[Image.Image] = []
        if is_main:
            logger.info(f"Running validation at step {step}...")
            step_dir = self.save_dir / f"step_{step:06d}"
            step_dir.mkdir(parents=True, exist_ok=True)

            generator = torch.Generator(device=device or "cpu").manual_seed(self.seed)

            use_embeds = (
                pipeline_kwargs_override is not None
                and (
                    (isinstance(pipeline_kwargs_override, dict) and "prompt_embeds" in pipeline_kwargs_override)
                    or isinstance(pipeline_kwargs_override, list)
                )
            )

            for i, prompt in enumerate(self.prompts):
                kwargs = {
                    "num_inference_steps": self.num_inference_steps,
                    "guidance_scale": self.guidance_scale,
                    "generator": generator,
                    "num_images_per_prompt": self.num_images_per_prompt,
                }
                if use_embeds:
                    kwargs["prompt"] = None
                    kwargs["negative_prompt"] = None
                    if isinstance(pipeline_kwargs_override, list):
                        kwargs.update(pipeline_kwargs_override[i % len(pipeline_kwargs_override)])
                    else:
                        for k, v in pipeline_kwargs_override.items():
                            if isinstance(v, torch.Tensor) and v.shape[0] > 1:
                                kwargs[k] = v[i % v.shape[0] : i % v.shape[0] + 1]
                            else:
                                kwargs[k] = v
                else:
                    kwargs["prompt"] = prompt
                    kwargs["negative_prompt"] = self.negative_prompt
                    if pipeline_kwargs_override:
                        kwargs.update(pipeline_kwargs_override)

                if conditioning_images is not None and i < len(conditioning_images):
                    kwargs["image"] = conditioning_images[i]
                    kwargs["controlnet_conditioning_scale"] = self.controlnet_conditioning_scale

                output = pipeline(**kwargs)
                for j, img in enumerate(output.images):
                    img.save(step_dir / f"prompt_{i:02d}_img_{j:02d}.png")
                    all_images.append(img)

                    cond_img = conditioning_images[i] if conditioning_images and i < len(conditioning_images) else None
                    gt_img = ground_truth_images[i] if ground_truth_images and i < len(ground_truth_images) else None

                    if cond_img is not None or gt_img is not None:
                        grid = _make_comparison_grid(img, cond_img, gt_img)
                        grid.save(step_dir / f"compare_{i:02d}_img_{j:02d}.png")
                        comparison_grids.append(grid)

            # ── img2img：每个 strength 独立生成一组，保存为 s{strength}_* ──────
            if img2img_data:
                # EulerDiscrete 支持 timesteps=[...] 自定义起始点；t2i 完成后才切换，
                # 避免污染 t2i / FID 生成（t2i 用 DPMSolverSDE，img2img 用 Euler）
                _saved_scheduler = pipeline.scheduler
                if img2img_scheduler is not None:
                    pipeline.scheduler = img2img_scheduler
                all_img2img: list[Image.Image] = []
                try:
                    for strength, i2i_latents, i2i_timesteps in img2img_data:
                        s_tag = f"s{strength:.1f}"
                        for i, prompt in enumerate(self.prompts):
                            i2i_kwargs = {
                                "guidance_scale": self.guidance_scale,
                                "generator": torch.Generator(device=device or "cpu").manual_seed(self.seed + i),
                                # img2img 传入自定义 latents（batch=1），必须用 num_images_per_prompt=1
                                # 否则 text embeds 被重复 4× 后与 latents batch 不匹配，
                                # 导致 Transformer cross-attention 内部 reshape 出错
                                "num_images_per_prompt": 1,
                                "timesteps": i2i_timesteps,
                                "latents": i2i_latents[i] if i < len(i2i_latents) else None,
                            }
                            if use_embeds:
                                i2i_kwargs["prompt"] = None
                                i2i_kwargs["negative_prompt"] = None
                                if isinstance(pipeline_kwargs_override, list):
                                    i2i_kwargs.update(pipeline_kwargs_override[i % len(pipeline_kwargs_override)])
                                else:
                                    for k, v in pipeline_kwargs_override.items():
                                        if isinstance(v, torch.Tensor) and v.shape[0] > 1:
                                            i2i_kwargs[k] = v[i % v.shape[0] : i % v.shape[0] + 1]
                                        else:
                                            i2i_kwargs[k] = v
                            else:
                                i2i_kwargs["prompt"] = prompt
                                i2i_kwargs["negative_prompt"] = self.negative_prompt
                                if pipeline_kwargs_override:
                                    i2i_kwargs.update(pipeline_kwargs_override)

                            out = pipeline(**i2i_kwargs)
                            for j, img in enumerate(out.images):
                                img.save(step_dir / f"{s_tag}_prompt_{i:02d}_img_{j:02d}.png")
                                all_img2img.append(img)

                                gt_img = ground_truth_images[i] if ground_truth_images and i < len(ground_truth_images) else None
                                if gt_img is not None:
                                    grid = _make_comparison_grid(img, None, gt_img)
                                    grid.save(step_dir / f"{s_tag}_compare_{i:02d}_img_{j:02d}.png")
                                    comparison_grids.append(grid)
                finally:
                    pipeline.scheduler = _saved_scheduler

                if all_img2img:
                    tb_logger.log_images("validation/img2img", all_img2img, step)

            tb_logger.log_images("validation/samples", all_images, step)
            if comparison_grids:
                tb_logger.log_images("validation/comparisons", comparison_grids, step)
            tb_logger.flush()
            logger.info(f"Validation complete: {len(all_images)} images saved to {step_dir}")
        # ────────────────────────────────────────────────────────────────────

        # ── 分布式 FID 计算 ───────────────────────────────────────────────────
        if self.fid_calculator is not None:
            self._run_fid_distributed(
                pipeline, step, tb_logger, device, accelerator, is_main,
                pipeline_kwargs_override=pipeline_kwargs_override,
            )
        # ────────────────────────────────────────────────────────────────────

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return all_images

    @torch.no_grad()
    def _run_fid_distributed(
        self,
        pipeline,
        step: int,
        tb_logger: TensorBoardLogger,
        device: torch.device,
        accelerator: Optional["Accelerator"],
        is_main: bool,
        pipeline_kwargs_override: Optional[dict] = None,
    ) -> None:
        """分布式 FID 计算：所有 Rank 均匀生成图像并提取特征，Rank 0 汇聚后计算 FID。

        流程：
          1. 每个 Rank 生成 fid_num_gen_images // num_processes 张图（使用不同 seed）
          2. 各 Rank 在本地用 DINOv2 提取特征，得到 (N/P, D) 的 Tensor
          3. accelerator.gather 将所有 Rank 的特征汇聚为 (N, D)
          4. 仅 Rank 0 根据真实图像统计量计算并记录 FID
        """
        if accelerator is not None and accelerator.num_processes > 1:
            num_processes = accelerator.num_processes
            process_index = accelerator.process_index
        else:
            num_processes = 1
            process_index = 0

        # 每卡生成的图像数（整除；余数部分忽略，对 FID 估计影响极小）
        num_gen_this_rank = self.fid_num_gen_images // num_processes
        seed_offset = process_index * num_gen_this_rank

        logger.info(
            f"[Rank {process_index}/{num_processes}] "
            f"Generating {num_gen_this_rank} FID images (seed_offset={seed_offset})..."
        )
        fid_images = self._generate_fid_images(
            pipeline, device, num_gen_this_rank, seed_offset,
            pipeline_kwargs_override=pipeline_kwargs_override,
        )

        # 本地提取 DINOv2 特征
        features_np = self.fid_calculator._extract_features(fid_images)          # (N/P, D)
        features_tensor = torch.from_numpy(features_np).to(device)

        # 跨卡 gather 特征
        if accelerator is not None and accelerator.num_processes > 1:
            gathered = accelerator.gather(features_tensor)  # (N, D)，所有 Rank 均可见
        else:
            gathered = features_tensor

        # 仅 Rank 0 计算 FID（需要真实图像统计量，只有 Rank 0 有）
        if is_main and self.fid_calculator.is_ready():
            try:
                gathered_np = gathered.cpu().float().numpy()  # (N, D)
                gen_mu, gen_sigma = FIDCalculator._calc_stats(gathered_np)
                fid_score = _compute_fid(
                    self.fid_calculator._real_mu,
                    self.fid_calculator._real_sigma,
                    gen_mu,
                    gen_sigma,
                )
                tb_logger.log_fid(fid_score, step)
                logger.info(
                    f"Step {step} — FID: {fid_score:.4f} "
                    f"(n_gen={gathered_np.shape[0]}, distributed across {num_processes} ranks)"
                )
            except Exception as e:
                logger.warning(f"FID 计算失败 (step={step}): {e}")

    @torch.no_grad()
    def _generate_fid_images(
        self,
        pipeline,
        device: torch.device,
        num_images: int,
        seed_offset: int = 0,
        pipeline_kwargs_override: Optional[dict] = None,
    ) -> list[Image.Image]:
        """为 FID 批量生成指定数量的图像。

        每次调用 pipeline 时传入一组 prompt 列表（长度为 fid_batch_size），
        并为每张图配置独立的 Generator，确保与逐张生成时的随机性完全一致。

        Args:
            num_images: 本 Rank 需要生成的图像数量
            seed_offset: seed 偏移量，确保不同 Rank 生成不同图像
            pipeline_kwargs_override: 文本编码器已卸载时传入预缓存嵌入，
                含 "prompt_embeds" 时自动省略 prompt/negative_prompt 字符串
        """
        fid_images: list[Image.Image] = []
        prompt_cycle = self.prompts * (num_images // len(self.prompts) + 1)
        use_embeds = pipeline_kwargs_override and "prompt_embeds" in pipeline_kwargs_override
        gen_device = device or "cpu"

        for batch_start in range(0, num_images, self.fid_batch_size):
            batch_indices = list(range(batch_start, min(batch_start + self.fid_batch_size, num_images)))
            actual_bs = len(batch_indices)

            # 每张图配置独立 Generator，保持与逐张生成时完全相同的随机性
            generators = [
                torch.Generator(device=gen_device).manual_seed(
                    self.seed + seed_offset + idx + 10000
                )
                for idx in batch_indices
            ]

            kwargs: dict = {
                "num_inference_steps": self.num_inference_steps,
                "guidance_scale": self.guidance_scale,
                "generator": generators,
                "num_images_per_prompt": 1,
            }

            if use_embeds:
                kwargs["prompt"] = None
                kwargs["negative_prompt"] = None
                for k, v in pipeline_kwargs_override.items():
                    if isinstance(v, torch.Tensor) and v.shape[0] != actual_bs:
                        repeats = (actual_bs + v.shape[0] - 1) // v.shape[0]
                        kwargs[k] = v.repeat(repeats, *([1] * (v.dim() - 1)))[:actual_bs]
                    else:
                        kwargs[k] = v
            else:
                kwargs["prompt"] = [prompt_cycle[idx % len(prompt_cycle)] for idx in batch_indices]
                kwargs["negative_prompt"] = self.negative_prompt
                if pipeline_kwargs_override:
                    kwargs.update(pipeline_kwargs_override)

            output = pipeline(**kwargs)
            fid_images.extend(output.images)

        logger.info(f"FID 生成完毕: {len(fid_images)} 张图像 (batch_size={self.fid_batch_size})")
        return fid_images
