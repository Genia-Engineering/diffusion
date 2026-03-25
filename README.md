# Stable Diffusion LoRA + ControlNet 训练框架

高性能训练框架，支持 **SD 1.5** 和 **SDXL** 的 LoRA 微调与 ControlNet 训练。

## 特性

- **LoRA 微调**: 原生实现 + PEFT 双模式，支持 UNet 和 Text Encoder 注入
- **ControlNet 训练**: 从预训练 UNet 初始化，含零卷积层
- **联合训练**: LoRA + ControlNet 模式 C（新风格 + 新控制）
- **性能优化**: bf16/fp16 混合精度、8-bit AdamW、xformers、梯度检查点
- **多 GPU**: 基于 accelerate 的分布式训练
- **Aspect Ratio Bucketing**: 动态宽高比分桶，避免 padding 浪费
- **TensorBoard**: 实时监控 loss/lr/生成图像

## 安装

```bash
pip install -r requirements.txt
```

## 数据准备

```
datasets/
├── train/
│   ├── image_001.png
│   ├── image_001.txt        # caption 文本文件，与图片同名
│   ├── image_002.jpg
│   ├── image_002.txt
│   └── ...
├── conditioning/             # ControlNet 条件图像（可选）
│   ├── image_001.png         # Canny/Depth/Pose 图，与训练图同名
│   └── ...
```

每张图像对应一个 `.txt` 文件存放 caption。如不提供条件图像目录，ControlNet 训练将在线提取 Canny 边缘图。

## 训练

### LoRA 微调

```bash
# SD 1.5 单卡训练
python scripts/train_lora.py --config configs/lora_sd15.yaml

# SDXL 多卡训练
accelerate launch --num_processes=4 scripts/train_lora.py \
  --config configs/lora_sdxl.yaml

# 命令行覆盖超参
accelerate launch scripts/train_lora.py \
  --config configs/lora_sd15.yaml \
  --override training.learning_rate=5e-5 training.train_batch_size=2
```

### ControlNet 训练

```bash
# SD 1.5
python scripts/train_controlnet.py --config configs/controlnet_sd15.yaml

# SDXL 多卡
accelerate launch --num_processes=4 scripts/train_controlnet.py \
  --config configs/controlnet_sdxl.yaml

# 联合 LoRA + ControlNet
accelerate launch scripts/train_controlnet.py \
  --config configs/controlnet_sd15.yaml \
  --override lora.rank=16 lora.alpha=16
```

### 恢复训练

```bash
# 从最新检查点恢复
python scripts/train_lora.py --config configs/lora_sd15.yaml --resume latest

# 从指定检查点恢复
python scripts/train_lora.py --config configs/lora_sd15.yaml \
  --resume outputs/lora_sd15/checkpoints/step_002000
```

## 超参数建议

针对 3000-4000 张图像的数据集：

| 任务 | 模型 | batch_size | grad_accum | effective_batch | lr | 总步数 | 约 epoch |
|------|------|-----------|-----------|----------------|---------|--------|---------|
| LoRA | SD1.5 | 4 | 2 | 8 | 1e-4 | 5000 | ~5 |
| LoRA | SDXL | 2 | 4 | 8 | 1e-4 | 10000 | ~5 |
| ControlNet | SD1.5 | 4 | 2 | 8 | 1e-5 | 10000 | ~10 |
| ControlNet | SDXL | 2 | 4 | 8 | 1e-5 | 16000 | ~8 |

## 输出目录结构

```
outputs/
├── tensorboard/              # TensorBoard 日志
├── samples/                  # 验证图像（每 500 step）
│   ├── step_000500/
│   │   ├── prompt_00_img_00.png
│   │   └── ...
│   └── step_001000/
└── checkpoints/              # 权重检查点（每 500 step，保留最近 3 份）
    ├── step_004000/
    │   ├── lora_unet.safetensors
    │   ├── optimizer.pt
    │   ├── lr_scheduler.pt
    │   └── training_state.json
    └── step_005000/
```

## 监控

```bash
tensorboard --logdir outputs/tensorboard --port 6006
```

可查看:
- `train/loss`: 训练损失曲线
- `train/learning_rate`: 学习率变化
- `train/grad_norm`: 梯度范数
- `validation/samples`: 每 500 step 的生成图像

## LoRA 权重使用

```python
from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe.load_lora_weights("outputs/lora_sd15/checkpoints/step_005000/lora_unet.safetensors")
image = pipe("your prompt").images[0]
```

## 项目结构

```
├── configs/          # YAML 配置文件
├── data/             # 数据加载与增强
├── models/           # LoRA 注入、ControlNet、模型加载
├── trainers/         # 训练循环（base + lora + controlnet）
├── utils/            # 日志、验证、检查点、显存优化
├── scripts/          # 训练入口脚本
└── outputs/          # 输出（图像、权重、日志）
```
