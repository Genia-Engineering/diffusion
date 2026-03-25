#!/bin/bash
# =============================================================
# 统一训练启动脚本 — 支持 LoRA / ControlNet / PixArt-Sigma × SD1.5 / SDXL
#
# 用法（参数顺序不限）:
#   bash train.sh lora sdxl                              # LoRA SDXL（默认配置）
#   bash train.sh lora sd15                              # LoRA SD1.5（默认配置）
#   bash train.sh controlnet sdxl                        # ControlNet SDXL（默认配置）
#   bash train.sh controlnet sd15                        # ControlNet SD1.5（默认配置）
#   bash train.sh controlnetxs sdxl                      # ControlNet-XS SDXL（默认配置）
#   bash train.sh pixart                                 # PixArt-Sigma 全量训练（默认配置）
#   bash train.sh lora sdxl resume                       # 恢复训练
#   bash train.sh smoke                                  # SDXL 冒烟测试
#   bash train.sh overfit                                # ControlNet SDXL 过拟合测试
#   bash train.sh overfitxs                              # ControlNet-XS SDXL 过拟合测试
#   bash train.sh overfitlora                            # LoRA SDXL 过拟合测试
#   bash train.sh overfitpixart                          # PixArt-Sigma 过拟合测试（96张）
#   bash train.sh overfitpixart4img                      # PixArt-Sigma 过拟合测试（4张，固定lr）
#   bash train.sh overfitpixart16img                     # PixArt-Sigma 过拟合测试（16张，固定lr）
#   bash train.sh pixart_cn                              # PixArt-Sigma ControlNet 训练
#   bash train.sh overfitpixart_cn                       # PixArt-Sigma ControlNet 过拟合测试
#   bash train.sh pixart_ic                              # PixArt-Sigma 图像条件训练 (VAE 模式，默认)
#   bash train.sh pixart_ic_overfit                      # PixArt-Sigma 图像条件过拟合测试 (32张)
#   bash train.sh pixart_ic_dinov2                       # PixArt-Sigma 图像条件训练 (DINOv2 模式)
#   bash train.sh cache --config configs/lora_sdxl.yaml  # 多卡预计算 VAE latents
#
#   # 自定义配置文件（--config 覆盖默认选择）:
#   bash train.sh controlnet sdxl --config configs/controlnet_sdxl_v2.yaml
#   bash train.sh lora sdxl --config configs/lora_sdxl_hires.yaml resume
#   bash train.sh smoke --config configs/smoke_test_sdxl.yaml
#   bash train.sh pixart --config configs/pixart_sigma_custom.yaml
# ★ 修改训练卡：只需改下面 GPUS 一行
# =============================================================

set -euo pipefail

# ── 默认值 ────────────────────────────────────────────────────
TASK="lora"
MODEL_TYPE="sdxl"
RESUME_FLAG=""
CUSTOM_CONFIG=""
ZERO2_FLAG=""

# ── 解析参数（顺序不限，支持 --config <path>） ───────────────
args=("$@")
i=0
while [[ $i -lt ${#args[@]} ]]; do
    arg="${args[$i]}"
    case "$arg" in
        lora)         TASK="lora"         ;;
        controlnet)   TASK="controlnet"   ;;
        controlnetxs) TASK="controlnetxs" ;;
        sdxl)         MODEL_TYPE="sdxl"   ;;
        sd15)         MODEL_TYPE="sd15"   ;;
        smoke)        TASK="smoke"        ;;
        overfit)      TASK="overfit"      ;;
        overfitxs)    TASK="overfitxs"    ;;
        overfitlora)      TASK="overfitlora"      ;;
        pixart)           TASK="pixart"           ;;
        overfitpixart)    TASK="overfitpixart"    ;;
        overfitpixart4img)  TASK="overfitpixart4img"  ;;
        overfitpixart16img) TASK="overfitpixart16img" ;;
        pixart_cn)        TASK="pixart_cn"        ;;
        overfitpixart_cn) TASK="overfitpixart_cn" ;;
        pixart_ic)        TASK="pixart_ic"        ;;
        pixart_ic_overfit) TASK="pixart_ic_overfit" ;;
        pixart_ic_dinov2) TASK="pixart_ic_dinov2" ;;
        cache)        TASK="cache"        ;;
        zero2)       ZERO2_FLAG="1" ;;
        resume)      RESUME_FLAG="resume" ;;
        --config)
            i=$(( i + 1 ))
            if [[ $i -ge ${#args[@]} ]]; then
                echo "[ERROR] --config 后需要跟配置文件路径"
                exit 1
            fi
            CUSTOM_CONFIG="${args[$i]}"
            ;;
        *)
            echo "[ERROR] 未知参数: $arg"
            echo "  支持: lora | controlnet | controlnetxs | pixart | pixart_cn | pixart_ic | pixart_ic_dinov2 | sd15 | sdxl | smoke | overfit | overfitxs | overfitlora | overfitpixart | overfitpixart_cn | zero2 | resume | --config <path>"
            exit 1
            ;;
    esac
    i=$(( i + 1 ))
done

# ── 项目目录 & GPU 配置 ──────────────────────────────────────
PROJ_DIR="/home/daiqing_tan/stable_diffusion_lora"

# ★ 在此修改使用哪几张卡，num_processes 会自动计算
GPUS="0,1,2,3"
NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)

# 根据卡数自动选择 accelerate 配置
if [[ "$NUM_GPUS" -le 2 ]]; then
    ACCELERATE_CFG="configs/accelerate_2gpu.yaml"
elif [[ "$NUM_GPUS" -eq 3 ]]; then
    ACCELERATE_CFG="configs/accelerate_3gpu.yaml"
else
    ACCELERATE_CFG="configs/accelerate_4gpu.yaml"
fi

# zero2 参数覆盖为 DeepSpeed ZeRO-2 配置
if [[ -n "$ZERO2_FLAG" ]]; then
    if [[ "$NUM_GPUS" -le 2 ]]; then
        ACCELERATE_CFG="configs/accelerate_zero2_2gpu.yaml"
    elif [[ "$NUM_GPUS" -eq 3 ]]; then
        ACCELERATE_CFG="configs/accelerate_zero2_3gpu.yaml"
    else
        ACCELERATE_CFG="configs/accelerate_zero2_4gpu.yaml"
    fi
fi

# ── 根据 TASK + MODEL_TYPE 选择配置文件和入口脚本 ────────────
declare -A CONFIG_MAP=(
    ["lora:sdxl"]="configs/lora_sdxl_floorplan.yaml"
    ["lora:sd15"]="configs/lora_sd15_floorplan.yaml"
    ["controlnet:sdxl"]="configs/controlnet_sdxl.yaml"
    ["controlnet:sd15"]="configs/controlnet_sd15.yaml"
    ["controlnetxs:sdxl"]="configs/controlnet_xs_sdxl.yaml"
    ["overfit:sdxl"]="configs/overfit_test_controlnet_sdxl.yaml"
    ["overfitxs:sdxl"]="configs/overfit_test_controlnet_xs_sdxl.yaml"
    ["overfitlora:sdxl"]="configs/overfit_test_lora_sdxl.yaml"
    ["pixart:sdxl"]="configs/pixart_sigma_floorplan.yaml"
    ["overfitpixart:sdxl"]="configs/overfit_test_pixart_sigma.yaml"
    ["overfitpixart4img:sdxl"]="configs/overfit_test_pixart_sigma_4img.yaml"
    ["overfitpixart16img:sdxl"]="configs/overfit_test_pixart_sigma_16img.yaml"
    ["pixart_cn:sdxl"]="configs/controlnet_pixart_sigma.yaml"
    ["overfitpixart_cn:sdxl"]="configs/overfit_test_controlnet_pixart_sigma.yaml"
    ["pixart_ic:sdxl"]="configs/pixart_sigma_img_cond_floorplan.yaml"
    ["pixart_ic_overfit:sdxl"]="configs/pixart_sigma_img_cond_overfit.yaml"
    ["pixart_ic_dinov2:sdxl"]="configs/pixart_sigma_img_cond_floorplan.yaml"
    ["smoke:sdxl"]="configs/smoke_test_sdxl.yaml"
)

declare -A SCRIPT_MAP=(
    ["lora"]="scripts/train_lora.py"
    ["controlnet"]="scripts/train_controlnet.py"
    ["controlnetxs"]="scripts/train_controlnet_xs.py"
    ["overfit"]="scripts/train_controlnet.py"
    ["overfitxs"]="scripts/train_controlnet_xs.py"
    ["overfitlora"]="scripts/train_lora.py"
    ["pixart"]="scripts/train_pixart_sigma.py"
    ["overfitpixart"]="scripts/train_pixart_sigma.py"
    ["overfitpixart4img"]="scripts/train_pixart_sigma.py"
    ["overfitpixart16img"]="scripts/train_pixart_sigma.py"
    ["pixart_cn"]="scripts/train_pixart_controlnet.py"
    ["overfitpixart_cn"]="scripts/train_pixart_controlnet.py"
    ["pixart_ic"]="scripts/train_pixart_img_cond.py"
    ["pixart_ic_overfit"]="scripts/train_pixart_img_cond.py"
    ["pixart_ic_dinov2"]="scripts/train_pixart_img_cond.py"
    ["smoke"]="scripts/train_lora.py"
    ["cache"]="scripts/precompute_latents.py"
)

KEY="${TASK}:${MODEL_TYPE}"
if [[ "$TASK" == "smoke" ]]; then
    KEY="smoke:sdxl"
elif [[ "$TASK" == "overfit" ]]; then
    KEY="overfit:sdxl"
elif [[ "$TASK" == "overfitxs" ]]; then
    KEY="overfitxs:sdxl"
elif [[ "$TASK" == "overfitlora" ]]; then
    KEY="overfitlora:sdxl"
elif [[ "$TASK" == "pixart" ]]; then
    KEY="pixart:sdxl"
elif [[ "$TASK" == "overfitpixart" ]]; then
    KEY="overfitpixart:sdxl"
elif [[ "$TASK" == "overfitpixart4img" ]]; then
    KEY="overfitpixart4img:sdxl"
elif [[ "$TASK" == "overfitpixart16img" ]]; then
    KEY="overfitpixart16img:sdxl"
elif [[ "$TASK" == "pixart_cn" ]]; then
    KEY="pixart_cn:sdxl"
elif [[ "$TASK" == "overfitpixart_cn" ]]; then
    KEY="overfitpixart_cn:sdxl"
elif [[ "$TASK" == "pixart_ic" ]]; then
    KEY="pixart_ic:sdxl"
elif [[ "$TASK" == "pixart_ic_overfit" ]]; then
    KEY="pixart_ic_overfit:sdxl"
elif [[ "$TASK" == "pixart_ic_dinov2" ]]; then
    KEY="pixart_ic_dinov2:sdxl"
fi

TRAIN_SCRIPT="${SCRIPT_MAP[$TASK]:-}"
if [[ -z "$TRAIN_SCRIPT" ]]; then
    echo "[ERROR] 不支持的任务类型: ${TASK}"
    echo "  支持: lora | controlnet | controlnetxs | pixart | pixart_cn | smoke | overfit | overfitxs | overfitlora | overfitpixart | overfitpixart_cn | cache"
    exit 1
fi

# cache 任务必须通过 --config 指定配置
if [[ "$TASK" == "cache" ]]; then
    if [[ -z "$CUSTOM_CONFIG" ]]; then
        echo "[ERROR] cache 任务需要 --config 指定配置文件"
        echo "  示例: bash train.sh cache --config configs/lora_sdxl.yaml"
        exit 1
    fi
    CONFIG="$CUSTOM_CONFIG"
    CONFIG_STEM=$(basename "${CONFIG}" .yaml)
    SESSION="cache_${CONFIG_STEM}"
    LOG_FILE="${PROJ_DIR}/logs/cache_${CONFIG_STEM}.log"
# --config 优先；否则从 CONFIG_MAP 查默认配置
elif [[ -n "$CUSTOM_CONFIG" ]]; then
    CONFIG="$CUSTOM_CONFIG"
    # 校验文件存在
    if [[ ! -f "${PROJ_DIR}/${CONFIG}" && ! -f "${CONFIG}" ]]; then
        echo "[ERROR] 配置文件不存在: ${CONFIG}"
        exit 1
    fi
    # 从文件名（去掉路径和扩展名）生成 session / log 名
    CONFIG_STEM=$(basename "${CONFIG}" .yaml)
    SESSION="${TASK}_${CONFIG_STEM}"
    LOG_FILE="${PROJ_DIR}/logs/${TASK}_${CONFIG_STEM}.log"
else
    CONFIG="${CONFIG_MAP[$KEY]:-}"
    if [[ -z "$CONFIG" ]]; then
        echo "[ERROR] 不支持的组合: task=${TASK} model=${MODEL_TYPE}"
        echo "  支持: lora×{sd15,sdxl}, controlnet×{sd15,sdxl}, controlnetxs×sdxl, pixart, pixart_cn, pixart_ic, pixart_ic_dinov2, smoke, overfit, overfitxs, overfitlora, overfitpixart, overfitpixart_cn"
        echo "  或使用 --config <path> 指定自定义配置文件"
        exit 1
    fi
    SESSION="${TASK}_train_${MODEL_TYPE}"
    LOG_FILE="${PROJ_DIR}/logs/${TASK}_${MODEL_TYPE}.log"

    if [[ "$TASK" == "smoke" ]]; then
        SESSION="lora_smoke_test"
        LOG_FILE="${PROJ_DIR}/logs/train_smoke.log"
    elif [[ "$TASK" == "overfit" ]]; then
        SESSION="controlnet_overfit_test"
        LOG_FILE="${PROJ_DIR}/logs/train_overfit.log"
    elif [[ "$TASK" == "overfitxs" ]]; then
        SESSION="controlnet_xs_overfit_test"
        LOG_FILE="${PROJ_DIR}/logs/train_overfit_xs.log"
    elif [[ "$TASK" == "overfitlora" ]]; then
        SESSION="lora_overfit_test"
        LOG_FILE="${PROJ_DIR}/logs/train_overfit_lora.log"
    elif [[ "$TASK" == "overfitpixart" ]]; then
        SESSION="pixart_overfit_test"
        LOG_FILE="${PROJ_DIR}/logs/train_overfit_pixart.log"
    elif [[ "$TASK" == "overfitpixart4img" ]]; then
        SESSION="pixart_overfit_4img"
        LOG_FILE="${PROJ_DIR}/logs/train_overfit_pixart_4img.log"
    elif [[ "$TASK" == "overfitpixart16img" ]]; then
        SESSION="pixart_overfit_16img"
        LOG_FILE="${PROJ_DIR}/logs/train_overfit_pixart_16img.log"
    elif [[ "$TASK" == "pixart_cn" ]]; then
        SESSION="pixart_cn_train"
        LOG_FILE="${PROJ_DIR}/logs/train_pixart_cn.log"
    elif [[ "$TASK" == "overfitpixart_cn" ]]; then
        SESSION="pixart_cn_overfit_test"
        LOG_FILE="${PROJ_DIR}/logs/train_overfit_pixart_cn.log"
    elif [[ "$TASK" == "pixart_ic" ]]; then
        SESSION="pixart_ic_vae_train"
        LOG_FILE="${PROJ_DIR}/logs/train_pixart_ic_vae.log"
    elif [[ "$TASK" == "pixart_ic_overfit" ]]; then
        SESSION="pixart_ic_overfit_test"
        LOG_FILE="${PROJ_DIR}/logs/train_pixart_ic_overfit.log"
    elif [[ "$TASK" == "pixart_ic_dinov2" ]]; then
        SESSION="pixart_ic_dinov2_train"
        LOG_FILE="${PROJ_DIR}/logs/train_pixart_ic_dinov2.log"
    fi
fi

# 从 YAML 中提取 output_dir 推断 checkpoint 目录（cache 任务不需要）
if [[ "$TASK" != "cache" ]]; then
    CKPT_DIR=$(grep -m1 'output_dir:' "${PROJ_DIR}/${CONFIG}" | sed 's/.*: *"\{0,1\}\([^"]*\)"\{0,1\}/\1/' | sed 's/ *#.*//')
    CKPT_DIR="${PROJ_DIR}/${CKPT_DIR}/checkpoints"
fi

echo "[INFO] 任务: ${TASK} | 模型: ${MODEL_TYPE} | 配置: ${CONFIG}"
echo "[INFO] 入口: ${TRAIN_SCRIPT} | GPU: ${GPUS} (${NUM_GPUS} 张)"

# ── 创建日志目录 ─────────────────────────────────────────────
mkdir -p "${PROJ_DIR}/logs"

# ── 判断恢复还是全新训练 ─────────────────────────────────────
RESUME_ARG=""
if [[ "$TASK" == "cache" ]]; then
    echo "[INFO] 预计算模式，跳过 checkpoint 逻辑"
elif [[ "$RESUME_FLAG" == "resume" ]]; then
    LATEST=$(ls -td "${CKPT_DIR}"/step_* 2>/dev/null | head -1)
    if [[ -z "$LATEST" ]]; then
        echo "[WARN] 未找到任何 checkpoint，改为全新训练"
    else
        RESUME_ARG="--resume latest"
        echo "[INFO] 从最新 checkpoint 恢复: $(basename "$LATEST")"
    fi
else
    echo "[INFO] 全新训练，不加载 checkpoint"
fi

# ── 检查 tmux 会话冲突 ──────────────────────────────────────
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[ERROR] tmux 会话 '${SESSION}' 已在运行"
    echo "  进入查看: tmux attach -t ${SESSION}"
    echo "  强制终止: tmux kill-session -t ${SESSION}"
    exit 1
fi

# ── 拼接训练命令 ─────────────────────────────────────────────
OVERRIDE_ARG=""
if [[ "$TASK" == "pixart_ic_dinov2" ]]; then
    OVERRIDE_ARG="--override model.image_encoder.type=dinov2"
fi

TRAIN_CMD="PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=${GPUS} accelerate launch \
    --config_file ${ACCELERATE_CFG} \
    --num_processes ${NUM_GPUS} \
    ${TRAIN_SCRIPT} \
    --config ${CONFIG} \
    ${RESUME_ARG} ${OVERRIDE_ARG}"

# ── tmux 后台启动 ────────────────────────────────────────────
tmux new-session -d -s "$SESSION" -x 220 -y 50 "
    cd ${PROJ_DIR}
    echo '[START] '$(date '+%Y-%m-%d %H:%M:%S') | tee -a ${LOG_FILE}
    echo '[CMD]   ${TRAIN_CMD}'                 | tee -a ${LOG_FILE}
    eval ${TRAIN_CMD} 2>&1 | tee -a ${LOG_FILE}
    echo '[END]   '$(date '+%Y-%m-%d %H:%M:%S') | tee -a ${LOG_FILE}
    echo '--- 训练结束，按任意键关闭 ---'
    read
"

# ── 打印操作提示 ─────────────────────────────────────────────
echo ""
echo "=================================================="
echo "  训练已在 tmux 后台启动  (Session: ${SESSION})"
echo "=================================================="
echo ""
echo "  进入 tmux 实时查看  : tmux attach -t ${SESSION}"
echo "  退出 tmux 保持后台  : Ctrl+B  然后  D"
echo "  tail 查看日志       : tail -f ${LOG_FILE}"
echo "  查看所有会话        : tmux ls"
echo "  停止训练            : tmux kill-session -t ${SESSION}"
echo ""
