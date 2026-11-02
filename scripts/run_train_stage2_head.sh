#!/bin/bash
#SBATCH --job-name=feiai_s2head
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=runs/stage2_head_%j.log

set -e
cd /public/home/liuhuan/workspace_xd/feiai_paper
PY=/public/home/liuhuan/workspace_xd/feiai_paper/.venv/bin/python

# 阶段2-读出模式: ViT + 重采样 + projector + 冻结 Qwen2.5-1.5B,
# 在 last-hidden-state 的每个结节坐标位置读隐状态 -> 小头预测 {blind/consensus} + 隐蔽度。
# 依赖: 先跑 build_volumes --all 产出 data/processed/ONE; grid 须与之一致。
$PY scripts/train_stage2_head.py \
    --one-dir data/processed/ONE \
    --llm-path models/Qwen2.5-1.5B-Instruct \
    --grid 128 192 192 --patch 16 16 16 --dim 384 --depth 8 --heads 6 \
    --num-queries 64 --resampler-depth 4 \
    --epochs 20 --batch 1 --lr 1e-4 --max-len 192 --subtlety-weight 0.1 \
    --tag stage2_head \
    --workers 4 --grad-ckpt --fp16

echo "[run_train_stage2_head] done"
