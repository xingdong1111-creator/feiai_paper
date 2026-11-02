#!/bin/bash
#SBATCH --job-name=feiai_stage2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=runs/stage2_%j.log

set -e
cd /public/home/liuhuan/workspace_xd/feiai_paper
PY=/public/home/liuhuan/workspace_xd/feiai_paper/.venv/bin/python

# 阶段2: 3D ViT + 重采样 + projector + 冻结 Qwen2.5-1.5B, LM loss 学 per-nodule JSON
# 依赖: 先跑 build_volumes --all 产出 data/processed/ONE; grid 须与之一致
$PY scripts/train_stage2.py \
    --one-dir data/processed/ONE \
    --llm-path models/Qwen2.5-1.5B-Instruct \
    --grid 128 192 192 --patch 16 16 16 --dim 384 --depth 8 --heads 6 \
    --num-queries 64 --resampler-depth 4 \
    --epochs 20 --batch 1 --lr 1e-4 --max-len 256 --tag stage2 \
    --workers 4 --grad-ckpt --fp16

echo "[run_train_stage2] done"
