#!/bin/bash
#SBATCH --job-name=feiai_ground
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=runs/grounding_%j.log

cd /public/home/liuhuan/workspace_xd/feiai_paper
PY=/public/home/liuhuan/workspace_xd/feiai_paper/.venv/bin/python

# Path B (grounding): 坐标作【输出】。ViT3D + 空间池化(4x6x6,保留3D排布) + projector + Qwen2.5(手写LoRA)。
# LLM 学生成 JSON [{coord:[x,y,z](0-1000), r, na, sub}, ...]。依赖先跑 build_volumes --all 产出 data/processed/ONE。
$PY scripts/train_grounding.py \
    --one-dir data/processed/ONE \
    --llm-path models/Qwen2.5-1.5B-Instruct \
    --grid 128 192 192 --patch 16 16 16 --dim 384 --depth 8 --heads 6 \
    --vis-grid 4 6 6 --lora-r 16 --lora-alpha 32 \
    --epochs 30 --batch 1 --lr 1e-4 --max-len 320 --tag grounding \
    --workers 4 --grad-ckpt --fp16

echo "[run_train_grounding] done"
