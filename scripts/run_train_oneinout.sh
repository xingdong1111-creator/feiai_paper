#!/bin/bash
#SBATCH --job-name=feiai_oneinout
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=runs/oneinout_%j.log

cd /public/home/liuhuan/workspace_xd/feiai_paper
PY=/public/home/liuhuan/workspace_xd/feiai_paper/.venv/bin/python

# 阶段1: one-in-one-out (整例进/全结节出), 3D编码器+坐标条件query头, 无LLM
$PY scripts/train_oneinout.py \
    --one-dir data/processed/ONE \
    --epochs 60 --batch 2 --lr 3e-4 --base 32 --tag oneinout --workers 4

echo "[run_train_oneinout] done"
