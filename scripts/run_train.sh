#!/bin/bash
#SBATCH --job-name=m1_blindspot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=runs/slurm_%j.log

# M1 基线训练提交脚本(由用户 sbatch 执行, 助手不代为提交)
# 用法: sbatch scripts/run_train.sh
cd /public/home/liuhuan/workspace_xd/feiai_paper
mkdir -p runs

.venv/bin/python scripts/train_baseline.py --model 3d --epochs 60 --batch 8 --lr 1e-3 --tag m1_3d
.venv/bin/python scripts/train_baseline.py --model 2d --epochs 60 --batch 8 --lr 1e-3 --tag m1_2d
