#!/bin/bash
#SBATCH --job-name=feiai_geval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=runs/eval_grounding_%j.log

set -e
cd /public/home/liuhuan/workspace_xd/feiai_paper
PY=/public/home/liuhuan/workspace_xd/feiai_paper/.venv/bin/python

# TEST 集最终评估(一次性, 不再调参): 生成->解析->与GT匹配, 报 recall@dist/坐标误差/blind AUC/na准确率。
# 提交时用 --dependency=afterok:<训练jobid> 挂在训练之后。
$PY scripts/eval_grounding.py \
    --one-dir data/processed/ONE \
    --ckpt runs/grounding/best_grounding.pt \
    --llm-path models/Qwen2.5-1.5B-Instruct \
    --grid 128 192 192 --patch 16 16 16 --dim 384 --depth 8 --heads 6 \
    --vis-grid 4 6 6 --lora-r 16 --lora-alpha 32 \
    --thresh 0.06

echo "[run_eval_grounding] done"
