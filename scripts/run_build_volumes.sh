#!/bin/bash
#SBATCH --job-name=feiai_buildvols
#SBATCH --partition=low
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=runs/build_volumes_%j.log

set -e
cd /public/home/liuhuan/workspace_xd/feiai_paper
# 全量预处理: 883 例整例 CT -> 固定网格 npz + 每结节归一化坐标/标签/split
# (CPU 密集, 约数小时; 默认 spacing 2mm, grid 128x192x192)
.venv/bin/python scripts/build_volumes.py --all --spacing 2 2 2 --grid 128 192 192

echo "[run_build_volumes] done"
