#!/bin/bash
# 项目环境搭建脚本(可重复执行)
# 策略: venv 继承 base 的 numpy/pandas/matplotlib/torch, 只补装缺失包,
#       并锁定有 manylinux2014(glibc2.17)预编译轮的版本, 避免源码编译(CentOS7 GCC4.8 太老)
set -e
PY=/public/home/liuhuan/anaconda/anaconda3/bin/python3
VENV=/public/home/liuhuan/workspace_xd/feiai_paper/.venv
MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple

echo "[1/4] 清理旧 venv"
rm -rf "$VENV"

echo "[2/4] 创建 venv(--system-site-packages 继承 base 包)"
$PY -m venv --system-site-packages "$VENV"

echo "[3/4] 升级 pip"
"$VENV/bin/pip" install -q --upgrade "pip<25" -i "$MIRROR"

echo "[4/4] 安装缺失依赖(锁定版本 + 只用预编译轮)"
"$VENV/bin/pip" install -q -i "$MIRROR" --prefer-binary --only-binary=:all: \
    "scipy==1.11.4" "scikit-learn==1.3.2" "SimpleITK==2.3.1" \
    nibabel pydicom huggingface_hub tqdm

echo "===== 验证 ====="
"$VENV/bin/python" - <<'PYEOF'
import numpy, pandas, matplotlib, scipy, sklearn, nibabel, SimpleITK, pydicom, huggingface_hub, torch
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("sklearn", sklearn.__version__)
print("nibabel", nibabel.__version__)
print("SimpleITK", SimpleITK.__version__)
print("hf_hub", huggingface_hub.__version__)
print("torch", torch.__version__, "cuda?", torch.cuda.is_available())
print("ALL OK")
PYEOF
