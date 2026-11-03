"""【已废弃 DEPRECATED】snapshot_download 版在本环境(httpx2)报 TypeError。
请改用 download_qwen.sh (curl 版)。本文件仅留作历史参考。"""
"""从 hf-mirror 下载 Qwen2.5-1.5B-Instruct 权重到 models/。"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from huggingface_hub import snapshot_download

REPO = "Qwen/Qwen2.5-1.5B-Instruct"
OUT = "/public/home/liuhuan/workspace_xd/feiai_paper/models/Qwen2.5-1.5B-Instruct"
print("下载", REPO, "->", OUT, flush=True)
p = snapshot_download(repo_id=REPO, local_dir=OUT, local_dir_use_symlinks=False,
                      allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model", "tokenizer*"])
print("DOWNLOADED", p, flush=True)
