#!/bin/bash
# 用 curl 从 hf-mirror 直接拉 Qwen2.5-1.5B-Instruct 权重(绕开 huggingface_hub 的 httpx2 兼容问题)。
set -u
BASE="https://hf-mirror.com/Qwen/Qwen2.5-1.5B-Instruct/resolve/main"
OUT="/public/home/liuhuan/workspace_xd/feiai_paper/models/Qwen2.5-1.5B-Instruct"
mkdir -p "$OUT"
FILES="config.json generation_config.json model.safetensors tokenizer.json tokenizer_config.json vocab.json merges.txt special_tokens_map.json"
for f in $FILES; do
  dst="$OUT/$f"
  if [ -s "$dst" ]; then echo "[SKIP] $f"; continue; fi
  echo "[GET ] $f"
  curl -L -C - --retry 5 --retry-delay 3 -sS -o "$dst" "$BASE/$f" && echo "[OK  ] $f $(du -h "$dst" | cut -f1)" || echo "[FAIL] $f"
done
echo "DONE -> $OUT"
ls -la "$OUT"
