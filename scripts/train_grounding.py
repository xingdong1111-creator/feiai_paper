"""Path B (grounding) 训练: ViT3D + 空间池化 + projector + Qwen2.5(手写LoRA)。
坐标是【输出】: LLM 学生成 JSON [{coord:[x,y,z],r,na,sub}, ...]。LM loss 只算目标 JSON 段。
可训: ViT/池化/projector(全量) + LLM LoRA; LLM base 冻结。GPU 运行, 支持梯度检查点+混合精度。
精度: --fp16 用于 V100(Volta, 无硬件 bf16, 用 GradScaler); --bf16 用于 A100/Ampere+。
"""
import argparse
import json
import os
import sys
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/public/home/liuhuan/workspace_xd/feiai_paper")
from src.vlm.grounding import build_groundingvlm
from src.vlm.grounding_data import GroundingDataset
from src.vlm.stage2_data import collate_stage2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one-dir", default="/public/home/liuhuan/workspace_xd/feiai_paper/data/processed/ONE")
    ap.add_argument("--llm-path", default="/public/home/liuhuan/workspace_xd/feiai_paper/models/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--grid", type=int, nargs=3, default=[128, 192, 192], help="须与 build_volumes 的 grid 一致")
    ap.add_argument("--patch", type=int, nargs=3, default=[16, 16, 16])
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--vis-grid", type=int, nargs=3, default=[4, 6, 6], help="空间池化后网格(保留3D排布)")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--tag", default="grounding")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--bf16", action="store_true", help="Ampere+(A100/3090) 用; V100 勿用")
    ap.add_argument("--fp16", action="store_true", help="V100(Volta) 用; 配 GradScaler 防梯度下溢")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)

    llm_dtype = torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)
    model = build_groundingvlm(args.llm_path, img_grid=tuple(args.grid), patch=tuple(args.patch),
                               dim=args.dim, depth=args.depth, heads=args.heads,
                               vis_grid=tuple(args.vis_grid), lora_r=args.lora_r, lora_alpha=args.lora_alpha,
                               llm_dtype=llm_dtype).to(device)
    if args.grad_ckpt:
        model.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrancy": False})
        model.llm.config.use_cache = False
    tok = model.tokenizer
    pad_id = tok.pad_token_id
    ntr = sum(p.numel() for p in model.trainable_parameters())
    print("llm_dim:", model.llm_dim, "| vis tokens:", model.num_vis, "| LoRA层:", model.n_lora,
          "| trainable(M): %.2f" % (ntr / 1e6), flush=True)

    tr = GroundingDataset(args.one_dir, "train", tok, max_len=args.max_len)
    va = GroundingDataset(args.one_dir, "val", tok, max_len=args.max_len)
    print("cases train/val:", len(tr), len(va), flush=True)
    coll = lambda b: collate_stage2(b, pad_id)
    trl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=args.workers, collate_fn=coll)
    val = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=args.workers, collate_fn=coll)

    opt = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=1e-2)
    use_amp = device == "cuda" and (args.fp16 or args.bf16)
    amp_dtype = torch.float16 if args.fp16 else torch.bfloat16
    amp = torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype)
    # GradScaler 仅 fp16 需要(bf16 动态范围足够); 关闭时 scale/step/update 自动透传
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and args.fp16))
    print("precision:", "fp16" if args.fp16 else ("bf16" if args.bf16 else "fp32"),
          "| amp:", use_amp, "| gradscaler:", scaler.is_enabled(), flush=True)

    out_dir = "/public/home/liuhuan/workspace_xd/feiai_paper/runs/" + args.tag
    os.makedirs(out_dir, exist_ok=True)
    best = float("inf")
    for ep in range(1, args.epochs + 1):
        model.vit.train(); model.pool.train(); model.projector.train()
        tot = 0.0; nb = 0
        for b in trl:
            vol = b["volume"].to(device)
            ids = b["input_ids"].to(device); attn = b["attention_mask"].to(device); labs = b["labels"].to(device)
            opt.zero_grad()
            with amp:
                loss = model(vol, ids, attn, labels=labs).loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss); nb += 1
        model.vit.eval(); model.pool.eval(); model.projector.eval()
        vt = 0.0; vn = 0
        with torch.no_grad():
            for b in val:
                vol = b["volume"].to(device)
                ids = b["input_ids"].to(device); attn = b["attention_mask"].to(device); labs = b["labels"].to(device)
                with amp:
                    vt += float(model(vol, ids, attn, labels=labs).loss); vn += 1
        tr_loss = tot / max(nb, 1); va_loss = vt / max(vn, 1)
        print("ep%03d train_loss=%.4f val_loss=%.4f" % (ep, tr_loss, va_loss), flush=True)
        if va_loss < best:
            best = va_loss
            torch.save({"vit": model.vit.state_dict(), "pool": model.pool.state_dict(),
                        "projector": model.projector.state_dict(),
                        "lora": {k: v for k, v in model.llm.state_dict().items() if "lora_" in k},
                        "epoch": ep, "val_loss": best},
                       os.path.join(out_dir, "best_grounding.pt"))
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"best_val_loss": best, "epochs": args.epochs,
                   "vis_grid": args.vis_grid, "lora_r": args.lora_r}, f, indent=2)
    print("done best_val_loss=%.4f -> %s" % (best, out_dir), flush=True)


if __name__ == "__main__":
    main()
