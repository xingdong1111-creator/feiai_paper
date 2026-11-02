"""阶段2-读出模式训练: ViT + 重采样 + projector + 冻结 Qwen2.5, 在 last-hidden-state 上接小头,
对每个结节坐标位置读出隐状态 -> 预测 {blind/consensus} + 隐蔽度回归。不生成文字。
只训视觉侧(ViT/重采样/projector/头); LLM 冻结。GPU 运行, 支持梯度检查点+bf16。
"""
import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, "/public/home/liuhuan/workspace_xd/feiai_paper")
from src.vlm.stage2 import build_qwenvlm
from src.vlm.stage2_data import Stage2HeadDataset, collate_head


def auc(scores, labels):
    """Mann-Whitney 秩法 AUC(含并列平均)。"""
    scores = np.asarray(scores, float); labels = np.asarray(labels, int)
    n1 = int((labels == 1).sum()); n0 = int((labels == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]; l = labels[order]
    ranks = np.arange(1, len(s) + 1, dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[i:j + 1] = ranks[i:j + 1].mean()
        i = j + 1
    sum_pos = ranks[l == 1].sum()
    return float((sum_pos - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one-dir", default="/public/home/liuhuan/workspace_xd/feiai_paper/data/processed/ONE")
    ap.add_argument("--llm-path", default="/public/home/liuhuan/workspace_xd/feiai_paper/models/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--grid", type=int, nargs=3, default=[128, 192, 192], help="须与 build_volumes 的 grid 一致")
    ap.add_argument("--patch", type=int, nargs=3, default=[16, 16, 16])
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--num-queries", type=int, default=64)
    ap.add_argument("--resampler-depth", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--subtlety-weight", type=float, default=0.1, help="隐蔽度回归 loss 权重(0=关闭)")
    ap.add_argument("--tag", default="stage2_head")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--bf16", action="store_true", help="Ampere+(A100/3090) 用; V100 勿用")
    ap.add_argument("--fp16", action="store_true", help="V100(Volta) 用; 配 GradScaler 防梯度下溢")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)

    llm_dtype = torch.float32
    if device == "cuda" and args.fp16:
        llm_dtype = torch.float16
    elif device == "cuda" and args.bf16:
        llm_dtype = torch.bfloat16
    elif args.fp16 or args.bf16:
        print("WARN: 请求半精度但无 CUDA, LLM 权重回退 fp32", flush=True)
    model = build_qwenvlm(args.llm_path, img_grid=tuple(args.grid), patch=tuple(args.patch),
                          dim=args.dim, depth=args.depth, heads=args.heads,
                          num_queries=args.num_queries, resampler_depth=args.resampler_depth,
                          llm_dtype=llm_dtype).to(device)
    if args.grad_ckpt:
        model.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrancy": False})
    tok = model.tokenizer
    pad_id = tok.pad_token_id
    print("llm_dim:", model.llm_dim, "| trainable(M): %.2f" % (
        sum(p.numel() for p in model.trainable_parameters()) / 1e6), flush=True)

    tr = Stage2HeadDataset(args.one_dir, "train", tok, max_len=args.max_len)
    va = Stage2HeadDataset(args.one_dir, "val", tok, max_len=args.max_len)
    print("cases train/val:", len(tr), len(va), flush=True)
    coll = lambda b: collate_head(b, pad_id)
    trl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=args.workers, collate_fn=coll)
    val = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=args.workers, collate_fn=coll)

    opt = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=1e-2)
    use_amp = device == "cuda" and (args.fp16 or args.bf16)
    amp_dtype = torch.float16 if args.fp16 else torch.bfloat16
    amp = torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and args.fp16))
    print("precision:", "fp16" if args.fp16 else ("bf16" if args.bf16 else "fp32"),
          "| amp:", use_amp, "| gradscaler:", scaler.is_enabled(), flush=True)

    def compute_loss(b):
        vol = b["volume"].to(device)
        ids = b["input_ids"].to(device); attn = b["attention_mask"].to(device)
        ro = b["readout_idx"].to(device); labs = b["labels"].to(device)
        valid = b["valid"].to(device); subt = b["subtlety"].to(device)
        with amp:
            logits, sp = model.forward_head(vol, ids, attn, ro)
            logits = logits.float(); sp = sp.float()
            lg = logits[valid]; lb = labs[valid]
            loss = F.cross_entropy(lg, lb) if lb.numel() else logits.sum() * 0.0
            if args.subtlety_weight > 0 and valid.any():
                # 隐蔽度 1~5 归一到 [-1,1] 再 MSE
                tgt = (subt[valid] - 3.0) / 2.0
                loss = loss + args.subtlety_weight * F.mse_loss(sp[valid], tgt)
        return loss, logits, valid, labs

    out_dir = "/public/home/liuhuan/workspace_xd/feiai_paper/runs/" + args.tag
    os.makedirs(out_dir, exist_ok=True)
    best_auc = -1.0
    for ep in range(1, args.epochs + 1):
        model.vit.train(); model.resampler.train(); model.projector.train()
        model.head.train(); model.subtlety_head.train()
        tot = 0.0; nb = 0
        for b in trl:
            opt.zero_grad()
            loss, _, _, _ = compute_loss(b)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss); nb += 1
        # 验证: loss + AUC
        model.vit.eval(); model.resampler.eval(); model.projector.eval()
        model.head.eval(); model.subtlety_head.eval()
        vt = 0.0; vn = 0; all_s = []; all_y = []
        with torch.no_grad():
            for b in val:
                loss, logits, valid, labs = compute_loss(b)
                vt += float(loss); vn += 1
                prob = torch.softmax(logits, -1)[..., 1]
                all_s.append(prob[valid].cpu().numpy()); all_y.append(labs[valid].cpu().numpy())
        tr_loss = tot / max(nb, 1); va_loss = vt / max(vn, 1)
        va_auc = auc(np.concatenate(all_s) if all_s else [], np.concatenate(all_y) if all_y else [])
        print("ep%03d train_loss=%.4f val_loss=%.4f val_auc=%.4f" % (ep, tr_loss, va_loss, va_auc), flush=True)
        if va_auc == va_auc and va_auc > best_auc:   # 非 nan 且更优
            best_auc = va_auc
            torch.save({"vit": model.vit.state_dict(), "resampler": model.resampler.state_dict(),
                        "projector": model.projector.state_dict(), "head": model.head.state_dict(),
                        "subtlety_head": model.subtlety_head.state_dict(),
                        "epoch": ep, "val_auc": best_auc, "val_loss": va_loss},
                       os.path.join(out_dir, "best_visual.pt"))
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"best_val_auc": best_auc, "epochs": args.epochs,
                   "subtlety_weight": args.subtlety_weight}, f, indent=2)
    print("done best_val_auc=%.4f -> %s" % (best_auc, out_dir), flush=True)


if __name__ == "__main__":
    main()
