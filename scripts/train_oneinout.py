"""one-in-one-out 阶段1: 整例 CT 进, 一次前向输出该例所有结节的 blind/consensus 预测。
模型 = 3D 编码器 + 坐标条件 query 头(无 LLM)。按 patient 划分, CE(masked by valid), AUC 评估。
"""
import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata
from torch.utils.data import DataLoader

sys.path.insert(0, "/public/home/liuhuan/workspace_xd/feiai_paper")
from src.vlm.model import build_oneinout
from src.vlm.dataset import WholeCaseDataset, collate_cases


def auc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = rankdata(p)
    return float((r[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@torch.no_grad()
def evaluate(model, loader, device, diam_bin=None):
    model.eval()
    all_p, all_y, all_d = [], [], []
    for b in loader:
        vol = b["volume"].to(device); coords = b["coords"].to(device)
        valid = b["valid"].to(device)
        logits = model(vol, coords)
        prob = torch.softmax(logits, -1)[:, :, 1]
        m = valid > 0.5
        mc = m.cpu()
        all_p.append(prob[mc].numpy())
        all_y.append(b["labels"][mc].numpy())
        if diam_bin is not None:
            all_d.append((b["diam"][mc] < diam_bin).float().numpy())
    p = np.concatenate(all_p) if all_p else np.array([])
    y = np.concatenate(all_y) if all_y else np.array([])
    res = {"auc": auc(y, p), "n": int(len(y)),
           "acc": float(((p >= 0.5).astype(int) == y).mean()) if len(y) else float("nan")}
    for c in (0, 1):
        res["recall_%d" % c] = float(((p >= 0.5).astype(int)[y == c] == c).mean()) if (y == c).sum() else float("nan")
    if all_d:
        d = np.concatenate(all_d)
        for name, mask in [("small", d == 1), ("large", d == 0)]:
            res["auc_%s" % name] = auc(y[mask], p[mask])
    return res, p, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one-dir", default="/public/home/liuhuan/workspace_xd/feiai_paper/data/processed/ONE")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--tag", default="oneinout")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--eval-diameter-bin", type=float, default=6.0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)

    tr = WholeCaseDataset(args.one_dir, "train", augment=True, seed=0)
    va = WholeCaseDataset(args.one_dir, "val", augment=False)
    te = WholeCaseDataset(args.one_dir, "test", augment=False)
    print("cases train/val/test:", len(tr), len(va), len(te), flush=True)

    def mk(ds, sh):
        return DataLoader(ds, batch_size=args.batch, shuffle=sh, num_workers=args.workers,
                          collate_fn=collate_cases)
    trl, val, tel = mk(tr, True), mk(va, False), mk(te, False)

    # 类别权重(train 的 valid 结节)
    cnt = np.zeros(2)
    for i in range(len(tr)):
        s = tr[i]
        m = s["valid"] > 0.5
        cnt += np.bincount(s["labels"][m].numpy(), minlength=2)
    w = torch.tensor((cnt.sum() / (2 * np.maximum(cnt, 1))), dtype=torch.float32, device=device)
    print("train valid nodules blind/consensus:", cnt.tolist(), "weights:", w.tolist(), flush=True)

    model = build_oneinout(base=args.base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    amp = torch.amp.autocast("cuda", enabled=(device == "cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    out_dir = "/public/home/liuhuan/workspace_xd/feiai_paper/runs/" + args.tag
    os.makedirs(out_dir, exist_ok=True)
    best = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0; nb = 0
        for b in trl:
            vol = b["volume"].to(device); coords = b["coords"].to(device)
            labels = b["labels"].to(device); valid = b["valid"].to(device)
            opt.zero_grad()
            with amp:
                logits = model(vol, coords)
                ce = F.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1), weight=w, reduction="none")
                ce = ce.reshape(labels.shape)
                loss = (ce * valid).sum() / valid.sum().clamp(min=1)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot += float(loss); nb += 1
        sched.step()
        vres, _, _ = evaluate(model, val, device)
        print("ep%03d loss=%.4f val_auc=%.4f val_acc=%.4f" % (ep, tot / max(nb, 1), vres["auc"], vres["acc"]), flush=True)
        if not np.isnan(vres["auc"]) and vres["auc"] > best:
            best = vres["auc"]
            torch.save({"model": model.state_dict(), "epoch": ep, "val_auc": best},
                       os.path.join(out_dir, "best.pt"))
    # test
    ck = torch.load(os.path.join(out_dir, "best.pt"), map_location=device)
    model.load_state_dict(ck["model"])
    tres, _, _ = evaluate(model, tel, device, diam_bin=args.eval_diameter_bin)
    tres["best_val_auc"] = best
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(tres, f, indent=2, ensure_ascii=False)
    print("TEST:", json.dumps(tres, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
