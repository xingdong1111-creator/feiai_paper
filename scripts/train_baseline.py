"""M1 基线训练: blind vs consensus 二分类。

用法(登录节点冒烟):  .venv/bin/python scripts/train_baseline.py --epochs 1
用法(SLURM 正式):    见 scripts/run_train.sh(由用户 sbatch 提交)

输出: runs/<tag>/metrics.json + preds_<split>.csv + best.pt
"""
import argparse
import csv
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.dataset import BlindConsensusDataset
from src.model import build_model

BASE = "/public/home/liuhuan/workspace_xd/feiai_paper"
MANIFEST = os.path.join(BASE, "data", "processed", "M1", "manifest.csv")


def evaluate(model, loader, device):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs.append(torch.softmax(logits, 1)[:, 1].cpu().numpy())
            labels.append(y.cpu().numpy())
    return np.concatenate(probs), np.concatenate(labels)


def auc(y, p):
    # Mann-Whitney U 秩法 AUC(并列用平均秩)
    from scipy.stats import rankdata
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(p)
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="3d", choices=["3d", "2d"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    tag = args.tag or ("m1_%s" % args.model)
    out_dir = os.path.join(BASE, "runs", tag)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = args.model
    tr = BlindConsensusDataset(MANIFEST, "train", mode=mode, augment=True, seed=1)
    va = BlindConsensusDataset(MANIFEST, "val", mode=mode)
    te = BlindConsensusDataset(MANIFEST, "test", mode=mode)
    trl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=args.workers)
    val = DataLoader(va, batch_size=args.batch, num_workers=args.workers)
    tel = DataLoader(te, batch_size=args.batch, num_workers=args.workers)

    # 类别权重(应对 blind/consensus 不均衡)
    ys = np.array([int(r["label"]) for r in tr.rows])
    cnt = np.bincount(ys, minlength=2).astype(float)
    w = torch.tensor(len(ys) / (2.0 * np.maximum(cnt, 1)), dtype=torch.float).to(device)
    crit = nn.CrossEntropyLoss(weight=w)

    model = build_model(args.model).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_auc, best_state = -1.0, None
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for x, y in trl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
        sched.step()
        vp, vy = evaluate(model, val, device)
        vauc = auc(vy, vp)
        if not np.isnan(vauc) and vauc > best_auc:
            best_auc = vauc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 5 == 0 or ep == 0:
            print("ep%02d loss=%.3f valAUC=%.3f best=%.3f" % (ep, tot / len(tr), vauc, best_auc),
                  flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, os.path.join(out_dir, "best.pt"))

    # test 评估 + 分层
    tp, ty = evaluate(model, tel, device)
    tacc = ((tp > 0.5).astype(int) == ty).mean()
    rec1 = ((tp > 0.5)[ty == 1].mean()) if (ty == 1).any() else float("nan")
    rec0 = ((tp <= 0.5)[ty == 0].mean()) if (ty == 0).any() else float("nan")
    metrics = {
        "model": args.model, "epochs": args.epochs,
        "val_auc_best": float(best_auc),
        "test_auc": float(auc(ty, tp)),
        "test_acc": float(tacc),
        "test_recall_consensus": float(rec1),
        "test_recall_blind": float(rec0),
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
    }
    # 直径分层(中位数切分) 控混杂初探
    te_meta = [te.meta(i) for i in range(len(te))]
    diam = np.array([float(m["diameter_mm"] or 0) for m in te_meta])
    med = np.median(diam) if len(diam) else 0
    for name, sel in [("small", diam <= med), ("large", diam > med)]:
        if sel.sum() and (ty[sel].sum() and (1 - ty[sel]).sum()):
            metrics["test_auc_%s" % name] = float(auc(ty[sel], tp[sel]))

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(out_dir, "preds_test.csv"), "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["patient_id", "nodule_id", "label", "prob", "diameter_mm"])
        for m, p, y in zip(te_meta, tp, ty):
            wtr.writerow([m["patient_id"], m["nodule_id"], y, "%.4f" % p, m["diameter_mm"]])

    print("\n===== TEST =====")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
