"""Test 集最终评估(grounding, 一次性): 载入训练好的模型, 在 split=test 上 生成->解析->与GT匹配,
报告 检测recall@dist / 平均坐标误差 / blind-vs-consensus AUC / na准确率 / subtlety MAE。
test 在训练与模型选择全程未碰, 此处为最终评估(不再调参)。GPU 运行。
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict
import numpy as np
import torch

ROOT = "/public/home/liuhuan/workspace_xd/feiai_paper"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from src.vlm.grounding import build_groundingvlm
from src.vlm.grounding_data import PROMPT
from infer_grounding import parse_pred


def auc(scores, labels):
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
    return float((ranks[l == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one-dir", default=os.path.join(ROOT, "data/processed/ONE"))
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "runs/grounding/best_grounding.pt"))
    ap.add_argument("--llm-path", default=os.path.join(ROOT, "models/Qwen2.5-1.5B-Instruct"))
    ap.add_argument("--grid", type=int, nargs=3, default=[128, 192, 192])
    ap.add_argument("--patch", type=int, nargs=3, default=[16, 16, 16])
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--vis-grid", type=int, nargs=3, default=[4, 6, 6])
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--thresh", type=float, default=0.06, help="匹配阈值(归一化3D距离)")
    ap.add_argument("--max-cases", type=int, default=0, help="0=全部test")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_groundingvlm(args.llm_path, img_grid=tuple(args.grid), patch=tuple(args.patch),
                               dim=args.dim, depth=args.depth, heads=args.heads,
                               vis_grid=tuple(args.vis_grid), lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    if os.path.exists(args.ckpt):
        sd = torch.load(args.ckpt, map_location="cpu")
        model.vit.load_state_dict(sd["vit"]); model.pool.load_state_dict(sd["pool"])
        model.projector.load_state_dict(sd["projector"]); model.llm.load_state_dict(sd["lora"], strict=False)
        print("loaded ckpt epoch:", sd.get("epoch"))
    model.eval().to(device)
    tok = model.tokenizer
    p_ids = tok(PROMPT, add_special_tokens=True).input_ids
    p_t = torch.tensor([p_ids]); p_a = torch.ones(1, len(p_ids), dtype=torch.long)

    rows = [r for r in csv.DictReader(open(os.path.join(args.one_dir, "cases_manifest.csv")))
            if r["split"] == "test"]
    by = defaultdict(list)
    for r in rows:
        by[r["case_id"]].append(r)
    cases = sorted(by)
    if args.max_cases:
        cases = cases[:args.max_cases]
    print("test cases:", len(cases), "| device:", device, flush=True)

    gt_n = matched = 0
    errs = []; scores = []; labels = []
    na_ok = na_tot = 0; sub_errs = []
    with torch.no_grad():
        for ci, case in enumerate(cases):
            gts = by[case]
            vol = torch.from_numpy(np.load(os.path.join(args.one_dir, "volumes", case + ".npz"))["volume"]
                                   .astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            gen = model.generate(vol, p_t.to(device), p_a.to(device), max_new_tokens=320, do_sample=False)
            preds = parse_pred(tok.decode(gen[0], skip_special_tokens=True))
            pc = [(p["coord"][0] / 1000, p["coord"][1] / 1000, p["coord"][2] / 1000) for p in preds]
            used = set()
            for g in gts:
                gc = (float(g["nx"]), float(g["ny"]), float(g["nz"]))
                best, bd = None, 1e9
                for pi, c in enumerate(pc):
                    if pi in used:
                        continue
                    d = float(np.sqrt((gc[0] - c[0]) ** 2 + (gc[1] - c[1]) ** 2 + (gc[2] - c[2]) ** 2))
                    if d < bd:
                        bd, best = d, pi
                gt_n += 1
                if best is not None and bd <= args.thresh:
                    matched += 1; used.add(best); errs.append(bd)
                    p = preds[best]
                    if int(g["n_annotations"]) in (1, 4):
                        labels.append(1 if g["n_annotations"] == "1" else 0)
                        scores.append(-float(p["na"]))          # na 越低 -> 越 blind
                        na_tot += 1; na_ok += int(int(p["na"]) == int(g["n_annotations"]))
                        try:
                            sub_errs.append(abs(float(p["sub"]) - float(g["subtlety_score"])))
                        except (ValueError, TypeError):
                            pass
            if (ci + 1) % 20 == 0:
                print("  %d/%d cases, recall so far=%.3f" % (ci + 1, len(cases), matched / max(gt_n, 1)), flush=True)

    recall = matched / max(gt_n, 1)
    mean_err = float(np.mean(errs)) if errs else float("nan")
    a = auc(scores, labels)
    na_acc = na_ok / max(na_tot, 1)
    sub_mae = float(np.mean(sub_errs)) if sub_errs else float("nan")
    res = {"n_test_cases": len(cases), "gt_nodules": gt_n, "matched": matched,
           "recall@%.2f" % args.thresh: round(recall, 4),
           "mean_coord_err_norm": round(mean_err, 4),
           "blind_auc": round(a, 4) if a == a else None,
           "na_acc": round(na_acc, 4), "subtlety_mae": round(sub_mae, 3),
           "n_blind_eval": len(labels)}
    print("\n=== TEST 最终评估 ===")
    for k, v in res.items():
        print("  %s: %s" % (k, v))
    out = os.path.join(ROOT, "runs/grounding/metrics_test.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print("saved:", out)


if __name__ == "__main__":
    main()
