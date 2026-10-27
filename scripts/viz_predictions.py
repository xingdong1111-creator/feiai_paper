"""把结节判断标回原 CT 的具体切片与位置。
每个结节有质心 (x,y,z)(原图体素索引), 在 z 那张轴位切片的 (x,y) 画圈, 按类别/预测着色:
  blind(n_annotations=1)=红, consensus(=4)=绿, 其它(2/3)=橙。
可选 --preds JSON 用模型预测着色(键 case_id -> [{id,category,subtlety}]); 缺省用 LIDC 真值。
输出: 每结节轴位标记图 + 选定结节的轴/冠/矢三平面十字标记图。
"""
import argparse
import csv
import json
import os
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

BASE = "/public/home/liuhuan/workspace_xd/feiai_paper"
RAW = os.path.join(BASE, "data", "raw")
NODULES = os.path.join(BASE, "data", "meta", "nodules.csv")
FIG = os.path.join(BASE, "data", "figs")
WMIN, WMAX = -1200, 200
COLOR = {"blind": "red", "consensus": "lime", "other": "orange"}


def cat_of(n):
    return "blind" if n == "1" else "consensus" if n == "4" else "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--preds", default="", help="模型预测 JSON; 缺省用真值着色")
    args = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)

    rows = [r for r in csv.DictReader(open(NODULES)) if r["case_id"] == args.case]
    if not rows:
        print("no nodules for", args.case); return
    preds = {}
    if args.preds and os.path.exists(args.preds):
        for it in json.load(open(args.preds)).get(args.case, []):
            preds[str(it["id"])] = it

    img = sitk.ReadImage(os.path.join(RAW, "images", args.case + ".nii.gz"))
    ia = sitk.GetArrayFromImage(img)            # (Z,Y,X)
    sx, sy, sz = img.GetSpacing()
    Z = ia.shape[0]

    # ---- 图1: 每结节轴位标记 ----
    n = len(rows)
    cols = min(4, n); rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(4 * cols, 4 * rows_n), squeeze=False)
    for i, r in enumerate(rows):
        ax = axes[i // cols][i % cols]
        cx, cy, cz = float(r["centroid_x"]), float(r["centroid_y"]), int(round(float(r["centroid_z"])))
        cz = min(max(cz, 0), Z - 1)
        ax.imshow(ia[cz], cmap="gray", vmin=WMIN, vmax=WMAX)
        rad = (float(r["diameter_mm"]) / 2.0) / sx + 3
        cat = cat_of(r["n_annotations"])
        if preds:                                   # 用预测着色
            p = preds.get(r["nodule_id"], {})
            pcat = p.get("category", cat)
            color = COLOR.get(pcat, "cyan")
            lab = "nod%s pred:%s" % (r["nodule_id"], pcat)
        else:
            color = COLOR[cat]
            lab = "nod%s %s(NA=%s)" % (r["nodule_id"], cat, r["n_annotations"])
        ax.add_patch(Circle((cx, cy), rad, fill=False, edgecolor=color, linewidth=2))
        ax.plot([cx], [cy], marker="+", color=color, markersize=8)
        ax.set_title("%s | d=%.1fmm sub=%s mal=%s" % (lab, float(r["diameter_mm"]),
                     r["subtlety_score"], r["malignancy_score"]), fontsize=8, color="white")
        ax.set_xlabel("z=%d (x=%.0f,y=%.0f)" % (cz, cx, cy), fontsize=7, color="white")
        ax.axis("off")
    for j in range(n, rows_n * cols):
        axes[j // cols][j % cols].axis("off")
    fig.patch.set_facecolor("black")
    fig.suptitle("%s  nodule locations on axial slices (red=blind, green=consensus)" % args.case,
                 fontsize=11, color="white")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    f1 = os.path.join(FIG, "viz_axial_%s.png" % args.case)
    fig.savefig(f1, dpi=90, bbox_inches="tight", facecolor="black"); plt.close(fig)

    # ---- 图2: 第一个 blind(或首个)结节的三平面十字标记 ----
    target = next((r for r in rows if r["n_annotations"] == "1"), rows[0])
    cx, cy, cz = float(target["centroid_x"]), float(target["centroid_y"]), int(round(float(target["centroid_z"])))
    cz = min(max(cz, 0), Z - 1)
    axial = ia[cz]; coronal = ia[:, int(round(cy)), :]; sagittal = ia[:, :, int(round(cx))]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    axes[0].imshow(axial, cmap="gray", vmin=WMIN, vmax=WMAX)
    axes[0].add_patch(Circle((cx, cy), (float(target["diameter_mm"]) / 2) / sx + 3, fill=False, edgecolor="red", lw=2))
    axes[0].set_title("axial z=%d" % cz, fontsize=9); axes[0].axis("off")
    axes[1].imshow(coronal, cmap="gray", vmin=WMIN, vmax=WMAX, aspect="auto")
    axes[1].plot([cx], [cz], marker="+", color="red", markersize=12)
    axes[1].set_title("coronal y=%d" % int(round(cy)), fontsize=9); axes[1].axis("off")
    axes[2].imshow(sagittal, cmap="gray", vmin=WMIN, vmax=WMAX, aspect="auto")
    axes[2].plot([cy], [cz], marker="+", color="red", markersize=12)
    axes[2].set_title("sagittal x=%d" % int(round(cx)), fontsize=9); axes[2].axis("off")
    fig.suptitle("%s  nod%s (%s, d=%.1fmm) marked on 3 planes" % (
        args.case, target["nodule_id"], cat_of(target["n_annotations"]), float(target["diameter_mm"])), fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    f2 = os.path.join(FIG, "viz_3plane_%s.png" % args.case)
    fig.savefig(f2, dpi=90, bbox_inches="tight"); plt.close(fig)

    print("结节数:", n, "| 图1:", f1, "| 图2:", f2)


if __name__ == "__main__":
    main()
