"""数据对齐验证 + 可视化(方案 A 核心信号验证)。

对每个已下载病例:
  1. 校验 image 与 mask_annotation_count 尺寸/spacing 一致(空间对齐)
  2. 统计 mask 取值分布(应为 0-4 = 勾画该体素的医生人数)
  3. 对 CSV 里每个结节: 取其质心邻域 mask 最大值, 与 n_annotations 比对(标签自洽)
  4. 为代表性病例生成 overlay 切片图(CT 肺窗 + 按医生人数着色的 mask)

坐标约定(已实测): CSV 的 centroid_x/y/z 即体素索引, 对应 array[z, y, x]。
"""
import csv
import os
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import ndimage

BASE = "/public/home/liuhuan/workspace_xd/feiai_paper"
RAW = os.path.join(BASE, "data", "raw")
FIG = os.path.join(BASE, "data", "figs")
REPORT = os.path.join(BASE, "scripts", "_verify_report.txt")
os.makedirs(FIG, exist_ok=True)

# 医生人数 -> 颜色(1=红 盲区/漏诊危险, 4=绿 共识)
COUNT_COLORS = {
    1: (1.0, 0.0, 0.0),    # 红
    2: (1.0, 0.5, 0.0),    # 橙
    3: (1.0, 0.85, 0.0),   # 黄
    4: (0.0, 0.8, 0.0),    # 绿
}

def load_case(cid):
    img = sitk.ReadImage(os.path.join(RAW, "images", cid + ".nii.gz"))
    msk = sitk.ReadImage(os.path.join(RAW, "masks_annotation_count", cid + ".nii.gz"))
    return img, msk, sitk.GetArrayFromImage(img), sitk.GetArrayFromImage(msk)

def neighborhood_max(ma, z, y, x, r=4):
    Z, Y, X = ma.shape
    z0, z1 = max(0, int(z) - r), min(Z, int(z) + r + 1)
    y0, y1 = max(0, int(y) - r), min(Y, int(y) + r + 1)
    x0, x1 = max(0, int(x) - r), min(X, int(x) + r + 1)
    sub = ma[z0:z1, y0:y1, x0:x1]
    return int(sub.max()) if sub.size else 0

def make_overlay(cid, ia, ma, nodules, out_png):
    """为该病例的每个结节画一张轴向切片(CT肺窗 + mask按人数着色)。"""
    nodules = nodules[:4]
    n = len(nodules)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.6))
    if n == 1:
        axes = [axes]
    for ax, nd in zip(axes, nodules):
        z = int(round(float(nd["centroid_z"])))
        y = int(round(float(nd["centroid_y"])))
        x = int(round(float(nd["centroid_x"])))
        z = min(max(z, 0), ia.shape[0] - 1)
        ct = ia[z]                     # (Y, X)
        msk_slice = ma[z]              # (Y, X), 值 0-4
        ax.imshow(ct, cmap="gray", vmin=-1200, vmax=200)   # 肺窗
        # 构造 RGBA overlay
        rgba = np.zeros((*msk_slice.shape, 4), dtype=float)
        for v, c in COUNT_COLORS.items():
            sel = msk_slice == v
            rgba[sel, 0], rgba[sel, 1], rgba[sel, 2] = c
            rgba[sel, 3] = 0.55
        ax.imshow(rgba)
        ax.scatter([x], [y], s=120, facecolors="none", edgecolors="cyan", linewidths=1.5)
        cat = nd["category"]
        ax.set_title("nod%s | %d reader(s) | malig=%s | %.1fmm | %s" % (
            nd["nodule_id"], int(nd["n_annotations"]), nd["malignancy_score"],
            float(nd["diameter_mm"]) if nd["diameter_mm"] else 0.0, cat),
            fontsize=9)
        ax.axis("off")
    legend = [Patch(color=COUNT_COLORS[k], label="%d reader(s)" % k) for k in [1, 2, 3, 4]]
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle(cid, fontsize=11)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(out_png, dpi=95, bbox_inches="tight")
    plt.close(fig)

def main():
    rows = list(csv.DictReader(open(os.path.join(BASE, "data", "sample_cases.csv"))))
    # 只保留已下载的
    by_case = {}
    for r in rows:
        cid = r["case_id"]
        p = os.path.join(RAW, "images", cid + ".nii.gz")
        if os.path.exists(p):
            by_case.setdefault(cid, []).append(r)

    lines = []
    lines.append("=" * 70)
    lines.append("方案 A 数据对齐验证报告")
    lines.append("=" * 70)
    total_nod, match_nod, align_ok = 0, 0, 0
    fig_cases = []
    for cid, nods in by_case.items():
        img, msk, ia, ma = load_case(cid)
        same = img.GetSize() == msk.GetSize()
        spacing_same = np.allclose(img.GetSpacing(), msk.GetSpacing())
        if same and spacing_same:
            align_ok += 1
        vals, cnts = np.unique(ma, return_counts=True)
        vd = dict(zip(vals.tolist(), cnts.tolist()))
        lines.append("\n[%s] 结节数=%d" % (cid, len(nods)))
        lines.append("  尺寸对齐=%s (img%s msk%s) spacing一致=%s"
                     % (same, img.GetSize(), msk.GetSize(), spacing_same))
        lines.append("  mask取值(医生人数)->体素数: %s" % {k: v for k, v in vd.items() if k > 0})
        for nd in nods:
            z, y, x = float(nd["centroid_z"]), float(nd["centroid_y"]), float(nd["centroid_x"])
            obs = neighborhood_max(ma, z, y, x, r=4)
            exp = int(nd["n_annotations"])
            ok = (obs == exp)
            total_nod += 1
            match_nod += int(ok)
            lines.append("    nod%-2s 期望%d位医生 实测mask最大%d %s | 恶性度%s 直径%.1fmm [%s]"
                         % (nd["nodule_id"], exp, obs, "✓" if ok else "✗",
                            nd["malignancy_score"],
                            float(nd["diameter_mm"]) if nd["diameter_mm"] else 0.0,
                            nd["category"]))
        # 选代表病例出图: 含 blind 的优先
        if any(nd["category"] == "blind" for nd in nods):
            fig_cases.append((cid, ia, ma, nods))

    lines.append("\n" + "=" * 70)
    lines.append("汇总: 病例数=%d 尺寸全对齐=%d" % (len(by_case), align_ok))
    lines.append("结节标签自洽: %d/%d 匹配 (CSV n_annotations == mask邻域最大值)"
                 % (match_nod, total_nod))
    lines.append("=" * 70)

    # 出图: 前 5 个含盲区的病例
    made = []
    for cid, ia, ma, nods in fig_cases[:5]:
        png = os.path.join(FIG, "overlay_%s.png" % cid)
        make_overlay(cid, ia, ma, nods, png)
        made.append(png)

    with open(REPORT, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print("\n生成 overlay 图 %d 张:" % len(made))
    for m in made:
        print("  ", m)

if __name__ == "__main__":
    main()
