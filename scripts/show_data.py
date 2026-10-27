"""给用户直观展示"数据到底长什么样", 回应"VLM 为啥不能直接吃 3D CT"。

对指定病例(默认那个 33.6mm/恶性度5/仅1位医生发现的盲区结节):
  图A: 4x4 连续 16 张轴位切片(肺窗) -> 证明"是一摞2D切片", 且结节跨切片出现/变化
  图B: 轴位/冠位/矢位 中心重建 + 薄层MIP -> "把3D压进2D"的实用VLM输入
并打印该病例总切片数(证明"几百张")。
"""
import csv
import os
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/public/home/liuhuan/workspace_xd/feiai_paper"
RAW = os.path.join(BASE, "data", "raw")
FIG = os.path.join(BASE, "data", "figs")
CASE = "LIDC-IDRI-0020_6518747372"
WMIN, WMAX = -1200, 200   # 肺窗

# 取该病例 blind 结节质心
row = next(r for r in csv.DictReader(open(os.path.join(BASE, "data", "sample_cases.csv")))
           if r["case_id"] == CASE and r["category"] == "blind")
cx, cy, cz = (float(row["centroid_x"]), float(row["centroid_y"]), float(row["centroid_z"]))
print("病例:", CASE, "| blind nod%s 恶性度%s 直径%.1fmm" % (
    row["nodule_id"], row["malignancy_score"], float(row["diameter_mm"])))
print("质心体素(x,y,z): %.0f,%.0f,%.0f" % (cx, cy, cz))

img = sitk.ReadImage(os.path.join(RAW, "images", CASE + ".nii.gz"))
ia = sitk.GetArrayFromImage(img)          # (Z,Y,X)
Z, Y, X = ia.shape
sp = img.GetSpacing()
print("总切片数 Z =", Z, "| spacing =", sp, "| 物理z跨度 ≈ %.0f mm" % (Z * sp[2]))

# ---- 图A: 连续16张轴位切片 ----
z0 = int(round(cz)) - 8
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for k, ax in enumerate(axes.flat):
    z = z0 + k
    z = min(max(z, 0), Z - 1)
    ax.imshow(ia[z], cmap="gray", vmin=WMIN, vmax=WMAX)
    ax.set_title("axial z=%d" % z, fontsize=8)
    ax.axis("off")
fig.suptitle("%s  16 consecutive axial slices (lung window) - a stack of 2D slices; nodule appears/changes across z" % CASE, fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fa = os.path.join(FIG, "show_stack_%s.png" % CASE)
fig.savefig(fa, dpi=90, bbox_inches="tight")
plt.close(fig)

# ---- 图B: 三平面重建 + 薄层MIP ----
zi, yi, xi = int(round(cz)), int(round(cy)), int(round(cx))
axial = ia[zi]                      # (Y,X)
coronal = ia[:, yi, :]              # (Z,X)
sagittal = ia[:, :, xi]             # (Z,Y)
slab = ia[max(0, zi - 8):zi + 9]
mip = slab.max(axis=0)              # (Y,X) 薄层最大密度投影

fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
for ax, im, t in zip(axes, [axial, coronal, sagittal, mip],
                     ["axial @z=%d" % zi, "coronal @y=%d" % yi,
                      "sagittal @x=%d" % xi, "thin-slab MIP(17 slices)"]):
    ax.imshow(im, cmap="gray", vmin=WMIN, vmax=WMAX)
    ax.set_title(t, fontsize=9)
    ax.axis("off")
fig.suptitle("%s  axial/coronal/sagittal reformat + thin-slab MIP - 3D compressed into 2D (4 images, not hundreds)" % CASE, fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fb = os.path.join(FIG, "show_reformat_%s.png" % CASE)
fig.savefig(fb, dpi=90, bbox_inches="tight")
plt.close(fig)

print("\n图A:", fa)
print("图B:", fb)
