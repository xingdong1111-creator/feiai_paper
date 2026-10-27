"""M1 数据集构建: 裁 1mm 重采样的 48^3 patch, 输出 blind/consensus 二分类数据集。

流程(按 case 分组, 每 case 只重采样一次):
  1. 读 data/meta/nodules.csv, 只保留 blind(n_annotations==1, label=0) 与
     consensus(n_annotations==4, label=1), 剔除 2/3。
  2. 仅处理 data/raw/images 里已下载的 case(试点); --all 时不过滤(未来全量)。
  3. 每个 case: 重采样到 1x1x1 mm; 对每个结节用 CSV 质心(体素索引, 对应 array[z,y,x])
     转物理坐标再转重采样索引, 裁 48^3 patch(越界用 -1000 填充)。
  4. 写 data/processed/M1/patches/*.npz + manifest.csv(含 patient 划分 split)。

划分: 按 patient 70/15/15, 固定 seed, 杜绝泄漏。
"""
import argparse
import csv
import os
import numpy as np
import SimpleITK as sitk

BASE = "/public/home/liuhuan/workspace_xd/feiai_paper"
RAW = os.path.join(BASE, "data", "raw")
META = os.path.join(BASE, "data", "meta", "nodules.csv")
OUT = os.path.join(BASE, "data", "processed", "M1")
PATCH_DIR = os.path.join(OUT, "patches")
PATCH = 48
FILL_HU = -1000
SEED = 42


def resample_to_spacing(img, spacing=(1.0, 1.0, 1.0)):
    orig_sp = np.array(img.GetSpacing())
    orig_sz = np.array(img.GetSize())
    new_sz = np.ceil(orig_sz * orig_sp / np.array(spacing)).astype(int)
    rf = sitk.ResampleImageFilter()
    rf.SetOutputSpacing(spacing)
    rf.SetSize(new_sz.tolist())
    rf.SetOutputOrigin(img.GetOrigin())
    rf.SetOutputDirection(img.GetDirection())
    rf.SetInterpolator(sitk.sitkLinear)
    rf.SetDefaultPixelValue(FILL_HU)
    return rf.Execute(img)


def crop_patch(arr, cz, cy, cx, size=PATCH, fill=FILL_HU):
    """arr: (Z,Y,X); cz,cy,cx 为重采样后的体素中心索引。越界用 fill 填充。"""
    h = size // 2
    Z, Y, X = arr.shape
    out = np.full((size, size, size), fill, dtype=np.int16)
    z0, y0, x0 = int(round(cz)) - h, int(round(cy)) - h, int(round(cx)) - h
    for ax, (s, S) in enumerate(((z0, Z), (y0, Y), (x0, X))):
        pass
    # 计算源/目标重叠区间
    sz0, sz1 = max(z0, 0), min(z0 + size, Z)
    sy0, sy1 = max(y0, 0), min(y0 + size, Y)
    sx0, sx1 = max(x0, 0), min(x0 + size, X)
    if sz0 >= sz1 or sy0 >= sy1 or sx0 >= sx1:
        return out
    dz0, dy0, dx0 = sz0 - z0, sy0 - y0, sx0 - x0
    out[dz0:dz0 + (sz1 - sz0), dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = \
        arr[sz0:sz1, sy0:sy1, sx0:sx1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="不过滤已下载, 处理全部 case")
    args = ap.parse_args()

    os.makedirs(PATCH_DIR, exist_ok=True)
    rows = [r for r in csv.DictReader(open(META))
            if r["n_annotations"] in ("1", "4")]
    if not args.all:
        rows = [r for r in rows if os.path.exists(
            os.path.join(RAW, "images", r["case_id"] + ".nii.gz"))]

    # 按 case 分组
    by_case = {}
    for r in rows:
        by_case.setdefault(r["case_id"], []).append(r)

    # patient 划分
    patients = sorted({r["patient_id"] for r in rows})
    rng = np.random.RandomState(SEED)
    rng.shuffle(patients)
    n = len(patients)
    n_tr = int(0.7 * n)
    n_va = int(0.15 * n)
    split_of = {}
    for i, p in enumerate(patients):
        split_of[p] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")

    manifest = []
    for cid, nods in by_case.items():
        img = sitk.ReadImage(os.path.join(RAW, "images", cid + ".nii.gz"))
        res = resample_to_spacing(img)
        arr = sitk.GetArrayFromImage(res)  # (Z,Y,X)
        for nd in nods:
            # CSV 质心为原体素索引 (x,y,z) -> 物理点 -> 重采样索引
            phys = img.TransformContinuousIndexToPhysicalPoint(
                (float(nd["centroid_x"]), float(nd["centroid_y"]), float(nd["centroid_z"])))
            idx = res.TransformPhysicalPointToContinuousIndex(phys)
            cx, cy, cz = idx
            patch = crop_patch(arr, cz, cy, cx)
            label = 0 if nd["n_annotations"] == "1" else 1
            fname = "%s_nod%s.npz" % (cid, nd["nodule_id"])
            np.savez_compressed(os.path.join(PATCH_DIR, fname), patch=patch)
            manifest.append({
                "patch_file": fname,
                "case_id": cid,
                "patient_id": nd["patient_id"],
                "nodule_id": nd["nodule_id"],
                "label": label,
                "category": "blind" if label == 0 else "consensus",
                "split": split_of[nd["patient_id"]],
                "diameter_mm": nd["diameter_mm"],
                "subtlety_score": nd["subtlety_score"],
                "malignancy_score": nd["malignancy_score"],
            })
        print("case %s: %d 结节" % (cid, len(nods)), flush=True)

    fields = ["patch_file", "case_id", "patient_id", "nodule_id", "label",
              "category", "split", "diameter_mm", "subtlety_score", "malignancy_score"]
    with open(os.path.join(OUT, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(manifest)

    from collections import Counter
    print("\n总样本:", len(manifest))
    print("标签分布:", Counter(m["label"] for m in manifest))
    print("split 分布:", Counter(m["split"] for m in manifest))
    print("split x label:", Counter((m["split"], m["label"]) for m in manifest))


if __name__ == "__main__":
    main()
