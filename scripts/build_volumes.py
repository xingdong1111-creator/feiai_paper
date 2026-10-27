"""预处理: 把整例 CT 重采样到固定网格存 npz, 并算每结节在网格里的归一化坐标。
one-in-one-out 的数据基础: 一例 = 一个固定尺寸体积 + 一组结节(坐标+标签)。

输出:
  data/processed/ONE/volumes/{case_id}.npz   (volume: float16 (D,H,W), 已 HU->[0,1])
  data/processed/ONE/cases_manifest.csv      (每结节一行: case/patient/nodule/nx,ny,nz/valid/label/...)
默认只处理 data/raw/images 里已下载的 case; 加 --all 处理 nodules.csv 全部 case。
"""
import argparse
import csv
import os
from collections import defaultdict
import numpy as np
import SimpleITK as sitk

BASE = "/public/home/liuhuan/workspace_xd/feiai_paper"
RAW = os.path.join(BASE, "data", "raw")
NODULES = os.path.join(BASE, "data", "meta", "nodules.csv")
OUT_DIR = os.path.join(BASE, "data", "processed", "ONE")
FILL_HU = -1000
HU_LO, HU_HI = -1000, 400


def crop_pad_axis(arr, axis, target, fill):
    """把 arr 在 axis 上裁剪/零填充到 target 长度; 返回新数组和 off=(src_len-target)//2。"""
    src = arr.shape[axis]
    off = (src - target) // 2
    shape = list(arr.shape); shape[axis] = target
    out = np.full(shape, fill, dtype=arr.dtype)
    s0, s1 = max(0, off), min(src, off + target)     # 源区间
    d0 = max(0, -off)                                  # 目标起点
    if s1 > s0:
        sl_src = [slice(None)] * arr.ndim
        sl_dst = [slice(None)] * arr.ndim
        sl_src[axis] = slice(s0, s1)
        sl_dst[axis] = slice(d0, d0 + (s1 - s0))
        out[tuple(sl_dst)] = arr[tuple(sl_src)]
    return out, off


def resample(img, spacing):
    orig_sp = np.array(img.GetSpacing())
    orig_sz = np.array(img.GetSize())
    new_sz = np.ceil(orig_sz * orig_sp / np.array(spacing)).astype(int).tolist()
    rf = sitk.ResampleImageFilter()
    rf.SetOutputSpacing([float(s) for s in spacing])
    rf.SetSize(new_sz)
    rf.SetOutputOrigin(img.GetOrigin())
    rf.SetOutputDirection(img.GetDirection())
    rf.SetInterpolator(sitk.sitkLinear)
    rf.SetDefaultPixelValue(FILL_HU)
    return rf.Execute(img)


def to_fixed_grid(arr, grid, fill=FILL_HU):
    """arr (Z,Y,X) -> 裁剪/零填充到 grid (D,H,W); 返回 (out, off_zyx)。"""
    off = []
    for ax, tgt in enumerate(grid):
        arr, o = crop_pad_axis(arr, ax, tgt, fill)
        off.append(o)
    return arr, tuple(off)   # off 按 (z,y,x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spacing", type=float, nargs=3, default=[2.0, 2.0, 2.0])
    ap.add_argument("--grid", type=int, nargs=3, default=[128, 192, 192], help="(D,H,W)=(z,y,x)")
    ap.add_argument("--all", action="store_true", help="处理 nodules.csv 全部 case(否则仅已下载)")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--limit", type=int, default=0, help="仅处理前 N 例(冒烟用, 0=全部)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--val-size", type=float, default=0.15)
    args = ap.parse_args()
    grid = tuple(args.grid)   # (D,H,W) = (z,y,x)

    out_dir = args.out_dir
    os.makedirs(os.path.join(out_dir, "volumes"), exist_ok=True)
    rows = [r for r in csv.DictReader(open(NODULES))]
    rows = [r for r in rows if r["n_annotations"] in ("1", "4")]
    by_case = defaultdict(list)
    for r in rows:
        by_case[r["case_id"]].append(r)

    if args.all:
        cases = sorted(by_case)
    else:
        have = {f[:-7] for f in os.listdir(os.path.join(RAW, "images")) if f.endswith(".nii.gz")}
        cases = sorted(c for c in by_case if c in have)
    if args.limit:
        cases = cases[:args.limit]

    # 按 patient 划分
    patients = sorted({r["patient_id"] for c in cases for r in by_case[c]})
    rng = np.random.RandomState(args.seed)
    rng.shuffle(patients)
    n = len(patients)
    n_test = int(round(n * args.test_size))
    n_val = int(round(n * args.val_size))
    test_p, val_p = set(patients[:n_test]), set(patients[n_test:n_test + n_val])
    def split_of(p):
        return "test" if p in test_p else "val" if p in val_p else "train"

    manifest = []
    n_ok = n_skip = 0
    for i, cid in enumerate(cases, 1):
        ip = os.path.join(RAW, "images", cid + ".nii.gz")
        if not os.path.exists(ip):
            n_skip += 1
            continue
        img = sitk.ReadImage(ip)
        res = resample(img, args.spacing)
        arr = sitk.GetArrayFromImage(res).astype(np.float32)   # (Z,Y,X)
        arr = np.clip(arr, HU_LO, HU_HI)
        vol, off_zyx = to_fixed_grid(arr, grid)                # (D,H,W)
        vol = ((vol - HU_LO) / (HU_HI - HU_LO)).astype(np.float16)
        np.savez_compressed(os.path.join(out_dir, "volumes", cid + ".npz"), volume=vol)

        D, H, W = grid   # z,y,x
        for r in by_case[cid]:
            cx, cy, cz = float(r["centroid_x"]), float(r["centroid_y"]), float(r["centroid_z"])
            phys = img.TransformContinuousIndexToPhysicalPoint((cx, cy, cz))
            rx, ry, rz = res.TransformPhysicalPointToContinuousIndex(phys)   # 重采样后连续索引 (x,y,z)
            oz, oy, ox = off_zyx
            gx, gy, gz = rx - ox, ry - oy, rz - oz       # 固定网格连续索引 (x,y,z)
            valid = int(0 <= gx < W and 0 <= gy < H and 0 <= gz < D)
            nx = (gx + 0.5) / W; ny = (gy + 0.5) / H; nz = (gz + 0.5) / D
            manifest.append({
                "case_id": cid, "patient_id": r["patient_id"], "nodule_id": r["nodule_id"],
                "nx": round(nx, 6), "ny": round(ny, 6), "nz": round(nz, 6), "valid": valid,
                "label": 1 if r["n_annotations"] == "4" else 0,   # 1=consensus 0=blind
                "category": "consensus" if r["n_annotations"] == "4" else "blind",
                "n_annotations": r["n_annotations"],
                "diameter_mm": r["diameter_mm"],
                "subtlety_score": r["subtlety_score"],
                "malignancy_score": r["malignancy_score"],
                "split": split_of(r["patient_id"]),
            })
        n_ok += 1
        if i % 50 == 0 or i == len(cases):
            print("[%d/%d] done=%d skip=%d nodules=%d" % (i, len(cases), n_ok, n_skip, len(manifest)), flush=True)

    cols = list(manifest[0].keys())
    mp = os.path.join(out_dir, "cases_manifest.csv")
    with open(mp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(manifest)
    valid_nod = sum(m["valid"] for m in manifest)
    print("\n病例处理=%d 跳过=%d | 结节=%d (网格内=%d) -> %s" % (n_ok, n_skip, len(manifest), valid_nod, mp))
    print("网格=%s spacing=%s | 每例 npz ≈ %.1f MB" % (grid, tuple(args.spacing),
          grid[0]*grid[1]*grid[2]*2/1e6))


if __name__ == "__main__":
    main()
