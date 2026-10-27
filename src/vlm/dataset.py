"""整例 Dataset: 读 data/processed/ONE 的 npz + cases_manifest.csv。
一个样本 = 一例 CT(固定网格体积) + 该例所有结节(归一化坐标/标签)。
变长结节用 collate 填齐到 batch 内最大 N, 配 valid 掩码。
"""
import csv
import os
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset


class WholeCaseDataset(Dataset):
    def __init__(self, one_dir, split, augment=False, seed=0):
        self.vol_dir = os.path.join(one_dir, "volumes")
        self.augment = augment
        self.rng = np.random.RandomState(seed)
        rows = [r for r in csv.DictReader(open(os.path.join(one_dir, "cases_manifest.csv")))
                if r["split"] == split]
        by_case = defaultdict(list)
        for r in rows:
            by_case[r["case_id"]].append(r)
        self.cases = sorted(by_case)
        self.by_case = by_case

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, i):
        cid = self.cases[i]
        vol = np.load(os.path.join(self.vol_dir, cid + ".npz"))["volume"].astype(np.float32)  # (D,H,W)
        nod = self.by_case[cid]
        coords = np.array([[float(r["nx"]), float(r["ny"]), float(r["nz"])] for r in nod], dtype=np.float32)
        coords = np.clip(coords, 0.0, 1.0)
        labels = np.array([int(r["label"]) for r in nod], dtype=np.int64)
        valid = np.array([int(r["valid"]) for r in nod], dtype=np.float32)
        diam = np.array([float(r["diameter_mm"]) for r in nod], dtype=np.float32)

        if self.augment:
            for ax, ci in ((2, 0), (1, 1), (0, 2)):   # (z,y,x)->coord idx (x,y,z)
                if self.rng.rand() < 0.5:
                    vol = np.flip(vol, axis=ax).copy()
                    coords[:, ci] = 1.0 - coords[:, ci]
            k = self.rng.randint(4)
            if k:
                vol = np.rot90(vol, k, axes=(1, 2)).copy()   # 绕 z 旋转 (y,x)
                for _ in range(k):
                    x, y = coords[:, 0].copy(), coords[:, 1].copy()
                    coords[:, 0], coords[:, 1] = 1.0 - y, x

        return {"volume": torch.from_numpy(vol).unsqueeze(0),   # (1,D,H,W)
                "coords": torch.from_numpy(coords),              # (N,3)
                "labels": torch.from_numpy(labels),              # (N,)
                "valid": torch.from_numpy(valid),                # (N,)
                "diam": torch.from_numpy(diam),                  # (N,)
                "case_id": cid}


def collate_cases(batch):
    """体积同尺寸可堆叠; 结节填齐到 batch 内最大 N。"""
    vols = torch.stack([b["volume"] for b in batch])             # (B,1,D,H,W)
    Nmax = max(b["coords"].shape[0] for b in batch)
    B = len(batch)
    coords = torch.zeros(B, Nmax, 3)
    labels = torch.zeros(B, Nmax, dtype=torch.long)
    valid = torch.zeros(B, Nmax)
    diam = torch.zeros(B, Nmax)
    for bi, b in enumerate(batch):
        n = b["coords"].shape[0]
        coords[bi, :n] = b["coords"]
        labels[bi, :n] = b["labels"]
        valid[bi, :n] = b["valid"]
        diam[bi, :n] = b["diam"]
    return {"volume": vols, "coords": coords, "labels": labels,
            "valid": valid, "diam": diam, "case_ids": [b["case_id"] for b in batch]}
