"""M1 数据集: 读取 build_dataset.py 产出的 patch(.npz) + manifest.csv。

mode='3d': 返回 (1,48,48,48) 全 patch。
mode='2d': 返回 (3,48,48) 中央 3 个轴向切片(2.5D 对照基线)。
归一化: HU clip[-1000,400] -> [0,1]。
增强(仅 train): 随机轴向翻转 + 随机 90° 旋转。
"""
import csv
import os
import numpy as np
import torch
from torch.utils.data import Dataset

HU_MIN, HU_MAX = -1000.0, 400.0


def _norm(p):
    p = np.clip(p, HU_MIN, HU_MAX)
    return ((p - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)


def _augment(p, rng):
    # p: (Z,Y,X)
    for ax in range(3):
        if rng.rand() < 0.5:
            p = np.flip(p, axis=ax).copy()
    k = rng.randint(0, 4)
    if k:
        p = np.rot90(p, k, axes=(1, 2)).copy()   # 轴向平面旋转
    return p


class BlindConsensusDataset(Dataset):
    def __init__(self, manifest_csv, split, mode="3d", augment=False, seed=0):
        assert mode in ("3d", "2d")
        self.mode = mode
        self.augment = augment
        self.rng = np.random.RandomState(seed)
        self.patch_dir = os.path.join(os.path.dirname(os.path.abspath(manifest_csv)), "patches")
        self.rows = [r for r in csv.DictReader(open(manifest_csv)) if r["split"] == split]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        p = np.load(os.path.join(self.patch_dir, r["patch_file"]))["patch"]
        if self.augment:
            p = _augment(p, self.rng)
        p = _norm(p)
        if self.mode == "3d":
            x = torch.from_numpy(p).unsqueeze(0)              # (1,Z,Y,X)
        else:
            z = p.shape[0] // 2
            x = torch.from_numpy(p[z - 1:z + 2])              # (3,Y,X) 中央3切片
        y = torch.tensor(int(r["label"]), dtype=torch.long)
        return x, y

    def meta(self, i):
        return self.rows[i]
