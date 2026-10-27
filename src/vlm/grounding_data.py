"""Path B grounding 数据: prompt 不含坐标(坐标是【输出】), 目标 JSON 含坐标(归一化0-1000)+r+na+sub。
结节按 nodule_id 排序成固定顺序做 LM 监督。batch 复用 stage2_data.collate_stage2(键一致)。
"""
import csv
import json
import os
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset

PROMPT = ("You are a lung-nodule detector. Given the encoded chest-CT volume, detect all pulmonary "
          "nodules. Output a JSON array; each item has coord [x,y,z] (each value normalized to 0-1000), "
          "r (radius in mm), na (number of radiologists who found it, 1-4), sub (subtlety 1-5). "
          "Answer:")


def _clip1000(v):
    return max(0, min(1000, int(round(v))))


def build_target(rows):
    """目标 JSON: 每结节 {coord:[x,y,z](0-1000), r, na, sub}, 按 nodule_id 升序(固定顺序)。"""
    rows = sorted(rows, key=lambda r: int(r["nodule_id"]))
    items = []
    for r in rows:
        coord = [_clip1000(float(r["nx"]) * 1000), _clip1000(float(r["ny"]) * 1000),
                 _clip1000(float(r["nz"]) * 1000)]
        try:
            rad = round(float(r["diameter_mm"]) / 2.0, 1)
        except (ValueError, TypeError):
            rad = 0.0
        try:
            na = int(r["n_annotations"])
        except (ValueError, TypeError):
            na = 1
        try:
            sub = int(float(r["subtlety_score"]))
        except (ValueError, TypeError):
            sub = 3
        items.append({"coord": coord, "r": rad, "na": na, "sub": sub})
    return json.dumps(items, separators=(",", ":"))


class GroundingDataset(Dataset):
    def __init__(self, one_dir, split, tokenizer, max_len=320, augment=False, seed=0):
        self.vol_dir = os.path.join(one_dir, "volumes")
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment = augment          # grounding 下翻转会使坐标失配, 默认关闭
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

    def _load_volume(self, cid):
        vol = np.load(os.path.join(self.vol_dir, cid + ".npz"))["volume"].astype(np.float32)
        return torch.from_numpy(vol).unsqueeze(0)

    def __getitem__(self, i):
        cid = self.cases[i]
        rows = self.by_case[cid]
        vol = self._load_volume(cid)
        target = build_target(rows)
        p_ids = self.tokenizer(PROMPT, add_special_tokens=True).input_ids
        t_ids = self.tokenizer(target, add_special_tokens=False).input_ids + [self.tokenizer.eos_token_id]
        input_ids = (p_ids + t_ids)[:self.max_len]
        labels = ([-100] * len(p_ids) + t_ids)[:self.max_len]
        attn = [1] * len(input_ids)
        return {"volume": vol,
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "case_id": cid}
