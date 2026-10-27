"""阶段2 数据: 把"整例体积 + 该例结节坐标"构造成 prompt, 把"per-nodule JSON"作为目标,
tokenize 后视觉 token 前置(在 QwenVLM.forward 里做), LM loss 只算目标 JSON 段。
一次输入(整例) -> 一次输出(该例所有结节的 JSON)。
"""
import csv
import json
import os
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset

PROMPT_TMPL = (
    "You are a lung-nodule analyst. Given the encoded chest-CT volume and the normalized "
    "coordinates of {n} nodules, output a JSON array; each item has id, category "
    "(blind|consensus), subtlety (1-5). Coordinates (x,y,z): {coords}\nAnswer:"
)


def build_target(rows):
    """目标 JSON: 每结节 {id, category, subtlety}(用 LIDC 真值)。"""
    items = []
    for i, r in enumerate(rows, 1):
        items.append({
            "id": i,
            "category": "consensus" if r["n_annotations"] == "4" else "blind",
            "subtlety": int(float(r["subtlety_score"])) if r.get("subtlety_score") else 3,
        })
    return json.dumps(items, ensure_ascii=False)


def build_prompt(rows):
    coords = ", ".join("(%.2f,%.2f,%.2f)" % (float(r["nx"]), float(r["ny"]), float(r["nz"])) for r in rows)
    return PROMPT_TMPL.format(n=len(rows), coords=coords)


class Stage2Dataset(Dataset):
    def __init__(self, one_dir, split, tokenizer, max_len=256, augment=False, seed=0):
        self.vol_dir = os.path.join(one_dir, "volumes")
        self.tokenizer = tokenizer
        self.max_len = max_len
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

    def _load_volume(self, cid):
        vol = np.load(os.path.join(self.vol_dir, cid + ".npz"))["volume"].astype(np.float32)
        if self.augment:
            for ax in (0, 1, 2):
                if self.rng.rand() < 0.5:
                    vol = np.flip(vol, axis=ax).copy()
            k = self.rng.randint(4)
            if k:
                vol = np.rot90(vol, k, axes=(1, 2)).copy()
        return torch.from_numpy(vol).unsqueeze(0)

    def __getitem__(self, i):
        cid = self.cases[i]
        rows = self.by_case[cid]
        # 注意: 增强会翻转体积, 坐标需同步; 阶段2 先不做坐标-体积一致性增强(仅整体翻转对坐标影响小, 保守起见 augment=False 训练)
        vol = self._load_volume(cid)
        prompt = build_prompt(rows)
        target = build_target(rows)
        p_ids = self.tokenizer(prompt, add_special_tokens=True).input_ids
        t_ids = self.tokenizer(target, add_special_tokens=False).input_ids + [self.tokenizer.eos_token_id]
        input_ids = (p_ids + t_ids)[:self.max_len]
        labels = ([-100] * len(p_ids) + t_ids)[:self.max_len]
        attn = [1] * len(input_ids)
        return {"volume": vol,
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "case_id": cid}


def collate_stage2(batch, pad_id):
    vols = torch.stack([b["volume"] for b in batch])
    Lmax = max(b["input_ids"].shape[0] for b in batch)
    B = len(batch)
    ids = torch.full((B, Lmax), pad_id, dtype=torch.long)
    attn = torch.zeros(B, Lmax, dtype=torch.long)
    labs = torch.full((B, Lmax), -100, dtype=torch.long)
    for bi, b in enumerate(batch):
        n = b["input_ids"].shape[0]
        ids[bi, :n] = b["input_ids"]
        attn[bi, :n] = b["attention_mask"]
        labs[bi, :n] = b["labels"]
    return {"volume": vols, "input_ids": ids, "attention_mask": attn, "labels": labs,
            "case_ids": [b["case_id"] for b in batch]}


# ------------------------- last-hidden-state 读出模式 -------------------------
PROMPT_PREFIX = ("You are a lung-nodule analyst. Given the encoded chest-CT volume and the "
                 "normalized coordinates of nodules, predict for each nodule whether it is "
                 "easy for radiologists to miss. Coordinates (x,y,z): ")
PROMPT_SUFFIX = " Predict per nodule."


def build_prompt_with_spans(rows):
    """返回 (prompt 字符串, 每个结节坐标的字符区间 list[(start,end)])。"""
    prompt = PROMPT_PREFIX
    spans = []
    for r in rows:
        s = "(%.2f,%.2f,%.2f); " % (float(r["nx"]), float(r["ny"]), float(r["nz"]))
        start = len(prompt)
        prompt += s
        spans.append((start, len(prompt)))
    prompt += PROMPT_SUFFIX
    return prompt, spans


def _readout_positions(offsets, spans):
    """每个结节坐标字符区间 -> 该区间内最后一个 token 的下标（读出位置）。"""
    pos = []
    for (cs, ce) in spans:
        idxs = [i for i, (ts, te) in enumerate(offsets) if ts >= cs and te <= ce and te > ts]
        pos.append(idxs[-1] if idxs else max(0, len(offsets) - 1))
    return pos


class Stage2HeadDataset(Dataset):
    """读出模式: 返回 input_ids + 每个结节的读出 token 位置 + 标签(类别/隐蔽度)。"""

    def __init__(self, one_dir, split, tokenizer, max_len=192):
        self.vol_dir = os.path.join(one_dir, "volumes")
        self.tokenizer = tokenizer
        self.max_len = max_len
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
        rows = self.by_case[cid]
        vol = torch.from_numpy(np.load(os.path.join(self.vol_dir, cid + ".npz"))["volume"]
                               .astype(np.float32)).unsqueeze(0)
        prompt, spans = build_prompt_with_spans(rows)
        enc = self.tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False,
                             truncation=True, max_length=self.max_len)
        readout = _readout_positions(enc["offset_mapping"], spans)
        lab, subt, valid = [], [], []
        for r in rows:
            na = int(r["n_annotations"])
            c = 1 if na == 1 else 0 if na == 4 else -100
            lab.append(c); valid.append(c != -100)
            try: subt.append(float(r["subtlety_score"]))
            except (ValueError, TypeError): subt.append(0.0)
        return {"volume": vol,
                "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
                "readout_idx": torch.tensor(readout, dtype=torch.long),
                "labels": torch.tensor(lab, dtype=torch.long),
                "subtlety": torch.tensor(subt, dtype=torch.float32),
                "valid": torch.tensor(valid, dtype=torch.bool),
                "case_id": cid}


def collate_head(batch, pad_id):
    """体积 stack; 文本右 padding 到 Lmax(读出位置不变); 读出/标签/valid padding 到 Nmax。"""
    vols = torch.stack([b["volume"] for b in batch])
    Lmax = max(b["input_ids"].shape[0] for b in batch)
    Nmax = max(b["labels"].shape[0] for b in batch)
    B = len(batch)
    ids = torch.full((B, Lmax), pad_id, dtype=torch.long)
    attn = torch.zeros(B, Lmax, dtype=torch.long)
    readout = torch.zeros(B, Nmax, dtype=torch.long)
    labs = torch.full((B, Nmax), -100, dtype=torch.long)
    subt = torch.zeros(B, Nmax)
    valid = torch.zeros(B, Nmax, dtype=torch.bool)
    for bi, b in enumerate(batch):
        L = b["input_ids"].shape[0]; N = b["labels"].shape[0]
        ids[bi, :L] = b["input_ids"]; attn[bi, :L] = 1
        readout[bi, :N] = b["readout_idx"].clamp(max=L - 1)
        labs[bi, :N] = b["labels"]; subt[bi, :N] = b["subtlety"]; valid[bi, :N] = b["valid"]
    return {"volume": vols, "input_ids": ids, "attention_mask": attn, "readout_idx": readout,
            "labels": labs, "subtlety": subt, "valid": valid,
            "case_ids": [b["case_id"] for b in batch]}
