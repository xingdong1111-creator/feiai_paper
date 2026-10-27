"""从 hf-mirror 下载 CT 影像与标注计数 mask。
默认读 data/sample_cases.csv 的 case_id(去重, 试点);
--full 时改读 data/meta/nodules.csv 的全部 case_id(全量 ~1000 例)。
下载:
  images/{case_id}.nii.gz
  masks_annotation_count/{case_id}.nii.gz
支持断点续传(已存在且非空则跳过)。
"""
import csv
import os
import subprocess
import sys

BASE = "/public/home/liuhuan/workspace_xd/feiai_paper"
CASES = os.path.join(BASE, "data", "sample_cases.csv")
FULL = os.path.join(BASE, "data", "meta", "nodules.csv")
RAW = os.path.join(BASE, "data", "raw")
ENDPOINT = "https://hf-mirror.com/datasets/MedOtter/LIDC-IDRI/resolve/main"

SUBDIRS = ["images", "masks_annotation_count"]
for s in SUBDIRS:
    os.makedirs(os.path.join(RAW, s), exist_ok=True)

# 读取并去重 case_id
seen, case_ids = set(), []
src = FULL if "--full" in sys.argv else CASES
for r in csv.DictReader(open(src)):
    cid = r["case_id"]
    if cid and cid not in seen:
        seen.add(cid)
        case_ids.append(cid)

print("待下载病例数(去重): %d" % len(case_ids), flush=True)

def fetch(remote_rel, local_path):
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return "SKIP", os.path.getsize(local_path)
    url = "%s/%s" % (ENDPOINT, remote_rel)
    cmd = ["curl", "-sS", "-L", "--connect-timeout", "20",
           "--retry", "3", "-o", local_path, url]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        return "FAIL", p.stderr[:120]
    sz = os.path.getsize(local_path) if os.path.exists(local_path) else 0
    if sz == 0:
        return "EMPTY", 0
    return "OK", sz

ok, fail = 0, 0
for i, cid in enumerate(case_ids, 1):
    for s in SUBDIRS:
        remote = "%s/%s.nii.gz" % (s, cid)
        local = os.path.join(RAW, s, "%s.nii.gz" % cid)
        status, info = fetch(remote, local)
        if status in ("OK", "SKIP"):
            ok += 1
            mb = info / 1e6 if isinstance(info, int) else 0
            print("[%d/%d] %-22s %-8s %.2fMB" % (i, len(case_ids), s, status, mb), flush=True)
        else:
            fail += 1
            print("[%d/%d] %-22s %-8s %s" % (i, len(case_ids), s, status, info), flush=True)

print("\n完成: 成功 %d, 失败 %d" % (ok, fail), flush=True)
sys.exit(1 if fail else 0)
