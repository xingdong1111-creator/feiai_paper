"""选择 20 例代表性样本用于数据对齐验证。
策略: 覆盖"人眼盲区"(n_annotations=1)与"共识结节"(n_annotations=4)两类,
并优先纳入高恶性度的盲区结节(临床最危险的漏诊)。
输出 data/sample_cases.csv 供下载脚本使用。
"""
import csv
import os

BASE = "/public/home/liuhuan/workspace_xd/feiai_paper"
SRC = os.path.join(BASE, "data", "meta", "nodules.csv")
OUT_DIR = os.path.join(BASE, "data")
OUT = os.path.join(OUT_DIR, "sample_cases.csv")
os.makedirs(OUT_DIR, exist_ok=True)

rows = list(csv.DictReader(open(SRC)))

def to_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default

# 盲区: n_annotations == 1; 按恶性度降序(优先高危漏诊)
blind = [r for r in rows if r["n_annotations"] == "1"]
blind.sort(key=lambda r: to_int(r["malignancy_score"]), reverse=True)
# 共识: n_annotations == 4; 按恶性度降序
consensus = [r for r in rows if r["n_annotations"] == "4"]
consensus.sort(key=lambda r: to_int(r["malignancy_score"]), reverse=True)

# 各选 10 例, 尽量来自不同 patient
def pick_unique(patients_pool, k):
    seen, out = set(), []
    for r in patients_pool:
        pid = r["patient_id"]
        if pid in seen:
            continue
        seen.add(pid)
        out.append(r)
        if len(out) >= k:
            break
    return out

sel_blind = pick_unique(blind, 10)
sel_cons = pick_unique(consensus, 10)

fields = ["case_id", "patient_id", "nodule_id", "category",
          "n_annotations", "malignancy_score", "subtlety_score",
          "diameter_mm", "centroid_x", "centroid_y", "centroid_z"]

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in sel_blind:
        r2 = {k: r.get(k, "") for k in fields}
        r2["category"] = "blind"
        w.writerow(r2)
    for r in sel_cons:
        r2 = {k: r.get(k, "") for k in fields}
        r2["category"] = "consensus"
        w.writerow(r2)

print("已写出", OUT)
print("盲区样本 %d 例, 共识样本 %d 例" % (len(sel_blind), len(sel_cons)))
print("\n盲区样本(高危漏诊优先):")
for r in sel_blind:
    print("  %s nod%s 恶性度%s 直径%.1fmm" % (
        r["patient_id"], r["nodule_id"], r["malignancy_score"],
        float(r["diameter_mm"]) if r["diameter_mm"] else 0))
print("\n共识样本:")
for r in sel_cons[:5]:
    print("  %s nod%s 恶性度%s" % (r["patient_id"], r["nodule_id"], r["malignancy_score"]))
