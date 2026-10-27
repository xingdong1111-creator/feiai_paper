import csv
from collections import Counter

p = "/public/home/liuhuan/workspace_xd/feiai_paper/data/meta/nodules.csv"
rows = list(csv.DictReader(open(p)))
n = len(rows)
print("nodules.csv 结节总数:", n)

c = Counter(r["n_annotations"] for r in rows)
print('按"发现该结节的医生人数"(n_annotations)分布:')
for k in sorted(c, key=lambda x: int(x) if x.isdigit() else 99):
    print("  %s 位医生发现: %d 个 (%.1f%%)" % (k, c[k], 100.0 * c[k] / n))

sub = Counter(r.get("subtlety_score", "") for r in rows)
print("subtlety_score(隐蔽度)分布:")
for k in sorted(sub):
    label = k if k else "NA"
    print("  %s: %d" % (label, sub[k]))

blind = [r for r in rows if r["n_annotations"] == "1"]
print("\n盲区结节(仅1位医生发现)共 %d 个, 样例:" % len(blind))
for r in blind[:8]:
    dia = float(r["diameter_mm"]) if r["diameter_mm"] else 0.0
    print("  %s nod%s: 直径%.1fmm 恶性度%s 隐蔽度%s" % (
        r["patient_id"], r["nodule_id"], dia,
        r["malignancy_score"], r["subtlety_score"]))

# 盲区里有多少其实是高恶性度(危险漏诊)
danger = [r for r in blind if r["malignancy_score"].isdigit() and int(r["malignancy_score"]) >= 4]
print("\n盲区结节中恶性度>=4(高危却被多数医生漏掉)的: %d 个" % len(danger))
for r in danger[:8]:
    print("  %s nod%s: 直径%.1fmm 恶性度%s" % (
        r["patient_id"], r["nodule_id"], float(r["diameter_mm"]), r["malignancy_score"]))
