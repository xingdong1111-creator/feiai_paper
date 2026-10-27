"""Path B 推理: 整例CT -> GroundingVLM 生成 JSON(坐标作输出) -> 解析 -> 坐标映射回原图切片 -> 可视化。
未训练时生成是乱的; 训练后应输出规范 JSON。--demo 用 LIDC 真值 JSON 验证 parse+映射+viz 通路(无需模型/ONE)。
"""
import argparse
import csv
import json
import os
import re
import sys
import numpy as np
sys.path.insert(0, "/public/home/liuhuan/workspace_xd/feiai_paper")
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

ROOT = "/public/home/liuhuan/workspace_xd/feiai_paper"
NA_COLOR = {1: "red", 4: "lime"}


def _to1000(v):
    """坐标可能是 0-1000 整数或 0-1 浮点, 统一成 0-1000。"""
    v = float(v)
    return v * 1000 if abs(v) <= 1.5 else v


def normalize_item(d):
    c = d.get("coord", d.get("coords", d.get("box", [0, 0, 0])))
    if not isinstance(c, (list, tuple)) or len(c) < 3:
        return None
    return {"coord": [_to1000(c[0]), _to1000(c[1]), _to1000(c[2])],
            "r": float(d.get("r", d.get("radius", 0)) or 0),
            "na": int(d.get("na", d.get("n_annotations", 0)) or 0),
            "sub": float(d.get("sub", d.get("subtlety", 0)) or 0)}


def parse_pred(text):
    """鲁棒解析生成文本 -> list[{coord,r,na,sub}]。先 JSON, 再正则兜底抓坐标三元组。"""
    t = text.strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    s = t.find("["); e = t.rfind("]")
    if s >= 0 and e > s:
        try:
            obj = json.loads(t[s:e + 1])
            if isinstance(obj, list):
                items = [normalize_item(d) for d in obj if isinstance(d, dict)]
                items = [x for x in items if x]
                if items:
                    return items
        except Exception:
            pass
    out = []
    for m in re.finditer(r"\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]", t):
        out.append({"coord": [_to1000(m.group(1)), _to1000(m.group(2)), _to1000(m.group(3))],
                    "r": 0, "na": 0, "sub": 0})
    return out


def coord_to_voxel(coord, size):
    X, Y, Z = size
    return coord[0] / 1000 * X, coord[1] / 1000 * Y, coord[2] / 1000 * Z


def render_predictions(case, preds, out_png, raw_dir=os.path.join(ROOT, "data/raw/images")):
    img = sitk.ReadImage(os.path.join(raw_dir, case + ".nii.gz"))
    arr = sitk.GetArrayFromImage(img)                      # (Z,Y,X) HU
    X, Y, Z = img.GetSize(); sx, sy, sz = img.GetSpacing()
    n = max(1, len(preds))
    cols = min(4, n); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.8 * rows), squeeze=False)
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]; ax.axis("off")
        if i >= len(preds):
            continue
        p = preds[i]
        vx, vy, vz = coord_to_voxel(p["coord"], (X, Y, Z))
        zi = int(np.clip(round(vz), 0, Z - 1)); sl = arr[zi]
        ax.imshow(sl, cmap="gray", vmin=-1000, vmax=300)
        rv = max(3.0, (p["r"] / sx) if p["r"] > 0 else 4.0)
        col = NA_COLOR.get(p["na"], "orange")
        ax.add_patch(Circle((vx, vy), rv, fill=False, edgecolor=col, linewidth=2))
        ax.plot([vx - rv - 3, vx + rv + 3], [vy, vy], color=col, lw=0.8)
        ax.plot([vx, vx], [vy - rv - 3, vy + rv + 3], color=col, lw=0.8)
        ax.set_title("nod%d z=%d (x=%.0f,y=%.0f) r=%.1fmm na=%d sub=%.0f"
                     % (i + 1, zi, vx, vy, p["r"], p["na"], p["sub"]), fontsize=8, color=col)
    fig.suptitle("%s — %d predicted nodule(s)  [red=na1/blind, green=na4/consensus]" % (case, len(preds)), fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=110, bbox_inches="tight"); plt.close(fig)
    return out_png


def gt_preds_from_csv(case, meta_csv=os.path.join(ROOT, "data/meta/nodules.csv")):
    """从 LIDC 真值构造 preds(demo/对照用)。"""
    r = sitk.ImageFileReader()
    r.SetFileName(os.path.join(ROOT, "data/raw/images", case + ".nii.gz")); r.ReadImageInformation()
    X, Y, Z = r.GetSize()
    rows = sorted([x for x in csv.DictReader(open(meta_csv)) if x["case_id"] == case],
                  key=lambda t: int(t["nodule_id"]))
    preds = []
    for t in rows:
        preds.append({"coord": [float(t["centroid_x"]) / X * 1000, float(t["centroid_y"]) / Y * 1000,
                                float(t["centroid_z"]) / Z * 1000],
                      "r": float(t["diameter_mm"]) / 2.0, "na": int(t["n_annotations"]),
                      "sub": float(t["subtlety_score"] or 0)})
    return preds


def run_model(case, ckpt, one_dir, model_kw):
    """真实推理: 载入模型+ckpt, 从 ONE 体积生成 JSON, 解析。"""
    import torch
    from src.vlm.grounding import build_groundingvlm
    from src.vlm.grounding_data import PROMPT
    model = build_groundingvlm(model_kw["llm_path"], img_grid=tuple(model_kw["grid"]),
                               patch=tuple(model_kw["patch"]), dim=model_kw["dim"], depth=model_kw["depth"],
                               heads=model_kw["heads"], vis_grid=tuple(model_kw["vis_grid"]),
                               lora_r=model_kw["lora_r"], lora_alpha=model_kw["lora_alpha"])
    if ckpt and os.path.exists(ckpt):
        sd = torch.load(ckpt, map_location="cpu")
        model.vit.load_state_dict(sd["vit"]); model.pool.load_state_dict(sd["pool"])
        model.projector.load_state_dict(sd["projector"]); model.llm.load_state_dict(sd["lora"], strict=False)
        print("loaded ckpt:", ckpt, "(epoch %s)" % sd.get("epoch"))
    model.eval()
    vol = torch.from_numpy(np.load(os.path.join(one_dir, "volumes", case + ".npz"))["volume"]
                           .astype(np.float32)).unsqueeze(0).unsqueeze(0)
    tok = model.tokenizer
    p_ids = tok(PROMPT, add_special_tokens=True).input_ids
    gen = model.generate(vol, torch.tensor([p_ids]), torch.ones(1, len(p_ids), dtype=torch.long),
                         max_new_tokens=320, do_sample=False)
    text = tok.decode(gen[0], skip_special_tokens=True)
    print("=== 生成文本 ===\n" + text)
    return parse_pred(text), text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="LIDC-IDRI-0003_3480003264")
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "runs/grounding/best_grounding.pt"))
    ap.add_argument("--one-dir", default=os.path.join(ROOT, "data/processed/ONE"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--demo", action="store_true", help="用 LIDC 真值验证 parse+映射+viz(不调模型)")
    ap.add_argument("--llm-path", default=os.path.join(ROOT, "models/Qwen2.5-1.5B-Instruct"))
    ap.add_argument("--grid", type=int, nargs=3, default=[128, 192, 192])
    ap.add_argument("--patch", type=int, nargs=3, default=[16, 16, 16])
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--vis-grid", type=int, nargs=3, default=[4, 6, 6])
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    args = ap.parse_args()

    if args.demo:
        preds = gt_preds_from_csv(args.case)
        tag = "demo_gt"
    else:
        mk = {k: getattr(args, k) for k in ("llm_path", "grid", "patch", "dim", "depth", "heads", "vis_grid", "lora_r", "lora_alpha")}
        preds, _ = run_model(args.case, args.ckpt, args.one_dir, mk)
        tag = "pred"
    out = args.out or os.path.join(ROOT, "data/figs", "grounding_%s_%s.png" % (tag, args.case))
    render_predictions(args.case, preds, out)
    print("结节数:", len(preds), "-> 图:", out)


if __name__ == "__main__":
    main()
