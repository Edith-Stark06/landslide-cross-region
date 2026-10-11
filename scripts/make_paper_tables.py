"""Build the paper's results tables (markdown) from results/runs/*/result.json. Every number in the tables is read from a run file.

Run: python -I scripts/make_paper_tables.py  ->  results/tables/paper_tables.md
Cells without a run print 'n/a'. Cells with >1 seed print mean +- std (n); single-seed cells print the value and '(n=1)'.
"""
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = {}
for f in sorted((ROOT / "results" / "runs").glob("*/result.json")):
    r = json.loads(f.read_text())
    # the adapter configs carry the fold inside `name` (xd_..._adapter_...__holdout_X); strip it so name identifies the method only
    RUNS.setdefault((re.sub(r"__holdout_\w+$", "", r["config"]["name"]), r["config"]["fold"]), []).append(r)
DS = ["l4s", "hrgldd", "gdcld", "lmhld", "cas"]
NAMES = {"l4s": "L4S", "hrgldd": "HR-GLDD", "gdcld": "GDCLD", "lmhld": "LMHLD", "cas": "CAS"}


def cell(name, fold, ds, key, scale=100.0, nd=1):
    rs = RUNS.get((name, fold))
    if not rs:
        return "n/a"
    v = [scale * r["test"][ds][key] for r in rs]
    return f"{np.mean(v):.{nd}f}" + (f" ± {np.std(v, ddof=1):.{nd}f} (n={len(v)})" if len(v) > 1 else "")


out = []
# T1: in-dataset baselines
out.append("### Table: in-dataset baselines (protocol A; official splits; test = official test set)\n")
out.append("| Dataset | Input | Model | Params (M) | GFLOPs | mIoU | Landslide IoU | F1 | Precision | Recall | PR-AUC | ECE |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
for name, ds, inp, label in [
    ("baseline_l4s_unet", "l4s", "14 bands", "U-Net R34"), ("baseline_l4s_deeplabv3plus", "l4s", "14 bands", "DeepLabV3+ R34"),
    ("baseline_l4s_segformer_b0", "l4s", "14 bands", "SegFormer-B0"),
    ("inds_l4s_rgb_unet", "l4s", "RGB", "U-Net R34"), ("inds_l4s_rgb_segformer_b0", "l4s", "RGB", "SegFormer-B0"),
    ("inds_l4s_bank_adapter_segformer_b0", "l4s", "bank (adapter)", "SegFormer-B0 + adapter"),
    ("baseline_gdcld_unet", "gdcld", "RGB", "U-Net R34"), ("baseline_gdcld_deeplabv3plus", "gdcld", "RGB", "DeepLabV3+ R34"),
    ("baseline_gdcld_segformer_b0", "gdcld", "RGB", "SegFormer-B0"), ("baseline_gdcld_segformer_b0_e60", "gdcld", "RGB", "SegFormer-B0 (60 ep)")]:
    rs = RUNS.get((name, ds))
    if not rs:
        continue
    r = rs[0]
    out.append(f"| {NAMES[ds]} | {inp} | {label} | {r['n_params'] / 1e6:.1f} | {r['gflops_at_input']:.1f} | "
               + " | ".join(cell(name, ds, ds, k) for k in ("miou", "iou_landslide", "f1", "precision", "recall"))
               + f" | {cell(name, ds, ds, 'pr_auc', 1, 3)} | {cell(name, ds, ds, 'ece', 1, 3)} |")

# T2: cross-dataset (B-strict), by model
def strict_name(model, ds):
    base = {"segformer_b0": "xd_Bstrict_segformer_b0", "unet": "xd_Bstrict_unet", "adapter": "xd_Bstrict_adapter_segformer_b0",
            "student": "xd_Bstrict_kd_segformer_b0"}[model]
    return (base.replace("xd_Bstrict_", "xd_Bstrict_fixbgr_"), f"holdout_{ds}") if ds == "hrgldd" and model in ("segformer_b0", "unet") else (base, f"holdout_{ds}")


for metric, label, scale, nd in [("iou_landslide", "landslide IoU", 100, 1), ("f1", "F1", 100, 1), ("ece", "ECE", 1, 3)]:
    out.append(f"\n### Table: cross-dataset, protocol B-strict (event-isolated leave-one-dataset-out), {label}\n")
    out.append("| Model | " + " | ".join(NAMES[d] for d in DS) + " |")
    out.append("|---|" + "---|" * len(DS))
    for model, lab in [("unet", "U-Net R34"), ("segformer_b0", "SegFormer-B0"), ("adapter", "SegFormer-B0 + band adapter"),
                       ("student", "SegFormer-B0 student (distilled)")]:
        out.append(f"| {lab} | " + " | ".join(cell(*strict_name(model, d)[:1], strict_name(model, d)[1], d, metric, scale, nd) for d in DS) + " |")

# T3: overlap control
out.append("\n### Table: effect of event overlap, SegFormer-B0, landslide IoU (protocol B vs size-matched B vs B-strict)\n")
out.append("| Held-out | Plain B (full training set) | Plain B cut to B-strict size | B-strict |")
out.append("|---|---|---|---|")
for d in DS:
    full = "n/a (old HR-GLDD channel order)" if d == "hrgldd" else cell("xd_B_segformer_b0", f"holdout_{d}", d, "iou_landslide")
    out.append(f"| {NAMES[d]} | {full} | {cell('xd_Bsized_segformer_b0', f'holdout_{d}', d, 'iou_landslide')} | "
               f"{cell(*strict_name('segformer_b0', d)[:1], strict_name('segformer_b0', d)[1], d, 'iou_landslide')} |")

# T4: teacher vs student (if present)
rows = [(k, v[0]) for k, v in RUNS.items() if k[0].startswith("xd_Bstrict_kd") and "teacher" in v[0]]
if rows:
    out.append("\n### Table: teacher (SegFormer-B3 + LoRA) vs distilled student (SegFormer-B0), B-strict, landslide IoU\n")
    out.append("| Held-out | Teacher | Student (KD) | Student (no KD) |")
    out.append("|---|---|---|---|")
    for d in DS:
        rs = RUNS.get(("xd_Bstrict_kd_segformer_b0", f"holdout_{d}"))
        if rs:
            t = np.mean([100 * r["teacher"]["test"][d]["iou_landslide"] for r in rs])
            out.append(f"| {NAMES[d]} | {t:.1f} | {cell('xd_Bstrict_kd_segformer_b0', f'holdout_{d}', d, 'iou_landslide')} | "
                       f"{cell(*strict_name('segformer_b0', d)[:1], strict_name('segformer_b0', d)[1], d, 'iou_landslide')} |")

# T5: per-region failures (size-matched SegFormer, shows local failures)
out.append("\n### Table: per-region / per-event landslide IoU, SegFormer-B0 (plain B cut to B-strict size, seed 0)\n")
for d in ("cas", "gdcld", "lmhld"):
    rs = RUNS.get(("xd_Bsized_segformer_b0", f"holdout_{d}"))
    if rs and "by_group" in rs[0]["test"][d]:
        g = rs[0]["test"][d]["by_group"]
        out.append(f"- **{NAMES[d]}**: " + "; ".join(f"{k} {100 * v['iou_landslide']:.1f}" for k, v in sorted(g.items(), key=lambda kv: kv[1]["iou_landslide"])))

(ROOT / "results" / "tables").mkdir(parents=True, exist_ok=True)
(ROOT / "results" / "tables" / "paper_tables.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
