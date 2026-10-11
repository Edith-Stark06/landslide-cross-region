"""Multi-seed summary of the headline B-strict comparisons: per fold (mean +- std over seeds, n) and the 5-fold mean per seed.

Run: python -I scripts/seed_summary.py -> results/tables/seed_summary.md
Std is the sample std (ddof=1) over seeds; with n=3 it is itself very noisy. 'paired' lines compare methods fold by fold on seed-averaged IoU.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
DS = ["l4s", "hrgldd", "gdcld", "lmhld", "cas"]
R = {}   # (method, fold) -> {seed: metrics dict}
for f in sorted((ROOT / "results" / "runs").glob("*/result.json")):
    r = json.loads(f.read_text())
    name = re.sub(r"__holdout_\w+$", "", r["config"]["name"])
    ds = r["config"]["fold"].replace("holdout_", "")
    R.setdefault((name, ds), {})[r["seed"]] = r["test"][ds]

METHODS = {"U-Net R34": ("xd_Bstrict_unet", "xd_Bstrict_fixbgr_unet"),
           "SegFormer-B0 (RGB)": ("xd_Bstrict_segformer_b0", "xd_Bstrict_fixbgr_segformer_b0"),
           "SegFormer-B0 + band adapter": ("xd_Bstrict_adapter_segformer_b0",) * 2,
           "SegFormer-B0 student (KD)": ("xd_Bstrict_kd_segformer_b0",) * 2}


def runs(method, ds):
    a, b = METHODS[method]
    return R.get((b if ds == "hrgldd" else a, ds), {})


def fmt(v, nd=1):
    v = np.asarray(v, float)
    return f"{v.mean():.{nd}f} ± {v.std(ddof=1):.{nd}f} (n={len(v)})" if len(v) > 1 else f"{v.mean():.{nd}f} (n=1)"


out = []
for key, label, scale, nd in [("iou_landslide", "landslide IoU", 100, 1), ("f1", "F1", 100, 1), ("ece", "ECE", 1, 3)]:
    out.append(f"### B-strict {label}: mean ± std over seeds (n = number of seeds)\n")
    out.append("| Method | " + " | ".join(DS) + " | 5-dataset mean per seed |")
    out.append("|---|" + "---|" * (len(DS) + 1))
    for m in METHODS:
        cells, per_seed = [], {}
        for d in DS:
            rs = runs(m, d)
            cells.append(fmt([scale * x[key] for x in rs.values()], nd) if rs else "n/a")
            for s, x in rs.items():
                per_seed.setdefault(s, []).append(scale * x[key])
        full = [np.mean(v) for v in per_seed.values() if len(v) == len(DS)]
        out.append(f"| {m} | " + " | ".join(cells) + f" | {fmt(full, nd) if full else 'n/a (not all folds)'} |")
    out.append("")

out.append("### Paired comparisons on landslide IoU (seed-averaged per fold; difference = A - B)\n")
for a, b in [("SegFormer-B0 (RGB)", "U-Net R34"), ("SegFormer-B0 + band adapter", "SegFormer-B0 (RGB)"), ("SegFormer-B0 student (KD)", "SegFormer-B0 (RGB)")]:
    diffs, parts = [], []
    for d in DS:
        ra, rb = runs(a, d), runs(b, d)
        if not ra or not rb:
            continue
        da, db = np.mean([100 * x["iou_landslide"] for x in ra.values()]), np.mean([100 * x["iou_landslide"] for x in rb.values()])
        diffs.append(da - db); parts.append(f"{d} {da - db:+.1f} (n={len(ra)},{len(rb)})")
    if diffs:
        out.append(f"- **{a} − {b}**: " + "; ".join(parts) + f"; mean {np.mean(diffs):+.1f}, positive in {sum(x > 0 for x in diffs)}/{len(diffs)} folds")
out.append("\nWith 5 folds the smallest possible two-sided Wilcoxon p-value is 0.0625, so a paired test over folds cannot reach p < 0.05; we report "
           "differences and sign counts, not significance claims.")

# distilled-student reruns (seed 1) vs seed 0
out.append("\n### Distilled student: seed 0 vs seed 1 on the two folds that collapsed (landslide IoU)\n")
for d in ("l4s", "cas"):
    rs = runs("SegFormer-B0 student (KD)", d)
    out.append(f"- {d}: " + ", ".join(f"seed {s}: {100 * x['iou_landslide']:.1f}" for s, x in sorted(rs.items())))
(ROOT / "results" / "tables" / "seed_summary.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
