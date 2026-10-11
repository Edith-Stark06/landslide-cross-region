"""Paper figures from saved results only.

fig_pareto.png     : mean B-strict landslide IoU (over the 5 held-out datasets, seed-0 runs available) vs CPU latency at 128 px (4 threads).
fig_reliability.png: reliability diagram(s) from runs that saved `reliability` bins (runs launched after commit 054d87a); skipped if none.
fig_failures.png   : worst / best test tiles from runs that saved examples_<dataset>.npz; skipped if none.
Run: python -I scripts/make_figures.py   -> figures/*.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
RUNS = ROOT / "results" / "runs"
DS = ["l4s", "hrgldd", "gdcld", "lmhld", "cas"]


def load(name, fold):
    f = RUNS / f"{name}__{fold}__seed0" / "result.json"
    return json.loads(f.read_text()) if f.exists() else None


def strict_mean(name_for):
    vals = []
    for d in DS:
        r = load(*name_for(d))
        if r is None:
            return None
        vals.append(100 * r["test"][d]["iou_landslide"])
    return float(np.mean(vals)), vals


# ---------------------------------------------------------------- Pareto
lat = pd.read_csv(ROOT / "results" / "tables" / "latency_cpu.csv")
def latency(model):
    row = lat[(lat.model == model) & (lat.input_px == 128) & (lat.threads == lat.threads.max())]
    return float(row.median_ms.iloc[0]), float(row.params_M.iloc[0])

cands = [
    ("U-Net R34", lambda d: ("xd_Bstrict_unet", f"holdout_{d}") if d != "hrgldd" else ("xd_Bstrict_fixbgr_unet", f"holdout_{d}")),
    ("SegFormer-B0", lambda d: ("xd_Bstrict_segformer_b0", f"holdout_{d}") if d != "hrgldd" else ("xd_Bstrict_fixbgr_segformer_b0", f"holdout_{d}")),
    ("SegFormer-B0 + band adapter", lambda d: ("xd_Bstrict_adapter_segformer_b0", f"holdout_{d}")),
    ("SegFormer-B0 student (distilled)", lambda d: ("xd_Bstrict_kd_segformer_b0", f"holdout_{d}")),
]
fig, ax = plt.subplots(figsize=(5.6, 4))
for label, fn in cands:
    s = strict_mean(fn)
    if s is None:
        continue
    ms, pm = latency("SegFormer-B0 + band adapter" if "adapter" in label else "SegFormer-B0" if "SegFormer-B0" in label else label)
    ax.scatter(ms, s[0], s=40 + 8 * pm, label=f"{label} ({pm:.1f} M)")
    ax.annotate(f"{s[0]:.1f}", (ms, s[0]), textcoords="offset points", xytext=(6, 4), fontsize=8)
ax.set_xlabel("CPU latency, 128 px, 4 threads (ms, median; i5-1135G7)")
ax.set_ylabel("Mean B-strict landslide IoU over 5 held-out datasets")
ax.set_title("Accuracy vs. latency (single seed)", fontsize=10)
ax.legend(fontsize=7, loc="best")
ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(FIG / "fig_pareto.png", dpi=200); plt.close(fig)
print("wrote fig_pareto.png")

# ---------------------------------------------------------------- reliability
rel = []
for f in sorted(RUNS.glob("*/result.json")):
    r = json.loads(f.read_text())
    for ds, m in r["test"].items():
        if "reliability" in m:
            rel.append((r["run_id"], ds, m["reliability"], m["ece"]))
if rel:
    n = min(len(rel), 6)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 2.8), squeeze=False)
    for ax, (rid, ds, rb, ece) in zip(axes[0], rel[:n]):
        pts = [(c, a) for c, a in zip(rb["confidence"], rb["accuracy"]) if c is not None]
        ax.plot([0.5, 1], [0.5, 1], "k--", lw=0.8)
        ax.plot(*zip(*pts), "o-", ms=3)
        ax.set_title(f"{rid.split('__')[0][:22]}\n{ds}, ECE {ece:.3f}", fontsize=6)
        ax.set_xlabel("confidence", fontsize=7); ax.set_ylabel("accuracy", fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "fig_reliability.png", dpi=200); plt.close(fig)
    print("wrote fig_reliability.png from", len(rel), "evaluations")
else:
    print("no reliability data yet: fig_reliability.png skipped")

# ---------------------------------------------------------------- failures
ex = sorted(RUNS.glob("*/examples_*.npz"))
if ex:
    f = ex[0]
    z = np.load(f)
    k = len(z["tile"])
    fig, axes = plt.subplots(3, k, figsize=(1.5 * k, 4.8), squeeze=False)
    for j in range(k):
        img = np.clip(z["image"][j].astype(np.float32).transpose(1, 2, 0), -2.5, 2.5)
        img = (img - img.min()) / max(img.max() - img.min(), 1e-6)
        axes[0, j].imshow(img); axes[0, j].set_title(f"IoU {z['iou'][j]:.2f}", fontsize=6)
        axes[1, j].imshow(z["mask"][j], cmap="gray", vmin=0, vmax=1)
        axes[2, j].imshow(z["prob"][j].astype(np.float32), cmap="magma", vmin=0, vmax=1)
        for i in range(3):
            axes[i, j].axis("off")
    for i, t in enumerate(["image (normalised RGB)", "ground truth", "P(landslide)"]):
        axes[i, 0].text(-0.05, 0.5, t, transform=axes[i, 0].transAxes, rotation=90, va="center", ha="right", fontsize=6)
    fig.suptitle(f.parent.name + " / " + f.stem, fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "fig_failures.png", dpi=200); plt.close(fig)
    print("wrote fig_failures.png from", f)
else:
    print("no example tiles saved yet: fig_failures.png skipped")
