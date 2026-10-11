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
NAMES = {"l4s": "L4S", "hrgldd": "HR-GLDD", "gdcld": "GDCLD", "lmhld": "LMHLD", "cas": "CAS"}


def load_all(name, fold):
    """All seeds of one (method, fold) as {seed: result}. Adapter configs carry the fold in their name too, so both layouts are tried."""
    out = {}
    for pat in (f"{name}__{fold}__seed*", f"{name}__{fold}__{fold}__seed*"):
        for d in RUNS.glob(pat):
            if (d / "result.json").exists():
                r = json.loads((d / "result.json").read_text())
                out[r["seed"]] = r
    return out


def strict_mean(name_for):
    """(mean over seeds of the 5-dataset mean IoU, std over seeds, n seeds with all 5 folds, per-dataset seed means) or None."""
    per_ds = {d: load_all(*name_for(d)) for d in DS}
    seeds = set.intersection(*[set(v) for v in per_ds.values()]) if all(per_ds.values()) else set()
    if not seeds:
        return None
    per_seed = [np.mean([100 * per_ds[d][s]["test"][d]["iou_landslide"] for d in DS]) for s in sorted(seeds)]
    return float(np.mean(per_seed)), (float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0), len(per_seed)


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
    ax.errorbar(ms, s[0], yerr=s[1], fmt="none", ecolor="gray", capsize=3, lw=1)
    ax.scatter(ms, s[0], s=40 + 8 * pm, label=f"{label} ({pm:.1f} M, n={s[2]} seeds)", zorder=3)
    ax.annotate(f"{s[0]:.1f}", (ms, s[0]), textcoords="offset points", xytext=(8, 4), fontsize=8)
ax.set_xlabel("CPU latency, 128 px, 4 threads (ms, median; i5-1135G7)")
ax.set_ylabel("Mean B-strict landslide IoU over 5 held-out datasets")
ax.set_title("Accuracy vs. latency (mean ± std over seeds where n>1)", fontsize=9)
ax.legend(fontsize=7, loc="best")
ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(FIG / "fig_pareto.png", dpi=200); plt.close(fig)
print("wrote fig_pareto.png")

# ---------------------------------------------------------------- reliability
REL_METHODS = [("U-Net R34", "xd_Bstrict_unet", "xd_Bstrict_fixbgr_unet"), ("SegFormer-B0 (RGB)", "xd_Bstrict_segformer_b0", "xd_Bstrict_fixbgr_segformer_b0"),
               ("SegFormer-B0 + adapter", "xd_Bstrict_adapter_segformer_b0", "xd_Bstrict_adapter_segformer_b0")]
panels = {}
for label, nm, nm_hr in REL_METHODS:
    for d in DS:
        runs_ = load_all(nm_hr if d == "hrgldd" else nm, f"holdout_{d}")
        for seed, r in runs_.items():
            if "reliability" in r["test"][d]:
                panels.setdefault(d, {}).setdefault(label, []).append((seed, r["test"][d]["reliability"], r["test"][d]["ece"]))
if panels:
    fig, axes = plt.subplots(1, len(DS), figsize=(3.0 * len(DS), 3.1), squeeze=False)
    for ax, d in zip(axes[0], DS):
        ax.plot([0.5, 1], [0.5, 1], "k--", lw=0.8, label="perfect")
        for label, items in panels.get(d, {}).items():
            seed, rb, ece = sorted(items, key=lambda t: t[0])[0]                      # lowest seed that stored bins (seed 1)
            pts = [(c, a) for c, a in zip(rb["confidence"], rb["accuracy"]) if c is not None]
            ax.plot(*zip(*pts), "o-", ms=2.5, lw=1, label=f"{label} (ECE {ece:.3f})")
        ax.set_title(f"held-out: {NAMES.get(d, d)}", fontsize=8)
        ax.set_xlabel("confidence (winning class)", fontsize=7); ax.set_ylabel("accuracy", fontsize=7)
        ax.legend(fontsize=5, loc="upper left")
    fig.suptitle("Reliability under B-strict (one seed per curve; pixels in 15 confidence bins)", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig_reliability.png", dpi=200); plt.close(fig)
    print("wrote fig_reliability.png for", sorted(panels))
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
