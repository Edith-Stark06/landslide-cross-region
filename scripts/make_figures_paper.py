"""Two additional manuscript figures from the saved runs (seed means with per-seed points).

fig_strict_bars.png : B-strict landslide IoU per held-out dataset for the four methods, mean over seeds with individual seeds as dots.
fig_overlap.png     : event-overlap control (plain B / plain B cut to the B-strict size / B-strict), SegFormer-B0, seed 0, with the B-strict
                      three-seed mean +- std as a reference band.
Run: python -I scripts/make_figures_paper.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from resultsdb import DS, ROOT, load  # noqa: E402

R, INFO = load()
FIG = ROOT / "figures"
LAB = {"l4s": "L4S", "hrgldd": "HR-GLDD", "gdcld": "GDCLD", "lmhld": "LMHLD", "cas": "CAS"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# ---------------------------------------------------------------- B-strict bars
methods = [("unet", "U-Net (R34)", "#8c8c8c"), ("seg", "SegFormer-B0", "#1f77b4"), ("adp", "SegFormer-B0 + band adapter", "#2ca02c"),
           ("kd", "SegFormer-B0 distilled student", "#d62728")]
fig, ax = plt.subplots(figsize=(8.2, 3.6))
w = 0.2
for j, (m, lab, col) in enumerate(methods):
    xs, ms, sds = [], [], []
    for i, d in enumerate(DS):
        v = [100 * x["iou_landslide"] for x in R.get(("strict", m, d), {}).values()]
        if not v:
            continue
        x0 = i + (j - 1.5) * w
        ax.bar(x0, np.mean(v), w * 0.95, color=col, alpha=0.85, label=lab if i == 0 else None)
        ax.scatter([x0] * len(v), v, s=9, color="k", zorder=3)
        if len(v) > 1:
            ax.errorbar(x0, np.mean(v), yerr=np.std(v, ddof=1), color="k", capsize=2, lw=0.8)
ax.set_xticks(range(len(DS))); ax.set_xticklabels([LAB[d] for d in DS])
ax.set_ylabel("landslide IoU on the held-out dataset (%)")
ax.legend(fontsize=7, frameon=False, ncol=2)
ax.set_title("B-strict: mean over seeds (bars), individual seeds (dots), sample std (error bars; n = 3, distilled student n = 1 to 2)", fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "fig_strict_bars.png", dpi=220); plt.close(fig)

# ---------------------------------------------------------------- overlap control
fig, ax = plt.subplots(figsize=(8.4, 3.6))
labs = [("plainB", "Plain B (all other datasets; events may overlap)", "#ff7f0e"), ("sized", "Plain B, size-matched", "#bcbd22"),
        ("comp", "Composition-matched (LMHLD+CAS; events may overlap)", "#9467bd"), ("strict", "B-strict (LMHLD+CAS; events removed)", "#1f77b4")]
bw = 0.2
for j, (fam, lab, col) in enumerate(labs):
    first = True
    for i, d in enumerate(DS):
        v = R.get((fam, "seg", d), {}).get(0)
        if v is None:
            continue
        x0 = i + (j - 1.5) * bw
        ax.bar(x0, 100 * v["iou_landslide"], bw * 0.95, color=col, label=lab if first else None)
        first = False
        if fam in ("comp", "strict"):                                  # three-seed mean +- std for the two conditions that have seeds
            vs = [100 * x["iou_landslide"] for x in R[(fam, "seg", d)].values()]
            if len(vs) > 1:
                ax.errorbar(x0, np.mean(vs), yerr=np.std(vs, ddof=1), fmt="D", color="k", ms=3, capsize=2, lw=0.8,
                            label="3-seed mean $\\pm$ std" if (fam == "comp" and i == 0) else None)
ax.text(1 - 1.5 * bw, 1.5, "n/a", ha="center", fontsize=7)
ax.set_xticks(range(len(DS))); ax.set_xticklabels([LAB[d] for d in DS])
ax.set_ylabel("landslide IoU (%), SegFormer-B0")
ax.set_ylim(0, 66)
ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="upper left")
ax.set_title("Event-overlap controls (bars: seed 0; diamonds: mean $\\pm$ std over seeds; plain B for HR-GLDD not reported)", fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "fig_overlap.png", dpi=220); plt.close(fig)
print("wrote fig_strict_bars.png, fig_overlap.png")
