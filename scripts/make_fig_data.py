"""Figure 2: dataset overview. (a) tiles and landslide-pixel share per dataset; (b) which physical events each dataset contains
(from configs/dataset_events.json; 'inferred' event assignments are described in the file), showing why leave-one-dataset-out needs event isolation.

Run: python -I scripts/make_fig_data.py -> figures/fig_data_overview.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DS = ["l4s", "hrgldd", "gdcld", "lmhld", "cas"]
LAB = {"l4s": "L4S", "hrgldd": "HR-GLDD", "gdcld": "GDCLD", "lmhld": "LMHLD", "cas": "CAS"}
ev = json.loads((ROOT / "configs" / "dataset_events.json").read_text(encoding="utf-8"))
summ = pd.read_csv(ROOT / "results" / "data_summary.csv").set_index("dataset")


def events(d):
    return set(ev[d]["group_to_event"].values()) if ev[d]["tile_events"] else set(ev[d]["events"])


all_ev = sorted(set().union(*[events(d) for d in DS]), key=lambda e: (-sum(e in events(d) for d in DS), e))
mat = np.array([[1 if e in events(d) else 0 for d in DS] for e in all_ev])

fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 5.2), gridspec_kw=dict(width_ratios=[1, 1.25]))
x = np.arange(len(DS))
a.bar(x, [summ.loc[d, "tiles"] for d in DS], color="#4c72b0")
a.set_xticks(x); a.set_xticklabels([LAB[d] for d in DS]); a.set_ylabel("tiles after conversion")
a2 = a.twinx()
a2.plot(x, [summ.loc[d, "pos_pixel_ratio_pct"] for d in DS], "o", color="#c44e52")
a2.set_ylabel("landslide pixels (%)", color="#c44e52")
for i, d in enumerate(DS):
    a.text(i, summ.loc[d, "tiles"], f"{summ.loc[d, 'tile_px']}", ha="center", va="bottom", fontsize=7)
a.set_title("(a) size and class balance (labels: tile px)", fontsize=9)

b.imshow(mat, aspect="auto", cmap="Blues", vmin=0, vmax=1.4)
b.set_xticks(range(len(DS))); b.set_xticklabels([LAB[d] for d in DS], fontsize=8)
b.set_yticks(range(len(all_ev))); b.set_yticklabels(all_ev, fontsize=7)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        if mat[i, j]:
            tile_level = ev[DS[j]]["tile_events"]
            b.text(j, i, "tiles" if tile_level else "doc.", ha="center", va="center", fontsize=6, color="white")
b.set_title("(b) physical events per dataset ('tiles' = tiles carry the event id;\n'doc.' = event listed in the dataset paper only)", fontsize=8)
fig.tight_layout()
(ROOT / "figures").mkdir(exist_ok=True)
fig.savefig(ROOT / "figures" / "fig_data_overview.png", dpi=200)
print("events:", len(all_ev), "| events present in >=2 datasets:", int((mat.sum(1) >= 2).sum()), "| in >=3:", int((mat.sum(1) >= 3).sum()))
