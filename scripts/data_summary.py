"""Data summary table computed from the processed tiles (every number is measured, none quoted).

Run: python -I scripts/data_summary.py   -> results/data_summary.csv and results/data_summary.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from landslide import channels, data  # noqa: E402

LICENCE = {  # from the Zenodo records / READMEs read in verification_log.md
    "l4s": "CC BY-4.0 (Zenodo 10463239)", "hrgldd": "CC BY-4.0 (Zenodo 7189381)", "gdcld": "CC BY-4.0 (Zenodo 13612636)",
    "lmhld": "CC BY-4.0 (Zenodo 17258777)", "cas": "CC BY-NC-4.0 (Zenodo 10294997)",
}
CITE = {
    "l4s": "Ghorbanzadeh et al. 2022 (arXiv 2206.00515)", "hrgldd": "ESSD 15, 3283 (2023); author list not captured, UNVERIFIED",
    "gdcld": "Fang et al. 2024, ESSD 16, 4817", "lmhld": "Liu et al. 2025, arXiv 2502.19866",
    "cas": "Xu et al. 2024, Sci Data 11, 12",
}
rows = []
for name in data.available_datasets():
    imgs, msks, meta = data.load(name)
    pos = 0
    for a in range(0, len(msks), 2000):
        pos += int(np.asarray(msks[a:a + 2000]).sum())
    tot = msks.shape[0] * msks.shape[1] * msks.shape[2]
    groups = sorted(set(meta.group) - {"unknown"})
    partial = (meta.group == "unknown").any() and bool(groups)      # GDCLD: only the 852 test crops carry an event
    rows.append(dict(
        dataset=name, tiles=len(meta), tile_px=f"{imgs.shape[2]}x{imgs.shape[3]}", channels=imgs.shape[1], dtype=str(imgs.dtype),
        bank_channels=",".join(channels.available(name)),
        official_split="/".join(f"{(meta.official_split == s).sum()}" for s in ("train", "val", "test")) + (" (none; own group split)" if (meta.official_split == "none").all() else ""),
        pos_pixel_ratio_pct=round(100 * pos / tot, 3), tiles_with_landslide_pct=round(100 * float((meta.pos_ratio > 0).mean()), 2),
        n_events_or_regions=(f"{len(groups)} (test crops only; train/val unknown)" if partial else len(groups)) if groups else "unknown (not exposed)", licence=LICENCE.get(name, ""), citation=CITE.get(name, "")))
df = pd.DataFrame(rows)
out = ROOT / "results"
out.mkdir(exist_ok=True)
df.to_csv(out / "data_summary.csv", index=False)
(out / "data_summary.md").write_text(df.to_markdown(index=False) if hasattr(df, "to_markdown") else df.to_string(index=False), encoding="utf-8")
print(df.to_string(index=False))

