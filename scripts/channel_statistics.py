"""Per-channel mean reflectance of HR-GLDD and of LMHLD scenes whose channel order is documented (B,G,R,NIR), used in the paper's
discussion of the HR-GLDD band order. Needs the processed data. Writes paper/generated/channel_stats.json.
Run: python -I scripts/channel_statistics.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from landslide import data  # noqa: E402

out = {}
x, _, m = data.load("hrgldd")
a = np.asarray(x[::5], dtype=np.float64).mean(axis=(0, 2, 3))
out["hrgldd"] = dict(channels="array channels 0..3 (documented order unknown)", means=[float(v) for v in a], n_tiles=int(len(x[::5])))
x, _, m = data.load("lmhld")
SENS = {"Hokkaido_Japan": "PlanetScope", "Taiwan_China": "PlanetScope", "Emilia-Romagna_Italy": "Sentinel-2", "Gorkha_Nepal": "Sentinel-2",
        "Jiuzhaigou_China": "Gaofen-2", "Wenchuan_China": "IKONOS", "Rio de Janeiro_Brazil": "RapidEye"}
out["lmhld"] = {}
for ev, sensor in SENS.items():
    idx = np.where(m.group == ev)[0][::10]
    a = np.asarray(x[idx], dtype=np.float64).mean(axis=(0, 2, 3))
    out["lmhld"][ev] = dict(sensor=sensor, means_BGRNIR=[float(v) for v in a], n_tiles=int(len(idx)))
(ROOT / "paper" / "generated").mkdir(parents=True, exist_ok=True)
(ROOT / "paper" / "generated" / "channel_stats.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1)[:1500])
