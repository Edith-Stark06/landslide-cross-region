"""Collect results/runs/*/result.json into results/tables/runs.csv and a mean +- std summary (per config name and fold).

Run: python -I scripts/aggregate.py
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
rows = []
for f in sorted((ROOT / "results" / "runs").glob("*/result.json")):
    r = json.loads(f.read_text())
    for ds, m in r["test"].items():
        m = {k: v for k, v in m.items() if k != "by_group"}
        rows.append(dict(run=r["run_id"], name=re.sub(r"__holdout_\w+$", "", r["config"]["name"]), model=r["config"]["model"], fold=r["config"]["fold"], seed=r["seed"],
                         test_dataset=ds, mIoU=100 * m["miou"], IoU_landslide=100 * m["iou_landslide"], F1=100 * m["f1"],
                         precision=100 * m["precision"], recall=100 * m["recall"], PR_AUC=m["pr_auc"], ECE=m["ece"],
                         params_M=r["n_params"] / 1e6, GFLOPs=r["gflops_at_input"], input_px=r["input_hw"], train_s=r["train_seconds"],
                         n_test=r["n_test"], best_epoch=r["best_epoch"], code_revision=r["code_revision"][:7], device=r["device"]))
df = pd.DataFrame(rows)
out = ROOT / "results" / "tables"
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "runs.csv", index=False)
metrics = ["mIoU", "IoU_landslide", "F1", "precision", "recall", "PR_AUC", "ECE"]
g = df.groupby(["name", "test_dataset"])[metrics]
summ = g.mean().round(3).join(g.std().round(3), rsuffix="_std").join(df.groupby(["name", "test_dataset"]).seed.nunique().rename("n_seeds"))
summ.to_csv(out / "summary.csv")
print(summ.to_string())
print("\nstd is NaN where n_seeds == 1")
