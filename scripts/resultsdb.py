"""Single access point to all run results, used by the paper generators (numbers, tables, figures).

R[(family, method, dataset)] -> {seed: test metrics dict}   (metrics are fractions, e.g. iou_landslide = 0.24)
INFO[(family, method, dataset)][seed] -> run-level info (params, gflops, best_val_iou, train_seconds, teacher block, run_id)

families:  strict   B-strict (event-isolated leave-one-dataset-out), held-out dataset = `dataset`
           plainB   plain leave-one-dataset-out (events may overlap)      [HR-GLDD fold excluded: wrong channel order]
           sized    plain B with the training set cut to the B-strict size
           inds     in-dataset (protocol A); `method` encodes input and model, e.g. 'l4s14_unet'
methods:   unet, seg (SegFormer-B0 RGB), adp (SegFormer-B0 + band adapter), kd (distilled SegFormer-B0 student)
Excluded on purpose: results/runs_superseded/* (wrong HR-GLDD order); plain-B HR-GLDD fold.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
DS = ["l4s", "hrgldd", "gdcld", "lmhld", "cas"]

NAME_MAP = {
    "xd_Bstrict_unet": ("strict", "unet"), "xd_Bstrict_fixbgr_unet": ("strict", "unet"),
    "xd_Bstrict_segformer_b0": ("strict", "seg"), "xd_Bstrict_fixbgr_segformer_b0": ("strict", "seg"),
    "xd_Bstrict_adapter_segformer_b0": ("strict", "adp"), "xd_Bstrict_kd_segformer_b0": ("strict", "kd"),
    "xd_B_segformer_b0": ("plainB", "seg"), "xd_Bsized_segformer_b0": ("sized", "seg"),
    "baseline_l4s_unet": ("inds", "l4s14_unet"), "baseline_l4s_deeplabv3plus": ("inds", "l4s14_deeplab"),
    "baseline_l4s_segformer_b0": ("inds", "l4s14_seg"),
    "inds_l4s_rgb_unet": ("inds", "l4srgb_unet"), "inds_l4s_rgb_segformer_b0": ("inds", "l4srgb_seg"),
    "inds_l4s_bank_adapter_segformer_b0": ("inds", "l4sbank_adp"),
    "baseline_gdcld_unet": ("inds", "gdcld_unet"), "baseline_gdcld_deeplabv3plus": ("inds", "gdcld_deeplab"),
    "baseline_gdcld_segformer_b0": ("inds", "gdcld_seg"), "baseline_gdcld_segformer_b0_e60": ("inds", "gdcld_seg60"),
}


def _clean(name: str) -> str:
    return re.sub(r"__holdout_\w+$", "", name)      # adapter configs carry the fold inside `name`


def load():
    R, INFO = {}, {}
    for f in sorted(RUNS.glob("*/result.json")):
        r = json.loads(f.read_text())
        name = _clean(r["config"]["name"])
        if name not in NAME_MAP:
            continue
        fam, meth = NAME_MAP[name]
        ds = r["config"]["fold"].replace("holdout_", "") if fam != "inds" else r["config"]["fold"]
        if fam == "plainB" and ds == "hrgldd":
            continue
        key = (fam, meth, ds)
        R.setdefault(key, {})[r["seed"]] = r["test"][ds]
        INFO.setdefault(key, {})[r["seed"]] = dict(
            run_id=r["run_id"], n_params=r["n_params"], gflops=r["gflops_at_input"], input_hw=r["input_hw"],
            best_val_iou=r["best_val_iou"], best_epoch=r["best_epoch"], train_seconds=r["train_seconds"],
            n_train=r["n_train"], n_test=r["n_test"], teacher=r.get("teacher"), code_revision=r["code_revision"][:7],
            device=r["device"], torch=r["torch"], epochs=r["config"].get("epochs"), by_group=r["test"][ds].get("by_group"),
            reliability=r["test"][ds].get("reliability"), n_trainable=r.get("n_trainable"))
    return R, INFO


def mean_std(vals):
    v = np.asarray(list(vals), float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else None), len(v)


def per_seed_all(R, fam, meth, key="iou_landslide"):
    """Per-seed mean over the five datasets, only for seeds present in all five folds -> list of (seed, value*100)."""
    seeds = set.intersection(*[set(R.get((fam, meth, d), {})) for d in DS]) if all((fam, meth, d) in R for d in DS) else set()
    return [(s, 100 * float(np.mean([R[(fam, meth, d)][s][key] for d in DS]))) for s in sorted(seeds)]
