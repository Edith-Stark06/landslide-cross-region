"""Configs for step A of the method stage: the band adapter.

jobs_adapter_xd.json  : SegFormer-B0 + band adapter (mode bank), protocol B-strict, 5 folds, same settings as the RGB SegFormer-B0 baseline
                        (xd_Bstrict_segformer_b0), so the ONLY difference is the adapter + the extra bands.
jobs_l4s_refs.json    : in-dataset L4S references: RGB-only SegFormer-B0 and U-Net, and SegFormer-B0 + adapter on the full bank
                        (the existing 14-band runs used native input, the RGB-only ones give the like-for-like gap for the cross-dataset table).
Run: python -I scripts/make_configs_adapter.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write(name, **cfg):
    p = ROOT / "configs" / f"{name}.yaml"
    cfg = dict(name=name, **cfg)
    p.write_text("\n".join(f"{k}: {str(v).lower() if isinstance(v, bool) else v}" for k, v in cfg.items()) + "\n")
    return p.name


xd = []
for fold in ("holdout_l4s", "holdout_hrgldd", "holdout_gdcld", "holdout_lmhld", "holdout_cas"):
    n = write(f"xd_Bstrict_adapter_segformer_b0__{fold}", protocol="protocol_B_strict", fold=fold, model="adapter_segformer_b0", mode="bank",
              pretrained=True, epochs=10, batch_size=32, lr=2e-4, weight_decay=0.01, crop=128, max_val=3000, num_workers=3, seed=0)
    xd.append(dict(config=f"configs/{n}", seeds=[0], timeout_s=2400))
refs = []
for model, lr, wd, mode in (("segformer_b0", 2e-4, 0.01, "rgb"), ("unet", 3e-4, 1e-4, "rgb"), ("adapter_segformer_b0", 2e-4, 0.01, "bank")):
    n = write(f"inds_l4s_{mode}_{model}", protocol="protocol_A", fold="l4s", model=model, encoder="resnet34", mode=mode, pretrained=True,
              epochs=50, batch_size=32, lr=lr, weight_decay=wd, num_workers=2, seed=0)
    refs.append(dict(config=f"configs/{n}", seeds=[0], timeout_s=1800))
(ROOT / "configs" / "jobs_adapter_xd.json").write_text(json.dumps(xd, indent=1) + "\n")
(ROOT / "configs" / "jobs_l4s_refs.json").write_text(json.dumps(refs, indent=1) + "\n")
print(len(xd), "adapter xd jobs;", len(refs), "l4s reference jobs")
