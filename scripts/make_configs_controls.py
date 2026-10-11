"""Configs for the two follow-up experiments after the first cross-dataset round.

1. fixbgr: B-strict, held-out HR-GLDD only, SegFormer-B0 and U-Net, with the corrected HR-GLDD channel order (B,G,R,NIR in channels.py).
2. sized: plain protocol B (event overlap allowed) with the training set randomly cut to the SAME number of tiles as B-strict uses for that
   fold. Comparing 'sized' with B-strict isolates the effect of event overlap from the effect of having less training data.
Run: python -I scripts/make_configs_controls.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LR = {"segformer_b0": 2e-4, "unet": 3e-4}
WD = {"segformer_b0": 0.01, "unet": 1e-4}
base = dict(encoder="resnet34", mode="rgb", pretrained=True, epochs=10, batch_size=32, crop=128, max_val=3000, num_workers=3, seed=0)


def write(name, fold, protocol, model, **extra):
    cfg = dict(name=name, protocol=protocol, fold=fold, model=model, **base, lr=LR[model], weight_decay=WD[model], **extra)
    p = ROOT / "configs" / f"{name}__{fold}.yaml"
    p.write_text("\n".join(f"{k}: {str(v).lower() if isinstance(v, bool) else v}" for k, v in cfg.items()) + "\n")
    return dict(config=f"configs/{p.name}", seeds=[0], timeout_s=2400)


strict = json.loads((ROOT / "splits" / "protocol_B_strict.json").read_text())
jobs_fix = [write(f"xd_Bstrict_fixbgr_{m}", "holdout_hrgldd", "protocol_B_strict", m) for m in ("segformer_b0", "unet")]
jobs_sized = [write("xd_Bsized_segformer_b0", fold, "protocol_B", "segformer_b0", max_train=len(strict[fold]["train"]))
              for fold in ("holdout_l4s", "holdout_hrgldd", "holdout_gdcld", "holdout_lmhld", "holdout_cas")]
(ROOT / "configs" / "jobs_fixbgr.json").write_text(json.dumps(jobs_fix, indent=1) + "\n")
(ROOT / "configs" / "jobs_sized.json").write_text(json.dumps(jobs_sized, indent=1) + "\n")
print(len(jobs_fix), "fixbgr jobs;", len(jobs_sized), "sized jobs;", [(j["config"], ) for j in jobs_sized][:1])
