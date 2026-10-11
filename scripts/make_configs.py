"""Generate YAML configs and a jobs file for the cross-dataset protocols.

python -I scripts/make_configs.py B-strict segformer_b0 unet  -> configs/xd_Bstrict_<model>_<fold>.yaml + configs/jobs_Bstrict_<model>.json
All cross-dataset runs: RGB input (the only mode every dataset supports), random 128 px crops at native scale, 10 epochs, test = every
tile of the held-out dataset at its native size.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDS = ["holdout_l4s", "holdout_hrgldd", "holdout_gdcld", "holdout_lmhld", "holdout_cas"]
LR = {"segformer_b0": 2e-4, "unet": 3e-4}
WD = {"segformer_b0": 0.01, "unet": 1e-4}


def make(protocol: str, model: str, seeds=(0,), epochs=10, timeout=2400):
    tag = protocol.replace("protocol_", "").replace("_", "")
    jobs = []
    for fold in FOLDS:
        name = f"xd_{tag}_{model}"
        cfg = dict(name=name, protocol=protocol, fold=fold, model=model, encoder="resnet34", mode="rgb", pretrained=True, epochs=epochs,
                   batch_size=32, lr=LR[model], weight_decay=WD[model], crop=128, max_val=3000, num_workers=3, seed=0)
        path = ROOT / "configs" / f"{name}__{fold}.yaml"
        path.write_text("\n".join(f"{k}: {str(v).lower() if isinstance(v, bool) else v}" for k, v in cfg.items()) + "\n")
        jobs.append(dict(config=f"configs/{path.name}", seeds=list(seeds), timeout_s=timeout))
    (ROOT / "configs" / f"jobs_{tag}_{model}.json").write_text(json.dumps(jobs, indent=1) + "\n")
    return jobs


if __name__ == "__main__":
    proto = {"B": "protocol_B", "B-strict": "protocol_B_strict"}[sys.argv[1]]
    for m in sys.argv[2:]:
        print(m, len(make(proto, m)), "jobs")

