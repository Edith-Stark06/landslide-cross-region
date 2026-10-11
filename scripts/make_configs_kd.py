"""Configs for step B/C: SegFormer-B3 teacher (LoRA rank r, trained in the same job) distilled into the SegFormer-B0 student, B-strict, RGB.

The student settings are identical to the plain RGB SegFormer-B0 baseline (xd_Bstrict_segformer_b0): the only difference is the KD term.
Usage: python -I scripts/make_configs_kd.py   -> configs/xd_Bstrict_kd_segformer_b0__<fold>.yaml, jobs_kd_a.json (2 folds), jobs_kd_b.json (3 folds)
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def job(fold, rank=8, name="xd_Bstrict_kd_segformer_b0"):
    cfg = dict(name=name, protocol="protocol_B_strict", fold=fold, model="segformer_b0", encoder="resnet34", mode="rgb", pretrained=True,
               epochs=10, batch_size=32, lr=2e-4, weight_decay=0.01, crop=128, max_val=3000, num_workers=3, seed=0,
               teacher_model="segformer_b3", teacher_lora_rank=rank, teacher_epochs=10, teacher_lr=2e-4, kd_alpha=0.5, kd_temp=2.0)
    p = ROOT / "configs" / f"{name}__{fold}.yaml"
    p.write_text("\n".join(f"{k}: {str(v).lower() if isinstance(v, bool) else v}" for k, v in cfg.items()) + "\n")
    return dict(config=f"configs/{p.name}", seeds=[0], timeout_s=4200)


a = [job("holdout_l4s"), job("holdout_hrgldd")]
b = [job("holdout_gdcld"), job("holdout_lmhld"), job("holdout_cas")]
(ROOT / "configs" / "jobs_kd_a.json").write_text(json.dumps(a, indent=1) + "\n")
(ROOT / "configs" / "jobs_kd_b.json").write_text(json.dumps(b, indent=1) + "\n")
print(len(a), "+", len(b), "kd jobs")
