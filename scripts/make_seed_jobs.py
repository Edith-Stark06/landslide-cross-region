"""Extra seeds for the headline B-strict comparisons, split over two kernels so both can run at once.

kernel seedsA: SegFormer-B0 RGB (seeds 1,2; HR-GLDD fold uses the corrected-order config) + U-Net (seeds 1,2)
kernel seedsB: SegFormer-B0 + band adapter (seeds 1,2) + distilled student on the two folds with the largest student/baseline gap (seed 1)
All configs get save_examples: true so the new runs also store reliability bins and worst/best example tiles.
Run: python -I scripts/make_seed_jobs.py -> configs/jobs_seeds_a.json, configs/jobs_seeds_b.json
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CF = ROOT / "configs"
FOLDS = ["l4s", "hrgldd", "gdcld", "lmhld", "cas"]


def base_cfg(model, fold):
    if model == "unet" or model == "segformer_b0":
        return f"xd_Bstrict_fixbgr_{model}__holdout_hrgldd.yaml" if fold == "hrgldd" else f"xd_Bstrict_{model}__holdout_{fold}.yaml"
    if model == "adapter":
        return f"xd_Bstrict_adapter_segformer_b0__holdout_{fold}.yaml"
    if model == "kd":
        return f"xd_Bstrict_kd_segformer_b0__holdout_{fold}.yaml"


def job(model, fold, seeds, timeout):
    p = CF / base_cfg(model, fold)
    txt = p.read_text()
    if "save_examples" not in txt:
        p.write_text(txt.rstrip("\n") + "\nsave_examples: true\n")
    return dict(config=f"configs/{p.name}", seeds=seeds, timeout_s=timeout)


a = [job("segformer_b0", f, [1, 2], 2400) for f in FOLDS] + [job("unet", f, [1, 2], 2400) for f in FOLDS]
b = [job("adapter", f, [1, 2], 2400) for f in FOLDS] + [job("kd", f, [1], 4200) for f in ("l4s", "cas")]
(CF / "jobs_seeds_a.json").write_text(json.dumps(a, indent=1) + "\n")
(CF / "jobs_seeds_b.json").write_text(json.dumps(b, indent=1) + "\n")
print(len(a), "jobs in A (", sum(len(j["seeds"]) for j in a), "runs); ", len(b), "jobs in B (", sum(len(j["seeds"]) for j in b), "runs)")
for j in a[:1] + b[-1:]:
    print(j)
