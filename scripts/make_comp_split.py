"""Composition-matched control for the event-overlap experiment.

In every B-strict fold the training tiles come ONLY from LMHLD and CAS (Hokkaido 2018 is in all five datasets and L4S, HR-GLDD, GDCLD carry no
tile-level event ids, so they are dropped whole). The earlier size-matched control matched the NUMBER of tiles but drew them from all four other
datasets. This control keeps the same source datasets as B-strict (LMHLD and CAS, minus the held-out one) and the same number of training tiles,
but does NOT remove tiles that share events with the held-out dataset. B-strict vs this control therefore differs only in event overlap.

Built from the committed splits (no processed data needed):
  splits/protocol_Bcomp.json : per fold, train/val = the protocol-B train/val tiles of LMHLD and CAS only; test = the held-out dataset (as B/B-strict);
                               'target_train' = number of B-strict training tiles (runs subsample the pool to this size via max_train).
Run: python -I scripts/make_comp_split.py  -> splits/protocol_Bcomp.json, configs/xd_Bcomp_segformer_b0__*.yaml, configs/jobs_comp.json
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = {"lmhld", "cas"}
B = json.loads((ROOT / "splits" / "protocol_B.json").read_text())
S = json.loads((ROOT / "splits" / "protocol_B_strict.json").read_text())
out, jobs = {}, []
for ds in ["l4s", "hrgldd", "gdcld", "lmhld", "cas"]:
    f = f"holdout_{ds}"
    tr = [r for r in B[f]["train"] if r[0] in SRC]
    va = [r for r in B[f]["val"] if r[0] in SRC]
    target = len(S[f]["train"])
    assert {r[0] for r in S[f]["train"]} <= SRC, "B-strict training data should only come from LMHLD/CAS"
    assert len(tr) >= target, (ds, len(tr), target)
    out[f] = dict(train=tr, val=va, test=B[f]["test"], target_train=target)
    cfg = dict(name="xd_Bcomp_segformer_b0", protocol="protocol_Bcomp", fold=f, model="segformer_b0", encoder="resnet34", mode="rgb", pretrained=True,
               epochs=10, batch_size=32, lr=2e-4, weight_decay=0.01, crop=128, max_val=3000, num_workers=3, seed=0, max_train=target, save_examples=True)
    p = ROOT / "configs" / f"xd_Bcomp_segformer_b0__{f}.yaml"
    p.write_text("\n".join(f"{k}: {str(v).lower() if isinstance(v, bool) else v}" for k, v in cfg.items()) + "\n")
    jobs.append(dict(config=f"configs/{p.name}", seeds=[0, 1, 2], timeout_s=2400))
    print(f"{f:15s} pool {len(tr):6d} -> subsample to {target:6d} (pool/target = {len(tr) / target:.2f})")
(ROOT / "splits" / "protocol_Bcomp.json").write_text(json.dumps(out))
(ROOT / "configs" / "jobs_comp.json").write_text(json.dumps(jobs, indent=1) + "\n")
