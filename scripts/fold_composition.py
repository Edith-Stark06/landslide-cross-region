"""Which source datasets contribute training tiles in each fold of protocol B and B-strict (counts of train tiles per source dataset).
Run: python -I scripts/fold_composition.py -> prints a table; make_paper_assets.py builds the paper table from the same splits."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DS = ["l4s", "hrgldd", "gdcld", "lmhld", "cas"]
for proto in ("protocol_B", "protocol_B_strict"):
    d = json.loads((ROOT / "splits" / f"{proto}.json").read_text())
    print(proto)
    for ds in DS:
        c = Counter(r[0] for r in d["holdout_" + ds]["train"])
        print(f"  held-out {ds:7s} train tiles by source:", {k: c.get(k, 0) for k in DS}, "total", sum(c.values()))
