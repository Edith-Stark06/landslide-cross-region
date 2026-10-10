import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from landslide import channels, data  # noqa: E402

NAMES = data.available_datasets()
EXPECTED_C = {"l4s": 14, "hrgldd": 4, "gdcld": 3, "lmhld": 4, "cas": 3}


@pytest.mark.skipif(not NAMES, reason="no processed datasets found")
@pytest.mark.parametrize("name", NAMES)
def test_shapes_and_labels(name):
    imgs, msks, meta = data.load(name)
    assert imgs.ndim == 4 and msks.ndim == 3
    assert imgs.shape[0] == msks.shape[0] == len(meta)
    assert imgs.shape[2:] == msks.shape[1:]
    if name in EXPECTED_C:
        assert imgs.shape[1] == EXPECTED_C[name]
    for s in range(0, len(msks), max(1, len(msks) // 200)):       # sampled; full scan is done in the data summary
        assert set(np.unique(msks[s]).tolist()) <= {0, 1}
    assert set(meta.official_split.unique()) <= {"train", "val", "test", "none"}      # 'none': dataset ships no split (CAS)
    assert np.isfinite(np.asarray(imgs[0], dtype=np.float32)).all()


@pytest.mark.skipif(not NAMES, reason="no processed datasets found")
@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("mode", ["rgb", "bank"])
def test_unified_item(name, mode):
    ds = data.LandslideDataset([(name, 0), (name, 1)], mode=mode)
    it = ds[0]
    c = 3 if mode == "rgb" else len(channels.BANK)
    assert it["image"].shape[0] == c and it["avail"].shape == (c,)
    assert it["image"].shape[1:] == it["mask"].shape
    assert it["image"].dtype == np.float32 and set(np.unique(it["mask"]).tolist()) <= {0, 1}
    if mode == "bank":
        # unavailable channels are exactly zero
        missing = it["avail"] == 0
        assert np.all(it["image"][missing] == 0)


PROTOCOL_FILES = ["protocol_A", "protocol_B", "protocol_B_strict"] + [p.stem for p in sorted((ROOT / "splits").glob("protocol_C_*.json"))]


@pytest.mark.skipif(not {"gdcld", "l4s"} <= set(NAMES), reason="needs gdcld and l4s")
def test_mixed_resolution_batches_share_one_size_with_crop():
    """Cross-dataset training mixes 256 px (gdcld) and 128 px (l4s) tiles: a 128 crop must make every item 128x128 (keeping GSD)."""
    ds = data.LandslideDataset([("gdcld", 0), ("l4s", 0), ("gdcld", 1), ("l4s", 1)], mode="rgb", augment=True, crop=128)
    shapes = {ds[i]["image"].shape for i in range(len(ds))}
    assert shapes == {(3, 128, 128)}
    assert all(ds[i]["mask"].shape == (128, 128) for i in range(len(ds)))
    ev = data.LandslideDataset([("gdcld", 0)], mode="rgb", augment=False, crop=128)       # evaluation: never cropped
    assert ev[0]["image"].shape == (3, 256, 256)


@pytest.mark.parametrize("proto", PROTOCOL_FILES)
def test_no_split_overlap(proto):
    f = ROOT / "splits" / f"{proto}.json"
    if not f.exists():
        pytest.skip("run landslide.splits first")
    for fold, s in json.loads(f.read_text()).items():
        tr, va, te = ({tuple(r) for r in s[k]} for k in ("train", "val", "test"))
        assert not (tr & te), f"{proto}/{fold}: train/test overlap"
        assert not (va & te), f"{proto}/{fold}: val/test overlap"
        assert not (tr & va), f"{proto}/{fold}: train/val overlap"
        assert te, f"{proto}/{fold}: empty test set"


def test_protocol_b_holds_out_whole_dataset():
    f = ROOT / "splits" / "protocol_B.json"
    if not f.exists():
        pytest.skip("run landslide.splits first")
    for fold, s in json.loads(f.read_text()).items():
        held = fold.removeprefix("holdout_")
        assert {r[0] for r in s["test"]} == {held}
        assert held not in {r[0] for r in s["train"]} | {r[0] for r in s["val"]}


@pytest.mark.parametrize("proto", [p for p in PROTOCOL_FILES if p.startswith("protocol_C_")])
def test_protocol_c_holds_out_whole_group(proto):
    import pandas as pd
    name = proto.removeprefix("protocol_C_")
    meta = pd.read_csv(data.PROCESSED / name / "meta.csv")
    for fold, s in json.loads((ROOT / "splits" / f"{proto}.json").read_text()).items():
        held = fold.removeprefix("holdout_")
        te_groups = {meta.group.iat[i] for _, i in s["test"]}
        assert te_groups == {held}
        assert len(s["test"]) == int((meta.group == held).sum())              # every tile of the group is in test
        assert held not in {meta.group.iat[i] for _, i in s["train"] + s["val"]}


def test_b_strict_shares_no_documented_event():
    import pandas as pd
    f = ROOT / "splits" / "leakage_report.csv"
    if not f.exists():
        pytest.skip("run landslide.splits first")
    df = pd.read_csv(f)
    bs = df[df.protocol == "B_strict"]
    assert len(bs) > 0 and (bs.shared_events_train_test == 0).all()


def test_no_group_straddles_when_groups_known():
    import pandas as pd
    f = ROOT / "splits" / "leakage_report.csv"
    if not f.exists():
        pytest.skip("run landslide.splits first")
    df = pd.read_csv(f)
    assert (df.groups_straddling_train_test[df.protocol.str.startswith(("B", "C"))] == 0).all()
