"""Split builders (report Section 6) and the leakage check.

Splits are lists of (dataset, row_index) references into the processed layout, saved as JSON under splits/.
Rules: split by dataset (B), by event/region/source image (C), never by random tile. Datasets without a group id
(group == 'unknown') can only be held out as a whole.

  A  in-dataset: the dataset's official train/val/test split.
  B  leave-one-dataset-out: test = every tile of the held-out dataset; train/val = official train/val of the others.
  C  leave-one-group-out inside a dataset that exposes groups (events/regions).
  D  sensor transfer: defined explicitly by (train_datasets, test_dataset) lists.

Run:  python -I -m landslide.splits   (with sys.path containing src)
"""
from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path

import numpy as np

from . import data

OUT = Path(__file__).resolve().parents[2] / "splits"


def _refs(meta, name, mask) -> list[list]:
    return [[name, int(i)] for i in meta.index[mask]]


def split_labels(meta) -> "pd.Series":
    """Official split when the dataset ships one; otherwise (official_split == 'none', e.g. CAS) a deterministic
    *group-level* split: groups ordered by md5(name), filled greedily to ~15% test then ~15% val of the tiles.
    This is our own split, not comparable to prior work, and is reported as such."""
    import pandas as pd
    if not (meta.official_split == "none").all():
        return meta.official_split
    import itertools
    order = sorted(meta.group.unique(), key=lambda g: hashlib.md5(g.encode()).hexdigest())   # deterministic tie-break
    sizes, n = meta.group.value_counts(), len(meta)

    def best_subset(pool):    # group subset whose tile share is closest to 15% (exhaustive; <= 2^16 subsets)
        best, best_err = (), 9.0
        for r in range(1, len(pool) + 1):
            for sub in itertools.combinations(pool, r):
                err = abs(sum(sizes[g] for g in sub) / n - 0.15)
                if err < best_err - 1e-12:
                    best, best_err = sub, err
        return best

    test = best_subset(order)
    val = best_subset([g for g in order if g not in test])
    label = {g: "test" if g in test else "val" if g in val else "train" for g in order}
    return pd.Series([label[g] for g in meta.group], index=meta.index)


def protocol_a(names: list[str]) -> dict:
    folds = {}
    for n in names:
        _, _, meta = data.load(n)
        sp = split_labels(meta)
        folds[n] = {s: _refs(meta, n, sp == s) for s in ("train", "val", "test")}
    return folds


def protocol_b(names: list[str]) -> dict:
    folds = {}
    for held in names:
        tr, va = [], []
        for n in names:
            if n == held:
                continue
            _, _, meta = data.load(n)
            sp = split_labels(meta)
            tr += _refs(meta, n, sp == "train")
            va += _refs(meta, n, sp == "val")
        _, _, meta = data.load(held)
        folds[f"holdout_{held}"] = dict(train=tr, val=va, test=_refs(meta, held, meta.index >= 0))
    return folds


def _events() -> dict:
    return json.loads((Path(__file__).resolve().parents[2] / "configs" / "dataset_events.json").read_text(encoding="utf-8"))


def dataset_events(name: str) -> set[str]:
    e = _events()[name]
    return set(e["group_to_event"].values()) if e["tile_events"] else set(e["events"])


def protocol_b_strict(names: list[str]) -> dict:
    """Leave-one-dataset-out with event isolation. Training tiles are removed when their physical event also occurs in the
    held-out dataset: per tile for datasets whose tiles carry an event (lmhld, cas); the whole dataset otherwise.
    Residual overlap cannot be excluded where events are inferred (see configs/dataset_events.json)."""
    ev = _events()
    folds = {}
    for held in names:
        e_held = dataset_events(held)
        tr, va, dropped = [], [], {}
        for n in names:
            if n == held:
                continue
            _, _, meta = data.load(n)
            sp = split_labels(meta)
            if ev[n]["tile_events"]:
                tile_ev = meta.group.map(ev[n]["group_to_event"])
                keep = ~tile_ev.isin(e_held)
                dropped[n] = int((~keep).sum())
            else:
                keep = pd_true(meta) if not (dataset_events(n) & e_held) else ~pd_true(meta)
                dropped[n] = 0 if keep.all() else len(meta)
            tr += _refs(meta, n, (sp == "train") & keep)
            va += _refs(meta, n, (sp == "val") & keep)
        _, _, meta = data.load(held)
        folds[f"holdout_{held}"] = dict(train=tr, val=va, test=_refs(meta, held, meta.index >= 0), dropped_tiles=dropped,
                                        held_events=sorted(e_held))
    return folds


def pd_true(meta):
    import pandas as pd
    return pd.Series(True, index=meta.index)


def protocol_c(name: str) -> dict:
    """Leave-one-group-out within one dataset. Requires EVERY tile to have a known event/region: otherwise tiles of the
    held-out group could hide in train. Test = all tiles of group g; val = one other whole group (next in sorted order);
    train = every remaining group (all of its tiles, regardless of the dataset's own tile-level splits)."""
    _, _, meta = data.load(name)
    if (meta.group == "unknown").any():
        raise ValueError(f"{name}: some tiles have no event/region id, leave-one-event-out would be unverifiable")
    groups = sorted(meta.group.unique())
    if len(groups) < 3:
        raise ValueError(f"{name}: need >=3 groups")
    folds = {}
    for i, g in enumerate(groups):
        v = groups[(i + 1) % len(groups)]
        folds[f"holdout_{g}"] = dict(train=_refs(meta, name, ~meta.group.isin([g, v])), val=_refs(meta, name, meta.group == v),
                                     test=_refs(meta, name, meta.group == g))
    return folds


@functools.lru_cache(maxsize=None)
def _arrays(name: str):
    return data.load(name)


@functools.lru_cache(maxsize=None)
def tile_hash(name: str, k: int) -> str:
    imgs, msks, _ = _arrays(name)
    return hashlib.md5(np.asarray(imgs[k]).tobytes() + np.asarray(msks[k]).tobytes()).hexdigest()


def leakage_report(protocol: str, folds: dict, check_hashes: bool = True) -> list[dict]:
    """Per fold: overlap of references, groups straddling train/test, and byte-identical tiles across train/test."""
    rows = []
    for fold, s in folds.items():
        tr = {tuple(r) for r in s["train"]}
        va = {tuple(r) for r in s["val"]}
        te = {tuple(r) for r in s["test"]}
        row = dict(protocol=protocol, fold=fold, n_train=len(tr), n_val=len(va), n_test=len(te),
                   ref_overlap_train_test=len(tr & te), ref_overlap_val_test=len(va & te), ref_overlap_train_val=len(tr & va))
        grp = {}
        for ds in {r[0] for r in tr | va | te}:
            groups = _arrays(ds)[2].group.to_numpy()
            for (d, i) in tr | va | te:
                if d == ds:
                    grp[(d, i)] = groups[i]
        g_tr = {(d, grp[(d, i)]) for (d, i) in tr if grp[(d, i)] != "unknown"}
        g_te = {(d, grp[(d, i)]) for (d, i) in te if grp[(d, i)] != "unknown"}
        row["groups_straddling_train_test"] = len(g_tr & g_te)
        if "held_events" in s:                       # B-strict: no documented event may be shared between train and test
            ev = _events()
            train_ev = set()
            for ds in {r[0] for r in tr}:
                if ev[ds]["tile_events"]:
                    g = _arrays(ds)[2].group.to_numpy()
                    train_ev |= {ev[ds]["group_to_event"][g[i]] for (d, i) in tr if d == ds}
                else:
                    train_ev |= dataset_events(ds)
            row["shared_events_train_test"] = len(train_ev & set(s["held_events"]))
        row["n_unknown_group_tiles_in_test"] = sum(1 for r in te if grp[r] == "unknown")
        if check_hashes:
            h_tr = {tile_hash(d, i) for (d, i) in tr}
            row["identical_tiles_train_test"] = sum(1 for (d, i) in te if tile_hash(d, i) in h_tr)
        rows.append(row)
    return rows


def save(name: str, folds: dict) -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(folds))


def main() -> None:
    import sys

    import pandas as pd
    sys.stdout.reconfigure(encoding="utf-8")   # CAS region names contain full-width brackets

    names = data.available_datasets()
    print("datasets:", names)
    report = []
    a, b, bs = protocol_a(names), protocol_b(names), protocol_b_strict(names)
    save("protocol_A", a); save("protocol_B", b); save("protocol_B_strict", bs)
    report += leakage_report("A", a) + leakage_report("B", b) + leakage_report("B_strict", bs)
    for n in names:
        try:
            c = protocol_c(n); save(f"protocol_C_{n}", c); report += leakage_report(f"C:{n}", c)
        except ValueError as e:
            print("protocol C skipped:", e)
    df = pd.DataFrame(report)
    df.to_csv(OUT / "leakage_report.csv", index=False)
    print(df.to_string())


if __name__ == "__main__":
    main()
