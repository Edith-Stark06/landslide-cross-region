"""Unified dataset access over the processed layout produced by landslide.prepare."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import channels

PROCESSED = Path(os.environ.get("LANDSLIDE_PROCESSED", r"F:\LandslideData\processed"))


def available_datasets(root: Path = PROCESSED) -> list[str]:
    return sorted(p.name for p in root.iterdir() if (p / "meta.csv").exists()) if root.exists() else []


def load(name: str, root: Path = PROCESSED, mmap: bool = True):
    d = root / name
    imgs = np.load(d / "images.npy", mmap_mode="r" if mmap else None)
    msks = np.load(d / "masks.npy", mmap_mode="r" if mmap else None)
    meta = pd.read_csv(d / "meta.csv")
    return imgs, msks, meta


def channel_stats(name: str, root: Path = PROCESSED) -> dict:
    """Per-channel mean/std from the *official train split only* (never val/test). Cached as stats.json."""
    f = root / name / "stats.json"
    if f.exists():
        return json.loads(f.read_text())
    from .splits import split_labels   # lazy import: splits imports this module
    imgs, _, meta = load(name, root)
    idx = meta.index[split_labels(meta) == "train"].to_numpy()
    idx = idx[:: max(1, len(idx) // 500)]                    # <=500 tiles, deterministic subsample
    x = np.asarray(imgs[idx], dtype=np.float64)
    if imgs.dtype == np.uint8:
        x = x / 255.0
    stats = dict(mean=x.mean(axis=(0, 2, 3)).tolist(), std=(x.std(axis=(0, 2, 3)) + 1e-6).tolist(), n_tiles=int(len(idx)))
    try:
        f.write_text(json.dumps(stats))
    except OSError:                       # read-only input (Kaggle): stats.json ships with the dataset
        pass
    return stats


class LandslideDataset:
    """Items: dict(image float32 (C,H,W), mask int64 (H,W), avail float32 (C,), dataset str, index int).

    refs: list of (dataset, row index). mode: 'rgb' -> C=3; 'bank' -> C=len(channels.BANK) with availability mask.
    Normalisation uses per-dataset train-split statistics on the *source* channels, applied before channel mapping.
    """

    def __init__(self, refs, mode: str = "rgb", root: Path = PROCESSED, augment: bool = False, seed: int = 0,
                 crop: int | None = None):
        """mode: 'rgb' (3 ch), 'bank' (channel bank + availability mask) or 'native' (the dataset's own channels; only valid
        when all refs come from one dataset). crop: random crop size at native scale when augment=True (keeps the GSD)."""
        assert mode in ("rgb", "bank", "native")
        assert mode != "native" or len({r[0] for r in refs}) == 1, "native mode needs a single dataset"
        self.refs, self.mode, self.augment, self.crop = list(refs), mode, augment, crop
        self.rng = np.random.default_rng(seed)
        self._cache = {n: load(n, root) for n in {r[0] for r in self.refs}}
        self._stats = {n: channel_stats(n, root) for n in self._cache}

    def __len__(self):
        return len(self.refs)

    def __getitem__(self, i):
        name, k = self.refs[i]
        imgs, msks, _ = self._cache[name]
        x = np.asarray(imgs[k], dtype=np.float32)
        if imgs.dtype == np.uint8:
            x = x / 255.0
        s = self._stats[name]
        x = (x - np.asarray(s["mean"], np.float32)[:, None, None]) / np.asarray(s["std"], np.float32)[:, None, None]
        m = np.asarray(msks[k], dtype=np.int64)
        if self.mode == "rgb":
            x, avail = channels.to_rgb(x, name), np.ones(3, np.float32)
        elif self.mode == "native":
            avail = np.ones(x.shape[0], np.float32)
        else:
            x, avail = channels.to_bank(x, name)
        if self.augment and self.crop and x.shape[1] > self.crop:
            r0, c0 = (int(v) for v in self.rng.integers(0, x.shape[1] - self.crop + 1, size=2))
            x, m = x[:, r0:r0 + self.crop, c0:c0 + self.crop], m[r0:r0 + self.crop, c0:c0 + self.crop]
        if self.augment:
            if self.rng.random() < 0.5:
                x, m = x[:, :, ::-1], m[:, ::-1]
            if self.rng.random() < 0.5:
                x, m = x[:, ::-1, :], m[::-1, :]
            r = int(self.rng.integers(0, 4))
            x, m = np.rot90(x, r, axes=(1, 2)), np.rot90(m, r)
        return dict(image=np.ascontiguousarray(x), mask=np.ascontiguousarray(m), avail=avail, dataset=name, index=int(k))
