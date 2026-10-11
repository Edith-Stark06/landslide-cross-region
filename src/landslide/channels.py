"""Channel-handling strategy shared by every dataset.

All datasets are mapped into one fixed *channel bank* plus a per-sample availability mask, so a single
network (with a band adapter) can consume RGB, 4-band, 14-band and (later) SAR data.

Two input modes:
  - "rgb":  3 channels (R, G, B). Universal baseline mode; every optical dataset provides it.
  - "bank": len(BANK) channels, missing channels zero-filled, plus a {0,1} availability mask of the same length.

The source-band indices below come from the dataset documentation named in the comment next to each entry.
Anything not documented is marked ASSUMED and must be re-checked before it is relied on.
"""
from __future__ import annotations

import numpy as np

BANK = ["R", "G", "B", "NIR", "RE1", "SWIR1", "SWIR2", "SLOPE", "DEM", "VV", "VH"]
RGB = ["R", "G", "B"]

# source channel index (0-based) for each bank channel, per dataset
DATASET_CHANNELS: dict[str, dict[str, int]] = {
    # Landslide4Sense README (iarai/Landslide4Sense-2022): bands 1-12 = Sentinel-2 B1..B12 in order (B8A omitted),
    # band 13 = slope, band 14 = DEM  ->  B2=idx1 (blue), B3=idx2 (green), B4=idx3 (red), B5=idx4, B8=idx7, B11=idx10, B12=idx11.
    "l4s": {"R": 3, "G": 2, "B": 1, "NIR": 7, "RE1": 4, "SWIR1": 10, "SWIR2": 11, "SLOPE": 12, "DEM": 13},
    # HR-GLDD ESSD 15, 3283 text says "red, green, blue, and NIR", but the released arrays most likely follow PlanetScope's native
    # B,G,R,NIR order: channel means are 0.109 / 0.150 / 0.140 / 0.500, i.e. blue < red < green, exactly the pattern of the
    # LMHLD PlanetScope scenes whose order is documented as B,G,R,NIR (Hokkaido 336/412/386, ratio pattern identical), and unlike
    # R,G,B where red would be darker than blue. This is statistical evidence, NOT confirmation: UNVERIFIED.
    # (Runs made before 2026-10-11 used R,G,B for this dataset; see results/runs notes in docs/data_notes.md.)
    "hrgldd": {"B": 0, "G": 1, "R": 2, "NIR": 3},
    # GDCLD ESSD 16, 4817: 8-bit RGB.
    "gdcld": {"R": 0, "G": 1, "B": 2},
    # LMHLD (arXiv 2502.19866): "four spectral bands (blue, green, red, and near-infrared)" -> array order B,G,R,NIR.
    # Rio de Janeiro (RapidEye) ships 5 channels; prepare.py drops channel 3 (ASSUMED red edge) so the order stays B,G,R,NIR.
    "lmhld": {"B": 0, "G": 1, "R": 2, "NIR": 3},
    # CAS Landslide README: "RGB images in TIF format"; masks in `mask/` ({0,1}); `label/` is a 0/255 visualisation.
    "cas": {"R": 0, "G": 1, "B": 2},
}


def available(dataset: str) -> list[str]:
    return [c for c in BANK if c in DATASET_CHANNELS[dataset]]


def to_rgb(x: np.ndarray, dataset: str) -> np.ndarray:
    """x: (C, H, W) in the dataset's own channel order -> (3, H, W) R,G,B."""
    m = DATASET_CHANNELS[dataset]
    return x[[m["R"], m["G"], m["B"]]]


def to_bank(x: np.ndarray, dataset: str) -> tuple[np.ndarray, np.ndarray]:
    """x: (C, H, W) -> (bank (len(BANK), H, W) zero-filled, avail (len(BANK),) in {0,1})."""
    m = DATASET_CHANNELS[dataset]
    out = np.zeros((len(BANK),) + x.shape[1:], dtype=x.dtype)
    avail = np.zeros(len(BANK), dtype=np.float32)
    for i, c in enumerate(BANK):
        if c in m:
            out[i] = x[m[c]]
            avail[i] = 1.0
    return out, avail
