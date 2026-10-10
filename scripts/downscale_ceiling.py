"""How much does 4x downscaling (1024 -> 256, majority-vote masks) cost on the GDCLD test crops?

For every kept test crop (same cropping as prepare_gdcld) we take the full-resolution label, downscale it exactly as prepare.py does,
upsample it back (nearest) and compute IoU / F1 of that 'perfect 256-px predictor' against the full-resolution label. This is the best a
model working at 256 px could score if evaluated at 1024 px; it is NOT a model result.
Run: python -I scripts/downscale_ceiling.py
"""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parents[1]
zt = zipfile.ZipFile(r"F:\LandslideData\raw\gdcld\test_data.zip")
tp = fp = fn = tn = 0
per_event = {}
for n in sorted(x for x in zt.namelist() if x.startswith("test_data/") and x.endswith(".tif")):
    event = n.split("/")[1].capitalize()
    im = Image.open(io.BytesIO(zt.read(n))).convert("RGB")
    lab = Image.open(io.BytesIO(zt.read(n.replace("test_data/", "test_label/", 1)))).convert("L")
    w, h = min(im.width, lab.width), min(im.height, lab.height)
    ia, la = np.asarray(im)[:h, :w], np.asarray(lab)[:h, :w] > 0
    del im, lab
    for r in range(h // 1024):
        for c in range(w // 1024):
            t, l = ia[r * 1024:(r + 1) * 1024, c * 1024:(c + 1) * 1024], la[r * 1024:(r + 1) * 1024, c * 1024:(c + 1) * 1024]
            if (t.sum(axis=2) == 0).mean() > 0.5:
                continue
            small = l.reshape(256, 4, 256, 4).mean(axis=(1, 3)) >= 0.5
            up = np.repeat(np.repeat(small, 4, axis=0), 4, axis=1)
            a, b, c_, d = int((up & l).sum()), int((up & ~l).sum()), int((~up & l).sum()), int((~up & ~l).sum())
            tp += a; fp += b; fn += c_; tn += d
            e = per_event.setdefault(event, [0, 0, 0, 0]); e[0] += a; e[1] += b; e[2] += c_; e[3] += d
    del ia, la
    print("done", n, flush=True)


def stats(tp, fp, fn, tn):
    iou = tp / (tp + fp + fn); iou_bg = tn / (tn + fp + fn)
    return dict(iou_landslide=100 * iou, miou=50 * (iou + iou_bg), f1=100 * 2 * tp / (2 * tp + fp + fn), pos_ratio=100 * (tp + fn) / (tp + fp + fn + tn))


print("ALL", {k: round(v, 2) for k, v in stats(tp, fp, fn, tn).items()})
for e, v in per_event.items():
    print(e, {k: round(x, 2) for k, x in stats(*v).items()})
