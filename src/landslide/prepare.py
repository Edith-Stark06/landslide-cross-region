"""Convert raw datasets into one processed layout:

  <out>/<dataset>/images.npy   (N, C, H, W)  float16 (multispectral) or uint8 (RGB)
  <out>/<dataset>/masks.npy    (N, H, W)     uint8 in {0,1}
  <out>/<dataset>/meta.csv     one row per tile: idx, tile_id, official_split, group, pos_ratio, ...

`group` is the unit that must never straddle train/test (event, region or source image). When a dataset
does not expose events, group is "unknown" and the dataset can only be used as a whole.
Run:  python -I -m landslide.prepare l4s|hrgldd   (with PYTHONPATH=src)
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(r"F:\LandslideData\raw")
EXTRACTED = Path(r"F:\LandslideData\extracted")
OUT = Path(r"F:\LandslideData\processed")


def _write_meta(rows: list[dict], out: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.insert(0, "idx", np.arange(len(df)))
    df.to_csv(out / "meta.csv", index=False)
    return df


def prepare_l4s() -> None:
    import pyfive  # pure-python HDF5 reader; h5py is blocked on this machine by Windows Application Control

    out = OUT / "l4s"
    out.mkdir(parents=True, exist_ok=True)
    plan = [("train", "TrainData", 3799), ("val", "ValidData", 245), ("test", "TestData", 800)]
    n_total = sum(n for _, _, n in plan)
    imgs = np.lib.format.open_memmap(out / "images.npy", mode="w+", dtype=np.float16, shape=(n_total, 14, 128, 128))
    msks = np.lib.format.open_memmap(out / "masks.npy", mode="w+", dtype=np.uint8, shape=(n_total, 128, 128))
    rows, k, max_rel_err = [], 0, 0.0
    for split, folder, n in plan:
        for i in range(1, n + 1):
            with pyfive.File(str(EXTRACTED / "landslide4sense" / folder / "img" / f"image_{i}.h5")) as f:
                a = np.asarray(f["img"][:], dtype=np.float64)
            with pyfive.File(str(EXTRACTED / "landslide4sense" / folder / "mask" / f"mask_{i}.h5")) as f:
                m = np.asarray(f["mask"][:], dtype=np.uint8)
            h = a.transpose(2, 0, 1).astype(np.float16)
            denom = np.maximum(np.abs(a.transpose(2, 0, 1)), 1e-6)
            max_rel_err = max(max_rel_err, float(np.max(np.abs(h.astype(np.float64) - a.transpose(2, 0, 1)) / denom)))
            imgs[k], msks[k] = h, m
            rows.append(dict(tile_id=f"{folder}/image_{i}", official_split=split, group="unknown", pos_ratio=float(m.mean()),
                             native_h=128, native_w=128))
            k += 1
    imgs.flush(); msks.flush()
    _write_meta(rows, out)
    print(f"l4s: {k} tiles written; max relative float16 quantisation error {max_rel_err:.2e}")


def prepare_hrgldd() -> None:
    out = OUT / "hrgldd"
    out.mkdir(parents=True, exist_ok=True)
    parts = [("train", "train"), ("val", "val"), ("test", "test")]
    xs, ys, rows = [], [], []
    for split, p in parts:
        x = np.load(RAW / "hr_gldd" / f"{p}X.npy")            # (N,128,128,4) float32
        y = np.load(RAW / "hr_gldd" / f"{p}Y.npy")            # (N,128,128,1) float32
        xs.append(x.transpose(0, 3, 1, 2).astype(np.float16))
        yb = (y[..., 0] > 0.5).astype(np.uint8)
        ys.append(yb)
        rows += [dict(tile_id=f"{p}/{i}", official_split=split, group="unknown", pos_ratio=float(yb[i].mean()),
                      native_h=128, native_w=128) for i in range(len(x))]
        print(split, x.shape, "label values", np.unique(y).tolist())
    np.save(out / "images.npy", np.concatenate(xs))
    np.save(out / "masks.npy", np.concatenate(ys))
    _write_meta(rows, out)
    print(f"hrgldd: {len(rows)} tiles written")


LMHLD_SENSOR = {  # LMHLD paper Table 1 (read): sensor, spatial resolution in metres
    "Rio de Janeiro_Brazil": ("RapidEye", 5), "Hokkaido_Japan": ("PlanetScope", 3), "Emilia-Romagna_Italy": ("Sentinel-2", 10),
    "Gorkha_Nepal": ("Sentinel-2", 10), "Jiuzhaigou_China": ("Gaofen-2", 1), "Wenchuan_China": ("IKONOS", 0.8),
    "Taiwan_China": ("PlanetScope", 3),
}
LMHLD_SIZE = 128   # all events resized to a common tile size (native 32/64/128/224 px)


def prepare_lmhld() -> None:
    from PIL import Image   # torch is not used here: Windows Application Control sometimes blocks its DLLs on this machine

    def resize(a: np.ndarray, resample) -> np.ndarray:   # a: (N, C, H, W) float32 -> (N, C, S, S)
        out = np.empty(a.shape[:2] + (LMHLD_SIZE, LMHLD_SIZE), np.float32)
        for i in range(a.shape[0]):
            for c in range(a.shape[1]):
                out[i, c] = np.asarray(Image.fromarray(a[i, c], mode="F").resize((LMHLD_SIZE,) * 2, resample))
        return out

    base = EXTRACTED / "lmhld" / "LMHLD" / "LMHLD_dataset_different_patch_sizes"
    out = OUT / "lmhld"
    out.mkdir(parents=True, exist_ok=True)
    events = sorted(p.name for p in base.iterdir() if p.is_dir())
    shapes = {(ev, s): np.load(base / ev / f"{s}_images.npy", mmap_mode="r").shape for ev in events for s in ("train", "val", "test")}
    n_total = sum(sh[0] for sh in shapes.values())
    imgs = np.lib.format.open_memmap(out / "images.npy", mode="w+", dtype=np.float16, shape=(n_total, 4, LMHLD_SIZE, LMHLD_SIZE))
    msks = np.lib.format.open_memmap(out / "masks.npy", mode="w+", dtype=np.uint8, shape=(n_total, LMHLD_SIZE, LMHLD_SIZE))
    rows, k = [], 0
    for ev in events:
        name, native = ev.rsplit("_", 1)
        sensor, gsd = LMHLD_SENSOR[name]
        for split in ("train", "val", "test"):
            x = np.load(base / ev / f"{split}_images.npy", mmap_mode="r")
            y = np.load(base / ev / f"{split}_labels.npy", mmap_mode="r")
            for a in range(0, len(x), 256):
                xb = np.asarray(x[a:a + 256], dtype=np.float32)
                if xb.shape[1] == 5:                                   # Rio de Janeiro: drop assumed red-edge channel (idx 3)
                    xb = xb[:, [0, 1, 2, 4]]
                yb = np.asarray(y[a:a + 256, 0], dtype=np.float32)
                if int(native) != LMHLD_SIZE:
                    xb = resize(xb, Image.BOX if int(native) > LMHLD_SIZE else Image.BILINEAR)
                    yb = resize(yb[:, None], Image.NEAREST)[:, 0] > 0.5
                else:
                    yb = yb > 0.5
                n = len(xb)
                imgs[k:k + n] = xb.astype(np.float16)
                msks[k:k + n] = yb.astype(np.uint8)
                for j in range(n):
                    rows.append(dict(tile_id=f"{ev}/{split}/{a + j}", official_split=split, group=name, sensor=sensor, gsd_m=gsd,
                                     pos_ratio=float(yb[j].mean()), native_h=int(native), native_w=int(native)))
                k += n
    imgs.flush(); msks.flush()
    _write_meta(rows, out)
    print(f"lmhld: {k} tiles written ({len(events)} events)")


# ---------------------------------------------------------------- GDCLD

GD_SIZE = 256   # 1024 px source tiles -> 256 px (box average); masks by majority vote


def _gd_pair(args):
    import io
    from PIL import Image
    zip_path, img_name, lab_name = args
    global _ZF
    if "_ZF" not in globals() or _ZF[0] != zip_path:
        _ZF = (zip_path, zipfile.ZipFile(zip_path))
    z = _ZF[1]
    im = Image.open(io.BytesIO(z.read(img_name))).convert("RGB")
    lab = np.array(Image.open(io.BytesIO(z.read(lab_name))).convert("L"))
    x = np.asarray(im.resize((GD_SIZE, GD_SIZE), Image.BOX)).transpose(2, 0, 1)
    f = 1024 // GD_SIZE
    m = ((lab.reshape(GD_SIZE, f, GD_SIZE, f) > 0).mean(axis=(1, 3)) >= 0.5).astype(np.uint8)
    return x, m, float((lab > 0).mean())


def prepare_gdcld() -> None:
    import io
    from multiprocessing import Pool
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None   # trusted, MD5-verified Zenodo archive

    out = OUT / "gdcld"
    out.mkdir(parents=True, exist_ok=True)
    jobs, split_of = [], []
    for split, zname, dpre, lpre in (("train", "train_data.zip", "train_data/", "train_label/"), ("val", "val_data.zip", "val_data/", "val_label/")):
        zp = str(RAW / "gdcld" / zname)
        names = sorted(n for n in zipfile.ZipFile(zp).namelist() if n.startswith(dpre) and n.endswith(".tif"))
        for n in names:
            jobs.append((zp, n, lpre + n[len(dpre):])); split_of.append(split)
    # test: 7 large event images -> non-overlapping 1024 crops (common size of image/label), nodata tiles dropped
    zt = zipfile.ZipFile(str(RAW / "gdcld" / "test_data.zip"))
    test_tiles, test_rows, dropped = [], [], 0
    for n in sorted(x for x in zt.namelist() if x.startswith("test_data/") and x.endswith(".tif")):
        event = n.split("/")[1].capitalize()
        im = Image.open(io.BytesIO(zt.read(n))).convert("RGB")
        lab = Image.open(io.BytesIO(zt.read(n.replace("test_data/", "test_label/", 1)))).convert("L")
        w, h = min(im.width, lab.width), min(im.height, lab.height)
        ia, la = np.asarray(im)[:h, :w], np.asarray(lab)[:h, :w]
        for r in range(h // 1024):
            for c in range(w // 1024):
                t, l = ia[r * 1024:(r + 1) * 1024, c * 1024:(c + 1) * 1024], la[r * 1024:(r + 1) * 1024, c * 1024:(c + 1) * 1024]
                if (t.sum(axis=2) == 0).mean() > 0.5:
                    dropped += 1
                    continue
                x = np.asarray(Image.fromarray(t).resize((GD_SIZE, GD_SIZE), Image.BOX)).transpose(2, 0, 1)
                f = 1024 // GD_SIZE
                m = ((l.reshape(GD_SIZE, f, GD_SIZE, f) > 0).mean(axis=(1, 3)) >= 0.5).astype(np.uint8)
                test_tiles.append((x, m)); test_rows.append((f"{n.split('/')[-1]}#r{r}c{c}", event, float((l > 0).mean())))
        del ia, la, im, lab
    n_total = len(jobs) + len(test_tiles)
    imgs = np.lib.format.open_memmap(out / "images.npy", mode="w+", dtype=np.uint8, shape=(n_total, 3, GD_SIZE, GD_SIZE))
    msks = np.lib.format.open_memmap(out / "masks.npy", mode="w+", dtype=np.uint8, shape=(n_total, GD_SIZE, GD_SIZE))
    rows = []
    with Pool(4) as pool:
        for k, (x, m, full_pos) in enumerate(pool.imap(_gd_pair, jobs, chunksize=16)):
            imgs[k], msks[k] = x, m
            rows.append(dict(tile_id=jobs[k][1], official_split=split_of[k], group="unknown", sensor="mixed", gsd_m=np.nan,
                             pos_ratio=float(m.mean()), pos_ratio_full_res=full_pos, native_h=1024, native_w=1024))
            if k % 2000 == 0:
                print("gdcld", k, "/", len(jobs), flush=True)
    for j, ((x, m), (tid, event, full_pos)) in enumerate(zip(test_tiles, test_rows)):
        k = len(jobs) + j
        imgs[k], msks[k] = x, m
        rows.append(dict(tile_id=tid, official_split="test", group=event, sensor="mixed", gsd_m=np.nan, pos_ratio=float(m.mean()),
                         pos_ratio_full_res=full_pos, native_h=1024, native_w=1024))
    imgs.flush(); msks.flush()
    _write_meta(rows, out)
    print(f"gdcld: {n_total} tiles ({len(jobs)} train/val + {len(test_tiles)} test crops; {dropped} nodata crops dropped)")


# ---------------------------------------------------------------- CAS

CAS_SIZE = 256   # 512 px source tiles -> 256 px


def cas_region(zip_stem: str) -> str:
    """Physical region of a CAS zip. Several zips are the same area imaged by different sensors (e.g. Moxitaidi SAT/UAV-0.6m/
    UAV-1m, Tiburon Sentinel/Planet), so splits must group by region, not by zip."""
    r = zip_stem.replace("（", "(").split("(")[0].strip()
    return r.title() if r.islower() else r


def _cas_pair(args):
    import io
    from PIL import Image
    zip_path, img_name, mask_name = args
    global _ZF
    if "_ZF" not in globals() or _ZF[0] != zip_path:
        _ZF = (zip_path, zipfile.ZipFile(zip_path))
    z = _ZF[1]
    im = Image.open(io.BytesIO(z.read(img_name))).convert("RGB")
    mk = np.array(Image.open(io.BytesIO(z.read(mask_name))).convert("L"))
    x = np.asarray(im.resize((CAS_SIZE, CAS_SIZE), Image.BOX)).transpose(2, 0, 1)
    m = (np.asarray(Image.fromarray((mk > 0).astype(np.float32), mode="F").resize((CAS_SIZE, CAS_SIZE), Image.BOX)) >= 0.5).astype(np.uint8)
    return x, m, float((mk > 0).mean()), im.size[0], im.size[1]


def prepare_cas() -> None:
    from multiprocessing import Pool
    out = OUT / "cas"
    out.mkdir(parents=True, exist_ok=True)
    jobs, groups = [], []
    for zp in sorted((RAW / "cas").glob("*.zip")):
        if "shp" in zp.name:
            continue
        names = zipfile.ZipFile(zp).namelist()
        # extensions differ in case across zips (.tif / .TIF): pair img/<stem> with mask/<stem> case-insensitively
        mask_of = {n[5:].rsplit(".", 1)[0].lower(): n for n in names if n.startswith("mask/") and not n.endswith("/")}
        for n in sorted(x for x in names if x.startswith("img/") and not x.endswith("/")):
            stem = n[4:].rsplit(".", 1)[0]
            jobs.append((str(zp), n, mask_of[stem.lower()])); groups.append(zp.stem)
    imgs = np.lib.format.open_memmap(out / "images.npy", mode="w+", dtype=np.uint8, shape=(len(jobs), 3, CAS_SIZE, CAS_SIZE))
    msks = np.lib.format.open_memmap(out / "masks.npy", mode="w+", dtype=np.uint8, shape=(len(jobs), CAS_SIZE, CAS_SIZE))
    rows = []
    with Pool(4) as pool:
        for k, (x, m, full_pos, w, h) in enumerate(pool.imap(_cas_pair, jobs, chunksize=16)):
            imgs[k], msks[k] = x, m
            rows.append(dict(tile_id=f"{groups[k]}/{jobs[k][1][4:]}", official_split="none", group=cas_region(groups[k]), sensor=groups[k], gsd_m=np.nan,
                             pos_ratio=float(m.mean()), pos_ratio_full_res=full_pos, native_h=h, native_w=w))
            if k % 3000 == 0:
                print("cas", k, "/", len(jobs), flush=True)
    imgs.flush(); msks.flush()
    _write_meta(rows, out)
    print(f"cas: {len(jobs)} tiles, {len(set(groups))} region/sensor groups")


if __name__ == "__main__":
    {"l4s": prepare_l4s, "hrgldd": prepare_hrgldd, "lmhld": prepare_lmhld, "gdcld": prepare_gdcld, "cas": prepare_cas}[sys.argv[1]]()

