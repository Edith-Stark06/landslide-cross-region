"""Train and evaluate one segmentation run from a YAML config.

    python -m landslide.run --config configs/baseline_l4s_unet.yaml [--seed 1] [--out results/runs]

Config keys: name, protocol (splits/<protocol>.json), fold, model (unet|deeplabv3plus|segformer_b0), encoder, mode (native|rgb|bank),
pretrained, epochs, batch_size, lr, weight_decay, crop, seed, num_workers, max_train/max_eval (optional subsampling for smoke tests).
Writes <out>/<name>__<fold>__seed<k>/result.json with config, seed, code revision, library versions, per-epoch history and test metrics.
Test data are touched once, with the weights of the best validation epoch.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def code_revision() -> str:
    if os.environ.get("GIT_HASH"):
        return os.environ["GIT_HASH"]
    f = ROOT / "GIT_HASH"
    if f.exists():
        return f.read_text().strip()
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def build_model(cfg: dict, in_ch: int):
    import segmentation_models_pytorch as smp
    w = "imagenet" if cfg.get("pretrained", True) else None
    kind = cfg["model"]
    if kind == "unet":
        return smp.Unet(cfg.get("encoder", "resnet34"), encoder_weights=w, in_channels=in_ch, classes=2)
    if kind == "deeplabv3plus":
        return smp.DeepLabV3Plus(cfg.get("encoder", "resnet34"), encoder_weights=w, in_channels=in_ch, classes=2)
    if kind == "segformer_b0":
        return smp.Segformer("mit_b0", encoder_weights=w, in_channels=in_ch, classes=2)
    raise ValueError(kind)


def seed_everything(seed: int) -> None:
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def loss_fn(logits, y):
    import torch
    import torch.nn.functional as F
    ce = F.cross_entropy(logits, y)
    p = torch.softmax(logits, 1)[:, 1]
    t = (y == 1).float()
    dice = 1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1)
    return ce + dice


def evaluate(model, refs, cfg, device, batch_size=16, workers=2):
    import torch
    from torch.utils.data import DataLoader

    from . import data
    from .metrics import BinaryMeter
    out = {}
    for name in sorted({r[0] for r in refs}):
        ds = data.LandslideDataset([r for r in refs if r[0] == name], mode=cfg["mode"], augment=False)
        meter = BinaryMeter()
        model.eval()
        with torch.no_grad():
            for b in DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers):
                x = b["image"].to(device)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits = model(x)
                p = torch.softmax(logits.float(), 1)[:, 1].cpu().numpy()
                meter.update(p, b["mask"].numpy())
        out[name] = meter.compute()
    return out


def main(argv=None) -> dict:
    import torch
    from torch.utils.data import DataLoader

    from . import data

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "results" / "runs"))
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.seed is not None:
        cfg["seed"] = args.seed
    seed = int(cfg.get("seed", 0))
    run_id = f"{cfg['name']}__{cfg['fold']}__seed{seed}"
    out_dir = Path(args.out) / run_id
    if (out_dir / "result.json").exists():
        print("already done:", run_id)
        return json.loads((out_dir / "result.json").read_text())
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = json.loads((ROOT / "splits" / f"{cfg['protocol']}.json").read_text())[cfg["fold"]]
    rng = np.random.default_rng(seed)
    sub = lambda refs, n: refs if not n or len(refs) <= n else [refs[i] for i in sorted(rng.choice(len(refs), n, replace=False))]  # noqa: E731
    train, val, test = (sub(splits["train"], cfg.get("max_train")), sub(splits["val"], cfg.get("max_eval")), sub(splits["test"], cfg.get("max_eval")))

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = data.LandslideDataset(train, mode=cfg["mode"], augment=True, seed=seed, crop=cfg.get("crop"))
    in_ch = train_ds[0]["image"].shape[0]
    model = build_model(cfg, in_ch).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 3e-4), weight_decay=cfg.get("weight_decay", 1e-4))
    epochs = int(cfg.get("epochs", 30))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    loader = DataLoader(train_ds, batch_size=cfg.get("batch_size", 32), shuffle=True, drop_last=True, num_workers=cfg.get("num_workers", 2),
                        persistent_workers=cfg.get("num_workers", 2) > 0, generator=torch.Generator().manual_seed(seed))

    history, best, best_state, t0 = [], -1.0, None, time.time()
    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for b in loader:
            x, y = b["image"].to(device, non_blocking=True), b["mask"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x)
            loss = loss_fn(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * len(x); n += len(x)
        sched.step()
        v = evaluate(model, val, cfg, device)
        v_iou = float(np.mean([m["iou_landslide"] for m in v.values()]))
        history.append(dict(epoch=ep, train_loss=tot / max(1, n), val_iou=v_iou, seconds=time.time() - t0))
        print(f"[{run_id}] ep {ep} loss {tot / max(1, n):.4f} val IoU(landslide) {v_iou:.4f}", flush=True)
        if v_iou > best:
            best, best_state = v_iou, copy.deepcopy({k: t.detach().cpu() for k, t in model.state_dict().items()})
            best_ep = ep
    train_seconds = time.time() - t0
    model.load_state_dict(best_state)
    n_test = len(test)
    test = evaluate(model, test, cfg, device)

    from torch.utils.flop_counter import FlopCounterMode
    h = int(cfg.get("crop") or train_ds[0]["image"].shape[1])
    model.eval()
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        model(torch.zeros(1, in_ch, h, h, device=device))
    result = dict(config=cfg, seed=seed, run_id=run_id, code_revision=code_revision(), torch=torch.__version__,
                  device=torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu", n_params=n_params,
                  gflops_at_input=fc.get_total_flops() / 2e9, input_hw=h, n_train=len(train), n_val=len(val), n_test=n_test,
                  best_epoch=best_ep, best_val_iou=best, history=history, train_seconds=train_seconds, test=test)
    (out_dir / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result["test"][k] for k in result["test"]}, indent=1))
    return result


if __name__ == "__main__":
    main()
