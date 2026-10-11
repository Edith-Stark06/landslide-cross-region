"""Train and evaluate one segmentation run from a YAML config.

    python -m landslide.run --config configs/baseline_l4s_unet.yaml [--seed 1] [--out results/runs]

Config keys: name, protocol (splits/<protocol>.json), fold, model (unet|deeplabv3plus|segformer_b0|segformer_b3, optionally prefixed
adapter_), encoder, mode (native|rgb|bank), pretrained, epochs, batch_size, lr, weight_decay, crop, seed, num_workers,
max_train / max_val / max_test / max_eval (optional subsampling).
Distillation (optional): teacher_model (e.g. segformer_b3), teacher_lora_rank (0 = full fine-tuning), teacher_epochs, teacher_lr,
kd_alpha, kd_temp. The teacher is trained first in the same job (same train/val data), evaluated on the test set, frozen, and then
guides the student with  (1-a)*CE+Dice + a*T^2*KL(teacher_T || student_T).
Writes <out>/<name>__<fold>__seed<k>/result.json with config, seed, code revision, library versions, per-epoch history and test metrics.
Test data are touched once per model, with the weights of the best validation epoch.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import subprocess
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
    if kind.startswith("adapter_"):          # band adapter in front of a pretrained 3-channel network (mode: bank)
        from .adapter import AdaptedSegmenter
        inner = dict(cfg, model=kind.removeprefix("adapter_"))
        return AdaptedSegmenter(build_model(inner, 3), n_bank=in_ch)
    if kind == "unet":
        return smp.Unet(cfg.get("encoder", "resnet34"), encoder_weights=w, in_channels=in_ch, classes=2)
    if kind == "deeplabv3plus":
        return smp.DeepLabV3Plus(cfg.get("encoder", "resnet34"), encoder_weights=w, in_channels=in_ch, classes=2)
    if kind == "segformer_b0":
        return smp.Segformer("mit_b0", encoder_weights=w, in_channels=in_ch, classes=2)
    if kind == "segformer_b3":
        return smp.Segformer("mit_b3", encoder_weights=w, in_channels=in_ch, classes=2)
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


def forward(model, b, device):
    x = b["image"].to(device, non_blocking=True)
    return model(x, b["avail"].to(device)) if getattr(model, "needs_avail", False) else model(x)


def evaluate(model, refs, cfg, device, batch_size=16, workers=2, examples_dir=None):
    """Metrics per dataset in refs. With examples_dir (final test evaluation only) also saves reliability-diagram bins in the result and
    an examples_<dataset>.npz with the 6 worst and 6 best tiles (by tile IoU, among tiles with >=1% landslide pixels) for failure figures."""
    import torch
    from torch.utils.data import DataLoader

    from . import data
    from .metrics import BinaryMeter
    out = {}
    for name in sorted({r[0] for r in refs}):
        ds = data.LandslideDataset([r for r in refs if r[0] == name], mode=cfg["mode"], augment=False)
        groups = ds._cache[name][2].group.to_numpy()          # event / region id per tile ('unknown' where not exposed)
        meter, by_group, tile_idx, tile_iou, tile_pos = BinaryMeter(), {}, [], [], []
        model.eval()
        with torch.no_grad():
            for b in DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers):
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits = forward(model, b, device)
                p = torch.softmax(logits.float(), 1)[:, 1].cpu().numpy()
                t = b["mask"].numpy()
                meter.update(p, t)
                g = groups[b["index"].numpy()]
                for gname in np.unique(g):
                    if gname != "unknown":
                        by_group.setdefault(gname, BinaryMeter()).update(p[g == gname], t[g == gname])
                if examples_dir is not None:
                    pr, tt = p >= 0.5, t == 1
                    tp, fp, fn = (pr & tt).sum((1, 2)), (pr & ~tt).sum((1, 2)), (~pr & tt).sum((1, 2))
                    tile_idx += b["index"].numpy().tolist(); tile_iou += (tp / np.maximum(tp + fp + fn, 1)).tolist()
                    tile_pos += tt.mean((1, 2)).tolist()
        out[name] = meter.compute()
        if by_group:                                           # per-event / per-region breakdown (for the paper's failure analysis)
            out[name]["by_group"] = {k: m.compute() for k, m in by_group.items()}
        if examples_dir is not None:
            out[name]["reliability"] = meter.reliability()
            ok = [i for i, pz in enumerate(tile_pos) if pz >= 0.01]
            order = sorted(ok, key=lambda i: tile_iou[i])
            pick = order[:6] + order[-6:] if len(order) >= 12 else order
            local = {k: j for j, (_, k) in enumerate(ds.refs)}
            items = [ds[local[tile_idx[i]]] for i in pick]
            if items:
                batch = {k: torch.as_tensor(np.stack([it[k] for it in items])) for k in ("image", "mask", "avail")}
                with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    prob = torch.softmax(forward(model, batch, device).float(), 1)[:, 1].cpu().numpy()
                np.savez_compressed(Path(examples_dir) / f"examples_{name}.npz", tile=np.array([tile_idx[i] for i in pick]),
                                    iou=np.array([tile_iou[i] for i in pick]), image=batch["image"][:, :3].numpy().astype(np.float16),
                                    mask=batch["mask"].numpy().astype(np.uint8), prob=prob.astype(np.float16))
    return out


def fit(model, train_ds, val, cfg, device, seed, epochs, lr, wd, tag, teacher=None):
    """Train `model` on train_ds; return the best-validation-IoU weights. With a frozen `teacher`, adds the distillation term."""
    import torch
    from torch.utils.data import DataLoader

    from .lora import kd_loss
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    nw = cfg.get("num_workers", 2)
    loader = DataLoader(train_ds, batch_size=cfg.get("batch_size", 32), shuffle=True, drop_last=True, num_workers=nw,
                        persistent_workers=nw > 0, generator=torch.Generator().manual_seed(seed))
    history, best, best_state, best_ep, t0 = [], -1.0, None, 0, time.time()
    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for b in loader:
            y = b["mask"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = forward(model, b, device)
                if teacher is not None:
                    with torch.no_grad():
                        t_logits = forward(teacher, b, device)
            loss = loss_fn(logits.float(), y)
            if teacher is not None:
                loss = kd_loss(logits.float(), t_logits.float(), loss, cfg.get("kd_alpha", 0.5), cfg.get("kd_temp", 2.0))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * len(y); n += len(y)
        sched.step()
        v = evaluate(model, val, cfg, device)
        v_iou = float(np.mean([m["iou_landslide"] for m in v.values()]))
        history.append(dict(epoch=ep, train_loss=tot / max(1, n), val_iou=v_iou, seconds=time.time() - t0))
        print(f"[{tag}] ep {ep} loss {tot / max(1, n):.4f} val IoU(landslide) {v_iou:.4f}", flush=True)
        if v_iou > best:
            best, best_state, best_ep = v_iou, copy.deepcopy({k: t.detach().cpu() for k, t in model.state_dict().items()}), ep
    model.load_state_dict(best_state)
    return dict(history=history, best_val_iou=best, best_epoch=best_ep, train_seconds=time.time() - t0)


def profile(model, in_ch, h, device):
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    from .lora import count_params
    total, trainable = count_params(model)
    model.eval()
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        model(torch.zeros(1, in_ch, h, h, device=device))     # avail defaults to all-ones in the adapter wrapper
    return dict(n_params=total, n_trainable=trainable, gflops_at_input=fc.get_total_flops() / 2e9, input_hw=h)


def main(argv=None) -> dict:
    import torch

    from . import data
    from .lora import apply_lora

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
    # max_val caps the per-epoch validation set (checkpoint selection only). The test set is never subsampled unless max_test /
    # max_eval is set explicitly (smoke tests).
    train = sub(splits["train"], cfg.get("max_train"))
    val = sub(splits["val"], cfg.get("max_val", cfg.get("max_eval")))
    test_refs = sub(splits["test"], cfg.get("max_test", cfg.get("max_eval")))

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = data.LandslideDataset(train, mode=cfg["mode"], augment=True, seed=seed, crop=cfg.get("crop"))
    in_ch = train_ds[0]["image"].shape[0]
    h = int(cfg.get("crop") or train_ds[0]["image"].shape[1])

    teacher, teacher_res = None, None
    if cfg.get("teacher_model"):
        prefix = "adapter_" if cfg["mode"] == "bank" else ""
        teacher = build_model(dict(cfg, model=prefix + cfg["teacher_model"]), in_ch).to(device)
        rank = int(cfg.get("teacher_lora_rank", 0))
        if rank > 0:
            inner = teacher.net if hasattr(teacher, "net") else teacher
            n_wrapped = apply_lora(inner.encoder, rank)
            print(f"[teacher] LoRA rank {rank} on {n_wrapped} linear layers", flush=True)
            teacher.to(device)
        fit_t = fit(teacher, train_ds, val, cfg, device, seed, int(cfg.get("teacher_epochs", cfg.get("epochs", 10))),
                    cfg.get("teacher_lr", 2e-4), cfg.get("teacher_weight_decay", 0.01), f"{run_id}:teacher")
        teacher_res = dict(test=evaluate(teacher, test_refs, cfg, device), **fit_t, **profile(teacher, in_ch, h, device),
                           lora_rank=rank, model=cfg["teacher_model"])
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

    model = build_model(cfg, in_ch).to(device)
    fit_s = fit(model, train_ds, val, cfg, device, seed, int(cfg.get("epochs", 30)), cfg.get("lr", 3e-4), cfg.get("weight_decay", 1e-4),
                run_id, teacher=teacher)
    test = evaluate(model, test_refs, cfg, device, examples_dir=out_dir if cfg.get("save_examples") else None)
    prof = profile(model, in_ch, h, device)
    result = dict(config=cfg, seed=seed, run_id=run_id, code_revision=code_revision(), torch=torch.__version__,
                  device=torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu", n_params=prof["n_params"],
                  n_trainable=prof["n_trainable"], gflops_at_input=prof["gflops_at_input"], input_hw=prof["input_hw"],
                  n_train=len(train), n_val=len(val), n_test=len(test_refs), best_epoch=fit_s["best_epoch"],
                  best_val_iou=fit_s["best_val_iou"], history=fit_s["history"], train_seconds=fit_s["train_seconds"], test=test)
    if teacher_res is not None:
        result["teacher"] = teacher_res
    (out_dir / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result["test"][k] for k in result["test"]}, indent=1))
    return result


if __name__ == "__main__":
    main()
