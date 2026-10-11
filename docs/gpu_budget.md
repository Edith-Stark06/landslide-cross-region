# Kaggle GPU budget

Hard cap set by the user: **30 GPU hours per week**, and the whole project must fit in it. Session wall time is what Kaggle
counts, so each kernel's total runtime is logged here (from the kernel's own `done; total seconds` line plus ~1 min start-up).
Smoke tests before the first real run used about 0.1 h in total (3 short kernels; the first one hung for hours before it was
deleted, so its true quota use is unknown and is counted below as the worst case).

| Date | Kernel | Jobs | Measured wall time | Cumulative |
|---|---|---|---|---|
| 2026-10-10/11 | smoke (hung, deleted) | 1 | unknown, up to ~6 h before deletion | <= 6 h (worst case) |
| 2026-10-11 | smoke2, smoke3 | 2 | ~0.05 h | |
| 2026-10-11 | l4s-baselines (U-Net, DeepLabV3+, SegFormer-B0, 50 ep, 1 seed each) | 3 | 1572 s job time + ~1 min start-up = ~0.45 h (estimate was 1.5-2.5 h; actual training 458-574 s per model) | ~0.5 h + the hung smoke (<= 6 h) |

| 2026-10-11 02:59 | gdcld-baselines (3 models, 20 ep, 256 px, 1 seed each) | 3 | 5,500 s job time (U-Net 2,269 s, DeepLabV3+ 1,720 s, SegFormer-B0 1,462 s train) + ~1 min start-up = ~1.55 h (estimate 2 h) | ~2.1 h + the hung smoke (<= 6 h) |

Measured throughput (T4, AMP, batch 32): L4S 128 px, 3,799 tiles -> ~9-11 s/epoch. Extrapolation to GDCLD (256 px, 11,162 tiles,
~12x the pixels) is ~2 min/epoch -> ~40 min per 20-epoch run: an estimate until measured.

Rule: no kernel is launched without writing its estimated hours here first; the remaining budget is re-checked after each.


| 2026-10-11 | gdcld-e60 (SegFormer-B0, 60 ep, 1 seed; under-training check) | 1 | 4,157 s job time (4,131 s train) + ~1 min start-up = ~1.17 h (estimate 1.3 h) | ~3.3 h + the hung smoke (<= 6 h) |

| 2026-10-11 | xd-segformer (SegFormer-B0, protocol B-strict then B, 5 folds each, 10 ep, 128 px crops, 1 seed) | 10 | LAUNCHED. Estimate ~1.9 h (from measured ~11 Mpx/s training); per-fold cap 2,400 s | pending |
| 2026-10-11 | xd-unet (U-Net, protocol B-strict, 5 folds, 10 ep) | 5 | LAUNCHED. Estimate ~0.9 h (U-Net measured 1.55x slower than SegFormer-B0); per-fold cap 2,400 s | pending |
