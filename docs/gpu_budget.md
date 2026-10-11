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

| 2026-10-11 | xd-segformer (SegFormer-B0, protocol B-strict then B, 5 folds each, 10 ep, 128 px crops, 1 seed) | 10 | 6,320 s job time + ~1 min start-up = ~1.78 h (estimate 1.9 h); B-strict 5 folds 1,806 s + B 5 folds 4,514 s | 5.9 h |
| 2026-10-11 | xd-unet (U-Net, protocol B-strict, 5 folds, 10 ep) | 5 | 2,635 s job time + ~1 min start-up = ~0.75 h (estimate 0.9 h); folds took 245-669 s of training | (see running total below) |
| 2026-10-11 | xd kernels, failed launches (dataset list passed as one string; 3 pushes) | - | ~0.05 h each, ~0.1 h total, jobs failed within seconds | +0.1 h |


Running total of measured kernel time this week (2026-10-11): smoke 0.05 + l4s 0.45 + gdcld 1.55 + gdcld-e60 1.17 + failed launches 0.1 + xd-unet 0.75 + xd-segformer 1.78 = **5.85 h**, plus the hung first smoke kernel (unknown, <= 6 h). Remaining of 30 h: 24.1 h, or 18.1 h if the hung kernel is counted in full. Check Kaggle's own quota page for the true figure.

