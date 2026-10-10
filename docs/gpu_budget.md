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

Measured throughput (T4, AMP, batch 32): L4S 128 px, 3,799 tiles -> ~9-11 s/epoch. Extrapolation to GDCLD (256 px, 11,162 tiles,
~12x the pixels) is ~2 min/epoch -> ~40 min per 20-epoch run: an estimate until measured.

Rule: no kernel is launched without writing its estimated hours here first; the remaining budget is re-checked after each.
