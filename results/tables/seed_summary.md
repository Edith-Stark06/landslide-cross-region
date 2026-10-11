### B-strict landslide IoU: mean ± std over seeds (n = number of seeds)

| Method | l4s | hrgldd | gdcld | lmhld | cas | 5-dataset mean per seed |
|---|---|---|---|---|---|---|
| U-Net R34 | 17.8 ± 6.2 (n=3) | 30.4 ± 2.5 (n=3) | 17.8 ± 2.6 (n=3) | 17.5 ± 1.8 (n=3) | 3.9 ± 2.1 (n=3) | 17.5 ± 2.9 (n=3) |
| SegFormer-B0 (RGB) | 13.4 ± 9.4 (n=3) | 36.9 ± 0.6 (n=3) | 28.5 ± 0.2 (n=3) | 21.8 ± 6.2 (n=3) | 21.2 ± 2.1 (n=3) | 24.4 ± 2.9 (n=3) |
| SegFormer-B0 + band adapter | 13.0 ± 5.7 (n=3) | 39.4 ± 1.7 (n=3) | 29.6 ± 0.8 (n=3) | 19.3 ± 3.2 (n=3) | 17.8 ± 2.5 (n=3) | 23.8 ± 1.8 (n=3) |
| SegFormer-B0 student (KD) | 9.6 ± 0.8 (n=2) | 33.8 (n=1) | 27.4 (n=1) | 30.1 (n=1) | 15.1 ± 2.8 (n=2) | 22.9 (n=1) |

### B-strict F1: mean ± std over seeds (n = number of seeds)

| Method | l4s | hrgldd | gdcld | lmhld | cas | 5-dataset mean per seed |
|---|---|---|---|---|---|---|
| U-Net R34 | 29.9 ± 9.2 (n=3) | 46.6 ± 2.9 (n=3) | 30.2 ± 3.7 (n=3) | 29.8 ± 2.6 (n=3) | 7.5 ± 3.8 (n=3) | 28.8 ± 4.1 (n=3) |
| SegFormer-B0 (RGB) | 22.8 ± 14.0 (n=3) | 53.9 ± 0.7 (n=3) | 44.4 ± 0.2 (n=3) | 35.5 ± 8.4 (n=3) | 35.0 ± 2.9 (n=3) | 38.3 ± 4.2 (n=3) |
| SegFormer-B0 + band adapter | 22.8 ± 8.7 (n=3) | 56.5 ± 1.7 (n=3) | 45.6 ± 1.0 (n=3) | 32.2 ± 4.5 (n=3) | 30.1 ± 3.6 (n=3) | 37.5 ± 2.6 (n=3) |
| SegFormer-B0 student (KD) | 17.6 ± 1.4 (n=2) | 50.5 (n=1) | 43.0 (n=1) | 46.3 (n=1) | 26.2 ± 4.3 (n=2) | 36.3 (n=1) |

### B-strict ECE: mean ± std over seeds (n = number of seeds)

| Method | l4s | hrgldd | gdcld | lmhld | cas | 5-dataset mean per seed |
|---|---|---|---|---|---|---|
| U-Net R34 | 0.016 ± 0.009 (n=3) | 0.058 ± 0.006 (n=3) | 0.052 ± 0.003 (n=3) | 0.168 ± 0.008 (n=3) | 0.142 ± 0.003 (n=3) | 0.087 ± 0.003 (n=3) |
| SegFormer-B0 (RGB) | 0.026 ± 0.022 (n=3) | 0.055 ± 0.002 (n=3) | 0.044 ± 0.001 (n=3) | 0.044 ± 0.007 (n=3) | 0.108 ± 0.005 (n=3) | 0.055 ± 0.006 (n=3) |
| SegFormer-B0 + band adapter | 0.036 ± 0.029 (n=3) | 0.051 ± 0.003 (n=3) | 0.041 ± 0.002 (n=3) | 0.058 ± 0.016 (n=3) | 0.112 ± 0.002 (n=3) | 0.060 ± 0.009 (n=3) |
| SegFormer-B0 student (KD) | 0.034 ± 0.019 (n=2) | 0.056 (n=1) | 0.041 (n=1) | 0.027 (n=1) | 0.120 ± 0.001 (n=2) | 0.053 (n=1) |

### Paired comparisons on landslide IoU (seed-averaged per fold; difference = A - B)

- **SegFormer-B0 (RGB) − U-Net R34**: l4s -4.4 (n=3,3); hrgldd +6.5 (n=3,3); gdcld +10.7 (n=3,3); lmhld +4.3 (n=3,3); cas +17.3 (n=3,3); mean +6.9, positive in 4/5 folds
- **SegFormer-B0 + band adapter − SegFormer-B0 (RGB)**: l4s -0.3 (n=3,3); hrgldd +2.5 (n=3,3); gdcld +1.0 (n=3,3); lmhld -2.5 (n=3,3); cas -3.5 (n=3,3); mean -0.5, positive in 2/5 folds
- **SegFormer-B0 student (KD) − SegFormer-B0 (RGB)**: l4s -3.7 (n=2,3); hrgldd -3.1 (n=1,3); gdcld -1.1 (n=1,3); lmhld +8.3 (n=1,3); cas -6.1 (n=2,3); mean -1.1, positive in 1/5 folds

With 5 folds the smallest possible two-sided Wilcoxon p-value is 0.0625, so a paired test over folds cannot reach p < 0.05; we report differences and sign counts, not significance claims.

### Distilled student: seed 0 vs seed 1 on the two folds that collapsed (landslide IoU)

- l4s: seed 0: 10.2, seed 1: 9.1
- cas: seed 0: 13.1, seed 1: 17.1
