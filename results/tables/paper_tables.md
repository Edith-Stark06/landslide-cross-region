### Table: in-dataset baselines (protocol A; official splits; test = official test set)

| Dataset | Input | Model | Params (M) | GFLOPs | mIoU | Landslide IoU | F1 | Precision | Recall | PR-AUC | ECE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L4S | 14 bands | U-Net R34 | 24.5 | 2.1 | 73.3 | 48.0 | 64.9 | 62.9 | 67.1 | 0.673 | 0.010 |
| L4S | 14 bands | DeepLabV3+ R34 | 22.5 | 2.1 | 71.9 | 45.3 | 62.3 | 58.3 | 66.9 | 0.636 | 0.011 |
| L4S | 14 bands | SegFormer-B0 | 3.7 | 0.4 | 72.6 | 46.7 | 63.7 | 57.3 | 71.6 | 0.657 | 0.010 |
| L4S | RGB | U-Net R34 | 24.4 | 2.0 | 68.2 | 37.9 | 54.9 | 71.7 | 44.5 | 0.572 | 0.011 |
| L4S | RGB | SegFormer-B0 | 3.7 | 0.4 | 68.8 | 39.2 | 56.3 | 63.1 | 50.8 | 0.556 | 0.011 |
| L4S | bank (adapter) | SegFormer-B0 + adapter | 3.7 | 0.4 | 70.2 | 42.2 | 59.4 | 51.5 | 70.2 | 0.604 | 0.011 |
| GDCLD | RGB | U-Net R34 | 24.4 | 7.8 | 69.0 | 38.7 | 55.8 | 68.5 | 47.1 | 0.604 | 0.005 |
| GDCLD | RGB | DeepLabV3+ R34 | 22.4 | 7.9 | 67.0 | 34.8 | 51.7 | 63.2 | 43.7 | 0.562 | 0.005 |
| GDCLD | RGB | SegFormer-B0 | 3.7 | 1.7 | 70.1 | 40.9 | 58.1 | 72.6 | 48.4 | 0.654 | 0.004 |
| GDCLD | RGB | SegFormer-B0 (60 ep) | 3.7 | 1.7 | 69.6 | 40.0 | 57.1 | 61.2 | 53.6 | 0.529 | 0.005 |

### Table: cross-dataset, protocol B-strict (event-isolated leave-one-dataset-out), landslide IoU

| Model | L4S | HR-GLDD | GDCLD | LMHLD | CAS |
|---|---|---|---|---|---|
| U-Net R34 | 17.8 ± 6.2 (n=3) | 30.4 ± 2.5 (n=3) | 17.8 ± 2.6 (n=3) | 17.5 ± 1.8 (n=3) | 3.9 ± 2.1 (n=3) |
| SegFormer-B0 | 13.4 ± 9.4 (n=3) | 36.9 ± 0.6 (n=3) | 28.5 ± 0.2 (n=3) | 21.8 ± 6.2 (n=3) | 21.2 ± 2.1 (n=3) |
| SegFormer-B0 + band adapter | 13.0 ± 5.7 (n=3) | 39.4 ± 1.7 (n=3) | 29.6 ± 0.8 (n=3) | 19.3 ± 3.2 (n=3) | 17.8 ± 2.5 (n=3) |
| SegFormer-B0 student (distilled) | 9.6 ± 0.8 (n=2) | 33.8 | 27.4 | 30.1 | 15.1 ± 2.8 (n=2) |

### Table: cross-dataset, protocol B-strict (event-isolated leave-one-dataset-out), F1

| Model | L4S | HR-GLDD | GDCLD | LMHLD | CAS |
|---|---|---|---|---|---|
| U-Net R34 | 29.9 ± 9.2 (n=3) | 46.6 ± 2.9 (n=3) | 30.2 ± 3.7 (n=3) | 29.8 ± 2.6 (n=3) | 7.5 ± 3.8 (n=3) |
| SegFormer-B0 | 22.8 ± 14.0 (n=3) | 53.9 ± 0.7 (n=3) | 44.4 ± 0.2 (n=3) | 35.5 ± 8.4 (n=3) | 35.0 ± 2.9 (n=3) |
| SegFormer-B0 + band adapter | 22.8 ± 8.7 (n=3) | 56.5 ± 1.7 (n=3) | 45.6 ± 1.0 (n=3) | 32.2 ± 4.5 (n=3) | 30.1 ± 3.6 (n=3) |
| SegFormer-B0 student (distilled) | 17.6 ± 1.4 (n=2) | 50.5 | 43.0 | 46.3 | 26.2 ± 4.3 (n=2) |

### Table: cross-dataset, protocol B-strict (event-isolated leave-one-dataset-out), ECE

| Model | L4S | HR-GLDD | GDCLD | LMHLD | CAS |
|---|---|---|---|---|---|
| U-Net R34 | 0.016 ± 0.009 (n=3) | 0.058 ± 0.006 (n=3) | 0.052 ± 0.003 (n=3) | 0.168 ± 0.008 (n=3) | 0.142 ± 0.003 (n=3) |
| SegFormer-B0 | 0.026 ± 0.022 (n=3) | 0.055 ± 0.002 (n=3) | 0.044 ± 0.001 (n=3) | 0.044 ± 0.007 (n=3) | 0.108 ± 0.005 (n=3) |
| SegFormer-B0 + band adapter | 0.036 ± 0.029 (n=3) | 0.051 ± 0.003 (n=3) | 0.041 ± 0.002 (n=3) | 0.058 ± 0.016 (n=3) | 0.112 ± 0.002 (n=3) |
| SegFormer-B0 student (distilled) | 0.034 ± 0.019 (n=2) | 0.056 | 0.041 | 0.027 | 0.120 ± 0.001 (n=2) |

### Table: effect of event overlap, SegFormer-B0, landslide IoU (protocol B vs size-matched B vs B-strict)

| Held-out | Plain B (full training set) | Plain B cut to B-strict size | B-strict |
|---|---|---|---|
| L4S | 27.3 | 22.8 | 13.4 ± 9.4 (n=3) |
| HR-GLDD | n/a (old HR-GLDD channel order) | 38.1 | 36.9 ± 0.6 (n=3) |
| GDCLD | 45.3 | 38.9 | 28.5 ± 0.2 (n=3) |
| LMHLD | 24.9 | 19.4 | 21.8 ± 6.2 (n=3) |
| CAS | 48.3 | 46.7 | 21.2 ± 2.1 (n=3) |

### Table: teacher (SegFormer-B3 + LoRA) vs distilled student (SegFormer-B0), B-strict, landslide IoU

| Held-out | Teacher | Student (KD) | Student (no KD) |
|---|---|---|---|
| L4S | 26.1 | 9.6 ± 0.8 (n=2) | 13.4 ± 9.4 (n=3) |
| HR-GLDD | 38.6 | 33.8 | 36.9 ± 0.6 (n=3) |
| GDCLD | 33.3 | 27.4 | 28.5 ± 0.2 (n=3) |
| LMHLD | 30.1 | 30.1 | 21.8 ± 6.2 (n=3) |
| CAS | 27.9 | 15.1 ± 2.8 (n=2) | 21.2 ± 2.1 (n=3) |

### Table: per-region / per-event landslide IoU, SegFormer-B0 (plain B cut to B-strict size, seed 0)

- **CAS**: Palu 1.9; Lombok 9.6; Hokkaido Iburi-Tobu 19.6; Tiburon Peninsula 26.2; Wenchuan 26.7; Moxitaidi 37.7; Longxi River 41.8; Moxi town 43.5; Mengdong Township 50.2; Jiuzhai valley 55.5
- **GDCLD**: Mesetas 2.2; Lushan 13.2; Sumarta 31.1; Palu 31.8
- **LMHLD**: Wenchuan_China 4.5; Gorkha_Nepal 19.1; Taiwan_China 19.2; Jiuzhaigou_China 21.2; Emilia-Romagna_Italy 22.5; Hokkaido_Japan 28.3; Rio de Janeiro_Brazil 32.7
