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
| U-Net R34 | 19.5 | 29.9 | 17.5 | 18.2 | 2.3 |
| SegFormer-B0 | 24.1 | 36.7 | 28.6 | 22.8 | 22.1 |
| SegFormer-B0 + band adapter | 19.5 | 41.1 | 28.9 | 18.1 | 20.6 |
| SegFormer-B0 student (distilled) | n/a | n/a | n/a | n/a | n/a |

### Table: cross-dataset, protocol B-strict (event-isolated leave-one-dataset-out), F1

| Model | L4S | HR-GLDD | GDCLD | LMHLD | CAS |
|---|---|---|---|---|---|
| U-Net R34 | 32.6 | 46.0 | 29.8 | 30.7 | 4.4 |
| SegFormer-B0 | 38.9 | 53.7 | 44.4 | 37.1 | 36.2 |
| SegFormer-B0 + band adapter | 32.7 | 58.3 | 44.8 | 30.7 | 34.1 |
| SegFormer-B0 student (distilled) | n/a | n/a | n/a | n/a | n/a |

### Table: cross-dataset, protocol B-strict (event-isolated leave-one-dataset-out), ECE

| Model | L4S | HR-GLDD | GDCLD | LMHLD | CAS |
|---|---|---|---|---|---|
| U-Net R34 | 0.012 | 0.062 | 0.052 | 0.178 | 0.145 |
| SegFormer-B0 | 0.002 | 0.052 | 0.042 | 0.042 | 0.108 |
| SegFormer-B0 + band adapter | 0.004 | 0.050 | 0.039 | 0.044 | 0.111 |
| SegFormer-B0 student (distilled) | n/a | n/a | n/a | n/a | n/a |

### Table: effect of event overlap, SegFormer-B0, landslide IoU (protocol B vs size-matched B vs B-strict)

| Held-out | Plain B (full training set) | Plain B cut to B-strict size | B-strict |
|---|---|---|---|
| L4S | 27.3 | 22.8 | 24.1 |
| HR-GLDD | n/a (old HR-GLDD channel order) | 38.1 | 36.7 |
| GDCLD | 45.3 | 38.9 | 28.6 |
| LMHLD | 24.9 | 19.4 | 22.8 |
| CAS | 48.3 | 46.7 | 22.1 |

### Table: per-region / per-event landslide IoU, SegFormer-B0 (plain B cut to B-strict size, seed 0)

- **CAS**: Palu 1.9; Lombok 9.6; Hokkaido Iburi-Tobu 19.6; Tiburon Peninsula 26.2; Wenchuan 26.7; Moxitaidi 37.7; Longxi River 41.8; Moxi town 43.5; Mengdong Township 50.2; Jiuzhai valley 55.5
- **GDCLD**: Mesetas 2.2; Lushan 13.2; Sumarta 31.1; Palu 31.8
- **LMHLD**: Wenchuan_China 4.5; Gorkha_Nepal 19.1; Taiwan_China 19.2; Jiuzhaigou_China 21.2; Emilia-Romagna_Italy 22.5; Hokkaido_Japan 28.3; Rio de Janeiro_Brazil 32.7
