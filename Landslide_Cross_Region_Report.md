# Cross-Region, Lightweight Landslide Segmentation: Literature, Datasets and Study Plan

Prepared 2026-10-10. Based on web search results and abstracts; only the PLoS One landslide paper and the YOLOv11-Lite paper were read in full. Items marked **[verify]** must be checked against the original paper before you cite or rely on them.

---

## 1. Summary

**Topic.** Detect landslides in satellite and UAV imagery with deep learning, and make the model (a) work on regions and sensors it was not trained on, and (b) small and fast enough for field or edge use.

**Starting point.** Sharma et al., *PLoS One* 21(4): e0347324 (2026), "A multi-modal deep learning framework with GAN-based fusion for enhanced landslide detection."
- Four ImageNet CNN backbones (VGG16, DenseNet201, ResNet50, InceptionV3) fused by channel concatenation, followed by a GAN refinement stage.
- Evaluated separately on CAS Landslide, "MS2LandsNet" and GDCLD. F1 = 97.24 / 93.70 / 94.75.
- 210M parameters, 28.6 GFLOPs, 830 MB, ~48 h training on a V100, 88 ms/image.

**The opportunity.** The field has moved from in-benchmark scores to event-isolated, cross-domain evaluation (2025-2026), and to foundation models with parameter-efficient fine-tuning. A model that is heavy and tested only inside each dataset is now behind the state of the art.

**Recommended paper.** A study that (1) trains and tests with leave-one-dataset-out and leave-one-event-out protocols over 5+ public datasets, (2) compares CNN, transformer and foundation-model baselines under the same protocol, (3) distils the best model into an edge-sized student, and (4) reports calibration, false-alarm behaviour and measured edge latency.

**Main risk.** Close competitors appeared in 2026 (Sections 4 and 8). The contribution must be positioned as *efficiency + cross-sensor generalisation + deployment evidence*, not "higher F1".

---

## 2. Literature

### 2.1 Reviews

| Ref | Link | Use |
|---|---|---|
| Jiang, Ma, Mei (2026). Review: Deep learning for potential landslide identification: data, models, applications, challenges, opportunities. *NHESS* 26, 487. 400+ studies | https://nhess.copernicus.org/articles/26/487/2026/ | Lists the gaps you target: imbalance, weak generalisation across regions, compute cost, interpretability, need for broader benchmarks. |
| AI-centric review of deep learning for landslides with remote sensing. *Remote Sensing*, April 2024 | https://doaj.org/article/9df02793f6174279aa5adaacbbc95e65 | Background on EO + DL for landslides. **[verify authors/title]** |
| Ji et al. (2020). Landslide detection from an open satellite imagery and DEM dataset using attention-boosted CNNs. *Landslides* | https://gpcv.whu.edu.cn/data/Bijie_pages.html (dataset page) | Bijie dataset paper. |

### 2.2 Dataset papers (the benchmarks)

| Dataset | Reference | Link |
|---|---|---|
| Landslide4Sense (L4S) | Ghorbanzadeh et al. (2022), outcome of the 2022 Landslide4Sense competition. Winners used Swin, SegFormer, U-Net with hard-example mining, self-training, mix-up | https://arxiv.org/abs/2209.02556 |
| CAS Landslide | CAS Landslide Dataset: large-scale, multisensor. *Scientific Data* (2024); 20,865 images, 9 regions, satellite + UAV | DOI 10.1038/s41597-023-02847-z |
| GDCLD | Fang, Fan et al. (2024). *Earth Syst. Sci. Data* 16, 4817. 9 coseismic events; SegFormer best at 85.06% mIoU | https://doi.org/10.5194/essd-16-4817-2024 |
| HR-GLDD | High-resolution global landslide detector database, *ESSD* 15, 3283 (2023); 10 events, PlanetScope 3 m | https://essd.copernicus.org/articles/15/3283/2023/ |
| LMHLD | Large-scale multi-source high-resolution landslide dataset; 25,365 patches, 5 sensors, 7 events | https://arxiv.org/abs/2502.19866 |
| Sen12Landslides | Spatio-temporal dataset for satellite-based landslide detection; Sentinel-1/2 + DEM, 75,000 annotations, 15 regions | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12606323/ |
| PILD | Physics-informed landslide dataset (DLR Landslide Reference + GDCLD + GLaD4CD + Sen12Landslides harmonised); 7,890 samples, 55 events, event-isolated cross-domain protocol | introduced in https://arxiv.org/abs/2608.09325 |

### 2.3 Methods: cross-domain, foundation models, lightweight

| Ref | Link | Why it matters to you |
|---|---|---|
| Kowsher, Zhan, Chen (2026). Benchmark of segmentation models and adaptation strategies for landslide detection (GDCLD). CNN vs transformer vs foundation models; LoRA/AdaLoRA cut trainable parameters by up to 95% with similar accuracy | https://arxiv.org/abs/2604.16663 | **Closest competitor.** Single dataset (GDCLD). You extend to multi-dataset, cross-sensor and distillation. |
| Liu, Kwan, Wu, Li (2026). GeoPhysAdapter: scale-matched geophysical adaptation for cross-domain landslide mapping with vision foundation models. PILD, event-isolated; false alarms from foundation models in unseen regions | https://arxiv.org/abs/2608.09325 | **Closest competitor on cross-domain.** Uses terrain, material and rainfall priors. Your angle is compact models rather than priors. |
| LandsDANet: unsupervised domain adaptation for cross-domain landslide mapping (Rare Class Sampling, Wallis filter, contrastive loss). *Remote Sensing* 18(2):286 (2026) | https://www.mdpi.com/2072-4292/18/2/286 | Domain-adaptation baseline. |
| Domain-adaptation landslide work, *Land* 13(7):928 (2024) | https://www.mdpi.com/2073-445X/13/7/928 | Earlier UDA baseline. **[verify exact title]** |
| WILD-SAM: phase-aware expert adaptation of SAM for landslide detection in wrapped InSAR interferograms (2026) | https://arxiv.org/abs/2604.14540 | SAM adaptation, SAR modality. |
| Burange et al. (2025). Landslide detection and mapping across multi-source satellite data and regions (U-Net, DeepLabV3+, ResNet on L4S) | https://arxiv.org/abs/2507.01123 | Simple baseline numbers on L4S. |
| A comparative analysis of CNN-based models for landslide detection (2024) | https://arxiv.org/abs/2408.01692 | CNN baselines. |
| Ge et al. (2023). Lightweight landslide detection with CenterNet + Ghost-BottleNeck, YOLOv5x teacher distillation. *Remote Sensing* | https://doaj.org/article/243d0ca5e5c548f9961132a6a8373a8e | Only distillation-for-landslides precedent found. Detection, not segmentation, and one dataset. |
| Prithvi-EO-2.0 (IBM-NASA) adapted to landslide mapping; outperforms U-Net, Segformer, SwinV2-B, SatMAE and TerraMind in a 2025/26 study | search "Prithvi-EO-2.0 landslide mapping" **[verify citation]** | Foundation-model teacher candidate. |
| TerraFM (2025): unified multisensor EO foundation model; reported L4S mIoU 70.8, landslide IoU 43.1 | https://arxiv.org/abs/2506.06281 | Teacher candidate and L4S reference number. |
| DisasterNets: embedding machine learning in disaster mapping | https://arxiv.org/abs/2306.09815 | Background on disaster mapping pipelines. |

---

## 3. Dataset catalogue and download links

| Dataset | Sensors / resolution | Size | Modality | Link |
|---|---|---|---|---|
| Landslide4Sense | Sentinel-2 (B1-B12) + ALOS PALSAR slope and DEM, ~10 m | 3,799 train / 245 val / 800 test patches, 128x128, 14 bands | Multispectral + terrain | https://zenodo.org/records/10463239 ; HF: https://huggingface.co/datasets/harshinde/LandSlide4Sense |
| GDCLD | PlanetScope 3 m, Gaofen-6 2 m, Map World 0.5 m, UAV 0.2 m | 11,162 train + 4,459 val patches of 1024x1024; 9 events; ~1.39e9 landslide pixels; 8.01% positive | RGB | https://doi.org/10.5281/zenodo.13612636 |
| CAS Landslide | Satellite + UAV, 9 regions | 20,865 images (the PLoS paper used a 1,766-image GF-2 subset) | RGB / multispectral | Zenodo record 10294997 (as cited in the PLoS paper) **[verify]**; paper DOI 10.1038/s41597-023-02847-z |
| Bijie | TripleSat 0.8 m, plus DEM | 770 landslide + 2,003 non-landslide images | RGB + DEM | https://gpcv.whu.edu.cn/data/Bijie_pages.html |
| HR-GLDD | PlanetScope 3 m, 4 bands | 10 events (5 rainfall, 5 earthquake) | RGB + NIR | https://doi.org/10.5281/zenodo.7189381 |
| LMHLD | 5 sensors, 7 events (Wenchuan, Rio, Gorkha, Jiuzhaigou, Taiwan, Hokkaido, Emilia-Romagna) | 25,365 patches | RGB | https://doi.org/10.5281/zenodo.11424988 |
| Sen12Landslides | Sentinel-1 SAR, Sentinel-2, Copernicus DEM | 75,000 annotations, 15 regions, 12,000+ patches | SAR + optical + DEM | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12606323/ (follow the data link there) |
| PILD | Harmonised from DLR Landslide Reference, GDCLD, GLaD4CD, Sen12Landslides | 7,890 samples, 55 events | mixed | Link inside https://arxiv.org/abs/2608.09325 |

**Flags to check before using.**
- The PLoS paper gives the "MS2LandsNet" source as a Kaggle URL named `tekbahadurkshetri/landslide4sense`. That looks like Landslide4Sense, not a separate dataset **[verify]**.
- The PLoS paper describes GDCLD as "9,000+ patches" while the ESSD paper lists 11,162 train + 4,459 val patches **[verify which subset they used]**.
- Input bands differ (RGB vs 4-band vs 14-band vs SAR). You need a channel-agnostic design or a common-band subset.

---

## 4. What the existing paper does and where it falls short

| Aspect | Existing PLoS paper | Gap |
|---|---|---|
| Evaluation | Each dataset trained and tested alone; reports "cross-dataset" results | No event-isolated or leave-one-region-out split, so scores may reflect spatial leakage. Newer work (GeoPhysAdapter, PILD) treats this as the key test. |
| Model | Four ImageNet backbones concatenated + GAN; 210M params, 28.6 GFLOPs | Heavy, hard to deploy, no ablation showing all four are needed. |
| Baselines | CNN fusions | Missing U-Net/DeepLabV3+, SegFormer, Swin, Prithvi/TerraFM, and the L4S competition winners. |
| Adaptation | Full training for ~48 h | No parameter-efficient fine-tuning (LoRA), where the 2026 benchmark reports ~95% fewer trainable parameters. |
| Efficiency evidence | Inference 88 ms on V100 | No measurement on edge hardware (Jetson, CPU, mobile). |
| Reliability | Accuracy metrics | No calibration, uncertainty, false-alarm analysis or statistical tests over seeds. |
| Stated limits | None listed | Reviewers will ask for them. |

---

## 5. How to improve (the contribution)

1. **Protocol first.** Adopt event-isolated leave-one-dataset-out and leave-one-event-out evaluation (PILD-style). This alone separates the work from in-benchmark papers.
2. **Fair baseline suite under one protocol.** U-Net, DeepLabV3+, SegFormer-B0..B5, Swin-UperNet, Prithvi-EO-2.0 and/or TerraFM with full fine-tuning and LoRA, plus the existing fusion model.
3. **Sensor-agnostic input.** A learned band adapter so RGB, 4-band and 14-band data share one network, evaluated by zero-shot transfer between sensors.
4. **Teacher-student distillation.** Best foundation or transformer model as teacher; student SegFormer-B0/B1 or MobileNetV3-UNet. Compare logit, feature and boundary-aware distillation.
5. **Cross-domain tricks as ablations.** Style/Wallis normalisation, rare-class sampling, self-training on unlabeled target images, test-time adaptation. Compare with LandsDANet-style methods.
6. **Reliability.** Expected calibration error, Monte-Carlo or deep-ensemble uncertainty, false-alarm rate per km2, and the object-level false-positive analysis that GeoPhysAdapter uses.
7. **Deployment evidence.** Latency, memory and energy on a Jetson-class device and a CPU. Report parameters, GFLOPs, and mIoU/F1 vs latency Pareto plot.
8. **Open release.** Code, splits and trained weights.

---

## 6. Experimental plan

**Tasks.** Binary landslide segmentation (primary); optional object-level detection metrics.

**Splits.**
- A: in-dataset (for comparison with prior work).
- B: leave-one-dataset-out (train on all but one of L4S, GDCLD, CAS, HR-GLDD, LMHLD, Bijie).
- C: leave-one-event-out inside GDCLD, LMHLD and PILD.
- D: sensor transfer (e.g., train on PlanetScope sets, test on Sentinel-2 L4S).

**Metrics.** F1, IoU (landslide class), mIoU, precision/recall, PR-AUC, ECE, false alarms per km2, boundary F-score. Report mean +- std over at least 5 seeds; paired tests (Wilcoxon) or bootstrap CIs.

**Ablations.** Band adapter on/off; teacher choice; distillation type; LoRA rank; training-set size (10/25/50/100%); with/without GAN refinement (tests the original paper's idea).

**Efficiency table.** Params, GFLOPs, memory, latency on A100/V100, Jetson, CPU; training hours.

**Compute estimate.** Teacher LoRA fine-tuning and a dozen baselines is feasible on one 24-48 GB GPU in a few weeks. If you have no GPU, restrict to SegFormer-B0..B3 and LoRA only.

---

## 7. Timeline (about 12-14 weeks)

| Weeks | Work |
|---|---|
| 1-2 | Download datasets, unify formats, build splits, reproduce baselines on L4S and GDCLD |
| 3-5 | Full baseline suite under protocols A-D |
| 6-8 | Foundation-model fine-tuning (LoRA) and band adapter |
| 9-10 | Distillation, ablations, calibration and uncertainty |
| 11 | Edge benchmarking |
| 12-14 | Writing, figures, code release, journal selection |

---

## 8. Target journals (approximate impact factors, check JCR)

| Journal | Approx. IF | Fit |
|---|---|---|
| Int. J. Applied Earth Observation and Geoinformation | ~7-8 | Good fit for method + benchmark |
| IEEE Trans. Geoscience and Remote Sensing | ~8 | Good, strong competition |
| ISPRS J. Photogrammetry and Remote Sensing | ~10 | Stretch; needs strong novelty |
| Landslides (Springer) | ~5-6 | Safer fallback |
| Remote Sensing / IEEE JSTARS | ~4-5 | Fallback |
| Computers & Geosciences | ~4-5 | Fallback |

---

## 9. Risks and how to handle them

- **Novelty overlap.** 2604.16663 (PEFT benchmark on GDCLD) and 2608.09325 (cross-domain foundation-model adaptation) are close. Differentiate by multi-dataset protocol, sensor-agnostic input, distillation to edge and deployment measurements. Re-run a literature search just before submission, since this area moves monthly.
- **Data leakage.** Overlapping tiles from one event across train/test inflate scores. Split by event or region and state it.
- **Label inconsistency across datasets** (polygon quality, scale, positive ratio from 8% to higher). Report per-dataset results and handle imbalance explicitly.
- **Compute.** See Section 6.
- **Reproducibility of prior numbers.** Re-run baselines yourself rather than quoting published scores.

---

## 10. Immediate next steps

1. Download L4S, GDCLD and LMHLD first (all on Zenodo); reproduce a SegFormer-B0 baseline on each.
2. Resolve the verification flags in Section 3 (MS2LandsNet, CAS subset, GDCLD subset).
3. Read in full: 2604.16663, 2608.09325, the NHESS 2026 review, and the GDCLD and L4S papers.
4. Decide GPU availability; this sets the teacher choice.
