# Verification log

Date: 2026-10-10. Resolves the **[verify]** flags in `Landslide_Cross_Region_Report.md`.

**Method and limits.** Pages were read with a web-fetch tool that returns a model-written summary of the page, not raw text. Search results are snippets. Numbers marked *(read)* were returned from the cited page itself. Numbers marked *(snippet)* came from a search-result summary and must be re-checked against the PDF before they go in the paper. Where a publisher blocked access (HTTP 403 / login redirect) this is stated. Anything not confirmed is marked **UNVERIFIED**.

---

## 1. Does "MS2LandsNet" exist as its own dataset?

**Answer: No evidence that it is a dataset. It is a model name, and the Kaggle link is a re-upload of Landslide4Sense.**

- PLoS paper, reference [11] *(read)*: Lu W, Hu Y, Shao W, Wang H, Zhang Z, Wang M. "A multiscale feature fusion enhanced CNN with the multiscale channel attention mechanism for efficient landslide detection (MS2LandsNet) using medium-resolution remote sensing data." *Int. J. Digital Earth* 2024;17(1). https://doi.org/10.1080/17538947.2023.2300731. So "MS2LandsNet" is the **network** introduced in that paper.
- The PLoS paper gives the data source as `https://www.kaggle.com/datasets/tekbahadurkshetri/landslide4sense` and describes it as Sentinel-2, 10 m, "several thousand" tiles, 70/30 split *(read)*.
- Kaggle API metadata for that dataset *(read, via `kaggle datasets metadata`)*: title "Landslide4Sense", subtitle "Landslide mapping from Sentinel2 dataset.", "Licensed by IARAI GmbH.", licence field "other", source stated as https://github.com/iarai/Landslide4Sense-2022, related papers arXiv 2206.00515 and IEEE 9944085. Files are `.h5` tiles of 1,837,056 bytes each (consistent with 128x128x14 float64 plus header; this is my arithmetic, not a stated fact).
- Not confirmed: whether Lu et al. (2024) themselves used Landslide4Sense or a different Sentinel-2 set (**UNVERIFIED**; I did not open that paper). Not confirmed: that the Kaggle copy is byte-identical to the Zenodo release (to be checked by tile counts and hashes in Step 2).
- Search for "MS2LandsNet dataset" returned no dataset page.

**Consequence for the plan:** treat "MS2LandsNet" as Landslide4Sense. Do not list it as a sixth dataset.

## 2. Which GDCLD subset did the PLoS paper use?

**Answer: UNVERIFIED. The PLoS text does not identify a subset, and its number does not match either ESSD split.**

- PLoS paper *(read)*: GDCLD described as "over 9,000 high-resolution image patches"; sensors "PlanetScope, Gaofen-6, MapWorld and UAV orthomosaics"; own 70/30 train-test split with 15% of training for validation. No download URL or DOI for GDCLD was found in the parts of the page I could read. The text cites GDCLD as "Wang et al. [52]" while the reference list gives it as [12] Fang et al.; reference 52 is titled GDSNet. This looks like a citation mix-up in the PLoS paper.
- ESSD paper (Fang et al., *Earth Syst. Sci. Data* 16, 4817-4842, 2024) *(read)*: 11,162 training + 4,459 validation images, each 1024x1024 8-bit RGB with binary masks (0/1), 3:1 ratio, plus a test set of seven large images from four events. Nine events: Jiuzhaigou 2017, Mainling 2017, Hokkaido 2018, Palu 2018, Mesetas 2019, Nippes 2021, Sumatra 2022, Lushan 2022, Luding 2022. "Landslide samples 8.01%" is stated as a **sample ratio, not a pixel ratio**; about 1.39e9 landslide pixels. Data: https://doi.org/10.5281/zenodo.13612636 (code: https://doi.org/10.5281/zenodo.13956757). Article licence CC BY 4.0; dataset licence not stated in the paper (check the Zenodo record in Step 2). Some source imagery (UAV, Map World, Gaofen-6) is restricted and PlanetScope must come from Planet.
- 11,162 + 4,459 = 15,621 (my sum). "Over 9,000" matches none of 11,162, 4,459 or 15,621. The PLoS paper used some subset of unknown composition.
- Correction to the report's Section 4/10 wording: the report says the 8.01% figure is a positive ratio. It is a **share of samples**; the pixel positive ratio must be computed by me in Step 2.

## 3. CAS Landslide download location

**Answer: Confirmed.**

- Zenodo record https://zenodo.org/records/10294997 *(read)*: "CAS Landslide Dataset: A Large-Scale and Multisensor Dataset for Deep Learning-Based Landslide Detection", v1, 8 Dec 2023, DOI 10.5281/zenodo.10294997, licence **CC BY-NC 4.0**, 20,865 images, nine regions, 9.2 GB total, 17 zip files (e.g. Hokkaido Iburi-Tobu 745.7 MB, Jiuzhai valley UAV-0.2m 2.1 GB, Moxi town UAV-0.2m 1.0 GB, Longxi River SAT 672.8 MB, Moxitaidi SAT 398.4 MB, Tiburon Peninsula Sentinel 383.2 MB, ...). Authors: Yulin Xu, Chaojun Ouyang, Qingsong Xu, Dongpo Wang, Bo Zhao, Yutao Luo.
- Paper: Xu Y et al., *Sci Data* 2024;11(1):12, doi 10.1038/s41597-023-02847-z, PMID 38168493 *(PLoS reference [10] and search results agree)*. The Nature page itself blocked automated access, so the paper's own data-availability text was **not read**; the Zenodo page and the PLoS reference list are the evidence.
- PLoS paper *(read)* used "1,766 high-resolution images" (1,236 / 265 / 265) and calls the sensor "GF-2 ... and UAV" in the Results but "Sentinel-2/Landsat imagery and UAV orthomosaics" in the Data Availability statement. The two statements conflict. The report's phrase "1,766-image GF-2 subset" is therefore **only partly supported**; which Zenodo zips make up the 1,766 images is **UNVERIFIED**.
- Practical notes: the non-commercial licence is fine for research but must be stated; the CAS ESSD critique "high image overlap, no negative samples" (search snippet) means a tile-level split would leak, so we split by region/zip.

## 4. Prithvi-EO-2.0 landslide paper(s) and "Land 13(7):928"

**Prithvi-EO-2.0 landslide mapping: Confirmed, and the report's single citation was really two papers.**

1. Szwarcman D, Roy S, Fraccaro P, ... Moreno JB. "Prithvi-EO-2.0: A Versatile Multi-Temporal Foundation Model for Earth Observation Applications." arXiv 2412.02732 (v1 3 Dec 2024; v3 6 Mar 2026); IEEE Xplore 11296896 (listed in search). It contains a landslide task on **Landslide4Sense** (3,799 / 245 / 800 patches, six HLS bands upsampled to 224x224), baselines U-Net and U-Net++ (ResNet50) only *(read, Tables VII-VIII)*. Full-data results as printed (mIoU / F1): U-Net 70.4 / 59.7; U-Net++ 69.0 / 57.0; Prithvi-300M 71.3 / 60.7; Prithvi-600M 70.4 / 58.6. ~1% data (50 images): U-Net 59.7 / 35.0; Prithvi-300M 64.5 / 43.8; Prithvi-600M 67.0 / 49.7.
2. Li W, Wang S, Lee H, Lu C, Roy S, Ramachandran R, Hsu C-Y. "Landslide Hazard Mapping with Geospatial Foundation Models: Geographical Generalizability, Data Scarcity, and Band Adaptability." arXiv 2511.04474 (6 Nov 2025, v1). This is the paper behind the report's "outperforms U-Net, Segformer, SwinV2-B, SatMAE and TerraMind". Table 3 *(read)*, L4S test set, mIoU / F1: U-Net 70.44 / 59.66; U-Net++ 68.96 / 57.89; SegFormer 69.89 / 59.12; SwinV2-B 67.77 / 58.28; TransLandSeg 68.30 / 55.39; SatMAE-MS 67.48 / 53.72; TerraMind-base 69.77 / 58.12; Prithvi-300M 71.30 / 60.70; Prithvi-600M 70.41 / 58.56. Geographic generalisation (Table 6, mIoU) uses the DLR "Landslide Reference" set and an external GVLM-S2 set (56 samples, 11 regions). Only arXiv v1 was found; no journal version is mentioned on the abstract page. **I read about the first 100,000 of 144,645 characters**, so the end of the paper is unread.
   - **Overlap warning:** this paper already tests a *band adapter* (linear projection vs U-Net encoder head) for 14-/9-/6-/4-band inputs into Prithvi's 6-band interface. The "learned band adapter" in the plan is **not novel by itself**. Our novelty has to be the adapter inside a *compact* student, zero-shot between sensors with different resolutions, evaluated under leave-one-dataset-out.
   - Note the L4S baselines differ by ~0.3 F1 across these two papers for the same U-Net, which illustrates why we re-run baselines ourselves.

**"Land 13(7):928": Confirmed (via search results; MDPI direct fetch returned HTTP 403).**
- Chen W, Chen Z, Song D, He H, Li H, Zhu Y. "Landslide Detection Using the Unsupervised Domain-Adaptive Image Segmentation Method." *Land* 2024, 13(7), 928; 26 June 2024; doi 10.3390/land13070928. Method name LUDAS (multi-source, target-domain-adaptive, unsupervised inter-domain translation to make pseudo-target data). Sources: search summary of the MDPI notes page, MDPI PDF and the RePEc/IDEAS listing. Their benchmark numbers were not read.

## 5. Other corrections found while verifying

- **First author of the PLoS paper:** the report and the task brief say "Sharma et al." The page lists **Srivats R., Deepika Roselind Johnson (corresponding), Logeswari G., Saimirra R., Muskaan Siddiqui, Abhiram Sharma**. Sharma is the last author. Title as in the report. Published **30 April 2026** (received 31 Dec 2025, accepted 31 Mar 2026) *(read)*. Cite as Srivats et al. The volume/issue "21(4)" was not independently seen on the page.
- **PILD:** arXiv 2608.09325 (GeoPhysAdapter; Liu Z, Kwan M-P, Wu J, Li H; submitted 10 Aug 2026) says PILD is built from four public sources, 55 events, **7,890 test samples** *(read, abstract)*. Code and data: https://github.com/Liu-Zhihang/geophysadapter. The acronym expansion "Physics-informed landslide dataset" in the report was **not** on the abstract page: UNVERIFIED.
- **2604.16663** *(read, abstract)*: Kowsher M, Zhan W, Chen C; submitted 17 Apr 2026; GDCLD only; "reduces trainable parameters by up to 95%". Abstract names no specific models; whether it includes a foundation model beyond "large pretrained" ones is unread.
- TerraFM L4S numbers in the report (mIoU 70.8, IoU 43.1) and the LandsDANet, WILD-SAM, Burange, Ge et al. rows were **not** checked in this pass: UNVERIFIED.

## 6. Newer or overlapping 2026 work (search run 2026-10-10)

**Date caveat.** Today is 2026-10-10, the same day as the report, so nothing can be "after 2026-10-10". I searched for anything dated June to October 2026 or otherwise missing from the report. The search index showed **nothing dated after GeoPhysAdapter (10 Aug 2026)**, but two searches for arXiv IDs 2609/2610 returned nothing, which may reflect index coverage rather than absence. Re-run on arXiv listing pages just before submission. All entries below are from snippets/abstracts; none read in full.

| Paper | Date | Link | Overlap with plan |
|---|---|---|---|
| Vu H B. Clay-CNN Hybrids: Geospatial Foundation Models as Auxiliary Context for Landslide Detection | 12 Jun 2026 (v3 16 Jun) | https://arxiv.org/abs/2606.14081 | L4S only; Clay v1.5 + two-stage LoRA; best U-Net+Clay F1 64.5 +- 1.8 (3 seeds) vs U-Net 59.9 *(read, abstract)*. Partial overlap on LoRA; no cross-region, no distillation stated. |
| Sequential Feature Selection for Efficient Landslide Segmentation from Multi-Spectral Data | May 2026 | https://arxiv.org/abs/2605.09746 | Channel selection on L4S with a U-Net++ proxy. Overlaps band-handling idea. |
| DIGSFNet (Remote Sens. 18(16):2692) | 2026 | https://doi.org/10.3390/rs18162692 | Cross-modal teacher-student distillation + INT8 for landslide extraction *(snippet)*. **Overlaps the distillation claim**; need to read before claiming novelty. |
| Sentinel-1 SAR-based globally distributed co-seismic landslide detection (GMD 19, 167) | 2026 | https://gmd.copernicus.org/articles/19/167/2026/ | Trained on 11 events, tested on unseen Haiti 2021 and Sumatra 2022, F1 up to 82% *(snippet)*. Relevant to leave-one-event-out and SAR. |
| Domain generalization for semantic segmentation of remote sensing images via vision foundation model fine-tuning (ISPRS J.) | 2025/26 | https://www.sciencedirect.com/science/article/abs/pii/S0924271625003569 | PEFT + augmentation for DG; includes a landslide set *(snippet)*. |
| Lightweight semantic segmentation for co-seismic landslide identification using adaptive transfer learning (SLATL) | 2025/26 | https://www.sciencedirect.com/science/article/abs/pii/S0952197625037157 | Compact model, Hokkaido and Wenchuan *(snippet)*. |
| Multi-Modal Landslide Detection from Sentinel-1 SAR and Sentinel-2 Optical Imagery Using Multi-Encoder Vision Transformers and Ensemble Learning | Apr 2026 | https://arxiv.org/abs/2604.05959 | Multi-sensor fusion, not read. |
| TransLandSeg: transfer learning for landslide segmentation with a vision foundation model | 2024 | https://arxiv.org/abs/2403.10127 | Adapter-based SAM fine-tuning; already a baseline in 2511.04474. |
| LandslideAgent with Multimodal LandslideBench | Jun 2026 | https://arxiv.org/abs/2606.18661 | Agent/benchmark; probably tangential. Not read. |

**Net effect on novelty.** (a) PEFT on landslides: Kowsher 2026, Vu 2026, Li 2025 all exist. (b) Cross-domain with foundation models: GeoPhysAdapter, Li 2025. (c) Band adaptation: Li 2025. (d) Distillation to a small student for landslides: DIGSFNet and Ge 2023 exist. What is still not shown by any source I found: **one protocol combining leave-one-dataset-out over 5+ public datasets, a band-agnostic compact student, and measured CPU/edge latency plus calibration.** That claim rests on a search, not a proof; I will soften it in the paper.

---

## 7. Dataset-link checks done during Step 2 (Zenodo API, read-only)

- **LMHLD link in the report is wrong or at least misleading.** `10.5281/zenodo.11424988` is titled "GSR2SD: A Global Multi-Satellite Sensor High-Resolution Remote Sensing Dataset for Landslide Detection" (curator Guanting Liu, published 2024-06-02, 6,723 patches, five sensors, seven areas, files **restricted**, no files exposed). The arXiv paper 2502.19866 (Liu et al., 27 Feb 2025; 25,365 patches, seven areas) cites that DOI. Zenodo's version list for the same concept shows the LMHLD data as later versions: `10.5281/zenodo.11519933` (2024-06-02, LMHLD.rar 1.77 GB) and `10.5281/zenodo.17258777` (2025-10-04, LMHLD.rar 5.31 GB). Which version holds the 25,365 patches is **UNVERIFIED** until I open the archive and count. Licence on the GSR2SD record: CC BY 4.0; licences of the two LMHLD versions to be recorded on download.
- **Landslide4Sense Zenodo 10463239** (CC BY 4.0): TrainData.zip 2.31 GB, ValidData.zip 0.13 GB, TestData.zip 0.44 GB, with MD5s from Zenodo.
- **GDCLD Zenodo 13612636** (CC BY 4.0): train_data.zip 15.35 GB, val_data.zip 6.29 GB, test_data.zip 1.78 GB, "Future work.zip" 1.07 GB.
- **HR-GLDD Zenodo 7189381** (CC BY 4.0): trainX/trainY/valX/valY/testX/testY `.npy`, about 0.5 GB in total. The report says "10 events"; per-event identity is not in the file names, so an event-level split may be impossible with this release (**to check**).

- **Sen12Landslides** *(read, PMC12606323)*: Höhn P, Heidler K, Behling R, Zhu XX. *Sci Data* 12, 1772 (11 Nov 2025), doi 10.1038/s41597-025-06167-2. Data: https://huggingface.co/datasets/paulhoehn/Sen12Landslides (DOI 10.57967/hf/5883); code https://github.com/PaulH97/Sen12Landslides (article says it "will" be published: confirm live). 128x128 px at 10 m, 15-image time series per patch; S2 bands B02-B12 + SCL + mask + DEM; S1 VV/VH (not in dB). 13,628 S2 patches, ~75,000 annotated landslides (74,956 refined). Dataset size in GB not stated. Dataset licence not stated in the article (check the HF card).
- **PILD** *(read, GitHub README summary)*: PILD itself is a **metadata archive** (manifests, splits, protocol tables; Zenodo 10.5281/zenodo.19430714, 7,890 samples, 55 events). The imagery must be fetched from upstream: Sen12Landslides 4,979 + GDCLD 2,334 + DLR Landslide Reference 509 (10.5281/zenodo.17007637) + GLaD4CD v1 68 (10.5281/zenodo.14226448) = 7,890 (my sum matches). Split file: `metadata/pild_geo4_qc_v1/event_isolated_split_geo4_qc_v1.csv`. Licence of PILD/code not stated on the page. The README summary I could read did not define the `role` values: UNVERIFIED until I open the CSV.
- **Bijie**: landing page https://gpcv.whu.edu.cn/data/Bijie_pages.html confirmed by search results (770 landslide + 2,003 non-landslide images, TripleSat 0.8 m, May-Aug 2018, three-fold inspection; masks, DEM and boundary polygons). The page itself could not be fetched by my tool, so the **download mechanism and licence are UNVERIFIED**; it may need a manual download by you.
- **Technical finding:** `h5py` cannot load on this PC (Windows Application Control blocks one of its DLLs), so HDF5 is read with the pure-Python `pyfive` instead. PyTorch 2.14.1 CPU loads fine.
- **Landslide4Sense validation split, computed by me from `ValidData.zip`:** 245 tiles, each 128x128x14 float64 in an HDF5 key `img`; mask key `mask` uint8 with values {0,1}; positive-pixel ratio **1.71%** in this split (all 245 masks).

---

## Status summary

| Flag | Status |
|---|---|
| MS2LandsNet is a dataset | Not a dataset; Kaggle link = Landslide4Sense re-upload (Kaggle metadata read). Lu et al.'s own data: UNVERIFIED |
| GDCLD subset in PLoS | UNVERIFIED ("over 9,000" matches no ESSD split) |
| ESSD GDCLD numbers | Confirmed |
| CAS Zenodo 10294997 | Confirmed (record read); PLoS 1,766-image subset composition UNVERIFIED |
| CAS paper DOI | Confirmed via Zenodo + PLoS reference; Nature page blocked |
| Prithvi-EO-2.0 landslide paper | Confirmed, two papers (2412.02732 and 2511.04474) |
| Land 13(7):928 title/authors | Confirmed via search/RePEc (MDPI blocked) |
| New 2026 overlap | Listed; index shows nothing newer than 10 Aug 2026 |
| PLoS first author | Corrected: Srivats et al., not Sharma et al. |

## 8. Additional findings from Step 2 (2026-10-10)

- **LMHLD bands and sensors** *(read, PDF text of arXiv 2502.19866)*: `Each selected patch includes four spectral bands (blue, green, red, and near-infrared)`. Table 1: Rio de Janeiro RapidEye 5 m (2011.08); Hokkaido PlanetScope 3 m (2018.11); Emilia-Romagna Sentinel-2 10 m (2023.08); Gorkha Sentinel-2 10 m (2015.12); Jiuzhaigou Gaofen-2 1 m (2015.08); Wenchuan IKONOS 0.8 m (2008.05); Taiwan PlanetScope 3 m (2018.01). The report's "LMHLD = RGB" is therefore wrong. Rio's 5th channel is not explained: UNVERIFIED. Total patches in the arrays: 25,365 (matches the paper). Zenodo licence of record 17258777: CC BY 4.0.
- **CAS image count:** the dataset README says 22,049 RGB images; the Zenodo description and the delivered files give 20,865 (measured). The README sensor table lists Longxi River SAT as GF-1 and Hokkaido as 3 m, which does not match the PLoS paper's "GF-2" statement.
- **Sen12Landslides on Hugging Face** *(read via the HF API)*: licence tag CC-BY-4.0; `data_harmonized` 72.4 GB and `data_raw` 86.4 GB (158.8 GB in total). Not downloaded.
