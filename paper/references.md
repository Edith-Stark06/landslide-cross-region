# References and verification status

Status key: **[R]** read the relevant parts of the source itself; **[A]** read the abstract / landing page only; **[S]** seen only as a search-result
snippet; **[V]** bibliographic details written from memory, NOT checked against any source in this project: verify before submission.
Nothing here has been checked for correct page numbers beyond what is stated.

| # | Reference | Status | Notes |
|---|---|---|---|
| 1 | Srivats R, Johnson DR, Logeswari G, Saimirra R, Siddiqui M, Sharma A. A multi-modal deep learning framework with GAN-based fusion for enhanced landslide detection. PLoS One 2026 (published 2026-04-30). doi:10.1371/journal.pone.0347324 | R (dataset section, references) | The report's "Sharma et al." is wrong: Sharma is the last author. Volume/issue 21(4) not independently seen. |
| 2 | Ghorbanzadeh O et al. Landslide4Sense: Reference Benchmark Data and Deep Learning Models for Landslide Detection. arXiv:2206.00515 (2022); data Zenodo 10.5281/zenodo.10463239 | A (via dataset README / Zenodo record) | Paper not opened. |
| 3 | Fang C, Fan X, Wang X, Nava L, Zhong H, Dong X, Qi J, Catani F. A globally distributed dataset of coseismic landslide mapping via multi-source high-resolution remote sensing images. Earth Syst Sci Data 16, 4817-4842 (2024). doi:10.5194/essd-16-4817-2024 | R (dataset section) | The 85.06% mIoU figure in our planning notes was not found/read in this paper. |
| 4 | Xu Y, Ouyang C, Xu Q, Wang D, Zhao B, Luo Y. CAS landslide dataset: a large-scale and multisensor dataset for deep learning-based landslide detection. Sci Data 11, 12 (2024). doi:10.1038/s41597-023-02847-z | A (Zenodo record, PLoS reference list; Nature page blocked) | |
| 5 | HR-GLDD: A globally distributed high resolution landslide dataset. Earth Syst Sci Data 15, 3283 (2023). Zenodo 10.5281/zenodo.7189381 | R (text), authors **not captured** | Author list UNVERIFIED. |
| 6 | Liu G, Wang Y, Chen X, Du B, Li P, Wu Y, Fang Z. LMHLD: A large-scale multi-source high-resolution landslide dataset for landslide detection based on deep learning. arXiv:2502.19866 (2025); data Zenodo 10.5281/zenodo.17258777 | R (paper text via PDF) | The DOI cited in the arXiv abstract resolves to a differently titled record (GSR2SD); LMHLD is a later version. |
| 7 | Höhn P, Heidler K, Behling R, Zhu XX. A spatio-temporal dataset for satellite-based landslide detection (Sen12Landslides). Sci Data 12, 1772 (2025). doi:10.1038/s41597-025-06167-2 | R (PMC article) | Dataset not used (159 GB). |
| 8 | Liu Z, Kwan M-P, Wu J, Li H. GeoPhysAdapter: scale-matched geophysical adaptation for cross-domain landslide mapping with vision foundation models. arXiv:2608.09325 (2026) | A | |
| 9 | Kowsher M, Zhan W, Chen C. A benchmark study of segmentation models and adaptation strategies for landslide detection from satellite imagery. arXiv:2604.16663 (2026) | A | |
| 10 | Li W, Wang S, Lee H, Lu C, Roy S, Ramachandran R, Hsu C-Y. Landslide hazard mapping with geospatial foundation models: geographical generalizability, data scarcity, and band adaptability. arXiv:2511.04474 (2025) | R (Tables 3 and 6, band-adaptation description; last ~30% of the text unread) | Overlaps our band-adapter idea (linear and U-Net adapters into Prithvi). |
| 11 | Szwarcman D et al. Prithvi-EO-2.0: a versatile multi-temporal foundation model for earth observation applications. arXiv:2412.02732 (v3 2026) | R (landslide task, Tables VII-VIII) | |
| 12 | Vu HB. Clay-CNN hybrids: leveraging geospatial foundation models as auxiliary context for landslide detection. arXiv:2606.14081 (2026) | A | |
| 13 | Chen W, Chen Z, Song D, He H, Li H, Zhu Y. Landslide detection using the unsupervised domain-adaptive image segmentation method. Land 13(7), 928 (2024). doi:10.3390/land13070928 | S | |
| 14 | DIGSFNet (cross-modal teacher-student distillation + INT8 for landslide extraction). Remote Sens. 18(16), 2692 (2026). doi:10.3390/rs18162692 | S | Authors not captured; read before claiming novelty. |
| 15 | Lu W, Hu Y, Shao W, Wang H, Zhang Z, Wang M. A multiscale feature fusion enhanced CNN with the multiscale channel attention mechanism for efficient landslide detection (MS2LandsNet) using medium-resolution remote sensing data. Int J Digital Earth 17(1) (2024). doi:10.1080/17538947.2023.2300731 | A (PLoS reference list) | |
| 16 | Jiang, Ma, Mei. Review: deep learning for potential landslide identification. Nat Hazards Earth Syst Sci 26, 487 (2026) | S (from the planning report; never opened) | Not cited in the draft yet. |
| 17 | Xie E et al. SegFormer: simple and efficient design for semantic segmentation with transformers. NeurIPS 2021 | **V** | Verify. |
| 18 | Ronneberger O, Fischer P, Brox T. U-Net: convolutional networks for biomedical image segmentation. MICCAI 2015 | **V** | Verify. |
| 19 | Chen L-C, Zhu Y, Papandreou G, Schroff F, Adam H. Encoder-decoder with atrous separable convolution for semantic image segmentation (DeepLabV3+). ECCV 2018 | **V** | Verify. |
| 20 | CAS Landslide Dataset README (Zenodo 10.5281/zenodo.10294997), including the per-region sensor and licence table | R | |
| 21 | Hu EJ et al. LoRA: low-rank adaptation of large language models. ICLR 2022 | **V** | Verify. |
| 22 | Hinton G, Vinyals O, Dean J. Distilling the knowledge in a neural network. arXiv:1503.02531 (2015) | **V** | Verify. |
| 23 | Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks. ICML 2017 | **V** | Not yet cited in the text; for the ECE definition. |
| 24 | Iakubovskii P. Segmentation Models PyTorch (software) | **V** | Verify version used (installed fresh on each Kaggle run; record the version in result files: TODO). |
| 25 | TerraFM; TerraMind; SatMAE; SwinV2; AdaLoRA; LandsDANet; WILD-SAM; Burange et al. 2025; Ge et al. 2023 | not cited | Listed in the planning report; not opened. |
