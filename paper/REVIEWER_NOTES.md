# Self-review: what a Q1/Q2 reviewer is likely to say, and what we did about it

Written for the authors, not for submission. Nothing here is a prediction of acceptance: whether a journal is Q1 or Q2 and whether a paper is
accepted depends on the journal, the editor and the reviewers, and cannot be guaranteed by drafting quality.

## What the paper is (and is not)
A *benchmark-and-protocol* paper with *honest negative results*, not a new-method paper. Its strengths are the event-isolated protocol, the controls,
multi-seed reporting, calibration and CPU-latency evidence, and the data-quality findings. Its weakness is that no new method outperforms a baseline.
Venues that publish evaluation/benchmark studies suit it better than venues that expect a new architecture; choose the framing accordingly.

## Likely criticisms (ordered by how much they would hurt)
| # | Criticism | What the manuscript already does | What is still open |
|---|---|---|---|
| 1 | "The cross-dataset training pool is only two datasets (LMHLD, CAS) under B-strict, so this is not a general leave-one-dataset-out." | States it prominently (Sec. 3.3, Table composition); adds a composition-matched control that holds sources and size fixed. | A design with more sources would need tile-level event ids for L4S/HR-GLDD/GDCLD, which do not exist. Could add a variant that keeps those datasets and removes only the documented overlapping *events* if event ids can be recovered. |
| 2 | "Single architecture family, 10 epochs, RGB only, no tuning." | Declared in Limitations. | Longer training (e.g. 30 epochs) and DeepLabV3+ in the cross-dataset protocol; a second transformer (e.g. SegFormer-B1/B2) to test whether the ranking is architecture- or size-driven. |
| 3 | "n = 3 seeds; no significance; five folds." | Reports mean +- std, per-seed values, sign counts, and says no significance can be claimed. | More seeds for the controls; bootstrap over tiles/events is possible but tiles are not independent. |
| 4 | "Event membership is partly inferred from place names; residual leakage cannot be excluded." | Marks inferred assignments; conservative merging; leakage report. | Check each inferred assignment against the original inventories; ask dataset authors. |
| 5 | "No foundation models (Prithvi/TerraMind), so the LoRA/teacher story is weak." | Declared; negative result for one configuration only. | A foundation-model teacher needs more GPU memory/time than the 30 h/week budget allowed. |
| 6 | "Protocol C (leave-one-event-out) is defined but not run." | Declared as not run. | About 3-4 GPU hours (estimate from measured fold times) for SegFormer-B0, one seed; on hold at the user's request. |
| 7 | "Quoted literature numbers came through a retrieval tool, not the original tables." | Flags them (E2), keeps them in one file, never mixes them with our measurements. | Re-check every value and every reference against the originals before submission (reference statuses are in `references.json`). |
| 8 | "HR-GLDD channel order is inferred and the decision followed a test failure." | Discloses it, tests only two orders, explains the weak statistics. | Contact the dataset authors, or check with the raw GeoTIFF/inventory metadata. |
| 9 | "GDCLD and CAS were downscaled; detail lost." | Quantifies the GDCLD ceiling (a perfect 256 px predictor scores far above our models). | Full-resolution GDCLD/CAS runs would cost roughly 4x and 2x the pixels of the current runs. |
| 10 | "No edge device." | CPU latency only, stated. | A Jetson or phone measurement. |

## Things that were found and corrected during drafting (so they do not recur)
* The "85.06% mIoU" GDCLD figure is the dataset paper's *validation* score; its *test* score is 72.84%. Our test mIoU is close to the latter.
* GDCLD's test split is event-isolated from its training data according to its own paper; its validation split is not.
* B-strict trains only on LMHLD and CAS; the first size-matched overlap control did not hold the source datasets fixed.
* The HR-GLDD channel-statistics argument is weak (it matches one of two documented PlanetScope scenes).
* Single-seed L4S results were very unstable (SegFormer-B0 seed 0 = 24.1, three-seed mean = 13.4).

## Before any submission
1. Author list, affiliations, funding, competing interests, CRediT, AI-assistance disclosure (placeholders are marked in the manuscript).
2. Convert `manuscript.tex` to the journal's template; compile (a PDF has not been built on this machine because the LaTeX class files are not installed locally; see the README).
3. Verify all references and external numbers (see `paper/references.json` status fields and `paper/literature_values.json`).
4. Decide the licence of the repository and check the redistribution of example image tiles (CAS is CC BY-NC 4.0).
