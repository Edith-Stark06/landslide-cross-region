# Cross-region landslide segmentation: event-isolated evaluation of compact models

Code, splits and results for the paper draft in `paper/draft.md`. Everything reported there is generated from the files in this repository;
nothing has been submitted or published. Planning notes: `Landslide_Cross_Region_Report.md`; what was verified against sources:
`verification_log.md`; data peculiarities: `docs/data_notes.md`; GPU-hour ledger: `docs/gpu_budget.md`.

## What you need

| Step | Needs | Where it ran |
|---|---|---|
| Download + convert data | ~45 GB disk, internet | local Windows PC (data on `F:\LandslideData`, edit the paths in `src/landslide/prepare.py` and `data.py` / env `LANDSLIDE_PROCESSED`) |
| Build splits, leakage report, unit tests, tables, figures, CPU latency | Python 3.12, `pip install numpy pandas pillow pyyaml pytest matplotlib tabulate tifffile pyfive torch torchvision segmentation-models-pytorch` | local CPU |
| Training (all `results/runs/*`) | one GPU (Tesla T4 on Kaggle) | Kaggle private kernels, see below |

`h5py` was blocked by Windows Application Control on the authors' machine, so HDF5 is read with `pyfive`. Python >= 3.12 is assumed; the
`segmentation-models-pytorch` version used on Kaggle was whatever was current at run time and was **not recorded** (local verification used 0.5.0).

## One command per artefact (run from the repository root; `$PY` = the Python of your environment)

| Artefact | Command | Output |
|---|---|---|
| Data summary table (Table 1) | `$PY -I scripts/data_summary.py` | `results/data_summary.md` |
| Splits (A, B, B-strict, C) + leakage report | `$PY -I -c "import sys; sys.path.insert(0,'src'); from landslide import splits; splits.main()"` | `splits/*.json`, `splits/leakage_report.csv` |
| Unit tests (shapes, labels, no split overlap, metrics, adapter, LoRA) | `$PY -m pytest tests -q` | |
| All result tables (baselines, cross-dataset, overlap, teacher/student) | `$PY -I scripts/make_paper_tables.py` | `results/tables/paper_tables.md` |
| Multi-seed means ± std and paired differences | `$PY -I scripts/seed_summary.py` | `results/tables/seed_summary.md` |
| Event-overlap control table | `$PY -I scripts/compare_overlap.py` | `results/tables/overlap_control.txt` |
| Per-run table | `$PY -I scripts/aggregate.py` | `results/tables/runs.csv`, `summary.csv` |
| CPU latency (Table: efficiency) | `$PY -I scripts/benchmark_latency.py` | `results/tables/latency_cpu.md` |
| Figure 1 (Pareto), 3 (reliability), 4 (failure cases) | `$PY -I scripts/make_figures.py` | `figures/fig_pareto.png`, `fig_reliability.png`, `fig_failures.png` |
| Figure 2 (data overview) | `$PY -I scripts/make_fig_data.py` | `figures/fig_data_overview.png` |
| 4x downscaling ceiling on GDCLD test crops | `$PY -I scripts/downscale_ceiling.py` (needs the raw GDCLD archive) | printed |

The tables read `results/runs/*/result.json`; each file holds the config, seed, git revision, library versions, per-epoch history and test
metrics (with per-event breakdowns where events are known). Superseded runs made with a wrong HR-GLDD channel order are in
`results/runs_superseded/` and are excluded.

## Re-running the experiments

1. `scripts/zenodo_get.ps1` downloads and MD5-checks the Zenodo records (L4S 10463239, HR-GLDD 7189381, GDCLD 13612636, LMHLD 17258777,
   CAS 10294997); `src/landslide/prepare.py` (`prepare_l4s`, `prepare_hrgldd`, `prepare_gdcld`, `prepare_lmhld`, `prepare_cas`) converts them.
2. Upload each processed folder as a private Kaggle dataset named `landslide-xr-<name>` (`scripts/upload_*.ps1`; needs a Kaggle token).
3. A job list is a JSON file of `{config, seeds, timeout_s}`. Generate configs and jobs with the `scripts/make_configs*.py` /
   `make_seed_jobs.py` scripts, package the code with `scripts/pack_code.ps1`, create a kernel with `scripts/make_kernel.ps1`, push with
   `kaggle kernels push`. `scripts/kaggle_entry.py` runs `python -m landslide.run --config <yaml> --seed <k>` for every job.
4. Pull logs and `result.json` with `kaggle kernels output`, copy into `results/runs/`.

Experiments and their jobs: in-dataset baselines (`configs/baseline_*.yaml`), cross-dataset B-strict and B (`make_configs.py`), corrected
HR-GLDD order and size-matched overlap control (`make_configs_controls.py`), band adapter (`make_configs_adapter.py`), LoRA teacher +
distillation (`make_configs_kd.py`), extra seeds (`make_seed_jobs.py`).

## Known limitations of the repository
* Training scripts contain absolute Windows paths for the data on the authors' drive; change them or set `LANDSLIDE_PROCESSED`.
* Bijie, Sen12Landslides and PILD were not used (see `verification_log.md`). Protocol C splits are built and checked but were not run.
* Library versions of the Kaggle runs (other than torch) were not recorded.
* Raw data are not redistributed; check each dataset's licence (CAS is CC BY-NC 4.0, the others CC BY 4.0 per their Zenodo records).
