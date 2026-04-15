# TriFuseSurv

TriFuseSurv is a compact OPSCC survival package centered on the recommended contour-aware workflow:

- CT-only shared encoder
- internal PT/LN localization heads
- ROI tokens generated from predicted soft masks
- joint OS/DSS/DFS survival plus localization training in one graph

`preprocess -> prepare survival table -> make splits -> contour-aware survival CV`

## Keepers

The packaged shell surface is intentionally small now:

- `scripts/install_env.sh`
- `scripts/build_zip_package.sh`
- `scripts/run_contour_aware_survival_serious.sh`
- `scripts/run_contour_aware_cindex_search.sh`
- `scripts/run_contour_aware_cindex_search_h100.sh`
- `scripts/preprocessing/export_swinunetr.sh`
- `scripts/preprocessing/prepare_opscc_tabular.sh`
- `scripts/preprocessing/make_cv_splits.sh`
- `scripts/survival/train_contour_aware_survival.sh`
- `scripts/survival/train_contour_aware_survival_lora.sh`
- `scripts/survival/evaluate_oof_cindex.sh`

The main package code remains under `src/trifusesurv/`.

## Install

```bash
./scripts/install_env.sh
source .venv/bin/activate
```

If you need a non-default PyTorch wheel source, set `TORCH_INDEX_URL` before running `scripts/install_env.sh`.

## Workspace Layout

The package expects this relative layout from the workspace root that contains `TriFuseSurv_package`:

- `OPSCC`
- `opscc_survival_time_event.csv`
- `clinical_covariate.csv`
- `cohort_radiomics_patient_wide.csv`

## Recommended Runs

Full serious workflow:

```bash
./scripts/run_contour_aware_survival_serious.sh
```

Recommended 2x H100 80 GB setting search:

```bash
./scripts/run_contour_aware_cindex_search_h100.sh
```

Resume the `meaningful_base` H100 search trial in-place to 100 total epochs:

```bash
./scripts/run_contour_aware_meaningful_base_100ep.sh
```

Fresh 100-epoch contour-aware run sized for roughly 70 GB VRAM on H100 80 GB:

```bash
./scripts/run_contour_aware_base70gb_100ep.sh
```

Meaningful 4-fold contour-aware search capped around 75 GB VRAM with 30 epochs per fold run:

```bash
./scripts/run_contour_aware_cindex_search_75gb_30ep.sh
```

That search uses:

- contour-aware image encoder
- shared fusion trunk with three endpoint-specific survival heads: OS, DSS, DFS
- 4-fold CV
- OOF c-index ranking
- 8 focused settings around the current strongest `h1095` region
- 30 epochs per fold-training run, fixed in the script
- compares auxiliary multitask survival weighting plus a small set of higher-signal structural changes: finer time bins, no-multiscale, longer teacher forcing, weaker localization supervision, and a no-multiscale + locweak combination
- targets roughly 75 GB VRAM on H100 80GB
- automatic multi-GPU scheduling across detected GPUs

## Rebuild The Zip

```bash
./scripts/build_zip_package.sh
```
