# Project status

_Last updated: 2026-10-11 (session 1)._ Only executed and verified results are listed.

## Environment check (done)
- Working directory verified: `...\Main project\prototype` (was empty; no prior Git repo).
- **Path discrepancy:** the data folder is `project data\dataset` (singular), not `datasets`. It is used as the default
  `INVENTORY_DATA_DIR`. It contains 3 ZIP archives; `project data` also holds an abstract, a PPT PDF and the IEEE papers.
- Python 3.13.15 (single interpreter), Git 2.53. Project venv `.venv` created; all dependencies install and import
  on 3.13 (torch 2.14.1 CPU, SB3 2.9.0, gymnasium 1.4.0, LightGBM 4.7.0, streamlit 1.65.0).
- Git initialized locally on `master`. The existing global identity was left unchanged. Nothing pushed.

## Completed
| Stage | Status | Evidence |
|---|---|---|
| 1 Discovery, safe extraction, audit, EDA | ✅ | `data/raw/extraction_manifest.json` (no rejected members), `results/reports/data_audit.json`, `docs/data_audit.md` |
| 2 Preprocessing & features | ✅ | 1,200 series × 46 weeks, 0 duplicate keys; leakage test passes |
| 3 Forecasting | ✅ | rolling-origin LightGBM (Poisson, 31 leaves, selected on val); test RMSE 0.569 vs 0.588 MA(8), 0.746 naive |
| 4 Uncertainty | ✅ | Poisson interval selected on val; test coverage 96.5% (nominal 80%, discreteness), 79.6% on positive-demand weeks |
| 5 Risk | ✅ | P(stockout) reliability on test: Brier 0.035, bins well aligned where n ≥ 100 |
| 6 Distances/costs | ✅ (simulated) | no real locations exist → SIMULATED coordinates; OSRM path implemented, unexercised |
| 7 Environment | ✅ | Gymnasium `check_env` passes; conservation/feasibility enforced and tested |
| 9 Baselines | ✅ | no-transfer + 3 rule variants |
| 11 Dashboard | ✅ code | 7 tabs; reads artifacts only |
| 12 Tests | ✅ | 70 tests passing (`python -m pytest`) |
| 13 Docs, notebook | ✅ | `docs/*.md`, `notebooks/pipeline_walkthrough.ipynb` |

## In progress / interrupted
- **Stage 8 RL training: interrupted.** The full run (2 variants × 3 seeds × 200k steps) was stopped by Claude
  Code because the machine ran critically low on memory, not because of a training error.
  - Completed: `models/rl/rl_point_seed0` (best validation saving 419 vs no-transfer, Poisson scenarios).
  - Partial: `models/rl/rl_uncertainty_seed0` holds only an intermediate `best_model.zip`, so it must be retrained.
  - Not started: seeds 1 and 2 for both variants.
  - Earlier 100k-step pilot (seed 0) showed learning for the uncertainty variant: validation saving 489 → 942.
  - To resume: `python scripts/run_rl.py` (≈27 min, all runs) or `python scripts/run_rl.py --seeds 1 2`,
    with other memory-heavy apps closed.
- **Stage 10 final evaluation: not yet run** (`python scripts/run_evaluation.py` after RL training).
  `docs/experiments.md` will be written from its generated outputs. No RL results are claimed yet.
- Dashboard headless test (`tests/test_dashboard.py::test_dashboard_runs_headless_on_artifacts`) is skipped
  until the evaluation artifacts exist.

## Pending / not done
- Dataset licences and citations must be confirmed on the Kaggle/Mendeley web pages (not in the archives).
- Literature review and positioning (no novelty claims made).
- The Colab notebook has not been executed on Colab itself (it uses the same scripts that run locally).

## Blocked
- Real road distances: blocked by the absence of store locations in the data.
