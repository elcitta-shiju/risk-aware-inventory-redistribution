# Project status

_Last updated: 2026-10-11 (session 1, after RL training and evaluation)._ Only executed and verified results are listed.

## Environment
- Working directory: `...\Main project\prototype`. Data folder: `project data\dataset`. (The brief said
  `datasets`; the actual folder name is `dataset`.)
- Python 3.13.15, project venv `.venv`. Versions are in `requirements.txt`.
- Git: local repository on `master`. Global identity unchanged. **Nothing pushed.**

## Completed
| Stage | Status | Evidence |
|---|---|---|
| 1 Discovery, safe extraction, audit, EDA | ✅ | `data/raw/extraction_manifest.json`, `results/reports/data_audit.json`, `docs/data_audit.md` |
| 2 Preprocessing & features | ✅ | 1,200 series × 46 weeks; leakage test passes |
| 3 Forecasting | ✅ | test RMSE 0.569 (LightGBM) vs 0.588 MA(8), 0.746 naive |
| 4 Uncertainty | ✅ | Poisson interval selected on val; 79.6% test coverage on positive-demand weeks (nominal 80%) |
| 5 Risk | ✅ | P(stockout) Brier 0.035 on test |
| 6 Distances/costs | ✅ (simulated) | no real store locations in the data |
| 7 Environment | ✅ | Gymnasium `check_env` passes; conservation/feasibility tested |
| 8 RL training | ✅ | **all 6 runs complete** (2 variants × seeds 0,1,2, 200k steps each), trained one run at a time after the earlier memory-related stop; `models/rl/training_summary.json` |
| 9 Baselines | ✅ | no-transfer + 3 rule variants |
| 10 Test evaluation | ✅ | `results/metrics/results.json`, `results/reports/evaluation_report.md`; no missing checkpoints |
| 13 Docs | ✅ | `docs/experiments.md` written from the generated results |

## Main results (test, modeled costs, simulated distances; see docs/experiments.md)
- The upper-bound rule saves 2,276 ± 641 vs no redistribution (Poisson scenarios). The point-forecast rule loses money (−187).
- RL with uncertainty saves 2,028 ± 438, versus 1,845 ± 388 for RL with point forecasts. The paired difference
  is +184, 95% CI [128, 239], but it is seed-dependent.
- RL did not beat the best rule under Poisson demand (−248, CI [−442, −53]). No method is a clear overall winner.

## Blocked
- **Dashboard (Stage 11) cannot run on this machine.** `streamlit run dashboard/app.py` crashes at startup with a
  native access violation (exit 139 / 3221225477). Investigation:
  - The crash is inside CPython's built-in `_asyncio.current_task`, and appears only once a large set of modules
    is loaded (e.g. `import altair` or `import streamlit`).
  - It does not reproduce with any single compiled package. Base Python with only the standard library works,
    and pandas 2.3.3 vs 3.0.6 made no difference.
  - Likely an interpreter-level fault in this Python 3.13.15 setup.
  - The dashboard code itself never gets to run, so it is untested at runtime.
  - Suggested fix (needs your approval, as it installs a new interpreter): create the venv with Python 3.12, or
    reinstall/repair Python 3.13, then rerun `python -m pytest`.
- Real road distances: blocked by the absence of store locations in the data.

## Tests
`python -m pytest` → **70 passed, 1 failed**. The failure is `test_dashboard_runs_headless_on_artifacts`,
caused by the Streamlit crash above.

## Pending (outside the code)
- Confirm dataset licences and citations on the Kaggle and Mendeley pages.
- Literature review (no novelty claims are made).
- Run the notebook on Colab itself.
- Optional: cost-sensitivity analysis and more RL seeds (see docs/experiments.md, "Not done").
