# Project status

_Last updated: 2026-10-11 (session 1, after moving to Python 3.12)._ Only executed and verified results are listed.

## Environment
- Working directory: `...\Main project\prototype`. Data folder: `project data\dataset`. (The brief said
  `datasets`; the actual folder name is `dataset`.)
- **Python 3.12.10**, project venv `.venv`. Versions are in `requirements.txt`.
  - Python 3.12 was installed for the user account via winget; Python 3.13 remains installed and is still the system default.
  - The previous 3.13 venv is kept as `.venv-py313` (git-ignored); it can be deleted.
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
| 8 RL training | ✅ | all 6 runs (2 variants × seeds 0,1,2, 200k steps); `models/rl/training_summary.json` |
| 9 Baselines | ✅ | no-transfer + 3 rule variants |
| 10 Test evaluation | ✅ | `results/metrics/results.json`, `results/reports/evaluation_report.md` |
| 11 Dashboard | ✅ | real `streamlit run` server starts and passes its health check; the headless test runs all 7 tabs on real artifacts |
| 12 Tests | ✅ | **71 passed, 0 failed** (`python -m pytest`, Python 3.12.10) |
| 13 Docs | ✅ | README, `docs/*.md` incl. `experiments.md` |

## Python 3.12 migration (verified)
- Under Python 3.13.15 the dashboard could not start: a native access violation inside `asyncio` once Streamlit
  was loaded. The same pinned packages work under 3.12.10.
- All stages except RL training were rerun under 3.12. `panel`, `features`, `forecasts`, `risk` and the evaluation
  summary are **identical** to the 3.13 outputs.
- The RL agents were trained under 3.13. They load and reproduce the same evaluation under 3.12, and a short
  PPO training smoke test passes under 3.12. They were not retrained.

## Main results (test, modeled costs, simulated distances; see docs/experiments.md)
- The upper-bound rule saves 2,276 ± 641 vs no redistribution (Poisson scenarios). The point-forecast rule loses money (−187).
- RL with uncertainty saves 2,028 ± 438, versus 1,845 ± 388 for RL with point forecasts. The paired difference is
  +184, 95% CI [128, 239], but it is seed-dependent.
- RL did not beat the best rule under Poisson demand. No method is a clear overall winner.

## Blocked
- Real road distances: the data has no store locations.

## Pending (outside the code)
- Confirm dataset licences and citations on the Kaggle and Mendeley pages.
- Literature review (no novelty claims are made).
- The notebook's Colab setup cells are untested on Colab (not planned; documented as untested).
- Optional: cost-sensitivity analysis and more RL seeds (see docs/experiments.md, "Not done").
