# Project status

_Last updated: 2026-10-11 (session 1, after 5-seed RL and cost sensitivity)._ Only executed and verified results are listed.

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
| 8 RL training | ✅ | 10 runs (2 variants × seeds 0–4, 200k steps); `models/rl/training_summary.json` |
| 9 Baselines | ✅ | no-transfer + 3 rule variants |
| 10 Test evaluation | ✅ | `results/metrics/results.json`, `results/reports/evaluation_report.md` (RL mean ± std across 5 seeds, Welch test) |
| 10b Cost sensitivity | ✅ | 15 one-at-a-time cost settings; `results/reports/cost_sensitivity_report.md`, `results/figures/cost_sensitivity.png` |
| 11 Dashboard | ✅ | redesigned for store managers (5 tabs: Decision Support, Store Overview, Results, About, Technical Details; light theme in `.streamlit/config.toml`). Real server checked in a browser: all tabs load without errors; the headless test exercises every tab and the main selections |
| 12 Tests | ✅ | see the last test run recorded in the latest commit message |
| 13 Docs | ✅ | README, `docs/*.md` incl. `experiments.md` |

## Main results (test, modeled costs, simulated distances; see docs/experiments.md)
- The upper-bound rule saves 2,276 ± 641 vs no redistribution (Poisson scenarios). The point-forecast rule loses money (−187).
- RL across 5 training seeds: uncertainty 2,021 ± 182, point 1,885 ± 132 (mean ± std of per-seed means).
  The paired scenario-level difference is +136, 95% CI [88, 184], but the seed-level Welch test is not
  significant (p = 0.22).
- RL did not beat the best rule at base costs. Under cost sensitivity, the ranking changes: RL has the
  highest mean saving when transfers are expensive or lost sales are cheap (because it moves fewer
  units; it does not observe costs and was not retrained), the rules lead when transfers are cheap or
  lost sales are expensive, and at a fixed cost of 32 per shipment no redistribution is cheapest.

## Blocked
- Real road distances: the data has no store locations.

## Pending (outside the code)
- Confirm dataset licences and citations on the Kaggle and Mendeley pages.
- Literature review (no novelty claims are made).
- The notebook's Colab setup cells are untested on Colab (not planned; documented as untested).
- Optional: retrain RL per cost setting; a rule using the expected transfer benefit; same-seed retraining check across Python versions.
