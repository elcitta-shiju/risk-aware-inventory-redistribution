# Risk-Aware Inventory Redistribution Using Uncertainty-Aware Demand Forecasting and Reinforcement Learning

Final-year B.Tech (AI & Data Science) project. The system forecasts weekly store-level demand for
retail SKUs, estimates the forecast uncertainty, scores stockout and excess-inventory risk, and
recommends inter-store stock transfers. It compares rule-based policies and PPO reinforcement-learning
agents, with and without uncertainty information, in a simulator built from real sales and stock data.

> **Scope note.** Transfer outcomes are *simulated* from historical data. Store locations are
> *simulated* because the data has none, and cost parameters are documented assumptions. No claim of
> novelty or of real-world superiority is made. See [assumptions and limitations](docs/assumptions_and_limitations.md).

## Data
Primary: **Mendeley "Retail Transactions and Stocks Data"**, with 125,751 sales transactions and
284,755 stock-on-hand validity records over 40 stores and 2,326 SKUs (Jun 2025 – Apr 2026). The two
Kaggle files were audited but not used: one is internally inconsistent, the other appears randomly
generated, and their keys are incompatible ([data audit](docs/data_audit.md)). Original archives are read-only; they are
extracted into `data/raw/`.

## Pipeline
| Stage | Module | Script |
|---|---|---|
| Safe extraction, discovery, audit | `inventory_rl.data.{extract,loaders,audit}` | `extract_data.py`, `run_data_audit.py` |
| Weekly store × SKU panel, stock reconstruction, features | `inventory_rl.data.preprocess`, `inventory_rl.features.build` | `run_preprocess.py` |
| Rolling-origin forecasting (baselines + global LightGBM) | `inventory_rl.forecasting.models` | `run_forecasting.py` |
| Prediction intervals (Poisson, QR, CQR, conformal) | `inventory_rl.uncertainty.intervals` | `run_forecasting.py` |
| Inventory risk + calibration | `inventory_rl.risk.assessment` | `run_risk.py` |
| Distances & transfer cost (simulated / OSRM-ready) | `inventory_rl.routing.network` | `run_risk.py` |
| Simulator + Gymnasium env | `inventory_rl.environment.*` | — |
| PPO with a shared per-store policy | `inventory_rl.agents.{ppo,policy}` | `run_rl.py` |
| Rule-based baselines | `inventory_rl.baselines.rules` | `run_evaluation.py` |
| Evaluation, report | `inventory_rl.evaluation.*` | `run_evaluation.py` |
| Cost-assumption sensitivity | `inventory_rl.evaluation.sensitivity` | `run_sensitivity.py` |
| Dashboard | `dashboard/app.py` | `streamlit run dashboard/app.py` |

## Quick start (Windows PowerShell)
```powershell
cd "C:\Users\ElcittaShiju\OneDrive\Desktop\Main project\prototype"
.\.venv\Scripts\Activate.ps1
python scripts/run_pipeline.py            # full pipeline (~1 h; add --skip-rl to reuse trained agents)
python -m pytest                          # run the tests
streamlit run dashboard/app.py            # open http://localhost:8501
```
First-time setup (creating `.venv` with **Python 3.12**, installing packages) is described in
[docs/reproducibility.md](docs/reproducibility.md). Python 3.13 is not supported: on the development machine
it crashed inside `asyncio` when Streamlit was loaded.
The raw-data folder defaults to `..\project data\dataset`. Override it with `INVENTORY_DATA_DIR`
(see `.env.example`).

## Results
Generated, never hand-written: [`results/reports/evaluation_report.md`](results/reports/evaluation_report.md),
`results/metrics/results.json` and figures in `results/figures/`. A summary and its interpretation are in
[docs/experiments.md](docs/experiments.md).

## Repository layout
```
config/config.yaml        all parameters (assumptions marked)
src/inventory_rl/         package: data, features, forecasting, uncertainty, risk, routing,
                          environment, agents, baselines, evaluation, visualization, utils
scripts/                  one script per stage + run_pipeline.py
dashboard/                Streamlit app (reads saved artifacts only)
notebooks/                walkthrough calling the package (Colab setup cells included but untested on Colab)
tests/                    pytest suite (synthetic fixtures)
docs/                     data dictionary, audit, methodology, assumptions, experiments, reproducibility
data/, models/, results/  generated (git-ignored except reports/metrics)
```

## Documentation
[Data dictionary](docs/data_dictionary.md) · [Data audit](docs/data_audit.md) · [Methodology](docs/methodology.md) ·
[Assumptions & limitations](docs/assumptions_and_limitations.md) · [Experiments](docs/experiments.md) ·
[Reproducibility](docs/reproducibility.md) · [Project status](PROJECT_STATUS.md)

## Connecting to GitHub later
The repository is local only, and nothing has been pushed. To publish to your account:
```powershell
git remote add origin https://github.com/<your-account>/<repo-name>.git
git push -u origin master
```
Datasets, `.venv`, `.env`, and generated data and models are excluded by `.gitignore`. Check the dataset
licences before making the repository public.
