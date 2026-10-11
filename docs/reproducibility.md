# Reproducibility

## Environment
- Windows 11, Python 3.13.15, CPU only. Exact library versions are in `requirements.txt`.
- Seeds: `seed: 42` (forecasting), RL seeds `[0, 1, 2]`, evaluation scenario seeds 10,000–10,019,
  validation scenario seeds 20,000+, simulated store locations seed 7. All are set in code or config.
- LightGBM uses multithreading. Results are deterministic for a fixed thread count; tiny differences
  across machines are possible.
- PPO on CPU is deterministic per seed on the same machine. Across machines and library versions,
  small numeric differences can change the selected checkpoint.

## Setup (PowerShell, from the `prototype` folder)
```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .                      # optional: makes `inventory_rl` importable everywhere
copy .env.example .env                # then edit INVENTORY_DATA_DIR if your archives live elsewhere
```

## Full run
```powershell
python scripts/run_pipeline.py                    # ~35 min total (RL training is ~27 min)
python -m pytest                                  # 70 tests
streamlit run dashboard/app.py
```

## Stage by stage
| Stage | Command | Main outputs |
|---|---|---|
| Extract archives (safe) | `python scripts/extract_data.py` | `data/raw/*`, `data/raw/extraction_manifest.json` |
| Data audit + EDA | `python scripts/run_data_audit.py` | `results/reports/data_audit.json`, `results/figures/eda_*.png` |
| Preprocess + features | `python scripts/run_preprocess.py` | `data/processed/{panel,features}.parquet`, `unit_economics.csv`, `results/reports/preprocessing_report.json` |
| Forecasts + intervals | `python scripts/run_forecasting.py` | `data/processed/forecasts.parquet`, `models/forecast_config.json`, `results/metrics/{forecast,interval}_metrics.json` |
| Risk + network | `python scripts/run_risk.py` | `data/processed/risk.parquet`, `results/metrics/{imbalance.csv,risk_summary.json,store_network.csv}` |
| RL training | `python scripts/run_rl.py` (`--timesteps 20000 --seeds 0` for a quick run) | `models/rl/*/{best_model.zip,validation_curve.csv,meta.json}` |
| Evaluation | `python scripts/run_evaluation.py` | `results/metrics/{evaluation_summary.csv,results.json}`, `results/reports/evaluation_report.md`, figures |

Each stage reads only earlier stages' artifacts, so a stage can be rerun alone. Generated data and
models are git-ignored and can be fully regenerated from the original archives.

## Google Colab
`notebooks/pipeline_walkthrough.ipynb` mounts Drive, sets `INVENTORY_DATA_DIR`, installs the
requirements and calls the same scripts and modules. No logic is duplicated in the notebook.
