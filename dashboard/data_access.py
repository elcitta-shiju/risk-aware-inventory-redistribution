"""Artifact loading helpers for the dashboard. Every loader returns None when a stage has not run."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from inventory_rl.utils.paths import FIGURES_DIR, METRICS_DIR, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR  # noqa: E402

ARTIFACTS = {
    "Data audit (Stage 1)": REPORTS_DIR / "data_audit.json",
    "Weekly panel (Stage 2)": PROCESSED_DIR / "panel.parquet",
    "Forecasts + intervals (Stages 3-4)": PROCESSED_DIR / "forecasts.parquet",
    "Risk table (Stage 5)": PROCESSED_DIR / "risk.parquet",
    "Store network (Stage 6)": METRICS_DIR / "store_network.csv",
    "RL checkpoints (Stage 8)": MODELS_DIR / "rl",
    "Evaluation results (Stage 10)": METRICS_DIR / "results.json",
}


def artifact_status() -> pd.DataFrame:
    return pd.DataFrame([{"stage": k, "available": p.exists(), "path": str(p.relative_to(ROOT))}
                         for k, p in ARTIFACTS.items()])


def load_json(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def load_table(path: Path) -> pd.DataFrame | None:
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def figure(name: str) -> Path | None:
    p = FIGURES_DIR / name
    return p if p.exists() else None


def rl_checkpoints() -> list[Path]:
    d = MODELS_DIR / "rl"
    return sorted(d.glob("*/best_model.zip")) if d.exists() else []


def product_labels(groups: list[str]) -> dict[str, str]:
    """Readable product names ('<subcategory> (<code>)') from the extracted sales file, if available."""
    raw = PROCESSED_DIR.parent / "raw" / "mendeley_retail_transactions_stocks" / "retail_sales_ml_apl.csv"
    names: dict[str, str] = {}
    if raw.exists():
        try:
            df = pd.read_csv(raw, usecols=["Product No", "Product Subcategory"]).drop_duplicates("Product No")
            names = dict(zip(df["Product No"], df["Product Subcategory"]))
        except (ValueError, OSError):
            names = {}
    return {g: f"{names[g]} ({g})" if g in names else g for g in groups}


def best_validation_run(variant: str = "rl_uncertainty") -> Path | None:
    """Checkpoint of the training run with the best VALIDATION score (test results are not used)."""
    summary = load_json(MODELS_DIR / "rl" / "training_summary.json")
    if not summary:
        return None
    runs = [r for r in summary["runs"] if r.get("variant") == variant]
    for run in sorted(runs, key=lambda r: -r["best_val_saving_poisson"]):
        path = MODELS_DIR / "rl" / f"{variant}_seed{run['seed']}" / "best_model.zip"
        if path.exists():
            return path
    return None
