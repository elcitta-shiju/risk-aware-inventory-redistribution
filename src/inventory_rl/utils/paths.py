"""Project paths resolved from the repository root and environment variables."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Default matches the author's machine; override with INVENTORY_DATA_DIR (or a .env file).
_DEFAULT_DATA_DIR = PROJECT_ROOT.parent / "project data" / "dataset"


def _load_dotenv() -> None:
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_file, override=False)
    except ImportError:  # python-dotenv is optional
        pass


_load_dotenv()


def source_data_dir() -> Path:
    """Directory holding the original (read-only) dataset archives."""
    return Path(os.environ.get("INVENTORY_DATA_DIR", _DEFAULT_DATA_DIR))


DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
# Weekly sheets added from the dashboard, and the forecasts/risk/transfers rebuilt from them
# (overridable, e.g. so tests never touch a user's uploads).
UPLOADS_DIR = Path(os.environ.get("INVENTORY_UPLOADS_DIR", DATA_DIR / "uploads"))
LIVE_DIR = Path(os.environ.get("INVENTORY_LIVE_DIR", DATA_DIR / "live"))
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
REPORTS_DIR = RESULTS_DIR / "reports"
CONFIG_DIR = PROJECT_ROOT / "config"


def ensure_dirs() -> None:
    for d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
