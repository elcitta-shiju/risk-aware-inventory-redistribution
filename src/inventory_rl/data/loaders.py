"""Loaders for the extracted raw datasets. Column names are standardized in memory only."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from inventory_rl.utils.paths import RAW_DIR

MENDELEY_DIR = "mendeley_retail_transactions_stocks"
OPEN_ENDED_DATE = "9999-12-31"  # 'current record' sentinel in the Mendeley inventory table


def snake(name: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", name.strip().lower()).strip("_")


def _read(path: Path, **kw) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run scripts/extract_data.py first")
    df = pd.read_csv(path, low_memory=False, **kw)
    df.columns = [snake(c) for c in df.columns]
    return df


def load_mendeley_sales(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Transaction-level sales. `qty_sold` < 0 rows are returns (`is_return` == 1)."""
    df = _read(raw_dir / MENDELEY_DIR / "retail_sales_ml_apl.csv")
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], format="%Y-%m-%d", errors="raise")
    return df


def load_mendeley_inventory(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Stock-on-hand validity intervals [start_date, end_date] per store/product/stock status.

    `end_date` == 9999-12-31 marks the current (open) record; it is kept as NaT in `end_date`
    and flagged in `is_open`.
    """
    df = _read(raw_dir / MENDELEY_DIR / "retail_inventory_ml_apl.csv")
    df["is_open"] = df["end_date"].eq(OPEN_ENDED_DATE)
    df["start_date"] = pd.to_datetime(df["start_date"], format="%Y-%m-%d")
    df["end_date"] = pd.to_datetime(df["end_date"].where(~df["is_open"]), format="%Y-%m-%d")
    return df


def discover_raw_files(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Inventory of all extracted data files with sizes."""
    rows = [
        {"dataset": p.parent.name, "file": p.name, "format": p.suffix.lstrip("."), "size_mb": p.stat().st_size / 1e6}
        for p in sorted(raw_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in {".csv", ".xlsx", ".json", ".parquet"}
    ]
    return pd.DataFrame(rows, columns=["dataset", "file", "format", "size_mb"])
