"""Leakage-safe time-series features. Every feature for week t uses information up to week t-1,
except `opening_stock`, which is known at the start of week t (decision time)."""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "gross_qty"


def build_features(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = panel.sort_values(["store", "group", "week_start"]).copy()
    g = df.groupby(["store", "group"], sort=False)[TARGET]
    for lag in cfg["lags"]:
        df[f"f_lag_{lag}"] = g.shift(lag)
    shifted = g.shift(1)
    sg = shifted.groupby([df.store, df.group], sort=False)
    for w in cfg["rolling_windows"]:
        df[f"f_roll_mean_{w}"] = sg.transform(lambda s: s.rolling(w, min_periods=1).mean())
        df[f"f_roll_std_{w}"] = sg.transform(lambda s: s.rolling(w, min_periods=2).std())
    df["f_expanding_mean"] = sg.transform(lambda s: s.expanding().mean())
    df["f_weeks_since_sale"] = sg.transform(_weeks_since_positive)
    df["f_promo_share_lag_1"] = df.groupby(["store", "group"], sort=False)["promo_share"].shift(1)
    df["f_opening_stock"] = df["opening_stock"]
    df["f_cover_weeks"] = df["opening_stock"] / (df[f"f_roll_mean_{cfg['rolling_windows'][0]}"] + 1.0)
    # Cross-sectional signals from the previous week (all stores / all groups), lagged.
    tot_group = df.groupby(["group", "week_start"])[TARGET].transform("sum")
    tot_store = df.groupby(["store", "week_start"])[TARGET].transform("sum")
    df["f_group_total_lag_1"] = tot_group.groupby([df.store, df.group], sort=False).shift(1)
    df["f_store_total_lag_1"] = tot_store.groupby([df.store, df.group], sort=False).shift(1)
    df["f_weekofyear"] = df.week_start.dt.isocalendar().week.astype(int)
    df["f_month"] = df.week_start.dt.month
    df["f_store"] = df.store.astype("category").cat.codes
    df["f_group"] = df.group.astype("category").cat.codes
    return df.reset_index(drop=True)


def _weeks_since_positive(s: pd.Series) -> pd.Series:
    """Applied to the 1-week-shifted target: 1 means the previous week had a sale; NaN if none yet."""
    out, last = np.full(len(s), np.nan), None
    for i, v in enumerate(s.to_numpy()):
        if not np.isnan(v) and v > 0:
            last = i
        if last is not None:
            out[i] = i - last + 1
    return pd.Series(out, index=s.index)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("f_")]
