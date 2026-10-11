"""Arrange panel + forecasts into dense (week x store) arrays per product for simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ARRAY_FIELDS = ["stock0", "receipts", "demand_obs", "lam", "yhat", "lo", "hi", "cqr_lo", "cqr_hi"]


@dataclass
class ProductTensors:
    product: str
    split: str
    weeks: list[pd.Timestamp]
    stock0: np.ndarray       # historical opening stock (NaN = no record)
    receipts: np.ndarray     # historical net receipts/adjustments (NaN -> 0)
    demand_obs: np.ndarray   # observed gross sales (censored demand)
    lam: np.ndarray          # simulator demand intensity (smoothed observed sales; NOT seen by agents)
    yhat: np.ndarray         # as-of point forecast
    lo: np.ndarray           # selected interval (Poisson) lower bound
    hi: np.ndarray           # selected interval upper bound
    cqr_lo: np.ndarray       # conformalized quantile-regression bounds
    cqr_hi: np.ndarray
    unit_margin: float
    unit_cost: float

    @property
    def n_weeks(self) -> int:
        return len(self.weeks)


def demand_intensity(y: pd.Series, keys: pd.DataFrame, window: int = 5) -> pd.Series:
    """Centered rolling mean of observed sales per series (simulator ground-truth proxy)."""
    return y.groupby([keys.store, keys.group], sort=False).transform(
        lambda s: s.rolling(window, center=True, min_periods=1).mean()
    )


def build_tensors(fc: pd.DataFrame, econ: pd.DataFrame, stores: list[str], split: str,
                  lam_window: int = 5) -> dict[str, ProductTensors]:
    """fc: forecasts table (one row per store/group/week with y, yhat, lo, hi, lo_cqr, hi_cqr,
    opening_stock, receipts_est). Only weeks of `split` are used, so the simulator for one split
    never touches data from another split."""
    d = fc[fc.split == split].sort_values(["group", "store", "week_start"]).copy()
    d["lam"] = demand_intensity(d.y, d[["store", "group"]], lam_window)
    econ = econ.set_index("group")
    out = {}
    for prod, g in d.groupby("group"):
        weeks = sorted(g.week_start.unique())

        def arr(col, fill=np.nan):
            p = g.pivot(index="week_start", columns="store", values=col).reindex(index=weeks, columns=stores)
            return p.to_numpy(float) if np.isnan(fill) else p.fillna(fill).to_numpy(float)

        out[prod] = ProductTensors(
            product=prod, split=split, weeks=[pd.Timestamp(w) for w in weeks],
            stock0=arr("opening_stock"), receipts=arr("receipts_est", 0.0), demand_obs=arr("y", 0.0),
            lam=arr("lam", 0.0), yhat=arr("yhat", 0.0), lo=arr("lo", 0.0), hi=arr("hi", 0.0),
            cqr_lo=arr("lo_cqr", 0.0), cqr_hi=arr("hi_cqr", 0.0),
            unit_margin=float(econ.loc[prod, "unit_margin"]), unit_cost=float(econ.loc[prod, "unit_cost"]),
        )
    return out
