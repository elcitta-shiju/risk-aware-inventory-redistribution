"""Stage 5: inventory risk measures at decision time (start of week t).

Definitions (D = next-week demand, s = opening stock):
  p_stockout          P(D >= s) under a Poisson(yhat) demand model -- a MODEL-BASED probability
                      whose calibration is checked empirically (see `stockout_calibration`).
                      For s > 0 the event "D >= s" is observable as "sold out" (sales >= s).
  expected_shortage   E[(D - s)^+] under Poisson(yhat), in units.
  shortfall_upper     max(0, hi - s): units missing to cover the upper prediction bound.
  excess_units        max(0, s - Q_{1-alpha/2}(Poisson(H * yhat))): stock beyond the upper bound
                      of demand over an H-week horizon (H is a documented assumption).
  surplus_point       max(0, s - ceil(yhat))      transferable without breaching the point forecast
  surplus_upper       max(0, s - hi)              transferable while still covering the upper bound
Missing stock records are kept as NaN and flagged; they are never silently set to zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def p_at_least(s: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """P(D >= s) for D ~ Poisson(mu); equals 1 when s <= 0."""
    s = np.asarray(s, float)
    mu = np.clip(np.asarray(mu, float), 1e-9, None)
    return np.where(s <= 0, 1.0, stats.poisson.sf(s - 1, mu))


def expected_shortage(s: np.ndarray, mu: np.ndarray, kmax: int = 60) -> np.ndarray:
    """E[(D - s)^+] for D ~ Poisson(mu), computed by summation over a truncated support."""
    s = np.clip(np.nan_to_num(np.asarray(s, float)), 0, None)
    mu = np.clip(np.asarray(mu, float), 1e-9, None)
    k = np.arange(kmax + 1)
    pmf = stats.poisson.pmf(k[None, :], mu[:, None])
    return (np.clip(k[None, :] - s[:, None], 0, None) * pmf).sum(axis=1)


def assess(df: pd.DataFrame, alpha: float, excess_horizon_weeks: int) -> pd.DataFrame:
    """df needs: store, group, week_start, opening_stock, yhat, hi. Returns risk table."""
    out = df.copy()
    s = out.opening_stock
    out["stock_missing"] = s.isna()
    out["p_stockout"] = np.where(s.isna(), np.nan, p_at_least(s.fillna(0), out.yhat))
    out["expected_shortage"] = np.where(s.isna(), np.nan, expected_shortage(s.fillna(0), out.yhat))
    out["shortfall_upper"] = (out.hi - s).clip(lower=0)
    horizon_hi = stats.poisson.ppf(1 - alpha / 2, np.clip(out.yhat * excess_horizon_weeks, 1e-9, None))
    out["excess_units"] = (s - horizon_hi).clip(lower=0)
    out["surplus_point"] = (s - np.ceil(out.yhat)).clip(lower=0)
    out["surplus_upper"] = (s - out.hi).clip(lower=0)
    out["cover_weeks"] = s / out.yhat.clip(lower=1e-3)
    out["risk_class"] = np.select(
        [out.stock_missing, out.p_stockout >= 0.5, out.shortfall_upper > 0, out.excess_units > 0],
        ["unknown stock", "high stockout risk", "below upper bound", "excess"],
        "balanced",
    )
    return out


def imbalance_by_product_week(risk: pd.DataFrame) -> pd.DataFrame:
    """Units that could in principle be rebalanced: min(total upper surplus, total upper shortfall)."""
    g = risk.groupby(["group", "week_start"])
    out = g.agg(
        stores_short=("shortfall_upper", lambda x: int((x > 0).sum())),
        stores_surplus=("surplus_upper", lambda x: int((x > 0).sum())),
        shortfall_units=("shortfall_upper", "sum"),
        surplus_units=("surplus_upper", "sum"),
        expected_shortage=("expected_shortage", "sum"),
        excess_units=("excess_units", "sum"),
    ).reset_index()
    out["rebalanceable_units"] = np.minimum(out.shortfall_units, out.surplus_units)
    return out


def stockout_calibration(risk: pd.DataFrame, bins: int = 10) -> tuple[pd.DataFrame, float]:
    """Reliability of p_stockout on rows with known positive stock; event = sales >= stock.

    Returns (binned table, Brier score)."""
    d = risk[(risk.opening_stock > 0) & risk.y.notna()].copy()
    d["event"] = (d.y >= d.opening_stock).astype(float)
    brier = float(np.mean((d.p_stockout - d.event) ** 2))
    d["bin"] = pd.cut(d.p_stockout, np.linspace(0, 1, bins + 1), include_lowest=True)
    tab = d.groupby("bin", observed=True).agg(predicted=("p_stockout", "mean"), observed=("event", "mean"),
                                              n=("event", "size")).reset_index()
    tab["bin"] = tab["bin"].astype(str)
    return tab, brier
