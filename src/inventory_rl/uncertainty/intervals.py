"""Stage 4: prediction intervals for one-week-ahead demand.

Methods (all produce [lo, hi] at nominal level 1 - alpha):
  poisson   parametric Poisson quantiles around the point forecast (assumes Poisson demand)
  qr        LightGBM quantile regression (tau = alpha/2, 1 - alpha/2), uncalibrated
  cqr       conformalized quantile regression (Romano et al., 2019) with a ROLLING calibration
            window: for week t the correction is computed only from out-of-sample scores of the
            previous `window` weeks, so no future information is used.
  conformal split-conformal around the point forecast with scale-normalized residuals,
            same rolling calibration window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def poisson_interval(mu: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    mu = np.clip(np.asarray(mu, float), 1e-9, None)
    return stats.poisson.ppf(alpha / 2, mu), stats.poisson.ppf(1 - alpha / 2, mu)


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Split-conformal threshold: the k-th smallest score, k = ceil((n+1)(1-alpha)).

    Returns +inf when k > n (too few calibration points for the requested level)."""
    scores = np.sort(np.asarray(scores, float))
    scores = scores[np.isfinite(scores)]
    n = len(scores)
    if n == 0:
        return float("nan")
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(scores[k - 1]) if k <= n else float("inf")


def rolling_conformal(
    df: pd.DataFrame, score_col: str, alpha: float, window: int, week_col: str = "week_start"
) -> pd.Series:
    """For each week t, conformal quantile of `score_col` over weeks [t-window, t-1] (y known)."""
    weeks = sorted(df[week_col].unique())
    q = {}
    for i, w in enumerate(weeks):
        past = weeks[max(0, i - window): i]
        q[w] = conformal_quantile(df.loc[df[week_col].isin(past), score_col].to_numpy(), alpha) if past else np.nan
    return df[week_col].map(q)


def add_intervals(df: pd.DataFrame, alpha: float, window: int) -> pd.DataFrame:
    """df needs columns: y, yhat, q_lo, q_hi, scale. Adds lo_/hi_ columns for each method."""
    out = df.copy()
    out["lo_poisson"], out["hi_poisson"] = poisson_interval(out.yhat, alpha)
    out["lo_qr"], out["hi_qr"] = out.q_lo.clip(lower=0), np.maximum(out.q_hi, out.q_lo).clip(lower=0)

    out["_s_cqr"] = np.maximum(out.q_lo - out.y, out.y - out.q_hi)
    qc = rolling_conformal(out, "_s_cqr", alpha, window)
    out["lo_cqr"] = (out.q_lo - qc).clip(lower=0)
    out["hi_cqr"] = (out.q_hi + qc).clip(lower=0)

    denom = out.scale.fillna(0) + 1.0
    out["_s_conf"] = (out.y - out.yhat).abs() / denom
    qn = rolling_conformal(out, "_s_conf", alpha, window)
    out["lo_conformal"] = (out.yhat - qn * denom).clip(lower=0)
    out["hi_conformal"] = out.yhat + qn * denom
    return out.drop(columns=["_s_cqr", "_s_conf"])


METHODS = ["poisson", "qr", "cqr", "conformal"]


def integerize(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Demand is integer-valued: [lo, hi] contains the same integers as [ceil lo, floor hi].

    A real interval containing no integer (e.g. [0.2, 0.7]) is widened to {ceil lo}, so the
    integer interval is never empty (this can only increase coverage).
    """
    lo_i = np.ceil(np.asarray(lo, float) - 1e-9)
    hi_i = np.floor(np.asarray(hi, float) + 1e-9)
    return lo_i + 0.0, np.maximum(hi_i, lo_i) + 0.0  # + 0.0 normalizes -0.0
