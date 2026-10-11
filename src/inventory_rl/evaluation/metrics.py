"""Forecast and interval metrics. All functions ignore rows where the actual is NaN."""

from __future__ import annotations

import numpy as np


def _clean(y, *arrs):
    y = np.asarray(y, dtype=float)
    arrs = [np.asarray(a, dtype=float) for a in arrs]
    m = np.isfinite(y)
    for a in arrs:
        m &= np.isfinite(a)
    return (y[m], *[a[m] for a in arrs])


def mae(y, yhat) -> float:
    y, yhat = _clean(y, yhat)
    return float(np.mean(np.abs(y - yhat))) if len(y) else float("nan")


def rmse(y, yhat) -> float:
    y, yhat = _clean(y, yhat)
    return float(np.sqrt(np.mean((y - yhat) ** 2))) if len(y) else float("nan")


def wape(y, yhat) -> float:
    """Sum |error| / sum |actual|. Undefined (NaN) when total actual demand is zero."""
    y, yhat = _clean(y, yhat)
    denom = np.sum(np.abs(y))
    return float(np.sum(np.abs(y - yhat)) / denom) if denom > 0 else float("nan")


def bias(y, yhat) -> float:
    """Mean forecast error (positive = over-forecast)."""
    y, yhat = _clean(y, yhat)
    return float(np.mean(yhat - y)) if len(y) else float("nan")


def coverage(y, lo, hi) -> float:
    y, lo, hi = _clean(y, lo, hi)
    return float(np.mean((y >= lo) & (y <= hi))) if len(y) else float("nan")


def mean_width(lo, hi) -> float:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return float(np.nanmean(hi - lo))


def interval_score(y, lo, hi, alpha: float) -> float:
    """Winkler / interval score (lower is better): width + (2/alpha) * miss distance."""
    y, lo, hi = _clean(y, lo, hi)
    s = (hi - lo) + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)
    return float(np.mean(s)) if len(y) else float("nan")


def pinball(y, q, tau: float) -> float:
    y, q = _clean(y, q)
    d = y - q
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d))) if len(y) else float("nan")


def point_metrics(y, yhat) -> dict[str, float]:
    return {"MAE": mae(y, yhat), "RMSE": rmse(y, yhat), "WAPE": wape(y, yhat), "Bias": bias(y, yhat)}


def interval_metrics(y, lo, hi, alpha: float) -> dict[str, float]:
    y_, hi_ = _clean(y, hi)
    return {
        "coverage": coverage(y, lo, hi),
        "nominal": 1 - alpha,
        "upper_coverage": float(np.mean(y_ <= hi_)) if len(y_) else float("nan"),
        "upper_nominal": 1 - alpha / 2,
        "mean_width": mean_width(lo, hi),
        "interval_score": interval_score(y, lo, hi, alpha),
    }
