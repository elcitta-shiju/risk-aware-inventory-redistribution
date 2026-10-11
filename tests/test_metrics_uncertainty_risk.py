import numpy as np
import pandas as pd
import pytest
from scipy import stats

from inventory_rl.evaluation import metrics as M
from inventory_rl.risk.assessment import assess, expected_shortage, p_at_least, stockout_calibration
from inventory_rl.routing.network import simulated_network, transfer_cost
from inventory_rl.uncertainty.intervals import conformal_quantile, integerize, poisson_interval, rolling_conformal


# ---------------------------------------------------------------- metrics
def test_point_metrics():
    y, f = np.array([0, 2, 4.0]), np.array([1, 2, 2.0])
    assert M.mae(y, f) == pytest.approx(1.0)
    assert M.rmse(y, f) == pytest.approx(np.sqrt(5 / 3))
    assert M.wape(y, f) == pytest.approx(3 / 6)
    assert M.bias(y, f) == pytest.approx(-1 / 3)


def test_wape_undefined_for_zero_demand_and_nan_ignored():
    assert np.isnan(M.wape([0, 0], [1, 1]))
    assert M.mae([1, np.nan], [0, 5]) == 1.0


def test_interval_metrics():
    y, lo, hi = np.array([0, 1, 5.0]), np.zeros(3), np.full(3, 2.0)
    assert M.coverage(y, lo, hi) == pytest.approx(2 / 3)
    # width 2 + (2/0.2) * 3 for the miss, averaged
    assert M.interval_score(y, lo, hi, 0.2) == pytest.approx((2 + 2 + 2 + 30) / 3)
    m = M.interval_metrics(y, lo, hi, 0.2)
    assert all(np.isfinite(v) for v in m.values())


# ---------------------------------------------------------------- uncertainty
def test_conformal_quantile_finite_sample():
    s = np.arange(1, 11, dtype=float)
    assert conformal_quantile(s, 0.2) == 9.0           # ceil(11*0.8)=9 -> 9th smallest
    assert np.isnan(conformal_quantile(np.array([]), 0.2))


def test_rolling_conformal_uses_only_past_weeks():
    w = pd.date_range("2026-01-05", periods=4, freq="W-MON")
    df = pd.DataFrame({"week_start": np.repeat(w, 2), "s": [1, 1, 2, 2, 3, 3, 1000, 1000]})
    q = rolling_conformal(df, "s", alpha=0.5, window=2)
    assert np.isnan(q.iloc[0])                          # no past -> undefined
    assert q.iloc[6] <= 3                               # week 4 never sees its own huge scores


def test_integerize_nonempty_and_poisson():
    lo, hi = integerize(np.array([0.2, 0.0]), np.array([0.7, 2.5]))
    assert list(lo) == [1, 0] and list(hi) == [1, 2]
    plo, phi = poisson_interval(np.array([2.0]), 0.2)
    assert plo[0] == stats.poisson.ppf(0.1, 2) and phi[0] == stats.poisson.ppf(0.9, 2)


# ---------------------------------------------------------------- risk
def test_p_at_least_and_expected_shortage():
    assert p_at_least(np.array([0.0]), np.array([0.5]))[0] == 1.0
    assert p_at_least(np.array([1.0]), np.array([0.5]))[0] == pytest.approx(1 - np.exp(-0.5))
    mu, s = 1.3, 2
    k = np.arange(80)
    brute = np.sum(np.clip(k - s, 0, None) * stats.poisson.pmf(k, mu))
    assert expected_shortage(np.array([s]), np.array([mu]))[0] == pytest.approx(brute)


def test_assess_keeps_missing_stock_unknown():
    df = pd.DataFrame({"store": ["a", "b", "c"], "group": "g", "week_start": pd.Timestamp("2026-01-05"),
                       "opening_stock": [np.nan, 0.0, 10.0], "yhat": [1.0, 1.0, 0.2], "hi": [2.0, 2.0, 1.0],
                       "y": [0, 1, 0]})
    r = assess(df, alpha=0.2, excess_horizon_weeks=4)
    assert r.risk_class.iloc[0] == "unknown stock" and np.isnan(r.p_stockout.iloc[0])
    assert r.risk_class.iloc[1] == "high stockout risk"
    assert r.excess_units.iloc[2] > 0 and r.surplus_upper.iloc[2] == 9
    tab, brier = stockout_calibration(r)
    assert 0 <= brier <= 1


# ---------------------------------------------------------------- routing
def test_simulated_network_and_cost():
    net = simulated_network(["a", "b", "c"], 100, 1.3, seed=1)
    d = net.distance_km
    assert net.provenance == "SIMULATED"
    np.testing.assert_allclose(d, d.T)
    assert (np.diag(d) == 0).all() and (d[~np.eye(3, dtype=bool)] > 0).all()
    assert transfer_cost(0, 50, 8, 0.03) == 0
    assert transfer_cost(2, 50, 8, 0.03) == pytest.approx(8 + 0.03 * 2 * 50)
