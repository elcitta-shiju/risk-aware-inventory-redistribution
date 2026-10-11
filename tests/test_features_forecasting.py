import numpy as np
import pandas as pd

from inventory_rl.features.build import build_features, feature_columns
from inventory_rl.forecasting.models import LGBMSpec, baseline_forecasts, rolling_origin_forecast

FCFG = {"lags": [1, 2], "rolling_windows": [4, 8]}


def synthetic_panel(n_weeks=12, seed=0):
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2025-06-02", periods=n_weeks, freq="W-MON")
    rows = []
    for st in ["S1", "S2"]:
        for g in ["A", "B"]:
            for w in weeks:
                rows.append({"store": st, "group": g, "week_start": w, "gross_qty": float(rng.poisson(2)),
                             "promo_share": 0.0, "opening_stock": 5.0, "split": "train"})
    return pd.DataFrame(rows)


def test_lag_features_use_only_past():
    p = synthetic_panel()
    f = build_features(p, FCFG).sort_values(["store", "group", "week_start"])
    s = f[(f.store == "S1") & (f.group == "A")]
    assert np.isnan(s.f_lag_1.iloc[0])
    np.testing.assert_array_equal(s.f_lag_1.iloc[1:].to_numpy(), s.gross_qty.iloc[:-1].to_numpy())


def test_features_unchanged_when_future_changes():
    """Leakage check: altering the target in week t must not change any feature of week <= t."""
    p = synthetic_panel()
    t = p.week_start.unique()[6]
    p2 = p.copy()
    p2.loc[p2.week_start >= t, "gross_qty"] += 100
    cols = feature_columns(build_features(p, FCFG))
    f1 = build_features(p, FCFG).set_index(["store", "group", "week_start"]).sort_index()
    f2 = build_features(p2, FCFG).set_index(["store", "group", "week_start"]).sort_index()
    mask = f1.index.get_level_values("week_start") <= t
    pd.testing.assert_frame_equal(f1.loc[mask, cols], f2.loc[mask, cols])


def test_rolling_origin_trains_only_on_past(monkeypatch):
    p = synthetic_panel()
    f = build_features(p, FCFG)
    seen = []

    class Recorder:
        def fit(self, X, y):
            seen.append(len(X))
            return self

        def predict(self, X):
            return np.full(len(X), -1.0)  # negative -> must be clipped to 0

    monkeypatch.setattr(LGBMSpec, "model", lambda self, seed: Recorder())
    weeks = sorted(f.week_start.unique())
    origins = [pd.Timestamp(w) for w in weeks[4:6]]
    out = rolling_origin_forecast(f, LGBMSpec(), origins)
    assert seen == [4 * 4, 5 * 4]           # 4 series x number of weeks strictly before origin
    assert (out.pred == 0).all()
    assert set(out.week_start) == set(origins)


def test_lgbm_forecast_runs_and_is_nonnegative():
    f = build_features(synthetic_panel(), FCFG)
    origins = [pd.Timestamp(sorted(f.week_start.unique())[-1])]
    out = rolling_origin_forecast(f, LGBMSpec(params={"n_estimators": 10, "verbose": -1, "min_child_samples": 2}), origins)
    assert len(out) == 4 and (out.pred >= 0).all() and np.isfinite(out.pred).all()


def test_baselines_match_features():
    f = build_features(synthetic_panel(), FCFG)
    b = baseline_forecasts(f)
    np.testing.assert_array_equal(b.naive.to_numpy(), f.f_lag_1.to_numpy())
