"""Stage 3: demand forecasting with simple baselines and a global LightGBM model.

All forecasts are produced by rolling-origin evaluation: for each origin week t the model is
fit only on rows with week_start < t and predicts week t (one-week-ahead). Hence every forecast,
including those for training weeks used by the RL simulator, is out-of-sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from inventory_rl.features.build import TARGET, feature_columns

KEYS = ["store", "group", "week_start"]


def baseline_forecasts(feats: pd.DataFrame) -> pd.DataFrame:
    """Naive, moving-average and expanding-mean forecasts (all use data up to t-1 only)."""
    out = feats[KEYS].copy()
    out["naive"] = feats["f_lag_1"]
    out["ma4"] = feats["f_roll_mean_4"]
    out["ma8"] = feats["f_roll_mean_8"]
    out["expanding_mean"] = feats["f_expanding_mean"]
    return out


@dataclass
class LGBMSpec:
    objective: str = "poisson"            # 'poisson' | 'regression' | 'quantile'
    alpha: float | None = None            # quantile level when objective == 'quantile'
    params: dict = field(default_factory=dict)

    def model(self, seed: int) -> lgb.LGBMRegressor:
        kw = dict(self.params, objective=self.objective, random_state=seed, n_jobs=-1)
        if self.objective == "quantile":
            kw["alpha"] = self.alpha
        return lgb.LGBMRegressor(**kw)


def rolling_origin_forecast(
    feats: pd.DataFrame, spec: LGBMSpec, origins: list[pd.Timestamp], seed: int = 42, return_models: bool = False
):
    """Fit on weeks < origin, predict the origin week, for every origin. Returns predictions."""
    cols = feature_columns(feats)
    preds, models = [], {}
    for t in origins:
        train = feats[(feats.week_start < t) & feats[TARGET].notna()]
        test = feats[feats.week_start == t]
        m = spec.model(seed)
        m.fit(train[cols], train[TARGET])
        p = test[KEYS].copy()
        p["pred"] = np.clip(m.predict(test[cols]), 0, None)
        preds.append(p)
        if return_models:
            models[t] = m
    out = pd.concat(preds, ignore_index=True)
    return (out, models) if return_models else out


def forecast_origins(feats: pd.DataFrame, min_history_weeks: int) -> list[pd.Timestamp]:
    weeks = sorted(feats.week_start.unique())
    return [pd.Timestamp(w) for w in weeks[min_history_weeks:]]


def feature_importance(model: lgb.LGBMRegressor, cols: list[str]) -> pd.Series:
    return pd.Series(model.booster_.feature_importance("gain"), index=cols).sort_values(ascending=False)
