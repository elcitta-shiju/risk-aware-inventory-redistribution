"""Stages 3-4: forecasting model selection, rolling-origin forecasts and prediction intervals.

Outputs: data/processed/forecasts.parquet, models/forecast_config.json,
         results/metrics/forecast_metrics.json, results/metrics/interval_metrics.json, figures.
"""

import _bootstrap  # noqa: F401

import itertools
import time

import numpy as np
import pandas as pd

from inventory_rl.evaluation.metrics import interval_metrics, point_metrics
from inventory_rl.features.build import TARGET, feature_columns
from inventory_rl.forecasting.models import (
    KEYS,
    LGBMSpec,
    baseline_forecasts,
    feature_importance,
    forecast_origins,
    rolling_origin_forecast,
)
from inventory_rl.uncertainty.intervals import METHODS, add_intervals, integerize
from inventory_rl.utils.config import load_config, save_json
from inventory_rl.utils.paths import FIGURES_DIR, METRICS_DIR, MODELS_DIR, PROCESSED_DIR, ensure_dirs
from inventory_rl.visualization import plots


def main() -> None:
    ensure_dirs()
    cfg = load_config()
    fcfg, ucfg, seed = cfg["forecasting"], cfg["uncertainty"], cfg["seed"]
    alpha = ucfg["alpha"]
    feats = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    origins = forecast_origins(feats, fcfg["min_history_weeks"])
    split_of = feats.drop_duplicates("week_start").set_index("week_start").split
    val_origins = [t for t in origins if split_of[t] == "val"]

    # ---- 1. model selection on the validation weeks only ---------------------------------
    t0 = time.time()
    base = fcfg["lgbm"]
    grid = []
    for obj, leaves in itertools.product(["poisson", "regression"], [7, 15, 31]):
        spec = LGBMSpec(objective=obj, params={**base, "num_leaves": leaves})
        p = rolling_origin_forecast(feats, spec, val_origins, seed)
        p = p.merge(feats[KEYS + [TARGET]], on=KEYS)
        grid.append({"objective": obj, "num_leaves": leaves, **point_metrics(p[TARGET], p.pred)})
    grid = pd.DataFrame(grid).sort_values("RMSE")
    best = grid.iloc[0]
    print(f"model selection ({time.time() - t0:.0f}s):\n{grid.round(4).to_string(index=False)}")
    point_spec = LGBMSpec(objective=best.objective, params={**base, "num_leaves": int(best.num_leaves)})

    # ---- 2. rolling-origin forecasts for all origin weeks -------------------------------
    t0 = time.time()
    point, models = rolling_origin_forecast(feats, point_spec, origins, seed, return_models=True)
    q_specs = {
        "q_lo": LGBMSpec("quantile", alpha / 2, point_spec.params),
        "q_hi": LGBMSpec("quantile", 1 - alpha / 2, point_spec.params),
    }
    fc = point.rename(columns={"pred": "yhat"})
    for name, spec in q_specs.items():
        fc = fc.merge(rolling_origin_forecast(feats, spec, origins, seed).rename(columns={"pred": name}), on=KEYS)
    print(f"rolling forecasts: {len(origins)} origins in {time.time() - t0:.0f}s")

    fc = fc.merge(feats[KEYS + [TARGET, "split", "f_roll_std_8", "opening_stock"]], on=KEYS)
    fc = fc.rename(columns={TARGET: "y", "f_roll_std_8": "scale"})
    fc = fc.merge(baseline_forecasts(feats), on=KEYS)

    # ---- 3. intervals; method chosen on validation interval score -----------------------
    fc = add_intervals(fc, alpha, ucfg["calibration_weeks"])
    for m in METHODS:
        fc[f"lo_{m}"], fc[f"hi_{m}"] = integerize(fc[f"lo_{m}"], fc[f"hi_{m}"])
    imetrics = {
        split: {m: interval_metrics(d.y, d[f"lo_{m}"], d[f"hi_{m}"], alpha) for m in METHODS}
        for split, d in fc.groupby("split")
    }
    chosen = min(METHODS, key=lambda m: imetrics["val"][m]["interval_score"])
    fc["lo"], fc["hi"] = fc[f"lo_{chosen}"], fc[f"hi_{chosen}"]
    print("interval method chosen on validation:", chosen)

    # ---- 4. point metrics per split ------------------------------------------------------
    fmetrics = {}
    for split, d in fc.groupby("split"):
        fmetrics[split] = {
            name: point_metrics(d.y, d[col])
            for name, col in [("LightGBM", "yhat"), ("Naive (t-1)", "naive"), ("MA(4)", "ma4"),
                              ("MA(8)", "ma8"), ("Expanding mean", "expanding_mean")]
        }
        fmetrics[split]["Zero forecast"] = point_metrics(d.y, np.zeros(len(d)))
        # zero-demand vs positive-demand weeks (intermittency diagnostic)
        fmetrics[split]["LightGBM | weeks with y>0"] = point_metrics(d.y[d.y > 0], d.yhat[d.y > 0])
        fmetrics[split]["LightGBM | weeks with y=0"] = point_metrics(d.y[d.y == 0], d.yhat[d.y == 0])

    fc.to_parquet(PROCESSED_DIR / "forecasts.parquet", index=False)
    cols = feature_columns(feats)
    imp = feature_importance(models[origins[-1]], cols)
    save_json({"point_model": {"objective": point_spec.objective, **point_spec.params},
               "quantile_levels": [alpha / 2, 1 - alpha / 2], "interval_method": chosen,
               "alpha": alpha, "calibration_weeks": ucfg["calibration_weeks"],
               "selection_grid_val": grid.to_dict("records"), "features": cols,
               "feature_importance_gain_last_origin": imp.round(2).to_dict(),
               "n_origins": len(origins), "first_origin": str(origins[0].date())},
              MODELS_DIR / "forecast_config.json")
    save_json(fmetrics, METRICS_DIR / "forecast_metrics.json")
    save_json({"chosen": chosen, "by_split": imetrics}, METRICS_DIR / "interval_metrics.json")

    # ---- 5. figures ----------------------------------------------------------------------
    plots.forecast_vs_actual(fc, FIGURES_DIR / "forecast_vs_actual_total.png",
                             "Weekly demand, all top-30 SKUs x 40 stores (rolling one-week-ahead)")
    top = fc.groupby(["store", "group"]).y.sum().sort_values(ascending=False).index[:4].tolist()
    plots.series_panel(fc, top, FIGURES_DIR / "forecast_series_examples.png")
    test = fc[fc.split == "test"]
    cov = test.groupby("group").apply(lambda d: pd.Series({"coverage": ((d.y >= d.lo) & (d.y <= d.hi)).mean()}))
    plots.coverage_by_group(cov, 1 - alpha, FIGURES_DIR / "interval_coverage_by_sku_test.png",
                            f"Test coverage by SKU ({chosen}, nominal {1 - alpha:.0%})")
    rows = [{"method": m, **imetrics["test"][m]} for m in METHODS]
    plots.bar_compare(pd.DataFrame(rows), "coverage", "method", FIGURES_DIR / "interval_coverage_methods_test.png",
                      f"Test coverage by interval method (nominal {1 - alpha:.0%})")
    print(pd.DataFrame(fmetrics["test"]).T.round(4))
    print(pd.DataFrame(imetrics["test"]).T.round(4))


if __name__ == "__main__":
    main()
