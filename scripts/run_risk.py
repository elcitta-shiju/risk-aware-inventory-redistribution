"""Stages 5-6: inventory risk tables, stockout-probability calibration, store network.

Outputs: data/processed/risk.parquet, results/metrics/imbalance.csv,
         results/metrics/stockout_calibration.csv, results/metrics/risk_summary.json,
         results/metrics/store_network.csv (with provenance), figures.
"""

import _bootstrap  # noqa: F401

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from inventory_rl.risk.assessment import assess, imbalance_by_product_week, stockout_calibration
from inventory_rl.routing.network import build_network
from inventory_rl.utils.config import load_config, save_json
from inventory_rl.utils.paths import FIGURES_DIR, METRICS_DIR, PROCESSED_DIR, ensure_dirs
from inventory_rl.visualization.plots import BLUE, RED, _save

if __name__ == "__main__":
    ensure_dirs()
    cfg = load_config()
    alpha = cfg["uncertainty"]["alpha"]
    fc = pd.read_parquet(PROCESSED_DIR / "forecasts.parquet")
    risk = assess(fc, alpha, cfg["environment"]["excess_horizon_weeks"])
    keep = ["store", "group", "week_start", "split", "y", "yhat", "lo", "hi", "opening_stock", "stock_missing",
            "p_stockout", "expected_shortage", "shortfall_upper", "excess_units", "surplus_point",
            "surplus_upper", "cover_weeks", "risk_class"]
    risk[keep].to_parquet(PROCESSED_DIR / "risk.parquet", index=False)

    imb = imbalance_by_product_week(risk)
    imb = imb.merge(risk[["week_start", "split"]].drop_duplicates(), on="week_start")
    imb.to_csv(METRICS_DIR / "imbalance.csv", index=False)

    summary = {}
    for split in ("val", "test"):
        tab, brier = stockout_calibration(risk[risk.split == split])
        tab.to_csv(METRICS_DIR / f"stockout_calibration_{split}.csv", index=False)
        r = risk[risk.split == split]
        summary[split] = {
            "brier_p_stockout": brier,
            "risk_class_share": r.risk_class.value_counts(normalize=True).round(4).to_dict(),
            "mean_rebalanceable_units_per_product_week": float(imb[imb.split == split].rebalanceable_units.mean()),
            "share_product_weeks_with_rebalanceable_units": float((imb[imb.split == split].rebalanceable_units > 0).mean()),
        }
    save_json(summary, METRICS_DIR / "risk_summary.json")

    stores = sorted(fc.store.unique())
    net = build_network(stores, cfg)
    net.to_frame().to_csv(METRICS_DIR / "store_network.csv", index=False)
    pd.DataFrame(net.distance_km, index=stores, columns=stores).round(2).to_csv(METRICS_DIR / "distance_matrix_km.csv")

    # figures --------------------------------------------------------------------------
    tab, _ = stockout_calibration(risk[risk.split == "test"])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color="grey", ls="--", label="perfect calibration")
    ax.plot(tab.predicted, tab.observed, "o-", color=BLUE, label="Poisson(yhat) P(sold out)")
    for _, row in tab.iterrows():
        ax.annotate(f"n={row.n}", (row.predicted, row.observed), fontsize=7, xytext=(3, -8), textcoords="offset points")
    ax.set(title="Stockout probability reliability (test, stock>0)", xlabel="Predicted P(D >= stock)",
           ylabel="Observed share sold out")
    ax.legend()
    _save(fig, FIGURES_DIR / "risk_stockout_reliability_test.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    wk = imb.groupby("week_start")[["shortfall_units", "surplus_units", "rebalanceable_units"]].sum()
    wk.plot(ax=ax, color=[RED, BLUE, "black"])
    ax.set(title="Network shortfall vs surplus (upper-bound based), all top-30 SKUs", xlabel="Week", ylabel="Units")
    _save(fig, FIGURES_DIR / "risk_imbalance_over_time.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    risk[risk.split == "test"].risk_class.value_counts().plot.barh(ax=ax, color=BLUE)
    ax.set(title="Risk class of store-SKU-weeks (test)", xlabel="Count")
    _save(fig, FIGURES_DIR / "risk_class_distribution_test.png")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(net.coords[:, 0], net.coords[:, 1], color=BLUE)
    for s, (x, y) in zip(stores, net.coords):
        ax.annotate(s.replace("STR-", ""), (x, y), fontsize=7)
    ax.set(title=f"Store locations ({net.provenance} - not real)", xlabel="km", ylabel="km")
    _save(fig, FIGURES_DIR / "network_simulated_locations.png")
    print(summary)
