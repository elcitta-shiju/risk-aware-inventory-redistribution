"""Stage 1: profile the extracted Mendeley dataset and save EDA tables/figures.

Outputs: results/reports/data_audit.json, results/figures/eda_*.png
"""

import _bootstrap  # noqa: F401

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from inventory_rl.data import loaders
from inventory_rl.data.audit import json_safe, run_full_audit
from inventory_rl.data.preprocess import week_start
from inventory_rl.utils.config import save_json
from inventory_rl.utils.paths import FIGURES_DIR, REPORTS_DIR, ensure_dirs


def eda_figures() -> None:
    sales = loaders.load_mendeley_sales()
    sales = sales[sales.is_return == 0]
    sales["week"] = week_start(sales.transaction_date)

    fig, ax = plt.subplots(figsize=(10, 4))
    sales.groupby("week").qty_sold.sum().plot(ax=ax, color="#2a6f97")
    ax.set(title="Mendeley: total weekly units sold (all stores, returns excluded)", xlabel="Week", ylabel="Units")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_mendeley_weekly_sales.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    sales.groupby("product_category").qty_sold.sum().sort_values().plot.barh(ax=ax, color="#2a6f97")
    ax.set(title="Mendeley: units sold by product category", xlabel="Units")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_mendeley_category_sales.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    sales.groupby("store").qty_sold.sum().sort_values(ascending=False).plot.bar(ax=ax, color="#2a6f97")
    ax.set(title="Mendeley: units sold by store", ylabel="Units")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_mendeley_store_sales.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    ensure_dirs()
    report = json_safe(run_full_audit())
    save_json(report, REPORTS_DIR / "data_audit.json")
    eda_figures()
    print("Saved", REPORTS_DIR / "data_audit.json")
