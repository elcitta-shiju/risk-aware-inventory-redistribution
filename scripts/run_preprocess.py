"""Stage 2: build the weekly store x product panel and its feature table.

Outputs: data/processed/panel.parquet, data/processed/features.parquet,
         data/processed/unit_economics.csv, results/reports/preprocessing_report.json
"""

import _bootstrap  # noqa: F401

from inventory_rl.data import loaders
from inventory_rl.data.preprocess import PanelSpec, build_panel, quality_report, select_top_groups, unit_economics
from inventory_rl.features.build import build_features
from inventory_rl.utils.config import load_config, save_json
from inventory_rl.utils.paths import PROCESSED_DIR, REPORTS_DIR, ensure_dirs

if __name__ == "__main__":
    ensure_dirs()
    cfg = load_config()
    spec = PanelSpec.from_config(cfg)
    sales, inv = loaders.load_mendeley_sales(), loaders.load_mendeley_inventory()
    groups = select_top_groups(sales, spec, cfg["data"].get("top_n_products"))
    sales = sales[sales[spec.group_col].isin(groups)]
    inv = inv[inv[spec.group_col].isin(groups)]
    panel = build_panel(sales, inv, spec)
    panel.to_parquet(PROCESSED_DIR / "panel.parquet", index=False)
    unit_economics(sales, spec, groups).to_csv(PROCESSED_DIR / "unit_economics.csv", index=False)
    feats = build_features(panel, cfg["features"])
    feats.to_parquet(PROCESSED_DIR / "features.parquet", index=False)
    report = {
        "config": {"data": cfg["data"], "splits": cfg["splits"], "features": cfg["features"]},
        "selected_groups": groups,
        "quality": quality_report(panel),
        "feature_columns": [c for c in feats.columns if c.startswith("f_")],
    }
    save_json(report, REPORTS_DIR / "preprocessing_report.json")
    print(report["quality"])
