"""Data-quality profiling for the raw datasets (Stage 1)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from inventory_rl.data import loaders


def profile(df: pd.DataFrame, key: list[str] | None = None, date_cols: list[str] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "rows": int(len(df)),
        "columns": {c: str(t) for c, t in df.dtypes.items()},
        "missing": {c: int(n) for c, n in df.isna().sum().items() if n},
        "duplicate_rows": int(df.duplicated().sum()),
    }
    if key:
        out["duplicate_keys"] = int(df.duplicated(key).sum())
    for c in date_cols or []:
        out[f"{c}_range"] = [str(df[c].min()), str(df[c].max())]
    num = df.select_dtypes("number")
    out["numeric_summary"] = num.describe().T[["mean", "min", "50%", "max"]].round(3).to_dict("index")
    out["negative_counts"] = {c: int((num[c] < 0).sum()) for c in num.columns if (num[c] < 0).any()}
    return out


def _lag1_autocorr(df: pd.DataFrame, keys: list[str], col: str) -> float:
    return float(df.groupby(keys)[col].apply(lambda s: s.autocorr(1)).mean())


def audit_kaggle(df: pd.DataFrame, name: str) -> dict[str, Any]:
    keys = ["store_id", "product_id"]
    df = df.sort_values(keys + ["date"])
    res = profile(df, key=["date", *keys], date_cols=["date"])
    res["n_stores"] = int(df.store_id.nunique())
    res["n_products"] = int(df.product_id.nunique())
    res["n_dates"] = int(df.date.nunique())
    res["max_categories_per_store_product"] = int(df.groupby(keys).category.nunique().max())
    res["max_regions_per_store"] = int(df.groupby("store_id").region.nunique().max())
    res["units_sold_lag1_autocorr_mean"] = round(_lag1_autocorr(df, keys, "units_sold"), 4)
    nxt = df.groupby(keys)["inventory_level"].shift(-1)
    balance = df.inventory_level - df.units_sold + df.units_ordered
    res["corr_next_inventory_vs_flow_balance"] = round(float(nxt.corr(balance)), 4)
    res["units_sold_exceeds_inventory_rows"] = int((df.units_sold > df.inventory_level).sum())
    if "demand_forecast" in df:
        res["corr_units_sold_vs_provided_demand_forecast"] = round(float(df.units_sold.corr(df.demand_forecast)), 4)
        res["negative_demand_forecast_rows"] = int((df.demand_forecast < 0).sum())
    if "demand" in df:
        res["units_sold_gt_demand_rows"] = int((df.units_sold > df.demand).sum())
        res["corr_units_sold_vs_demand"] = round(float(df.units_sold.corr(df.demand)), 4)
    res["dataset"] = name
    return res


def audit_mendeley(sales: pd.DataFrame, inv: pd.DataFrame) -> dict[str, Any]:
    res: dict[str, Any] = {"dataset": "Mendeley Retail Transactions and Stocks"}
    res["sales"] = profile(sales, date_cols=["transaction_date"])
    res["sales"]["n_stores"] = int(sales.store.nunique())
    res["sales"]["n_products"] = int(sales.product_no.nunique())
    res["sales"]["return_rows"] = int(sales.is_return.sum())
    res["sales"]["returns_with_nonnegative_qty"] = int(((sales.is_return == 1) & (sales.qty_sold >= 0)).sum())
    res["sales"]["sales_with_negative_qty"] = int(((sales.is_return == 0) & (sales.qty_sold < 0)).sum())
    res["sales"]["single_valued_columns"] = [c for c in sales.columns if sales[c].nunique(dropna=False) == 1]
    for c in ["sales_type", "product_division"]:
        res["sales"][f"{c}_counts"] = sales[c].value_counts().to_dict()
    pairs = sales.groupby(["store", "product_no"]).size()
    res["sales"]["transactions_per_store_product"] = pairs.describe().round(2).to_dict()
    res["sales"]["stores_per_product"] = sales.groupby("product_no").store.nunique().describe().round(2).to_dict()
    res["sales"]["n_categories"] = int(sales.product_category.nunique())

    key = ["store", "product_no", "stock_status"]
    res["inventory"] = profile(inv, key=["start_date", *key], date_cols=["start_date", "end_date"])
    res["inventory"]["open_records"] = int(inv.is_open.sum())
    res["inventory"]["start_after_end"] = int((inv.start_date > inv.end_date).sum())
    res["inventory"]["negative_qty_rows"] = int((inv.qty_on_hand < 0).sum())
    res["inventory"]["zero_qty_rows"] = int((inv.qty_on_hand == 0).sum())
    inv = inv.sort_values([*key, "start_date"])
    g = inv.groupby(key)
    gap = (inv.start_date - g.end_date.shift()).dt.days
    res["inventory"]["interval_gap_days_counts"] = {str(k): int(v) for k, v in gap.value_counts().head(8).items()}
    res["inventory"]["overlapping_intervals"] = int((gap < 1).sum())
    # Consistency: on contiguous transitions, does the stock change equal minus that day's net sales?
    dq = inv.qty_on_hand - g.qty_on_hand.shift()
    day = sales.groupby(["store", "product_no", "transaction_date"]).qty_sold.sum()
    t = inv.loc[gap == 1, ["store", "product_no", "start_date"]].assign(dq=dq[gap == 1])
    idx = pd.MultiIndex.from_frame(t[["store", "product_no", "start_date"]])
    t["sold"] = day.reindex(idx).fillna(0).to_numpy()
    res["inventory"]["contiguous_transitions"] = int(len(t))
    res["inventory"]["share_stock_change_equals_minus_sales"] = round(float((t.dq == -t.sold).mean()), 4)
    res["inventory"]["share_increase_without_sale (receipts/transfers in)"] = round(
        float(((t.dq > 0) & (t.sold == 0)).mean()), 4
    )
    res["inventory"]["single_valued_columns"] = [c for c in inv.columns if inv[c].nunique(dropna=False) == 1]
    sp_inv = set(map(tuple, inv[["store", "product_no"]].drop_duplicates().to_numpy()))
    sp_sales = set(map(tuple, sales[["store", "product_no"]].drop_duplicates().to_numpy()))
    res["linkage"] = {
        "store_products_in_sales": len(sp_sales),
        "store_products_in_inventory": len(sp_inv),
        "sales_pairs_without_inventory_record": len(sp_sales - sp_inv),
        "stores_match": sorted(set(sales.store)) == sorted(set(inv.store)),
    }
    return res


def run_full_audit() -> dict[str, Any]:
    sales = loaders.load_mendeley_sales()
    inv = loaders.load_mendeley_inventory()
    return {
        "files": loaders.discover_raw_files().to_dict("records"),
        "mendeley": audit_mendeley(sales, inv),
        "kaggle_sales_data": audit_kaggle(loaders.load_kaggle_sales_data(), "Kaggle sales_data.csv"),
        "kaggle_retail_store_inventory": audit_kaggle(
            loaders.load_kaggle_retail_store_inventory(), "Kaggle retail_store_inventory.csv"
        ),
    }


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    return obj
