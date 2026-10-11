"""Stage 2: build a weekly store x product-group panel from the Mendeley data.

Outputs one row per (store, group, week_start) with:
  gross_qty     units sold excluding returns (demand proxy; censored by stock-outs)
  return_qty    units returned
  net_qty       gross_qty - return_qty
  promo_share   share of gross units sold at non-full price
  opening_stock / closing_stock  reconstructed stock-on-hand (NaN = no stock record, NOT zero)
  receipts_est  closing - opening + net_qty  (inferred replenishment / transfers in)
  split         train | val | test
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

GROUP_COL_MAP = {
    "Product Category": "product_category",
    "Product Subcategory": "product_subcategory",
    "Product Division": "product_division",
    "Product No": "product_no",
}


@dataclass(frozen=True)
class PanelSpec:
    first_week_start: pd.Timestamp
    last_week_end: pd.Timestamp
    group_col: str
    train_end: pd.Timestamp
    val_end: pd.Timestamp

    @classmethod
    def from_config(cls, cfg: dict) -> "PanelSpec":
        d, s = cfg["data"], cfg["splits"]
        spec = cls(
            first_week_start=pd.Timestamp(d["first_full_week_start"]),
            last_week_end=pd.Timestamp(d["last_full_week_end"]),
            group_col=GROUP_COL_MAP[d["product_level"]],
            train_end=pd.Timestamp(s["train_end"]),
            val_end=pd.Timestamp(s["val_end"]),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if self.first_week_start.dayofweek != 0:
            raise ValueError("first_week_start must be a Monday")
        if self.last_week_end.dayofweek != 6:
            raise ValueError("last_week_end must be a Sunday")
        if not (self.first_week_start < self.train_end < self.val_end < self.last_week_end):
            raise ValueError("split boundaries must be strictly chronological")

    @property
    def weeks(self) -> pd.DatetimeIndex:
        return pd.date_range(self.first_week_start, self.last_week_end, freq="W-MON")


def select_top_groups(sales: pd.DataFrame, spec: PanelSpec, n: int | None) -> list[str]:
    """Top-n groups by gross units sold in the TRAINING period only (avoids look-ahead selection)."""
    end = spec.train_end + pd.Timedelta(days=6)
    s = sales[(sales.transaction_date >= spec.first_week_start) & (sales.transaction_date <= end) & (sales.is_return == 0)]
    ranked = s.groupby(spec.group_col).qty_sold.sum().sort_values(ascending=False, kind="stable")
    return ranked.index.tolist() if n is None else ranked.index[:n].tolist()


def unit_economics(sales: pd.DataFrame, spec: PanelSpec, groups: list[str]) -> pd.DataFrame:
    """Observed average unit price, unit cost (COGS) and unit margin per group from TRAINING sales."""
    end = spec.train_end + pd.Timedelta(days=6)
    s = sales[(sales.transaction_date <= end) & (sales.is_return == 0) & (sales.qty_sold > 0)]
    s = s[s[spec.group_col].isin(groups)]
    g = s.groupby(spec.group_col)[["qty_sold", "sales_amount", "cogs"]].sum()
    out = pd.DataFrame({"unit_price": g.sales_amount / g.qty_sold, "unit_cost": g.cogs / g.qty_sold})
    out["unit_margin"] = out.unit_price - out.unit_cost
    return out.rename_axis("group").reset_index()


def week_start(dates: pd.Series) -> pd.Series:
    """Monday of the ISO week containing each date."""
    return (dates - pd.to_timedelta(dates.dt.dayofweek, unit="D")).dt.normalize()


def assign_split(week: pd.Series, spec: PanelSpec) -> pd.Series:
    return pd.Series(
        np.select([week <= spec.train_end, week <= spec.val_end], ["train", "val"], "test"), index=week.index
    )


def weekly_sales(sales: pd.DataFrame, spec: PanelSpec) -> pd.DataFrame:
    s = sales.copy()
    s["week_start"] = week_start(s["transaction_date"])
    s = s[(s.week_start >= spec.first_week_start) & (s.week_start <= spec.weeks[-1])]
    is_ret = (s.is_return == 1) | (s.qty_sold < 0)
    s["gross_qty"] = np.where(is_ret, 0.0, s.qty_sold.clip(lower=0))
    s["return_qty"] = np.where(is_ret, -s.qty_sold.clip(upper=0), 0.0)
    s["promo_qty"] = np.where(s.sales_type.ne("Full Price"), s.gross_qty, 0.0)
    s["gross_amount"] = np.where(is_ret, 0.0, s.sales_amount)
    g = s.groupby(["store", spec.group_col, "week_start"], as_index=False)[
        ["gross_qty", "return_qty", "promo_qty", "gross_amount"]
    ].sum()
    return g.rename(columns={spec.group_col: "group"})


def stock_at(inv: pd.DataFrame, date: pd.Timestamp, group_col: str) -> pd.DataFrame:
    """End-of-day stock on `date` per (store, group), summing all stock statuses of the SKUs.

    Negative book stock is clipped to zero for the physical stock but counted in `neg_book_records`.
    Store-groups with no covering record are absent (i.e. unknown, not zero).
    """
    covers = (inv.start_date <= date) & (inv.end_date.isna() | (inv.end_date >= date))
    cur = inv.loc[covers, ["store", group_col, "qty_on_hand"]]
    out = cur.groupby(["store", group_col]).agg(
        stock=("qty_on_hand", lambda q: q.clip(lower=0).sum()),
        book_stock=("qty_on_hand", "sum"),
        neg_book_records=("qty_on_hand", lambda q: int((q < 0).sum())),
        n_records=("qty_on_hand", "size"),
    )
    return out.reset_index().rename(columns={group_col: "group"})


def weekly_stock(inv: pd.DataFrame, spec: PanelSpec) -> pd.DataFrame:
    """Closing stock at each week end (Sunday) plus the Sunday before the first week."""
    ends = [spec.first_week_start - pd.Timedelta(days=1)] + [w + pd.Timedelta(days=6) for w in spec.weeks]
    frames = []
    for d in ends:
        f = stock_at(inv, d, spec.group_col)
        f["week_start"] = d - pd.Timedelta(days=6)  # the week this Sunday closes
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def build_panel(sales: pd.DataFrame, inv: pd.DataFrame, spec: PanelSpec) -> pd.DataFrame:
    ws = weekly_sales(sales, spec)
    st = weekly_stock(inv, spec)
    # Series universe defined from the TRAINING period only (no look-ahead in series selection).
    train_ws = ws[ws.week_start <= spec.train_end]
    train_st = st[(st.week_start <= spec.train_end) & (st.stock > 0)]
    pairs = pd.concat([train_ws[["store", "group"]], train_st[["store", "group"]]]).drop_duplicates()
    grid = pairs.merge(pd.DataFrame({"week_start": spec.weeks}), how="cross")
    panel = grid.merge(ws, on=["store", "group", "week_start"], how="left")
    sales_cols = ["gross_qty", "return_qty", "promo_qty", "gross_amount"]
    panel[sales_cols] = panel[sales_cols].fillna(0.0)  # no transaction in a week == zero sales (observed)
    panel["net_qty"] = panel.gross_qty - panel.return_qty
    panel["promo_share"] = np.where(panel.gross_qty > 0, panel.promo_qty / panel.gross_qty.where(panel.gross_qty > 0), 0.0)

    close = st.rename(columns={"stock": "closing_stock", "book_stock": "closing_book_stock"})
    panel = panel.merge(
        close[["store", "group", "week_start", "closing_stock", "closing_book_stock", "neg_book_records"]],
        on=["store", "group", "week_start"],
        how="left",
    )
    prev = close[["store", "group", "week_start", "closing_stock"]].copy()
    prev["week_start"] = prev.week_start + pd.Timedelta(days=7)
    panel = panel.merge(
        prev.rename(columns={"closing_stock": "opening_stock"}), on=["store", "group", "week_start"], how="left"
    )
    panel["stock_recorded"] = panel.closing_stock.notna()
    panel["receipts_est"] = panel.closing_stock - panel.opening_stock + panel.net_qty
    panel["split"] = assign_split(panel.week_start, spec)
    panel = panel.sort_values(["store", "group", "week_start"]).reset_index(drop=True)
    return panel


def quality_report(panel: pd.DataFrame) -> dict:
    return {
        "rows": int(len(panel)),
        "series": int(panel.groupby(["store", "group"]).ngroups),
        "stores": int(panel.store.nunique()),
        "groups": int(panel.group.nunique()),
        "weeks": int(panel.week_start.nunique()),
        "week_range": [str(panel.week_start.min().date()), str(panel.week_start.max().date())],
        "rows_per_split": panel.split.value_counts().to_dict(),
        "zero_sales_share": round(float((panel.gross_qty == 0).mean()), 4),
        "closing_stock_missing_share": round(float(panel.closing_stock.isna().mean()), 4),
        "opening_stock_missing_share": round(float(panel.opening_stock.isna().mean()), 4),
        "rows_with_negative_book_stock_records": int((panel.neg_book_records.fillna(0) > 0).sum()),
        "negative_receipts_est_share": round(float((panel.receipts_est < 0).mean()), 4),
        "stockout_weeks_share (opening_stock==0)": round(float((panel.opening_stock == 0).mean()), 4),
        "units_sold_in_weeks_with_missing_stock_record": float(panel.loc[panel.opening_stock.isna(), "gross_qty"].sum()),
        "units_sold_total": float(panel.gross_qty.sum()),
        "duplicate_keys": int(panel.duplicated(["store", "group", "week_start"]).sum()),
    }
