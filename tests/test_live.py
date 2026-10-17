import io
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from inventory_rl.live.upload import clean_weekly_sheet, normalize_code, parse_numbers, parse_weeks, read_sheet

STORES = ["STR-1006", "STR-1044"]
PRODUCTS = ["PROD-100358", "PROD-100435"]
LAST = pd.Timestamp("2026-04-13")   # Monday of the last week already in the data
TODAY = pd.Timestamp("2026-06-01")
ROOT = Path(__file__).resolve().parents[1]


def clean(csv: str, **kw):
    return clean_weekly_sheet(read_sheet(csv.encode(), "sheet.csv"), STORES, PRODUCTS, LAST, today=TODAY, **kw)


def messages(res) -> str:
    return " | ".join(i.message for i in res.issues)


# ---------------------------------------------------------------- reading and matching columns
def test_alternative_headers_title_rows_and_extra_columns():
    res = clean("Weekly report,,,,,\n,,,,,\n"
                "Week Ending,Shop,SKU,Qty Sold,SOH,Sales Value\n"
                "26/04/2026,1006,100358,3,10,226.50\n")
    assert res.ok, messages(res)
    assert res.column_map == {"week": "Week Ending", "store": "Shop", "product": "SKU", "units_sold": "Qty Sold",
                              "stock_start": "SOH"}
    row = res.weekly.iloc[0]
    assert (row.store, row.group, row.week_start) == ("STR-1006", "PROD-100358", pd.Timestamp("2026-04-20"))
    assert row.gross_qty == 3 and row.stock_start == 10 and np.isnan(row.stock_end)
    assert "Sales Value" in messages(res)  # reported as ignored, never read as units


def test_missing_required_column_is_an_error():
    res = clean("Week,Store,Units sold\n2026-04-20,STR-1006,3\n")
    assert not res.ok and res.issues[0].level == "error" and "Product" in res.issues[0].message


def test_semicolon_separated_file():
    res = clean("Week;Store;Product;Units sold\n2026-04-20;STR-1006;PROD-100358;2\n")
    assert res.ok and res.weekly.gross_qty.tolist() == [2]


def test_excel_file(tmp_path):
    pytest.importorskip("openpyxl")
    buf = io.BytesIO()
    pd.DataFrame({"Week": [pd.Timestamp("2026-04-22")], "Store": [1044], "Product": ["prod-100435"],
                  "Units sold": [5]}).to_excel(buf, index=False)
    res = clean_weekly_sheet(read_sheet(buf.getvalue(), "s.xlsx"), STORES, PRODUCTS, LAST, today=TODAY)
    assert res.ok, messages(res)
    assert res.weekly.iloc[0][["store", "group", "gross_qty"]].tolist() == ["STR-1044", "PROD-100435", 5]


# ---------------------------------------------------------------- value cleaning
def test_code_normalisation():
    s = pd.Series(["1006", "str-1006", "Store 1006", "STR 1006", "1006.0", "9999", "", None])
    out = normalize_code(s, STORES)
    assert out.iloc[:5].tolist() == ["STR-1006"] * 5 and out.iloc[5:].isna().all()


def test_number_parsing():
    num, bad = parse_numbers(pd.Series(["1,234", " 5 ", "(3)", "n/a", "-", "", "abc"]))
    assert num.tolist()[:3] == [1234, 5, -3] and num.iloc[3:].isna().all()
    assert bad.tolist() == [False] * 6 + [True]


def test_week_formats_all_move_to_monday():
    weeks, _ = parse_weeks(pd.Series(["2026-04-22", "2026-W17", "Week 17 2026", "26/04/2026", "46137"]))
    assert set(weeks) == {pd.Timestamp("2026-04-20")}


def test_ambiguous_dates_follow_the_rest_of_the_file():
    weeks, note = parse_weeks(pd.Series(["04/05/2026", "25/05/2026"]))  # 25/05 only fits day/month
    assert weeks.tolist() == [pd.Timestamp("2026-05-04"), pd.Timestamp("2026-05-25")]
    assert "day/month/year" in note


def test_row_level_fixes_and_drops():
    res = clean("Week,Store,Product,Units sold,Units returned,Stock at start of week,Units sold on promotion\n"
                "2026-04-20,STR-1006,PROD-100358,-2,0,5,0\n"     # negative sale -> return
                "2026-04-20,STR-1006,PROD-100435,2.6,0,-4,9\n"   # rounded, stock clipped, promo capped
                "2026-04-20,STR-9999,PROD-100358,1,0,1,0\n"      # unknown store
                "2026-04-20,STR-1006,PROD-999,1,0,1,0\n"         # product not covered
                "2026-04-06,STR-1044,PROD-100358,1,0,1,0\n"      # week already in the data
                "2026-07-06,STR-1044,PROD-100358,1,0,1,0\n"      # future week
                "2026-04-27,STR-1044,PROD-100358,1,0,1,0\n"
                "2026-04-27,STR-1044,PROD-100358,1,0,1,0\n"      # exact duplicate
                "2026-04-27,STR-1044,PROD-100435,1,0,4,0\n"
                "2026-04-27,1044,100435,2,0,4,0\n"               # same key, other row -> added up
                "Total,,,10,,,\n")
    w = res.weekly.set_index(["store", "group", "week_start"])
    a = w.loc[("STR-1006", "PROD-100358", pd.Timestamp("2026-04-20"))]
    assert (a.gross_qty, a.return_qty) == (0, 2)
    b = w.loc[("STR-1006", "PROD-100435", pd.Timestamp("2026-04-20"))]
    assert (b.gross_qty, b.stock_start, b.promo_qty) == (3, 0, 3)
    assert w.loc[("STR-1044", "PROD-100358", pd.Timestamp("2026-04-27"))].gross_qty == 1
    assert w.loc[("STR-1044", "PROD-100435", pd.Timestamp("2026-04-27"))].gross_qty == 3
    assert len(w) == 4
    text = messages(res)
    for phrase in ("STR-9999", "PROD-999", "already in the data", "not started yet", "exact duplicate",
                   "added together", "counted as returns", "rounded", "set to 0", "capped", "total"):
        assert phrase in text, phrase


def test_nothing_usable_is_an_error():
    res = clean("Week,Store,Product,Units sold\n2026-04-06,STR-1006,PROD-100358,1\n")
    assert not res.ok and any(i.level == "error" for i in res.issues)


# ---------------------------------------------------------------- sheets in other layouts
def test_unrecognised_columns_can_be_matched_by_hand():
    csv = "When,Where,What,How many,Left over\n2026-04-21,1006,100358,4,7\n"
    res = clean(csv)
    assert not res.ok and set(res.missing) == {"week", "store", "product", "units_sold"}
    assert res.columns_in_file == ["When", "Where", "What", "How many", "Left over"]
    res = clean(csv, mapping={"week": "When", "store": "Where", "product": "What", "units_sold": "How many",
                              "stock_end": "Left over"})
    assert res.ok, messages(res)
    assert res.weekly.iloc[0][["store", "gross_qty", "stock_end"]].tolist() == ["STR-1006", 4, 7]


def test_hand_matching_overrides_and_excludes_columns():
    csv = "Week,Store,Product,Qty,Units sold,Stock\n2026-04-21,1006,100358,9,4,7\n"
    res = clean(csv, mapping={"units_sold": "Units sold", "stock_start": None})
    assert res.ok, messages(res)
    row = res.weekly.iloc[0]
    assert row.gross_qty == 4 and np.isnan(row.stock_start)


def test_weeks_across_the_columns():
    res = clean("Store,Product,Stock,2026-04-20,2026-04-27,2026-W19\n"
                "1006,100358,5,3,,1\n"
                "1044,100435,2,0,2,4\n")
    assert res.ok, messages(res)
    w = res.weekly.set_index(["store", "group", "week_start"]).gross_qty
    assert w[("STR-1006", "PROD-100358", pd.Timestamp("2026-04-20"))] == 3
    assert ("STR-1006", "PROD-100358", pd.Timestamp("2026-04-27")) not in w.index   # blank cell: no row
    assert w[("STR-1044", "PROD-100435", pd.Timestamp("2026-05-04"))] == 4
    assert len(w) == 5
    assert "weeks run across" in messages(res) and "Stock could not be used" in messages(res)


def test_products_across_the_columns():
    res = clean("Week,Store,PROD-100358,100435\n2026-04-20,1006,3,1\n2026-04-20,1044,0,2\n")
    assert res.ok, messages(res)
    assert sorted(res.weekly.gross_qty) == [0, 1, 2, 3] and set(res.weekly.group) == set(PRODUCTS)
    assert "products run across" in messages(res)


def test_one_store_or_one_week_for_the_whole_file():
    csv = "Product,Units sold\n100358,3\n100435,1\n"
    res = clean(csv)
    assert not res.ok and set(res.missing) == {"week", "store"}
    res = clean(csv, store="STR-1044", week=pd.Timestamp("2026-04-23"))
    assert res.ok, messages(res)
    assert set(res.weekly.store) == {"STR-1044"} and set(res.weekly.week_start) == {pd.Timestamp("2026-04-20")}


def test_daily_rows_become_weeks_with_first_and_last_stock():
    rows = ["Date,Store,Product,Units sold,Opening stock,Closing stock"]
    for d in range(7):   # Mon 20 Apr .. Sun 26 Apr, stock falls by 1 a day
        rows.append(f"{pd.Timestamp('2026-04-20') + pd.Timedelta(days=d):%d/%m/%Y},1006,100358,1,{10 - d},{9 - d}")
    rows.append("27/04/2026,1006,100358,2,3,1")   # one day of the next week
    res = clean("\n".join(rows) + "\n")
    assert res.ok, messages(res)
    w = res.weekly.set_index("week_start")
    assert w.loc[pd.Timestamp("2026-04-20"), ["gross_qty", "stock_start", "stock_end"]].tolist() == [7, 10, 3]
    assert w.loc[pd.Timestamp("2026-04-27"), "gross_qty"] == 2
    text = messages(res)
    assert "added up into weekly totals" in text and "only 1 day(s)" in text
    assert "exact duplicate" not in text and "added together" not in text


def test_daily_rows_with_one_stock_column_take_the_last_figure_as_end_stock():
    res = clean("Txn Date,Store,Product,Qty,On Hand\n"
                "20/04/2026,1006,100358,1,8\n22/04/2026,1006,100358,2,6\n25/04/2026,1006,100358,1,4\n")
    assert res.ok, messages(res)
    assert res.weekly.iloc[0][["gross_qty", "stock_start", "stock_end"]].tolist() == [4, 8, 4]
    assert "'Txn Date' → Week" in messages(res)


def test_workbook_reads_every_sales_sheet_and_sheet_names_as_stores():
    pytest.importorskip("openpyxl")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as xw:
        pd.DataFrame({"Notes": ["exported from the till"]}).to_excel(xw, sheet_name="About", index=False)
        for store, units in (("1006", 3), ("Store 1044", 5)):
            pd.DataFrame({"Week": ["2026-04-20"], "SKU": [100358], "Qty Sold": [units]}).to_excel(
                xw, sheet_name=store, index=False)
    raw = read_sheet(buf.getvalue(), "book.xlsx")
    assert isinstance(raw, dict) and len(raw) == 3
    res = clean_weekly_sheet(raw, STORES, PRODUCTS, LAST, today=TODAY)
    assert res.ok, messages(res)
    assert res.weekly.set_index("store").gross_qty.to_dict() == {"STR-1006": 3, "STR-1044": 5}
    assert res.sheets_used == ["1006", "Store 1044"] and "skipped: About" in messages(res)


def test_title_line_without_separators_above_the_table():
    res = clean("Weekly sales export\nWeek,Store,Product,Units sold\n2026-04-20,1006,100358,2\n")
    assert res.ok, messages(res)
    assert res.weekly.gross_qty.tolist() == [2]


# ---------------------------------------------------------------- weekly table for uploaded weeks
def tiny_base() -> pd.DataFrame:
    """Two stores x one product, two weeks, with closing stock known in the last week."""
    rows = []
    for w in pd.date_range("2026-04-06", periods=2, freq="W-MON"):
        for s, close in (("S1", 5.0), ("S2", np.nan)):
            rows.append({"store": s, "group": "P", "week_start": w, "gross_qty": 1.0, "return_qty": 0.0,
                         "promo_qty": 0.0, "gross_amount": 10.0, "net_qty": 1.0, "promo_share": 0.0,
                         "closing_stock": close, "closing_book_stock": close, "neg_book_records": 0.0,
                         "opening_stock": close, "stock_recorded": not np.isnan(close), "receipts_est": 0.0,
                         "split": "test"})
    return pd.DataFrame(rows)


def uploads(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["store", "group", "week_start", "gross_qty", "return_qty", "promo_qty",
                                     "stock_start", "stock_end"])
    df["week_start"] = pd.to_datetime(df.week_start)
    return df


def test_live_panel_stock_and_reporting_rules():
    from inventory_rl.live.update import build_live_panel

    up = uploads([
        ("S1", "P", "2026-04-20", 2, 0, 0, np.nan, np.nan),   # start from last week's close (5), end estimated (3)
        ("S2", "P", "2026-04-20", 1, 0, 0, 4, 3),             # stock given
        # week of 27 Apr missing entirely; week of 4 May: only S1 reported, with a start stock
        ("S1", "P", "2026-05-04", 1, 0, 0, 6, np.nan),
    ])
    p = build_live_panel(tiny_base(), up).set_index(["store", "week_start"])
    w1, w2, w3, nxt = pd.date_range("2026-04-20", periods=4, freq="W-MON")
    assert p.loc[("S1", w1), ["opening_stock", "closing_stock", "stock_estimated"]].tolist() == [5, 3, False]
    assert p.loc[("S2", w1), ["opening_stock", "closing_stock"]].tolist() == [4, 3]
    # missing week: sales unknown; S1's opening carries the estimated close forward (flagged)
    assert np.isnan(p.loc[("S1", w2), "gross_qty"]) and p.loc[("S1", w2), "stock_estimated"]
    assert p.loc[("S1", w3), ["gross_qty", "opening_stock", "stock_estimated"]].tolist() == [1, 6, False]
    assert np.isnan(p.loc[("S2", w3), "gross_qty"])            # S2 not reported that week
    assert p.loc[("S1", nxt), "split"] == "next" and p.loc[("S1", nxt), "opening_stock"] == 5
    assert p.loc[("S1", nxt), "stock_estimated"]               # 6 - 1 sold, no end stock given
    assert set(p.split) == {"new", "next"}


def test_reported_store_missing_product_sold_zero():
    from inventory_rl.live.update import build_live_panel

    base = tiny_base()
    base = pd.concat([base, base.assign(group="Q")], ignore_index=True)
    p = build_live_panel(base, uploads([("S1", "P", "2026-04-20", 2, 0, 0, 6, 4)]))
    q = p[(p.store == "S1") & (p.group == "Q") & (p.split == "new")].iloc[0]
    assert q.gross_qty == 0 and q.opening_stock == 5            # carried from the last known close


# ---------------------------------------------------------------- end to end on the real artifacts
def test_upload_update_and_dashboard_end_to_end(tmp_path):
    """Clean a sheet, refresh forecasts/risk/transfers into temporary folders, then render the pages that
    show uploaded data. Skipped when the pipeline artifacts are not present."""
    needed = [ROOT / "data/processed/panel.parquet", ROOT / "data/processed/forecasts.parquet",
              ROOT / "models/forecast_config.json", ROOT / "models/rl/training_summary.json"]
    if not all(p.exists() for p in needed):
        pytest.skip("pipeline artifacts not generated yet")
    code = f"""
import sys
sys.path.insert(0, {str(ROOT / 'dashboard')!r})
import pandas as pd
from inventory_rl.live import update as up
from inventory_rl.live.upload import clean_weekly_sheet, read_sheet
from inventory_rl.utils.config import load_config
from inventory_rl.baselines.rules import RuleBasedPolicy
base = up.base_panel(); stores, prods = up.known_codes(base); last = base.week_start.max()
w = last + pd.Timedelta(days=7)
rows = ["Week,Store,Product,Units sold,Stock at start of week,Stock at end of week"]
for i, s in enumerate(stores):
    for j, g in enumerate(prods):
        rows.append(f"{{w:%d/%m/%Y}},{{s}},{{g}},{{(i + j) % 3}},{{(i * j) % 7}},{{max(0, (i * j) % 7 - (i + j) % 3)}}")
data = "\\n".join(rows).encode()
res = clean_weekly_sheet(read_sheet(data, "w.csv"), stores, prods, last, today=w + pd.Timedelta(days=7))
assert res.ok and len(res.weekly) == len(stores) * len(prods), [i.message for i in res.issues]
up.save_upload(res.weekly, data, "w.csv")
s = up.run_update(load_config(), {{"Rule: cover a high estimate": RuleBasedPolicy("upper")}}, progress=lambda m: None)
assert up.live_ready() and s["next_week"] == str((w + pd.Timedelta(days=7)).date())
risk = pd.read_parquet(up.LIVE_FILES["risk.parquet"])
assert set(risk.split) == {{"new", "next"}} and risk.yhat.notna().all() and risk.hi.notna().all()
assert (risk[risk.split == "next"].opening_stock.notna()).all()

from streamlit.testing.v1 import AppTest
at = AppTest.from_file({str(ROOT / 'dashboard' / 'app.py')!r}, default_timeout=300).run()
assert not at.exception, [e.value for e in at.exception]
assert at.radio[0].value.startswith("Next week"), at.radio[0].value
assert {{"Demand outlook", "Stock risk", "Recommended transfer"}} <= {{x.value for x in at.subheader}}
pages = {{i["url_pathname"] or "decision_support": h for h, i in at._registered_pages.items()}}
for name in ("store_overview", "add-data"):
    at._page_hash = pages[name]; at.run()
    assert not at.exception, (name, [e.value for e in at.exception])
assert at.metric[1].value == "1"   # weeks added
up.clear_uploads()
assert not up.live_ready() and up.load_uploads() is None
print("LIVE_OK")
"""
    env = {**__import__("os").environ, "INVENTORY_UPLOADS_DIR": str(tmp_path / "uploads"),
           "INVENTORY_LIVE_DIR": str(tmp_path / "live"), "PYTHONPATH": str(ROOT / "src")}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=900, env=env, cwd=ROOT)
    assert r.returncode == 0 and "LIVE_OK" in r.stdout, r.stderr[-3000:]


def test_terminal_add_data_flow(tmp_path):
    """Drive the terminal version with typed answers: upload a sheet, refresh the plan, open Decision Support on
    the added data, then remove it. Uses temporary folders; skipped without the pipeline artifacts."""
    if not (ROOT / "data/processed/panel.parquet").exists() or not (ROOT / "models/forecast_config.json").exists():
        pytest.skip("pipeline artifacts not generated yet")
    panel = pd.read_parquet(ROOT / "data/processed/panel.parquet", columns=["store", "group", "week_start"])
    week = panel.week_start.max() + pd.Timedelta(days=7)
    sheet = tmp_path / "week.csv"
    pd.DataFrame([{"Week": f"{week:%Y-%m-%d}", "Store": s, "Product": g, "Units sold": 1, "Stock at start of week": 3,
                   "Stock at end of week": 2} for s in sorted(panel.store.unique())
                  for g in sorted(panel.group.unique())]).to_csv(sheet, index=False)
    answers = ["3", "1", f'"{sheet}"', "y", "6", "",   # Add New Data: upload, confirm, back to menu
               "1", "", "", "", "", "",                 # Decision Support on the added data, all defaults
               "3", "5", "remove", "4", "", "0"]        # remove the added data, exit
    env = {**__import__("os").environ, "INVENTORY_UPLOADS_DIR": str(tmp_path / "uploads"),
           "INVENTORY_LIVE_DIR": str(tmp_path / "live"), "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run([sys.executable, str(ROOT / "dashboard" / "terminal.py")], input="\n".join(answers) + "\n",
                       capture_output=True, text=True, encoding="utf-8", timeout=900, env=env, cwd=ROOT)
    out = re.sub(r"\x1b\[[0-9;]*m", "", r.stdout)  # drop colour codes
    assert r.returncode == 0 and "Traceback" not in out + r.stderr, r.stderr[-3000:]
    assert "Rows that can be used: 1,200" in out
    assert f"The plan for the week of {week + pd.Timedelta(days=7):%d %b %Y} is ready" in out
    assert "(next week)" in out and "Recommended transfer" in out
    assert "All added data was removed." in out
    assert not any((tmp_path / "live").iterdir())
