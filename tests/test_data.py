import zipfile

import numpy as np
import pandas as pd
import pytest

from inventory_rl.data.extract import extract_archive, is_safe_member
from inventory_rl.data.loaders import discover_raw_files, snake
from inventory_rl.data.preprocess import PanelSpec, assign_split, build_panel, stock_at, week_start


def spec(**kw):
    base = dict(first_week_start=pd.Timestamp("2025-06-02"), last_week_end=pd.Timestamp("2025-06-22"),
                group_col="product_no", train_end=pd.Timestamp("2025-06-02"), val_end=pd.Timestamp("2025-06-09"))
    base.update(kw)
    return PanelSpec(**base)


# ---------------------------------------------------------------- discovery / extraction
@pytest.mark.parametrize("name,ok", [("a/data.csv", True), ("../evil.csv", False), ("/abs.csv", False),
                                     ("C:/x.csv", False), ("run.exe", False), ("notes.md", True)])
def test_is_safe_member(name, ok):
    assert is_safe_member(name) is ok


def test_extract_archive_never_overwrites(tmp_path):
    z = tmp_path / "src.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("folder/data.csv", "a,b\n1,2\n")
        zf.writestr("../escape.csv", "x")
        zf.writestr("tool.exe", "x")
    raw = tmp_path / "raw"
    rec = extract_archive(z, raw)
    assert rec.extracted == ["data.csv"]
    assert set(rec.rejected) == {"../escape.csv", "tool.exe"}
    rec2 = extract_archive(z, raw)  # second run: same size -> skipped, not rewritten
    assert rec2.skipped_existing == ["data.csv"] and rec2.extracted == []
    assert z.exists()  # original archive untouched
    files = discover_raw_files(raw)
    assert list(files.file) == ["data.csv"]


def test_snake():
    assert snake("Qty on hand") == "qty_on_hand"
    assert snake("Holiday/Promotion") == "holiday_promotion"


# ---------------------------------------------------------------- preprocessing
def test_panel_spec_validation():
    with pytest.raises(ValueError):
        spec(first_week_start=pd.Timestamp("2025-06-03")).validate()
    with pytest.raises(ValueError):
        spec(train_end=pd.Timestamp("2025-06-16"), val_end=pd.Timestamp("2025-06-09")).validate()


def test_week_start_is_monday():
    d = pd.Series(pd.to_datetime(["2025-06-02", "2025-06-08", "2025-06-09"]))
    assert list(week_start(d).dt.strftime("%Y-%m-%d")) == ["2025-06-02", "2025-06-02", "2025-06-09"]


def test_assign_split_is_chronological():
    s = spec()
    weeks = pd.Series(s.weeks)
    split = assign_split(weeks, s)
    assert list(split) == ["train", "val", "test"]


def test_stock_at_distinguishes_missing_from_zero(tiny_inventory):
    st = stock_at(tiny_inventory, pd.Timestamp("2025-06-22"), "product_no").set_index(["store", "group"])
    assert st.loc[("S1", "A"), "stock"] == 0              # explicit zero record
    assert ("S2", "B") not in st.index                     # no record -> absent (unknown)
    assert st.loc[("S2", "A"), "stock"] == 0               # negative book stock clipped
    assert st.loc[("S2", "A"), "neg_book_records"] == 1


def test_build_panel(tiny_sales, tiny_inventory):
    p = build_panel(tiny_sales, tiny_inventory, spec()).set_index(["store", "group", "week_start"])
    w1, w2, w3 = pd.Timestamp("2025-06-02"), pd.Timestamp("2025-06-09"), pd.Timestamp("2025-06-16")
    assert p.loc[("S1", "A", w1), "gross_qty"] == 3
    assert p.loc[("S1", "A", w2), "gross_qty"] == 1 and p.loc[("S1", "A", w2), "return_qty"] == 1
    assert p.loc[("S1", "A", w1), "opening_stock"] == 5   # stock at Sunday 2025-06-01
    assert p.loc[("S1", "A", w1), "closing_stock"] == 2
    assert p.loc[("S1", "A", w1), "receipts_est"] == 2 - 5 + 3
    assert np.isnan(p.loc[("S2", "B", w3), "closing_stock"])  # unknown, not zero
    assert not p.index.duplicated().any()
    assert list(p.reset_index().groupby("week_start").split.first()) == ["train", "val", "test"]
