"""Read and clean a retailer's weekly sales-and-stock sheet (CSV or Excel), as typed or exported.

One row = one store, one product, one week. Column names are matched loosely (case, spaces and
common alternative names), and the values are cleaned:
  - week: any date inside the week (several date formats) or an ISO week such as 2026-W17;
    every date is moved to the Monday of its week
  - store / product codes: '1006', 'str 1006', 'Store 1006' all become STR-1006 (same for PROD-)
  - numbers: thousands separators, spaces, '(3)' and text such as 'n/a' or '-' are handled
  - negative units sold are counted as returns; negative stock is set to 0
  - exact duplicate rows are dropped; split rows for the same store/product/week are added up
Rows that cannot be used (unknown store or product, missing week, week already in the data, week
in the future) are dropped. Every change is reported as an `Issue` in plain language.

The result has one row per (store, group, week_start) in the units of the weekly panel; stock that
was not given stays missing (NaN) and is filled from neighbouring weeks later (see update.py).
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["store", "group", "week_start"]
VALUE_COLS = ["gross_qty", "return_qty", "promo_qty", "stock_start", "stock_end"]


@dataclass(frozen=True)
class SheetColumn:
    name: str            # column name in the template
    required: bool
    meaning: str
    aliases: tuple[str, ...]


# canonical name -> column definition. Aliases are compared after lower-casing and removing
# everything except letters and digits ("Qty Sold" -> "qtysold").
COLUMNS: dict[str, SheetColumn] = {
    "week": SheetColumn("Week", True, "Any date in the week (e.g. the Monday), or a week such as 2026-W17.",
                        ("week", "weekstart", "weekstarting", "weekbeginning", "weekcommencing", "weekof",
                         "weekending", "weekend", "weekenddate", "date", "salesdate", "period", "isoweek")),
    "store": SheetColumn("Store", True, "Store code, e.g. STR-1006 or just 1006.",
                         ("store", "storeid", "storecode", "storeno", "storenumber", "shop", "shopid", "branch",
                          "branchid", "outlet", "location", "site")),
    "product": SheetColumn("Product", True, "Product code, e.g. PROD-100358 or just 100358.",
                           ("product", "productid", "productcode", "productno", "productnumber", "sku", "skuid",
                            "item", "itemid", "itemcode", "itemno", "article", "articleno")),
    "units_sold": SheetColumn("Units sold", True, "Units sold in the week (returns not subtracted).",
                              ("unitssold", "units", "qtysold", "quantitysold", "qty", "quantity", "sold", "salesunits", "salesqty", "unitsales", "grossunits", "grossqty")),
    "units_returned": SheetColumn("Units returned", False, "Units customers brought back in the week.",
                                  ("unitsreturned", "returns", "returned", "returnqty", "returnunits", "qtyreturned",
                                   "returnsunits")),
    "stock_start": SheetColumn("Stock at start of week", False, "Units in the store when the week began.",
                               ("stockatstartofweek", "stockstart", "startstock", "openingstock", "openstock",
                                "openinginventory", "stockonhand", "soh", "onhand", "stock", "inventory",
                                "stockmonday", "beginningstock", "beginninginventory")),
    "stock_end": SheetColumn("Stock at end of week", False, "Units left when the week ended.",
                             ("stockatendofweek", "stockend", "endstock", "closingstock", "closestock",
                              "closinginventory", "endinginventory", "endingstock", "stocksunday")),
    "units_on_promotion": SheetColumn("Units sold on promotion", False,
                                      "Of the units sold, how many were sold at a discount.",
                                      ("unitssoldonpromotion", "promounits", "promotionunits", "promoqty",
                                       "discountedunits", "unitsondiscount", "markdownunits", "saleunits",
                                       "promo", "promotion", "unitsonpromotion")),
}

# Headers about money are never matched to a units or stock column.
MONEY_WORDS = ("value", "amount", "revenue", "price", "cost", "margin", "turnover", "eur", "usd", "gbp", "inr")
MISSING_TEXT = {"", "-", "--", "na", "n/a", "nan", "none", "null", "nil", "?", "#n/a", "unknown", "missing"}
TOTAL_TEXT = re.compile(r"^\s*(grand\s*)?(sub\s*)?totals?\s*$", re.I)


@dataclass
class Issue:
    level: str        # 'error' (nothing can be used) | 'warning' (rows dropped or changed) | 'info'
    message: str
    rows: int = 0     # rows affected


@dataclass
class CleanResult:
    weekly: pd.DataFrame                          # one row per store/group/week_start
    issues: list[Issue] = field(default_factory=list)
    column_map: dict[str, str] = field(default_factory=dict)   # canonical -> column found in the file
    rows_read: int = 0

    @property
    def ok(self) -> bool:
        return not any(i.level == "error" for i in self.issues) and len(self.weekly) > 0


def _norm(text) -> str:
    return re.sub(r"[^0-9a-z]", "", str(text).lower())


def _sheet_row_numbers(index: pd.Index, header_row: int) -> str:
    """Spreadsheet row numbers (header = its own row, first data row below it) for a short example list."""
    nums = [int(i) + header_row + 2 for i in index[:5]]
    more = "" if len(index) <= 5 else f" and {len(index) - 5} more"
    return ", ".join(map(str, nums)) + more


# ------------------------------------------------------------------------------- reading the file
def read_sheet(data: bytes, filename: str) -> pd.DataFrame:
    """Read every cell as text (no type guessing), so cleaning sees the file as typed."""
    suffix = Path(filename).suffix.lower()
    if suffix in (".xlsx", ".xlsm", ".xls"):
        try:
            return pd.read_excel(io.BytesIO(data), header=None, dtype=object)
        except ImportError as e:
            raise ValueError("Reading Excel files needs the 'openpyxl' package (pip install openpyxl), "
                             "or save the sheet as CSV.") from e
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    sep = max([",", ";", "\t", "|"], key=lambda s: text[:5000].count(s))
    return pd.read_csv(io.StringIO(text), header=None, dtype=str, sep=sep, skip_blank_lines=False,
                       keep_default_na=False, engine="python")


def match_columns(header: list) -> dict[str, int]:
    """canonical name -> column position. An exact alias match wins over a partial one."""
    names = [_norm(h) for h in header]
    found: dict[str, int] = {}
    for key, col in COLUMNS.items():
        exact = [i for i, n in enumerate(names) if n in col.aliases and i not in found.values()]
        if exact:
            found[key] = exact[0]
    for key, col in COLUMNS.items():  # second pass: headers that contain an alias, e.g. "units sold (qty)"
        if key in found:
            continue
        loose = [i for i, n in enumerate(names) if i not in found.values() and n
                 and not any(w in n for w in MONEY_WORDS) and any(len(a) >= 5 and a in n for a in col.aliases)]
        if loose:
            found[key] = loose[0]
    return found


def find_header(raw: pd.DataFrame, max_scan: int = 15) -> tuple[int, dict[str, int]]:
    """The header is the first row (of the top `max_scan`) that names the most template columns,
    so title rows above the table are skipped."""
    best = (0, {})
    for r in range(min(max_scan, len(raw))):
        m = match_columns(raw.iloc[r].tolist())
        required = sum(k in m for k, c in COLUMNS.items() if c.required)
        if required > sum(k in best[1] for k, c in COLUMNS.items() if c.required):
            best = (r, m)
        if required == 4:
            break
    return best


# ------------------------------------------------------------------------------- value cleaning
def parse_numbers(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Returns (numbers, bad) where bad marks cells that had text that is not a number."""
    txt = s.astype(str).str.strip()
    missing = s.isna() | txt.str.lower().isin(MISSING_TEXT)
    t = txt.str.replace(r"[\s,']", "", regex=True).str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    num = pd.to_numeric(t.where(~missing), errors="coerce")
    return num, num.isna() & ~missing


_ISO_WEEK = re.compile(r"^\s*(\d{4})\s*-?\s*w\s*(\d{1,2})\s*$", re.I)
_WEEK_YEAR = re.compile(r"^\s*w(?:ee)?k?\s*(\d{1,2})\s*[-/ ,]+\s*(\d{4})\s*$", re.I)


def parse_weeks(s: pd.Series) -> tuple[pd.Series, str]:
    """Monday of each week. Returns (dates, note on how ambiguous dates were read)."""
    txt = s.astype(str).str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    # values that are already dates (Excel cells)
    is_dt = s.map(lambda v: isinstance(v, (pd.Timestamp, np.datetime64)) or hasattr(v, "isoformat"))
    out[is_dt] = pd.to_datetime(s[is_dt], errors="coerce")
    rest = ~is_dt & ~txt.str.lower().isin(MISSING_TEXT) & s.notna()
    # ISO weeks: 2026-W17, 2026W17 ; and "Week 17 2026", "wk17-2026"
    for rx, yi, wi in ((_ISO_WEEK, 1, 2), (_WEEK_YEAR, 2, 1)):
        m = txt[rest].str.extract(rx)
        ok = m[0].notna()
        if ok.any():
            idx = m.index[ok]
            out[idx] = pd.to_datetime(m.loc[ok, yi - 1] + "-W" + m.loc[ok, wi - 1].str.zfill(2) + "-1",
                                      format="%G-W%V-%u", errors="coerce")
            rest[idx] = False
    note = ""
    if rest.any():
        vals = txt[rest]
        serial = pd.to_numeric(vals, errors="coerce")
        is_serial = serial.between(20000, 80000)  # Excel day numbers saved as text
        out[vals.index[is_serial]] = pd.Timestamp("1899-12-30") + pd.to_timedelta(serial[is_serial], unit="D")
        vals = vals[~is_serial]
        if len(vals):
            iso = vals.str.match(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}")
            out[vals.index[iso]] = pd.to_datetime(vals[iso].str.replace(r"[/.]", "-", regex=True).str[:10],
                                                  format="%Y-%m-%d", errors="coerce")
            other = vals[~iso]
            if len(other):
                dmy = pd.to_datetime(other, dayfirst=True, errors="coerce", format="mixed")
                mdy = pd.to_datetime(other, dayfirst=False, errors="coerce", format="mixed")
                # prefer the reading that parses more cells, then the one spanning fewer weeks
                def score(d):
                    return (d.notna().sum(), -d.dt.to_period("W").nunique())
                use_dmy = score(dmy) >= score(mdy)
                out[other.index] = dmy if use_dmy else mdy
                both = dmy.notna() & mdy.notna() & (dmy != mdy)
                if both.any():
                    ex = other[both].iloc[0]
                    note = (f"Dates that could be read either way (such as {ex}) were read as " +
                            ("day/month/year" if use_dmy else "month/day/year") + ", which fits the other dates "
                            "in the file best.")
    out = out.dt.normalize()
    return out - pd.to_timedelta(out.dt.dayofweek, unit="D"), note


def normalize_code(s: pd.Series, known: list[str]) -> pd.Series:
    """Map typed codes to known codes: exact (case-insensitive) or by their number."""
    by_upper = {k.upper(): k for k in known}
    by_number = {}
    for k in known:
        digits = re.findall(r"\d+", k)
        if len(digits) == 1:
            by_number[digits[0].lstrip("0") or "0"] = k

    def one(v) -> str | None:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        t = str(v).strip()
        if t.endswith(".0") and t[:-2].isdigit():  # 1006.0 from a numeric Excel cell
            t = t[:-2]
        if t.upper() in by_upper:
            return by_upper[t.upper()]
        compact = re.sub(r"[\s_]+", "-", t.upper())
        if compact in by_upper:
            return by_upper[compact]
        digits = re.findall(r"\d+", t)
        if len(digits) == 1:
            return by_number.get(digits[0].lstrip("0") or "0")
        return None

    return s.map(one)


# ------------------------------------------------------------------------------- main entry point
def clean_weekly_sheet(raw: pd.DataFrame, stores: list[str], products: list[str], last_loaded_week: pd.Timestamp,
                       today: pd.Timestamp | None = None) -> CleanResult:
    """raw: the sheet as read by `read_sheet` (no header applied). `stores`/`products` are the codes the
    models know; `last_loaded_week` is the Monday of the latest week already in the data."""
    today = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today)
    issues: list[Issue] = []
    empty = pd.DataFrame(columns=KEYS + VALUE_COLS)
    if raw.empty:
        return CleanResult(empty, [Issue("error", "The file is empty.")])

    header_row, cmap = find_header(raw)
    missing = [COLUMNS[k].name for k, c in COLUMNS.items() if c.required and k not in cmap]
    header = raw.iloc[header_row].tolist()
    column_map = {k: str(header[i]) for k, i in cmap.items()}
    if missing:
        return CleanResult(empty, [Issue(
            "error", f"Could not find the column(s) {', '.join(missing)}. Columns in the file: "
                     f"{', '.join(str(h) for h in header if str(h).strip() and str(h) != 'nan') or 'none'}. "
                     f"Rename them to match the template, or download the template and copy the data into it.")],
            column_map)

    renamed = {k: str(header[i]).strip() for k, i in cmap.items() if _norm(header[i]) != _norm(COLUMNS[k].name)}
    if renamed:
        issues.append(Issue("info", "Columns matched: " + "; ".join(
            f"'{v}' → {COLUMNS[k].name}" for k, v in renamed.items()) + "."))
    ignored = [str(h).strip() for i, h in enumerate(header) if i not in cmap.values()
               and str(h).strip() and str(h) != "nan"]
    if ignored:
        issues.append(Issue("info", f"Columns not needed and ignored: {', '.join(ignored)}."))

    body = raw.iloc[header_row + 1:].reset_index(drop=True)
    df = pd.DataFrame({k: body.iloc[:, i] for k, i in cmap.items()})
    blank = df.apply(lambda c: c.isna() | c.astype(str).str.strip().str.lower().isin(MISSING_TEXT)).all(axis=1)
    totals = df[["store", "product", "week"]].astype(str).apply(lambda c: c.str.match(TOTAL_TEXT)).any(axis=1)
    if totals.any():
        issues.append(Issue("info", f"Skipped {int(totals.sum())} total/subtotal row(s).", int(totals.sum())))
    df = df[~blank & ~totals]
    rows_read = len(df)
    if df.empty:
        return CleanResult(empty, [Issue("error", "The file has a header but no data rows.")] + issues, column_map,
                           rows_read)

    def drop(mask: pd.Series, message: str) -> None:
        nonlocal df
        n = int(mask.sum())
        if n:
            issues.append(Issue("warning", f"{message} ({n} row{'s' if n > 1 else ''}, sheet rows "
                                           f"{_sheet_row_numbers(df.index[mask], header_row)}).", n))
            df = df[~mask]

    # week ----------------------------------------------------------------------------------
    weeks, note = parse_weeks(df["week"])
    df = df.assign(week_start=weeks)
    if note:
        issues.append(Issue("info", note))
    drop(df.week_start.isna(), "Dropped rows whose week could not be read as a date")

    # store / product -----------------------------------------------------------------------
    df = df.assign(store_code=normalize_code(df["store"], stores),
                   group=normalize_code(df["product"], products))
    no_store = df["store"].isna() | df["store"].astype(str).str.strip().str.lower().isin(MISSING_TEXT)
    drop(no_store, "Dropped rows with no store")
    unknown = df.store_code.isna()
    if unknown.any():
        names = sorted(df.loc[unknown, "store"].astype(str).str.strip().unique())
        drop(unknown, f"Dropped rows for store(s) the model does not know: {', '.join(names[:8])}"
                      f"{' …' if len(names) > 8 else ''}. Only the {len(stores)} stores in the original data can be "
                      f"planned (new stores need the AI agent to be retrained)")
    no_prod = df["product"].isna() | df["product"].astype(str).str.strip().str.lower().isin(MISSING_TEXT)
    drop(no_prod, "Dropped rows with no product")
    unknown = df.group.isna()
    if unknown.any():
        names = sorted(df.loc[unknown, "product"].astype(str).str.strip().unique())
        drop(unknown, f"Dropped rows for product(s) outside the {len(products)} products the forecasts cover: "
                      f"{', '.join(names[:8])}{' …' if len(names) > 8 else ''}")
    df = df.assign(store=df.store_code).drop(columns=["store_code", "product"])

    # dates against the data already loaded ----------------------------------------------------
    drop(df.week_start <= last_loaded_week,
         f"Dropped rows for weeks already in the data (the data runs up to the week of "
         f"{last_loaded_week:%d %b %Y})")
    drop(df.week_start > today, "Dropped rows for weeks that have not started yet")

    # numbers ------------------------------------------------------------------------------------
    for key in ("units_sold", "units_returned", "stock_start", "stock_end", "units_on_promotion"):
        if key not in df:
            df[key] = np.nan
            continue
        num, bad = parse_numbers(df[key])
        if bad.any():
            examples = ", ".join(f"'{v}'" for v in df.loc[bad, key].astype(str).unique()[:3])
            issues.append(Issue("warning", f"{COLUMNS[key].name}: {int(bad.sum())} value(s) were not numbers "
                                           f"({examples}) and were treated as blank.", int(bad.sum())))
        df[key] = num
    drop(df.units_sold.isna(), "Dropped rows with no units sold")

    neg = df.units_sold < 0
    if neg.any():
        issues.append(Issue("warning", f"{int(neg.sum())} row(s) had negative units sold; these were counted as "
                                       f"returns.", int(neg.sum())))
        df.loc[neg, "units_returned"] = df.loc[neg, "units_returned"].fillna(0) - df.loc[neg, "units_sold"]
        df.loc[neg, "units_sold"] = 0.0
    for key in ("units_returned", "units_on_promotion"):
        n = int((df[key] < 0).sum())
        if n:
            issues.append(Issue("warning", f"{COLUMNS[key].name}: {n} negative value(s) were made positive.", n))
            df[key] = df[key].abs()
    for key in ("stock_start", "stock_end"):
        n = int((df[key] < 0).sum())
        if n:
            issues.append(Issue("warning", f"{COLUMNS[key].name}: {n} negative value(s) set to 0 (stock on the "
                                           f"shelf cannot be below zero).", n))
            df[key] = df[key].clip(lower=0)
    frac = pd.concat([(df[k].dropna() % 1 != 0) for k in
                      ("units_sold", "units_returned", "stock_start", "stock_end", "units_on_promotion")]).sum()
    if frac:
        issues.append(Issue("warning", f"{int(frac)} value(s) were not whole units and were rounded.", int(frac)))
    for key in ("units_sold", "units_returned", "stock_start", "stock_end", "units_on_promotion"):
        df[key] = df[key].round()
    over = df.units_on_promotion > df.units_sold
    if over.any():
        issues.append(Issue("warning", f"{int(over.sum())} row(s) had more units on promotion than units sold; "
                                       f"capped at units sold.", int(over.sum())))
        df.loc[over, "units_on_promotion"] = df.loc[over, "units_sold"]

    # duplicates ---------------------------------------------------------------------------------
    cols = ["store", "group", "week_start", "units_sold", "units_returned", "stock_start", "stock_end",
            "units_on_promotion"]
    dup = df.duplicated(cols)
    drop(dup, "Dropped exact duplicate rows")
    multi = df.duplicated(KEYS, keep=False)
    if multi.any():
        n_keys = df[multi].drop_duplicates(KEYS).shape[0]
        conflict = df[multi].groupby(KEYS)[["stock_start", "stock_end"]].nunique().gt(1).any(axis=1).sum()
        msg = (f"{int(multi.sum())} rows share the same store, product and week ({n_keys} "
               f"combination{'s' if n_keys > 1 else ''}); their units were added together")
        msg += (f", and where their stock figures differ ({int(conflict)}) the first one was kept."
                if conflict else ".")
        issues.append(Issue("warning", msg, int(multi.sum())))
    weekly = df.groupby(KEYS, as_index=False).agg(
        gross_qty=("units_sold", "sum"), return_qty=("units_returned", lambda x: x.sum(min_count=1)),
        promo_qty=("units_on_promotion", lambda x: x.sum(min_count=1)),
        stock_start=("stock_start", "first"), stock_end=("stock_end", "first"))
    weekly[VALUE_COLS] = weekly[VALUE_COLS].astype(float)
    weekly[["return_qty", "promo_qty"]] = weekly[["return_qty", "promo_qty"]].fillna(0.0)

    if weekly.empty:
        issues.append(Issue("error", "No usable rows are left after cleaning (see the messages above)."))
        return CleanResult(weekly, issues, column_map, rows_read)

    if weekly[["stock_start", "stock_end"]].isna().all(axis=None):
        issues.append(Issue("warning", "The sheet has no stock figures. Stock will be estimated from the last known "
                                       "stock minus units sold, assuming no deliveries, which makes the stock "
                                       "risk less reliable."))
    return CleanResult(weekly.sort_values(KEYS).reset_index(drop=True), issues, column_map, rows_read)


def template(stores: list[str], products: list[str], week: pd.Timestamp, n_rows: int = 3) -> pd.DataFrame:
    """A small example sheet with the template's column names (values are illustrative only)."""
    rows = [{"Week": f"{week:%Y-%m-%d}", "Store": stores[i % len(stores)], "Product": products[i % len(products)],
             "Units sold": [4, 0, 2][i % 3], "Units returned": [0, 0, 1][i % 3],
             "Stock at start of week": [10, 3, 6][i % 3], "Stock at end of week": [6, 3, 5][i % 3],
             "Units sold on promotion": [1, 0, 0][i % 3]} for i in range(n_rows)]
    return pd.DataFrame(rows, columns=[c.name for c in COLUMNS.values()])
