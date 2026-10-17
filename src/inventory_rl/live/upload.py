"""Read and clean a retailer's sales-and-stock sheet (CSV, text or Excel), in whatever layout it was exported.

The result is one row per store, product and week. The sheet does not have to follow the template:
  - columns: names are matched loosely (case, spaces and common alternative names); a column that is
    not recognised can be matched by hand (`mapping`), and title rows above the table are skipped
  - layout: one row per store/product/week (long), or with the weeks, products or stores running across
    the columns (wide: each cell is then read as units sold)
  - a file with no store or week column can be given one store / one week for all its rows; an Excel
    sheet named after a store (e.g. '1006' or 'Store 1006') is read as that store
  - workbooks: every sheet holding a sales table is read; other sheets are skipped
  - daily or transaction rows are added up into weeks (stock at the start of the week comes from the
    earliest date in the week, stock at the end from the latest)
Values are cleaned:
  - week: any date (several date formats) or an ISO week such as 2026-W17; every date is moved to the
    Monday of its week
  - store / product codes: '1006', 'str 1006', 'Store 1006' all become STR-1006 (same for PROD-)
  - numbers: thousands separators, spaces, '(3)' and text such as 'n/a' or '-' are handled
  - negative units sold are counted as returns; negative stock is set to 0
  - exact duplicate rows are dropped; split rows for the same store/product/date are added up
Rows that cannot be used (unknown store or product, missing week, week already in the data, week
in the future) are dropped. Every change is reported as an `Issue` in plain language.

The result has one row per (store, group, week_start) in the units of the weekly panel; stock that
was not given stays missing (NaN) and is filled from neighbouring weeks later (see update.py).
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["store", "group", "week_start"]
VALUE_COLS = ["gross_qty", "return_qty", "promo_qty", "stock_start", "stock_end"]
# sheet columns that hold quantities (the others say which store, product and week a row is for)
QUANTITIES = ("units_sold", "units_returned", "stock_start", "stock_end", "units_on_promotion")
EXCEL_SUFFIXES = (".xlsx", ".xlsm", ".xls")


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
                         "weekending", "weekend", "weekenddate", "date", "salesdate", "period", "isoweek",
                         "day", "transactiondate", "orderdate", "invoicedate", "businessdate")),
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
ACROSS_NAMES = {"week": "weeks", "store": "stores", "product": "products"}

# Headers about money are never matched to a units or stock column.
MONEY_WORDS = ("value", "amount", "revenue", "price", "cost", "margin", "turnover", "eur", "usd", "gbp", "inr")
# short aliases that may also match inside a longer header ("Txn Date"); other partial matches need 5+ letters
SHORT_PARTIAL = {"date"}
MISSING_TEXT = {"", "-", "--", "na", "n/a", "nan", "none", "null", "nil", "?", "#n/a", "unknown", "missing"}
TOTAL_TEXT = re.compile(r"^\s*(grand\s*)?(sub\s*)?totals?\s*$", re.I)
# a header cell or sheet name that is just a code: '1006', 'STR-1006', 'Store 1006', 'PROD_100358', '100358.0'
CODE_TEXT = re.compile(r"\s*[A-Za-z]{0,8}[\s\-_#:]*\d+(\.0)?\s*")


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
    columns_in_file: list[str] = field(default_factory=list)   # header names, for matching columns by hand
    missing: list[str] = field(default_factory=list)           # required columns (canonical) not found
    sheets_used: list[str] = field(default_factory=list)       # Excel sheets the rows came from

    @property
    def ok(self) -> bool:
        return not any(i.level == "error" for i in self.issues) and len(self.weekly) > 0


def _norm(text) -> str:
    return re.sub(r"[^0-9a-z]", "", str(text).lower())


def cell_text(v) -> str:
    """A cell as the user sees it: '' for blanks, dates without a time, 100358.0 as 100358."""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    if hasattr(v, "hour") and hasattr(v, "strftime"):
        return v.strftime("%Y-%m-%d" if (v.hour, v.minute, v.second) == (0, 0, 0) else "%Y-%m-%d %H:%M")
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).strip()


def _is_blank(s: pd.Series) -> pd.Series:
    return s.isna() | s.astype(str).str.strip().str.lower().isin(MISSING_TEXT)


def _row_list(labels: pd.Series) -> str:
    """Spreadsheet row numbers for a short example list."""
    more = "" if len(labels) <= 5 else f" and {len(labels) - 5} more"
    return ", ".join(labels.iloc[:5]) + more


# ------------------------------------------------------------------------------- reading the file
def read_sheet(data: bytes, filename: str) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """Read every cell as typed (no type guessing). A CSV or text file gives one table; an Excel
    workbook gives {sheet name: table} for all its sheets."""
    suffix = Path(filename).suffix.lower()
    if suffix in EXCEL_SUFFIXES:
        try:
            return pd.read_excel(io.BytesIO(data), header=None, dtype=object, sheet_name=None)
        except ImportError as e:
            package = "xlrd" if suffix == ".xls" else "openpyxl"
            raise ValueError(f"Reading {suffix} files needs the '{package}' package (pip install {package}), "
                             "or save the sheet as CSV.") from e
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    sep = max([",", ";", "\t", "|"], key=lambda s: text[:5000].count(s))
    # rows may have different lengths (a title line above the table, a ragged export): pad them
    rows = list(csv.reader(io.StringIO(text), delimiter=sep))
    width = max((len(r) for r in rows), default=0)
    return pd.DataFrame([r + [""] * (width - len(r)) for r in rows], dtype=str)


def match_columns(header: list, mapping: dict[str, str | None] | None = None) -> dict[str, int]:
    """canonical name -> column position. A column chosen by hand (`mapping`: canonical -> header text,
    or None for 'not in the file') wins, then an exact alias match, then a partial one."""
    texts = [cell_text(h) for h in header]
    names = [_norm(t) for t in texts]
    found: dict[str, int] = {}
    settled = set()
    for key, wanted in (mapping or {}).items():
        if wanted is None:
            settled.add(key)
        elif wanted in texts:
            found[key] = texts.index(wanted)
            settled.add(key)
    taken = set(found.values())
    for key, col in COLUMNS.items():
        if key in settled:
            continue
        exact = [i for i, n in enumerate(names) if n in col.aliases and i not in taken]
        if exact:
            found[key] = exact[0]
            taken.add(exact[0])
    for key, col in COLUMNS.items():  # second pass: headers that contain an alias, e.g. "units sold (qty)"
        if key in found or key in settled:
            continue
        loose = [i for i, n in enumerate(names) if i not in taken and n
                 and not any(w in n for w in MONEY_WORDS)
                 and any((len(a) >= 5 or a in SHORT_PARTIAL) and a in n for a in col.aliases)]
        if loose:
            found[key] = loose[0]
            taken.add(loose[0])
    return found


def _could_be_week(v) -> bool:
    if hasattr(v, "isoformat"):
        return True
    t = cell_text(v)
    if not re.search(r"\d", t):
        return False
    if re.fullmatch(r"\d+(\.\d+)?", t):  # a bare number is only a date as an Excel day number
        return 20000 <= float(t) <= 80000
    return True


@dataclass
class Layout:
    header_row: int
    header: list
    cmap: dict[str, int]                 # canonical -> position of an ordinary (long) column
    across: str | None = None            # 'week' | 'store' | 'product' when it runs across the columns
    across_cols: list[int] = field(default_factory=list)

    def missing(self, fixed) -> list[str]:
        have = set(self.cmap) | set(fixed) | ({self.across, "units_sold"} if self.across else set())
        return [k for k, c in COLUMNS.items() if c.required and k not in have]


def read_header(header: list, row: int, stores: list[str], products: list[str],
                mapping: dict[str, str | None] | None = None, fixed=()) -> Layout:
    """Which column is which, and whether weeks, products or stores run across the columns (a wide
    sheet: then header cells are dates or codes, and each cell below holds units sold). A row with no named
    column needs at least two such cells, so a data row is not taken for a wide header."""
    cmap = match_columns(header, mapping)
    free = [i for i, h in enumerate(header) if i not in cmap.values() and cell_text(h)]
    need = 1 if cmap else 2
    for dim, known in (("product", products), ("store", stores)):
        if dim in cmap or dim in fixed:
            continue
        cand = [i for i in free if CODE_TEXT.fullmatch(cell_text(header[i]))]
        codes = normalize_code(pd.Series([header[i] for i in cand], index=cand, dtype=object), known)
        if codes.notna().sum() >= need:
            return Layout(row, header, cmap, dim, codes.dropna().index.tolist())
    if "week" not in cmap and "week" not in fixed:
        cand = [i for i in free if _could_be_week(header[i])]
        if cand:
            weeks, _ = parse_weeks(pd.Series([header[i] for i in cand], index=cand, dtype=object))
            if weeks.notna().sum() >= need:
                return Layout(row, header, cmap, "week", weeks.dropna().index.tolist())
    return Layout(row, header, cmap)


def find_header(raw: pd.DataFrame, stores: list[str], products: list[str],
                mapping: dict[str, str | None] | None = None, fixed=(), max_scan: int = 15) -> Layout:
    """The header is the first row (of the top `max_scan`) that accounts for the most required columns,
    so title rows above the table are skipped. When no row names any, the widest row is taken as the header."""
    scan = range(min(max_scan, len(raw)))
    wide = max(scan, key=lambda r: sum(cell_text(v) != "" for v in raw.iloc[r]), default=0)
    best = read_header(raw.iloc[wide].tolist() if len(raw) else [], wide, stores, products, mapping, fixed)
    best_missing = len(best.missing(fixed))
    for r in scan:
        lay = read_header(raw.iloc[r].tolist(), r, stores, products, mapping, fixed)
        n = len(lay.missing(fixed))
        if n < best_missing:
            best, best_missing = lay, n
        if n == 0:
            break
    return best


def _table_rows(raw: pd.DataFrame, lay: Layout, fixed: dict, sheet: str) -> tuple[pd.DataFrame, int, list[str]]:
    """The sheet's data rows as one row per entry with canonical column names (wide sheets are turned into
    rows), blank and total rows removed. Returns (rows, number of total rows skipped, ignored quantity columns)."""
    body = raw.iloc[lay.header_row + 1:]
    ignored = []
    if lay.across:
        # each cell under a week/product/store column is units sold; per-row quantity columns cannot be split
        ignored = [cell_text(lay.header[i]) for k, i in lay.cmap.items() if k in QUANTITIES]
        ids = {k: body.iloc[:, i] for k, i in lay.cmap.items() if k not in QUANTITIES}
        parts = []
        for c in lay.across_cols:
            part = pd.DataFrame(ids, index=body.index)
            part[lay.across] = [lay.header[c]] * len(body)
            part["units_sold"] = body.iloc[:, c]
            parts.append(part[~_is_blank(part.units_sold)])
        df = pd.concat(parts) if parts else pd.DataFrame(columns=list(ids) + [lay.across, "units_sold"])
    else:
        df = pd.DataFrame({k: body.iloc[:, i] for k, i in lay.cmap.items()}, index=body.index)
    df = df[~df.apply(_is_blank).all(axis=1)] if len(df.columns) else df
    ids = [k for k in ("store", "product", "week") if k in df]
    totals = df[ids].astype(str).apply(lambda c: c.str.match(TOTAL_TEXT)).any(axis=1) if ids else df.index.isna()
    df = df[~totals].copy()
    for k, v in fixed.items():
        df[k] = [v] * len(df)
    where = f" of '{sheet}'" if sheet else ""
    df["_row"] = [f"{i + 1}{where}" for i in df.index]
    return df, int(totals.sum()), ignored


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


def parse_dates(s: pd.Series) -> tuple[pd.Series, str]:
    """The day of each cell (a week such as 2026-W17 gives its Monday). Returns (dates, note on how
    ambiguous dates were read)."""
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
    return out.dt.normalize(), note


def parse_weeks(s: pd.Series) -> tuple[pd.Series, str]:
    """Monday of each week. Returns (dates, note on how ambiguous dates were read)."""
    out, note = parse_dates(s)
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
def clean_weekly_sheet(raw: pd.DataFrame | dict[str, pd.DataFrame], stores: list[str], products: list[str],
                       last_loaded_week: pd.Timestamp, today: pd.Timestamp | None = None, *,
                       mapping: dict[str, str | None] | None = None, store: str | None = None,
                       week: pd.Timestamp | None = None) -> CleanResult:
    """raw: the file as read by `read_sheet` (no header applied; a dict for an Excel workbook). `stores`/
    `products` are the codes the models know; `last_loaded_week` is the Monday of the latest week already in
    the data. `mapping` (canonical column -> header text, or None for 'not in the file') overrides the
    automatic column matching; `store` / `week` apply to every row of a file that has no such column."""
    today = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today)
    issues: list[Issue] = []
    empty = pd.DataFrame(columns=KEYS + VALUE_COLS)
    sheets = {n: s for n, s in (raw if isinstance(raw, dict) else {"": raw}).items()
              if not s.empty and not s.map(lambda v: cell_text(v) == "").all(axis=None)}
    if not sheets:
        return CleanResult(empty, [Issue("error", "The file is empty.")])
    named = len(sheets) > 1
    given = {k: v for k, v in (("store", store), ("week", week)) if v is not None}

    # which column is which, sheet by sheet ------------------------------------------------------
    layouts: dict[str, tuple[Layout, dict]] = {}
    for name, s in sheets.items():
        lay = find_header(s, stores, products, mapping, given)
        fixed = dict(given)
        if "store" in lay.missing(fixed) and name and CODE_TEXT.fullmatch(name):  # a sheet named after a store
            code = normalize_code(pd.Series([name]), stores).iloc[0]
            if code is not None:
                fixed["store"] = code
        layouts[name] = (lay, fixed)
    usable = [n for n, (lay, fixed) in layouts.items() if not lay.missing(fixed)]

    def headers(names) -> list[str]:
        seen = dict.fromkeys(t for n in names for t in map(cell_text, layouts[n][0].header) if t and t != "nan")
        return list(seen)

    if not usable:
        name = min(layouts, key=lambda n: len(layouts[n][0].missing(layouts[n][1])))
        lay, fixed = layouts[name]
        missing = lay.missing(fixed)
        found = headers([name])
        where = f" (sheet '{name}')" if named else ""
        return CleanResult(empty, [Issue(
            "error", f"Could not find the column(s) {', '.join(COLUMNS[k].name for k in missing)}{where}. Columns "
                     f"in the file: {', '.join(found) or 'none'}. Choose which column is which"
                     + (", or say which store the whole file is for" if "store" in missing else "")
                     + (", or say which week the whole file is for" if "week" in missing else "")
                     + "; or rename the columns to match the template.")],
            {k: cell_text(lay.header[i]) for k, i in lay.cmap.items()}, columns_in_file=found, missing=missing,
            sheets_used=[name] if name else [])
    if named:
        skipped = [n for n in sheets if n not in usable]
        issues.append(Issue("info", f"Read {len(usable)} sheet(s): {', '.join(usable)}."
                            + (f" Sheets with no sales table were skipped: {', '.join(skipped)}." if skipped else "")))

    parts = []
    notes: dict[tuple[str, str, int], list[str]] = {}   # (level, message, rows) -> sheets it applies to
    from_name = {}                                      # sheet -> store read from its name

    def note(level: str, message: str, name: str, rows: int = 0) -> None:
        notes.setdefault((level, message, rows), []).append(name)

    for name in usable:
        lay, fixed = layouts[name]
        renamed = {k: cell_text(lay.header[i]) for k, i in lay.cmap.items()
                   if _norm(lay.header[i]) != _norm(COLUMNS[k].name)}
        if renamed:
            note("info", "Columns matched: " + "; ".join(f"'{v}' → {COLUMNS[k].name}" for k, v in renamed.items())
                 + ".", name)
        rows, n_totals, not_split = _table_rows(sheets[name], lay, fixed, name if named else "")
        if lay.across:
            heads = [cell_text(lay.header[i]) for i in lay.across_cols]
            note("info", f"The {ACROSS_NAMES[lay.across]} run across the columns ({len(heads)} column(s): "
                         f"{heads[0]}{f' … {heads[-1]}' if len(heads) > 1 else ''}); each cell was read as units sold.",
                 name)
            if not_split:
                note("warning", f"Column(s) {', '.join(not_split)} could not be used: in a sheet where the "
                                f"{ACROSS_NAMES[lay.across]} run across the columns, only units sold can be read.", name)
        used = set(lay.cmap.values()) | set(lay.across_cols)
        ignored = [t for i, h in enumerate(lay.header) if i not in used and (t := cell_text(h)) and t != "nan"]
        if ignored:
            note("info", f"Columns not needed and ignored: {', '.join(ignored)}.", name)
        if "store" in fixed and "store" not in given:
            from_name[name] = fixed["store"]
        if n_totals:
            note("info", f"Skipped {n_totals} total/subtotal row(s).", name, n_totals)
        parts.append(rows)
    for (level, message, n), names in notes.items():   # one message for all the sheets it applies to
        pre = "" if not named else "All sheets: " if len(names) == len(usable) else             ("Sheet " if len(names) == 1 else "Sheets ") + ", ".join(f"'{x}'" for x in names) + ": "
        issues.append(Issue(level, pre + message, n * len(names)))
    if from_name:
        issues.append(Issue("info", f"Sheet{'s' if len(from_name) > 1 else ''} with no store column, read as the "
                                    "store in the sheet name: " + ", ".join(f"'{k}' → {v}" for k, v in from_name.items())
                                    + "."))
    if store is not None:
        issues.append(Issue("info", f"Every row was given to store {store}, as chosen for the whole file."))
    if week is not None:
        issues.append(Issue("info", f"Every row was given the week of {pd.Timestamp(week):%d %b %Y}, as chosen for "
                                    f"the whole file."))

    first = layouts[usable[0]][0]
    column_map = {k: cell_text(first.header[i]) for k, i in first.cmap.items()}
    df = pd.concat(parts, ignore_index=True)
    rows_read = len(df)
    meta = dict(column_map=column_map, rows_read=rows_read, columns_in_file=headers(usable),
                sheets_used=[n for n in usable if n])
    if df.empty:
        return CleanResult(empty, [Issue("error", "The file has a header but no data rows.")] + issues, **meta)

    def drop(mask: pd.Series, message: str) -> None:
        nonlocal df
        n = int(mask.sum())
        if n:
            issues.append(Issue("warning", f"{message} ({n} row{'s' if n > 1 else ''}, sheet rows "
                                           f"{_row_list(df.loc[mask, '_row'])}).", n))
            df = df[~mask]

    # week ----------------------------------------------------------------------------------
    dates, note = parse_dates(df["week"])
    df = df.assign(date=dates, week_start=dates - pd.to_timedelta(dates.dt.dayofweek, unit="D"))
    if note:
        issues.append(Issue("info", note))
    drop(df.week_start.isna(), "Dropped rows whose week could not be read as a date")

    # store / product -----------------------------------------------------------------------
    df = df.assign(store_code=normalize_code(df["store"], stores),
                   group=normalize_code(df["product"], products))
    drop(_is_blank(df["store"]), "Dropped rows with no store")
    unknown = df.store_code.isna()
    if unknown.any():
        names = sorted(df.loc[unknown, "store"].astype(str).str.strip().unique())
        drop(unknown, f"Dropped rows for store(s) the model does not know: {', '.join(names[:8])}"
                      f"{' …' if len(names) > 8 else ''}. Only the {len(stores)} stores in the original data can be "
                      f"planned (new stores need the AI agent to be retrained)")
    drop(_is_blank(df["product"]), "Dropped rows with no product")
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
    drop(df.date > today, "Dropped rows dated in the future")

    # numbers ------------------------------------------------------------------------------------
    for key in QUANTITIES:
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
    frac = pd.concat([(df[k].dropna() % 1 != 0) for k in QUANTITIES]).sum()
    if frac:
        issues.append(Issue("warning", f"{int(frac)} value(s) were not whole units and were rounded.", int(frac)))
    for key in QUANTITIES:
        df[key] = df[key].round()
    over = df.units_on_promotion > df.units_sold
    if over.any():
        issues.append(Issue("warning", f"{int(over.sum())} row(s) had more units on promotion than units sold; "
                                       f"capped at units sold.", int(over.sum())))
        df.loc[over, "units_on_promotion"] = df.loc[over, "units_sold"]

    # duplicates, and several rows for one store/product/week -----------------------------------
    drop(df.duplicated(["store", "group", "date", *QUANTITIES]), "Dropped exact duplicate rows")
    multi = df.duplicated(KEYS + ["date"], keep=False)
    if multi.any():
        n_keys = df[multi].drop_duplicates(KEYS + ["date"]).shape[0]
        conflict = (df[multi].groupby(KEYS + ["date"])[["stock_start", "stock_end"]].nunique().gt(1)
                    .any(axis=1).sum())
        msg = (f"{int(multi.sum())} rows share the same store, product and date ({n_keys} "
               f"combination{'s' if n_keys > 1 else ''}); their units were added together")
        msg += (f", and where their stock figures differ ({int(conflict)}) the first start stock and the last end "
                f"stock were kept." if conflict else ".")
        issues.append(Issue("warning", msg, int(multi.sum())))
    if len(df) and df.groupby(["store", "week_start"]).date.nunique().median() > 1:  # daily or transaction rows
        issues.append(Issue("info", f"The file has rows for {df.date.nunique()} different dates over "
                                    f"{df.week_start.nunique()} week(s); they were added up into weekly totals. Stock "
                                    "at the start of a week is the earliest stock figure in that week, and stock at "
                                    "the end the latest."))
        days = df.groupby("week_start").date.agg(["nunique", "min", "max"])
        for w in sorted({days.index.min(), days.index.max()}):
            d = days.loc[w]
            if d["nunique"] < 7:
                issues.append(Issue("warning", f"The week of {w:%d %b %Y} has only {d['nunique']} day(s) of data "
                                    f"({d['min']:%a %d %b} to {d['max']:%a %d %b}). If part of the week is missing, "
                                    "its sales will look low; upload the full week when you can."))
        if df.stock_end.isna().all() and df.stock_start.notna().any():
            df["stock_end"] = df.stock_start
            issues.append(Issue("info", "The file has one stock figure per date, so stock at the end of a week is "
                                        "the latest figure in that week."))
    df = df.assign(_order=np.arange(len(df))).sort_values(["date", "_order"])
    weekly = df.groupby(KEYS, as_index=False).agg(
        gross_qty=("units_sold", "sum"), return_qty=("units_returned", lambda x: x.sum(min_count=1)),
        promo_qty=("units_on_promotion", lambda x: x.sum(min_count=1)),
        stock_start=("stock_start", "first"), stock_end=("stock_end", "last"))
    weekly[VALUE_COLS] = weekly[VALUE_COLS].astype(float)
    weekly[["return_qty", "promo_qty"]] = weekly[["return_qty", "promo_qty"]].fillna(0.0)

    if weekly.empty:
        issues.append(Issue("error", "No usable rows are left after cleaning (see the messages above)."))
        return CleanResult(weekly, issues, **meta)

    if weekly[["stock_start", "stock_end"]].isna().all(axis=None):
        issues.append(Issue("warning", "The sheet has no stock figures. Stock will be estimated from the last known "
                                       "stock minus units sold, assuming no deliveries, which makes the stock "
                                       "risk less reliable."))
    return CleanResult(weekly.sort_values(KEYS).reset_index(drop=True), issues, **meta)


def template(stores: list[str], products: list[str], week: pd.Timestamp, n_rows: int = 3) -> pd.DataFrame:
    """A small example sheet with the template's column names (values are illustrative only)."""
    rows = [{"Week": f"{week:%Y-%m-%d}", "Store": stores[i % len(stores)], "Product": products[i % len(products)],
             "Units sold": [4, 0, 2][i % 3], "Units returned": [0, 0, 1][i % 3],
             "Stock at start of week": [10, 3, 6][i % 3], "Stock at end of week": [6, 3, 5][i % 3],
             "Units sold on promotion": [1, 0, 0][i % 3]} for i in range(n_rows)]
    return pd.DataFrame(rows, columns=[c.name for c in COLUMNS.values()])
