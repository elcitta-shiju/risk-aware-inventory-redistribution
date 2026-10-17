"""Refresh forecasts, risk and transfer recommendations after new weekly data has been uploaded.

Nothing is re-tuned or retrained here: the forecast model uses the settings chosen by the original
pipeline (models/forecast_config.json), and transfers come from the already-trained AI agent and
the rule. The original study's files in data/processed and results/ are never changed; everything
built from uploaded data is written to data/live/.

Weekly table for uploaded weeks (same columns as data/processed/panel.parquet):
  - a store that has at least one row in a week is "reported": its products missing from the sheet
    sold 0 units that week. A store with no rows in a week has unknown sales (NaN), as do weeks
    missing from the uploads entirely.
  - stock at the start of week t = the sheet's start stock, else the end stock of week t-1.
    Stock at the end of week t = the sheet's end stock, else the start stock of week t+1, else an
    ESTIMATE: start stock - units sold + returns (assumes no deliveries). Opening stock that rests on
    an estimate is flagged in `stock_estimated`.
  - the week after the last uploaded week is the "next" week: the one to plan for.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from inventory_rl.environment.data import ProductTensors
from inventory_rl.environment.simulator import RedistributionSim, SimConfig
from inventory_rl.features.build import TARGET, build_features, feature_columns
from inventory_rl.forecasting.models import KEYS, LGBMSpec, baseline_forecasts
from inventory_rl.risk.assessment import assess
from inventory_rl.routing.network import build_network, transfer_cost
from inventory_rl.uncertainty.intervals import METHODS, add_intervals, integerize
from inventory_rl.utils.config import save_json
from inventory_rl.utils.paths import LIVE_DIR, MODELS_DIR, PROCESSED_DIR, UPLOADS_DIR

UPLOADED = UPLOADS_DIR / "weekly_data.parquet"
ORIGINALS = UPLOADS_DIR / "original_files"
LIVE_FILES = {name: LIVE_DIR / name for name in
              ("panel.parquet", "forecasts.parquet", "risk.parquet", "transfers.csv", "summary.json")}
RISK_COLS = ["store", "group", "week_start", "split", "y", "yhat", "lo", "hi", "opening_stock", "stock_missing",
             "stock_estimated", "p_stockout", "expected_shortage", "shortfall_upper", "excess_units",
             "surplus_point", "surplus_upper", "cover_weeks", "risk_class"]


# ------------------------------------------------------------------------------- stored uploads
def base_panel() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "panel.parquet")


def known_codes(panel: pd.DataFrame) -> tuple[list[str], list[str]]:
    return sorted(panel.store.unique()), sorted(panel.group.unique())


def load_uploads() -> pd.DataFrame | None:
    return pd.read_parquet(UPLOADED) if UPLOADED.exists() else None


def save_upload(weekly: pd.DataFrame, data: bytes, filename: str) -> list[pd.Timestamp]:
    """Keep the original file untouched and add the cleaned rows. Weeks that were uploaded before are
    replaced by the new file's version (so a corrected sheet can be re-uploaded). Returns replaced weeks."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    ORIGINALS.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w.\-]+", "_", Path(filename).name)[-100:] or "upload"
    (ORIGINALS / f"{stamp}_{safe}").write_bytes(data)

    new = weekly.assign(source_file=f"{stamp}_{safe}")
    old = load_uploads()
    replaced: list[pd.Timestamp] = []
    if old is not None:
        clash = old.week_start.isin(new.week_start.unique())
        replaced = sorted(pd.Timestamp(w) for w in old.loc[clash, "week_start"].unique())
        new = pd.concat([old[~clash], new], ignore_index=True)
    new.sort_values(KEYS).reset_index(drop=True).to_parquet(UPLOADED, index=False)
    return replaced


def review_upload(weekly: pd.DataFrame, base_last: pd.Timestamp, n_stores: int) -> list[str]:
    """Plain-language notes on how a cleaned sheet fits the data already added (shown before saving)."""
    def label(w) -> str:
        return f"week of {pd.Timestamp(w):%d %b %Y}"

    notes = []
    uploads = load_uploads()
    new_weeks = sorted(weekly.week_start.unique())
    before = set() if uploads is None else set(uploads.week_start)
    if again := [w for w in new_weeks if w in before]:
        notes.append("Already added before and will be replaced by this file: " + ", ".join(map(label, again)) + ".")
    after = sorted(before | set(new_weeks))
    gaps = [w for w in pd.date_range(base_last + pd.Timedelta(days=7), after[-1], freq="W-MON") if w not in after]
    if gaps:
        notes.append("No data for the " + ", ".join(map(label, gaps)) + ". These weeks will be treated as not "
                     "reported (sales unknown); upload them when you can for better forecasts.")
    per_week = weekly.groupby("week_start").store.nunique()
    if (short := per_week[per_week < n_stores]).any():
        notes.append("Stores reported: " + "; ".join(f"{label(k)}: {v} of {n_stores}" for k, v in short.items())
                     + ". Stores missing from a week are treated as not reported that week.")
    return notes


def added_weeks(uploads: pd.DataFrame) -> pd.DataFrame:
    """One row per uploaded week: stores, rows, units sold, share of rows with a start stock, source file."""
    return uploads.groupby("week_start").agg(
        stores=("store", "nunique"), rows=("store", "size"), units=("gross_qty", "sum"),
        stock_given=("stock_start", lambda x: f"{100 * x.notna().mean():.0f}%"),
        file=("source_file", "last")).reset_index()


def clear_uploads() -> None:
    """Remove all uploaded data and everything built from it (the original study is untouched)."""
    for d in (UPLOADS_DIR, LIVE_DIR):
        if d.exists():
            for p in d.iterdir():
                if p.name == ".gitkeep":
                    continue
                shutil.rmtree(p) if p.is_dir() else p.unlink()


def live_ready() -> bool:
    return all(p.exists() for p in LIVE_FILES.values())


# ------------------------------------------------------------------------------- weekly table
def build_live_panel(base: pd.DataFrame, uploads: pd.DataFrame) -> pd.DataFrame:
    """Rows for every uploaded week (gaps included) plus the next week, for every store/product series."""
    pairs = base[["store", "group"]].drop_duplicates().sort_values(["store", "group"])
    idx = pd.MultiIndex.from_frame(pairs)
    base_last = base.week_start.max()
    weeks = pd.date_range(base_last + pd.Timedelta(days=7), uploads.week_start.max(), freq="W-MON")
    up = uploads.set_index(KEYS)
    up_weeks = set(up.index.get_level_values("week_start"))
    reported = uploads.groupby("week_start").store.agg(set).to_dict()

    def week_rows(w):
        return up.xs(w, level="week_start").reindex(idx) if w in up_weeks else pd.DataFrame(index=idx, columns=up.columns,
                                                                                               dtype=float)

    last = base[base.week_start == base_last].set_index(["store", "group"]).reindex(idx)
    prev_close, prev_est = last.closing_stock.astype(float), pd.Series(False, index=idx)
    store_of = idx.get_level_values("store")
    frames = []
    for i, w in enumerate(weeks):
        u = week_rows(w)
        rep = store_of.isin(reported.get(w, set()))
        gross = u.gross_qty.astype(float).fillna(0.0).where(rep)
        ret = u.return_qty.astype(float).fillna(0.0).where(rep)
        promo = u.promo_qty.astype(float).fillna(0.0).where(rep)
        start = u.stock_start.astype(float)
        opening = start.fillna(prev_close)
        open_est = start.isna() & prev_est & opening.notna()
        nxt = week_rows(weeks[i + 1]).stock_start.astype(float) if i + 1 < len(weeks) else pd.Series(np.nan, index=idx)
        closing = u.stock_end.astype(float).fillna(nxt)
        est = closing.isna() & opening.notna() & gross.notna()
        closing = closing.where(~est, (opening - gross + ret).clip(lower=0))
        frames.append(pd.DataFrame({
            "week_start": w, "gross_qty": gross, "return_qty": ret, "promo_qty": promo, "closing_stock": closing,
            "opening_stock": opening, "stock_estimated": open_est, "stock_reported": rep}, index=idx))
        prev_close, prev_est = closing, est
    nxt_week = (weeks[-1] if len(weeks) else base_last) + pd.Timedelta(days=7)
    frames.append(pd.DataFrame({"week_start": nxt_week, "gross_qty": np.nan, "return_qty": np.nan,
                                "promo_qty": np.nan, "closing_stock": np.nan, "opening_stock": prev_close,
                                "stock_estimated": prev_est & prev_close.notna(), "stock_reported": False}, index=idx))
    p = pd.concat(frames).reset_index()
    p["gross_amount"] = np.nan
    p["net_qty"] = p.gross_qty - p.return_qty
    p["promo_share"] = np.where(p.gross_qty > 0, p.promo_qty / p.gross_qty.where(p.gross_qty > 0), 0.0)
    p.loc[p.gross_qty.isna(), "promo_share"] = np.nan
    p["closing_book_stock"] = p.closing_stock
    p["neg_book_records"] = np.nan
    p["stock_recorded"] = p.closing_stock.notna()
    p["receipts_est"] = p.closing_stock - p.opening_stock + p.net_qty
    p["split"] = np.where(p.week_start == nxt_week, "next", "new")
    return p[list(base.columns) + ["stock_estimated", "stock_reported"]].sort_values(KEYS).reset_index(drop=True)


# ------------------------------------------------------------------------------- forecasts
def forecast_weeks(feats: pd.DataFrame, origins: list[pd.Timestamp], fcfg: dict, seed: int,
                   progress: Callable[[str], None] = print) -> pd.DataFrame:
    """One-week-ahead point and quantile forecasts with the saved model settings (no re-tuning)."""
    params = {k: v for k, v in fcfg["point_model"].items() if k != "objective"}
    lo_q, hi_q = fcfg["quantile_levels"]
    specs = {"yhat": LGBMSpec(fcfg["point_model"]["objective"], None, params),
             "q_lo": LGBMSpec("quantile", lo_q, params), "q_hi": LGBMSpec("quantile", hi_q, params)}
    cols = feature_columns(feats)
    out = []
    for n, t in enumerate(origins, 1):
        progress(f"Forecasting the week of {t:%d %b %Y} ({n} of {len(origins)})")
        train = feats[(feats.week_start < t) & feats[TARGET].notna()]
        test = feats[feats.week_start == t]
        p = test[KEYS].copy()
        for name, spec in specs.items():
            p[name] = np.clip(spec.model(seed).fit(train[cols], train[TARGET]).predict(test[cols]), 0, None)
        out.append(p)
    return pd.concat(out, ignore_index=True)


def add_live_intervals(fc_new: pd.DataFrame, fc_base: pd.DataFrame, fcfg: dict) -> pd.DataFrame:
    """Likely ranges as in the original pipeline. The rolling calibration window reaches back into the
    original forecasts, so the first uploaded weeks are calibrated on real past errors."""
    window = fcfg["calibration_weeks"]
    hist_weeks = sorted(fc_base.week_start.unique())[-window:]
    hist = fc_base[fc_base.week_start.isin(hist_weeks)][KEYS + ["y", "yhat", "q_lo", "q_hi", "scale"]]
    both = add_intervals(pd.concat([hist.assign(_new=False), fc_new.assign(_new=True)], ignore_index=True),
                         fcfg["alpha"], window)
    out = both[both._new].drop(columns="_new").reset_index(drop=True)
    for m in METHODS:
        out[f"lo_{m}"], out[f"hi_{m}"] = integerize(out[f"lo_{m}"], out[f"hi_{m}"])
    out["lo"], out["hi"] = out[f"lo_{fcfg['interval_method']}"], out[f"hi_{fcfg['interval_method']}"]
    return out


# ------------------------------------------------------------------------------- transfers
def next_week_tensors(fc_next: pd.DataFrame, econ: pd.DataFrame, stores: list[str]) -> dict[str, ProductTensors]:
    """Single-week simulator inputs for the planning week (actual opening stock, as-of forecasts)."""
    econ = econ.set_index("group")
    week = pd.Timestamp(fc_next.week_start.iloc[0])
    out = {}
    for prod, g in fc_next.groupby("group"):
        g = g.set_index("store").reindex(stores)

        def arr(col, fill=0.0):
            v = g[col].to_numpy(float)[None, :]
            return v if fill is None else np.nan_to_num(v, nan=fill)

        z = np.zeros((1, len(stores)))
        out[prod] = ProductTensors(
            product=prod, split="next", weeks=[week], stock0=arr("opening_stock", None), receipts=z, demand_obs=z,
            lam=z, yhat=arr("yhat"), lo=arr("lo"), hi=arr("hi"), cqr_lo=arr("lo_cqr"), cqr_hi=arr("hi_cqr"),
            unit_margin=float(econ.loc[prod, "unit_margin"]), unit_cost=float(econ.loc[prod, "unit_cost"]))
    return out


def recommend_transfers(fc_next: pd.DataFrame, econ: pd.DataFrame, cfg: dict,
                        policies: dict[str, Callable]) -> pd.DataFrame:
    """policies: approach name -> policy(sim) -> [(src, dst, qty)]. One row per recommended shipment."""
    stores = sorted(fc_next.store.unique())
    network = build_network(stores, cfg)
    sim_cfg = SimConfig.from_config(cfg)
    sim = RedistributionSim(next_week_tensors(fc_next, econ, stores), network.distance_km, sim_cfg, "replay")
    rows = []
    for approach, policy in policies.items():
        for prod in sim.products:
            sim.reset(prod, 0)
            for s, d, q in sim.validate(policy(sim))[0]:
                km = float(network.distance_km[s, d])
                rows.append({"approach": approach, "group": prod, "from_store": stores[s], "to_store": stores[d],
                             "units": int(q), "km": round(km, 1), "cost": round(float(transfer_cost(
                                 q, km, sim_cfg.transfer_fixed, sim_cfg.transfer_per_unit_km)), 2)})
    return pd.DataFrame(rows, columns=["approach", "group", "from_store", "to_store", "units", "km", "cost"])


# ------------------------------------------------------------------------------- main entry point
def run_update(cfg: dict, policies: dict[str, Callable], progress: Callable[[str], None] = print) -> dict:
    uploads = load_uploads()
    if uploads is None or uploads.empty:
        raise ValueError("No uploaded data yet.")
    progress("Building the weekly table")
    base = base_panel()
    live = build_live_panel(base, uploads)
    progress("Preparing forecast inputs")
    feats = build_features(pd.concat([base, live.drop(columns=["stock_estimated", "stock_reported"])],
                                     ignore_index=True), cfg["features"])
    fcfg = json.loads((MODELS_DIR / "forecast_config.json").read_text(encoding="utf-8"))
    origins = sorted(pd.Timestamp(w) for w in live.week_start.unique())
    fc = forecast_weeks(feats, origins, fcfg, cfg["seed"], progress)

    progress("Working out likely ranges and stock risk")
    fc = fc.merge(feats[KEYS + [TARGET, "split", "f_roll_std_8", "opening_stock"]], on=KEYS)
    fc = fc.rename(columns={TARGET: "y", "f_roll_std_8": "scale"}).merge(baseline_forecasts(feats), on=KEYS)
    fc = add_live_intervals(fc, pd.read_parquet(PROCESSED_DIR / "forecasts.parquet"), fcfg)
    fc = fc.merge(live[KEYS + ["stock_estimated", "stock_reported"]], on=KEYS)
    risk = assess(fc, fcfg["alpha"], cfg["environment"]["excess_horizon_weeks"])

    progress("Recommending transfers for next week")
    econ = pd.read_csv(PROCESSED_DIR / "unit_economics.csv")
    nxt = fc[fc.split == "next"]
    transfers = recommend_transfers(nxt, econ, cfg, policies)

    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    live.to_parquet(LIVE_FILES["panel.parquet"], index=False)
    fc.to_parquet(LIVE_FILES["forecasts.parquet"], index=False)
    risk[RISK_COLS].to_parquet(LIVE_FILES["risk.parquet"], index=False)
    transfers.to_csv(LIVE_FILES["transfers.csv"], index=False)
    r_next = risk[risk.split == "next"]
    new_weeks = live[live.split == "new"]
    summary = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "original_data_last_week": str(base.week_start.max().date()),
        "uploaded_weeks": [str(w.date()) for w in sorted(uploads.week_start.unique())],
        "missing_weeks": [str(pd.Timestamp(w).date()) for w in sorted(set(new_weeks.week_start) -
                                                                       set(uploads.week_start))],
        "stores_reported_per_week": {str(pd.Timestamp(w).date()): int(g.stock_reported.groupby(g.store).any().sum())
                                     for w, g in new_weeks.groupby("week_start")},
        "next_week": str(pd.Timestamp(nxt.week_start.iloc[0]).date()),
        "next_week_pairs": int(len(r_next)),
        "next_week_stock_estimated": int(r_next.stock_estimated.sum()),
        "next_week_no_stock_record": int(r_next.stock_missing.sum()),
        "transfers": {a: {"shipments": int(len(g)), "units": int(g.units.sum())}
                      for a, g in transfers.groupby("approach")},
    }
    save_json(summary, LIVE_FILES["summary.json"])
    progress("Done")
    return summary
