"""Terminal version of the dashboard: the same pages and choices, shown as text menus in the console.

Run from the project root:  python dashboard/terminal.py
Like the Streamlit app it only READS saved results and trained models (nothing is trained on launch). The
"Add new data" page cleans a weekly sheet and refreshes next week's forecasts, risk and transfers with the
already-trained models (inventory_rl.live); the original study's results are never changed.
"""

from __future__ import annotations

import os
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_access as da  # noqa: E402
import explain as ex  # noqa: E402

from inventory_rl.live import update as live  # noqa: E402
from inventory_rl.live.upload import COLUMNS, clean_weekly_sheet, read_sheet, template  # noqa: E402
from inventory_rl.utils.config import load_config  # noqa: E402
from inventory_rl.utils.paths import METRICS_DIR, PROCESSED_DIR, REPORTS_DIR  # noqa: E402

if os.name == "nt":
    os.system("")  # turn on ANSI colours in the Windows console
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

RED, AMBER, GREEN, GREY, BOLD, DIM, CYAN, RESET = (
    "\033[91m", "\033[93m", "\033[92m", "\033[90m", "\033[1m", "\033[2m", "\033[96m", "\033[0m")
RISK_ANSI = {"Stockout risk": RED, "Excess stock": AMBER, "Normal": GREEN, "No stock record": GREY}
APPROACHES = {"ai": "Uncertainty-aware AI agent", "ai-point": "AI agent", "upper": "Rule: cover a high estimate",
              "cqr": "Rule: cover a calibrated high estimate", "point": "Rule: cover the forecast",
              "none": "No transfers"}
# the two approaches offered for planning (also used when refreshing the plan from added data)
APPROACH_NAMES = {APPROACHES[k]: k for k in ("ai", "upper")}

_cache: dict = {}


# ------------------------------------------------------------------------------------- loaders
def cached(key, fn):
    if key not in _cache:
        _cache[key] = fn()
    return _cache[key]


def table(path: Path):
    return cached(str(path), lambda: da.load_table(path))


def js(path: Path):
    return cached(str(path), lambda: da.load_json(path))


def live_file(name: str):
    """A table/summary rebuilt from uploaded data, read fresh each time (None until the first upload)."""
    if not live.live_ready():
        return None
    path = live.LIVE_FILES[name]
    return da.load_json(path) if path.suffix == ".json" else da.load_table(path)


def sim_context():
    def load():
        from inventory_rl.environment.factory import load_context

        print(f"{DIM}Loading forecasts, stock and store network...{RESET}")
        return load_context(load_config())
    return cached("ctx", load)


def policy(kind: str):
    def load():
        from inventory_rl.baselines.rules import RuleBasedPolicy, no_transfer

        if kind in ("ai", "ai-point"):
            from inventory_rl.agents.ppo import load_policy

            variant = "rl_uncertainty" if kind == "ai" else "rl_point"
            path = da.best_validation_run(variant)
            if path is not None:
                return load_policy(path, kind == "ai", load_config()["environment"], APPROACHES[kind])
            print(f"{AMBER}No trained AI agent found; using the 'cover a high estimate' rule.{RESET}")
            return RuleBasedPolicy("upper")
        return no_transfer if kind == "none" else RuleBasedPolicy(kind)
    return cached(("policy", kind), load)


def trajectory(kind: str, prod: str) -> list[tuple[np.ndarray, list]]:
    """Run the approach over the test weeks on recorded sales (as in the dashboard) and keep, for every
    week, the simulated opening stock and the transfers recommended for it."""
    def run():
        sim = sim_context().sim("test", "replay")
        sim.reset(prod, 0)
        pol, out = policy(kind), []
        while not sim.done:
            valid = sim.validate(pol(sim))[0]
            out.append((sim.stock.copy(), valid))
            sim.step(valid)
        return out
    return cached(("traj", kind, prod), run)


def example_case() -> tuple[int, str, str]:
    """Default selection: the latest week with an AI-recommended transfer and the destination of its largest one."""
    def find():
        ctx = sim_context()
        products = sorted(ctx.tensors["test"])
        n_weeks = ctx.tensors["test"][products[0]].n_weeks
        for wi in range(n_weeks - 1, -1, -1):
            best = None
            for prod in products:
                for _, dst, q in trajectory("ai", prod)[wi][1]:
                    if best is None or q > best[0]:
                        best = (q, ctx.network.stores[dst], prod)
            if best:
                return wi, best[1], best[2]
        return n_weeks - 1, ctx.network.stores[0], products[0]
    return cached("example", find)


# ------------------------------------------------------------------------------------- terminal helpers
def header(text: str) -> None:
    print(f"\n{BOLD}{CYAN}{'=' * 78}\n {text}\n{'=' * 78}{RESET}")


def sub(text: str) -> None:
    print(f"\n{BOLD}{text}{RESET}")


def note(text: str) -> None:
    import textwrap

    print(DIM + textwrap.fill(text, 78) + RESET)


def visible_len(s: str) -> int:
    import re

    return len(re.sub(r"\033\[[0-9;]*m", "", s))


def print_table(rows: list[list], headers: list[str]) -> None:
    cells = [[str(c) for c in r] for r in rows]
    widths = [max([len(h)] + [visible_len(r[i]) for r in cells]) for i, h in enumerate(headers)]
    print("  " + "  ".join(f"{BOLD}{h:<{w}}{RESET}" for h, w in zip(headers, widths)))
    print("  " + "  ".join("-" * w for w in widths))
    for r in cells:
        print("  " + "  ".join(c + " " * (w - visible_len(c)) for c, w in zip(r, widths)))


def hbar(value: float, top: float, width: int = 24) -> str:
    if top <= 0 or not np.isfinite(value):
        return ""
    n = int(round(width * abs(value) / top))
    return (GREEN if value >= 0 else RED) + "#" * n + RESET


def signed(x: float) -> str:
    colour = GREEN if x > 0.005 else RED if x < -0.005 else DIM
    return f"{colour}{x:+,.0f}{RESET}" if abs(x) >= 100 else f"{colour}{x:+,.2f}{RESET}"


def choose(prompt: str, options: list[str], default: int = 0, columns: int = 1) -> int:
    """Numbered menu; Enter keeps the default. Returns the chosen index."""
    print(f"\n{BOLD}{prompt}{RESET}")
    width = max(len(o) for o in options) + 6
    for i, o in enumerate(options):
        mark = "*" if i == default else " "
        end = "\n" if (i + 1) % columns == 0 or i == len(options) - 1 else ""
        print(f"{mark}{i + 1:>3}. {o:<{width}}".rstrip() if end else f"{mark}{i + 1:>3}. {o:<{width}}", end=end)
    while True:
        raw = input(f"Choose 1-{len(options)} [Enter = {default + 1}]: ").strip()
        if raw == "":
            return default
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        print(f"{RED}Please type a number from 1 to {len(options)}.{RESET}")


def pause() -> None:
    input(f"\n{DIM}Press Enter to go back to the menu...{RESET}")


def missing(obj, what: str) -> bool:
    if obj is None:
        print(f"{AMBER}{what} is not available yet. The project pipeline has to be run first.{RESET}")
        return True
    return False


# ===================================================================================== 1. decision support
def decision_support() -> None:
    header("DECISION SUPPORT")
    if live.live_ready():
        summ = live_file("summary.json")
        if choose("Data", [f"Next week, from your added data ({ex.week_label(summ['next_week'])})",
                           "Test weeks (project evaluation)"], 0) == 0:
            decision_support_live()
            return
    note("Choose a week, a store and a product to see the demand outlook, the stock risk and the recommended "
         "transfer. Press Enter at any question to keep the example case (marked *).")
    risk, fc = table(PROCESSED_DIR / "risk.parquet"), table(PROCESSED_DIR / "forecasts.parquet")
    if missing(risk, "Risk information") or missing(fc, "Demand forecasts"):
        return
    from inventory_rl.risk.assessment import assess
    from inventory_rl.routing.network import transfer_cost

    ctx, cfg = sim_context(), load_config()
    T = ctx.tensors["test"]
    products, stores = sorted(T), list(ctx.network.stores)
    names = da.product_labels(products)
    print(f"{DIM}Finding an example case...{RESET}")
    ex_week, ex_store, ex_prod = example_case()
    weeks = T[products[0]].weeks

    wi = choose("Planning week", [ex.week_label(w) for w in weeks], ex_week, columns=2)
    si = choose("Store", [ex.store_label(s) for s in stores], stores.index(ex_store), columns=5)
    pi = choose("Product", [names[g] for g in products], products.index(ex_prod), columns=3)
    kind = list(APPROACH_NAMES.values())[choose("Recommendation approach", list(APPROACH_NAMES), 0)]
    store, prod, P, week = stores[si], products[pi], T[products[pi]], pd.Timestamp(weeks[wi])
    stock, valid = trajectory(kind, prod)[wi]
    rk = assess(pd.DataFrame({"opening_stock": stock, "yhat": P.yhat[wi], "hi": P.hi[wi]}),
                cfg["uncertainty"]["alpha"], cfg["environment"]["excess_horizon_weeks"])

    header(f"{ex.store_label(store)} | {names[prod]} | {ex.week_label(week)} | {APPROACHES[kind]}")
    sub("Demand outlook (last weeks up to the planning week)")
    d = fc[(fc.store == store) & (fc.group == prod) & (fc.week_start <= week)].sort_values("week_start").tail(10)
    top = max(d.hi.max(), d.y.max(), 1)
    print_table([[f"{pd.Timestamp(r.week_start):%d %b %Y}" + (" <" if pd.Timestamp(r.week_start) == week else ""),
                  f"{r.yhat:.1f}", f"{r.lo:.0f}-{r.hi:.0f}", f"{r.y:.0f}",
                  f"{CYAN}{'=' * int(round(20 * r.yhat / top))}{RESET}"]
                 for r in d.itertuples()], ["Week", "Forecast", "Likely range", "Actual sales", "Forecast bar"])
    note("Forecast made one week ahead, with its likely range. '<' marks the planning week (its actual sales are "
         "shown for comparison).")

    r = rk.iloc[si]
    status = ex.risk_status(r.risk_class)
    p_out, stock_now, expected = float(r.p_stockout), float(r.opening_stock), float(P.yhat[wi][si])
    sub("Stock risk")
    print(f"  {RISK_ANSI[status]}{BOLD}{status.upper()}{RESET}  {ex.RISK_HELP[status]}")
    print(f"  In stock: {BOLD}{stock_now:.0f}{RESET}   Expected sales: {BOLD}{expected:.1f}{RESET} "
          f"(likely range {P.lo[wi][si]:.0f} to {P.hi[wi][si]:.0f})   Chance of running out next week: "
          f"{BOLD}{'Out of stock' if stock_now <= 0 else ex.chance(p_out)}{RESET}")

    def describe(s: int, dd: int, q: int) -> tuple[list, str]:
        km = float(ctx.network.distance_km[s, dd])
        cost = float(transfer_cost(q, km, ctx.sim_cfg.transfer_fixed, ctx.sim_cfg.transfer_per_unit_km))
        reason = ex.transfer_reason(ex.store_label(stores[s]), ex.store_label(stores[dd]), q, km, cost, stock[s],
                                    P.yhat[wi][s], stock[dd], P.yhat[wi][dd], float(rk.p_stockout.iloc[dd]))
        return [ex.store_label(stores[s]), ex.store_label(stores[dd]), q, f"{km:.0f}", ex.money(cost)], reason

    print_transfers(valid, si, store, status, p_out, stock_now, expected, describe)
    note(f"Stock levels are simulated as if the chosen approach had been in use since the week of "
         f"{pd.Timestamp(weeks[0]):%d %b %Y}, using the sales that actually happened. Stores without a stock count "
         f"at that start are treated as empty. Distances come from a simulated store map and costs use the "
         f"project's assumed shipping costs.")


def print_transfers(valid: list, si: int, store: str, status: str, p_out: float, stock_now: float, expected: float,
                    describe) -> None:
    sub("Recommended transfer")
    mine = [describe(s, dd, q) for s, dd, q in valid if si in (s, dd)]
    if mine:
        row, reason = mine[0]
        print(f"  {BOLD}{row[0]}  --{ex.units(row[2])}-->  {row[1]}{RESET}   distance {row[3]} km, "
              f"estimated cost {row[4]}")
        print(f"  {BOLD}Why:{RESET} {reason}")
        if len(mine) > 1:
            print(f"\n  This store is part of {len(mine)} transfers this week:")
            print_table([m[0] for m in mine], ["From", "To", "Units", "Distance (km)", "Cost"])
    else:
        print(f"  {BOLD}No transfer for {ex.store_label(store)}.{RESET} "
              + ex.no_transfer_reason(ex.store_label(store), status, p_out, 0.0 if np.isnan(stock_now) else stock_now,
                                      expected))
    if valid:
        sub(f"All transfers suggested for this product this week ({len(valid)})")
        print_table([describe(*v)[0] for v in valid], ["From", "To", "Units", "Distance (km)", "Cost"])


def decision_support_live() -> None:
    """Next week's plan built from the uploaded weekly sheets (inventory_rl.live.update)."""
    lrisk, lfc, ltr, summ = (live_file("risk.parquet"), live_file("forecasts.parquet"), live_file("transfers.csv"),
                             live_file("summary.json"))
    fc = table(PROCESSED_DIR / "forecasts.parquet")
    nxt = lrisk[lrisk.split == "next"]
    week = pd.Timestamp(summ["next_week"])
    stores = list(sim_context().network.stores)
    products = sorted(nxt.group.unique())
    names = da.product_labels(products)
    note(f"Plan for the week of {week:%d %b %Y}, using the data you added up to the week of "
         f"{pd.Timestamp(summ['uploaded_weeks'][-1]):%d %b %Y} (updated {summ['updated_at'].replace('T', ' ')}). "
         f"Press Enter at any question to keep the suggested case (marked *).")
    method = list(APPROACH_NAMES)[choose("Recommendation approach", list(APPROACH_NAMES), 0)]
    tr = ltr[ltr.approach == method]
    if len(tr):  # default selection: the largest recommended shipment
        top = tr.sort_values("units", ascending=False).iloc[0]
        ex_store, ex_prod = top.to_store, top.group
    else:  # otherwise the store and product expected to be short of the most units
        top = nxt.sort_values(["expected_shortage", "yhat"], ascending=False).iloc[0]
        ex_store, ex_prod = top.store, top.group
    print(f"  {DIM}{len(tr)} shipment(s) recommended across all products next week.{RESET}")
    si = choose("Store", [ex.store_label(s) for s in stores], stores.index(ex_store), columns=5)
    pi = choose("Product", [names[g] for g in products], products.index(ex_prod), columns=3)
    store, prod = stores[si], products[pi]
    rows = nxt[nxt.group == prod].set_index("store").reindex(stores)

    header(f"{ex.store_label(store)} | {names[prod]} | {ex.week_label(week)} (next week) | {method}")
    sub("Demand outlook (recent weeks and next week)")
    cols = ["week_start", "yhat", "y", "lo", "hi"]
    d = pd.concat([fc.loc[(fc.store == store) & (fc.group == prod), cols],
                   lfc.loc[(lfc.store == store) & (lfc.group == prod), cols]]).sort_values("week_start").tail(10)
    top_v = max(d.hi.max(), d.y.max(skipna=True) if d.y.notna().any() else 0, 1)
    print_table([[f"{pd.Timestamp(r.week_start):%d %b %Y}" + (" <" if pd.Timestamp(r.week_start) == week else ""),
                  f"{r.yhat:.1f}", f"{r.lo:.0f}-{r.hi:.0f}", "-" if pd.isna(r.y) else f"{r.y:.0f}",
                  f"{CYAN}{'=' * int(round(20 * r.yhat / top_v))}{RESET}"]
                 for r in d.itertuples()], ["Week", "Forecast", "Likely range", "Actual sales", "Forecast bar"])
    note("Forecast made one week ahead, with its likely range. '<' marks next week, which has no sales yet.")

    r = rows.loc[store]
    status = ex.risk_status(r.risk_class)
    stock_now, p_out, expected = float(r.opening_stock), float(r.p_stockout), float(r.yhat)
    sub("Stock risk")
    print(f"  {RISK_ANSI[status]}{BOLD}{status.upper()}{RESET}  {ex.RISK_HELP[status]}")
    print(f"  In stock: {BOLD}{'no record' if np.isnan(stock_now) else f'{stock_now:.0f}'}{RESET}"
          f"{' (estimated)' if r.stock_estimated else ''}   Expected sales: {BOLD}{expected:.1f}{RESET} "
          f"(likely range {r.lo:.0f} to {r.hi:.0f})   Chance of running out next week: {BOLD}"
          f"{'unknown' if np.isnan(stock_now) else 'Out of stock' if stock_now <= 0 else ex.chance(p_out)}{RESET}")
    if r.stock_estimated:
        note("Estimated stock: your sheets did not give it, so it is the last known stock minus units sold since, "
             "assuming no deliveries.")

    stock, p_all, yhat = (rows.opening_stock.to_numpy(float), rows.p_stockout.to_numpy(float),
                          rows.yhat.to_numpy(float))
    pick = tr[tr.group == prod]
    valid = [(stores.index(a), stores.index(b), int(q)) for a, b, q in zip(pick.from_store, pick.to_store, pick.units)]
    costs = {(stores.index(a), stores.index(b)): (km, c) for a, b, km, c in zip(pick.from_store, pick.to_store,
                                                                                pick.km, pick.cost)}

    def describe(s: int, dd: int, q: int) -> tuple[list, str]:
        km, cost = costs[(s, dd)]
        no_rec = bool(np.isnan(stock[dd]))
        reason = ex.transfer_reason(ex.store_label(stores[s]), ex.store_label(stores[dd]), q, km, cost, stock[s],
                                    yhat[s], 0.0 if no_rec else stock[dd], yhat[dd],
                                    1.0 if no_rec else float(p_all[dd]), dst_no_record=no_rec)
        return [ex.store_label(stores[s]), ex.store_label(stores[dd]), q, f"{km:.0f}", ex.money(cost)], reason

    print_transfers(valid, si, store, status, p_out, stock_now, expected, describe)
    note("Stock is the stock at the start of next week from your sheets; where a sheet left it out it is estimated, "
         "and stores with no stock figure at all are treated as empty when planning. Forecasts use the model settings "
         "chosen in the original study and the AI agent trained in the study. Distances come from a simulated store "
         "map and costs use the project's assumed shipping costs.")


# ===================================================================================== 2. store overview
def store_overview() -> None:
    header("STORE OVERVIEW")
    note("Which stores need attention in the chosen week. Red = may run out, amber = too much stock, "
         "green = normal, grey = no stock count in the data.")
    risk = table(PROCESSED_DIR / "risk.parquet")
    if missing(risk, "Risk information"):
        return
    test = risk[risk.split == "test"]
    lrisk = live_file("risk.parquet")
    if lrisk is not None:  # weeks from uploaded sheets, and the next week to plan for
        test = pd.concat([test, lrisk[test.columns]], ignore_index=True)
    test = test.assign(status=lambda d: d.risk_class.map(ex.risk_status))
    weeks, prods = sorted(test.week_start.unique()), sorted(test.group.unique())
    names = da.product_labels(prods)
    tag = test.drop_duplicates("week_start").set_index("week_start").split.map(
        {"new": " (your data)", "next": " (next week)"}).fillna("")
    wk = weeks[choose("Week", [ex.week_label(w) + tag[w] for w in weeks], len(weeks) - 1, columns=2)]
    pick = (["All products"] + prods)[choose("Product", ["All products"] + [names[g] for g in prods], 0, columns=3)]
    d = test[test.week_start == wk]
    if pick != "All products":
        d = d[d.group == pick]

    header(f"{ex.week_label(wk)}{tag[wk]} | {pick if pick == 'All products' else names[pick]}")
    n = d.store.nunique()
    print(f"  Stores with a stockout risk : {RED}{d[d.status == 'Stockout risk'].store.nunique()}{RESET} of {n}")
    print(f"  Store-product pairs at risk : {RED}{int((d.status == 'Stockout risk').sum())}{RESET}")
    print(f"  Pairs with excess stock     : {AMBER}{int((d.status == 'Excess stock').sum())}{RESET}")
    print(f"  Pairs with no stock count   : {GREY}{int((d.status == 'No stock record').sum())}{RESET}")

    if pick == "All products":
        counts = d.groupby(["store", "status"]).size().unstack(fill_value=0).reindex(columns=ex.RISK_ORDER, fill_value=0)
        counts = counts.sort_values(["Stockout risk", "Excess stock"], ascending=False)
        sub("Products per status, stores with the most stockout risks first")
        rows = []
        for s, c in counts.iterrows():
            bar = "".join(RISK_ANSI[k] + "#" * int(c[k]) + RESET for k in ex.RISK_ORDER)
            rows.append([ex.store_label(s), f"{RED}{c['Stockout risk']}{RESET}", f"{AMBER}{c['Excess stock']}{RESET}",
                         f"{GREEN}{c['Normal']}{RESET}", f"{GREY}{c['No stock record']}{RESET}", bar])
        print_table(rows, ["Store", "Risk", "Excess", "Normal", "No record", "Products"])
    else:
        sub("Every store for this product")
        rows = []
        for r in d.sort_values(["p_stockout", "store"], ascending=[False, True], na_position="last").itertuples():
            rows.append([ex.store_label(r.store), f"{RISK_ANSI[r.status]}{r.status}{RESET}",
                         "no record" if pd.isna(r.opening_stock) else f"{r.opening_stock:.0f}", f"{r.yhat:.1f}",
                         "unknown" if pd.isna(r.p_stockout) else ex.chance(r.p_stockout)])
        print_table(rows, ["Store", "Status", "In stock", "Expected sales", "Chance of running out"])


# ===================================================================================== 3. results
def results_page() -> None:
    header("RESULTS")
    note("How each approach performed on 8 recent weeks that were held back for testing, compared with making "
         "no transfers at all.")
    summary = table(METRICS_DIR / "evaluation_summary.csv")
    if missing(summary, "Test results"):
        return
    results = js(METRICS_DIR / "results.json")
    scen = ["poisson", "replay"][choose("Demand used in the test", ["Simulated demand (20 runs)", "Actual past sales"])]
    s = summary[summary.scenario == scen].set_index("policy")
    base = s.loc["No redistribution"]
    df = pd.DataFrame({
        "approach": [ex.POLICY_NAMES[p] for p in s.index],
        "lost_sales_avoided": [-ex.pct_change(s.loc[p, "lost_units"], base.lost_units) for p in s.index],
        "leftover_change": [ex.pct_change(s.loc[p, "avg_closing_stock"], base.avg_closing_stock) for p in s.index],
        "shipments": s.n_transfers.values, "transfer_spend": s.transfer_cost.values,
        "money_saved": s.saving_vs_noop.values, "spread": np.nan}, index=s.index)
    seeds = table(METRICS_DIR / "rl_saving_across_training_seeds.csv")
    if seeds is not None:
        for _, r in seeds[seeds.scenario == scen].iterrows():
            if r.policy in df.index:
                df.loc[r.policy, "spread"] = r["std"]
    show = df.drop(index="No redistribution").sort_values("money_saved", ascending=False)
    ua, point_ai = show.loc["RL (forecast + uncertainty)"], show.loc["RL (point forecast)"]

    header("Uncertainty-aware AI agent vs no transfers")
    print(f"  Lost sales avoided: {BOLD}{ua.lost_sales_avoided:.0f}%{RESET}   Change in leftover stock: "
          f"{BOLD}{ua.leftover_change:+.1f}%{RESET}   Shipments in 8 weeks: {BOLD}{ua.shipments:.0f}{RESET}   "
          f"Money saved: {BOLD}{ex.money(ua.money_saved)}{RESET}")

    sub("All approaches (sorted by money saved)")
    top = show.money_saved.abs().max()
    print_table([[r.approach, f"{r.lost_sales_avoided:.0f}%", f"{r.leftover_change:+.1f}%", f"{r.shipments:.0f}",
                  ex.money(r.transfer_spend),
                  signed(r.money_saved) + ("" if np.isnan(r.spread) else f" ± {r.spread:,.0f}"),
                  hbar(r.money_saved, top)] for r in show.itertuples()],
                ["Approach", "Lost sales avoided", "Leftover", "Shipments", "Shipping spend", "Money saved", ""])
    note("Money saved = cost of no transfers minus (lost-sale cost + holding cost + shipping cost), over 30 products "
         "and 40 stores. ± is the spread over 5 separately trained AI agents.")

    best = ex.best_policy(dict(zip(show.approach, show.money_saved)))
    losers = show[show.money_saved < 0].approach.tolist()
    takeaway = (f"In this test, {best} saved the most money. The uncertainty-aware AI agent saved "
                f"{ex.money(ua.money_saved)}, {'more' if ua.money_saved > point_ai.money_saved else 'less'} than the "
                f"AI agent without uncertainty ({ex.money(point_ai.money_saved)}), using {ua.shipments:.0f} shipments.")
    if losers:
        takeaway += f" {' and '.join(losers)} spent more on shipping than it saved."
    cmp_ = (results or {}).get("rl_seed_level_comparison", {}).get(
        f"RL (forecast + uncertainty) minus RL (point forecast) | {scen}")
    if cmp_:
        takeaway += (" The gap between the two AI agents is within their run-to-run variation, so it is not a "
                     "reliable difference." if cmp_["p_value"] >= 0.05 else
                     " The gap between the two AI agents is larger than their run-to-run variation.")
    sub("Takeaway")
    import textwrap

    print(textwrap.fill(takeaway, 78, initial_indent="  ", subsequent_indent="  "))

    sens = table(METRICS_DIR / "cost_sensitivity.csv")
    if sens is None:
        return
    sub("What if the cost assumptions are different?")
    costs = load_config()["costs"]
    sens = sens[sens.scenario == scen].copy()
    for param, label in ex.COST_PARAM_NAMES.items():
        part = sens[(sens.parameter == param) | (sens.parameter == "base")].assign(
            value=lambda d, p=param: d.value.fillna(float(costs[p])))
        lo, hi = part.value.min(), part.value.max()
        pick = {v: ex.best_policy({ex.POLICY_NAMES[p]: x for p, x in
                                   part[part.value == v].set_index("policy").saving_mean.items()}) for v in (lo, hi)}
        print(f"  - {label} (assumed {costs[param]:g}): at {lo:g} {BOLD}{pick[lo]}{RESET} saves the most; at {hi:g} "
              f"{BOLD}{pick[hi]}{RESET} {'is cheapest' if pick[hi] == 'No transfers' else 'saves the most'}.")
    params = list(ex.COST_PARAM_NAMES)
    choice = params[choose("Show the effect of", list(ex.COST_PARAM_NAMES.values()))]
    part = sens[(sens.parameter == choice) | (sens.parameter == "base")].assign(
        value=lambda d: d.value.fillna(float(costs[choice])))
    grid = part.pivot_table(index="policy", columns="value", values="saving_mean")
    grid = grid.reindex([p for p in ex.POLICY_NAMES if p in grid.index])
    sub(f"Money saved vs no transfers, by {ex.COST_PARAM_NAMES[choice].lower()}")
    print_table([[ex.POLICY_NAMES[p]] + [signed(v) for v in row] for p, row in grid.iterrows()],
                ["Approach"] + [f"{v:g}" + (" (assumed)" if v == float(costs[choice]) else "") for v in grid.columns])
    note("Negative = the approach loses money. The AI agents were trained once with the assumed costs and were not "
         "retrained for each what-if value.")


# ===================================================================================== 4. model performance
def model_performance() -> None:
    header("MODEL PERFORMANCE")
    sub("Pipeline status")
    status = da.artifact_status()
    print_table([[r.stage, f"{GREEN}Ready{RESET}" if r.available else f"{RED}Not run yet{RESET}"]
                 for r in status.itertuples()], ["Step", "Status"])

    audit, prep = js(REPORTS_DIR / "data_audit.json"), js(REPORTS_DIR / "preprocessing_report.json")
    if audit and prep:
        sub("Data quality")
        m, q = audit["mendeley"], prep["quality"]
        print_table([
            ["Sales transactions", f"{m['sales']['rows']:,}"],
            ["Stock records", f"{m['inventory']['rows']:,}"],
            ["Stores / products in the raw data", f"{m['sales']['n_stores']} / {m['sales']['n_products']:,}"],
            ["Sales period", " to ".join(x[:10] for x in m["sales"]["transaction_date_range"])],
            ["Return transactions", f"{m['sales']['return_rows']:,}"],
            ["Stock records with negative quantity", f"{m['inventory']['negative_qty_rows']:,}"],
            ["Weekly series used (store x product)",
             f"{q['series']:,} ({q['stores']} stores x {q['groups']} products, {q['weeks']} weeks)"],
            ["Store-weeks with no sales", f"{100 * q['zero_sales_share']:.0f}%"],
            ["Store-weeks with no stock count at the start", f"{100 * q['opening_stock_missing_share']:.0f}%"],
            ["Store-weeks starting with zero stock", f"{100 * q['stockout_weeks_share (opening_stock==0)']:.1f}%"],
            ["Duplicate records", f"{q['duplicate_keys']}"],
        ], ["Measure", "Value"])

    fm = js(METRICS_DIR / "forecast_metrics.json")
    if fm:
        sub("Forecast accuracy (test weeks)")
        labels = {"LightGBM": "Machine-learning model (used)", "Naive (t-1)": "Same as last week",
                  "MA(4)": "Average of last 4 weeks", "MA(8)": "Average of last 8 weeks", "Zero forecast": "Always zero"}
        print_table([[v, f"{fm['test'][k]['MAE']:.3f}", f"{fm['test'][k]['RMSE']:.3f}", f"{fm['test'][k]['Bias']:+.3f}"]
                     for k, v in labels.items() if k in fm["test"]],
                    ["Method", "Average error (units)", "Large-error score (RMSE)", "Bias (units)"])

    im = js(METRICS_DIR / "interval_metrics.json")
    if im:
        sub("Likely-range accuracy (test weeks)")
        labels = {"poisson": "Count-based range", "qr": "Quantile model", "cqr": "Calibrated quantile model",
                  "conformal": "Error-based range"}
        print_table([[labels.get(k, k) + (" (used)" if k == im["chosen"] else ""), f"{100 * v['coverage']:.0f}%",
                      f"{100 * v['nominal']:.0f}%", f"{v['mean_width']:.2f}"]
                     for k, v in im["by_split"]["test"].items()],
                    ["Method", "Sales inside the range", "Target", "Average width (units)"])

    per_run, train = table(METRICS_DIR / "rl_saving_per_training_seed.csv"), js(da.MODELS_DIR / "rl" / "training_summary.json")
    if per_run is not None and train:
        sub("AI agent training runs")
        val = {(r["variant"], r["seed"]): r["best_val_saving_poisson"] for r in train["runs"]}
        key = {"RL (point forecast)": "rl_point", "RL (forecast + uncertainty)": "rl_uncertainty"}
        print_table([[ex.POLICY_NAMES[r.policy], r.train_seed + 1, f"{val.get((key[r.policy], r.train_seed), np.nan):,.0f}",
                      f"{r.poisson:,.0f}", f"{r.replay:,.0f}"] for r in per_run.itertuples()],
                    ["Agent", "Run", "Saving, check weeks", "Saving, test (simulated)", "Saving, test (actual)"])
        note("The version kept from each run was chosen on the earlier check weeks, never on the test weeks.")

    sub("Cost assumptions")
    costs = load_config()["costs"]
    print_table([["Fixed cost per shipment", f"{costs['transfer_fixed']:g}"],
                 ["Cost per item per km", f"{costs['transfer_per_unit_km']:g}"],
                 ["Yearly holding cost (share of item cost)", f"{costs['holding_rate_per_year']:g}"],
                 ["Cost of a lost sale (times profit per item)", f"{costs['shortage_cost_multiplier']:g}"],
                 ["Profit and cost per item", "Taken from the sales data"],
                 ["Store distances", "Simulated map (no locations in the data)"]], ["Assumption", "Value"])

    cal = table(METRICS_DIR / "stockout_calibration_test.csv")
    if cal is not None:
        sub("Is the chance of running out trustworthy?")
        print_table([[f"{100 * r.predicted:.0f}%", f"{100 * r.observed:.0f}%", f"{r.n:,}"] for r in cal.itertuples()],
                    ["Predicted chance of running out", "Share that actually sold out", "Store-weeks"])
        note("If the two percentages are close, the predicted chance matched reality.")


# ===================================================================================== 5. week-by-week replay
def week_by_week() -> None:
    header("WEEK-BY-WEEK SIMULATION")
    note("Replays the 8 test weeks for one product. Each week shows the transfers the approach makes and its cost "
         "against making no transfers on the SAME demand. Transfers are simulated; sales and stock come from the data.")
    ctx = sim_context()
    T, stores = ctx.tensors["test"], ctx.network.stores
    products = sorted(T, key=lambda g: -T[g].demand_obs.sum())
    names = da.product_labels(products)
    prod = products[choose("Product (busiest first)",
                           [f"{names[g]} - {T[g].demand_obs.sum():.0f} units sold" for g in products], 0, columns=2)]
    kinds = list(APPROACHES)[:-1]
    kind = kinds[choose("Approach", [APPROACHES[k] for k in kinds], 0)]
    mode = ["replay", "poisson"][choose("Demand", ["Actual past sales", "Simulated demand (one random run)"], 0)]
    seed = 10_000
    if mode == "poisson":
        raw = input("Random seed [Enter = 10000]: ").strip()
        seed = int(raw) if raw.isdigit() else seed
    delay = [1.5, 0.0, -1][choose("Speed", ["Auto-play (1.5 s per week)", "All at once", "Press Enter for each week"])]

    sim, pol, p = ctx.sim("test", mode), policy(kind), T[prod]
    sim.reset(prod, 0, seed=seed)
    tot = dict(cost=0.0, noop=0.0, lost=0.0, demand=0.0, ships=0, units=0.0)
    while not sim.done:
        t, before = sim.t, sim.stock.copy()
        res = sim.step(pol(sim))
        for k, v in (("cost", res.total_cost), ("noop", res.noop_cost), ("lost", res.lost), ("demand", res.demand),
                     ("ships", res.n_transfers), ("units", res.units_transferred)):
            tot[k] += v
        print(f"\n{BOLD}{CYAN}--- Week {t + 1}/{sim.end}  ({ex.week_label(p.weeks[t])}) ---{RESET}")
        short = [i for i in range(len(stores)) if before[i] < np.ceil(p.yhat[t][i] - 1e-9)]
        print(f"  Stores below their forecast before the decision: {RED if short else GREEN}{len(short)}{RESET}"
              + (f" ({', '.join(ex.store_label(stores[i]) for i in short[:6])}{', ...' if len(short) > 6 else ''})"
                 if short else ""))
        for src, dst, q in res.transfers:
            print(f"  {ex.store_label(stores[src]):<10} --{q:>3} units--> {ex.store_label(stores[dst]):<10} "
                  f"{DIM}({sim.dist[src, dst]:.0f} km){RESET}")
        if not res.transfers:
            print(f"  {DIM}No transfers this week{RESET}")
        print(f"  Demand {res.demand:.0f} | sold {res.sales:.0f} | lost {res.lost:.0f} | "
              f"stores out of stock {RED if res.stockout_stores else GREEN}{res.stockout_stores}{RESET}")
        print(f"  Cost {res.total_cost:,.2f} (no transfers {res.noop_cost:,.2f}) -> saving "
              f"{signed(res.noop_cost - res.total_cost)} | running saving {signed(tot['noop'] - tot['cost'])}")
        if delay > 0:
            time.sleep(delay)
        elif delay < 0 and not sim.done:
            input(f"{DIM}Enter = next week{RESET}")

    sub(f"Summary: {APPROACHES[kind]}")
    print(f"  Total cost {tot['cost']:,.2f} vs {tot['noop']:,.2f} with no transfers -> saving "
          f"{signed(tot['noop'] - tot['cost'])}")
    print(f"  Lost sales {tot['lost']:.0f} of {tot['demand']:.0f}; {tot['ships']} shipments, {tot['units']:.0f} units moved")

    from inventory_rl.evaluation.experiment import run_episode

    sub("All approaches on the same demand")
    rows = sorted(((APPROACHES[k], run_episode(sim, policy(k), prod, 0, seed)) for k in kinds),
                  key=lambda x: -x[1]["saving_vs_noop"])
    top = max(abs(r["saving_vs_noop"]) for _, r in rows)
    print_table([[n, signed(r["saving_vs_noop"]), f"{r['lost_units']:.0f}", r["n_transfers"],
                  hbar(r["saving_vs_noop"], top)] for n, r in rows], ["Approach", "Saving", "Lost", "Shipments", ""])


# ===================================================================================== 6. add new data
LEVEL_ANSI = {"error": RED, "warning": AMBER, "info": DIM}


def yes(prompt: str) -> bool:
    return input(f"{prompt} [y/N]: ").strip().lower() in ("y", "yes")


def refresh_plan() -> dict | None:
    """Refresh forecasts, risk and next week's transfers from all uploaded data, printing progress."""
    policies = {name: policy(kind) for name, kind in APPROACH_NAMES.items()}
    try:
        s = live.run_update(load_config(), policies, progress=lambda m: print(f"  {DIM}{m}{RESET}"))
    except Exception as e:
        print(f"{RED}Updating the plan failed: {e}{RESET}")
        return None
    print(f"{GREEN}The plan for the week of {pd.Timestamp(s['next_week']):%d %b %Y} is ready: see Decision Support "
          f"and Store Overview.{RESET}")
    return s


def save_unique(path: Path, text: str) -> Path:
    n = 1
    while path.exists():
        path = path.with_name(f"{path.stem.rstrip('_0123456789')}_{n}{path.suffix}")
        n += 1
    path.write_text(text, encoding="utf-8")
    return path


def column_choices(res, stores: list[str], base_last: pd.Timestamp) -> dict | None:
    """Ask which column holds each required item that was not found (or which store / week the whole file is
    for). Returns keyword arguments for clean_weekly_sheet, or None if the user gives up."""
    sub("Some columns were not recognised")
    print(f"  Columns in the file: {', '.join(res.columns_in_file) or 'none'}")
    if not yes("Choose which column is which?"):
        return None
    out: dict = {"mapping": {}}
    for k in res.missing:
        options = res.columns_in_file + ["(not in the file)"]
        if k == "store":
            options.append("(the whole file is for one store)")
        if k == "week":
            options.append("(the whole file is for one week)")
        pick = options[choose(f"Which column holds '{COLUMNS[k].name}'? ({COLUMNS[k].meaning})", options,
                              len(res.columns_in_file))]
        if pick == "(the whole file is for one store)":
            out["store"] = stores[choose("Which store?", [f"{ex.store_label(s)} ({s})" for s in stores], 0, 4)]
        elif pick == "(the whole file is for one week)":
            weeks = list(pd.date_range(base_last + pd.Timedelta(days=7), pd.Timestamp.today().normalize(),
                                       freq="W-MON"))
            if not weeks:
                print(f"{RED}No week after the data already loaded has started yet.{RESET}")
                return None
            out["week"] = weeks[choose("Which week?", [ex.week_label(w) for w in weeks], len(weeks) - 1)]
        elif pick != "(not in the file)":
            out["mapping"][k] = pick
    return out


def upload_sheet(stores: list[str], products: list[str], base_last: pd.Timestamp) -> None:
    raw_path = input("\nPath of the sheet (CSV, text or Excel; you can drag the file here): ").strip().strip('"\'')
    if not raw_path:
        return
    path = Path(raw_path).expanduser()
    if not path.is_file():
        print(f"{RED}No file found at {path}.{RESET}")
        return
    data = path.read_bytes()
    try:
        raw = read_sheet(data, path.name)
        res = clean_weekly_sheet(raw, stores, products, base_last)
        choices: dict = {"mapping": {}}
        while res.missing and res.columns_in_file:  # ask until every required item is found, or the user stops
            new = column_choices(res, stores, base_last)
            if new is None:
                break
            choices = {**choices, **new, "mapping": {**choices["mapping"], **new["mapping"]}}
            res = clean_weekly_sheet(raw, stores, products, base_last, **choices)
    except Exception as e:  # unreadable file: show why instead of a traceback
        print(f"{RED}Could not read the file: {e}{RESET}")
        return

    w = res.weekly
    header("CHECK BEFORE ADDING")
    print(f"  Rows in the file: {BOLD}{res.rows_read:,}{RESET}   Rows that can be used: {BOLD}{len(w):,}{RESET}   "
          f"Weeks: {BOLD}{w.week_start.nunique() if len(w) else 0}{RESET}   "
          f"Stores: {BOLD}{w.store.nunique() if len(w) else 0}{RESET}")
    if res.issues:
        sub("What was found and fixed")
        for i in res.issues:
            print(LEVEL_ANSI.get(i.level, "") + textwrap.fill(f"- {i.message}", 78, subsequent_indent="  ") + RESET)
    if not res.ok:
        print(f"{RED}Nothing was added.{RESET}")
        return
    for text in live.review_upload(w, base_last, len(stores)):
        print(AMBER + textwrap.fill(f"- {text}", 78, subsequent_indent="  ") + RESET)
    sub(f"Cleaned data (first 10 of {len(w):,} rows)")
    names = da.product_labels(products)
    print_table([[f"{r.week_start:%d %b %Y}", ex.store_label(r.store), names[r.group], f"{r.gross_qty:.0f}",
                  f"{r.return_qty:.0f}", "-" if pd.isna(r.stock_start) else f"{r.stock_start:.0f}",
                  "-" if pd.isna(r.stock_end) else f"{r.stock_end:.0f}"] for r in w.head(10).itertuples()],
                ["Week", "Store", "Product", "Sold", "Returned", "Stock at start", "Stock at end"])
    if not yes("\nAdd this data and update the plan?"):
        print("Nothing was added.")
        return
    replaced = live.save_upload(w, data, path.name)
    print(f"{GREEN}Added {len(w):,} rows for {w.week_start.nunique()} week(s)"
          + (f", replacing {len(replaced)} earlier week(s)" if replaced else "") + f".{RESET}")
    refresh_plan()


def add_data_page() -> None:
    header("ADD NEW DATA")
    note("Add each week's sales and stock from a simple sheet, straight from your till or stock system. The sheet is "
         "checked and cleaned, then the forecasts, stock risk and recommended transfers are updated for the "
         "following week. The original data and the project's results are never changed.")
    panel = live.base_panel()
    stores, products = live.known_codes(panel)
    base_last = pd.Timestamp(panel.week_start.max())
    while True:
        uploads, summ = live.load_uploads(), live_file("summary.json")
        loaded_to = base_last if uploads is None else max(base_last, pd.Timestamp(uploads.week_start.max()))
        sub("Status")
        print(f"  Data runs up to: {BOLD}{ex.week_label(loaded_to)}{RESET}   Weeks you added: "
              f"{BOLD}{0 if uploads is None else uploads.week_start.nunique()}{RESET}   Planning week: "
              f"{BOLD}{ex.week_label(summ['next_week']) if summ else 'not yet'}{RESET}")
        if uploads is not None:
            added = live.added_weeks(uploads)
            print_table([[ex.week_label(r.week_start), r.stores, r.rows, f"{r.units:.0f}", r.stock_given, r.file]
                         for r in added.itertuples()],
                        ["Week", "Stores", "Rows", "Units sold", "Rows with start stock", "From file"])
            if not live.live_ready():
                print(f"{AMBER}The plan has not been updated for the data above (choose 'Update the plan').{RESET}")
        actions = ["Upload a weekly sheet", "How to prepare the sheet", "Save a blank template and the code list"]
        if uploads is not None:
            actions += ["Update the plan again", "Remove all added data"]
        actions.append("Back to the main menu")
        act = actions[choose("What would you like to do?", actions, 0)]
        if act == "Upload a weekly sheet":
            upload_sheet(stores, products, base_last)
        elif act == "How to prepare the sheet":
            sub("What the sheet must contain (CSV, text or Excel; the template is optional)")
            print_table([[c.name, "yes" if c.required else "optional", c.meaning] for c in COLUMNS.values()],
                        ["Item", "Needed", "What to put"])
            note("Column names do not have to match (e.g. Shop, SKU, Qty Sold, SOH, Closing Stock are recognised), "
                 "and columns that are not recognised can be matched by hand after choosing the file. Extra columns "
                 "and title rows are ignored and every sheet of a workbook is read. The weeks, products or stores "
                 "may run across the columns (each cell is then units sold), daily or till rows are added up into "
                 "weeks, and a file with no store or week column can be given one. Sales must be in units, not "
                 "money. A product left out of a store's rows counts "
                 "as 0 sold; a store with no rows in a week counts as not reported. Stock figures make the plan more "
                 f"reliable. Only the {len(stores)} stores and {len(products)} products the models were built on can "
                 "be planned. Uploading a week again replaces the earlier upload of that week.")
        elif act == "Save a blank template and the code list":
            names = da.product_labels(products)
            t = save_unique(Path.cwd() / "weekly_sheet_template.csv",
                            template(stores, products, loaded_to + pd.Timedelta(days=7)).to_csv(index=False))
            codes = pd.concat([
                pd.DataFrame({"Type": "Store", "Code": stores, "Name": [ex.store_label(s) for s in stores]}),
                pd.DataFrame({"Type": "Product", "Code": products, "Name": [names[g] for g in products]})])
            c = save_unique(Path.cwd() / "store_and_product_codes.csv", codes.to_csv(index=False))
            print(f"{GREEN}Saved {t} and {c}.{RESET}")
        elif act == "Update the plan again":
            refresh_plan()
        elif act == "Remove all added data":
            if input("Type 'remove' to delete every sheet you added and the plan built from it: ").strip() == "remove":
                live.clear_uploads()
                print(f"{GREEN}All added data was removed.{RESET}")
            else:
                print("Nothing was removed.")
        else:
            return


# ===================================================================================== menu
PAGES = [("Decision Support", decision_support), ("Store Overview", store_overview), ("Add New Data", add_data_page),
         ("Results", results_page), ("Model Performance", model_performance),
         ("Week-by-week simulation", week_by_week)]


def main() -> None:
    header("STOCK TRANSFER PLANNER (terminal)")
    note("Forecasts, stockout risk and RL transfer recommendations for 40 stores. All transfer results are "
         "simulated from past data; store locations and shipping costs are assumptions.")
    while True:
        print(f"\n{BOLD}Main menu{RESET}")
        for i, (name, _) in enumerate(PAGES, 1):
            print(f"  {i}. {name}")
        print("  0. Exit")
        raw = input("Choose a page: ").strip()
        if raw in ("0", "q", "exit"):
            return
        if raw.isdigit() and 1 <= int(raw) <= len(PAGES):
            PAGES[int(raw) - 1][1]()
            pause()
        else:
            print(f"{RED}Please type a number from 0 to {len(PAGES)}.{RESET}")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print()
