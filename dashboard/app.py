"""Streamlit dashboard: risk-aware inventory redistribution, written for store managers.

Run from the project root:  streamlit run dashboard/app.py
It only READS saved results and trained models (nothing is trained on launch). The "Add new data" page
cleans an uploaded weekly sheet and refreshes forecasts, risk and transfers for the next week with the
already-trained models (inventory_rl.live); the original study's results are never changed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_access as da  # noqa: E402
import explain as ex  # noqa: E402

from inventory_rl.live import update as live  # noqa: E402
from inventory_rl.live.upload import COLUMNS, clean_weekly_sheet, read_sheet, template  # noqa: E402
from inventory_rl.utils.config import load_config  # noqa: E402
from inventory_rl.utils.paths import METRICS_DIR, PROCESSED_DIR, REPORTS_DIR  # noqa: E402

BLUE, DARK, GREY = "#2A6F97", "#1F2933", "#8DA0B3"
# one colour per approach, used by every results chart
APPROACH_COLORS = {"Rule: cover the forecast": "#C3CDD6", "Rule: cover a high estimate": "#E07A1F",
                   "Rule: cover a calibrated high estimate": "#7A8B99", "AI agent": "#5DA9D6",
                   "Uncertainty-aware AI agent": "#1B4F72"}
APPROACH_SCALE = alt.Scale(domain=list(APPROACH_COLORS), range=list(APPROACH_COLORS.values()))

st.set_page_config(page_title="Stock Transfer Planner", page_icon="📦", layout="wide")
st.title("Stock Transfer Planner")
st.caption("Forecasts, stockout risk and RL transfer recommendations for 40 stores.")


# ------------------------------------------------------------------------------------- loaders
@st.cache_data
def table(path: str, version: float = 0.0):
    return da.load_table(Path(path))


@st.cache_data
def js(path: str, version: float = 0.0):
    return da.load_json(Path(path))


def live_file(name: str):
    """A table/summary rebuilt from uploaded data (None until the first upload). Keyed on the file's
    modification time, so a new upload is picked up without a restart."""
    path = live.LIVE_FILES[name]
    if not live.live_ready():
        return None
    version = path.stat().st_mtime
    return js(str(path), version) if path.suffix == ".json" else table(str(path), version)


@st.cache_data
def product_names(groups: tuple[str, ...]) -> dict[str, str]:
    return da.product_labels(list(groups))


@st.cache_resource
def sim_context():
    from inventory_rl.environment.factory import load_context

    return load_context(load_config())


@st.cache_resource
def recommender(kind: str):
    from inventory_rl.baselines.rules import RuleBasedPolicy

    if kind == "ai":
        path = da.best_validation_run("rl_uncertainty")
        if path is not None:
            from inventory_rl.agents.ppo import load_policy

            return load_policy(path, True, load_config()["environment"], "Uncertainty-aware AI agent")
    return RuleBasedPolicy("upper")


APPROACH_KINDS = {"Uncertainty-aware AI agent": "ai", "Rule: cover a high estimate": "rule"}


@st.cache_data
def plan(kind: str, prod: str, week_index: int) -> tuple[np.ndarray, list]:
    """Run the approach from the first test week up to `week_index` on recorded sales (as in the test),
    then return the simulated stock at that week and the transfers recommended for it."""
    ctx = sim_context()
    sim = ctx.sim("test", "replay")
    sim.reset(prod, 0)
    policy = recommender(kind)
    while sim.t < week_index:
        sim.step(policy(sim))
    return sim.stock.copy(), sim.validate(policy(sim))[0]


@st.cache_data
def example_case() -> tuple[int, str, str]:
    """Default selection: the latest week with an AI-recommended transfer, and the destination store and
    product of its largest transfer."""
    ctx = sim_context()
    products = sorted(ctx.tensors["test"])
    n_weeks = ctx.tensors["test"][products[0]].n_weeks
    for wi in range(n_weeks - 1, -1, -1):
        best = None
        for prod in products:
            for _, dst, q in plan("ai", prod, wi)[1]:
                if best is None or q > best[0]:
                    best = (q, ctx.network.stores[dst], prod)
        if best:
            return wi, best[1], best[2]
    return n_weeks - 1, ctx.network.stores[0], products[0]


def need(obj, what: str) -> bool:
    if obj is None:
        st.warning(f"{what} is not available yet. The project pipeline has to be run first.")
        return False
    return True


def explain_line(text: str) -> None:
    st.caption(text)


def risk_box(status: str) -> None:
    color = ex.RISK_COLORS[status]
    st.markdown(
        f"<div style='background:{color}14;border-left:6px solid {color};padding:12px 16px;border-radius:6px;"
        f"margin-bottom:8px'><div style='font-size:1.5rem;font-weight:700;color:{color}'>{status}</div>"
        f"<div style='color:{DARK}'>{ex.RISK_HELP[status]}</div></div>", unsafe_allow_html=True)


def bar(df: pd.DataFrame, value: str, title: str, fmt: str = ",.0f", err: str | None = None) -> alt.Chart:
    base = alt.Chart(df).encode(
        y=alt.Y("approach:N", sort=None, title=None, axis=alt.Axis(labelLimit=260)),
        x=alt.X(f"{value}:Q", title=title),
        tooltip=[alt.Tooltip("approach:N", title="Approach"), alt.Tooltip(f"{value}:Q", title=title, format=fmt)])
    chart = base.mark_bar().encode(color=alt.Color("approach:N", scale=APPROACH_SCALE, legend=None))
    if err:
        chart += base.mark_errorbar(color=DARK).encode(x=alt.X(f"lo_{value}:Q", title=title), x2=f"hi_{value}:Q")
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=DARK).encode(x="x:Q")
    return (chart + zero).properties(height=36 * len(df) + 20)


risk = table(str(PROCESSED_DIR / "risk.parquet"))
fc = table(str(PROCESSED_DIR / "forecasts.parquet"))
summary = table(str(METRICS_DIR / "evaluation_summary.csv"))
results = js(str(METRICS_DIR / "results.json"))

# ===================================================================================== 1. decision support
def outlook_chart(d: pd.DataFrame, week: pd.Timestamp, note: str) -> None:
    """Forecast line, likely-range band and actual sales for one store and product."""
    st.subheader("Demand outlook")
    long = pd.concat([d[["week_start", "yhat"]].rename(columns={"yhat": "units"}).assign(series="Forecast"),
                      d[["week_start", "y"]].rename(columns={"y": "units"}).assign(series="Actual sales")]).dropna()
    band = alt.Chart(d).mark_area(opacity=0.18, color=BLUE).encode(
        x=alt.X("week_start:T", title=None), y=alt.Y("lo:Q", title="Units per week"), y2="hi:Q")
    lines = alt.Chart(long).mark_line(point=True).encode(
        x="week_start:T", y="units:Q",
        color=alt.Color("series:N", scale=alt.Scale(domain=["Forecast", "Actual sales"], range=[BLUE, DARK]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("week_start:T", title="Week"), "series:N",
                 alt.Tooltip("units:Q", format=".1f", title="Units")])
    mark = alt.Chart(pd.DataFrame({"w": [week]})).mark_rule(strokeDash=[4, 4], color=GREY).encode(x="w:T")
    st.altair_chart((band + lines + mark).properties(height=300), width="stretch")
    explain_line(note)


def stock_risk_panel(status: str, stock_now: float, expected: float, lo: float, hi: float, p_out: float,
                     stock_help: str | None = None) -> None:
    st.subheader("Stock risk")
    risk_box(status)
    m1, m2 = st.columns(2)
    m1.metric("In stock", "No record" if np.isnan(stock_now) else f"{stock_now:.0f}", border=True, help=stock_help)
    m2.metric("Expected sales", f"{expected:.1f}", border=True, help=f"Likely range {lo:.0f} to {hi:.0f}")
    st.metric("Chance of running out next week",
              "Unknown" if np.isnan(stock_now) else "Out of stock" if stock_now <= 0 else ex.chance(p_out), border=True)


def show_transfers(valid: list, si: int, store: str, status: str, p_out: float, stock_now: float, expected: float,
                   describe) -> None:
    """The selected store's recommended transfer(s), with the reason, and all transfers for the product."""
    st.subheader("Recommended transfer")
    mine = [describe(s, dd, q) for s, dd, q in valid if si in (s, dd)]
    if mine:
        t = mine[0]
        c = st.columns(5)
        c[0].metric("From", t["From"], border=True)
        c[1].metric("To", t["To"], border=True)
        c[2].metric("Quantity", ex.units(t["Units"]), border=True)
        c[3].metric("Distance", f"{t['Distance (km)']} km", border=True)
        c[4].metric("Estimated cost", ex.money(t["Cost"]), border=True)
        with st.container(border=True):
            st.markdown(f"**Why:** {t['_reason']}")
        if len(mine) > 1:
            st.markdown(f"This store is part of {len(mine)} transfers this week:")
            st.dataframe(pd.DataFrame(mine).drop(columns="_reason"), hide_index=True, width="stretch")
    else:
        with st.container(border=True):
            st.markdown(f"**No transfer for {ex.store_label(store)}.** "
                        + ex.no_transfer_reason(ex.store_label(store), status, p_out,
                                                0.0 if np.isnan(stock_now) else stock_now, expected))
    if valid:
        with st.expander(f"All transfers suggested for this product this week ({len(valid)})"):
            st.dataframe(pd.DataFrame([describe(*v) for v in valid]).drop(columns="_reason"),
                         hide_index=True, width="stretch")


def approach_radio() -> str:
    return st.radio("Recommendation approach", list(APPROACH_KINDS), horizontal=True,
                    help="The AI agent learned when to move stock from past data. The rule tops every store "
                         "up to a high estimate of next week's demand. The Results page compares both.")


def decision_support() -> None:
    st.header("Decision support")
    if live.live_ready():
        summ = live_file("summary.json")
        mode = st.radio("Data", [f"Next week, from your added data ({ex.week_label(summ['next_week'])})",
                                 "Test weeks (project evaluation)"], horizontal=True,
                        help="Your added data: forecasts and transfers for the week after the last week you uploaded. "
                             "Test weeks: the 8 held-back weeks used to evaluate the project.")
        if mode.startswith("Next week"):
            decision_support_live()
            return
    decision_support_test()


def decision_support_test() -> None:
    explain_line("Choose a week, a store and a product to see the demand outlook, the stock risk and the "
                 "recommended transfer.")
    if need(risk, "Risk information") and need(fc, "Demand forecasts"):
        from inventory_rl.risk.assessment import assess
        from inventory_rl.routing.network import transfer_cost

        ctx = sim_context()
        cfg = load_config()
        T = ctx.tensors["test"]
        products = sorted(T)
        stores = list(ctx.network.stores)
        names = product_names(tuple(products))
        ex_week, ex_store, ex_prod = example_case()
        c1, c2, c3 = st.columns(3)
        weeks = T[products[0]].weeks
        week_opts = [ex.week_label(w) for w in weeks]
        wi = week_opts.index(c1.selectbox("Planning week", week_opts, index=ex_week))
        store_opts = {ex.store_label(s): s for s in stores}
        store = store_opts[c2.selectbox("Store", list(store_opts), index=stores.index(ex_store))]
        prod_opts = {names[g]: g for g in products}
        prod = prod_opts[c3.selectbox("Product", list(prod_opts), index=products.index(ex_prod))]
        method = approach_radio()
        week = pd.Timestamp(weeks[wi])
        si = stores.index(store)
        P = T[prod]
        stock, valid = plan(APPROACH_KINDS[method], prod, wi)
        rk = assess(pd.DataFrame({"opening_stock": stock, "yhat": P.yhat[wi], "hi": P.hi[wi]}),
                    cfg["uncertainty"]["alpha"], cfg["environment"]["excess_horizon_weeks"])

        left, right = st.columns([2, 1])
        with left:
            d = fc[(fc.store == store) & (fc.group == prod) & (fc.week_start <= week)].sort_values("week_start").tail(16)
            outlook_chart(d, week, "Blue line: forecast made one week ahead. Shaded band: likely range of sales. "
                                   "Dashed line: the planning week (its actual sales are shown for comparison).")
        r = rk.iloc[si]
        status = ex.risk_status(r.risk_class)
        p_out, stock_now, expected = float(r.p_stockout), float(r.opening_stock), float(P.yhat[wi][si])
        with right:
            stock_risk_panel(status, stock_now, expected, P.lo[wi][si], P.hi[wi][si], p_out)

        def describe(s: int, dd: int, q: int) -> dict:
            km = float(ctx.network.distance_km[s, dd])
            cost = float(transfer_cost(q, km, ctx.sim_cfg.transfer_fixed, ctx.sim_cfg.transfer_per_unit_km))
            return {"From": ex.store_label(stores[s]), "To": ex.store_label(stores[dd]), "Units": q,
                    "Distance (km)": round(km), "Cost": round(cost, 2), "_reason": ex.transfer_reason(
                        ex.store_label(stores[s]), ex.store_label(stores[dd]), q, km, cost, stock[s],
                        P.yhat[wi][s], stock[dd], P.yhat[wi][dd], float(rk.p_stockout.iloc[dd]))}

        show_transfers(valid, si, store, status, p_out, stock_now, expected, describe)
        explain_line(f"Stock levels here are simulated as if the chosen approach had been in use since the "
                     f"week of {pd.Timestamp(weeks[0]):%d %b %Y}, using the sales that actually happened, so they can differ "
                     f"from the recorded stock in Store Overview. Stores without a stock count at that start are "
                     f"treated as empty. Distances come from a simulated store map, because the data has no store "
                     f"locations. Costs use the project's assumed shipping costs, in the same currency as the sales "
                     f"data. A store always keeps enough stock for its own expected sales before sending any.")


def decision_support_live() -> None:
    """Next week's plan built from the uploaded weekly sheets (inventory_rl.live.update)."""
    lrisk, lfc, ltr, summ = (live_file("risk.parquet"), live_file("forecasts.parquet"), live_file("transfers.csv"),
                             live_file("summary.json"))
    nxt = lrisk[lrisk.split == "next"]
    week = pd.Timestamp(summ["next_week"])
    stores = list(sim_context().network.stores)
    products = sorted(nxt.group.unique())
    names = product_names(tuple(products))
    explain_line(f"Plan for the week of {week:%d %b %Y}, using the data you added up to the week of "
                 f"{pd.Timestamp(summ['uploaded_weeks'][-1]):%d %b %Y} (updated {summ['updated_at'].replace('T', ' ')}).")

    method = approach_radio()
    tr = ltr[ltr.approach == method]
    if len(tr):  # default selection: the largest recommended shipment
        top = tr.sort_values("units", ascending=False).iloc[0]
        ex_store, ex_prod = top.to_store, top.group
    else:  # otherwise the store and product expected to be short of the most units
        top = nxt.sort_values(["expected_shortage", "yhat"], ascending=False).iloc[0]
        ex_store, ex_prod = top.store, top.group
    c1, c2, c3 = st.columns(3)
    c1.selectbox("Planning week", [f"{ex.week_label(week)} (next week)"], disabled=True)
    store_opts = {ex.store_label(s): s for s in stores}
    store = store_opts[c2.selectbox("Store", list(store_opts), index=stores.index(ex_store), key="live_store")]
    prod_opts = {names[g]: g for g in products}
    prod = prod_opts[c3.selectbox("Product", list(prod_opts), index=products.index(ex_prod), key="live_prod")]
    si = stores.index(store)
    rows = nxt[nxt.group == prod].set_index("store").reindex(stores)

    left, right = st.columns([2, 1])
    with left:
        cols = ["week_start", "yhat", "y", "lo", "hi"]
        hist = pd.concat([fc.loc[(fc.store == store) & (fc.group == prod), cols],
                          lfc.loc[(lfc.store == store) & (lfc.group == prod), cols]]).sort_values("week_start").tail(16)
        outlook_chart(hist, week, "Blue line: forecast made one week ahead. Shaded band: likely range of sales. "
                                  "Dashed line: next week, which has no sales yet.")
    r = rows.loc[store]
    status = ex.risk_status(r.risk_class)
    stock_now, p_out, expected = float(r.opening_stock), float(r.p_stockout), float(r.yhat)
    with right:
        stock_risk_panel(status, stock_now, expected, r.lo, r.hi, p_out,
                         stock_help="Estimated: your sheets did not give this stock, so it is the last known stock "
                                    "minus units sold since, assuming no deliveries." if r.stock_estimated else None)
        if r.stock_estimated:
            st.caption("This stock figure is an estimate: your sheet had no stock figure for it.")

    stock = rows.opening_stock.to_numpy(float)
    p_all = rows.p_stockout.to_numpy(float)
    yhat = rows.yhat.to_numpy(float)
    pick = tr[tr.group == prod]
    valid = [(stores.index(a), stores.index(b), int(q)) for a, b, q in zip(pick.from_store, pick.to_store, pick.units)]
    costs = {(stores.index(a), stores.index(b)): (km, c) for a, b, km, c in zip(pick.from_store, pick.to_store,
                                                                                pick.km, pick.cost)}

    def describe(s: int, dd: int, q: int) -> dict:
        km, cost = costs[(s, dd)]
        no_rec = bool(np.isnan(stock[dd]))
        return {"From": ex.store_label(stores[s]), "To": ex.store_label(stores[dd]), "Units": q,
                "Distance (km)": round(km), "Cost": round(cost, 2), "_reason": ex.transfer_reason(
                    ex.store_label(stores[s]), ex.store_label(stores[dd]), q, km, cost, stock[s], yhat[s],
                    0.0 if no_rec else stock[dd], yhat[dd], 1.0 if no_rec else float(p_all[dd]), dst_no_record=no_rec)}

    show_transfers(valid, si, store, status, p_out, stock_now, expected, describe)
    explain_line("Stock is the stock at the start of next week from your sheets (the end-of-week stock of the last "
                 "uploaded week). Where a sheet left it out, it is estimated; stores with no stock figure at all are "
                 "treated as empty when planning. Forecasts use the model settings chosen in the original study, and "
                 "the AI agent is the one trained in the study. Distances come from a simulated store map and costs "
                 "use the project's assumed shipping costs.")

# ===================================================================================== 2. store overview
def store_overview() -> None:
    st.header("Store overview")
    explain_line("Which stores need attention in the chosen week. Red = may run out, amber = too much stock, "
                 "green = normal, grey = no stock count in the data.")
    if need(risk, "Risk information"):
        test = risk[risk.split == "test"]
        lrisk = live_file("risk.parquet")
        if lrisk is not None:  # weeks from uploaded sheets, and the next week to plan for
            test = pd.concat([test, lrisk[test.columns]], ignore_index=True)
        test = test.assign(status=lambda d: d.risk_class.map(ex.risk_status))
        weeks = sorted(test.week_start.unique())
        prods = sorted(test.group.unique())
        names = product_names(tuple(prods))
        c1, c2 = st.columns(2)
        tag = test.drop_duplicates("week_start").set_index("week_start").split.map(
            {"new": " (your data)", "next": " (next week)"}).fillna("")
        week_opts = {ex.week_label(w) + tag[w]: w for w in weeks}
        wk = week_opts[c1.selectbox("Week", list(week_opts), index=len(weeks) - 1, key="ov_week")]
        prod_opts = {"All products": "All products", **{names[g]: g for g in prods}}
        pick = prod_opts[c2.selectbox("Product", list(prod_opts), key="ov_prod")]
        d = test[test.week_start == wk]
        if pick != "All products":
            d = d[d.group == pick]
        n_stores = d.store.nunique()
        c = st.columns(4)
        c[0].metric("Stores with a stockout risk", f"{d[d.status == 'Stockout risk'].store.nunique()} of {n_stores}",
                    border=True)
        c[1].metric("Store-product pairs at risk", int((d.status == "Stockout risk").sum()), border=True)
        c[2].metric("Pairs with excess stock", int((d.status == "Excess stock").sum()), border=True)
        c[3].metric("Pairs with no stock count", int((d.status == "No stock record").sum()), border=True)
        color = alt.Color("status:N", title=None, scale=alt.Scale(domain=ex.RISK_ORDER, range=list(ex.RISK_COLORS.values())),
                          legend=alt.Legend(orient="top"))
        d = d.assign(store_name=d.store.map(ex.store_label))
        if pick == "All products":
            counts = d.groupby(["store_name", "status"]).size().rename("products").reset_index()
            order = (counts[counts.status == "Stockout risk"].sort_values("products", ascending=False).store_name.tolist())
            order += [s for s in sorted(counts.store_name.unique()) if s not in order]
            chart = alt.Chart(counts).mark_bar().encode(
                y=alt.Y("store_name:N", sort=order, title=None, axis=alt.Axis(labelOverlap=False)), x=alt.X("products:Q", title="Number of products"),
                color=color, order=alt.Order("status_rank:Q"),
                tooltip=[alt.Tooltip("store_name:N", title="Store"), alt.Tooltip("status:N", title="Status"),
                         alt.Tooltip("products:Q", title="Products")],
            ).transform_calculate(status_rank=f"indexof({ex.RISK_ORDER}, datum.status)")
            st.altair_chart(chart.properties(height=22 * len(order) + 40), width="stretch")
            explain_line("Each bar is one store, split by the status of its products. Stores with the most products "
                         "at risk of running out are at the top.")
        else:
            d = d.sort_values("store").reset_index(drop=True)
            d = d.assign(col=d.index % 8, row=d.index // 8, num=d.store.str.split("-").str[-1],
                         stock_txt=d.opening_stock.map(lambda v: "no record" if pd.isna(v) else f"{v:.0f}"),
                         chance_txt=d.p_stockout.map(lambda v: "unknown" if pd.isna(v) else ex.chance(v)))
            base = alt.Chart(d).encode(x=alt.X("col:O", axis=None), y=alt.Y("row:O", axis=None))
            tiles = base.mark_rect(cornerRadius=6, stroke="white", strokeWidth=3).encode(
                color=color, tooltip=[alt.Tooltip("store_name:N", title="Store"), alt.Tooltip("status:N", title="Status"),
                                      alt.Tooltip("stock_txt:N", title="In stock"),
                                      alt.Tooltip("yhat:Q", title="Expected sales", format=".1f"),
                                      alt.Tooltip("chance_txt:N", title="Chance of running out")])
            text = base.mark_text(color="white", fontWeight="bold").encode(text="num:N")
            st.altair_chart((tiles + text).properties(height=70 * (d.row.max() + 1)), width="stretch")
            explain_line("Each tile is a store (number shown). Hover over a tile to see its stock, expected sales "
                         "and chance of running out.")

# ===================================================================================== 3. add new data
@st.cache_data
def base_info() -> tuple[list[str], list[str], pd.Timestamp]:
    panel = live.base_panel()
    stores, products = live.known_codes(panel)
    return stores, products, pd.Timestamp(panel.week_start.max())


def run_live_update(status) -> dict:
    """Refresh forecasts, risk and next week's transfers from all uploaded data, reporting progress."""
    policies = {name: recommender(kind) for name, kind in APPROACH_KINDS.items()}
    return live.run_update(load_config(), policies, progress=status.write)


def add_data_page() -> None:
    st.header("Add new data")
    explain_line("Upload each week's sales and stock as a simple sheet, straight from your till or stock system. "
                 "The sheet is checked and cleaned here, then the forecasts, stock risk and recommended transfers "
                 "are updated for the following week.")
    stores, products, base_last = base_info()
    uploads = live.load_uploads()
    summ = live_file("summary.json")
    loaded_to = base_last if uploads is None else max(base_last, pd.Timestamp(uploads.week_start.max()))
    if msg := st.session_state.pop("upload_message", None):
        st.success(msg)

    c = st.columns(3)
    c[0].metric("Data runs up to", ex.week_label(loaded_to), border=True,
                help=f"The original data ends with the week of {base_last:%d %b %Y}.")
    c[1].metric("Weeks you added", 0 if uploads is None else uploads.week_start.nunique(), border=True)
    c[2].metric("Planning week", ex.week_label(summ["next_week"]) if summ else "Not yet", border=True)

    with st.expander("How to prepare the sheet", expanded=uploads is None):
        st.markdown("One row per **store, product and week**. CSV or Excel. Column names do not have to match "
                    "exactly; common alternatives are recognised, extra columns are ignored, and title rows above "
                    "the table are skipped.")
        st.dataframe(pd.DataFrame([{"Column": c.name, "Needed": "Yes" if c.required else "Optional",
                                    "What to put": c.meaning,
                                    "Also recognised": ", ".join(a for a in c.aliases[1:6])}
                                   for c in COLUMNS.values()]), hide_index=True, width="stretch")
        st.markdown(
            "- A product with no sales in a store can be left out, as long as the store has other rows that week: it "
            "is then counted as 0 sold.\n"
            "- A store with no rows at all in a week is treated as *not reported* (sales unknown), not as zero sales.\n"
            "- Stock figures make the plan much more reliable. Without them, stock is estimated from the last known "
            "stock minus units sold, assuming no deliveries.\n"
            f"- Only the {len(stores)} stores and {len(products)} products the models were built on can be planned. "
            "Other rows are reported and skipped.\n"
            "- Uploading a week again replaces the earlier upload of that week, so a corrected sheet can be re-uploaded.")
        names = product_names(tuple(products))
        d1, d2 = st.columns(2)
        d1.download_button("Download a blank template (CSV)",
                           template(stores, products, loaded_to + pd.Timedelta(days=7)).to_csv(index=False),
                           "weekly_sheet_template.csv", "text/csv", width="stretch")
        codes = pd.concat([pd.DataFrame({"Type": "Store", "Code": stores, "Name": [ex.store_label(s) for s in stores]}),
                           pd.DataFrame({"Type": "Product", "Code": products, "Name": [names[g] for g in products]})])
        d2.download_button("Download the store and product codes (CSV)", codes.to_csv(index=False),
                           "store_and_product_codes.csv", "text/csv", width="stretch")

    key = f"uploader_{st.session_state.get('upload_round', 0)}"
    file = st.file_uploader("Weekly sheet", type=["csv", "txt", "xlsx", "xlsm"], key=key)
    if file is not None:
        data = file.getvalue()
        try:
            raw = read_sheet(data, file.name)
        except Exception as e:  # unreadable file: show why instead of a traceback
            st.error(f"Could not read the file: {e}")
            return
        res = clean_weekly_sheet(raw, stores, products, base_last)

        st.subheader("Check before adding")
        w = res.weekly
        c = st.columns(4)
        c[0].metric("Rows in the file", f"{res.rows_read:,}", border=True)
        c[1].metric("Rows that can be used", f"{len(w):,}", border=True)
        c[2].metric("Weeks", w.week_start.nunique() if len(w) else 0, border=True)
        c[3].metric("Stores", w.store.nunique() if len(w) else 0, border=True)
        for i in res.issues:
            {"error": st.error, "warning": st.warning}.get(i.level, st.info)(i.message)
        if res.ok:
            new_weeks = sorted(w.week_start.unique())
            for text in live.review_upload(w, base_last, len(stores)):
                st.warning(text)
            preview = w.assign(store=w.store.map(ex.store_label), group=w.group.map(names),
                               week_start=w.week_start.dt.strftime("%d %b %Y")).rename(columns={
                "store": "Store", "group": "Product", "week_start": "Week", "gross_qty": "Units sold",
                "return_qty": "Units returned", "promo_qty": "Units on promotion",
                "stock_start": "Stock at start", "stock_end": "Stock at end"})
            with st.expander(f"Cleaned data ({len(w):,} rows)"):
                st.dataframe(preview, hide_index=True, width="stretch", height=300)
            if st.button("Add this data and update the plan", type="primary"):
                replaced = live.save_upload(w, data, file.name)
                with st.status("Updating forecasts and recommendations...", expanded=True) as status:
                    try:
                        s = run_live_update(status)
                    except Exception as e:
                        status.update(label="The update failed", state="error")
                        st.error(f"The data was saved, but updating the plan failed: {e}")
                        return
                    status.update(label="Updated", state="complete")
                st.cache_data.clear()
                st.session_state["upload_round"] = st.session_state.get("upload_round", 0) + 1
                st.session_state["upload_message"] = (
                    f"Added {len(w):,} rows for {len(new_weeks)} week(s)"
                    + (f" (replacing {len(replaced)} earlier week(s))" if replaced else "")
                    + f". The plan for the week of {pd.Timestamp(s['next_week']):%d %b %Y} is ready on the Decision Support "
                      f"and Store Overview pages.")
                st.rerun()
        return

    if uploads is not None:
        st.subheader("Data you have added")
        added = live.added_weeks(uploads)
        st.dataframe(added.assign(week_start=added.week_start.map(ex.week_label)).rename(columns={
            "week_start": "Week", "stores": "Stores", "rows": "Rows", "units": "Units sold",
            "stock_given": "Rows with start stock", "file": "From file"}), hide_index=True, width="stretch")
        if not live.live_ready():
            st.warning("The plan has not been updated for the data above.")
        if st.button("Update the plan again"):
            with st.status("Updating forecasts and recommendations...", expanded=True) as status:
                run_live_update(status)
                status.update(label="Updated", state="complete")
            st.cache_data.clear()
            st.rerun()
        with st.expander("Remove added data"):
            st.markdown("Deletes every sheet you added and the plan built from it. The original data and the "
                        "project's results are not affected.")
            if st.checkbox("Yes, remove all the data I added") and st.button("Remove"):
                live.clear_uploads()
                st.cache_data.clear()
                st.session_state["upload_message"] = "All added data was removed."
                st.rerun()

# ===================================================================================== 4. results
def results_page() -> None:
    st.header("Results")
    explain_line("How each approach performed on 8 recent weeks that were held back for testing, compared with "
                 "making no transfers at all.")
    if need(summary, "Test results"):
        scen_label = st.radio("Demand used in the test", ["Simulated demand (20 runs)", "Actual past sales"],
                              horizontal=True,
                              help="Simulated demand varies around past sales. Actual past sales understate demand "
                                   "in weeks when a store had sold out.")
        scen = "poisson" if scen_label.startswith("Simulated") else "replay"
        s = summary[summary.scenario == scen].set_index("policy")
        base = s.loc["No redistribution"]
        df = pd.DataFrame({
            "approach": [ex.POLICY_NAMES[p] for p in s.index],
            "lost_sales_avoided": [-ex.pct_change(s.loc[p, "lost_units"], base.lost_units) for p in s.index],
            "leftover_change": [ex.pct_change(s.loc[p, "avg_closing_stock"], base.avg_closing_stock) for p in s.index],
            "shipments": s.n_transfers.values, "transfer_spend": s.transfer_cost.values,
            "money_saved": s.saving_vs_noop.values}, index=s.index)
        seeds = table(str(METRICS_DIR / "rl_saving_across_training_seeds.csv"))
        df["lo_money_saved"] = df.money_saved
        df["hi_money_saved"] = df.money_saved
        if seeds is not None:
            for _, r in seeds[seeds.scenario == scen].iterrows():
                if r.policy in df.index:
                    df.loc[r.policy, ["lo_money_saved", "hi_money_saved"]] = [r["mean"] - r["std"], r["mean"] + r["std"]]
        show = df.drop(index="No redistribution")
        ua = show.loc["RL (forecast + uncertainty)"]
        c = st.columns(4)
        c[0].metric("Lost sales avoided", f"{ua.lost_sales_avoided:.0f}%", border=True,
                    help="Uncertainty-aware AI agent, compared with no transfers")
        c[1].metric("Change in leftover stock", f"{ua.leftover_change:+.1f}%", border=True)
        c[2].metric("Shipments in 8 weeks", f"{ua.shipments:.0f}", border=True)
        c[3].metric("Money saved", ex.money(ua.money_saved), border=True)
        explain_line("The four cards show the uncertainty-aware AI agent. Money saved = cost of no transfers minus "
                     "(lost-sale cost + holding cost + shipping cost), over 30 products and 40 stores.")

        best = ex.best_policy(dict(zip(show.approach, show.money_saved)))
        point_ai = show.loc["RL (point forecast)"]
        losers = show[show.money_saved < 0].approach.tolist()
        takeaway = (f"**Takeaway:** in this test, **{best}** saved the most money. The uncertainty-aware AI agent "
                    f"saved {ex.money(ua.money_saved)}, {'more' if ua.money_saved > point_ai.money_saved else 'less'} "
                    f"than the AI agent without uncertainty ({ex.money(point_ai.money_saved)}), using "
                    f"{ua.shipments:.0f} shipments.")
        if losers:
            takeaway += f" {' and '.join(losers)} spent more on shipping than it saved."
        cmp_ = (results or {}).get("rl_seed_level_comparison", {}).get(
            f"RL (forecast + uncertainty) minus RL (point forecast) | {scen}")
        if cmp_:
            takeaway += (" The gap between the two AI agents is within their run-to-run variation, so it is not a "
                         "reliable difference." if cmp_["p_value"] >= 0.05 else
                         " The gap between the two AI agents is larger than their run-to-run variation.")
        with st.container(border=True):
            st.markdown(takeaway)

        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**Fewer stockouts**")
            st.altair_chart(bar(show, "lost_sales_avoided", "Lost sales avoided (%)", ".0f"), width="stretch")
            explain_line("Share of lost sales (customers who found the shelf empty) that each approach prevented.")
        with g2:
            st.markdown("**Leftover stock**")
            st.altair_chart(bar(show, "leftover_change", "Change in average leftover stock (%)", ".1f"), width="stretch")
            explain_line("Transfers move stock between stores rather than removing it, so total leftover stock "
                         "changes only slightly.")
        g3, g4 = st.columns(2)
        with g3:
            st.markdown("**Shipments made**")
            st.altair_chart(bar(show, "shipments", "Shipments in 8 weeks", ",.0f"), width="stretch")
            explain_line("Fewer shipments for a similar saving means fewer unnecessary moves. The data does not mark "
                         "individual shipments as necessary or not.")
        with g4:
            st.markdown("**Money saved**")
            st.altair_chart(bar(show, "money_saved", "Money saved vs no transfers", ",.0f", err="money_saved"),
                            width="stretch")
            explain_line("Black lines on the AI bars show the spread over 5 separately trained agents.")

        st.subheader("What if the cost assumptions are different?")
        explain_line("Shipping and lost-sale costs are not in the data, so the project assumes them. Here the test is "
                     "repeated with one assumed cost changed at a time.")
        sens = table(str(METRICS_DIR / "cost_sensitivity.csv"))
        if need(sens, "The cost what-if analysis"):
            costs = load_config()["costs"]
            sens = sens[sens.scenario == scen].copy()
            lines_txt = []
            for param in ex.COST_PARAM_NAMES:
                sub = sens[(sens.parameter == param) | (sens.parameter == "base")].assign(
                    value=lambda d, p=param: d.value.fillna(float(costs[p])))
                lo, hi = sub.value.min(), sub.value.max()
                pick_lo = ex.best_policy({ex.POLICY_NAMES[p]: v for p, v in
                                          sub[sub.value == lo].set_index("policy").saving_mean.items()})
                pick_hi = ex.best_policy({ex.POLICY_NAMES[p]: v for p, v in
                                          sub[sub.value == hi].set_index("policy").saving_mean.items()})
                lines_txt.append(f"- **{ex.COST_PARAM_NAMES[param]}** (assumed {costs[param]:g}): at {lo:g} "
                                 f"**{pick_lo}** saves the most; at {hi:g} **{pick_hi}** "
                                 f"{'is cheapest' if pick_hi == 'No transfers' else 'saves the most'}.")
            st.markdown("\n".join(lines_txt))
            param_opts = {v: k for k, v in ex.COST_PARAM_NAMES.items()}
            choice = param_opts[st.selectbox("Show the effect of", list(param_opts))]
            sub = sens[(sens.parameter == choice) | (sens.parameter == "base")].assign(
                value=lambda d: d.value.fillna(float(costs[choice])), approach=lambda d: d.policy.map(ex.POLICY_NAMES))
            ticks = sorted(sub.value.unique())
            chart = alt.Chart(sub).mark_line(point=True).encode(
                x=alt.X("value:Q", title=ex.COST_PARAM_NAMES[choice],
                        scale=alt.Scale(type="log", domain=[ticks[0], ticks[-1]], nice=False),
                        axis=alt.Axis(values=ticks, format="~g")),
                y=alt.Y("saving_mean:Q", title="Money saved vs no transfers"),
                color=alt.Color("approach:N", title=None, scale=APPROACH_SCALE,
                                legend=alt.Legend(orient="top", columns=3, labelLimit=320)),
                tooltip=[alt.Tooltip("approach:N", title="Approach"), alt.Tooltip("value:Q", title="Cost value"),
                         alt.Tooltip("saving_mean:Q", title="Money saved", format=",.0f")])
            zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=DARK).encode(y="y:Q")
            st.altair_chart((chart + zero).properties(height=320), width="stretch")
            explain_line("Below the black line an approach loses money. The AI agents were trained once with the "
                         "assumed costs and were not retrained for each what-if value.")

# ===================================================================================== 5. model performance
def model_performance() -> None:
    st.header("Model performance")
    explain_line("Data quality, model accuracy and how the results were produced, for readers who want more detail.")

    st.subheader("Pipeline status")
    status = da.artifact_status()
    steps = {"Data audit": "Data checks", "Weekly panel": "Weekly sales and stock table",
             "Forecasts + intervals": "Forecasts and likely ranges", "Risk table": "Risk scores",
             "Store network": "Store distances", "RL checkpoints": "Trained AI agents",
             "Evaluation results": "Test results"}
    step = status.stage.str.replace(r"\s*\(.*\)", "", regex=True)
    st.dataframe(pd.DataFrame({"Step": step.map(steps).fillna(step),
                               "Status": status.available.map({True: "Ready", False: "Not run yet"})}),
                 hide_index=True, width="content")

    audit = js(str(REPORTS_DIR / "data_audit.json"))
    prep = js(str(REPORTS_DIR / "preprocessing_report.json"))
    if audit and prep:
        st.subheader("Data quality")
        m, q = audit["mendeley"], prep["quality"]
        rows = [
            ("Sales transactions", f"{m['sales']['rows']:,}"),
            ("Stock records", f"{m['inventory']['rows']:,}"),
            ("Stores / products in the raw data", f"{m['sales']['n_stores']} / {m['sales']['n_products']:,}"),
            ("Sales period", " to ".join(x[:10] for x in m["sales"]["transaction_date_range"])),
            ("Return transactions", f"{m['sales']['return_rows']:,}"),
            ("Stock records with negative quantity", f"{m['inventory']['negative_qty_rows']:,}"),
            ("Weekly series used (store x product)", f"{q['series']:,} ({q['stores']} stores x {q['groups']} products, "
                                                    f"{q['weeks']} weeks)"),
            ("Store-weeks with no sales", f"{100 * q['zero_sales_share']:.0f}%"),
            ("Store-weeks with no stock count at the start", f"{100 * q['opening_stock_missing_share']:.0f}%"),
            ("Store-weeks starting with zero stock", f"{100 * q['stockout_weeks_share (opening_stock==0)']:.1f}%"),
            ("Duplicate records", f"{q['duplicate_keys']}"),
        ]
        st.dataframe(pd.DataFrame(rows, columns=["Measure", "Value"]), hide_index=True, width="stretch")
        explain_line("Two other public datasets were checked and not used: one was internally inconsistent and the "
                     "other looked randomly generated.")

    fm = js(str(METRICS_DIR / "forecast_metrics.json"))
    if fm:
        st.subheader("Forecast accuracy (test weeks)")
        names = {"LightGBM": "Machine-learning model (used)", "Naive (t-1)": "Same as last week",
                 "MA(4)": "Average of last 4 weeks", "MA(8)": "Average of last 8 weeks",
                 "Zero forecast": "Always zero"}
        acc = pd.DataFrame([{"Method": v, "Average error (units)": fm["test"][k]["MAE"],
                             "Large-error score (RMSE)": fm["test"][k]["RMSE"], "Bias (units)": fm["test"][k]["Bias"]}
                            for k, v in names.items() if k in fm["test"]])
        st.dataframe(acc.round(3), hide_index=True, width="stretch")
        explain_line("Most store-weeks sell nothing, so 'always zero' has a small average error but misses every "
                     "sale; the large-error score is the fairer comparison.")

    im = js(str(METRICS_DIR / "interval_metrics.json"))
    if im:
        st.subheader("Likely-range accuracy (test weeks)")
        labels = {"poisson": "Count-based range", "qr": "Quantile model", "cqr": "Calibrated quantile model",
                  "conformal": "Error-based range"}
        rng = pd.DataFrame([{"Method": labels.get(k, k) + (" (used)" if k == im["chosen"] else ""),
                             "Sales inside the range": f"{100 * v['coverage']:.0f}%",
                             "Target": f"{100 * v['nominal']:.0f}%", "Average width (units)": round(v["mean_width"], 2)}
                            for k, v in im["by_split"]["test"].items()])
        st.dataframe(rng, hide_index=True, width="stretch")
        explain_line("Weekly sales are small whole numbers, so ranges often cover more than their target. The method "
                     "was chosen on earlier weeks, not on the test weeks.")

    per_run = table(str(METRICS_DIR / "rl_saving_per_training_seed.csv"))
    train = js(str(da.MODELS_DIR / "rl" / "training_summary.json"))
    if per_run is not None and train:
        st.subheader("AI agent training runs")
        val = {(r["variant"], r["seed"]): r["best_val_saving_poisson"] for r in train["runs"]}
        key = {"RL (point forecast)": "rl_point", "RL (forecast + uncertainty)": "rl_uncertainty"}
        runs = per_run.assign(**{
            "Agent": per_run.policy.map(ex.POLICY_NAMES),
            "Run": per_run.train_seed + 1,
            "Saving on earlier check weeks": [val.get((key[p], s)) for p, s in zip(per_run.policy, per_run.train_seed)],
            "Saving on test weeks (simulated demand)": per_run.poisson,
            "Saving on test weeks (actual sales)": per_run.replay})
        st.dataframe(runs[["Agent", "Run", "Saving on earlier check weeks", "Saving on test weeks (simulated demand)",
                           "Saving on test weeks (actual sales)"]].round(0), hide_index=True, width="stretch")
        explain_line("Each agent was trained 5 times from different random starting points. The version kept from "
                     "each run was chosen on the earlier check weeks, never on the test weeks.")

    st.subheader("Cost assumptions")
    costs = load_config()["costs"]
    st.dataframe(pd.DataFrame([
        ("Fixed cost per shipment", f"{costs['transfer_fixed']:g}"),
        ("Cost per item per km", f"{costs['transfer_per_unit_km']:g}"),
        ("Yearly holding cost (share of item cost)", f"{costs['holding_rate_per_year']:g}"),
        ("Cost of a lost sale (times profit per item)", f"{costs['shortage_cost_multiplier']:g}"),
        ("Profit and cost per item", "Taken from the sales data"),
        ("Store distances", "Simulated map (no locations in the data)"),
    ], columns=["Assumption", "Value"]), hide_index=True, width="stretch")

    cal = table(str(METRICS_DIR / "stockout_calibration_test.csv"))
    if cal is not None:
        st.subheader("Is the chance of running out trustworthy?")
        cal = cal.assign(predicted_pct=100 * cal.predicted, observed_pct=100 * cal.observed)
        diag = alt.Chart(pd.DataFrame({"x": [0, 100], "y": [0, 100]})).mark_line(
            strokeDash=[4, 4], color=GREY).encode(x="x:Q", y="y:Q")
        pts = alt.Chart(cal).mark_line(point=alt.OverlayMarkDef(size=80), color=BLUE).encode(
            x=alt.X("predicted_pct:Q", title="Predicted chance of running out (%)", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("observed_pct:Q", title="Share that actually sold out (%)", scale=alt.Scale(domain=[0, 100])),
            tooltip=[alt.Tooltip("predicted_pct:Q", title="Predicted (%)", format=".0f"),
                     alt.Tooltip("observed_pct:Q", title="Actually sold out (%)", format=".0f"),
                     alt.Tooltip("n:Q", title="Store-weeks")])
        st.altair_chart((diag + pts).properties(height=320), width="stretch")
        explain_line("Test weeks with stock on hand, grouped by predicted chance. Points on the dashed line mean the "
                     "predicted chance matched reality. Few store-weeks have a predicted chance above 50%, so the "
                     "right-hand points rest on very little data (hover to see counts).")


st.navigation([st.Page(decision_support, title="Decision Support", default=True),
               st.Page(store_overview, title="Store Overview"),
               st.Page(add_data_page, title="Add New Data", url_path="add-data"),
               st.Page(results_page, title="Results", url_path="results"),
               st.Page(model_performance, title="Model Performance")]).run()
