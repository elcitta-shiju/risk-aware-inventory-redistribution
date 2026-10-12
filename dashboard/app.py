"""Streamlit dashboard: risk-aware inventory redistribution, written for store managers.

Run from the project root:  streamlit run dashboard/app.py
It only READS saved results and trained models (nothing is trained on launch).
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
def table(path: str):
    return da.load_table(Path(path))


@st.cache_data
def js(path: str):
    return da.load_json(Path(path))


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
def decision_support() -> None:
    st.header("Decision support")
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
        method = st.radio("Recommendation approach", ["Uncertainty-aware AI agent", "Rule: cover a high estimate"],
                          horizontal=True,
                          help="The AI agent learned when to move stock from past data. The rule tops every store "
                               "up to a high estimate of next week's demand. The Results page compares both.")
        week = pd.Timestamp(weeks[wi])
        si = stores.index(store)
        P = T[prod]
        stock, valid = plan("ai" if method.startswith("Uncertainty") else "rule", prod, wi)
        rk = assess(pd.DataFrame({"opening_stock": stock, "yhat": P.yhat[wi], "hi": P.hi[wi]}),
                    cfg["uncertainty"]["alpha"], cfg["environment"]["excess_horizon_weeks"])

        left, right = st.columns([2, 1])
        with left:
            st.subheader("Demand outlook")
            d = fc[(fc.store == store) & (fc.group == prod) & (fc.week_start <= week)].sort_values("week_start").tail(16)
            long = pd.concat([d[["week_start", "yhat"]].rename(columns={"yhat": "units"}).assign(series="Forecast"),
                              d[["week_start", "y"]].rename(columns={"y": "units"}).assign(series="Actual sales")])
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
            explain_line("Blue line: forecast made one week ahead. Shaded band: likely range of sales. "
                         "Dashed line: the planning week (its actual sales are shown for comparison).")

        r = rk.iloc[si]
        status = ex.risk_status(r.risk_class)
        p_out, stock_now, expected = float(r.p_stockout), float(r.opening_stock), float(P.yhat[wi][si])
        with right:
            st.subheader("Stock risk")
            risk_box(status)
            m1, m2 = st.columns(2)
            m1.metric("In stock", f"{stock_now:.0f}", border=True)
            m2.metric("Expected sales", f"{expected:.1f}", border=True,
                      help=f"Likely range {P.lo[wi][si]:.0f} to {P.hi[wi][si]:.0f}")
            st.metric("Chance of running out next week", "Out of stock" if stock_now <= 0 else ex.chance(p_out),
                      border=True)

        st.subheader("Recommended transfer")

        def describe(s: int, dd: int, q: int) -> dict:
            km = float(ctx.network.distance_km[s, dd])
            cost = float(transfer_cost(q, km, ctx.sim_cfg.transfer_fixed, ctx.sim_cfg.transfer_per_unit_km))
            return {"From": ex.store_label(stores[s]), "To": ex.store_label(stores[dd]), "Units": q,
                    "Distance (km)": round(km), "Cost": round(cost, 2), "_reason": ex.transfer_reason(
                        ex.store_label(stores[s]), ex.store_label(stores[dd]), q, km, cost, stock[s],
                        P.yhat[wi][s], stock[dd], P.yhat[wi][dd], float(rk.p_stockout.iloc[dd]))}

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
                            + ex.no_transfer_reason(ex.store_label(store), status, p_out, stock_now, expected))
        if valid:
            with st.expander(f"All transfers suggested for this product this week ({len(valid)})"):
                st.dataframe(pd.DataFrame([describe(*v) for v in valid]).drop(columns="_reason"),
                             hide_index=True, width="stretch")
        explain_line(f"Stock levels here are simulated as if the chosen approach had been in use since the "
                     f"week of {pd.Timestamp(weeks[0]):%d %b %Y}, using the sales that actually happened, so they can differ "
                     f"from the recorded stock in Store Overview. Stores without a stock count at that start are "
                     f"treated as empty. Distances come from a simulated store map, because the data has no store "
                     f"locations. Costs use the project's assumed shipping costs, in the same currency as the sales "
                     f"data. A store always keeps enough stock for its own expected sales before sending any.")

# ===================================================================================== 2. store overview
def store_overview() -> None:
    st.header("Store overview")
    explain_line("Which stores need attention in the chosen week. Red = may run out, amber = too much stock, "
                 "green = normal, grey = no stock count in the data.")
    if need(risk, "Risk information"):
        test = risk[risk.split == "test"].assign(status=lambda d: d.risk_class.map(ex.risk_status))
        weeks = sorted(test.week_start.unique())
        prods = sorted(test.group.unique())
        names = product_names(tuple(prods))
        c1, c2 = st.columns(2)
        week_opts = {ex.week_label(w): w for w in weeks}
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

# ===================================================================================== 3. results
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

# ===================================================================================== 4. model performance
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
               st.Page(results_page, title="Results", url_path="results"),
               st.Page(model_performance, title="Model Performance")]).run()
