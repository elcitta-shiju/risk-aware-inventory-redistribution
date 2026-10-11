"""Streamlit dashboard: risk-aware inventory redistribution.

Run from the project root:  streamlit run dashboard/app.py
It only READS saved artifacts (no model training on launch).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_access as da  # noqa: E402

from inventory_rl.utils.config import load_config  # noqa: E402
from inventory_rl.utils.paths import METRICS_DIR, PROCESSED_DIR, REPORTS_DIR  # noqa: E402

st.set_page_config(page_title="Inventory Redistribution", layout="wide")
st.title("Risk-Aware Inventory Redistribution")
st.caption("Uncertainty-aware demand forecasting + reinforcement learning. Mendeley 'Retail Transactions and "
           "Stocks' data, top-30 SKUs x 40 stores, weekly. Transfer outcomes are SIMULATED; store locations are "
           "SIMULATED; cost parameters are assumptions.")


@st.cache_data
def table(path: str):
    return da.load_table(Path(path))


@st.cache_data
def js(path: str):
    return da.load_json(Path(path))


@st.cache_resource
def sim_context():
    from inventory_rl.environment.factory import load_context

    return load_context(load_config())


def need(obj, what: str) -> bool:
    if obj is None:
        st.warning(f"{what} not found. Run the pipeline first (see README: `python scripts/run_pipeline.py`).")
        return False
    return True


tabs = st.tabs(["Overview", "Sales & stock", "Forecasts", "Risk", "Transfers", "Evaluation", "Assumptions"])

# ------------------------------------------------------------------------------------- overview
with tabs[0]:
    st.subheader("Pipeline artifacts")
    st.dataframe(da.artifact_status(), hide_index=True, use_container_width=True)
    audit = js(str(REPORTS_DIR / "data_audit.json"))
    prep = js(str(REPORTS_DIR / "preprocessing_report.json"))
    if need(audit, "Data audit"):
        m = audit["mendeley"]
        c = st.columns(4)
        c[0].metric("Sales transactions", f"{m['sales']['rows']:,}")
        c[1].metric("Stock records", f"{m['inventory']['rows']:,}")
        c[2].metric("Stores", m["sales"]["n_stores"])
        c[3].metric("Products (SKUs)", m["sales"]["n_products"])
        st.markdown("**Data-quality highlights (Mendeley, primary dataset)**")
        st.json({
            "sales date range": m["sales"]["transaction_date_range"],
            "return rows (negative qty)": m["sales"]["return_rows"],
            "negative stock records": m["inventory"]["negative_qty_rows"],
            "stock change = -sales on contiguous days": m["inventory"]["share_stock_change_equals_minus_sales"],
            "kaggle retail_store_inventory lag-1 autocorr (≈0 => random-looking)":
                audit["kaggle_retail_store_inventory"]["units_sold_lag1_autocorr_mean"],
        })
    if prep:
        st.markdown("**Weekly panel quality**")
        st.json(prep["quality"])

# ------------------------------------------------------------------------------------- history
panel = table(str(PROCESSED_DIR / "panel.parquet"))
with tabs[1]:
    if need(panel, "Weekly panel"):
        c1, c2 = st.columns(2)
        prod = c1.selectbox("Product (SKU)", sorted(panel.group.unique()), key="h_prod")
        stores = sorted(panel.store.unique())
        sel = c2.multiselect("Stores", stores, default=stores[:3], key="h_store")
        d = panel[(panel.group == prod) & panel.store.isin(sel)]
        st.markdown("Weekly units sold (returns excluded)")
        st.line_chart(d.pivot(index="week_start", columns="store", values="gross_qty"))
        st.markdown("Opening stock (gaps = no stock record, *not* zero)")
        st.line_chart(d.pivot(index="week_start", columns="store", values="opening_stock"))
        net = panel[panel.group == prod].groupby("week_start")[["gross_qty", "opening_stock"]].sum()
        st.markdown("Network total for this SKU")
        st.line_chart(net)

# ------------------------------------------------------------------------------------- forecasts
fc = table(str(PROCESSED_DIR / "forecasts.parquet"))
with tabs[2]:
    if need(fc, "Forecasts"):
        c1, c2 = st.columns(2)
        prod = c1.selectbox("Product", sorted(fc.group.unique()), key="f_prod")
        store = c2.selectbox("Store", sorted(fc.store.unique()), key="f_store")
        d = fc[(fc.group == prod) & (fc.store == store)].set_index("week_start").sort_index()
        st.line_chart(d[["y", "yhat", "lo", "hi"]].rename(columns={
            "y": "actual", "yhat": "forecast", "lo": "lower bound", "hi": "upper bound"}))
        st.caption("One-week-ahead rolling-origin forecasts (each fitted only on earlier weeks).")
        fm = js(str(METRICS_DIR / "forecast_metrics.json"))
        im = js(str(METRICS_DIR / "interval_metrics.json"))
        if fm:
            st.markdown("**Point forecast accuracy (test)**")
            st.dataframe(pd.DataFrame(fm["test"]).T.round(4), use_container_width=True)
        if im:
            st.markdown(f"**Prediction intervals (test). Method selected on validation: `{im['chosen']}`**")
            st.dataframe(pd.DataFrame(im["by_split"]["test"]).T.round(3), use_container_width=True)
        for f in ["forecast_vs_actual_total.png", "interval_coverage_by_sku_test.png"]:
            if da.figure(f):
                st.image(str(da.figure(f)))

# ------------------------------------------------------------------------------------- risk
risk = table(str(PROCESSED_DIR / "risk.parquet"))
with tabs[3]:
    if need(risk, "Risk table"):
        c1, c2 = st.columns(2)
        prod = c1.selectbox("Product", sorted(risk.group.unique()), key="r_prod")
        weeks = sorted(risk.week_start.unique())
        wk = c2.select_slider("Decision week", weeks, value=weeks[-8], format_func=lambda w: str(pd.Timestamp(w).date()))
        d = risk[(risk.group == prod) & (risk.week_start == wk)].sort_values("p_stockout", ascending=False)
        cols = ["store", "opening_stock", "yhat", "lo", "hi", "p_stockout", "expected_shortage", "shortfall_upper",
                "surplus_upper", "excess_units", "risk_class", "y"]
        st.dataframe(d[cols].round(3), hide_index=True, use_container_width=True)
        st.caption("p_stockout = P(demand >= stock) under Poisson(forecast) - a model-based probability whose "
                   "calibration is checked on held-out data (reliability plot below). y = actual sales that week.")
        c = st.columns(3)
        c[0].metric("Potential destinations (shortfall vs upper bound)", int((d.shortfall_upper > 0).sum()))
        c[1].metric("Potential sources (surplus above upper bound)", int((d.surplus_upper > 0).sum()))
        c[2].metric("Unknown stock (no record)", int(d.stock_missing.sum()))
        for f in ["risk_stockout_reliability_test.png", "risk_imbalance_over_time.png"]:
            if da.figure(f):
                st.image(str(da.figure(f)))

# ------------------------------------------------------------------------------------- transfers
with tabs[4]:
    st.markdown("Suggested transfers for one product and one week, computed from that week's opening stock and "
                "as-of forecasts. Quantities respect the source protection rule (a source keeps at least "
                "ceil(forecast)).")
    if need(fc, "Forecasts") and need(table(str(METRICS_DIR / "store_network.csv")), "Store network"):
        from inventory_rl.baselines.rules import RuleBasedPolicy
        from inventory_rl.routing.network import transfer_cost

        ctx = sim_context()
        cfg = ctx.cfg
        split = st.radio("Period", ["test", "val", "train"], horizontal=True)
        T = ctx.tensors[split]
        c1, c2, c3 = st.columns(3)
        prod = c1.selectbox("Product", sorted(T), key="t_prod")
        wi = c2.selectbox("Week", range(T[prod].n_weeks), format_func=lambda i: str(T[prod].weeks[i].date()))
        policies = {"Rule-based (upper bound)": RuleBasedPolicy("upper"), "Rule-based (point)": RuleBasedPolicy("point"),
                    "Rule-based (CQR bound)": RuleBasedPolicy("cqr")}
        for ck in da.rl_checkpoints():
            unc = ck.parent.name.startswith("rl_uncertainty")
            label = f"RL ({'forecast + uncertainty' if unc else 'point forecast'}) - {ck.parent.name}"
            policies[label] = ("rl", ck, unc)
        pname = c3.selectbox("Policy", list(policies))
        sim = ctx.sim(split, "replay")
        sim.reset(prod, wi)
        pol = policies[pname]
        if isinstance(pol, tuple):
            from inventory_rl.agents.ppo import load_policy

            pol = load_policy(pol[1], pol[2], cfg["environment"], pname)
        transfers = pol(sim)
        valid, invalid, _ = sim.validate(transfers)
        stores = ctx.network.stores
        rows = [{"from": stores[s], "to": stores[d], "qty": q, "distance_km (simulated)": round(sim.dist[s, d], 1),
                 "modeled cost": round(float(transfer_cost(q, sim.dist[s, d], sim.cfg.transfer_fixed,
                                                           sim.cfg.transfer_per_unit_km)), 2),
                 "source stock": sim.stock[s], "dest stock": sim.stock[d],
                 "dest forecast": round(sim.p.yhat[wi][d], 2), "feasible": (s, d, q) in valid} for s, d, q in transfers]
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        else:
            st.info("No transfer recommended for this product and week.")
        st.caption(f"Missing stock records at this week (treated as 0 in the simulation): {sim.stock_missing_initial}. "
                   f"Infeasible units proposed: {invalid:g}.")

# ------------------------------------------------------------------------------------- evaluation
with tabs[5]:
    res = js(str(METRICS_DIR / "results.json"))
    if need(res, "Evaluation results"):
        summ = pd.DataFrame(res["summary"])
        scen = st.radio("Demand scenario", ["poisson", "replay"], horizontal=True,
                        help="poisson: simulated demand around smoothed observed sales; replay: observed sales "
                             "(censored by historical stock-outs, so a pessimistic view of transfer benefits)")
        show = ["policy", "fill_rate", "stockout_rate", "lost_units", "transfer_cost", "total_cost", "saving_vs_noop",
                "saving_vs_noop_std", "n_transfers", "units_transferred", "avg_transfer_km", "invalid_action_rate"]
        st.dataframe(summ[summ.scenario == scen][show].round(3), hide_index=True, use_container_width=True)
        st.markdown("**Paired differences in saving (same scenarios)**")
        st.dataframe(pd.DataFrame(res["paired_differences"]).T.round(1), use_container_width=True)
        if res.get("rl_across_training_seeds"):
            st.markdown("**RL saving across training seeds (mean ± std of per-seed means)**")
            seeds = pd.DataFrame(res["rl_across_training_seeds"])
            st.dataframe(seeds[seeds.scenario == scen].round(1), hide_index=True, use_container_width=True)
        if res.get("missing_rl_checkpoints"):
            st.warning("Some RL checkpoints were missing at evaluation time.")
        for f in [f"eval_saving_{scen}.png", f"eval_fill_rate_{scen}.png", "rl_validation_curves.png"]:
            if da.figure(f):
                st.image(str(da.figure(f)))
        rep = REPORTS_DIR / "evaluation_report.md"
        if rep.exists():
            with st.expander("Generated evaluation report"):
                st.markdown(rep.read_text(encoding="utf-8"))
        sens = REPORTS_DIR / "cost_sensitivity_report.md"
        if sens.exists():
            with st.expander("Cost-assumption sensitivity (generated)"):
                if da.figure("cost_sensitivity.png"):
                    st.image(str(da.figure("cost_sensitivity.png")))
                st.markdown(sens.read_text(encoding="utf-8"))

# ------------------------------------------------------------------------------------- assumptions
with tabs[6]:
    doc = Path(__file__).resolve().parents[1] / "docs" / "assumptions_and_limitations.md"
    st.markdown(doc.read_text(encoding="utf-8") if doc.exists() else "docs/assumptions_and_limitations.md not found.")
    st.markdown("**Active configuration**")
    st.json(load_config())
