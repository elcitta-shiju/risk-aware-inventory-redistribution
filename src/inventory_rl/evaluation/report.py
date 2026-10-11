"""Generate a concise Markdown evaluation report from computed results (no hard-coded numbers)."""

from __future__ import annotations

import pandas as pd

from inventory_rl.utils.paths import REPORTS_DIR

COLS = ["policy", "fill_rate", "stockout_rate", "lost_units", "avg_closing_stock", "transfer_cost", "total_cost",
        "saving_vs_noop", "saving_vs_noop_std", "n_transfers", "units_transferred", "avg_transfer_km",
        "invalid_action_rate"]


def _md_table(df: pd.DataFrame, digits: int = 3) -> str:
    df = df.copy()
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    rows = []
    for _, r in df.iterrows():
        cells = [f"{v:,.{digits}f}" if isinstance(v, float) else str(v) for v in r]
        rows.append("| " + " | ".join(cells) + " |")
    return head + "\n".join(rows) + "\n"


def _verdict(d: dict) -> str:
    if d["n_scenarios"] < 2:
        return "single scenario - no interval"
    if d["ci95_low"] > 0:
        return "positive (95% CI excludes 0)"
    if d["ci95_high"] < 0:
        return "negative (95% CI excludes 0)"
    return "not distinguishable from 0"


def write_report(results: dict, summary: pd.DataFrame, per_seed: pd.DataFrame) -> None:
    lines = ["# Evaluation report (generated)", "",
             "All values are computed by `scripts/run_evaluation.py` on the held-out **test** weeks. Costs are",
             "**modeled** (transfer, holding and distance parameters are assumptions; unit margins are observed).",
             f"Store distances are **{results['network_provenance']}**. Simulation results are not evidence of",
             "performance in a live retail operation.", "",
             f"- Products: {results['products']} SKUs, stores: {results['stores']}, "
             f"Poisson demand scenarios: {results['n_poisson_scenarios']} (+1 historical replay)",
             f"- Interval method (selected on validation): `{results['interval_method']}`", ""]
    for scen, desc in [("poisson", "Poisson demand scenarios (mean +/- std over scenario x training seed)"),
                       ("replay", "Historical replay (observed, stock-censored sales as demand)")]:
        s = summary[summary.scenario == scen]
        lines += [f"## Policy comparison - {desc}", "", _md_table(s[[c for c in COLS if c in s]].assign(
            policy=lambda d: d.policy.astype(str)), 2), ""]
    lines += ["## Paired differences in total saving (same scenarios)", "",
              "| comparison | mean | 95% CI | n | reading |", "|---|---|---|---|---|"]
    for k, d in results["paired_differences"].items():
        lines.append(f"| {k} | {d['mean']:,.1f} | [{d['ci95_low']:,.1f}, {d['ci95_high']:,.1f}] | "
                     f"{d['n_scenarios']} | {_verdict(d)} |")
    lines += ["", "The CI treats demand scenarios as independent replicates; it does not capture uncertainty",
              "from the choice of products, period, or cost assumptions.", "",
              "## RL robustness across training seeds (mean saving per scenario)", "",
              _md_table(per_seed.round(1), 1), "",
              "## Forecast accuracy (test, one-week-ahead)", "",
              _md_table(pd.DataFrame(results["forecast_test"]).T.reset_index().rename(columns={"index": "model"}), 4),
              "", "## Prediction intervals (test)", "",
              _md_table(pd.DataFrame(results["intervals_test"]).T.reset_index().rename(columns={"index": "method"}), 3)]
    if results["missing_rl_checkpoints"]:
        lines += ["", "**Warning:** missing RL checkpoints: " + ", ".join(results["missing_rl_checkpoints"])]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")
