"""Stage 10: run any policy over identical scenarios and compute operational metrics.

Scenario set for a split: for every product, one episode starting at the first week of the
split (horizon = split length), under
  - 'replay'  : observed sales as demand (1 scenario; censored, i.e. a lower bound on demand)
  - 'poisson' : K Poisson demand scenarios with fixed seeds (identical across policies).

Metric definitions (per episode, then averaged over episodes):
  fill_rate          sum(sales) / sum(demand)
  stockout_rate      store-weeks with lost demand / store-weeks with positive demand
  lost_units         sum of unmet demand
  avg_closing_stock  mean over weeks of network closing stock (inventory level / excess proxy)
  holding_cost, shortage_cost, transfer_cost, total_cost  (modeled; see assumptions)
  saving_vs_noop     no-transfer cost - policy cost on the same demand path
  n_transfers, units_transferred, avg_transfer_km (unit-weighted)
  invalid_action_rate  infeasible proposed units / proposed units (0 if nothing proposed)
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from inventory_rl.environment.simulator import RedistributionSim

Policy = Callable[[RedistributionSim], list]


def run_episode(sim: RedistributionSim, policy: Policy, product: str, start: int, seed: int) -> dict:
    sim.reset(product, start, seed=seed)
    acc = dict(demand=0.0, sales=0.0, lost=0.0, holding_cost=0.0, shortage_cost=0.0, transfer_cost=0.0,
               noop_cost=0.0, n_transfers=0, units_transferred=0.0, transfer_km=0.0, invalid_units=0.0,
               proposed_units=0.0, stockout_store_weeks=0, demand_store_weeks=0, closing_stock_sum=0.0)
    weeks = 0
    while not sim.done:
        r = sim.step(policy(sim))
        weeks += 1
        for k in ("demand", "sales", "lost", "holding_cost", "shortage_cost", "transfer_cost", "noop_cost",
                  "units_transferred", "transfer_km", "invalid_units", "proposed_units"):
            acc[k] += getattr(r, k)
        acc["n_transfers"] += r.n_transfers
        acc["stockout_store_weeks"] += r.stockout_stores
        acc["demand_store_weeks"] += r.demand_stores
        acc["closing_stock_sum"] += r.closing_stock
    total = acc["holding_cost"] + acc["shortage_cost"] + acc["transfer_cost"]
    return {
        "product": product, "start": start, "seed": seed, "weeks": weeks,
        "demand": acc["demand"], "sales": acc["sales"], "lost_units": acc["lost"],
        "fill_rate": acc["sales"] / acc["demand"] if acc["demand"] > 0 else np.nan,
        "stockout_rate": (acc["stockout_store_weeks"] / acc["demand_store_weeks"]
                          if acc["demand_store_weeks"] > 0 else np.nan),
        "avg_closing_stock": acc["closing_stock_sum"] / max(weeks, 1),
        "holding_cost": acc["holding_cost"], "shortage_cost": acc["shortage_cost"],
        "transfer_cost": acc["transfer_cost"], "total_cost": total,
        "noop_cost": acc["noop_cost"], "saving_vs_noop": acc["noop_cost"] - total,
        "n_transfers": acc["n_transfers"], "units_transferred": acc["units_transferred"],
        "avg_transfer_km": acc["transfer_km"] / acc["units_transferred"] if acc["units_transferred"] else np.nan,
        "invalid_action_rate": acc["invalid_units"] / acc["proposed_units"] if acc["proposed_units"] else 0.0,
    }


def evaluate_policy(sims: dict[str, RedistributionSim], policy: Policy, n_poisson: int,
                    base_seed: int = 10_000) -> pd.DataFrame:
    """sims: {'replay': sim, 'poisson': sim} built on the same split. Same scenarios for every policy."""
    rows = []
    for mode, sim in sims.items():
        seeds = [base_seed] if mode == "replay" else [base_seed + k for k in range(n_poisson)]
        for prod in sim.products:
            for sd in seeds:
                rows.append({"scenario": mode, **run_episode(sim, policy, prod, 0, sd)})
    return pd.DataFrame(rows)


SUMMARY_COLS = ["fill_rate", "stockout_rate", "lost_units", "avg_closing_stock", "holding_cost", "shortage_cost",
                "transfer_cost", "total_cost", "saving_vs_noop", "n_transfers", "units_transferred",
                "avg_transfer_km", "invalid_action_rate"]


def summarize(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Per-episode metrics summed over products per scenario seed, then mean/std across seeds.

    Network-level fill/stockout rates are recomputed from totals (ratio of sums)."""
    df = df.reset_index(drop=True).assign(
        km_units=lambda d: (d.avg_transfer_km * d.units_transferred).fillna(0.0))
    agg = df.groupby(by + ["scenario", "seed"]).agg(
        demand=("demand", "sum"), sales=("sales", "sum"), lost_units=("lost_units", "sum"),
        avg_closing_stock=("avg_closing_stock", "sum"), holding_cost=("holding_cost", "sum"),
        shortage_cost=("shortage_cost", "sum"), transfer_cost=("transfer_cost", "sum"),
        total_cost=("total_cost", "sum"), saving_vs_noop=("saving_vs_noop", "sum"),
        n_transfers=("n_transfers", "sum"), units_transferred=("units_transferred", "sum"),
        km_units=("km_units", "sum"),
        stockout_rate=("stockout_rate", "mean"), invalid_action_rate=("invalid_action_rate", "mean"),
    ).reset_index()
    agg["fill_rate"] = agg.sales / agg.demand
    agg["avg_transfer_km"] = np.where(agg.units_transferred > 0, agg.km_units / agg.units_transferred.where(agg.units_transferred > 0), np.nan)
    g = agg.groupby(by + ["scenario"])[SUMMARY_COLS]
    mean, std = g.mean(), g.std(ddof=1)
    out = mean.join(std, rsuffix="_std")
    out["n_seeds"] = g.size()
    return out.reset_index()
