"""One-at-a-time sensitivity of policy savings to the ASSUMED cost parameters.

Each setting changes exactly one cost parameter of the base configuration and re-simulates every
policy on the same test scenarios. Policies are NOT retrained: RL agents trained under the base
costs are evaluated as fixed policies. Rule-based policies react to transfer costs through their
cost-per-unit cap; RL actions do not depend on cost parameters (they are not observed).
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from inventory_rl.environment.factory import SimContext
from inventory_rl.environment.simulator import RedistributionSim, SimConfig
from inventory_rl.evaluation.experiment import evaluate_policy

# config key under `costs` -> SimConfig field (holding is converted from per-year to per-week)
PARAMS = {
    "transfer_fixed": "transfer_fixed",
    "transfer_per_unit_km": "transfer_per_unit_km",
    "holding_rate_per_year": "holding_rate_per_week",
    "shortage_cost_multiplier": "shortage_cost_multiplier",
}


def settings(cfg: dict) -> list[dict]:
    """Base setting plus one setting per (parameter, alternative value) in cfg['sensitivity']."""
    base = cfg["costs"]
    out = [{"parameter": "base", "value": None}]
    for key, values in cfg["sensitivity"].items():
        if key not in PARAMS:
            raise ValueError(f"unknown sensitivity parameter: {key}")
        out += [{"parameter": key, "value": float(v)} for v in values if float(v) != float(base[key])]
    return out


def sim_config_for(base: SimConfig, parameter: str, value: float | None) -> SimConfig:
    if parameter == "base":
        return base
    field = PARAMS[parameter]
    return replace(base, **{field: value / 52 if parameter == "holding_rate_per_year" else value})


def run_setting(ctx: SimContext, sim_cfg: SimConfig, policies: list[tuple[str, int, object]],
                n_poisson: int, split: str = "test") -> pd.DataFrame:
    """Evaluate (label, train_seed, policy) triples under one cost configuration."""
    dist = ctx.network.distance_km
    sims = {m: RedistributionSim(ctx.tensors[split], dist, sim_cfg, m) for m in ("replay", "poisson")}
    frames = [evaluate_policy(sims, pol, n_poisson).assign(policy=label, train_seed=seed)
              for label, seed, pol in policies]
    return pd.concat(frames, ignore_index=True)


def summarize_setting(ep: pd.DataFrame) -> pd.DataFrame:
    """Saving per policy and scenario type.

    saving_mean: mean over demand scenarios (and training seeds) of the network total saving.
    saving_std_scenarios: std over demand scenarios of the (seed-averaged) total saving.
    saving_std_train_seeds: std over training seeds of each seed's mean saving (RL only; NaN for rules).
    """
    tot = ep.groupby(["policy", "train_seed", "scenario", "seed"]).agg(
        saving=("saving_vs_noop", "sum"), lost=("lost_units", "sum"), demand=("demand", "sum"),
        units=("units_transferred", "sum"), transfer_cost=("transfer_cost", "sum")).reset_index()
    per_seed = tot.groupby(["policy", "train_seed", "scenario"]).saving.mean()
    seed_std = per_seed.groupby(["policy", "scenario"]).std(ddof=1)
    by_scen = tot.groupby(["policy", "scenario", "seed"])[["saving", "lost", "demand", "units",
                                                           "transfer_cost"]].mean()
    g = by_scen.groupby(["policy", "scenario"])
    out = pd.DataFrame({
        "saving_mean": g.saving.mean(), "saving_std_scenarios": g.saving.std(ddof=1),
        "saving_std_train_seeds": seed_std,
        "fill_rate": 1 - g.lost.sum() / g.demand.sum(),
        "units_transferred": g.units.mean(), "transfer_cost": g.transfer_cost.mean(),
        "n_train_seeds": per_seed.groupby(["policy", "scenario"]).size(),
    }).reset_index()
    out.loc[out.n_train_seeds < 2, "saving_std_train_seeds"] = float("nan")
    return out
