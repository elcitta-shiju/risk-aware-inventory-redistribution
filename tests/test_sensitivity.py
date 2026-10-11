import numpy as np
import pandas as pd
import pytest

from inventory_rl.baselines.rules import RuleBasedPolicy, no_transfer
from inventory_rl.environment.simulator import RedistributionSim, SimConfig
from inventory_rl.evaluation.experiment import evaluate_policy
from inventory_rl.evaluation.sensitivity import settings, sim_config_for, summarize_setting

CFG = {"costs": {"transfer_fixed": 8.0, "transfer_per_unit_km": 0.03, "holding_rate_per_year": 0.25,
                 "shortage_cost_multiplier": 1.0},
       "sensitivity": {"transfer_fixed": [4.0, 8.0, 16.0], "shortage_cost_multiplier": [2.0]}}


def test_settings_one_at_a_time_and_skip_base_duplicates():
    s = settings(CFG)
    assert s[0] == {"parameter": "base", "value": None}
    assert [(x["parameter"], x["value"]) for x in s[1:]] == [
        ("transfer_fixed", 4.0), ("transfer_fixed", 16.0), ("shortage_cost_multiplier", 2.0)]


def test_unknown_parameter_rejected():
    with pytest.raises(ValueError):
        settings({**CFG, "sensitivity": {"lead_time": [1]}})


def test_sim_config_for_changes_only_one_field():
    base = SimConfig(transfer_fixed=8.0, transfer_per_unit_km=0.03, holding_rate_per_week=0.25 / 52)
    assert sim_config_for(base, "base", None) is base
    h = sim_config_for(base, "holding_rate_per_year", 0.5)
    assert h.holding_rate_per_week == pytest.approx(0.5 / 52)
    assert (h.transfer_fixed, h.transfer_per_unit_km) == (8.0, 0.03)
    f = sim_config_for(base, "transfer_fixed", 32.0)
    assert f.transfer_fixed == 32.0 and f.holding_rate_per_week == base.holding_rate_per_week


def test_shortage_multiplier_scales_noop_shortage_cost(tensors, dist):
    base = SimConfig(transfer_fixed=5.0, transfer_per_unit_km=0.1, holding_rate_per_week=0.0, horizon=4)
    costs = []
    for cfg in (base, sim_config_for(base, "shortage_cost_multiplier", 2.0)):
        sims = {"replay": RedistributionSim(tensors, dist, cfg, "replay")}
        costs.append(evaluate_policy(sims, no_transfer, 1).noop_cost.sum())
    assert costs[0] > 0
    assert costs[1] == pytest.approx(2 * costs[0])


def test_summarize_setting_reports_seed_std_only_for_multiple_seeds(tensors, dist):
    cfg = SimConfig(transfer_fixed=5.0, transfer_per_unit_km=0.1, holding_rate_per_week=0.01, horizon=4)
    sims = {"poisson": RedistributionSim(tensors, dist, cfg, "poisson")}
    frames = [evaluate_policy(sims, RuleBasedPolicy("upper"), 3).assign(policy="rule", train_seed=-1)]
    for seed in (0, 1):  # two "training seeds" of a fake learned policy
        frames.append(evaluate_policy(sims, RuleBasedPolicy("point" if seed else "upper"), 3)
                      .assign(policy="learned", train_seed=seed))
    out = summarize_setting(pd.concat(frames, ignore_index=True)).set_index("policy")
    assert np.isnan(out.loc["rule", "saving_std_train_seeds"])
    assert out.loc["learned", "n_train_seeds"] == 2
    assert np.isfinite(out.loc["learned", "saving_std_train_seeds"])
    assert 0 <= out.loc["rule", "fill_rate"] <= 1
