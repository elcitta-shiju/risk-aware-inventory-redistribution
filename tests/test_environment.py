import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from inventory_rl.baselines.rules import RuleBasedPolicy, no_transfer
from inventory_rl.environment.gym_env import RedistributionEnv, action_to_transfers
from inventory_rl.environment.simulator import RedistributionSim, SimConfig
from inventory_rl.evaluation.experiment import evaluate_policy, run_episode, summarize


def test_reset_uses_historical_stock(sim):
    np.testing.assert_array_equal(sim.stock, [10, 0, 2])
    assert sim.t == 0 and not sim.done


def test_keep_level_and_available(sim):
    np.testing.assert_array_equal(sim.keep_level(), [1, 2, 1])
    np.testing.assert_array_equal(sim.available(), [9, 0, 1])


def test_valid_transfer_conserves_inventory_and_costs(sim):
    total0 = sim.stock.sum()
    r = sim.step([(0, 1, 3)])
    assert r.units_transferred == 3 and r.n_transfers == 1 and r.invalid_units == 0
    assert r.transfer_cost == pytest.approx(5 + 0.1 * 3 * 10)
    # conservation: closing = opening + receipts(0) - sales
    assert sim.stock.sum() == pytest.approx(total0 - r.sales)
    assert (sim.stock >= 0).all()


@pytest.mark.parametrize("bad", [[(0, 0, 2)], [(0, 1, -3)], [(0, 1, np.nan)], [(5, 1, 1)], [(1, 0, 1)]])
def test_invalid_transfers_rejected(sim, bad):
    stock0 = sim.stock.copy()
    v, invalid, _ = sim.validate(bad)
    assert v == []
    r = sim.step(bad)
    assert r.units_transferred == 0 and r.transfer_cost == 0
    assert (sim.stock <= stock0).all()  # nothing appears from nowhere


def test_source_cannot_drop_below_keep_level(sim):
    v, invalid, proposed = sim.validate([(0, 1, 50)])
    assert v == [(0, 1, 9)] and invalid == 41 and proposed == 50
    v, invalid, _ = sim.validate([(0, 1, 6), (0, 2, 6)])  # cumulative availability is respected
    assert sum(q for _, _, q in v) == 9


def test_noop_reward_is_zero_and_shadow_matches(sim):
    while not sim.done:
        r = sim.step([])
        assert sim.reward(r) == pytest.approx(0.0)
        assert r.total_cost == pytest.approx(r.noop_cost)


def test_demand_and_sales_accounting(sim):
    r = sim.step([])
    # demand [1,2,1], stock [10,0,2] -> sales [1,0,1], lost [0,2,0]
    assert (r.demand, r.sales, r.lost) == (4, 2, 2)
    assert r.shortage_cost == pytest.approx(2 * 20.0)
    assert r.stockout_stores == 1 and r.demand_stores == 3


def test_helpful_transfer_has_positive_reward(sim):
    r = sim.step([(0, 1, 2)])
    assert r.lost == 0 and sim.reward(r) > 0


def test_episode_terminates(sim):
    for _ in range(4):
        sim.step([])
    assert sim.done
    with pytest.raises(RuntimeError):
        sim.step([])


def test_poisson_mode_reproducible(tensors, dist):
    cfg = SimConfig(horizon=4)
    a, b = (RedistributionSim(tensors, dist, cfg, "poisson") for _ in range(2))
    a.reset("P1", 0, seed=3)
    b.reset("P1", 0, seed=3)
    assert [a.step([]).demand for _ in range(4)] == [b.step([]).demand for _ in range(4)]


# ---------------------------------------------------------------- projection / gym
@pytest.mark.parametrize("seed", range(20))
def test_action_to_transfers_always_feasible(sim, seed):
    a = np.random.default_rng(seed).uniform(-1, 1, 3)
    out = action_to_transfers(a, sim, 2.0)
    assert all(s != d and q > 0 for s, d, q in out)
    _, invalid, _ = sim.validate(out)
    assert invalid == 0


def test_zero_action_targets_point_forecast(sim):
    # stock [10,0,2], yhat [1,2,0.5] -> targets round(yhat) = [1,2,0]; store 1 needs 2 from nearest store 0
    assert action_to_transfers(np.zeros(3), sim, 2.0) == [(0, 1, 2)]


def test_gym_env_api(sim):
    for unc in (False, True):
        env = RedistributionEnv(sim, unc, 2.0)
        check_env(env, skip_render_check=True)
        obs, info = env.reset(seed=0, options={"product": "P1", "start": 0})
        assert obs.shape == env.observation_space.shape and np.isfinite(obs).all()
        steps, done = 0, False
        while not done:
            obs, r, done, trunc, info = env.step(env.action_space.sample())
            assert np.isfinite(r)
            steps += 1
        assert steps == 4


# ---------------------------------------------------------------- baselines & evaluation
@pytest.mark.parametrize("target", ["point", "upper", "cqr"])
def test_rule_policy_feasible(sim, target):
    pol = RuleBasedPolicy(target)
    tr = pol(sim)
    v, invalid, _ = sim.validate(tr)
    assert invalid == 0 and len(v) == len(tr)
    assert any(d == 1 for _, d, _ in tr)  # the empty store is served


def test_evaluation_metrics_finite(tensors, dist):
    cfg = SimConfig(horizon=4)
    sims = {"replay": RedistributionSim(tensors, dist, cfg, "replay"),
            "poisson": RedistributionSim(tensors, dist, cfg, "poisson")}
    df = evaluate_policy(sims, RuleBasedPolicy("upper"), n_poisson=2)
    df["policy"] = "upper"
    df0 = evaluate_policy(sims, no_transfer, n_poisson=2).assign(policy="none")
    assert (df0.saving_vs_noop.abs() < 1e-9).all()
    s = summarize(df.assign(policy="upper"), ["policy"])
    num = s.select_dtypes("number").drop(columns=[c for c in s if c.endswith("_std")])
    assert np.isfinite(num.drop(columns=["avg_transfer_km"]).to_numpy()).all()
    ep = run_episode(sims["replay"], no_transfer, "P1", 0, 0)
    assert 0 <= ep["fill_rate"] <= 1 and ep["invalid_action_rate"] == 0
