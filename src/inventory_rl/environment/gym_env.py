"""Gymnasium wrapper around RedistributionSim.

Observation (per store, concatenated over the N stores, then 2 global features):
  point variant:        [log1p(stock), log1p(available), yhat]
  uncertainty variant:  [log1p(stock), log1p(available), yhat, hi_poisson, cqr_lo, cqr_hi]
  globals:              [fraction of episode elapsed, log(unit margin)]
All are known at the start of the week (stock and as-of forecasts).

Action: Box(-1, 1, N) = per-store target-stock adjustment relative to the point forecast:
    target_i = max(0, round(yhat_i + a_i * max_target_offset))
Targets are converted to transfers by the shared matching layer (environment/matching.py):
stores below target receive units from the nearest stores holding stock above
max(target, protected level). Hence a = 0 means "hold about one week of expected demand",
a > 0 builds a safety buffer, a < 0 releases stock. Self-transfers, negative quantities and
over-drawing a source are impossible by construction, and the action space stays N-dimensional.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from inventory_rl.environment.matching import transfers_from_targets
from inventory_rl.environment.simulator import RedistributionSim


def per_store_features(use_uncertainty: bool) -> int:
    return 6 if use_uncertainty else 3


def action_to_transfers(action: np.ndarray, sim: RedistributionSim, max_offset: float) -> list[tuple[int, int, int]]:
    a = np.clip(np.asarray(action, float), -1, 1)
    target = np.clip(np.round(sim.p.yhat[sim.t] + a * max_offset), 0, None)
    return transfers_from_targets(sim.stock, target, sim.keep_level(), sim.dist,
                                  sim.cfg.transfer_fixed, sim.cfg.transfer_per_unit_km)


def build_obs(sim: RedistributionSim, use_uncertainty: bool) -> np.ndarray:
    p, t = sim.p, sim.t
    feats = [np.log1p(sim.stock), np.log1p(sim.available()), p.yhat[t]]
    if use_uncertainty:
        feats += [p.hi[t], p.cqr_lo[t], p.cqr_hi[t]]
    per_store = np.stack(feats, axis=1).ravel()
    elapsed = (t - sim.start) / max(1, sim.end - sim.start)
    return np.concatenate([per_store, [elapsed, np.log(max(p.unit_margin, 1.0))]]).astype(np.float32)


class RedistributionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, sim: RedistributionSim, use_uncertainty: bool, max_target_offset: float = 2.0,
                 fixed_product: str | None = None, fixed_start: int | None = None):
        super().__init__()
        self.sim, self.use_uncertainty, self.max_offset = sim, use_uncertainty, max_target_offset
        self.fixed_product, self.fixed_start = fixed_product, fixed_start
        n = sim.n_stores
        k = per_store_features(use_uncertainty)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n * k + 2,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        rng = self.np_random
        product = options.get("product") or self.fixed_product or self.sim.products[rng.integers(len(self.sim.products))]
        n_weeks = self.sim.data[product].n_weeks
        max_start = max(0, n_weeks - self.sim.cfg.horizon)
        start = options.get("start", self.fixed_start)
        start = int(rng.integers(max_start + 1)) if start is None else int(start)
        self.sim.reset(product, start, seed=int(rng.integers(2**31 - 1)))
        return build_obs(self.sim, self.use_uncertainty), {"product": product, "start": start}

    def step(self, action):
        res = self.sim.step(action_to_transfers(action, self.sim, self.max_offset))
        reward = self.sim.reward(res)
        terminated = self.sim.done
        obs = (np.zeros(self.observation_space.shape, np.float32) if terminated
               else build_obs(self.sim, self.use_uncertainty))
        info = {"total_cost": res.total_cost, "noop_cost": res.noop_cost, "lost": res.lost,
                "units_transferred": res.units_transferred, "invalid_units": res.invalid_units}
        return obs, float(reward), terminated, False, info
