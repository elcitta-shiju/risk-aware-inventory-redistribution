"""Stage 9: conventional (non-learning) redistribution policies.

A policy is a callable `policy(sim) -> list[(src, dst, qty)]` evaluated in the same simulator,
on the same scenarios and under the same feasibility constraints as the RL agents.

RuleBasedPolicy (documented risk-and-cost rule):
  target_i  = ceil(yhat_i)        ('point')   -- cover expected demand
            = hi_i  (Poisson q90) ('upper')   -- cover the upper prediction bound
            = ceil(cqr_hi_i)      ('cqr')     -- cover the conformalized QR upper bound
  need_i    = max(0, target_i - stock_i)                      (projected shortage)
  avail_j   = max(0, stock_j - max(ceil(yhat_j), target_j))   (surplus beyond own target)
  Destinations are served in decreasing order of need, then of stockout probability; each is
  supplied by the nearest sources with surplus. A shipment is made only if its modeled cost per
  unit does not exceed the unit margin (a transfer should not cost more than the sale is worth).
"""

from __future__ import annotations

import numpy as np

from inventory_rl.environment.matching import transfers_from_targets
from inventory_rl.environment.simulator import RedistributionSim
from inventory_rl.risk.assessment import p_at_least


def no_transfer(sim: RedistributionSim) -> list:
    return []


class RuleBasedPolicy:
    def __init__(self, target: str = "point", max_cost_per_unit_share: float = 1.0):
        if target not in ("point", "upper", "cqr"):
            raise ValueError(target)
        self.target, self.share = target, max_cost_per_unit_share
        self.name = f"Rule-based ({target})"

    def targets(self, sim: RedistributionSim) -> np.ndarray:
        p, t = sim.p, sim.t
        if self.target == "point":
            return np.ceil(p.yhat[t] - 1e-9)
        return p.hi[t].copy() if self.target == "upper" else np.ceil(p.cqr_hi[t] - 1e-9)

    def __call__(self, sim: RedistributionSim) -> list[tuple[int, int, int]]:
        p, t = sim.p, sim.t
        return transfers_from_targets(
            sim.stock, self.targets(sim), sim.keep_level(), sim.dist, sim.cfg.transfer_fixed,
            sim.cfg.transfer_per_unit_km, max_cost_per_unit=self.share * p.unit_margin,
            priority=p_at_least(sim.stock, p.yhat[t]),
        )
