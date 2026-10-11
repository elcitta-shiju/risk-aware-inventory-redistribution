"""Stage 7: weekly multi-store inventory redistribution simulator for one product.

Weekly sequence for week t (all stores of one product):
  1. Decision: a list of transfers (src, dst, qty) is proposed using information available at
     the start of week t (opening stock, as-of forecasts).
  2. Feasibility: each transfer must have src != dst, a positive integer qty, and the source may
     only send units above its protected level keep_t = ceil(yhat_t) (it must keep at least its
     own expected demand). Infeasible quantities are clipped and counted as invalid units.
  3. Transfers arrive before demand (lead time 0 - ASSUMPTION).
  4. Historical net receipts/adjustments for week t are replayed: stock <- max(0, stock + r_t).
  5. Demand D_t is realized ('replay' = observed sales, 'poisson' = Poisson(lambda_t)).
     sales = min(D, stock), lost = D - sales.
  6. Cost = margin * lost + holding * closing stock + transfer costs.
A shadow "no-transfer" trajectory with the SAME demand and receipts is simulated in parallel;
the reward is the cost saving versus that shadow (a policy-independent baseline that reduces
variance without changing the optimal policy).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from inventory_rl.environment.data import ProductTensors
from inventory_rl.routing.network import transfer_cost


@dataclass
class SimConfig:
    transfer_fixed: float = 15.0
    transfer_per_unit_km: float = 0.02
    holding_rate_per_week: float = 0.25 / 52     # ASSUMPTION: 25%/year of unit cost
    shortage_cost_multiplier: float = 1.0        # lost sale costs this x observed unit margin
    horizon: int = 8
    reward_scale: float = 10.0

    @classmethod
    def from_config(cls, cfg: dict) -> "SimConfig":
        c = cfg["costs"]
        return cls(transfer_fixed=c["transfer_fixed"], transfer_per_unit_km=c["transfer_per_unit_km"],
                   holding_rate_per_week=c["holding_rate_per_year"] / 52,
                   shortage_cost_multiplier=c["shortage_cost_multiplier"],
                   horizon=cfg["environment"]["horizon_weeks"], reward_scale=cfg["environment"]["reward_scale"])


@dataclass
class StepResult:
    demand: float
    sales: float
    lost: float
    closing_stock: float
    holding_cost: float
    shortage_cost: float
    transfer_cost: float
    n_transfers: int
    units_transferred: float
    transfer_km: float            # unit-weighted km
    invalid_units: float
    proposed_units: float
    stockout_stores: int          # stores with demand > available stock
    demand_stores: int            # stores with positive demand
    noop_cost: float
    transfers: list = field(default_factory=list)

    @property
    def total_cost(self) -> float:
        return self.holding_cost + self.shortage_cost + self.transfer_cost


class RedistributionSim:
    def __init__(self, data: dict[str, ProductTensors], distance_km: np.ndarray, cfg: SimConfig,
                 demand_mode: str = "poisson", seed: int | None = None):
        if demand_mode not in ("poisson", "replay"):
            raise ValueError("demand_mode must be 'poisson' or 'replay'")
        self.data, self.dist, self.cfg, self.demand_mode = data, distance_km, cfg, demand_mode
        self.products = sorted(data)
        self.n_stores = distance_km.shape[0]
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ episode control
    def reset(self, product: str, start: int, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.p = self.data[product]
        if not 0 <= start < self.p.n_weeks:
            raise ValueError("start week out of range")
        self.start, self.t = start, start
        self.end = min(start + self.cfg.horizon, self.p.n_weeks)
        s0 = self.p.stock0[start]
        self.stock_missing_initial = int(np.isnan(s0).sum())
        # Missing stock record -> 0 units FOR SIMULATION ONLY (flagged in stock_missing_initial).
        self.stock = np.nan_to_num(s0, nan=0.0).round()
        self.shadow = self.stock.copy()

    @property
    def done(self) -> bool:
        return self.t >= self.end

    def keep_level(self) -> np.ndarray:
        """Hard protection: a source keeps at least ceil(point forecast) units."""
        return np.ceil(self.p.yhat[self.t] - 1e-9)

    def available(self) -> np.ndarray:
        return np.clip(self.stock - self.keep_level(), 0, None)

    # ------------------------------------------------------------------ dynamics
    def _demand(self) -> np.ndarray:
        if self.demand_mode == "replay":
            return self.p.demand_obs[self.t].copy()
        return self.rng.poisson(self.p.lam[self.t]).astype(float)

    def validate(self, transfers) -> tuple[list[tuple[int, int, int]], float, float]:
        """Clip a proposed transfer list to feasibility. Returns (valid, invalid_units, proposed_units)."""
        avail = self.available()
        valid, invalid, proposed = [], 0.0, 0.0
        for src, dst, qty in transfers:
            q = float(qty)
            proposed += max(q, 0.0)
            ok_idx = 0 <= src < self.n_stores and 0 <= dst < self.n_stores and src != dst
            if not ok_idx or not np.isfinite(q) or q <= 0:
                invalid += max(q, 0.0) if np.isfinite(q) else 0.0
                continue
            qi = int(np.floor(q + 1e-9))
            send = min(qi, int(avail[src]))
            invalid += q - send
            if send > 0:
                avail[src] -= send
                valid.append((int(src), int(dst), send))
        return valid, invalid, proposed

    def step(self, transfers) -> StepResult:
        if self.done:
            raise RuntimeError("episode finished; call reset()")
        valid, invalid, proposed = self.validate(transfers)
        before = self.stock.sum()
        tcost, units, km_units = 0.0, 0.0, 0.0
        for src, dst, q in valid:
            self.stock[src] -= q
            self.stock[dst] += q
            km = self.dist[src, dst]
            tcost += float(transfer_cost(q, km, self.cfg.transfer_fixed, self.cfg.transfer_per_unit_km))
            units += q
            km_units += q * km
        if abs(self.stock.sum() - before) > 1e-9:
            raise RuntimeError("transfers did not conserve inventory")
        if (self.stock < 0).any():
            raise RuntimeError("negative inventory after transfers")

        r = self.p.receipts[self.t]
        demand = self._demand()
        res_main = self._settle(self.stock, r, demand)
        res_shadow = self._settle(self.shadow, r, demand)
        self.stock, self.shadow = res_main["closing"], res_shadow["closing"]
        noop_cost = res_shadow["holding"] + res_shadow["shortage"]
        out = StepResult(
            demand=float(demand.sum()), sales=res_main["sales"], lost=res_main["lost"],
            closing_stock=float(self.stock.sum()), holding_cost=res_main["holding"],
            shortage_cost=res_main["shortage"], transfer_cost=tcost, n_transfers=len(valid),
            units_transferred=units, transfer_km=km_units, invalid_units=invalid, proposed_units=proposed,
            stockout_stores=res_main["stockout_stores"], demand_stores=int((demand > 0).sum()),
            noop_cost=noop_cost, transfers=valid,
        )
        self.t += 1
        return out

    def _settle(self, stock: np.ndarray, receipts: np.ndarray, demand: np.ndarray) -> dict:
        s = np.clip(stock + receipts, 0, None)
        sales = np.minimum(demand, s)
        lost = demand - sales
        closing = s - sales
        return {
            "closing": closing, "sales": float(sales.sum()), "lost": float(lost.sum()),
            "holding": float(closing.sum() * self.cfg.holding_rate_per_week * self.p.unit_cost),
            "shortage": float(lost.sum() * self.cfg.shortage_cost_multiplier * self.p.unit_margin),
            "stockout_stores": int((lost > 0).sum()),
        }

    def reward(self, res: StepResult) -> float:
        return (res.noop_cost - res.total_cost) / self.cfg.reward_scale
