"""Assemble simulators for each split from the processed artifacts."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from inventory_rl.environment.data import ProductTensors, build_tensors
from inventory_rl.environment.simulator import RedistributionSim, SimConfig
from inventory_rl.routing.network import StoreNetwork, build_network
from inventory_rl.utils.paths import PROCESSED_DIR


@dataclass
class SimContext:
    network: StoreNetwork
    tensors: dict[str, dict[str, ProductTensors]]   # split -> product -> tensors
    sim_cfg: SimConfig
    cfg: dict

    def sim(self, split: str, demand_mode: str, seed: int | None = None) -> RedistributionSim:
        return RedistributionSim(self.tensors[split], self.network.distance_km, self.sim_cfg, demand_mode, seed)

    def sims(self, split: str) -> dict[str, RedistributionSim]:
        return {"replay": self.sim(split, "replay"), "poisson": self.sim(split, "poisson")}


def load_context(cfg: dict) -> SimContext:
    fc = pd.read_parquet(PROCESSED_DIR / "forecasts.parquet")
    panel = pd.read_parquet(PROCESSED_DIR / "panel.parquet", columns=["store", "group", "week_start", "receipts_est"])
    fc = fc.merge(panel, on=["store", "group", "week_start"], how="left")
    econ = pd.read_csv(PROCESSED_DIR / "unit_economics.csv")
    stores = sorted(fc.store.unique())
    network = build_network(stores, cfg)
    tensors = {s: build_tensors(fc, econ, stores, s) for s in ("train", "val", "test")}
    return SimContext(network, tensors, SimConfig.from_config(cfg), cfg)
