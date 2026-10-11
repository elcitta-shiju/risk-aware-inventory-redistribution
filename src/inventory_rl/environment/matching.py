"""Convert per-store target stock levels into feasible transfers (shared by rules and RL).

need_i  = max(0, target_i - stock_i)
avail_j = max(0, stock_j - max(keep_j, target_j))      keep_j = hard protected level
Destinations are served in decreasing order of need (ties: higher priority first); each is
supplied from the nearest sources with available units. Optionally a shipment is skipped when
its modeled cost per unit exceeds `max_cost_per_unit`.
"""

from __future__ import annotations

import numpy as np

from inventory_rl.routing.network import transfer_cost


def transfers_from_targets(stock: np.ndarray, target: np.ndarray, keep: np.ndarray, dist: np.ndarray,
                           fixed: float, per_unit_km: float, max_cost_per_unit: float | None = None,
                           priority: np.ndarray | None = None) -> list[tuple[int, int, int]]:
    target = np.ceil(np.asarray(target, float) - 1e-9)
    need = np.floor(np.clip(target - stock, 0, None))
    avail = np.floor(np.clip(stock - np.maximum(keep, target), 0, None))
    if need.sum() == 0 or avail.sum() == 0:
        return []
    prio = np.zeros(len(stock)) if priority is None else priority
    order = sorted(np.flatnonzero(need > 0), key=lambda i: (-need[i], -prio[i]))
    out = []
    for d in order:
        for s in np.argsort(dist[:, d], kind="stable"):
            if need[d] <= 0:
                break
            if s == d or avail[s] <= 0:
                continue
            q = int(min(need[d], avail[s]))
            if q < 1:
                continue
            if max_cost_per_unit is not None:
                if float(transfer_cost(q, dist[s, d], fixed, per_unit_km)) / q > max_cost_per_unit:
                    continue
            out.append((int(s), int(d), q))
            need[d] -= q
            avail[s] -= q
    return out
