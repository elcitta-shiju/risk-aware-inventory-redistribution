"""Stage 6: store locations, distances and transfer costs.

The Mendeley data contains NO store addresses or coordinates (only anonymized IDs such as
STR-1006). Therefore, by default, store coordinates are SIMULATED (seeded, uniform in a square
region) and road distance is approximated as Euclidean distance x a circuity factor. Every
output carries a `provenance` field so simulated values are never confused with observations.

If a CSV of real coordinates (columns: store, lat, lon) is supplied and online routing is
enabled (ENABLE_ONLINE_ROUTING=1), road distances can be fetched from an OSRM server with
on-disk caching. This path is optional; nothing else in the project requires network access.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from inventory_rl.utils.paths import RESULTS_DIR

OSRM_PUBLIC = "https://router.project-osrm.org"


@dataclass
class StoreNetwork:
    stores: list[str]
    coords: np.ndarray           # (N, 2) km for simulated, (lat, lon) for real
    distance_km: np.ndarray      # (N, N), zero diagonal
    provenance: str              # 'SIMULATED' | 'OSRM' | 'HAVERSINE'

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"store": self.stores, "x": self.coords[:, 0], "y": self.coords[:, 1],
                             "provenance": self.provenance})


def simulated_network(stores: list[str], region_km: float, circuity: float, seed: int) -> StoreNetwork:
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, region_km, size=(len(stores), 2))
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)) * circuity
    return StoreNetwork(list(stores), xy, d, "SIMULATED")


def haversine_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat, lon = np.radians(lat), np.radians(lon)
    dlat, dlon = lat[:, None] - lat[None, :], lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def osrm_distance_matrix(lat: np.ndarray, lon: np.ndarray, base_url: str = OSRM_PUBLIC,
                         cache: Path | None = None, timeout: float = 30.0) -> np.ndarray:
    """Road distances (km) from the OSRM table service. Results are cached; raises on failure.

    Note: the public demo server has a fair-use policy (no heavy use); use a self-hosted OSRM
    instance for anything beyond a few requests.
    """
    cache = cache or RESULTS_DIR / "cache" / "osrm_table.json"
    key = ";".join(f"{a:.6f},{b:.6f}" for a, b in zip(lon, lat))
    if cache.exists():
        stored = json.loads(cache.read_text())
        if key in stored:
            return np.array(stored[key])
    url = f"{base_url}/table/v1/driving/{key}?annotations=distance"
    with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 (fixed https URL)
        body = json.loads(r.read())
    if body.get("code") != "Ok":
        raise RuntimeError(f"OSRM error: {body.get('code')} {body.get('message')}")
    d = np.array(body["distances"], dtype=float) / 1000.0
    if np.isnan(d).any():
        raise RuntimeError("OSRM returned missing routes for some store pairs")
    cache.parent.mkdir(parents=True, exist_ok=True)
    stored = json.loads(cache.read_text()) if cache.exists() else {}
    stored[key] = d.tolist()
    cache.write_text(json.dumps(stored))
    return d


def build_network(stores: list[str], cfg: dict, coordinates_csv: Path | None = None) -> StoreNetwork:
    """Real coordinates if provided (OSRM when enabled, else haversine x circuity); otherwise simulated."""
    ncfg = cfg["network"]
    if coordinates_csv is not None and coordinates_csv.exists():
        c = pd.read_csv(coordinates_csv).set_index("store").loc[stores]
        lat, lon = c.lat.to_numpy(float), c.lon.to_numpy(float)
        if os.environ.get("ENABLE_ONLINE_ROUTING") == "1":
            try:
                return StoreNetwork(stores, np.c_[lat, lon], osrm_distance_matrix(lat, lon), "OSRM")
            except Exception as e:  # network failure -> documented offline fallback
                print(f"OSRM unavailable ({e}); falling back to haversine x circuity")
        return StoreNetwork(stores, np.c_[lat, lon], haversine_km(lat, lon) * ncfg["circuity_factor"], "HAVERSINE")
    return simulated_network(stores, ncfg["region_km"], ncfg["circuity_factor"], ncfg["location_seed"])


def transfer_cost(units: np.ndarray | float, km: np.ndarray | float, fixed: float, per_unit_km: float):
    """Modeled cost of one shipment: fixed handling + per unit-km. Zero units -> zero cost."""
    units = np.asarray(units, float)
    return np.where(units > 0, fixed + per_unit_km * units * np.asarray(km, float), 0.0)
