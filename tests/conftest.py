import numpy as np
import pandas as pd
import pytest

from inventory_rl.environment.data import ProductTensors
from inventory_rl.environment.simulator import RedistributionSim, SimConfig


def make_tensors(n_weeks: int = 4, product: str = "P1") -> ProductTensors:
    w, n = n_weeks, 3
    z = np.zeros((w, n))
    stock0 = np.tile([10.0, 0.0, 2.0], (w, 1))
    demand = np.tile([1.0, 2.0, 1.0], (w, 1))
    yhat = np.tile([1.0, 2.0, 0.5], (w, 1))
    return ProductTensors(
        product=product, split="test", weeks=list(pd.date_range("2026-01-05", periods=w, freq="W-MON")),
        stock0=stock0, receipts=z.copy(), demand_obs=demand, lam=demand.copy(), yhat=yhat,
        lo=z.copy(), hi=yhat + 1, cqr_lo=z.copy(), cqr_hi=yhat + 1, unit_margin=20.0, unit_cost=40.0,
    )


@pytest.fixture
def tensors():
    return {"P1": make_tensors()}


@pytest.fixture
def dist():
    return np.array([[0.0, 10.0, 50.0], [10.0, 0.0, 40.0], [50.0, 40.0, 0.0]])


@pytest.fixture
def sim(tensors, dist):
    cfg = SimConfig(transfer_fixed=5.0, transfer_per_unit_km=0.1, holding_rate_per_week=0.01, horizon=4)
    s = RedistributionSim(tensors, dist, cfg, demand_mode="replay", seed=0)
    s.reset("P1", 0)
    return s


@pytest.fixture
def tiny_sales():
    """Two stores, two products, three full weeks (Mon 2025-06-02 .. Sun 2025-06-22)."""
    rows = [
        ("2025-06-02", "S1", "A", 1, 0), ("2025-06-03", "S1", "A", 2, 0), ("2025-06-10", "S1", "A", 1, 0),
        ("2025-06-11", "S1", "A", -1, 1), ("2025-06-04", "S2", "B", 3, 0), ("2025-06-20", "S2", "A", 1, 0),
    ]
    df = pd.DataFrame(rows, columns=["transaction_date", "store", "product_no", "qty_sold", "is_return"])
    df["transaction_date"] = pd.to_datetime(df.transaction_date)
    df["sales_type"] = "Full Price"
    df["sales_amount"] = df.qty_sold * 10.0
    df["cogs"] = df.qty_sold * 5.0
    return df


@pytest.fixture
def tiny_inventory():
    rows = [  # store, product, start, end, qty
        ("S1", "A", "2025-05-01", "2025-06-01", 5), ("S1", "A", "2025-06-02", "2025-06-09", 2),
        ("S1", "A", "2025-06-10", None, 0),          # explicit zero stock (open record)
        ("S2", "B", "2025-05-15", "2025-06-15", 4),  # then no record -> unknown (NaN), not zero
        ("S2", "A", "2025-06-16", None, -1),         # negative book stock -> clipped to 0
    ]
    df = pd.DataFrame(rows, columns=["store", "product_no", "start_date", "end_date", "qty_on_hand"])
    df["is_open"] = df.end_date.isna()
    df["start_date"] = pd.to_datetime(df.start_date)
    df["end_date"] = pd.to_datetime(df.end_date)
    df["stock_status"] = "Full Price"
    return df
