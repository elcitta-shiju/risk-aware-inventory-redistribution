"""Plain-language labels and explanations for the dashboard (pure functions, no Streamlit)."""

from __future__ import annotations

import math

import pandas as pd

# Risk status shown to users. Colours: red = stockout risk, amber = excess stock, green = normal.
RISK_COLORS = {"Stockout risk": "#D64545", "Excess stock": "#E9A23B", "Normal": "#3BA55C",
               "No stock record": "#9AA5B1"}
RISK_ORDER = list(RISK_COLORS)
RISK_HELP = {
    "Stockout risk": "Stock is unlikely to cover next week's demand.",
    "Excess stock": "More stock than is likely to sell in the next 8 weeks.",
    "Normal": "Stock is in line with expected demand.",
    "No stock record": "The data has no stock count for this store and product this week.",
}
_RISK_CLASS_MAP = {"high stockout risk": "Stockout risk", "below upper bound": "Stockout risk",
                   "excess": "Excess stock", "balanced": "Normal", "unknown stock": "No stock record"}

# Evaluation policy names -> names shown to users.
POLICY_NAMES = {
    "No redistribution": "No transfers",
    "Rule-based (point)": "Rule: cover the forecast",
    "Rule-based (upper bound)": "Rule: cover a high estimate",
    "Rule-based (CQR bound)": "Rule: cover a calibrated high estimate",
    "RL (point forecast)": "AI agent",
    "RL (forecast + uncertainty)": "Uncertainty-aware AI agent",
}

COST_PARAM_NAMES = {
    "transfer_fixed": "Fixed cost per shipment",
    "transfer_per_unit_km": "Cost per item per km",
    "holding_rate_per_year": "Yearly cost of holding stock (share of item cost)",
    "shortage_cost_multiplier": "Cost of a lost sale (times the profit per item)",
}


def risk_status(risk_class: str) -> str:
    return _RISK_CLASS_MAP.get(risk_class, "Normal")


def store_label(code: str) -> str:
    return f"Store {str(code).split('-')[-1]}"


def week_label(ts) -> str:
    return f"Week of {pd.Timestamp(ts):%d %b %Y}"


def money(x: float) -> str:
    return f"{x:,.0f}" if abs(x) >= 100 else f"{x:,.2f}".rstrip("0").rstrip(".")


def units(x: float) -> str:
    x = float(x)
    if x.is_integer():
        n = int(x)
        return f"{n} unit" if n == 1 else f"{n} units"
    return f"{x:.1f} units"


def chance(p: float) -> str:
    return f"{100 * p:.0f}%"


def pct_change(new: float, base: float) -> float:
    return float("nan") if base == 0 else 100.0 * (new - base) / base


def transfer_reason(src: str, dst: str, qty: int, km: float, cost: float, src_stock: float,
                    src_expected: float, dst_stock: float, dst_expected: float, dst_p_out: float,
                    dst_no_record: bool = False) -> str:
    """One plain-English sentence explaining a recommended transfer."""
    if dst_no_record:
        need = f"{dst} has no stock record this week (treated as empty) and expects to sell about {dst_expected:.1f}"
    elif dst_stock <= 0:
        need = f"{dst} is out of stock and expects to sell about {dst_expected:.1f} next week"
    elif dst_p_out >= 0.5:
        need = (f"{dst} is likely to run out next week ({chance(dst_p_out)} chance): it has {units(dst_stock)} "
                f"and expects to sell about {dst_expected:.1f}")
    else:
        need = (f"{dst} may run short next week ({chance(dst_p_out)} chance of running out): it has "
                f"{units(dst_stock)} and expects to sell about {dst_expected:.1f}")
    spare = f"{src} has spare stock ({units(src_stock)} against about {src_expected:.1f} expected sales)"
    return f"{need}. {spare}. Move {units(qty)} over {km:.0f} km, estimated cost {money(cost)}."


def no_transfer_reason(store: str, status: str, p_out: float, stock: float, expected: float) -> str:
    if status == "No stock record":
        return (f"{store} has no stock record this week, so no transfer is suggested for it. "
                f"Expected sales: about {expected:.1f}.")
    if status == "Stockout risk" and stock <= 0:
        return (f"{store} is out of stock (expected sales about {expected:.1f}), but no transfer to it is "
                f"recommended this week: other stores have little spare stock of this product, or the "
                f"recommended approach judged a shipment not worthwhile.")
    if status == "Stockout risk":
        return (f"{store} may run short ({chance(p_out)} chance of running out), but no transfer to it is "
                f"recommended this week: other stores have little spare stock of this product, or the "
                f"recommended approach judged a shipment not worthwhile.")
    if status == "Excess stock":
        return (f"{store} holds more than it is likely to sell ({units(stock)} against about {expected:.1f} "
                f"expected next week), but no other store needs this product enough to justify a shipment.")
    return (f"{store} has enough stock for next week ({units(stock)} against about {expected:.1f} expected), "
            f"so no transfer is needed.")


def best_policy(means: dict[str, float]) -> str:
    """Name of the highest mean saving; 'No transfers' when every approach loses money."""
    finite = {k: v for k, v in means.items() if v is not None and not math.isnan(v)}
    if not finite or max(finite.values()) <= 0:
        return "No transfers"
    return max(finite, key=finite.get)
