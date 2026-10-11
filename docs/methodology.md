# Methodology

```
raw ZIPs ──extract──► data/raw ──audit──► panel (store×SKU×week) ──features──► rolling-origin LightGBM
   │                                                                                │
   │                                      prediction intervals (Poisson/QR/CQR/conformal) ◄┘
   │                                                    │
   │                    risk measures (P(stockout), expected shortage, surplus, excess)
   │                                                    │
   └── store network (SIMULATED coords) ──► transfer cost ──► weekly simulator (Gymnasium)
                                                              │            │
                                                 rule-based policies    PPO agents (point / +uncertainty)
                                                              └──── same test scenarios ────► evaluation
```

## 1. Unit of analysis and splits
Weekly (Mon–Sun) × store × SKU for the 30 SKUs with the most **training-period** units sold, across all
40 stores (1,200 series × 46 full weeks, 2025-06-02 → 2026-04-19). Partial first/last weeks are dropped.

| split | weeks | used for |
|---|---|---|
| train | 2025-06-02 → 2025-12-22 (30) | model fitting, RL training episodes |
| val | 2025-12-29 → 2026-02-16 (8) | forecast model / interval method selection, RL checkpoint selection |
| test | 2026-02-23 → 2026-04-13 (8) | final evaluation only |

The series universe and the SKU selection use training data only. Weekly opening stock is the
reconstructed end-of-day stock on the preceding Sunday (sum over all stock-status pools of the SKU).

## 2. Demand forecasting (Stage 3)
- Target: `gross_qty` (units sold excluding returns), a **censored** proxy for demand.
- Baselines: naive (t−1), MA(4), MA(8), expanding mean, zero forecast.
- Model: one global LightGBM over all series (pooling is essential with 83% zero weeks). Features
  are lags, rolling statistics, weeks since last sale, lagged promo share, opening stock, cover, lagged
  SKU-wide and store-wide totals, calendar, and store/SKU codes. All features use information up to t−1,
  except opening stock, which is known at decision time. A test perturbs future targets and asserts that
  past features are unchanged.
- **Rolling origin:** for every week t ≥ 2025-07-28 the model is refit on weeks < t and predicts t.
  Every forecast, including those the simulator uses for training weeks, is therefore out-of-sample.
- Selection: objective {Poisson, L2} × num_leaves {7, 15, 31} chosen by validation RMSE. RMSE is used
  because the forecast serves as an *expected* demand. MAE rewards forecasting the median, which is
  0 for most series (the zero forecast has the lowest MAE).

## 3. Uncertainty (Stage 4)
Nominal 80% central intervals (α = 0.2), integer-valued:
- **Poisson** quantiles around the point forecast (parametric).
- **QR:** LightGBM quantile regression at τ = 0.1 / 0.9 (rolling origin, like the point model).
- **CQR:** conformalized QR (Romano et al., 2019).
- **Normalized split conformal** around the point forecast (scale = rolling std).

The conformal variants use a *rolling* calibration window of the previous 8 weeks of out-of-sample
scores, so week t is never calibrated on itself or the future. The method is chosen by **validation
interval score** (Winkler score, a proper scoring rule combining width and misses). Reported metrics are
two-sided coverage, one-sided upper-bound coverage, mean width and interval score per split, plus
coverage per SKU.

Interpretation for decisions: the upper bound `hi` is the stock level that covers next-week demand
with ~90% (one-sided) probability. Wide intervals at a destination raise its target; at a source they
reduce what it can safely give.

## 4. Inventory risk (Stage 5)
Defined in `risk/assessment.py`: `p_stockout = P(D ≥ s)` under Poisson(ŷ) is a *model-based* probability.
Its calibration is checked on held-out weeks with stock > 0, where "D ≥ s" is observable as "sold out"
(reliability table + Brier score). Expected shortage is E[(D−s)+]. Also computed: shortfall versus the
upper bound, excess beyond the 8-week upper demand bound, transferable surplus (point / upper), and per
SKU-week imbalance = min(total shortfall, total surplus).

## 5. Distances and costs (Stage 6)
No store locations exist in the data, so coordinates are **simulated**: seeded uniform points in a
250 km square, with road distance = Euclidean × 1.3 circuity. Every output is labelled `SIMULATED`.
`routing/network.py` also supports real coordinates (haversine, or OSRM road distances with caching
and offline fallback) when a `store,lat,lon` CSV is provided. Transfer cost = fixed per shipment +
per unit-km (assumptions).

## 6. Simulator and RL environment (Stage 7)
One episode = one SKU, all 40 stores, 8 weekly steps. Each week:
1. Transfers are decided.
2. Feasibility: src ≠ dst, positive integer quantity, and a source keeps ≥ ceil(ŷ) units, which also
   rules out negative stock.
3. Transfers arrive (lead time 0).
4. Historical net receipts/adjustments are replayed.
5. Demand is realized.
6. Sales = min(demand, stock), and lost sales are recorded.
7. Cost = observed unit margin × lost units + holding (25%/yr of observed unit cost) × closing stock
   + transfer cost.

Inventory conservation and non-negativity are checked at every step, raising an error if violated.

**Demand scenarios.** *Replay* uses the observed sales of that week; it is censored, so it understates
demand at stores that were out of stock. *Poisson* uses Poisson(λ), where λ is a centred 5-week rolling
mean of observed sales within the same split. λ is a simulator ground-truth proxy that agents never
observe; agents only see as-of forecasts.

**Reward** = (cost of a parallel no-transfer trajectory with identical demand and receipts − policy
cost) / 10. The no-transfer cost does not depend on the agent's actions, so the optimal policy is
unchanged, but variance drops.

**Observation:** per store, [log1p stock, log1p available, ŷ]. The uncertainty variant adds
[Poisson hi, CQR lo, CQR hi]. Two globals: episode progress and log unit margin.

**Action:** Box(−1, 1)^40, a per-store target stock `round(ŷ + 2a)`. A deterministic matching layer
(`environment/matching.py`, shared with the rule-based baseline) sends units from the nearest stores
with stock above max(target, protection) to stores below target. Every action is therefore feasible,
and the action space stays 40-dimensional rather than 40² × quantity levels.

Design iterations, all executed and kept in the history:
- (a) Raw offer/request actions with a flat MLP: worse than no-transfer on validation.
- (b) Same with a dead-zone: collapsed to "never transfer".
- (c) Target-level actions: learns.

## 7. RL training (Stage 8)
PPO (Stable-Baselines3 2.9) with a **parameter-shared per-store actor** (`agents/policy.py`): the same
small network maps each store's features plus a mean/max-pooled network embedding to that store's
action, deep-sets style. This makes the policy permutation-equivariant across stores. Training settings:
8 parallel training-split environments, 200k steps, γ = 0.97, 3 seeds per variant. Every 20k steps the
deterministic policy is evaluated on the validation split, and the best checkpoint is kept (selection
uses validation only).

## 8. Baselines and evaluation (Stages 9–10)
Policies:
- No redistribution.
- Rule-based with target = ceil(ŷ) (*point*), Poisson upper bound (*upper*), or CQR upper bound (*cqr*).
  Destinations are ranked by need then stockout probability and served by the nearest surplus; a shipment
  is skipped if its cost per unit exceeds the unit margin.
- RL with point forecasts.
- RL with forecasts + uncertainty.

All are evaluated on the **test** split (8 weeks, 30 SKUs) under the same 1 replay scenario + 20 Poisson
scenarios with fixed seeds. Metric definitions are in `evaluation/experiment.py`. Variability is
reported across scenarios and training seeds, and paired differences on identical scenarios come
with normal 95% CIs.
