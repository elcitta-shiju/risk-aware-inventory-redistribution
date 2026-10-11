# Experiments

All numbers below are copied from generated outputs:
- `results/metrics/evaluation_summary.csv`, `results/metrics/results.json`
- `results/metrics/rl_saving_per_training_seed.csv`, `results/metrics/rl_saving_across_training_seeds.csv`
- `results/reports/evaluation_report.md`
- `results/metrics/cost_sensitivity.csv`, `results/reports/cost_sensitivity_report.md`,
  `results/figures/cost_sensitivity.png`

Rerun with `python scripts/run_evaluation.py` and `python scripts/run_sensitivity.py`.

## Experiment matrix

| # | Policy | Uses forecast | Uses uncertainty | Learns |
|---|---|---|---|---|
| 1 | No redistribution | – | – | – |
| 2 | Rule-based (point): target = ceil(ŷ) | ✓ | – | – |
| 3 | Rule-based (upper bound): target = Poisson q90 | ✓ | ✓ | – |
| 4 | Rule-based (CQR bound): target = CQR upper | ✓ | ✓ | – |
| 5 | RL (point forecast): PPO, 5 training seeds | ✓ | – | ✓ |
| 6 | RL (forecast + uncertainty): PPO, 5 training seeds | ✓ | ✓ | ✓ |

**Shared conditions:**
- Test split: 8 weeks, 2026-02-23 → 2026-04-13.
- 30 SKUs × 40 stores, same simulator, same feasibility rules, same cost parameters.
- Same demand: 20 Poisson scenarios (fixed seeds) plus 1 historical replay.

**RL training:**
- 200k steps per run, training split only; seeds 0–4 for each variant (10 runs).
- The checkpoint for each run was chosen by validation saving.
- Seeds 0–2 were trained under Python 3.13 and seeds 3–4 under Python 3.12, with identical package
  versions (see `docs/reproducibility.md`).
- Best validation saving (Poisson):

  | Variant | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
  |---|---|---|---|---|---|
  | point | 419 | 569 | 578 | 526 | 750 |
  | uncertainty | 956 | 547 | 962 | 671 | 916 |

## Results on the test split

Saving = modeled cost of no redistribution − policy cost, summed over 30 SKUs and 8 weeks. Costs are
**modeled**: unit margins are observed, while transfer and holding costs and the distances are
assumptions or simulations.

**Poisson demand scenarios** (mean ± std over scenarios; for RL, the ± here is over scenario × training
seed; the across-seed spread is in the next table)

| Policy | Fill rate | Lost units | Transfer cost | Saving vs none | Units moved |
|---|---|---|---|---|---|
| No redistribution | 0.84 | 355.0 | 0 | 0 | 0 |
| Rule-based (point) | 0.89 | 260.6 | 4,682 | −187 ± 560 | 477 |
| Rule-based (upper bound) | 0.89 | 247.4 | 2,913 | **2,276 ± 641** | 325 |
| Rule-based (CQR bound) | 0.88 | 275.2 | 2,006 | 1,985 ± 390 | 213 |
| RL (point forecast) | 0.87 | 293.5 | 1,304 | 1,885 ± 410 | 136 |
| RL (forecast + uncertainty) | 0.87 | 287.3 | 1,474 | 2,021 ± 425 | 155 |

**RL saving across training seeds** (mean ± std, ddof = 1, of each seed's mean saving; 5 seeds)

| Variant | Poisson: mean ± std | Poisson: min – max | Replay: mean ± std | Replay: min – max |
|---|---|---|---|---|
| RL (point forecast) | 1,885 ± 132 | 1,779 – 2,111 | 1,081 ± 99 | 985 – 1,249 |
| RL (forecast + uncertainty) | 2,021 ± 182 | 1,754 – 2,253 | 1,176 ± 129 | 1,044 – 1,321 |

Per seed (Poisson / replay):

| Variant | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|
| point | 1,861 / 1,053 | 1,805 / 1,042 | 1,867 / 1,077 | 1,779 / 985 | 2,111 / 1,249 |
| uncertainty | 2,078 / 1,087 | 1,754 / 1,044 | 2,253 / 1,320 | 1,961 / 1,121 | 2,058 / 1,309 |

**Historical replay** (observed, stock-censored sales; one scenario)

| Policy | Fill rate | Lost units | Transfer cost | Saving vs none | Units moved |
|---|---|---|---|---|---|
| No redistribution | 0.91 | 209 | 0 | 0 | 0 |
| Rule-based (point) | 0.92 | 182 | 4,149 | −2,495 | 410 |
| Rule-based (upper bound) | 0.93 | 167 | 2,018 | 493 | 210 |
| Rule-based (CQR bound) | 0.93 | 166 | 1,213 | **1,331** | 114 |
| RL (point forecast) | 0.92 | 178 | 846 | 1,081 ± 99 (seeds) | 80 |
| RL (forecast + uncertainty) | 0.92 | 174 | 955 | 1,176 ± 129 (seeds) | 91 |

**Paired differences in saving** (same 20 Poisson scenarios, RL averaged over seeds, normal 95% CI)

| Comparison | Mean | 95% CI |
|---|---|---|
| Rule upper − Rule point | +2,463 | [2,301, 2,624] |
| RL uncertainty − RL point | +136 | [88, 184] |
| RL uncertainty − Rule upper | −255 | [−457, −54] |

**Seed-level comparison** (Welch t-test; each training seed is one replicate, 5 vs 5)

| Comparison | Mean difference | Welch t | p-value |
|---|---|---|---|
| RL uncertainty − RL point (Poisson) | +136 | 1.35 | 0.217 |
| RL uncertainty − RL point (replay) | +95 | 1.30 | 0.231 |

## Interpretation (limited to what the results support)

1. **Uncertainty information helped the rule-based policy clearly.** The rule that covers the point
   forecast ships many single units and loses money: −187 under Poisson demand and −2,495 under replay.
   Targeting the upper prediction bound instead gives the largest Poisson saving (+2,276). The paired
   difference is large, and its CI excludes 0.
2. **For RL, the benefit of uncertainty features is small and not established across training seeds.**
   With 5 seeds, the uncertainty-aware agent has a higher mean saving (2,021 ± 182 vs 1,885 ± 132).
   Treating demand scenarios as replicates, the paired difference (+136) has a CI that excludes 0.
   Treating training seeds as replicates, however, the difference is not significant (Welch p = 0.22),
   and the seed ranges overlap: the worst uncertainty seed (1,754) is below the worst point seed (1,779).
   The scenario-level CI understates the variability that comes from training.
3. **RL did not beat the best rule-based policy under Poisson demand at the base costs.** The upper-bound
   rule saves 255 more than the uncertainty-aware RL agent (CI [−457, −54]), and none of the 5 uncertainty
   seeds exceeds the rule (best seed 2,253 vs 2,276). In the single replay scenario, both RL variants save
   more than the upper-bound rule but less than the CQR rule. No method is a clear overall winner.
4. **RL agents are more conservative.** They move fewer units (136–155 vs 213–477 for the rules) and spend
   less on transfers, but leave more lost sales.
5. **Replay versus Poisson.** All savings are smaller under replay, which is consistent with demand censoring:
   observed demand at stores that were out of stock is zero. See the limitations doc.

## Cost-assumption sensitivity

One assumed cost parameter is changed at a time; the others stay at the base values. All 6 policies are
re-simulated on the same test scenarios. **RL agents are the base-cost checkpoints and are not retrained**,
and they do not observe costs, so their transfers are identical in every setting; only the cost of those
transfers changes. Rule-based policies skip a shipment whose cost per unit exceeds the unit margin, but in
practice this cap almost never binds (units moved by the rules are unchanged except at the highest fixed
cost). Full tables: `results/reports/cost_sensitivity_report.md`. Figure: `results/figures/cost_sensitivity.png`.

**Poisson saving vs no redistribution** (RL = mean ± std across 5 training seeds; 12 of the 15 settings shown)

| Setting | Rule point | Rule upper | Rule CQR | RL point | RL uncertainty | Highest mean |
|---|---|---|---|---|---|---|
| **Base** (fixed 8, 0.03/unit-km, holding 0.25, shortage ×1) | −187 | 2,276 | 1,985 | 1,885 ± 132 | 2,021 ± 182 | Rule upper |
| Fixed cost 2 | 2,607 | 3,943 | 3,165 | 2,656 ± 263 | 2,888 ± 288 | Rule upper |
| Fixed cost 16 | −3,912 | 54 | 411 | 856 ± 48 | 865 ± 66 | RL uncertainty |
| Fixed cost 32 | −4,700 | −4,113 | −2,710 | −1,200 ± 397 | −1,446 ± 273 | No redistribution |
| Per unit-km 0.0075 | 531 | 2,794 | 2,309 | 2,091 ± 166 | 2,260 ± 211 | Rule upper |
| Per unit-km 0.06 | −1,143 | 1,585 | 1,552 | 1,609 ± 88 | 1,703 ± 145 | RL uncertainty |
| Per unit-km 0.12 | −2,975 | 209 | 688 | 1,059 ± 22 | 1,066 ± 77 | RL uncertainty |
| Holding 0.10 / yr | −201 | 2,230 | 1,948 | 1,857 ± 128 | 1,990 ± 179 | Rule upper |
| Holding 1.00 / yr | −114 | 2,504 | 2,166 | 2,024 ± 154 | 2,176 ± 201 | Rule upper |
| Shortage ×0.5 | −2,422 | −281 | 20 | 314 ± 42 | 299 ± 30 | RL point |
| Shortage ×2 | 4,284 | 7,389 | 5,915 | 5,027 ± 477 | 5,464 ± 537 | Rule upper |
| Shortage ×3 | 8,755 | 12,503 | 9,845 | 8,169 ± 821 | 8,907 ± 893 | Rule upper |

"Highest mean" compares means only, with no significance test. Close cases (for example RL point vs RL
uncertainty at fixed cost 16, 856 vs 865) are within one seed std.

**Findings:**
1. **The holding rate barely matters.** Over 0.10–1.00 per year, every saving changes by less than 300 and
   the ranking is unchanged (upper-bound rule first under Poisson, CQR rule first under replay).
2. **Transfer costs and the lost-sale value decide the ranking.** When transfers are cheap or lost sales are
   expensive, the rules that move more units (upper bound, CQR) save the most. When transfers become
   expensive (fixed cost 16, or ≥ 0.06 per unit-km) or lost sales are cheap (×0.5), the RL agents have the
   highest mean saving because they move fewer units. This is a consequence of their conservative behaviour
   learned at base costs, **not cost-awareness**: the agents do not observe costs and were not retrained.
3. **At a fixed cost of 32 per shipment, every policy loses money**, so not redistributing is cheapest.
   The rules' margin cap does not prevent this because it compares shipment cost with the full unit margin,
   not with the expected benefit of the transfer.
4. **The point-forecast rule has the lowest saving in 14 of 15 Poisson settings and in all 15 replay
   settings.** The exception is a lost-sale multiplier of 3, where it is ahead of RL point. Covering an
   uncertainty-based upper bound is consistently better than covering the point forecast.
5. **RL uncertainty vs RL point:** the uncertainty variant has the higher mean in 13 of the 15 settings
   (both under Poisson and under replay), but the difference is usually smaller than the across-seed std, consistent with the
   non-significant seed-level test at base costs.

These findings come from a simulator with simulated distances and assumed costs. They are not evidence of
performance in real stores.

## Not done
- Retraining the RL agents under each cost setting (would test whether RL can *learn* cost-appropriate
  behaviour; about 50 min of training per setting on the development laptop).
- A rule that compares transfer cost with the *expected* benefit (P(stockout) × margin) instead of the full margin.
- Hyperparameter tuning of PPO beyond the documented design iterations.
- Lead time > 0, and SKUs beyond the top 30.
