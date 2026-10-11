# Experiments

All numbers below are copied from generated outputs:
- `results/metrics/evaluation_summary.csv`
- `results/metrics/results.json`
- `results/metrics/rl_saving_per_training_seed.csv`
- `results/reports/evaluation_report.md`

Rerun with `python scripts/run_evaluation.py`.

## Experiment matrix

| # | Policy | Uses forecast | Uses uncertainty | Learns |
|---|---|---|---|---|
| 1 | No redistribution | – | – | – |
| 2 | Rule-based (point): target = ceil(ŷ) | ✓ | – | – |
| 3 | Rule-based (upper bound): target = Poisson q90 | ✓ | ✓ | – |
| 4 | Rule-based (CQR bound): target = CQR upper | ✓ | ✓ | – |
| 5 | RL (point forecast): PPO, 3 training seeds | ✓ | – | ✓ |
| 6 | RL (forecast + uncertainty): PPO, 3 training seeds | ✓ | ✓ | ✓ |

**Shared conditions:**
- Test split: 8 weeks, 2026-02-23 → 2026-04-13.
- 30 SKUs × 40 stores, same simulator, same feasibility rules, same cost parameters.
- Same demand: 20 Poisson scenarios (fixed seeds) plus 1 historical replay.

**RL training:**
- 200k steps per run, training split only.
- The checkpoint for each run was chosen by validation saving.
- Best validation saving (Poisson):

  | Variant | seed 0 | seed 1 | seed 2 |
  |---|---|---|---|
  | point | 419 | 569 | 578 |
  | uncertainty | 956 | 547 | 962 |

## Results on the test split

Saving = modeled cost of no redistribution − policy cost, summed over 30 SKUs and 8 weeks. Costs are
**modeled**: unit margins are observed, while transfer and holding costs and the distances are
assumptions or simulations.

**Poisson demand scenarios** (mean ± std over scenarios; for RL, also over training seeds)

| Policy | Fill rate | Lost units | Transfer cost | Saving vs none | Units moved |
|---|---|---|---|---|---|
| No redistribution | 0.84 | 355.0 | 0 | 0 | 0 |
| Rule-based (point) | 0.89 | 260.6 | 4,682 | −187 ± 560 | 477 |
| Rule-based (upper bound) | 0.89 | 247.4 | 2,913 | **2,276 ± 641** | 325 |
| Rule-based (CQR bound) | 0.88 | 275.2 | 2,006 | 1,985 ± 390 | 213 |
| RL (point forecast) | 0.87 | 296.1 | 1,220 | 1,845 ± 388 | 127 |
| RL (forecast + uncertainty) | 0.87 | 287.1 | 1,481 | 2,028 ± 438 | 155 |

**Historical replay** (observed, stock-censored sales; one scenario)

| Policy | Fill rate | Lost units | Transfer cost | Saving vs none | Units moved |
|---|---|---|---|---|---|
| No redistribution | 0.91 | 209 | 0 | 0 | 0 |
| Rule-based (point) | 0.92 | 182 | 4,149 | −2,495 | 410 |
| Rule-based (upper bound) | 0.93 | 167 | 2,018 | 493 | 210 |
| Rule-based (CQR bound) | 0.93 | 166 | 1,213 | **1,331** | 114 |
| RL (point forecast) | 0.92 | 179 | 800 | 1,058 ± 18 (seeds) | 76 |
| RL (forecast + uncertainty) | 0.92 | 174 | 974 | 1,151 ± 149 (seeds) | 93 |

**Paired differences in saving** (same 20 Poisson scenarios, normal 95% CI)

| Comparison | Mean | 95% CI |
|---|---|---|
| Rule upper − Rule point | +2,463 | [2,301, 2,624] |
| RL uncertainty − RL point | +184 | [128, 239] |
| RL uncertainty − Rule upper | −248 | [−442, −53] |

**RL saving per training seed** (Poisson / replay)

| Variant | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| point | 1,861 / 1,053 | 1,805 / 1,042 | 1,867 / 1,077 |
| uncertainty | 2,078 / 1,087 | 1,754 / 1,044 | 2,253 / 1,320 |

## Interpretation (limited to what the results support)

1. **Uncertainty information helped the rule-based policy clearly.** The rule that covers the point
   forecast ships many single units and loses money: −187 under Poisson demand and −2,495 under replay.
   Targeting the upper prediction bound instead gives the largest Poisson saving (+2,276). The paired
   difference is large, and its CI excludes 0.
2. **For RL, uncertainty features gave a small, seed-dependent benefit.** Averaged over training seeds, the
   uncertainty-aware agent saves +184 more than the point agent on the same Poisson scenarios (CI excludes 0).
   Per seed, however, seed 1 of the uncertainty agent (1,754) is below all three point-agent seeds. With
   only 3 training seeds, the robustness of this effect is not established.
3. **RL did not beat the best rule-based policy under Poisson demand.** The upper-bound rule saves 248 more
   than the uncertainty-aware RL agent (CI [−442, −53]). In the single replay scenario, both RL agents save
   more than the upper-bound rule but less than the CQR rule. No method is a clear overall winner.
4. **RL agents are more conservative.** They move fewer units (127–155 vs 213–477 for the rules) and spend
   less on transfers, but leave more lost sales.
5. **Replay versus Poisson.** All savings are smaller under replay, which is consistent with demand censoring:
   observed demand at stores that were out of stock is zero. See the limitations doc.

These findings come from a simulator with simulated distances and assumed costs. They are sensitive to
the transfer-cost parameters, and they are not evidence of performance in real stores.

## Not done
- Sensitivity analysis over the cost assumptions (transfer fixed/per-km cost, lost-sale multiplier).
- More training seeds, and hyperparameter tuning of PPO beyond the documented design iterations.
- Lead time > 0, and SKUs beyond the top 30.
