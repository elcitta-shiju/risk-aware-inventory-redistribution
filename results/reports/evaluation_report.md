# Evaluation report (generated)

All values are computed by `scripts/run_evaluation.py` on the held-out **test** weeks. Costs are
**modeled** (transfer, holding and distance parameters are assumptions; unit margins are observed).
Store distances are **SIMULATED**. Simulation results are not evidence of
performance in a live retail operation.

- Products: 30 SKUs, stores: 40, Poisson demand scenarios: 20 (+1 historical replay)
- Interval method (selected on validation): `poisson`

## Policy comparison - Poisson demand scenarios (mean +/- std over scenario x training seed)

| policy | fill_rate | stockout_rate | lost_units | avg_closing_stock | transfer_cost | total_cost | saving_vs_noop | saving_vs_noop_std | n_transfers | units_transferred | avg_transfer_km | invalid_action_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| No redistribution | 0.84 | 0.32 | 355.00 | 2,742.51 | 0.00 | 23,461.05 | -0.00 | 0.00 | 0.00 | 0.00 | nan | 0.00 |
| Rule-based (point) | 0.89 | 0.27 | 260.55 | 2,730.31 | 4,681.55 | 23,647.59 | -186.54 | 559.95 | 465.65 | 477.35 | 66.79 | 0.00 |
| Rule-based (upper bound) | 0.89 | 0.27 | 247.35 | 2,700.35 | 2,913.41 | 21,185.08 | 2,275.97 | 641.12 | 277.80 | 325.15 | 70.91 | 0.00 |
| Rule-based (CQR bound) | 0.88 | 0.28 | 275.20 | 2,709.99 | 2,005.98 | 21,476.31 | 1,984.74 | 389.67 | 196.70 | 213.30 | 67.69 | 0.00 |
| RL (point forecast) | 0.87 | 0.29 | 296.12 | 2,719.27 | 1,220.12 | 21,616.54 | 1,844.51 | 387.92 | 120.20 | 126.65 | 68.33 | 0.00 |
| RL (forecast + uncertainty) | 0.87 | 0.29 | 287.13 | 2,714.81 | 1,480.83 | 21,432.74 | 2,028.31 | 438.32 | 145.13 | 155.05 | 69.06 | 0.00 |


## Policy comparison - Historical replay (observed, stock-censored sales as demand)

| policy | fill_rate | stockout_rate | lost_units | avg_closing_stock | transfer_cost | total_cost | saving_vs_noop | saving_vs_noop_std | n_transfers | units_transferred | avg_transfer_km | invalid_action_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| No redistribution | 0.91 | 0.25 | 209.00 | 2,663.50 | 0.00 | 16,997.36 | 0.00 | nan | 0.00 | 0.00 | nan | 0.00 |
| Rule-based (point) | 0.92 | 0.22 | 182.00 | 2,671.25 | 4,149.04 | 19,492.15 | -2,494.79 | nan | 410.00 | 410.00 | 70.65 | 0.00 |
| Rule-based (upper bound) | 0.93 | 0.22 | 167.00 | 2,641.62 | 2,017.64 | 16,504.24 | 493.12 | nan | 187.00 | 210.00 | 82.80 | 0.00 |
| Rule-based (CQR bound) | 0.93 | 0.22 | 166.00 | 2,641.00 | 1,212.99 | 15,666.63 | 1,330.74 | nan | 113.00 | 114.00 | 90.35 | 0.00 |
| RL (point forecast) | 0.92 | 0.23 | 179.00 | 2,647.88 | 800.26 | 15,939.72 | 1,057.64 | 18.03 | 76.00 | 76.00 | 84.27 | 0.00 |
| RL (forecast + uncertainty) | 0.92 | 0.23 | 173.67 | 2,644.54 | 974.07 | 15,846.78 | 1,150.58 | 148.50 | 92.67 | 92.67 | 83.94 | 0.00 |


## Paired differences in total saving (same scenarios)

| comparison | mean | 95% CI | n | reading |
|---|---|---|---|---|
| RL (forecast + uncertainty) minus RL (point forecast) | poisson | 183.8 | [128.3, 239.3] | 20 | positive (95% CI excludes 0) |
| RL (forecast + uncertainty) minus RL (point forecast) | replay | 92.9 | [nan, nan] | 1 | single scenario - no interval |
| RL (forecast + uncertainty) minus Rule-based (upper bound) | poisson | -247.7 | [-442.0, -53.3] | 20 | negative (95% CI excludes 0) |
| RL (forecast + uncertainty) minus Rule-based (upper bound) | replay | 657.5 | [nan, nan] | 1 | single scenario - no interval |
| Rule-based (upper bound) minus Rule-based (point) | poisson | 2,462.5 | [2,301.3, 2,623.7] | 20 | positive (95% CI excludes 0) |
| Rule-based (upper bound) minus Rule-based (point) | replay | 2,987.9 | [nan, nan] | 1 | single scenario - no interval |

The CI treats demand scenarios as independent replicates; it does not capture uncertainty
from the choice of products, period, or cost assumptions.

## RL robustness across training seeds (mean saving per scenario)

| policy | train_seed | poisson | replay |
|---|---|---|---|
| RL (forecast + uncertainty) | 0 | 2,078.0 | 1,087.5 |
| RL (forecast + uncertainty) | 1 | 1,754.4 | 1,044.1 |
| RL (forecast + uncertainty) | 2 | 2,252.6 | 1,320.2 |
| RL (point forecast) | 0 | 1,861.5 | 1,053.2 |
| RL (point forecast) | 1 | 1,805.5 | 1,042.2 |
| RL (point forecast) | 2 | 1,866.6 | 1,077.5 |


## Forecast accuracy (test, one-week-ahead)

| model | MAE | RMSE | WAPE | Bias |
|---|---|---|---|---|
| LightGBM | 0.2808 | 0.5692 | 1.1947 | 0.0115 |
| Naive (t-1) | 0.3049 | 0.7456 | 1.2974 | -0.0016 |
| MA(4) | 0.2822 | 0.6026 | 1.2010 | -0.0183 |
| MA(8) | 0.2855 | 0.5880 | 1.2149 | -0.0418 |
| Expanding mean | 0.3720 | 0.6137 | 1.5829 | 0.0174 |
| Zero forecast | 0.2350 | 0.6830 | 1.0000 | -0.2350 |
| LightGBM | weeks with y>0 | 0.8924 | 1.1760 | 0.6116 | -0.7794 |
| LightGBM | weeks with y=0 | 0.1633 | 0.3474 | nan | 0.1633 |


## Prediction intervals (test)

| method | coverage | nominal | upper_coverage | upper_nominal | mean_width | interval_score |
|---|---|---|---|---|---|---|
| poisson | 0.965 | 0.800 | 0.967 | 0.900 | 0.674 | 1.121 |
| qr | 0.934 | 0.800 | 0.938 | 0.900 | 0.400 | 1.245 |
| cqr | 0.934 | 0.800 | 0.938 | 0.900 | 0.400 | 1.245 |
| conformal | 0.785 | 0.800 | 0.915 | 0.900 | 0.015 | 2.561 |
