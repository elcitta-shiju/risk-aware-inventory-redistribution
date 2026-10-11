# Assumptions and limitations

## Observed vs simulated vs assumed

| Quantity | Status | Source / value |
|---|---|---|
| Sales, returns, sales type | **observed** | Mendeley sales file |
| Stock on hand (weekly) | **observed** (reconstructed) | Mendeley validity intervals; negatives clipped; missing kept NaN |
| Net receipts / adjustments | **derived** from observations | closing − opening + net sales |
| Unit margin, unit cost per SKU | **observed** | training-period sales amount and COGS |
| Store coordinates and distances | **SIMULATED** | seeded uniform points in a 250 km square, Euclidean × 1.3 |
| Transfer cost | **assumption** | 8 per shipment + 0.03 per unit-km |
| Holding cost | **assumption** | 25% of unit cost per year |
| Lost-sale cost | assumption on observed basis | 1 × observed unit margin |
| Transfer lead time | **assumption** | 0 weeks (arrive before the week's demand) |
| Source protection | design rule | a source keeps ≥ ceil(point forecast) units |
| Poisson demand scenarios | **simulation** | Poisson(centred 5-week mean of observed sales) |
| Excess horizon | assumption | 8 weeks |

All assumptions are configurable in `config/config.yaml`.

## Limitations
1. **Censored demand.** Sales cannot exceed stock, so true demand at stocked-out stores is unknown.
   Replay scenarios therefore *understate* the benefit of moving stock to empty stores, while Poisson
   scenarios rely on smoothed observed sales and still inherit some censoring.
2. **No real geography.** Distances are simulated, so transfer costs and routes are illustrative. The
   OSRM integration is implemented but not exercised (no coordinates exist).
3. **Short history.** About 11 months: no full seasonal cycle, and val/test are only 8 weeks each.
   Results may not generalize to other seasons.
4. **Sparse SKU demand** (83% zero weeks). Point-forecast gains over simple baselines are modest, and
   two-sided interval coverage overshoots its nominal level because of discreteness. CQR makes
   practically no correction because the conformal quantile falls in the zero-demand mass.
5. **Scope.** 30 SKUs × 40 stores; one product per episode (no joint shipments across SKUs); no
   vehicle capacities, no store capacity limits, no replenishment decisions (historical receipts are
   replayed as exogenous).
6. **Simulation only.** The RL agent is trained and evaluated in a simulator built from historical data.
   Nothing here is validated in a live retail operation.
7. **Statistical scope of comparisons.** Confidence intervals treat demand scenarios as replicates. They do
   not cover uncertainty from the cost assumptions, SKU selection or time period. Results are sensitive
   to the transfer-cost parameters.
8. **Novelty.** No claim of novelty or superiority over the literature is made. Positioning against
   prior work (e.g., the IEEE papers collected in `project data/ieee papers`) requires a literature review.
9. **Dataset licences** were not included in the archives and must be confirmed on the source pages.
