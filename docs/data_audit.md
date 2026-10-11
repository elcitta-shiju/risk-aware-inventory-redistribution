# Data audit and dataset selection

Reproduce with `python scripts/run_data_audit.py` → `results/reports/data_audit.json`, EDA figures
`results/figures/eda_*.png`. Weekly-panel quality: `results/reports/preprocessing_report.json`.

## 1. Mendeley "Retail Transactions and Stocks Data" — **selected as primary**

**Sales (125,751 rows).** No duplicate rows; no missing values except the fully empty `Reason of Return`.
7,491 return rows, all with negative quantity and `Is Return = 1` (no contradictions). 40 stores,
2,326 SKUs, 2025-06-01 → 2026-04-24. Demand at SKU-store level is very sparse: a store-SKU pair has
a median of 2 transactions in the whole period. Each SKU is sold in 5–40 stores (median 19), so the
same product exists in many stores, which is what redistribution needs.

**Inventory (284,755 rows).** Validity intervals with no overlaps (0 overlapping intervals per
store/SKU/status) and no start-after-end rows. 22,701 open records (`9999-12-31`). 1,380 rows with
**negative** stock on hand (book-stock errors), clipped to 0 for physical stock but counted
(`neg_book_records`) and kept in `closing_book_stock`.

**Consistency check between the two files.** For 203,777 contiguous interval transitions, the change
in stock equals minus that day's net sales in **79.9%** of cases; 13.9% are increases without a sale
(receipts or transfers in). This supports reconstructing weekly stock-on-hand from the intervals.
2,588 of 50,447 sold store-SKU pairs have no stock record at all.

**Missing ≠ zero.** In the weekly top-30-SKU panel, 29.4% of store-SKU-weeks have no covering stock
record, and **497 units (3.6% of sales) were sold in weeks with no stock record**. Missing stock is
therefore kept as NaN and never silently converted to zero. Only the simulator needs a number, and it
uses 0 for such cells, explicitly flagged (`stock_missing_initial`).

**Why SKU level and not category level.** At store × category level the median cover is ~18 weeks and
only 0.8% of store-weeks start with zero stock, so there is little to redistribute. At SKU level (top-30 SKUs
by *training* sales), 7.5% of store-weeks start at zero stock, and in an exploratory check during the audit
~48% of those had another store holding ≥3 units of the same SKU in the same week, which is a genuine
redistribution opportunity (the reproducible version is `results/metrics/imbalance.csv`). The cost is intermittent demand: mean
0.25 units/store-week, 82.7% zero weeks, lag-1 autocorrelation ≈ 0.10.

**Limitations.** Anonymized IDs; no store locations; one channel and store type; ~11 months of history
(no full seasonal cycle); sales are censored by stock-outs (true demand unobserved when stock = 0);
licence/citation not included in the archive.

## 2. Kaggle `sales_data.csv` — secondary, not used for redistribution

5 stores × 20 products, daily, 2022-01-01 → 2024-01-30, no missing values, no duplicate keys,
category/region consistent per store-product. However:
- `Units Sold > Demand` in **21,000 rows (27.6%)**, so `Demand` cannot be true demand (sales cannot exceed demand).
- The next-day inventory agrees only weakly with `inventory − sold + ordered` (corr 0.53), so stock dynamics are not
  reliably reproducible.
- Only 5 stores, and it has the hallmarks of a synthetic dataset (no documentation).

## 3. Kaggle `retail_store_inventory.csv` — not used

5 stores × 20 products, daily, 2022-01-01 → 2024-01-01. Strong evidence of random generation:
- Lag-1 autocorrelation of units sold ≈ **−0.002** (no temporal structure to forecast).
- A store-product pair takes up to **5 different categories** and a store up to **4 regions** over time.
- Next-day inventory is uncorrelated with the stock-flow balance (corr ≈ 0.000).
- The provided `Demand Forecast` correlates **0.997** with same-day `Units Sold` and is negative in 673 rows.
  It is a target leak, not a forecast.

## 4. Integration decision

The datasets have **incompatible keys** (`STR-1xxx`/`PROD-xxxxxx` vs `S001`/`P0001`), non-overlapping
periods (2025–26 vs 2022–24) and different product domains. **They are not concatenated or merged.**
All forecasting, risk, RL and evaluation use the Mendeley data only. The Kaggle files remain audited
and documented in case they are needed for a separate sanity check.
