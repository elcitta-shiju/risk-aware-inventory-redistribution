# Risk-Aware Adaptive Inventory Redistribution Using Uncertainty-Aware Demand Forecasting


## The problem

In a chain of shops, one store often runs out of a product while another store has more than it can sell.
Moving stock between stores can save those lost sales, but every shipment costs money. The question is
**which** stock to move, **where**, and **when**.

## What this project does

- It **forecasts** next week's sales for every store and product, with a **likely range** (uncertainty).
- It uses the forecast and the current stock to **flag risk**: likely to run out, too much stock, or normal.
- It **recommends store-to-store transfers** using an **AI agent** trained with reinforcement learning
  (learning by trial and error in a simulator), and compares it with simple rules.
- It tests every approach on 8 recent weeks that were held back, and shows everything in a dashboard.

> **Important:** all transfer results are **simulated** from past data. Nothing was tried in real stores.
> Store locations are simulated, and shipping costs are assumptions (see Limitations).

## Key features

- **Demand forecast with an uncertainty range**: a machine-learning model (LightGBM) predicts next week's sales
  and gives a low-to-high likely range.
- **Stock risk flags**:
  - red = stockout risk;
  - amber = excess stock;
  - green = normal;
  - grey = no stock count in the data.
- **AI-recommended transfers**: from which store, to which store, how many units, distance and cost, with a
  plain-English reason.
- **Comparison with simple rules**: "cover the forecast", "cover a high estimate" and "cover a calibrated high
  estimate", plus "no transfers".
- **Cost sensitivity**: the test is repeated with each assumed cost changed, to see if conclusions still hold.
- **Streamlit dashboard** for store managers, with 5 tabs:
  - Decision Support
  - Store Overview
  - Results
  - About the Project
  - Technical Details

## How it works, step by step

1. **Data cleaning.** Real sales transactions and stock records are checked and cleaned:
   - returns are separated from sales;
   - negative stock values are set to 0 but counted;
   - missing stock counts are kept as "unknown", never treated as zero.
2. **Weekly data.** The data becomes one row per store × product × week, for the 30 best-selling products
   (chosen using training weeks only) in 40 stores over 46 weeks. The weeks are split into training, validation
   (checking) and test.
3. **Forecasting.** One LightGBM model for all stores and products is re-trained every week on past weeks only
   and predicts the next week.
4. **Uncertainty.** Four ways of building an 80% likely range were compared. The count-based (Poisson) range
   was chosen because it scored best on the validation weeks.
5. **Risk.** For every store and product: the chance of running out next week, the expected shortage, and
   whether stock is more than 8 weeks of likely sales.
6. **Store network.** The data has no store addresses, so the 40 stores are placed on a **simulated** map
   (250 km square). Shipping cost = 8 per shipment + 0.03 per unit per km (assumed).
7. **AI agent (reinforcement learning).**
   - A simulator replays each product across all stores for 8 weeks.
   - A PPO agent learns, for each store, how much stock to aim for. A shared matching step turns these targets
     into shipments from the nearest store with spare stock.
   - Two agents were trained, one seeing only the forecast and one also seeing the uncertainty range. Each was
     trained 5 times.
8. **Evaluation.**
   - All approaches are run on the 8 test weeks with exactly the same demand: 20 simulated demand runs plus the
     actual past sales.
   - We measure lost sales, leftover stock, shipments and money saved compared with making no transfers.
   - A cost-sensitivity test changes one assumed cost at a time.
9. **Dashboard.** A Streamlit app reads the saved results and models. It does not retrain anything.

## Dataset

Mendeley Data, *"Retail Transactions and Stocks Data"*:
- 125,751 sales transactions and 284,755 stock records;
- 40 stores and 2,326 products;
- 1 June 2025 to 24 April 2026.

It was chosen because it has both sales **and** stock histories, the same product is sold in many stores
(which is needed for transfers), and the two files agree with each other.


Details: [docs/data_audit.md](docs/data_audit.md).

The raw data files are **not** included in this repository; only small processed tables needed by the
dashboard are included. The dataset licences must be confirmed on the original Mendeley and Kaggle pages.

## Key results (test weeks, from the saved results files)

All money values are **modelled** costs in the dataset's currency, totalled over 30 products, 40 stores and
8 weeks, compared with making no transfers.

**Forecast.** The model's large-error score (RMSE) is **0.569**, better than an 8-week average (0.588) and
"same as last week" (0.746). The gain is modest, because 83% of store-weeks have zero sales.

**Uncertainty range.** It contains the actual sales in 97% of test weeks (target 80%). It is wider than needed
because weekly sales are small whole numbers.

**Money saved, simulated demand** (20 runs):

| Approach | Money saved vs no transfers | Lost sales avoided | Shipments |
|---|---|---|---|
| Rule: cover a high estimate | **2,276** | 30% | 278 |
| Uncertainty-aware AI agent | 2,021 ± 182 | 19% | 144 |
| Rule: cover a calibrated high estimate | 1,985 | 22% | 197 |
| AI agent (forecast only) | 1,885 ± 132 | 17% | 129 |
| Rule: cover the forecast | −187 (loses money) | 27% | 466 |

For the AI agents, ± is the spread across 5 separately trained agents.

**In plain words:**
- The simple **"cover a high estimate" rule saved the most** under simulated demand.
- With actual past sales, the **"calibrated high estimate" rule** saved the most (1,331, against 1,176 ± 129
  for the uncertainty-aware AI).
- **No approach won everywhere.**
- **Using uncertainty clearly helped the rules**: covering the high estimate instead of the forecast saved
  2,463 more.
- For the **AI**, the uncertainty information gave only a small gain (+136). Across the 5 training runs this is
  **not statistically reliable** (p = 0.22).
- The AI makes **far fewer shipments**. When shipping is expensive (fixed cost 16 per shipment, or 0.06 per
  unit per km and above), the AI agents save the most. At 32 per shipment, no approach is worth it.

Full results: [docs/experiments.md](docs/experiments.md) and
[results/reports/evaluation_report.md](results/reports/evaluation_report.md).

## Folder structure

```
config/config.yaml        all settings and assumptions (costs, weeks, model settings)
src/inventory_rl/         the project code
  data/                   safe extraction, loading, data checks, weekly table
  features/               forecasting inputs (lags, averages, stock cover, ...)
  forecasting/            LightGBM and simple baseline forecasts
  uncertainty/            likely ranges (Poisson, quantile, conformal)
  risk/                   chance of running out, excess stock, risk flags
  routing/                simulated store map, distances, shipping cost
  environment/            weekly simulator and its reinforcement-learning wrapper
  agents/                 PPO training and the per-store AI network
  baselines/              the simple rules
  evaluation/             test metrics, reports, cost sensitivity
  visualization/          saved figures
scripts/                  one script per step, plus run_pipeline.py for everything
dashboard/                Streamlit app (app.py), plain-language text (explain.py), file loading
.streamlit/config.toml    dashboard theme (light)
data/processed/           small processed tables the dashboard reads (raw data is not included)
models/                   trained AI agents (selected versions) and forecast settings
results/                  metrics, reports and figures produced by the pipeline
docs/                     data audit, data dictionary, methodology, experiments, limitations,
                          reproducibility
notebooks/                walkthrough notebook
tests/                    automated tests (pytest)
```

Useful documents:
- [Methodology](docs/methodology.md)
- [Assumptions and limitations](docs/assumptions_and_limitations.md)
- [Reproducibility](docs/reproducibility.md)

## How to install and run

**Requirements:** **Python 3.12** (3.13 is not supported; the dashboard crashed under 3.13 on our machine).
Windows, macOS or Linux; CPU is enough.

```bash
git clone https://github.com/<account>/risk-aware-inventory-redistribution.git
cd risk-aware-inventory-redistribution

# create and activate a virtual environment
py -3.12 -m venv .venv            # Windows   (macOS/Linux: python3.12 -m venv .venv)
.\.venv\Scripts\Activate.ps1      # Windows   (macOS/Linux: source .venv/bin/activate)

# install packages (the first line gets the smaller CPU-only PyTorch)
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# open the dashboard (run from the project folder so the theme is applied)
streamlit run dashboard/app.py
```

The dashboard opens at http://localhost:8501 and works straight away with the files in this repository.
Product names appear as codes (for example `PROD-100803`), because the readable names come from the raw data
file, which is not included.

**Optional: re-run the whole pipeline** (about 1 hour). Get the dataset ZIP archives, point
`INVENTORY_DATA_DIR` to their folder (copy `.env.example` to `.env` and edit it), then:

```bash
python scripts/run_pipeline.py      # all steps; add --skip-rl to reuse the trained agents
python -m pytest                    # run the tests
```

## Limitations and assumptions

- **Simulated transfers.** All results come from a simulator built on past data, not from real stores.
- **Simulated store locations.** The data has no addresses; distances come from a random map
  (straight line × 1.3).
- **Assumed costs.**
  - Shipping: 8 per shipment + 0.03 per unit per km.
  - Holding: 25% of item cost per year.
  - Lost sale: 1 × profit per item.
  - Profit and item cost per product come from the data.
- **No delivery delay.** Shipments are assumed to arrive before the week's sales.
- **Hidden demand.** When a shelf was empty, customers who wanted the item were not recorded, so past sales
  understate demand.
- **Missing stock counts.** 29% of store-weeks have no stock count; the simulator treats a missing starting
  count as 0.
- **Short history and sparse sales.** About 11 months of data; 83% of store-weeks have zero sales.
- **Limited scope.**
  - 30 products and 40 stores, one product at a time.
  - No truck or shelf capacity limits.
  - The AI agents were not retrained for each cost setting.

See [docs/assumptions_and_limitations.md](docs/assumptions_and_limitations.md).

## Future scope 

- Use **real store locations** and road distances, and the retailer's **real shipping and holding costs**.
- Add **delivery delays**, truck capacities, and shipping several products together.
- Retrain the AI agent for different cost levels, and try a rule that compares shipping cost with the expected
  benefit.
- Extend to more products and a longer history (a full year of seasons).
- Run a **pilot in a few real stores** and compare with the simulated results.


