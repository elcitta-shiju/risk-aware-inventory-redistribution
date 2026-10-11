# Dashboard walkthrough and team guide

This guide explains the whole project and every part of the dashboard in simple English. 
**How to open the dashboard** (from PowerShell):

```powershell
cd prototype
.\.venv\Scripts\Activate.ps1
streamlit run dashboard/app.py
```

It opens in a browser at http://localhost:8501. The first load can take a little while, because the
dashboard prepares the simulator and searches for a good starting example. After that it is fast.

**Where the numbers in this guide come from.** Every number below was read from the saved project results
(`results/`, `data/processed/`, `models/`) or recomputed with the same functions the dashboard uses. If the
pipeline is re-run, small numbers may change. Where we are not sure how something works, the guide says so.

**Words used in this guide**

- **Store-week**: one store in one week. **Store-product pair**: one store and one product.
- **SKU** (stock-keeping unit): one specific product, for example `PROD-100803`. We use "product" for SKU.
- **Simulation**: a computer re-play of what *would* have happened, using past data. Nothing in this project
  was tried in a real store.

---

## Contents

1. [Project overview](#part-1--project-overview)
2. [Every dashboard feature](#part-2--every-dashboard-feature)
3. [Hands-on exercises](#part-3--hands-on-exploration-exercises)
4. [10-minute demo script](#part-4--10-minute-demo-script-for-our-teacher)
5. [How the project was built](#part-5--how-the-project-was-built-step-by-step)
6. [20 likely teacher questions](#part-6--20-likely-teacher-questions)
7. [Limitations and assumptions](#part-7--limitations-and-assumptions)
8. [Glossary](#part-8--glossary)

---

## Part 1 — Project overview

### What the project does (5 sentences)

1. It takes about **11 months** of real sales and stock records from a chain of 40 shops (June 2025 to
   April 2026) and turns them into a weekly table for the 30 best-selling products.
2. A **machine-learning model** (a program that learns patterns from past data) **forecasts** (predicts) how
   many units of each product each store will sell next week, and gives a **likely range** around that guess.
3. From the forecast and the current stock, it gives every store a **risk flag**: likely to run out (red),
   too much stock (amber), normal (green), or no stock count in the data (grey).
4. It **recommends moving stock between stores** (a "transfer"), using either an **AI agent** trained by
   **reinforcement learning** (learning by trial and error in a simulator) or a simple rule.
5. It tests every approach on 8 recent weeks that were held back, and reports how many lost sales each
   approach avoids and how much money it saves after paying for shipping.

### Who would use it and for what

| Who | What they would use it for |
|---|---|
| **Store manager** | Check if their store is about to run out of a product next week, and whether stock is coming from (or should go to) another store. |
| **Regional / inventory planner** | See which stores need attention this week, and approve or reject suggested transfers between stores. |
| **Management** | Compare approaches (no transfers, simple rules, AI) and see if transfers are worth their shipping cost. |
| **Our teacher / examiners** | Understand the data, the method, the results and the limitations. |

> Important: this is a **prototype on past data**. The recommendations show what the system *would have*
> suggested. They have not been used in real stores.

---

## Part 2 — Every dashboard feature

### How to read this part

For each feature we give five things:

- **a) Shows**: what you see on screen.
- **b) Means**: the real-world meaning, with a small store example.
- **c) How it is made**: the file and function that produce it, the data or model it uses, and the
  calculation in simple words.
- **d) Used for**: how a manager or planner would use it.
- **e) Example**: a specific week, store and product where you can see it.

**Main files behind the dashboard**

| File | What it does |
|---|---|
| `dashboard/app.py` | Builds every tab, chart, card and table. |
| `dashboard/explain.py` | Turns numbers into plain words: risk status names and colours, store and week labels, the "Why" sentences, approach names. |
| `dashboard/data_access.py` | Reads saved files, finds product names, picks which trained AI agent to use, checks which pipeline results exist. |
| `src/inventory_rl/...` | The project code the dashboard calls: risk formulas, the simulator, the rules, the AI agent loader, shipping cost. |
| `.streamlit/config.toml` | The light theme (white background, dark text). |

**The two kinds of data the dashboard shows** (this matters a lot):

1. **Recorded data**: what really happened in the shops (stock counts and sales from the dataset). Used by
   the **Store Overview** tab and by the forecast chart.
2. **Simulated data**: what stock *would have been* if an approach had been running since the first test
   week (week of 23 Feb 2026), using the sales that really happened. Used by the **Decision Support** tab.

These two can differ for the same store and week. Exercise 1.4 shows an example.

---

### 2.0 Things on every page

#### Page title "Stock Transfer Planner" and the line under it

- **a) Shows**: the dashboard name, and the caption "Forecasts next week's demand for each store and product,
  flags stores at risk, and suggests moving stock between stores. Built on past sales and stock data.
  Transfer results are simulated."
- **b) Means**: a one-line summary of the tool. The last sentence warns that transfer results are not real.
- **c) How it is made**: `st.title` and `st.caption` at the top of `dashboard/app.py`. The browser tab shows
  the same name and a 📦 icon (`st.set_page_config`).
- **d) Used for**: telling a new user what the page is for.
- **e) Example**: visible on every tab.

#### The tab bar (5 tabs)

- **a) Shows**: Decision Support, Store Overview, Results, About the Project, Technical Details.
- **b) Means**: they are ordered by who needs them most: the manager's daily decision first, the details last.
- **c) How it is made**: `st.tabs([...])` in `dashboard/app.py`. The selected tab is underlined in blue.
- **d) Used for**: moving between tasks.
- **e) Example**: click "Results" to jump to the test results.

#### Light theme

- **a) Shows**: white background, dark text.
- **b) Means**: easier to read and to print.
- **c) How it is made**: `.streamlit/config.toml` (`base = "light"`, `backgroundColor = "#FFFFFF"`). Streamlit
  only reads this file when you start the dashboard **from the project folder**.
- **d) Used for**: readability and screenshots.
- **e) Example**: every page.

#### "Deploy" button and ⋮ menu (top right)

- **a) Shows**: built-in Streamlit controls (settings, rerun, print, and so on).
- **b) Means**: these are **not** project features. "Deploy" would publish the app online; we do **not**
  use it (the data and results stay on our computer).
- **c) How it is made**: added automatically by Streamlit.
- **d) Used for**: you can ignore them. "Rerun" in the menu reloads the page.
- **e) Example**: top-right corner of every page.

#### Yellow warning "… is not available yet"

- **a) Shows**: for example "Risk information is not available yet. The project pipeline has to be run first."
- **b) Means**: a result file is missing, so that part cannot be drawn.
- **c) How it is made**: the `need()` function in `dashboard/app.py` checks whether the file was loaded.
- **d) Used for**: knowing that you must run `python scripts/run_pipeline.py` first.
- **e) Example**: you will not see it on our machine, because all results exist.

#### Built-in tools on charts and tables

- **Charts**: hover over a point or bar to see a **tooltip** (a small box with the exact numbers). Charts also
  have a small menu (⋯) to save the chart as an image, and a full-screen button. These are Altair and Streamlit
  built-ins (**Altair** is the charting library).
- **Tables**: hover over a table to see a toolbar with **download as CSV**, **search** and **full screen**.
  Click a column header to sort. These are Streamlit built-ins.

---

### 2.1 Tab 1 — Decision Support

The tab's purpose: pick a week, a store and a product, and see the demand outlook, the stock risk and the
recommended transfer on one screen.

#### Heading "Decision support" and its caption

- **a) Shows**: "Choose a week, a store and a product to see the demand outlook, the stock risk and the
  recommended transfer."
- **b) Means**: instructions for the tab.
- **c) How it is made**: `st.header` and the `explain_line()` helper (a grey caption) in `dashboard/app.py`.
- **d) Used for**: orientation.
- **e) Example**: top of the tab.

#### Dropdown: "Planning week"

- **a) Shows**: the 8 test weeks, labelled "Week of 23 Feb 2026" … "Week of 13 Apr 2026".
- **b) Means**: the week you are planning for. Think of it as "it is Monday morning of this week; what should
  we do?". A week runs Monday to Sunday.
- **c) How it is made**: the list comes from the simulator's test weeks (`ctx.tensors["test"]`, built by
  `src/inventory_rl/environment/data.py`). Labels come from `explain.week_label()`. Only **test** weeks are
  offered, because those are the weeks the trained models never saw.
- **d) Used for**: choosing the decision week.
- **e) Example**: "Week of 6 Apr 2026" has many interesting cases (see Part 4).

#### Dropdown: "Store"

- **a) Shows**: the 40 stores, labelled "Store 1006", "Store 1010", …
- **b) Means**: the store you are responsible for. Store numbers are the dataset's codes `STR-1006` etc. with the
  "STR-" removed. They are anonymous; we do not know the real shop names or towns.
- **c) How it is made**: list from the store network (`ctx.network.stores`); labels from `explain.store_label()`.
- **d) Used for**: looking at your own store.
- **e) Example**: "Store 1044".

#### Dropdown: "Product"

- **a) Shows**: the 30 products, for example "Scholar Footwear Flight Classic Zephyr (PROD-100803)".
- **b) Means**: the product (SKU) you are checking. The readable part is the product subcategory from the
  data; the code in brackets is the exact product, because several products share one subcategory name.
- **c) How it is made**: `data_access.product_labels()` reads the "Product Subcategory" column from the
  extracted sales file. If that file is missing, only the code is shown.
- **d) Used for**: checking one product at a time.
- **e) Example**: PROD-100803 (a "Scholar Footwear Flight Classic Zephyr" item; profit per unit about 41).

#### Default selection (what you see when the page first opens)

- **a) Shows**: Week of 13 Apr 2026, Store 1080, product PROD-100800.
- **b) Means**: the dashboard opens on a case that **has** a recommended transfer, so you immediately see what
  the tool does.
- **c) How it is made**: `example_case()` in `dashboard/app.py` starts from the latest week and looks for the
  largest transfer recommended by the AI agent; it opens on that transfer's receiving store and product.
- **d) Used for**: a quick first impression.
- **e) Example**: on open you see "Store 1080 is out of stock … Move 1 unit over 42 km, estimated cost 9.27."

#### Radio buttons: "Recommendation approach" (and its ? help icon)

- **a) Shows**: two choices: **Uncertainty-aware AI agent** (default) and **Rule: cover a high estimate**.
  The help icon says: "The AI agent learned when to move stock from past data. The rule tops every store up to
  a high estimate of next week's demand. The Results tab compares both."
- **b) Means**: who makes the recommendation.
  - The **AI agent** looks at each store's stock, its forecast and the forecast's likely range, and decides a
    target stock level for every store.
  - The **rule** is simple: "every store should hold at least the high estimate of next week's sales; fill
    any gap from the nearest store with spare stock."
- **c) How it is made**: `recommender()` in `dashboard/app.py`.
  - AI: loads the trained model chosen by `data_access.best_validation_run("rl_uncertainty")`. It picks the
    uncertainty-aware training run with the best score on the **validation weeks** (earlier check weeks, not
    the test weeks). Today that is **Run 3** (file `models/rl/rl_uncertainty_seed2/best_model.zip`). The model
    is loaded with `load_policy()` in `src/inventory_rl/agents/ppo.py`.
  - Rule: `RuleBasedPolicy("upper")` in `src/inventory_rl/baselines/rules.py`.
- **d) Used for**: comparing what the two approaches suggest for the same situation.
- **e) Example**: Week of 6 Apr 2026, Store 1044, PROD-100803. The AI sends **1 unit**; the rule sends
  **4 units** (Exercise 1.3).

> **How the recommendation is calculated (applies to the whole tab).** The function `plan()` in
> `dashboard/app.py` starts at the first test week (23 Feb 2026) with the recorded stock. It then runs the
> chosen approach week by week until the planning week, using the sales that really happened. This is the
> same way the approaches were tested. It returns:
> - the **simulated stock** of every store in the planning week, and
> - the **transfers** the approach recommends for that week.
>
> Because each approach makes different transfers in earlier weeks, the two approaches can show
> **different stock** for the same store and week.

#### Chart: "Demand outlook"

The chart has several parts.

| Part | a) Shows | b) Means |
|---|---|---|
| **Blue line with dots: "Forecast"** | The model's forecast for each week: how many units it expected to sell. | Store 1044 was expected to sell 5.7 units of PROD-100803 in the week of 6 Apr. |
| **Dark line with dots: "Actual sales"** | Units really sold that week (returns not counted). | Store 1044 actually sold 2 units that week. |
| **Light blue shaded band** | The **likely range** of sales for each week (from a low to a high value). | "Probably between 3 and 9 units." Sales outside the band are a surprise. |
| **Grey dashed vertical line** | The planning week. | "This is the week we are deciding for." Its actual sales are shown only for comparison; a real manager would not know them yet. |
| **Legend** (top) | Which colour is the forecast and which is actual sales. | — |
| **Y-axis "Units per week"** | Number of units. | Values are small (0–9), because one store sells few units of one product. |
| **X-axis** | The last 16 weeks up to the planning week. | Shows recent history. |
| **Tooltip** (hover) | Week, series name, units with one decimal. | — |

- **c) How it is made**: data from `data/processed/forecasts.parquet` (made by `scripts/run_forecasting.py`).
  - Forecast `yhat`: a **LightGBM** model (a fast machine-learning method that combines many small decision
    trees) was re-trained every week on all earlier weeks only, then predicted the next week. So every forecast
    is a real "one-week-ahead" forecast. Code: `rolling_origin_forecast()` in
    `src/inventory_rl/forecasting/models.py`.
  - Actual sales `y`: units sold that week, excluding returns (`gross_qty` in the weekly table).
  - Band `lo` to `hi`: the 10th and 90th **percentiles** of a **Poisson distribution** whose average is the
    forecast. A Poisson distribution is a standard formula for counting random events, such as customers
    buying an item. "90th percentile" means sales should be at or below `hi` in about 9 weeks out of 10.
    Code: `poisson_interval()` in `src/inventory_rl/uncertainty/intervals.py`. This method was chosen because it
    scored best on the validation weeks.
  - Drawn with Altair in the "Demand outlook" block of `dashboard/app.py`.
- **d) Used for**: judging whether the forecast is believable. Has the product been selling? Was there a
  recent spike? A wide band means "very uncertain, keep a buffer".
- **e) Example**: Week of 6 Apr 2026, Store 1044, PROD-100803. Store 1044 sold **7** units in the week of
  30 Mar (a spike). The model then forecast **5.7** for 6 Apr, with a range of **3 to 9**. Actual sales were
  only **2**. The model reacted strongly to the spike, which shows that forecasts can be wrong.

The grey caption under the chart repeats the meaning of the line, the band and the dashed line.

#### Box: Stock risk status (coloured box)

- **a) Shows**: one of these statuses, with a short explanation:

  | Status | Colour | Explanation shown |
  |---|---|---|
  | **Stockout risk** | red | "Stock is unlikely to cover next week's demand." |
  | **Excess stock** | amber (orange) | "More stock than is likely to sell in the next 8 weeks." |
  | **Normal** | green | "Stock is in line with expected demand." |
  | **No stock record** | grey | "The data has no stock count for this store and product this week." |

- **b) Means**: a traffic light for this store and product in the planning week.
- **c) How it is made**: `assess()` in `src/inventory_rl/risk/assessment.py`, applied to the **simulated**
  stock. The rules are checked in this order:
  1. If there is no stock count → **No stock record**. This cannot happen in this tab, because the simulation
     treats missing counts as 0, so grey only appears in Store Overview.
  2. If the chance of running out is **50% or more** → **Stockout risk**.
  3. If stock is **below the high estimate** `hi` → also shown as **Stockout risk**. Internally this is a
     separate, milder class, "below upper bound"; the dashboard groups it with red.
  4. If stock is **more than the high estimate of 8 weeks of sales** → **Excess stock**. The high estimate is
     the 90th percentile of a Poisson distribution with average 8 × forecast.
  5. Otherwise → **Normal**.

  Names and colours come from `explain.risk_status()` and `explain.RISK_COLORS`.
- **d) Used for**: deciding which stores need action first.
- **e) Example** (Week of 6 Apr 2026, PROD-100803):
  - Store **1044**: 5 in stock, about 5.7 expected → **Stockout risk** (red).
  - Store **1237**: 9 in stock, about 0.7 expected per week → **Excess stock** (amber). Its 8-week high
    estimate is 8 units and it has 9.

#### Card: "In stock"

- **a) Shows**: units on the shelf at the start of the planning week.
- **b) Means**: Store 1044 has 5 pairs of these shoes on Monday morning.
- **c) How it is made**: the **simulated** stock from `plan()`. It starts from the recorded stock in the week
  of 23 Feb and then follows the chosen approach. If a store had no stock count on 23 Feb, it starts at 0.
- **d) Used for**: comparing stock with expected sales.
- **e) Example**: 6 Apr, Store 1044, PROD-100803 → **5**.

#### Card: "Expected sales" (and its ? icon)

- **a) Shows**: the forecast for the planning week, with one decimal. The ? icon shows "Likely range X to Y".
- **b) Means**: on average the store is expected to sell this many units. A decimal like 0.4 means "about one
  sale every two or three weeks".
- **c) How it is made**: the forecast `yhat` (LightGBM, see the chart above). The range in the ? icon is `lo`
  to `hi` (Poisson 10th–90th percentiles).
- **d) Used for**: judging how much stock is needed.
- **e) Example**: 6 Apr, Store 1044, PROD-100803 → **5.7**, likely range **3 to 9**.

#### Card: "Chance of running out next week"

- **a) Shows**: a percentage, or **"Out of stock"** when stock is 0.
- **b) Means**: the chance that customers will want **at least as many units as the store has**, so the
  shelf ends up empty. With 0 in stock the store is already out.
- **c) How it is made**: `p_at_least()` in `src/inventory_rl/risk/assessment.py`:

  **chance = P(demand ≥ stock)**, where demand follows a Poisson distribution with average = forecast.

  For example, with stock 5 and forecast 5.72 the chance is 67.5%. The text is formatted by `explain.chance()`.
- **d) Used for**: ranking urgency.
- **e) Example**: 6 Apr, Store 1044, PROD-100803 → **68%**. For the default case (13 Apr, Store 1080,
  PROD-100800) with the AI → **Out of stock**.

> **Is this percentage trustworthy?** It is checked in Technical Details ("Is the chance of running out
> trustworthy?"). It is well matched below about 50%, but above 50% there are very few test cases and the
> real share was lower than predicted. Treat high percentages as "high risk", not as exact numbers.

#### Section "Recommended transfer": the 5 cards

Shown when the selected store **sends or receives** stock in the planning week.

| Card | a) Shows | b) Means |
|---|---|---|
| **From** | The store that sends stock. | Store 1237 sends. |
| **To** | The store that receives stock. | Store 1044 receives. |
| **Quantity** | Number of units ("1 unit", "4 units"). | Move 1 pair of shoes. |
| **Distance** | Kilometres between the two stores. | 29 km. **Simulated** distance (see below). |
| **Estimated cost** | Shipping cost of this move. | 8.88 money units. |

- **c) How it is made**:
  - Transfers come from `plan()` (above). The AI and the rule both turn **target stock levels** into
    shipments with `transfers_from_targets()` in `src/inventory_rl/environment/matching.py`:
    - **need** of a store = target − stock (if positive);
    - **spare** of a store = stock − max(its own target, the amount it must keep);
    - a store **must keep** at least its own expected sales, rounded up (`keep_level()` in
      `src/inventory_rl/environment/simulator.py`);
    - stores with the biggest need are served first, from the **nearest** store with spare stock.
  - The **targets** are different for the two approaches:
    - **Rule**: target = `hi` (the high estimate). The rule also skips a shipment if its cost per unit is
      more than the product's profit per unit (`rules.py`).
    - **AI**: target = forecast + an adjustment between −2 and +2 units, chosen by the trained network for each
      store (`action_to_transfers()` in `src/inventory_rl/environment/gym_env.py`). The AI has **no**
      cost-per-unit check; it learned during training whether shipments pay off.
  - **Distance**: from the simulated store map (`src/inventory_rl/routing/network.py`, see Part 5.6).
  - **Cost**: `transfer_cost()` in `routing/network.py`:

    **cost = 8 (fixed cost per shipment) + 0.03 × units × km**

    For example, 1 unit over 29.4 km = 8 + 0.03 × 1 × 29.4 = **8.88**. 4 units = 8 + 0.03 × 4 × 29.4 = **11.53**.
    The cost is in the same currency as the sales data; the dataset does not name the currency.
  - If a store is in several transfers, the cards show the first one.
- **d) Used for**: the actual action: "send X units from A to B; it will cost about Y".
- **e) Example**: 6 Apr 2026, Store 1044, PROD-100803, AI → From **Store 1237**, To **Store 1044**,
  **1 unit**, **29 km**, **8.88**.

#### Box: "Why: …"

- **a) Shows**: a generated sentence explaining the transfer.
- **b) Means**: the reason in plain words, so a manager can check it makes sense.
- **c) How it is made**: `explain.transfer_reason()`. It fills in a template with the numbers:
  - If the receiving store has 0 units: "… is out of stock and expects to sell about X next week."
  - If its chance of running out is ≥ 50%: "… is likely to run out next week (N% chance): it has S units and
    expects to sell about X."
  - Otherwise: "… may run short next week (N% chance of running out) …"
  - Then: "[Sender] has spare stock (S units against about X expected sales). Move Q units over D km,
    estimated cost C."

  The sentence explains the **situation**. It does not read the AI's "mind". The AI network does not give
  reasons by itself; we describe the numbers it reacted to.
- **d) Used for**: trusting (or questioning) the recommendation.
- **e) Example** (6 Apr, Store 1044, PROD-100803, AI): "Store 1044 is likely to run out next week (68% chance):
  it has 5 units and expects to sell about 5.7. Store 1237 has spare stock (9 units against about 0.7 expected
  sales). Move 1 unit over 29 km, estimated cost 8.88."

#### Table: "This store is part of N transfers this week"

- **a) Shows**: appears only when the selected store sends or receives more than one shipment. Columns:
  **From, To, Units, Distance (km), Cost**.
- **b) Means**: a busy store, for example one sender supplying several stores.
- **c) How it is made**: the transfers from `plan()` that involve the selected store.
- **d) Used for**: seeing the full workload for one store.
- **e) Example**: Week of 16 Mar 2026, PROD-140871, AI, **Store 1074**. It sends 1 unit each to Stores 1075,
  1094, 1219, 1234 and 1235 (5 transfers).

#### Box: "No transfer for Store X." (when the store has no transfer)

- **a) Shows**: one of four sentences from `explain.no_transfer_reason()`:
  1. **Out of stock, but no transfer**: "… is out of stock (expected sales about X), but no transfer to it is
     recommended this week: other stores have little spare stock of this product, or the recommended approach
     judged a shipment not worthwhile."
  2. **At risk, but no transfer**: the same idea with the chance of running out.
  3. **Excess stock**: "… holds more than it is likely to sell …, but no other store needs this product enough
     to justify a shipment."
  4. **Normal**: "… has enough stock for next week …, so no transfer is needed."
- **b) Means**: why nothing happens for this store.
- **c) How it is made**: chosen by the store's status. The wording "or the recommended approach judged a
  shipment not worthwhile" is a general explanation. The dashboard does **not** know the exact internal
  reason the AI chose not to ship.
- **d) Used for**: knowing that "no transfer" is a decision, not an error.
- **e) Example**: 6 Apr 2026, Store 1044, **PROD-100801**, AI → case 1. The store is out of stock but expects
  only about 0.1 sales.

#### Expander: "All transfers suggested for this product this week (N)"

- **a) Shows**: a collapsible section (click to open) listing every transfer for this product and week, across
  all stores. Same columns: From, To, Units, Distance (km), Cost.
- **b) Means**: the full plan for the product, not just your store.
- **c) How it is made**: all transfers returned by `plan()`.
- **d) Used for**: a planner approving the whole week's moves for one product.
- **e) Example**: 6 Apr 2026, PROD-100801. The AI suggests **0** (the section does not appear); the rule
  suggests **13** transfers.

#### Caption at the bottom of the tab

- **a) Shows**: "Stock levels here are simulated as if the chosen approach had been in use since the week of
  23 Feb 2026 … can differ from the recorded stock in Store Overview. Stores without a stock count at that start
  are treated as empty. Distances come from a simulated store map … Costs use the project's assumed shipping
  costs … A store always keeps enough stock for its own expected sales before sending any."
- **b–d)**: the key honesty notes for this tab. Read it before demoing.
- **e) Example**: bottom of Tab 1.

---

### 2.2 Tab 2 — Store Overview

The tab's purpose: see **which stores need attention** this week, using the **recorded** stock from the data
(not the simulation).

#### Heading "Store overview" and caption

- **a) Shows**: "Which stores need attention in the chosen week. Red = may run out, amber = too much stock,
  green = normal, grey = no stock count in the data."
- **c) How it is made**: `st.header` and `explain_line()`.

#### Dropdown: "Week"

- **a) Shows**: the 8 test weeks; the default is the latest (Week of 13 Apr 2026).
- **b–d)**: the week to inspect.
- **c) How it is made**: test-week rows of `data/processed/risk.parquet` (made by `scripts/run_risk.py`).
- **e) Example**: "Week of 23 Feb 2026" is much calmer than later weeks (see the cards).

#### Dropdown: "Product"

- **a) Shows**: "All products" (default) plus the 30 products.
- **b) Means**: look at the whole range, or at one product.
- **c) How it is made**: filters the risk table. "All products" shows a **bar chart**; one product shows a
  **tile grid**.
- **e) Example**: choose PROD-100803 in Week of 6 Apr 2026.

#### The 4 metric cards

| Card | a) Shows | c) How it is calculated |
|---|---|---|
| **Stores with a stockout risk** | "X of 40" | Number of different stores with at least one red product. |
| **Store-product pairs at risk** | a count | Number of (store, product) combinations that are red. |
| **Pairs with excess stock** | a count | Number of amber combinations. |
| **Pairs with no stock count** | a count | Number of grey combinations (no stock record that week). |

- **b) Means**: the size of the problem this week.
- **c) More detail**: the status uses the same `assess()` rules as Tab 1, but on **recorded** opening stock.
  The status is pre-computed in `risk.parquet` and named by `explain.risk_status()`.
- **d) Used for**: a quick weekly health check of the whole chain.
- **e) Examples** (All products):

  | Week | Stores at risk | Pairs at risk | Excess | No stock count |
  |---|---|---|---|---|
  | 23 Feb 2026 | 28 of 40 | 59 | 364 | 475 |
  | 6 Apr 2026 | 39 of 40 | 233 | 97 | 361 |
  | 13 Apr 2026 | 39 of 40 | 220 | 146 | 364 |

> **Important finding:** on 13 Apr, **219 of the 220 red pairs have zero recorded stock**. Most "Stockout risk"
> in this tab means "already empty", not "about to run out". Many of these products sell very rarely, so an
> empty shelf is not always urgent.

#### Bar chart (when "All products" is selected)

- **a) Shows**: one horizontal bar per store. Each bar is split into coloured pieces: **red** (Stockout risk),
  **amber** (Excess stock), **green** (Normal), **grey** (No stock record). The length of each piece is the
  number of products with that status (x-axis "Number of products", 0–30). The legend at the top names the
  colours. Hovering shows Store, Status and Products.
- **b) Means**: Store 1026 has 13 of its 30 products in red on 13 Apr: that store needs attention.
- **c) How it is made**: the "store overview" block of `dashboard/app.py` counts products per store and status.
  Stores are sorted by red count (most red at the top); stores with no red go at the bottom. Colours come from
  `explain.RISK_COLORS`.
- **d) Used for**: deciding which stores to call or visit first.
- **e) Example**: 13 Apr 2026. **Store 1026 and Store 1350** are at the top with 13 red products each.
  **Store 1308** is at the bottom with 0 red.

Caption: "Each bar is one store, split by the status of its products. Stores with the most products at risk of
running out are at the top."

#### Tile grid (when one product is selected)

- **a) Shows**: 40 rounded squares (5 rows × 8), one per store, coloured by status, with the store number in
  white. Hovering shows **Store, Status, In stock** (or "no record"), **Expected sales**, and **Chance of
  running out** (or "unknown").
- **b) Means**: a map-like view of one product across all stores. It is **not** a geographic map; tiles are
  simply in store-number order.
- **c) How it is made**: the same risk table filtered to one product; tiles drawn with Altair rectangles.
- **d) Used for**: finding sender and receiver stores for one product: red tiles need stock, amber tiles have
  extra.
- **e) Example**: Week of 6 Apr 2026, PROD-100803:
  - **3 red** tiles: 1044, 1109, 1219;
  - **1 amber** tile: 1237;
  - 1 grey and 35 green tiles.
  - Hovering over 1044 shows In stock 5, Expected sales 5.7, Chance 68%.

Caption: "Each tile is a store (number shown). Hover over a tile to see its stock, expected sales and chance of
running out."

---

### 2.3 Tab 3 — Results

The tab's purpose: show how well each approach did over the **8 test weeks** (23 Feb – 19 Apr 2026), for all
30 products and 40 stores, compared with **making no transfers at all**.

#### The six approaches (names used in the charts)

| Name on screen | What it is | Code |
|---|---|---|
| No transfers | Baseline: never move stock. | `no_transfer` in `baselines/rules.py` |
| **Rule: cover the forecast** | Target = expected sales (rounded up). | `RuleBasedPolicy("point")` |
| **Rule: cover a high estimate** | Target = high estimate `hi` (Poisson 90th percentile). | `RuleBasedPolicy("upper")` |
| **Rule: cover a calibrated high estimate** | Target = upper bound from a different range method (CQR, see Glossary). | `RuleBasedPolicy("cqr")` |
| **AI agent** | Trained AI that sees only stock and the forecast. | `rl_point_seed*` models |
| **Uncertainty-aware AI agent** | Trained AI that also sees the likely ranges. | `rl_uncertainty_seed*` models |

Names come from `explain.POLICY_NAMES`. Each approach has its own colour in every chart: the "cover a high
estimate" rule is **orange**, the uncertainty-aware AI is **dark blue**, the AI agent is **light blue**, and
the other rules are greys.

#### Radio buttons: "Demand used in the test" (and its ? icon)

- **a) Shows**: **Simulated demand (20 runs)** (default) or **Actual past sales**. The help icon says:
  "Simulated demand varies around past sales. Actual past sales understate demand in weeks when a store had sold
  out."
- **b) Means**: two ways to test.
  - **Actual past sales** ("replay"): replay what customers really bought. Problem: if a shelf was empty, we
    never saw the customers who wanted to buy, so true demand is hidden. This is called **censored demand**.
  - **Simulated demand**: 20 random versions of each week. Demand is drawn from a Poisson distribution whose
    average is a smoothed version of real sales (a 5-week moving average centred on that week).
- **c) How it is made**: `scripts/run_evaluation.py` → `results/metrics/evaluation_summary.csv`. The simulator
  is in `src/inventory_rl/environment/simulator.py` and the metrics in `src/inventory_rl/evaluation/experiment.py`.
  Every approach sees exactly the same 20 demand runs (fixed random seeds 10,000–10,019).
- **d) Used for**: checking that a conclusion holds under both views of demand.
- **e) Example**: the best approach changes. With simulated demand it is "Rule: cover a high estimate"; with
  actual past sales it is "Rule: cover a calibrated high estimate".

#### The 4 metric cards (always for the Uncertainty-aware AI agent)

| Card | a) Simulated demand | a) Actual past sales | c) Formula |
|---|---|---|---|
| **Lost sales avoided** (? "Uncertainty-aware AI agent, compared with no transfers") | **19%** | **17%** | (lost units with no transfers − lost units with the AI) ÷ lost units with no transfers. With simulated demand: (355 − 287) ÷ 355. |
| **Change in leftover stock** | **−1.0%** | **−0.7%** | Change in average closing stock (stock left at the end of each week, summed over stores and products) compared with no transfers. |
| **Shipments in 8 weeks** | **144** | **91** | Average number of shipments over the 8 weeks, all 30 products. |
| **Money saved** | **2,021** | **1,176** | Total cost with no transfers − total cost with the AI (see below). |

- **b) Means** (simulated demand): over 8 weeks the AI prevents about one in five lost sales, makes 144 small
  shipments, and the chain ends up about 2,021 money units better off after paying for shipping.
- **c) How "Money saved" is calculated**: for every week and store, the simulator adds up three costs:
  - **lost-sale cost** = units customers wanted but could not buy × profit per unit (profit = average price −
    average cost, taken from the training-period sales);
  - **holding cost** = stock left at the end of the week × unit cost × 25% ÷ 52 (25% a year, an assumption);
  - **shipping cost** = 8 per shipment + 0.03 × units × km.

  Money saved = (total cost with no transfers) − (total cost with this approach), on the same demand.
  For AI agents, the number is the average over **5 separately trained agents** × 20 demand runs.
- **d) Used for**: judging whether the AI is worth using.
- **e) Example**: visible as soon as you open the tab.

Caption: "The four cards show the uncertainty-aware AI agent. Money saved = cost of no transfers minus
(lost-sale cost + holding cost + shipping cost), over 30 products and 40 stores."

#### Box: "Takeaway"

- **a) Shows**, with simulated demand: "in this test, **Rule: cover a high estimate** saved the most money. The
  uncertainty-aware AI agent saved 2,021, more than the AI agent without uncertainty (1,885), using 144
  shipments. Rule: cover the forecast spent more on shipping than it saved. The gap between the two AI agents is
  within their run-to-run variation, so it is not a reliable difference."
- **b) Means**: the honest one-line summary. The AI did well but not best; the uncertainty information helped
  the AI only a little, and that gap could be luck.
- **c) How it is made**: generated in `dashboard/app.py` from the numbers. Nothing is typed in by hand.
  - "Saved the most" uses `explain.best_policy()`, which picks the highest money saved, or "No transfers" if
    every approach loses money.
  - "Spent more on shipping than it saved" lists approaches with negative money saved.
  - "Not a reliable difference" comes from a **Welch t-test** (a standard statistics test of whether two
    averages really differ) over the 5 training runs of each AI, stored in `results/metrics/results.json`. The
    p-value is 0.22 (simulated) and 0.23 (actual). A p-value above 0.05 means the difference could easily be
    chance.
- **d) Used for**: the main conclusion for management.
- **e) Example**: switch to "Actual past sales"; the takeaway now names "Rule: cover a calibrated high estimate"
  (1,331).

#### Chart "Fewer stockouts": Lost sales avoided (%)

- **a) Shows**: one bar per approach (no-transfers is left out, because it is 0 by definition).
- **b) Means**: the share of lost sales each approach prevented. A **lost sale** is a customer who found the
  shelf empty.
- **c) How it is made**: same formula as the card, for each approach (`bar()` helper in `dashboard/app.py`).
- **d) Used for**: comparing service level.
- **e) Example** (simulated):

  | Approach | Lost sales avoided |
  |---|---|
  | Rule: cover a high estimate | 30% |
  | Rule: cover the forecast | 27% |
  | Rule: cover a calibrated high estimate | 22% |
  | Uncertainty-aware AI | 19% |
  | AI agent | 17% |

Caption: "Share of lost sales (customers who found the shelf empty) that each approach prevented."

#### Chart "Leftover stock": Change in average leftover stock (%)

- **a) Shows**: small negative bars (between about −0.4% and −1.5% with simulated demand).
- **b) Means**: transfers do not remove stock from the chain. They only move it, so total leftover stock hardly
  changes. It falls a little because more of it gets sold.
- **c) How it is made**: % change of `avg_closing_stock` compared with no transfers.
- **d) Used for**: showing that this project solves **where** stock is, not **how much** stock exists.
- **e) Example**: "Rule: cover a high estimate" −1.5%; uncertainty-aware AI −1.0%.

Caption: "Transfers move stock between stores rather than removing it, so total leftover stock changes only
slightly."

#### Chart "Shipments made": Shipments in 8 weeks

- **a) Shows**: the number of shipments per approach.
- **b) Means**: workload and cost. Fewer shipments for a similar saving means fewer unnecessary moves.
- **c) How it is made**: `n_transfers` in the evaluation summary.
- **d) Used for**: judging operational effort (staff time, trucks).
- **e) Example** (simulated): "Rule: cover the forecast" makes **466** shipments and still loses money.
  "Rule: cover a high estimate" makes 278, the uncertainty-aware AI 144, the AI agent 129.

Caption: "Fewer shipments for a similar saving means fewer unnecessary moves. The data does not mark individual
shipments as necessary or not." This is honest: the project has **no direct measure of "unnecessary
transfers"**, only these indirect signs.

#### Chart "Money saved": Money saved vs no transfers (with black lines)

- **a) Shows**: bars of money saved, with a black vertical line at 0. The two AI bars have a thin black
  horizontal line (an **error bar**).
- **b) Means**: below 0 = worse than doing nothing. The error bar shows how much the result varies between
  **5 separately trained AI agents**: from the average minus one standard deviation to the average plus one.
  A **standard deviation** measures how spread out values are.
- **c) How it is made**: `saving_vs_noop` from the evaluation summary. The error bar comes from
  `results/metrics/rl_saving_across_training_seeds.csv` (mean ± std of the 5 runs).
- **d) Used for**: the bottom line.
- **e) Example** (simulated):
  - Rule: cover a high estimate **2,276**
  - Rule: cover a calibrated high estimate **1,985**
  - Uncertainty-aware AI **2,021 ± 182**
  - AI agent **1,885 ± 132**
  - Rule: cover the forecast **−187**

Caption: "Black lines on the AI bars show the spread over 5 separately trained agents."

#### Section "What if the cost assumptions are different?"

Caption: "Shipping and lost-sale costs are not in the data, so the project assumes them. Here the test is
repeated with one assumed cost changed at a time."

**The 4 bullet sentences**

- **a) Shows**: for each assumed cost, which approach saves the most at the lowest and highest value tried.
- **b) Means** (simulated demand):
  - **Fixed cost per shipment** (assumed 8): at 2, Rule: cover a high estimate saves the most; at 32, **No
    transfers** is cheapest (every approach loses money).
  - **Cost per item per km** (assumed 0.03): at 0.0075 the rule saves the most; at 0.12 the uncertainty-aware AI
    saves the most.
  - **Yearly cost of holding stock** (assumed 0.25): the rule wins at both ends; this cost hardly matters.
  - **Cost of a lost sale** (assumed 1 × profit): at 0.5 the AI agent saves the most; at 3 the rule saves the
    most.
- **c) How it is made**: `scripts/run_sensitivity.py` and `src/inventory_rl/evaluation/sensitivity.py` re-run the
  whole test 15 times (base costs plus 14 changed settings) → `results/metrics/cost_sensitivity.csv`. The
  sentences are generated by `dashboard/app.py` with `explain.best_policy()`. **The AI agents were not
  retrained** for each setting; they keep the behaviour they learned with the base costs.
- **d) Used for**: knowing which approach to choose if our real costs are different from the assumptions.
- **e) Example**: see Exercise 3.4.

**Dropdown: "Show the effect of"**

- **a) Shows**: the four cost names: Fixed cost per shipment, Cost per item per km, Yearly cost of holding stock
  (share of item cost), Cost of a lost sale (times the profit per item).
- **c) How it is made**: picks which rows of `cost_sensitivity.csv` the chart draws.

**Line chart**

- **a) Shows**: x-axis = the cost value (only the tried values are marked: for example 2, 4, 8, 16, 32), on a
  **log scale**, meaning each step doubles. The y-axis is money saved. There is one coloured line per approach,
  a black horizontal line at 0, a legend at the top, and a tooltip with the approach, cost value and money saved.
- **b) Means**: how each approach's saving changes as the cost assumption changes. A line crossing below the
  black line starts losing money.
- **d) Used for**: seeing where the lines cross, i.e. where the best choice changes.
- **e) Example**: "Fixed cost per shipment". At 16:
  - uncertainty-aware AI **865**, AI agent **856**;
  - Rule: cover a calibrated high estimate 411;
  - Rule: cover a high estimate only **54**.

  The AI wins at high shipping cost because it ships less. This is **not** because it understands costs; it
  never sees them.

Caption: "Below the black line an approach loses money. The AI agents were trained once with the assumed costs
and were not retrained for each what-if value."

---

### 2.4 Tab 4 — About the Project

- **a) Shows**: five text blocks:
  - **The problem**
  - **The data** (with the number of products and stores filled in from the data: 30 and 40)
  - **How it works** (5 numbered steps: Forecast, Uncertainty, Risk, Recommendation, Testing)
  - **Limitations** (6 bullets)
  - **Future work** (4 bullets)
- **b) Means**: the project story in plain language.
- **c) How it is made**: Markdown text in `dashboard/app.py`. Only the two counts are computed (from the risk
  table).
- **d) Used for**: explaining the project to a non-technical audience.
- **e) Example**: read it aloud in the demo (Part 4, minute 9).

---

### 2.5 Tab 5 — Technical Details

Caption: "Data quality, model accuracy and how the results were produced, for readers who want more detail."

#### Table "Pipeline status"

- **a) Shows**: two columns, **Step** and **Status** ("Ready" or "Not run yet"). The 7 steps are:
  1. Data checks
  2. Weekly sales and stock table
  3. Forecasts and likely ranges
  4. Risk scores
  5. Store distances
  6. Trained AI agents
  7. Test results
- **b) Means**: whether each stage of the project has been run on this computer.
- **c) How it is made**: `data_access.artifact_status()` checks that each output file or folder exists.
- **d) Used for**: troubleshooting.
- **e) Example**: all "Ready" on our machine.

#### Table "Data quality" (columns Measure, Value)

| Measure | Value | Meaning |
|---|---|---|
| Sales transactions | 125,751 | Rows in the sales file. |
| Stock records | 284,755 | Rows in the stock file. |
| Stores / products in the raw data | 40 / 2,326 | Before we chose the top 30 products. |
| Sales period | 2025-06-01 to 2026-04-24 | About 11 months. |
| Return transactions | 7,491 | Rows where a customer returned an item. |
| Stock records with negative quantity | 1,380 | Impossible values (book-keeping errors). |
| Weekly series used (store x product) | 1,200 (40 stores x 30 products, 46 weeks) | What the project models. |
| Store-weeks with no sales | 83% | Most weeks a store sells none of a given product. |
| Store-weeks with no stock count at the start | 29% | Stock unknown, not zero. |
| Store-weeks starting with zero stock | 7.5% | Shelf already empty on Monday. |
| Duplicate records | 0 | No repeated rows in the weekly table. |

- **c) How it is made**: from `results/reports/data_audit.json` (`scripts/run_data_audit.py`) and
  `results/reports/preprocessing_report.json` (`scripts/run_preprocess.py`).
- Caption: "Two other public datasets were checked and not used: one was internally inconsistent and the other
  looked randomly generated." (Details in Part 5.1.)

#### Table "Forecast accuracy (test weeks)"

Columns: **Method**, **Average error (units)** (MAE), **Large-error score (RMSE)**, **Bias (units)**.

| Method | Average error | Large-error score | Bias |
|---|---|---|---|
| Machine-learning model (used) | 0.281 | **0.569** | +0.012 |
| Same as last week | 0.305 | 0.746 | −0.002 |
| Average of last 4 weeks | 0.282 | 0.603 | −0.018 |
| Average of last 8 weeks | 0.285 | 0.588 | −0.042 |
| Always zero | **0.235** | 0.683 | −0.235 |

- **b) Means**:
  - **MAE** (mean absolute error) is the average size of the miss in units.
  - **RMSE** (root mean squared error) is like MAE but punishes big misses more.
  - **Bias** is the average of (forecast − actual): positive means the forecast is too high on average.
- **c) How it is made**: `src/inventory_rl/evaluation/metrics.py` → `results/metrics/forecast_metrics.json`.
- **d) Used for**: proving the model is better than simple methods.
- **e) Example**: "Always zero" has the lowest average error but misses every sale, which is useless for stock
  planning. That is why we judge by RMSE, where our model is best (0.569). The caption under the table says the
  same.

#### Table "Likely-range accuracy (test weeks)"

Columns: **Method**, **Sales inside the range**, **Target**, **Average width (units)**.

| Method | Inside the range | Target | Width |
|---|---|---|---|
| Count-based range (used) | 97% | 80% | 0.67 |
| Quantile model | 93% | 80% | 0.4 |
| Calibrated quantile model | 93% | 80% | 0.4 |
| Error-based range | 79% | 80% | 0.02 |

- **b) Means**: an 80% range should contain actual sales about 80% of the time.
  - The ranges cover more than that, because sales are small whole numbers: a range "0 to 1" catches all the
    zero weeks.
  - "Width" is the average gap between low and high.
- **c) How it is made**: `scripts/run_forecasting.py` and `uncertainty/intervals.py` →
  `results/metrics/interval_metrics.json`. The "(used)" method was chosen on the **validation** weeks using the
  **interval score**, a score that rewards narrow ranges and penalises misses.
- **d) Used for**: trusting the shaded band in Tab 1.
- **e) Example**: the error-based range hits 79% but is almost zero-width (0.02), so it is not useful.

Caption: "Weekly sales are small whole numbers, so ranges often cover more than their target. The method was
chosen on earlier weeks, not on the test weeks."

#### Table "AI agent training runs"

Columns: **Agent**, **Run** (1–5), **Saving on earlier check weeks**, **Saving on test weeks (simulated
demand)**, **Saving on test weeks (actual sales)**.

| Agent | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 |
|---|---|---|---|---|---|
| Uncertainty-aware: check weeks | 956 | 547 | **962** | 671 | 916 |
| Uncertainty-aware: test (simulated / actual) | 2,078 / 1,087 | 1,754 / 1,044 | 2,253 / 1,320 | 1,961 / 1,121 | 2,058 / 1,309 |
| AI agent: check weeks | 419 | 569 | 578 | 526 | 750 |
| AI agent: test (simulated / actual) | 1,861 / 1,053 | 1,805 / 1,042 | 1,867 / 1,077 | 1,779 / 985 | 2,111 / 1,249 |

- **b) Means**: each AI was trained 5 times with different random starts. A "run" is one complete training.
  - "Earlier check weeks" are the **validation** weeks (29 Dec 2025 – 16 Feb 2026), used to choose the best
    saved version during training.
  - Results differ between runs, which is why we report averages ± spread.
- **c) How it is made**: `models/rl/training_summary.json` and
  `results/metrics/rl_saving_per_training_seed.csv`. "Run" = training seed + 1.
- **d) Used for**: showing that results are not from one lucky run.
- **e) Example**: Uncertainty-aware **Run 3** has the best check-week saving (962). That is the agent the
  Decision Support tab uses. Its test result was also the best, but it was **not** chosen using the test weeks.

#### Table "Cost assumptions"

| Assumption | Value |
|---|---|
| Fixed cost per shipment | 8 |
| Cost per item per km | 0.03 |
| Yearly holding cost (share of item cost) | 0.25 |
| Cost of a lost sale (times profit per item) | 1 |
| Profit and cost per item | Taken from the sales data |
| Store distances | Simulated map (no locations in the data) |

- **c) How it is made**: read from `config/config.yaml` (`costs` section).
- **d) Used for**: making clear what is assumed.

#### Chart "Is the chance of running out trustworthy?"

- **a) Shows**:
  - x-axis = **predicted** chance of running out (%);
  - y-axis = **share that actually sold out** (%);
  - a blue line with dots (one dot per group of store-weeks: 0–10%, 10–20%, …);
  - a grey dashed diagonal (perfect agreement);
  - a tooltip with Predicted (%), Actually sold out (%) and Store-weeks (how many cases are in the dot).
- **b) Means**: if we say "30%", about 30% of such cases should really sell out. Dots near the dashed line are
  good.
- **c) How it is made**: `stockout_calibration()` in `risk/assessment.py` → `results/metrics/stockout_calibration_test.csv`.
  Only test weeks with stock on hand are used, because "sold out" can only be observed when there was stock.
  The overall **Brier score** (average squared error of probabilities; lower is better) is 0.035
  (`risk_summary.json`).
- **d) Used for**: knowing how far to trust the percentage in Tab 1.
- **e) Example**:

  | Predicted | Actually sold out | Store-weeks |
  |---|---|---|
  | 1% | 2% | 4,335 |
  | 14% | 12% | 387 |
  | 25% | 23% | 170 |
  | 35% | 42% | 100 |
  | 44% | 44% | 32 |
  | 54% | 17% | **12** |
  | 65% | 0% | **4** |

  The predictions are good up to about 50%. Above that there are very few cases, and reality was lower than
  predicted.

Caption: "Test weeks with stock on hand, grouped by predicted chance. Points on the dashed line mean the
predicted chance matched reality. Few store-weeks have a predicted chance above 50%, so the right-hand points
rest on very little data (hover to see counts)."

---

## Part 3 — Hands-on exploration exercises

Do these with the dashboard open. Each says what to select, what you should see, what it means, and gives a
question with its answer.

### Tab 1 — Decision Support

**Exercise 1.1: The default example**
- **Do**: open the dashboard (no changes). You get Week of 13 Apr 2026, Store 1080, PROD-100800, Uncertainty-aware AI agent.
- **See**:
  - red **Stockout risk**; In stock **0**; Expected sales **0.4**; Chance **Out of stock**;
  - transfer From **Store 1114** To **Store 1080**, **1 unit**, **42 km**, cost **9.27**.
- **Meaning**: Store 1080 has nothing on the shelf. Store 1114 has 4 units and expects to sell only about 0.4,
  so it can spare one.
- **Question**: how is 9.27 calculated?
  **Answer**: 8 (fixed) + 0.03 × 1 unit × about 42.4 km ≈ 9.27.

**Exercise 1.2: Same case, switch to the rule**
- **Do**: keep everything; click **Rule: cover a high estimate**.
- **See**: Store 1080 now has **1** in stock, status **Normal**, and "No transfer for Store 1080 … has enough
  stock for next week".
- **Meaning**: the stock is different because each approach is simulated from 23 Feb with its **own** earlier
  transfers. Under the rule, Store 1080 already had 1 unit (from an earlier transfer or from different
  stock movements), so it needs nothing this week.
- **Question**: why can "In stock" differ between the two approaches for the same store and week?
  **Answer**: because the stock is simulated; earlier transfers by each approach change later stock.

**Exercise 1.3: AI versus rule on a clear case (1 unit vs 4 units)**
- **Do**: Planning week **Week of 6 Apr 2026**, Store **1044**, Product **PROD-100803**.
- **See with the AI**:
  - red **Stockout risk**; In stock **5**; Expected sales **5.7** (? shows range 3 to 9); Chance **68%**;
  - transfer **Store 1237 → Store 1044, 1 unit, 29 km, 8.88**;
  - expander "All transfers … (1)".
- **See with the rule**: same status and stock, but **4 units**, cost **11.53**, and "All transfers … (4)".
  The rule also sends 1 unit each 1198 → 1109, 1107 → 1219 and 1353 → 1082.
- **Meaning**:
  - The rule fills Store 1044 up to the high estimate: 9 − 5 = **4 units**.
  - The AI adds a smaller buffer of 1 unit. It learned that big buffers often go unsold, because sales are
    usually low.
  - In reality Store 1044 sold only **2** that week, so the AI's smaller move was enough this time.
- **Question**: where does the rule's "4" come from?
  **Answer**: the target is the high estimate (9), and the store has 5, so it needs 9 − 5 = 4.

**Exercise 1.4: The sender with extra stock**
- **Do**: Week of 6 Apr 2026, Store **1237**, PROD-100803 (AI).
- **See**:
  - amber **Excess stock**; In stock **9**; Expected sales **0.7**; Chance **0%**;
  - the same transfer, but now Store 1237 is in the **From** card.
- **Meaning**: Store 1237 has more than 8 weeks' worth (its 8-week high estimate is 8). It is a natural donor.
- **Question**: after sending 1 unit, how many must Store 1237 keep at least?
  **Answer**: its own expected sales rounded up: 0.7 → **1 unit**. The "keep level" rule protects every sender.

**Exercise 1.5: Rule ships, AI does not**
- **Do**: Week of 6 Apr 2026, Store **1044**, Product **PROD-100801**.
- **See with the AI**: red, In stock 0, Expected sales **0.1**, "No transfer for Store 1044 … is out of stock
  (expected sales about 0.1) …", and no expander.
- **See with the rule**: transfer **Store 1237 → Store 1044, 1 unit, 29 km, 8.88**, and "All transfers … (13)".
- **Meaning**: the rule ships whenever stock is below the high estimate (here 1), even for products that almost
  never sell. The AI waits.
- **Question**: is the rule's shipment worth it?
  **Answer**: probably not. With expected sales of 0.11, the chance of selling even one unit next week is only
  about 10%. The profit per unit of PROD-100801 is about 41, so the expected lost sale is worth about
  0.10 × 41 ≈ **4**, but the shipment costs **8.88**. This is the kind of "unnecessary transfer" the AI avoids.

### Tab 2 — Store Overview

**Exercise 2.1: This week's health check**
- **Do**: open Store Overview (default: Week of 13 Apr 2026, All products).
- **See**: cards **39 of 40**, **220**, **146**, **364**. Store 1026 and Store 1350 are at the top of the chart.
- **Meaning**: almost every store has at least one empty or nearly empty product.
- **Question**: of the 220 red pairs, how many have zero recorded stock?
  **Answer**: 219. Red here mostly means "already empty".

**Exercise 2.2: A calmer week**
- **Do**: choose **Week of 23 Feb 2026**.
- **See**: **28 of 40**, **59**, **364**, **475**.
- **Meaning**: fewer red, more amber and many unknown stock counts at the start of the test period.
- **Question**: what does a grey piece in a bar mean?
  **Answer**: no stock record for that store and product that week (unknown, not zero).

**Exercise 2.3: One product's tiles**
- **Do**: Week of **6 Apr 2026**, product **PROD-100803**.
- **See**: red tiles **1044, 1109, 1219**; amber tile **1237**; 1 grey; the rest green. Hover over 1044 to see
  In stock 5, Expected 5.7, Chance 68%.
- **Meaning**: this matches Tab 1. The likely transfer is from the amber store to a red store.
- **Question**: is the tile grid a geographic map?
  **Answer**: no. Tiles are in store-number order; real locations are unknown.

**Exercise 2.4: Recorded versus simulated stock**
- **Do**: Week of **13 Apr 2026**, product **PROD-100800**; hover over tile **1080**.
- **See**: Store 1080 is **green**, In stock **1** (recorded).
- **Compare**: in Tab 1 (same week, store and product, AI) Store 1080 shows **0** in stock and red.
- **Meaning**: Store Overview shows what was **recorded**; Decision Support shows the **simulated** stock
  under the chosen approach.
- **Question**: which one should you use to see what really happened?
  **Answer**: Store Overview (recorded data).

### Tab 3 — Results

**Exercise 3.1: Read the cards and takeaway**
- **Do**: open Results (Simulated demand).
- **See**: **19%**, **−1.0%**, **144**, **2,021**; the takeaway names "Rule: cover a high estimate".
- **Question**: did the AI beat every rule?
  **Answer**: no. "Rule: cover a high estimate" saved 2,276; the uncertainty-aware AI saved 2,021.

**Exercise 3.2: Switch to actual past sales**
- **Do**: click **Actual past sales**.
- **See**: **17%**, **−0.7%**, **91**, **1,176**. The takeaway names "Rule: cover a calibrated high estimate".
- **Meaning**: all savings are smaller, because past sales hide demand when shelves were empty.
- **Question**: which approach loses money under both views?
  **Answer**: "Rule: cover the forecast" (−187 simulated, −2,495 actual).

**Exercise 3.3: Read the error bars**
- **Do**: look at the "Money saved" chart (Simulated).
- **See**: black lines on the two AI bars.
- **Question**: what do they show?
  **Answer**: the spread (± one standard deviation) across 5 separately trained agents: 2,021 ± 182 and
  1,885 ± 132. They overlap, so the two AIs are not reliably different.

**Exercise 3.4: Expensive shipping**
- **Do**: in "Show the effect of" choose **Fixed cost per shipment**; hover over the points at **16** and **32**.
- **See**:
  - at 16: uncertainty-aware AI 865, AI agent 856, calibrated rule 411, high-estimate rule 54;
  - at 32: every line is below 0.
- **Meaning**: when each shipment is expensive, fewer shipments win, and at 32 doing nothing is best.
- **Question**: does this prove the AI understands costs?
  **Answer**: no. It never sees costs and was not retrained; it simply ships less.

**Exercise 3.5: Cheap lost sales**
- **Do**: choose **Cost of a lost sale (times the profit per item)**; hover at **0.5**.
- **See**: AI agent **314**, uncertainty-aware AI 299, calibrated rule 20, high-estimate rule **−281**.
- **Question**: why does the high-estimate rule lose money here?
  **Answer**: if a lost sale is worth less, its many shipments cost more than the sales they save.

### Tab 4 — About the Project

**Exercise 4.1**: Read "How it works".
- **Question**: which step uses reinforcement learning?
- **Answer**: step 4, Recommendation.

**Exercise 4.2**: Read "Limitations".
- **Question**: why are distances not real?
- **Answer**: the data has no store locations, so a simulated map is used.

**Exercise 4.3**: Read "Future work".
- **Question**: name one thing needed before using the tool in real shops.
- **Answer**: real store locations and shipping costs, delivery delays, or a pilot in a few stores.

### Tab 5 — Technical Details

**Exercise 5.1: Forecast accuracy**
- **Do**: find the "Always zero" row.
- **Question**: why is its average error the lowest, yet we don't use it?
- **Answer**: 83% of store-weeks sell nothing, so guessing 0 is usually right, but it misses every actual sale.
  Our model has the best RMSE (0.569).

**Exercise 5.2: Likely ranges**
- **Question**: the used range catches 97% of sales but the target is 80%. Is that bad?
- **Answer**: it is wider than needed, because sales are small whole numbers. It still had the best interval
  score on the validation weeks.

**Exercise 5.3: Training runs**
- **Question**: which uncertainty-aware run does Tab 1 use, and why?
- **Answer**: Run 3. It has the best saving on the earlier check (validation) weeks (962). It was not chosen on
  test results.

**Exercise 5.4: Trusting the percentage**
- **Do**: hover over the right-most dots of the last chart.
- **Question**: how many store-weeks support the 65% dot?
- **Answer**: 4. That is too few to trust.

---

## Part 4 — 10-minute demo script for our teacher

Before the demo:
- Open the dashboard 5 minutes early, so the first load is done.
- Check the default view shows Store 1080.

| Time | Click / show | What to say |
|---|---|---|
| **0:00–1:00** | Start on the title. | "Our project is a stock transfer planner for a 40-store retail chain. It forecasts next week's sales for each store and product, flags stores at risk of running out or holding too much, and recommends moving stock between stores. We compare an AI agent trained with reinforcement learning against simple rules. Everything is tested on 8 weeks of real data the models never saw. Transfers are simulated; we did not run this in real shops." |
| **1:00–2:00** | **Store Overview**, default (13 Apr, All products). Point at the 4 cards and the top bars. | "This is the weekly health check from the recorded data. 39 of 40 stores have at least one product at risk. Red means may run out, amber means too much stock, grey means the data has no stock count. Store 1026 has 13 products in red. One honest note: most red here means the shelf is already empty, often for products that sell rarely." |
| **2:00–3:00** | Week **6 Apr 2026**, Product **PROD-100803**. Point at red tile 1044 and amber tile 1237; hover over both. | "Now one product. **(a) Store 1044 is at risk of running out**: 5 in stock, 5.7 expected, 68% chance of selling out. **(b) Store 1237 has extra stock**: 9 units but only 0.7 expected sales a week." |
| **3:00–5:00** | **Decision Support**: week 6 Apr 2026, Store **1044**, PROD-100803, Uncertainty-aware AI agent. Show the chart, the risk box, the cards and the "Why". | "The blue line is our forecast, the shaded band the likely range of 3 to 9 units. The dashed line is the planning week. **(c) Here is the recommended transfer**: 1 unit from Store 1237 to Store 1044, 29 km, cost 8.88. The 'Why' sentence explains it in plain words. The cost is 8 per shipment plus 0.03 per unit per km. The distance comes from a simulated map, because the data has no addresses." |
| **5:00–6:30** | Click **Rule: cover a high estimate**. Open "All transfers…". | "**(d) The difference between the AI and the simple rule.** The rule fills the store up to the high estimate of 9, so it sends 4 units for 11.53, and makes 4 shipments for this product. The AI sends 1. Actual sales that week were 2, so the smaller move was enough." |
| **6:30–7:15** | Change product to **PROD-100801** (same week and store). Switch between AI and rule. | "A second example. This product sells about 0.1 units a week here. The rule still ships, 13 shipments for this product this week; the AI ships nothing. With about a 10% chance of a sale worth 41, the expected benefit is around 4, but the shipment costs 8.88. The AI learned to avoid moves like this." |
| **7:15–8:45** | **Results** (Simulated demand). Cards, takeaway, Money saved chart. Then "Show the effect of" → Fixed cost per shipment. | "Over 8 test weeks, the uncertainty-aware AI avoided 19% of lost sales and saved 2,021 after shipping costs, with 144 shipments. The best approach was actually the simple high-estimate rule, at 2,276. The rule that only covers the forecast loses money. The difference between our two AI agents is not statistically reliable. Costs are assumptions, so we tested them: when shipping gets expensive, the AI saves the most because it ships less, and at 32 per shipment nobody should ship." |
| **8:45–9:30** | **About the Project**, scroll to Limitations. | "Main limitations: simulated distances, assumed costs, past sales hide demand when shelves were empty, many weeks have no stock count, and no delivery delay is modelled." |
| **9:30–10:00** | **Technical Details**: point at Forecast accuracy and the training-runs table. | "Our forecast beats simple methods on the large-error score. Each AI was trained 5 times, and we report averages with their spread, so results are not from one lucky run. Thank you; happy to take questions." |

---

## Part 5 — How the project was built, step by step

The whole pipeline runs with `python scripts/run_pipeline.py` (about one hour on a laptop). Each step is a
separate script. A **pipeline** is a chain of steps where each step uses the previous step's output.

Main **libraries** (ready-made code packages):
- **pandas** and **numpy**: tables and numbers.
- **pyarrow**: reads and writes `.parquet` files (a compact table file format).
- **scipy**: Poisson formulas.
- **LightGBM**: forecasting (it uses the **scikit-learn** interface).
- **Gymnasium**: the standard "game" interface for the simulator.
- **Stable-Baselines3** and **PyTorch**: AI training.
- **matplotlib**: saved figures.
- **Streamlit** and **Altair**: the dashboard and its charts.
- **PyYAML**: the settings file.
- **pytest**: automatic tests (81 tests, all passing at the time of writing).

### 5.0 Safe extraction of the data

- **What**: copy the CSV files out of the original ZIP archives into `data/raw/`.
- **Why**: never touch the originals. The extractor skips unsafe paths and never overwrites files.
- **File**: `scripts/extract_data.py`, `src/inventory_rl/data/extract.py`.
- **Result**: 2 Mendeley files and 2 Kaggle files extracted, with a record in `data/raw/extraction_manifest.json`.

### 5.1 Data checks and cleaning, and why Mendeley was chosen

- **What**: a full check of every dataset; then cleaning rules applied while building the weekly table.
- **Files**: `scripts/run_data_audit.py`, `src/inventory_rl/data/audit.py`, `src/inventory_rl/data/loaders.py`,
  `src/inventory_rl/data/preprocess.py`.
- **Libraries**: pandas, numpy, matplotlib.

**How each problem was handled**

| Problem | How many | What we did |
|---|---|---|
| **Returns** (customer brings an item back) | **7,491** sales rows (6% of 125,751). All have negative quantity and "Is Return = 1"; no contradictions. | Not counted as sales. Their quantity goes into a separate `return_qty` column, and the forecast target `gross_qty` counts only real purchases (`weekly_sales()`). |
| **Negative stock** (book-keeping errors, e.g. −2 on the shelf) | **1,380** stock records (range down to −9). In the weekly table, **555** store-product-weeks contained at least one such record. | Clipped to 0 when adding up physical stock (`stock_at()`), but counted in `neg_book_records`, with the original sum kept in `closing_book_stock`. |
| **Missing stock records** (no stock count covering that week) | **29.4%** of store-product-weeks have no opening stock count (28.0% no closing count). **497 units (3.6% of sales)** were sold in such weeks, which proves missing ≠ zero. | Kept as **unknown (NaN)**, never set to 0 in the data. The dashboard shows them as grey "No stock record". Only the simulator needs a number: it uses 0 for a missing count at the start of an episode, and flags it. |
| **"Open" stock records** (end date 9999-12-31, meaning "still valid") | **22,701** records. | Treated as "valid until today" (`is_open`). |
| **Several stock pools per product** (Full Price, Promo, Markdown…) | — | Added together per store and product. |
| **Weeks with no transactions** | 83% of store-product-weeks. | Recorded as **0 sales** (an observed zero). |
| **Partial first and last weeks** | Data starts on a Sunday (1 Jun 2025) and ends mid-week. | Dropped; we keep 46 full Monday–Sunday weeks, 2 Jun 2025 – 19 Apr 2026. |
| **Empty or constant columns** | "Reason of Return" is fully empty; "Sales Channel" has one value. | Not used. |
| **Duplicates** | 0 duplicate rows. | Nothing needed. |
| **Inconsistency check** | Stock change = minus that day's sales in **79.9%** of 203,777 consecutive stock records; 13.9% are increases without a sale (deliveries or transfers in). | Confirms that weekly stock can be rebuilt from the stock file. |

**Why the Mendeley dataset was chosen**
- Real-looking retail data: transactions **and** stock histories, 40 stores and 2,326 products.
- The same product is sold in many stores (median 19 stores per product), which is exactly what transfers need.
- The two files agree with each other (the 79.9% check above).

**Why the Kaggle datasets were not used** (from `docs/data_audit.md`)
- `sales_data.csv`: units sold are **greater than demand** in 27.6% of rows (impossible), stock movements do
  not add up (correlation only 0.53), and there are only 5 stores.
- `retail_store_inventory.csv`: looks **randomly generated**.
  - Sales show no week-to-week pattern (autocorrelation ≈ −0.002).
  - A store changes region and a product changes category over time.
  - Its "demand forecast" is almost identical to same-day sales (correlation 0.997), which is cheating
    information, not a forecast.
- The Kaggle and Mendeley files have different store and product codes, different years and different
  products, so they cannot be merged.

### 5.2 Weekly data preparation

- **What**: turn transactions and stock records into one row per **store × product × week**.
  - Choose the **top 30 products** by units sold in the **training period only**. Choosing with later data
    would be "looking into the future".
  - Compute, for every row:
    - **sales**: units sold excluding returns;
    - **returns**;
    - **promo share**: the share of units sold at a discount;
    - **opening stock**: Sunday-night stock before the week;
    - **closing stock**;
    - **estimated deliveries**: closing − opening + net sales.
  - Assign each week to train, validation or test:
    - **train** (30 weeks): 2 Jun – 22 Dec 2025;
    - **validation** (8 weeks): 29 Dec 2025 – 16 Feb 2026;
    - **test** (8 weeks): 23 Feb – 13 Apr 2026, week starts.
  - Compute **profit per unit** and **cost per unit** per product from training-period sales.
- **Why**: weekly decisions are realistic for transfers. SKU level (not category level) is where stores really
  run out: at category level only 0.8% of store-weeks start empty, at SKU level 7.5%.
- **Files**: `scripts/run_preprocess.py`, `src/inventory_rl/data/preprocess.py`,
  `src/inventory_rl/features/build.py`.
- **Libraries**: pandas, numpy, pyarrow.
- **Result**:
  - 1,200 series (40 × 30) × 46 weeks = **55,200 rows** (`data/processed/panel.parquet`).
  - Average sales: only **0.25 units per store-product-week**; **82.7%** of weeks have zero sales.
  - Profit per unit ranges from 18 to 94 (median 43).

**Features** (the inputs the forecast model learns from), all using only information from **before** the week:
- sales 1, 2, 3, 4 and 8 weeks ago;
- 4- and 8-week averages and spreads;
- weeks since the last sale;
- last week's promo share;
- opening stock, and stock cover (stock ÷ recent average sales);
- last week's total sales of the product across stores, and of the store across products;
- week of year, month;
- store and product codes.

A test checks that changing future sales does not change past features. This guards against **data leakage**,
i.e. accidentally using information from the future.

### 5.3 Forecasting

- **What**: one **global LightGBM** model (one model for all 1,200 series) predicts next week's sales. It is
  **re-trained every week** on all earlier weeks: the **rolling origin** method, for 38 weeks from 28 Jul 2025.
  - Settings: 6 combinations were tried (Poisson or ordinary objective × 7, 15 or 31 leaves). The **Poisson
    objective with 31 leaves** won on the validation weeks.
  - Compared against simple methods: same as last week, 4- and 8-week averages, long-run average, always zero.
- **Why**: one shared model is needed because each single series is mostly zeros. Re-training weekly means
  every forecast is honest (made without future data).
- **Files**: `scripts/run_forecasting.py`, `src/inventory_rl/forecasting/models.py`,
  `src/inventory_rl/evaluation/metrics.py`.
- **Libraries**: LightGBM, pandas, numpy.
- **Result** (test weeks): RMSE **0.569**, versus 0.588 (8-week average) and 0.746 (same as last week).
  Only a modest gain, because demand is sparse. The most important inputs (last re-training) were stock cover,
  last week's total sales of the product, the product code and opening stock.

### 5.4 Uncertainty ranges (likely ranges)

- **What**: four ways to build an 80% likely range (low–high) around each forecast:
  - **Poisson** (count formula around the forecast);
  - **Quantile regression** (LightGBM predicting the 10th and 90th percentiles directly);
  - **CQR**, conformalized quantile regression (quantile regression widened or narrowed using the errors of the
    last 8 weeks);
  - **error-based conformal** (forecast ± a margin learned from recent errors).

  The ranges are rounded to whole units. The method with the best **interval score** on the validation weeks
  was chosen: **Poisson**.
- **Why**: a forecast of 0.4 means little on its own; a range tells us how much safety stock is needed. The
  rules and the uncertainty-aware AI use the high end of the range.
- **Files**: `src/inventory_rl/uncertainty/intervals.py`, `scripts/run_forecasting.py`.
- **Libraries**: scipy, LightGBM, numpy.
- **Result**: the Poisson range covers 97% of test weeks (target 80%), with average width 0.67 units. It is
  wider than nominal because sales are small whole numbers. Test coverage on weeks with at least one sale is
  79.6% (`PROJECT_STATUS.md`).

### 5.5 Risk flags

- **What**: for every store, product and week we compute:
  - **chance of running out** = P(demand ≥ stock), with Poisson demand;
  - **expected shortage** = average units short;
  - **shortfall** below the high estimate;
  - **excess** above the 8-week high estimate;
  - the **status** (red, amber, green or grey).
- **Why**: managers need a simple flag; the AI and rules use the same numbers.
- **Files**: `scripts/run_risk.py`, `src/inventory_rl/risk/assessment.py`.
- **Libraries**: scipy, pandas.
- **Result**:
  - Test-week mix: 37.9% normal, 34.5% no stock record, 14.0% excess, 13.2% high stockout risk, 0.4% below the
    high estimate.
  - The chance of running out is well calibrated up to about 50% (Brier score **0.035**).

### 5.6 Store network (locations and distances)

- **What**: because the data has **no store addresses**, we placed the 40 stores at random points in a
  250 km × 250 km square (fixed random seed 7, so it is the same every run). Road distance = straight-line
  distance × 1.3, a common allowance for winding roads.
  - Distances range from **7.9 km to 381 km** (median 172 km).
  - Shipping cost = 8 per shipment + 0.03 per unit per km.
  - The code can use real coordinates and real road distances (from the free OSRM routing service) if a file of
    store locations is ever provided. This path is unused.
- **Why**: transfers need a distance to have a cost. Every distance output is labelled "SIMULATED" so it is
  never mistaken for real data.
- **Files**: `src/inventory_rl/routing/network.py` (run by `scripts/run_risk.py`).
- **Libraries**: numpy.
- **Result**: `results/metrics/store_network.csv` and `distance_matrix_km.csv`.

### 5.7 AI agent training

- **What**: a **simulator** (a "game" built from the past data) plays one product across 40 stores for
  8 weeks. Each week:
  1. decide transfers;
  2. shipments arrive immediately;
  3. recorded deliveries are added;
  4. customers arrive;
  5. sales and lost sales are counted;
  6. costs are added up.

  The AI's score each week (its **reward**) is the cost of doing nothing minus the cost of its decisions.
  - **Inputs to the AI**, for each store:
    - stock and spare stock;
    - the forecast;
    - for the uncertainty-aware version, also the high estimate and the CQR range;
    - plus how far into the 8 weeks we are, and the product's profit level.
  - **Output**: for each store, an adjustment between −2 and +2 units to the forecast, which becomes the target
    stock. The same matching step as the rules turns targets into shipments.
  - **Algorithm**: **PPO** (Proximal Policy Optimization, a popular, stable reinforcement-learning method)
    from Stable-Baselines3.
    - The network uses the **same small brain for every store** (a "shared per-store" design), so it learns
      one store-level rule from all 40 stores at once.
    - 200,000 training steps per run.
    - Checked every 20,000 steps on the validation weeks; the best version is kept.
  - Two agents (with and without uncertainty), each trained **5 times** with different random starts:
    10 runs, about 50 minutes in total.
  - Two earlier designs failed and were replaced; they are documented in `docs/methodology.md`. One did worse
    than doing nothing; the other learned to never transfer.
- **Why**: to see whether a learning approach can decide better than fixed rules, and whether giving it
  uncertainty information helps.
- **Files**:
  - `scripts/run_rl.py`;
  - `src/inventory_rl/environment/` (`simulator.py`, `gym_env.py`, `matching.py`, `data.py`, `factory.py`);
  - `src/inventory_rl/agents/ppo.py`, `src/inventory_rl/agents/policy.py`.
- **Libraries**: Gymnasium, Stable-Baselines3, PyTorch, numpy.
- **Result**: trained models in `models/rl/`. Validation savings: 419–750 (AI agent) and 547–962
  (uncertainty-aware).
- **Observed (not designed)**: in the test simulation on actual past sales, **every shipment the chosen AI agent
  recommends is exactly 1 unit**, while the rule sometimes sends 2–4. We have not studied why; it is probably
  because a ±2 adjustment around small forecasts rarely creates a need above 1 unit.

### 5.8 Evaluation

- **What**: all 6 approaches are run on the **test weeks** for all 30 products, under 20 simulated demand runs
  plus actual past sales, using exactly the same demand for every approach.
  - Metrics: fill rate, lost units, leftover stock, shipments, shipping cost, total cost, money saved.
  - For the AIs: mean ± spread over 5 runs, and a Welch t-test.
- **Why**: a fair, like-for-like comparison on data never used for training or choosing.
- **Files**: `scripts/run_evaluation.py`, `src/inventory_rl/evaluation/experiment.py`,
  `src/inventory_rl/evaluation/report.py`.
- **Result** (simulated demand, money saved vs no transfers):

  | Approach | Money saved |
  |---|---|
  | Rule: cover a high estimate | **2,276** |
  | Uncertainty-aware AI | 2,021 ± 182 |
  | Rule: cover a calibrated high estimate | 1,985 |
  | AI agent | 1,885 ± 132 |
  | Rule: cover the forecast | −187 |

  - With actual past sales the calibrated rule is best (1,331).
  - Uncertainty helped the **rules** a lot: high-estimate rule vs forecast-only rule = +2,463.
  - It helped the **AI** only a little: +136, not statistically reliable across runs (p = 0.22).
  - Full report: `results/reports/evaluation_report.md`, `docs/experiments.md`.

### 5.9 Cost sensitivity

- **What**: repeat the test with one assumed cost changed at a time:
  - fixed cost 2, 4, 16, 32;
  - per unit-km 0.0075, 0.015, 0.06, 0.12;
  - holding 0.10, 0.50, 1.00 per year;
  - lost-sale multiplier 0.5, 2, 3.

  That is 15 settings including the base.
- **Why**: costs are assumptions, so conclusions must be checked against them.
- **Files**: `scripts/run_sensitivity.py`, `src/inventory_rl/evaluation/sensitivity.py`.
- **Result**:
  - The holding cost hardly matters.
  - Cheap shipping or expensive lost sales favour the high-estimate rule.
  - Expensive shipping or cheap lost sales favour the AI, because it ships less.
  - At 32 per shipment, no transfers is best.
  - The AI was **not retrained** for these settings. Report: `results/reports/cost_sensitivity_report.md`.

### 5.10 Dashboard

- **What**: a 5-tab Streamlit web page that **only reads** saved results and models; it trains nothing.
- **Why**: to make the results usable by a non-technical store manager.
- **Files**: `dashboard/app.py`, `dashboard/explain.py`, `dashboard/data_access.py`, `.streamlit/config.toml`.
- **Libraries**: Streamlit, Altair, pandas.
- **Result**:
  - Automatic tests open every tab and change the main selections without errors (`tests/test_dashboard.py`).
  - The real server was also checked in a browser.

---

## Part 6 — 20 likely teacher questions

1. **Is this used in real stores?**
   No. It is a simulation on past data. The recommendations show what the system would have suggested.

2. **Where is the data from?**
   The public Mendeley dataset "Retail Transactions and Stocks Data": 125,751 sales rows and 284,755 stock
   records, 40 stores, June 2025 – April 2026. The licence still has to be confirmed on the dataset page.

3. **Why not the Kaggle data?**
   One file has sales greater than demand in 27.6% of rows; the other looks randomly generated, and its
   "forecast" copies same-day sales. Their codes and years also don't match Mendeley.

4. **Why only 30 products?**
   They are the best sellers in the training period. With 2,326 products most series are almost empty.
   30 × 40 stores = 1,200 series was enough to learn from, and fast enough to simulate.

5. **How accurate is the forecast?**
   RMSE 0.569 on the test weeks, better than the 8-week average (0.588) and "same as last week" (0.746). The gain
   is modest, because 83% of weeks have zero sales.

6. **Why is "always zero" better on average error?**
   Because most weeks have no sales. It would never recommend any stock, so we use RMSE, which punishes the
   big misses that matter for stock.

7. **What does the shaded band mean?**
   The likely range of next week's sales: Poisson 10th to 90th percentile around the forecast. On test weeks
   it contained the actual sales 97% of the time.

8. **How is "chance of running out" calculated, and can we trust it?**
   P(demand ≥ stock) with Poisson demand around the forecast. It matches reality well up to about 50%; above
   50% there are very few cases and it was over-confident.

9. **Are the store distances real?**
   No. The data has no addresses, so stores are placed at random on a 250 km map (fixed seed), with road
   distance = straight line × 1.3. It is clearly labelled as simulated.

10. **Where do the costs come from?**
    Profit and item cost per product come from the sales data. Shipping (8 + 0.03 per unit-km), holding (25% a
    year) and lost-sale cost (1 × profit) are assumptions; we tested how results change when they change.

11. **What is reinforcement learning, in one sentence?**
    The AI tries actions in a simulator many times and is rewarded when its decisions cost less than doing
    nothing, so it gradually learns better decisions.

12. **What exactly does the AI decide?**
    For every store, how far above or below the forecast (between −2 and +2 units) the stock target should be.
    A fixed matching step then sends units from the nearest store with spare stock.

13. **Did the AI beat the rules?**
    Not overall. With simulated demand the high-estimate rule saved more (2,276 vs 2,021). The AI did beat the
    forecast-only rule, and it does best when shipping is expensive.

14. **Did uncertainty information help?**
    For the rules, clearly (+2,463 when using the high estimate instead of the forecast). For the AI, only
    slightly (+136), and across 5 training runs this is not statistically reliable (p = 0.22).

15. **Why train each AI 5 times?**
    Training involves randomness. One run can be lucky. Five runs let us report an average and its spread
    honestly.

16. **How do you know you didn't cheat by using test data?**
    The products, settings, range method and AI versions were all chosen using training or validation weeks
    only. The test weeks were used once, for the final evaluation. Forecasts are re-trained weekly on past
    weeks only.

17. **Why does Decision Support show different stock from Store Overview?**
    Decision Support simulates each approach from 23 Feb onward, so its earlier transfers change later stock.
    Store Overview shows the recorded stock.

18. **What about missing stock records?**
    29% of store-weeks have no stock count. We keep them as unknown (grey); we never pretend they are zero in the
    data. Only the simulator treats a missing starting count as 0, which may understate some stores' stock.

19. **Why are savings smaller with actual past sales?**
    When a shelf was empty, customers who wanted the item were never recorded, so past sales understate demand.
    Moving stock to empty stores therefore looks less useful than it probably is.

20. **What would you do next?**
    Use real store locations and shipping costs, add delivery delays and truck capacity, retrain the AI for
    different cost levels, and run a small pilot in a few stores.

---

## Part 7 — Limitations and assumptions

**What is simulated**
- **Transfer outcomes**: all results come from a simulator built on past data; nothing was tried in shops.
- **Store locations and distances**: random map, 250 km square, straight line × 1.3.
- **Simulated demand**: Poisson draws around a 5-week centred average of real sales.
- **Stock in Decision Support**: simulated from 23 Feb 2026 under the chosen approach.

**What is assumed** (all in `config/config.yaml`)
- Shipping cost: **8** per shipment + **0.03** per unit per km.
- Holding cost: **25%** of item cost per year.
- Lost-sale cost: **1 ×** profit per unit (profit itself comes from the data).
- **No delivery delay**: shipments arrive before the week's sales.
- A sending store always keeps at least its own expected sales, rounded up.
- "Excess" means more than the high estimate of **8 weeks** of sales.
- Missing starting stock is treated as **0** in the simulator.
- Recorded deliveries are replayed as they happened; the system does not decide orders from suppliers.

**Other limitations**
- **Censored demand**: when the shelf was empty, true demand was not recorded.
- **Short history**: about 11 months, so there is no full year of seasons. Validation and test are only 8 weeks
  each.
- **Sparse sales**: 83% of store-weeks have zero sales, so forecasts are only modestly better than simple
  methods, and ranges are wider than their target.
- **Scope**:
  - 30 products and 40 stores, one product at a time;
  - no truck capacity, no store space limits, no combined shipments of several products;
  - one sales channel.
- **Statistics**: confidence intervals treat demand runs as independent. They do not cover uncertainty from the
  choice of products, period or cost assumptions. The AI comparison has only 5 runs per agent.
- **AI not retrained per cost setting** in the what-if analysis, and it never sees costs.
- **High stockout percentages** (above 50%) are based on very few cases.
- **Store Overview "red"** mostly means "already empty", often for products that rarely sell.
- **Explanations** ("Why" sentences) describe the situation, not the AI's internal reasoning.
- **Python version**: the dashboard needs Python 3.12; it crashed under Python 3.13 on our machine.
- **Licences** of the datasets must still be confirmed before publishing.
- **No claim of novelty** is made; a literature review is still needed.

---

## Part 8 — Glossary

Terms shown in the dashboard (plus a few used in this guide), one simple sentence each.

| Term | Meaning |
|---|---|
| **Actual sales** | Units customers really bought in that week, not counting returns. |
| **AI agent** | The trained decision-maker that sees stock and the forecast only. |
| **All products** | Option that shows every product at once in Store Overview. |
| **Approach** | A way of deciding transfers (no transfers, one of three rules, or one of two AI agents). |
| **Average error (MAE)** | The average size of the forecast's miss, in units. |
| **Average width** | The average gap between the low and high ends of the likely range. |
| **Bias** | Average of (forecast − actual); positive means the forecast tends to be too high. |
| **Calibrated quantile model (CQR)** | A likely-range method whose high and low ends are adjusted using recent forecast errors. |
| **Censored demand** | Demand that was never recorded because the shelf was empty. |
| **Chance of running out** | Probability that customers want at least as many units as the store has. |
| **Check weeks (validation weeks)** | Weeks (29 Dec 2025 – 16 Feb 2026) used to choose settings and AI versions, never for final results. |
| **Cost** | Shipping cost of a transfer: 8 + 0.03 × units × km. |
| **Cost of a lost sale** | Money lost when a customer can't buy: assumed equal to the profit per unit. |
| **Cost per item per km** | Assumed shipping cost for moving one unit one kilometre (0.03). |
| **Count-based range** | The likely range from the Poisson formula around the forecast (the one used). |
| **Demand** | How many units customers want, which can be more than they could buy. |
| **Demand outlook** | The chart of forecast, actual sales and likely range for one store and product. |
| **Distance (km)** | Simulated road distance between two stores. |
| **Error bar** | The thin line on the AI bars showing the spread across 5 trained agents. |
| **Error-based range** | A likely-range method built from the size of recent forecast errors. |
| **Estimated cost** | The shipping cost of the shown transfer. |
| **Excess stock (amber)** | More stock than is likely to sell in the next 8 weeks. |
| **Expected sales** | The forecast: average number of units the store should sell next week. |
| **Fixed cost per shipment** | Assumed cost of making any shipment, however small (8). |
| **Forecast** | The model's prediction of next week's sales. |
| **From / To** | The sending store and the receiving store of a transfer. |
| **High estimate** | The top of the likely range: sales should be at or below it about 9 weeks in 10. |
| **Holding cost** | Cost of keeping unsold stock (assumed 25% of item cost per year). |
| **In stock** | Units on the shelf at the start of the planning week (simulated in Decision Support, recorded in Store Overview). |
| **Large-error score (RMSE)** | An error measure that punishes big misses more than small ones. |
| **Leftover stock** | Stock still on the shelf at the end of a week. |
| **Likely range** | The band of sales values the model considers likely (80% range). |
| **Lost sales** | Units customers wanted but couldn't buy because the shelf was empty. |
| **Lost sales avoided** | Share of the no-transfer lost sales that an approach prevented. |
| **Machine-learning model** | A program that learns patterns from past data to make predictions; here, LightGBM. |
| **Money saved** | Total cost with no transfers minus total cost with an approach (lost sales + holding + shipping). |
| **No stock record (grey)** | The data has no stock count for that store, product and week. |
| **No transfers** | The baseline approach that never moves stock. |
| **Normal (green)** | Stock is in line with expected demand. |
| **Out of stock** | The store has zero units. |
| **Pairs (store-product pairs)** | One store combined with one product. |
| **Pipeline status** | Which project steps have been run on this computer. |
| **Planning week** | The week you are making a decision for. |
| **Poisson** | A standard formula for counts of random events, used for demand and ranges. |
| **Product** | One specific item (SKU), shown as its subcategory name and code. |
| **Quantile model** | A likely-range method where the model directly predicts the low and high ends. |
| **Quantity** | Number of units to move. |
| **Recommendation approach** | Choice between the uncertainty-aware AI agent and the high-estimate rule in Decision Support. |
| **Recommended transfer** | The suggested shipment for the selected store. |
| **Reinforcement learning** | Training an AI by trial and error in a simulator, rewarding cheaper outcomes. |
| **Rule: cover a calibrated high estimate** | Fill each store up to the CQR high estimate. |
| **Rule: cover a high estimate** | Fill each store up to the Poisson high estimate (the best approach in the simulated-demand test). |
| **Rule: cover the forecast** | Fill each store up to its forecast; it ships a lot and loses money. |
| **Run (training run)** | One complete training of an AI agent from a different random start. |
| **Sales inside the range** | Share of weeks where actual sales fell inside the likely range. |
| **Shipments** | Individual transfers between two stores. |
| **Simulated demand (20 runs)** | Test using 20 random versions of each week's demand around real sales. |
| **Actual past sales** | Test that replays what customers really bought. |
| **Stockout risk (red)** | The store is likely to run out next week (or is below the high estimate, or already empty). |
| **Store** | One of the 40 shops, shown by its anonymous number. |
| **Store Overview** | The tab showing every store's status for a week, from recorded data. |
| **Takeaway** | The automatically written one-paragraph conclusion in Results. |
| **Target** | The intended coverage of the likely range (80%). |
| **Test weeks** | The 8 held-back weeks (23 Feb – 19 Apr 2026) used only for the final results. |
| **Transfer** | Moving units of a product from one store to another. |
| **Uncertainty-aware AI agent** | The trained decision-maker that also sees the likely ranges. |
| **What if the cost assumptions are different?** | Section that repeats the test with one assumed cost changed. |
| **Why** | The generated sentence explaining a transfer in plain words. |
| **Yearly cost of holding stock** | The assumed holding cost as a share of item cost per year (0.25). |
