
<h2 align="left">Technical Skills</h2>

<p align="left">
  <img src="https://skillicons.dev/icons?i=py,sklearn,pytorch,tensorflow,anaconda,jupyter,vscode,git,github,arduino,raspberrypi&perline=6" />
</p>


<div align="center">

# ThreadLine
### Demand Forecasting & Capacity Planning for Garment Manufacturing

*A decision-support system that forecasts team-level demand, optimizes daily staffing under a mixed-integer program, and validates itself against real historical outcomes — not a single optimistic number.*

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.3-blue)
![PuLP](https://img.shields.io/badge/PuLP-MILP%20Optimization-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/status-validated-success)

**[Live Dashboard](#) · [Analysis Notebook](notebooks/ThreadLine_Analysis.ipynb) · [Figures](figures/)**

</div>

---

## TL;DR

A garment factory's forecasting-and-staffing decisions were never optimized against
actual demand — teams were staffed to rough historical averages, with overtime as the
reactive lever. This project builds the system that would have done better, and then
proves it did, with a walk-forward backtest against what actually happened.

| | |
|---|---|
| **Realistic savings** (walk-forward backtest, forecast error included) | **~72%** (±16pp, 5 folds) |
| **Perfect-hindsight ceiling** (same days, demand known exactly) | ~77% |
| **Forecast accuracy** | XGBoost MAPE 0.31 vs. Naive 0.57 vs. LSTM 0.78 |
| **Operating insight** | Below a ~5x backlog-to-overtime cost ratio, absorbing delay beats paying overtime — above it, the reverse |
| **Dataset** | 1,197 team-day observations, 24 teams, 2 departments, ~59 days |

Every number above is reproduced by running `notebooks/ThreadLine_Analysis.ipynb` top to
bottom — nothing here is asserted without a cell that produces it.

<div align="center">
<img src="figures/04_model_comparison.png" width="700">
</div>

---

## Table of Contents

- [Why this is more than a forecasting notebook](#why-this-is-more-than-a-forecasting-notebook)
- [Architecture](#architecture)
- [Results](#results)
- [Skills demonstrated](#skills-demonstrated)
- [Repo structure](#repo-structure)
- [Running it](#running-it)
- [Honest limitations](#honest-limitations)
- [What I'd do next](#what-id-do-next)

---

## Why this is more than a forecasting notebook

Most portfolio projects stop at "here's a model with good accuracy." Three decisions in
this one exist specifically because that wasn't good enough:

**1. The forecast target is department-aware, not one-size-fits-all.**
WIP (work-in-progress) isn't tracked for the finishing department at all — it's
structurally absent, not missing-at-random. Forecasting it anyway, or filling it with
zero, silently erases half the dataset's signal and — confirmed by actually computing
it — sends a naively-calculated MAPE to `inf`. Fixed by using each department's real
workload signal: WIP for sewing, overtime for finishing.

**2. The capacity constraint is unit-aware, not uniformly applied.**
A garment-piece capacity formula (`workers × time ÷ SMV`) is meaningless for a
department whose demand signal is measured in minutes, not pieces. The MILP dispatches
by department instead of forcing one formula on both — and the README says so, instead
of quietly hoping nobody checks the units.

**3. Savings are reported in three layers, not one triumphant percentage.**

<div align="center">
<img src="figures/11_three_layer_validation.png" width="700">
</div>

- **Layer A** — is the forecast even trustworthy? (walk-forward MAPE, not a single lucky split)
- **Layer B** — best case if demand were known perfectly (isolates optimization value)
- **Layer C** — what the system would have *actually* delivered, forecast error included

The report states plainly that **C < B** — the gap between them is the honest cost of
forecast uncertainty. A hindsight-only number would have been a more impressive slide
and a less true one.

---

## Architecture

```
Raw CSV
   │
   ▼
Phase 1-3   Data audit → cleaning → department-aware feature engineering → EDA
   │
   ▼
Phase 4     Demand forecasting — Naive / XGBoost / LSTM, P50 + P90 quantiles
   │
   ▼
Phase 5     Capacity optimization (MILP) — worker + overtime allocation per team/day
   │
   ▼
Phase 6     Pipeline integration — one callable entry point, config-driven
   │
   ▼
Phase 7     Scenario & sensitivity analysis — surge, labor shortage, cost-ratio sweep
   │
   ▼
Phase 9     Impact validation — walk-forward backtest vs. real historical outcomes
```

Phase 8 (decision dashboard) is `app.py` — run separately with `streamlit run app.py`.
A Streamlit app needs a running server context; it can't execute as a sequential
notebook cell, so it's deliberately kept out of the analysis notebook.

---

## Results

### Forecasting (Phase 4)

XGBoost beats both a naive persistence baseline and an LSTM benchmark — the LSTM
underperforming is reported as the expected result of ~59 days of training history, not
smoothed over by re-tuning until it wins.

| Model | MAPE | RMSE |
|---|---|---|
| Naive (yesterday = today) | 0.574 | 1206 |
| **XGBoost** | **0.306** | **463** |
| LSTM | 0.777 | 1199 |

### Capacity optimization (Phase 5)

<div align="center">
<img src="figures/07_backlog_persistence.png" width="700">
</div>

The optimizer doesn't just minimize cost — it surfaces which teams are *structurally*
under-resourced (chronic backlog even under the best achievable schedule) versus which
ones just need smarter daily reallocation. That distinction is a real staffing
recommendation, not a modeling artifact.

### Scenario stress-testing (Phase 7)

<div align="center">
<img src="figures/10_scenario_and_sweep.png" width="700">
</div>

A 10% labor shortage costs the plant more than a single team's 20% downtime — a finding
about total slack in the system, not any one team's fragility. The cost-ratio sweep
identifies the exact tipping point (~5x) where overtime becomes cheaper than letting
backlog accumulate — a board-ready operating threshold, not a generic insight.

### Validation (Phase 9)

<div align="center">
<img src="figures/08_actual_vs_hindsight_cost.png" width="600">
</div>

Backtested against what management actually staffed historically — not a synthetic
baseline — using the exact same cost formula the optimizer itself uses, so the
comparison is apples-to-apples rather than favorably constructed.

---

## Skills demonstrated

`Time series forecasting` `Quantile regression` `Mixed-integer programming (MILP)`
`Feature engineering` `Walk-forward validation` `Backtesting methodology`
`Scenario & sensitivity analysis` `Interactive dashboards` `Data quality auditing`
`Production-pipeline design` — plus the harder-to-teach skill this README tries to make
visible throughout: **treating every number as something to verify, not just compute.**

---

## Repo structure

```
threadline/
├── README.md
├── requirements.txt
├── app.py                     # Phase 8 dashboard — streamlit run app.py
├── phase6_pipeline.py          # imported by app.py: clean/forecast/optimize pipeline
├── phase7_scenarios.py         # imported by app.py: scenario perturbation layer
├── phase9_validation.py        # walk-forward + end-to-end backtest logic
├── generate_figures.py         # regenerates everything in figures/
├── data/
│   └── garments_worker_productivity.csv
├── notebooks/
│   └── ThreadLine_Analysis.ipynb   # Phases 1-3, 4, 5, 6, 7, 9 — executed, outputs included
└── figures/                    # 13 report-ready static figures
```

---

## Running it

```bash
git clone <your-repo-url>
cd threadline
pip install -r requirements.txt

# Full analysis — forecasting, optimization, scenarios, validation
jupyter notebook notebooks/ThreadLine_Analysis.ipynb

# Interactive dashboard
streamlit run app.py

# Regenerate all 13 report figures (~5 min — retrains XGBoost/LSTM, runs both backtests)
python generate_figures.py data/garments_worker_productivity.csv figures
```

---

## Honest limitations

Being upfront about these is part of the analysis, not a disclaimer bolted on after it.

- **~59 days of data.** Every validation figure is directional, not statistically
  powered — Phase 9 says so explicitly, including a fold-to-fold standard deviation of
  ±16pp that a bigger dataset would narrow, not eliminate through better modeling.
- **Finishing has no independent demand signal.** WIP isn't tracked there; overtime
  stands in as a proxy, which means finishing's "backlog" is structurally zero by
  construction — not evidence of good staffing. Sewing carries the meaningful half of
  that story.
- **Cost assumptions are asserted, not observed.** The source data has no wage or
  backlog-penalty field. Every assumed rate lives in one place
  (`phase6_pipeline.py`'s `DEFAULT_CONFIG`), not buried across formulas.
- **The MILP can't model finishing's worker-to-output relationship** — there's no
  SMV-equivalent throughput rate for that department in this dataset, so reallocating
  workers away from finishing has no modeled effect on its own overtime requirement.
  Flagged as a direction for future work, not quietly assumed away.

## What I'd do next

- A full production year to shrink the validation confidence intervals from directional
  to statistically powered.
- Real wage and backlog-penalty data to replace the documented cost assumptions with
  observed ones.
- A finishing-specific throughput measure (if one exists operationally) to let the MILP
  model worker reallocation effects there the way it already does for sewing.

---

## Dataset

[Garment Worker Productivity](https://archive.ics.uci.edu/dataset/597/productivity+prediction+of+garment+employees)
— 1,197 team-day observations across sewing and finishing departments, Jan–Mar 2015.

<div align="center">

*Built end-to-end: data audit → forecasting → optimization → validated impact.*

</div>
