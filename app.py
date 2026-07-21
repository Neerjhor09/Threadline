"""
ThreadLine — Phase 8: Decision Dashboard (Streamlit)
==========================================================================================
Run:
    streamlit run app.py

Requires phase6_pipeline.py and phase7_scenarios.py in the same directory.

CORRECTIONS vs. the draft (each validated against the actual pipeline before fixing):

1. WRONG IMPORT SOURCE: `from pipeline import ...` doesn't match any file this project
   actually has -- fixed to import from phase6_pipeline.py (clean_pipeline,
   forecast_workload, build_base_stats, optimize_schedule, DEFAULT_CONFIG) and
   phase7_scenarios.py (apply_scenario), which is where this logic actually lives.

2. TEAM vs TEAM_ID, again, in the team selector, the schedule pivot, and the
   downtime dropdown. Fixed to team_id everywhere, matching every other phase.

3. FORECAST TAB WOULD SHOW "inf%" MAPE FOR EVERY FINISHING TEAM: the draft plots
   `workload_pressure` as "Actual WIP" and computes MAPE against it directly. WIP is
   constant 0 for the entire finishing department, and MAPE divides by |y_true| --
   confirmed by running it: selecting finishing_1 in the original code returns a MAPE
   of literal `inf`, not a large-but-finite number. Fixed to use `demand_proxy`
   (already in forecast_df's output from Phase 4/6 -- no extra merge needed), guarded
   against zero-true-value rows the same way Phase 4's evaluation was, with the
   excluded-row count shown rather than hidden. Also relabels the chart/metric
   per department ("WIP (pieces)" for sewing, "Overtime (minutes)" for finishing) --
   showing "Actual WIP" for a finishing team would be actively misleading given
   finishing has no WIP signal at all (Phase 1's audit finding).

4. base_stats BUILT INLINE, WRONG SHAPE: `{'smv': df.groupby('team')...}` reuses the
   same team/team_id bug and doesn't match the nested per-team_id structure
   build_base_stats() (and everything downstream) actually expects. Replaced with a
   direct call to build_base_stats(df).

5. NO CACHING ON THE MILP SOLVE: Streamlit reruns the entire script top-to-bottom on
   every widget interaction anywhere in the app -- moving a What-If slider in tab 3
   would silently re-solve the Tab 2 baseline schedule too, every time, even though
   nothing about it changed. Wrapped the baseline solve in its own @st.cache_data
   function, separate from load_data(), since a MILP solve and a data-cleaning step
   have very different costs and don't need to invalidate together.

6. INFEASIBLE BASELINE WASN'T GUARDED: Tab 2 called optimize_schedule directly (which
   doesn't raise on a non-optimal status, unlike the solve_scenario wrapper) but never
   checked `status` before using total_cost -- if bounds were ever tightened to the
   point of infeasibility, the tab would silently show a stale or meaningless number
   instead of an error. Added the same explicit status check used in Phase 7.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from phase6_pipeline import (
    clean_pipeline, forecast_workload, build_base_stats, optimize_schedule, DEFAULT_CONFIG,
)
from phase7_scenarios import apply_scenario

st.set_page_config(page_title="ThreadLine — Capacity Planner", layout="wide")

NAVY = "#1C2B39"
AMBER = "#B25B25"
RAW_CSV_PATH = "data/garments_worker_productivity.csv"


@st.cache_data
def load_data():
    df = clean_pipeline(RAW_CSV_PATH)
    forecast_df = forecast_workload(df)
    base_stats = build_base_stats(df)
    return df, forecast_df, base_stats


@st.cache_data
def compute_baseline_schedule(_forecast_df, _base_stats):
    """Cached separately from load_data(): this is a MILP solve, not just data
    prep, and shouldn't re-run every time an unrelated widget fires elsewhere
    in the app (see fix #5). Underscore-prefixed args tell Streamlit not to
    spend time hashing large DataFrames it already knows are constant here."""
    return optimize_schedule(_forecast_df, _base_stats, DEFAULT_CONFIG)


def guarded_mape(y_true, y_pred):
    """MAPE excluding exact-zero true values -- dividing by a true zero sends
    the metric toward its epsilon-guarded ceiling instead of contributing a
    real 0 error. Confirmed this isn't hypothetical: computing the unguarded
    version directly on a finishing team's demand_proxy returns `inf`."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mask = y_true > 0
    excluded = int((~mask).sum())
    if mask.sum() == 0:
        return np.nan, excluded
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask])), excluded


df, forecast_df, base_stats = load_data()

st.title("ThreadLine — Demand Forecasting & Capacity Planner")
st.caption("Forecast-driven production scheduling for garment manufacturing")

tab1, tab2, tab3 = st.tabs(["📈 Forecast", "🗓️ Schedule", "⚙️ What-If"])

# ============================================================================
# TAB 1 — FORECAST: is the model trustworthy?
# ============================================================================
with tab1:
    col1, col2 = st.columns([1, 3])
    with col1:
        team_sel = st.selectbox("Team", sorted(forecast_df["team_id"].unique()))

    team_forecast = forecast_df[forecast_df["team_id"] == team_sel].sort_values("date")
    is_sewing = team_forecast["department"].iloc[0] == "sewing"
    demand_label = "WIP (pieces)" if is_sewing else "Overtime (minutes)"

    with col2:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(team_forecast["date"], team_forecast["demand_proxy"],
                label=f"Actual {demand_label}", color=NAVY)
        ax.plot(team_forecast["date"], team_forecast["forecast_p50"],
                label="Forecast (P50)", color=AMBER, linestyle="--")
        ax.fill_between(team_forecast["date"], team_forecast["forecast_p50"], team_forecast["forecast_p90"],
                         alpha=0.15, color=AMBER, label="P50–P90 range")
        ax.legend()
        ax.set_title(f"{team_sel} — Workload Forecast ({demand_label})")
        st.pyplot(fig)

    mape, excluded = guarded_mape(team_forecast["demand_proxy"], team_forecast["forecast_p50"])
    st.metric("Forecast MAPE (this team)", f"{mape:.1%}" if not np.isnan(mape) else "n/a")
    if excluded:
        st.caption(f"{excluded} day(s) excluded from MAPE (zero true demand that day — "
                   f"a percentage error against zero isn't a meaningful number).")

# ============================================================================
# TAB 2 — SCHEDULE: what's the optimizer recommending, and what does it cost?
# ============================================================================
with tab2:
    baseline_result = compute_baseline_schedule(forecast_df, base_stats)

    if baseline_result["status"] != "Optimal":
        st.error(f"Baseline schedule solve returned '{baseline_result['status']}' — "
                 f"check worker/overtime bounds before trusting anything else in this tab.")
    else:
        schedule_df = baseline_result["schedule"]  # already a DataFrame -- optimize_schedule
                                                      # returns one directly

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Cost", f"{baseline_result['total_cost']:,.0f}")
        c2.metric("Total Overtime (min)", f"{schedule_df['overtime_min'].sum():,.0f}")
        c3.metric("Total Backlog", f"{schedule_df['backlog'].sum():,.0f}")

        st.subheader("Worker Allocation by Team")
        pivot = schedule_df.pivot(index="date", columns="team_id", values="workers")
        st.line_chart(pivot)

        st.subheader("Full Schedule")
        st.dataframe(schedule_df, use_container_width=True)
        st.download_button("Download schedule.csv", schedule_df.to_csv(index=False), "schedule.csv")

# ============================================================================
# TAB 3 — WHAT-IF: how does the schedule change if conditions shift?
# ============================================================================
with tab3:
    st.subheader("Stress-Test the Schedule")

    col1, col2, col3 = st.columns(3)
    with col1:
        demand_scenario = st.radio("Demand scenario", ["Expected (P50)", "Surge (P90)"])
    with col2:
        team_options = ["None"] + sorted(forecast_df["team_id"].unique().tolist())
        downtime_team = st.selectbox("Simulate downtime on team", team_options)
        downtime_pct = st.slider("Downtime severity (%)", 0, 50, 0, disabled=(downtime_team == "None")) / 100
    with col3:
        workforce_shock = st.slider("Labor shortage (%)", 0, 30, 0) / 100

    scenario = {"demand_column": "forecast_p90" if demand_scenario == "Surge (P90)" else "forecast_p50"}
    if downtime_team != "None" and downtime_pct > 0:
        scenario["downtime_teams"] = {downtime_team: downtime_pct}
    if workforce_shock > 0:
        scenario["workforce_shock_pct"] = workforce_shock

    if st.button("Run Scenario", type="primary"):
        fdf, stats, demand_col = apply_scenario(forecast_df, base_stats, scenario)
        cfg = dict(DEFAULT_CONFIG)
        cfg["demand_column"] = demand_col
        scenario_result = optimize_schedule(fdf, stats, cfg)

        if scenario_result["status"] != "Optimal":
            st.error(f"No feasible schedule under these conditions ({scenario_result['status']}). "
                     f"This itself is a finding — current staffing bounds can't absorb this "
                     f"combination of shocks.")
        elif baseline_result["status"] != "Optimal":
            st.warning("Scenario solved, but the baseline itself didn't — % vs. baseline isn't "
                       "meaningful right now. Showing absolute cost only.")
            st.metric("Scenario Cost", f"{scenario_result['total_cost']:,.0f}")
        else:
            delta_pct = (scenario_result["total_cost"] / baseline_result["total_cost"] - 1) * 100
            st.metric("Scenario Cost", f"{scenario_result['total_cost']:,.0f}",
                      delta=f"{delta_pct:+.1f}% vs. baseline")
