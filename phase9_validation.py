"""
ThreadLine — Phase 9: Impact Validation
==========================================================================================
Run directly:
    python phase9_validation.py path/to/garments_worker_productivity.csv [output_dir]

Requires phase6_pipeline.py in the same directory.

CORRECTIONS vs. the draft (each validated against the actual pipeline before fixing):

1. TEAM vs TEAM_ID / workload_pressure vs demand_proxy, again, in all four functions
   (actual_historical_cost, hindsight_backtest, walk_forward_forecast_eval,
   realized_cost_of_schedule). Fixed throughout, matching every other phase.

2. UNIFORM SMV CAPACITY FORMULA, again: `(workers*shift_min + overtime)/smv` applied
   to every team assumes demand is in WIP pieces -- true for sewing, not for
   finishing (Phase 5's unit-mismatch fix). Factored into a shared
   `_capacity_provided()` helper, dispatched by department exactly like the MILP
   itself, so the backtest evaluates schedules using the SAME cost logic that
   produced them -- an inconsistent evaluation formula would make Layer B/C
   comparisons meaningless even if each one were internally correct.

3. A GENUINE DATA-STRUCTURAL FINDING, not a code bug: because finishing has no
   independent WIP signal, its demand_proxy IS over_time (Phase 1-3's fix). That
   means actual_historical_cost's finishing backlog is IDENTICALLY ZERO by
   construction -- confirmed on the real data: demand_proxy == over_time for every
   single finishing row, always. This isn't finishing being well-staffed, it's
   using a variable as its own ground truth. Reported explicitly (with a
   department breakdown) rather than silently folded into one combined number
   that would misleadingly suggest all 24 teams contributed to the finding.

4. base_stats BUILT INLINE WITH THE WRONG SHAPE in both hindsight_backtest and
   end_to_end_backtest -- replaced with build_base_stats(df), matching every
   downstream phase's expectation.

5. BACKLOG PENALTY: single `backlog_penalty_per_unit` doesn't exist in the
   corrected config (Phase 5 split it into per-piece and per-minute versions,
   since a "unit" means a different thing in each department). Dispatched through
   the same `_backlog_penalty()` helper used everywhere else.

6. NO INFEASIBILITY HANDLING IN THE BACKTEST LOOPS: the draft's Step 6 says "every
   optimize_schedule call should return Optimal" as something to check by eye
   afterward -- but nothing in the loop actually stops a non-optimal fold from
   being silently averaged in as if its cost were meaningful. Fixed to record
   status per fold and exclude non-optimal folds from the mean/std instead of
   contaminating them.

9. LAYER B vs LAYER C SCOPE MISMATCH -- this is the one worth understanding, not
   just fixing. The doc's own Step 6 says "expect C < B"; running it produced
   B=44.1%, C=72.0% -- backwards, and a real red flag rather than something to
   report as-is. Cause: Layer B (hindsight_backtest) was computed over the FULL
   59-day dataset, while Layer C (end_to_end_backtest) only covers the ~15 days
   that happen to fall inside the 5 backtest fold windows. A whole-dataset ceiling
   compared against a 15-day realistic result isn't the same comparison -- confirmed
   by recomputing Layer B restricted to those same 15 days: 77.2%, which correctly
   sits above Layer C's 72.0%. Fixed to report BOTH a full-period hindsight ceiling
   (useful on its own) and a backtest-scope-matched hindsight figure -- the latter,
   not the former, is what Layer C should actually be measured against.

7. realized_cost_of_schedule re-wrapped an already-a-DataFrame in pd.DataFrame(...)
   -- harmless but redundant (optimize_schedule already returns one); removed.

8. SELF-INFLICTED MERGE COLLISION, found by actually running this: an early version
   of walk_forward_forecast_eval merged forecast_df (which already carries
   demand_proxy as the ground truth alongside forecast_p50) against df's
   demand_proxy again, colliding column names into demand_proxy_x/demand_proxy_y
   and raising `KeyError: 'demand_proxy'` the moment the code tried to read it back.
   Fixed by using forecast_df's own demand_proxy directly -- the merge was
   unnecessary in the first place.
"""

import sys
import numpy as np
import pandas as pd

from phase6_pipeline import clean_pipeline, forecast_workload, build_base_stats, optimize_schedule, DEFAULT_CONFIG


# ============================================================================
# SHARED COST LOGIC -- same dispatch the MILP itself uses, so the backtest
# evaluates schedules with the same formula that produced them
# ============================================================================
def _capacity_provided(department, workers, overtime_min, smv, shift_min):
    if department == "sewing":
        return (workers * shift_min + overtime_min) / smv
    return overtime_min  # finishing: demand_proxy is already in minutes, no SMV term


def _backlog_penalty(department, config):
    return config["backlog_penalty_per_piece"] if department == "sewing" else config["backlog_penalty_per_min"]


# ============================================================================
# STEP 1 — what actually happened (management's real historical decisions)
# ============================================================================
def actual_historical_cost(df, config):
    df = df.sort_values(["team_id", "date"]).copy()
    records = []
    for team_id, g in df.groupby("team_id"):
        dept = g["department"].iloc[0]
        prev_backlog = 0.0
        for row in g.itertuples():
            capacity = _capacity_provided(dept, row.no_of_workers, row.over_time, row.smv, config["shift_min"])
            backlog = max(0.0, row.demand_proxy + prev_backlog - capacity)
            cost = config["overtime_cost_per_min"] * row.over_time + _backlog_penalty(dept, config) * backlog
            records.append({"team_id": team_id, "department": dept, "date": row.date,
                             "backlog": backlog, "cost": cost})
            prev_backlog = backlog
    hist_df = pd.DataFrame(records)
    return hist_df, float(hist_df["cost"].sum())


# ============================================================================
# STEP 2 — Layer B: optimizer value under perfect hindsight
# ============================================================================
def hindsight_backtest(df, config):
    """Feeds the ACTUAL realized demand_proxy to the optimizer as if it were the
    forecast -- isolates optimization value from forecast-error cost."""
    base_stats = build_base_stats(df)
    hindsight_demand = df[["team_id", "date", "demand_proxy"]].rename(
        columns={"demand_proxy": config["demand_column"]}
    )
    result = optimize_schedule(hindsight_demand, base_stats, config)
    if result["status"] != "Optimal":
        raise RuntimeError(f"Hindsight backtest did not solve to optimality: {result['status']}")
    return result["total_cost"]


# ============================================================================
# STEP 3 — Layer A: forecast accuracy, walk-forward
# ============================================================================
def walk_forward_forecast_eval(df, n_folds=5, horizon=3):
    dates = sorted(df["date"].unique())
    fold_starts = np.linspace(int(len(dates) * 0.5), len(dates) - horizon - 1, n_folds).astype(int)

    results = []
    for start in fold_starts:
        cutoff = dates[start]
        forecast_df = forecast_workload(df, cutoff_date=cutoff)
        window_end_idx = min(start + horizon, len(dates) - 1)
        window_dates = dates[start + 1: window_end_idx + 1]
        window = forecast_df[forecast_df["date"].isin(window_dates)]
        if window.empty:
            continue

        # forecast_workload's output already carries demand_proxy (the ground
        # truth for that period) alongside forecast_p50 -- merging it again
        # from df would collide column names and silently produce
        # demand_proxy_x/demand_proxy_y instead of a usable column
        true, pred = window["demand_proxy"], window["forecast_p50"]
        mask = true > 0
        mape = float((true[mask] - pred[mask]).abs().div(true[mask]).mean()) if mask.any() else np.nan
        results.append({"cutoff": cutoff, "mape": mape, "n_obs": len(window),
                         "n_excluded_zero_true": int((~mask).sum())})

    return pd.DataFrame(results)


# ============================================================================
# STEP 4 — Layer C: end-to-end realistic backtest
# ============================================================================
def realized_cost_of_schedule(schedule_df, actual_df, config):
    merged = schedule_df.merge(
        actual_df[["team_id", "date", "demand_proxy", "smv"]], on=["team_id", "date"]
    ).sort_values(["team_id", "date"])
    if merged.empty:
        return 0.0

    records = []
    for team_id, g in merged.groupby("team_id"):
        dept = g["department"].iloc[0]
        prev_backlog = 0.0
        for row in g.itertuples():
            capacity = _capacity_provided(dept, row.workers, row.overtime_min, row.smv, config["shift_min"])
            backlog = max(0.0, row.demand_proxy + prev_backlog - capacity)
            cost = config["overtime_cost_per_min"] * row.overtime_min + _backlog_penalty(dept, config) * backlog
            records.append(cost)
            prev_backlog = backlog
    return float(sum(records))


def end_to_end_backtest(df, n_folds=5, horizon=3, config=None):
    config = config or DEFAULT_CONFIG
    dates = sorted(df["date"].unique())
    fold_starts = np.linspace(int(len(dates) * 0.5), len(dates) - horizon - 1, n_folds).astype(int)
    base_stats = build_base_stats(df)

    fold_results = []
    for start in fold_starts:
        cutoff = dates[start]
        forecast_df = forecast_workload(df, cutoff_date=cutoff)
        window_end_idx = min(start + horizon, len(dates) - 1)
        window_dates = dates[start + 1: window_end_idx + 1]
        window_forecast = forecast_df[forecast_df["date"].isin(window_dates)]
        if window_forecast.empty:
            continue

        result = optimize_schedule(window_forecast, base_stats, config)
        if result["status"] != "Optimal":
            # a non-optimal fold is a real finding, not something to silently
            # average in as if its cost number meant anything
            fold_results.append({"cutoff": cutoff, "status": result["status"],
                                  "realized_cost": np.nan, "actual_cost": np.nan})
            continue

        realized = realized_cost_of_schedule(result["schedule"], df, config)
        _, actual_cost = actual_historical_cost(df[df["date"].isin(window_dates)], config)
        fold_results.append({"cutoff": cutoff, "status": result["status"],
                              "realized_cost": realized, "actual_cost": actual_cost})

    return pd.DataFrame(fold_results)


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    raw_path = sys.argv[1] if len(sys.argv) > 1 else "garments_worker_productivity.csv"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "outputs"
    import os
    os.makedirs(output_dir, exist_ok=True)

    df = clean_pipeline(raw_path)

    # --- Step 1 ---
    hist_df, hist_total_cost = actual_historical_cost(df, DEFAULT_CONFIG)
    print(f"Actual historical cost (all teams): {hist_total_cost:,.0f}")

    by_dept = hist_df.groupby("department").agg(cost=("cost", "sum"), backlog=("backlog", "sum"))
    print("\nHistorical cost/backlog by department:")
    print(by_dept.round(1).to_string())
    if by_dept.loc["finishing", "backlog"] == 0:
        print("\nNote: finishing backlog is exactly 0 -- expected, not a finding. Finishing's "
              "demand_proxy IS over_time (no independent WIP signal exists for that department), "
              "so 'demand' and 'capacity provided' are definitionally the same number there. Any "
              "backlog-based savings story for finishing would be circular; treat sewing's figures "
              "as the meaningful ones for that part of the narrative.")

    if hist_total_cost <= 0:
        print("\nWARNING: historical cost is ~0 -- the opportunity for savings is genuinely small. "
              "Report that rather than tuning cost-ratio assumptions until a bigger number appears.")

    # --- Step 2: Layer B ---
    hindsight_cost = hindsight_backtest(df, DEFAULT_CONFIG)
    hindsight_savings_pct = (1 - hindsight_cost / hist_total_cost) * 100
    print(f"\nOptimizer cost (perfect hindsight, full {df['date'].nunique()}-day period): {hindsight_cost:,.0f}")
    print(f"Layer B — full-period hindsight ceiling: {hindsight_savings_pct:.1f}%")

    # --- Step 3: Layer A ---
    wf_results = walk_forward_forecast_eval(df)
    print("\nWalk-forward forecast evaluation:")
    print(wf_results.to_string(index=False))
    print(f"Layer A — mean walk-forward MAPE: {wf_results['mape'].mean():.1%} "
          f"(std: {wf_results['mape'].std():.1%})")

    # --- Step 4: Layer C ---
    e2e = end_to_end_backtest(df)
    e2e_valid = e2e[e2e["status"] == "Optimal"].copy()
    if len(e2e_valid) < len(e2e):
        print(f"\nNote: {len(e2e) - len(e2e_valid)} of {len(e2e)} end-to-end folds did not solve "
              f"to optimality and are excluded from the savings estimate below -- an infeasible "
              f"fold under a realistic forecast is itself a finding about staffing-bound tightness.")
    e2e_valid["savings_pct"] = (1 - e2e_valid["realized_cost"] / e2e_valid["actual_cost"]) * 100
    print("\nEnd-to-end backtest by fold:")
    print(e2e_valid.to_string(index=False))

    mean_e2e = e2e_valid["savings_pct"].mean()
    std_e2e = e2e_valid["savings_pct"].std()
    print(f"\nLayer C — mean end-to-end savings: {mean_e2e:.1f}% (std: {std_e2e:.1f}%)")

    if std_e2e is not None and not np.isnan(std_e2e) and std_e2e < 1e-6 and len(e2e_valid) > 1:
        print("WARNING: savings_pct is suspiciously identical across folds -- check that "
              "forecast_workload's cutoff_date is actually being respected and no future data "
              "is leaking into earlier folds.")

    # --- Layer B, restricted to the SAME days as the Layer C backtest folds ---
    # Comparing a whole-dataset hindsight ceiling against a partial-days realistic
    # result isn't apples-to-apples (see fix #9) -- this is the number that
    # Layer C should actually be checked against.
    dates_all = sorted(df["date"].unique())
    fold_starts = np.linspace(int(len(dates_all) * 0.5), len(dates_all) - 3 - 1, 5).astype(int)
    backtest_days = sorted(set(
        d for start in fold_starts
        for d in dates_all[start + 1: min(start + 3, len(dates_all) - 1) + 1]
    ))
    df_matched = df[df["date"].isin(backtest_days)]
    hist_df_matched, hist_cost_matched = actual_historical_cost(df_matched, DEFAULT_CONFIG)
    hindsight_cost_matched = hindsight_backtest(df_matched, DEFAULT_CONFIG)
    hindsight_savings_matched = (1 - hindsight_cost_matched / hist_cost_matched) * 100
    print(f"\nLayer B (scope-matched to the {len(backtest_days)} backtest days): "
          f"{hindsight_savings_matched:.1f}%")

    if mean_e2e > hindsight_savings_matched + 1e-6:
        print(f"WARNING: Layer C ({mean_e2e:.1f}%) exceeds scope-matched Layer B "
              f"({hindsight_savings_matched:.1f}%) -- perfect information should never do worse "
              f"than a realistic forecast. Investigate before trusting either number.")
    else:
        print(f"Sanity check OK: Layer C ({mean_e2e:.1f}%) <= scope-matched Layer B "
              f"({hindsight_savings_matched:.1f}%), as expected.")

    # --- Step 5: summary ---
    summary = pd.DataFrame([
        {"Layer": "B — Hindsight ceiling (full period)", "Result": f"{hindsight_savings_pct:.1f}% savings"},
        {"Layer": "B — Hindsight (scope-matched to backtest)", "Result": f"{hindsight_savings_matched:.1f}% savings"},
        {"Layer": "A — Forecast accuracy (walk-forward)", "Result": f"{wf_results['mape'].mean():.1%} mean MAPE"},
        {"Layer": "C — End-to-end (realistic)", "Result": f"{mean_e2e:.1f}% savings (std {std_e2e:.1f}%)"},
    ])
    print("\n" + "=" * 60)
    print(summary.to_string(index=False))
    print("=" * 60)
    print(f"\nHeadline sentence: on the same days used for backtesting, hindsight optimization "
          f"could save ~{hindsight_savings_matched:.0f}%, but realistic forecast error narrows "
          f"that to ~{mean_e2e:.0f}% achievable today (±{std_e2e:.0f}pp across a "
          f"{len(e2e_valid)}-fold walk-forward backtest on ~{df['date'].nunique()} days of data -- "
          f"directional, not statistically powered; would narrow with a full year of production "
          f"history).")

    hist_df.to_csv(f"{output_dir}/phase9_historical_cost.csv", index=False)
    wf_results.to_csv(f"{output_dir}/phase9_walk_forward_mape.csv", index=False)
    e2e_valid.to_csv(f"{output_dir}/phase9_end_to_end_backtest.csv", index=False)
    print(f"\nSaved detail tables to {output_dir}/")
