"""
ThreadLine — Phase 7: Scenario & Sensitivity Analysis
==========================================================================================
Run directly:
    python phase7_scenarios.py path/to/garments_worker_productivity.csv [output_dir]

Requires phase6_pipeline.py in the same directory -- imports clean_pipeline,
forecast_workload, optimize_schedule, build_base_stats, DEFAULT_CONFIG from it
rather than duplicating that logic here.

CORRECTIONS vs. the draft (each validated against the actual pipeline before fixing):

1. TEAM vs TEAM_ID, again: `fdf['team'] == team` and `stats['smv'][team]` use the raw
   number that collides sewing/finishing. Every scenario definition (`rush_order`,
   `downtime_teams`) now takes a team_id string like 'sewing_1', matching every other
   phase.

2. DOWNTIME DOESN'T WORK ON FINISHING AS WRITTEN: the draft models capacity loss by
   inflating SMV. But Phase 5's unit-mismatch fix means finishing's capacity
   constraint doesn't use SMV at all -- it's overtime-minutes-direct. Inflating a
   finishing team's SMV would silently do nothing to its schedule. Fixed with a
   department-dispatched perturbation: sewing still inflates SMV (dimensionally
   correct there); finishing scales its forecasted demand up instead, which has the
   equivalent effect -- more minutes required to clear the same real workload.

3. EFFICIENCY: `run_scenario` in the draft calls `clean_pipeline` and
   `forecast_workload` FRESH for every single scenario -- with 6 base scenarios plus
   a 7-point cost-ratio sweep, that's 13 full reruns, each retraining two XGBoost
   models that produce an IDENTICAL forecast every time, since none of these
   scenarios change the input data the forecaster sees. Fixed the same way Phase 6
   was: prepare the forecast ONCE, run every scenario against that cached result.

4. INFEASIBILITY WOULD CRASH THE WHOLE BATCH: Phase 6's `optimize_schedule` wrapper
   raises RuntimeError on a non-optimal solve, which is correct for a single
   production run but wrong here -- the doc itself says an infeasible Compound Shock
   scenario "is itself a finding," but a raised exception would kill the batch before
   the other 12 scenarios/sweep points ever ran. Fixed to catch a non-optimal status
   per-scenario, record it in the results table (cost as NaN, status preserved), and
   continue the sweep.

5. BACKLOG PENALTY: the draft sweeps a single `backlog_penalty_per_unit`, which
   doesn't exist in the corrected config (Phase 5 split it into
   backlog_penalty_per_piece and backlog_penalty_per_min because a backlog "unit" is
   a different thing -- and different cost -- in each department). Both are scaled
   by the same ratio together, and overtime/backlog totals are reported per
   department too, since sewing and finishing operate on very different absolute
   scales and could have different tipping points.

6. RUSH ORDER EXAMPLE DATE: the draft's example, `pd.Timestamp('2015-02-10')`, falls
   inside the TRAINING period on this dataset, not the forecast window (which starts
   2015-02-28) -- the scenario would silently match zero rows and do nothing. Fixed
   with a real in-window example, and added a check that warns (rather than silently
   no-ops) if a scenario target isn't found.

7. MONOTONICITY: the doc says cost should "strictly increase from Baseline -> Surge
   -> Compound Shock" but leaves it as something you eyeball in a printed table.
   Turned into an explicit check that prints a clear pass/fail instead of relying on
   a person to notice quietly.
"""

import sys
import copy
import numpy as np
import pandas as pd

from phase6_pipeline import (
    clean_pipeline, forecast_workload, build_base_stats, optimize_schedule, DEFAULT_CONFIG,
)


# ============================================================================
# SCENARIO PERTURBATION LAYER
# ============================================================================
def apply_scenario(forecast_df, base_stats, scenario):
    fdf = forecast_df.copy()
    stats = copy.deepcopy(base_stats)
    demand_col = scenario.get("demand_column", "forecast_p50")

    if scenario.get("rush_order"):
        team_id, date, extra_units = scenario["rush_order"]
        date = pd.Timestamp(date)
        mask = (fdf["team_id"] == team_id) & (fdf["date"] == date)
        if not mask.any():
            print(f"  Note: rush_order target ({team_id}, {date.date()}) not found in "
                  f"the forecast window -- scenario has no effect on this run.")
        fdf.loc[mask, demand_col] = fdf.loc[mask, demand_col] + extra_units

    if scenario.get("downtime_teams"):
        for team_id, pct_loss in scenario["downtime_teams"].items():
            if team_id not in stats:
                print(f"  Note: downtime target '{team_id}' not found -- skipped.")
                continue
            if stats[team_id]["department"] == "sewing":
                # inflating SMV = more minutes needed per piece; dimensionally
                # correct since sewing's capacity constraint is piece/SMV-based
                stats[team_id]["smv"] = stats[team_id]["smv"] / (1 - pct_loss)
            else:
                # finishing's capacity constraint is overtime-minutes-direct, no
                # SMV term -- so scale the forecasted demand up instead, which has
                # the equivalent effect (more minutes required for the same work)
                mask = fdf["team_id"] == team_id
                fdf.loc[mask, demand_col] = fdf.loc[mask, demand_col] / (1 - pct_loss)

    if scenario.get("workforce_shock_pct"):
        shock = scenario["workforce_shock_pct"]
        for team_id in stats:
            stats[team_id]["base_workers"] = max(1.0, stats[team_id]["base_workers"] * (1 - shock))

    return fdf, stats, demand_col


# ============================================================================
# SCENARIO RUNNER (operates on an already-prepared forecast -- see fix #3)
# ============================================================================
def run_scenario(forecast_df, base_stats, scenario_name, scenario=None, base_config=None):
    scenario = scenario or {}
    config = dict(base_config or DEFAULT_CONFIG)

    fdf, stats, demand_col = apply_scenario(forecast_df, base_stats, scenario)
    config["demand_column"] = demand_col

    result = optimize_schedule(fdf, stats, config)

    if result["status"] != "Optimal":
        # infeasibility under a stress scenario is a finding, not a crash --
        # record it and let the rest of the sweep continue (see fix #4)
        return {"scenario": scenario_name, "status": result["status"],
                "total_cost": np.nan, "total_overtime": np.nan, "total_backlog": np.nan,
                "sewing_overtime": np.nan, "sewing_backlog": np.nan,
                "finishing_overtime": np.nan, "finishing_backlog": np.nan}

    sched = result["schedule"]
    by_dept = sched.groupby("department").agg(overtime=("overtime_min", "sum"), backlog=("backlog", "sum"))

    return {
        "scenario": scenario_name, "status": result["status"], "total_cost": result["total_cost"],
        "total_overtime": sched["overtime_min"].sum(), "total_backlog": sched["backlog"].sum(),
        "sewing_overtime": by_dept.loc["sewing", "overtime"] if "sewing" in by_dept.index else 0.0,
        "sewing_backlog": by_dept.loc["sewing", "backlog"] if "sewing" in by_dept.index else 0.0,
        "finishing_overtime": by_dept.loc["finishing", "overtime"] if "finishing" in by_dept.index else 0.0,
        "finishing_backlog": by_dept.loc["finishing", "backlog"] if "finishing" in by_dept.index else 0.0,
    }


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    raw_path = sys.argv[1] if len(sys.argv) > 1 else "garments_worker_productivity.csv"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "outputs"

    import os
    os.makedirs(output_dir, exist_ok=True)

    # prepared ONCE, reused across every scenario and every sweep point (fix #3)
    df = clean_pipeline(raw_path)
    forecast_df = forecast_workload(df)
    base_stats = build_base_stats(df)

    # pick a real, in-window example team/date for the rush-order demo, rather
    # than hardcoding a date that might fall outside the forecast horizon
    example_team = sorted(forecast_df["team_id"].unique())[0]
    example_date = sorted(forecast_df["date"].unique())[2]

    scenarios = [
        ("Baseline (P50)", {}),
        ("Demand Surge (P90)", {"demand_column": "forecast_p90"}),
        ("Rush Order", {"rush_order": (example_team, example_date, 500)}),
        ("Line Downtime -- 20%", {"downtime_teams": {example_team: 0.20}}),
        ("Labor Shortage (10%)", {"workforce_shock_pct": 0.10}),
        ("Compound Shock", {"demand_column": "forecast_p90", "downtime_teams": {example_team: 0.20}}),
    ]

    print(f"Rush order / downtime example team: {example_team}, example date: "
          f"{pd.Timestamp(example_date).date()}\n")

    results = [run_scenario(forecast_df, base_stats, name, scen) for name, scen in scenarios]
    results_df = pd.DataFrame(results)

    baseline_cost = results_df.loc[results_df["scenario"] == "Baseline (P50)", "total_cost"].values[0]
    results_df["cost_vs_baseline_pct"] = (results_df["total_cost"] / baseline_cost - 1) * 100
    print(results_df[["scenario", "status", "total_cost", "cost_vs_baseline_pct",
                       "total_overtime", "total_backlog"]].to_string(index=False))

    # explicit pass/fail instead of something to eyeball (fix #7)
    ordered = results_df.set_index("scenario")
    checks = [
        ("Demand Surge >= Baseline",
         ordered.loc["Demand Surge (P90)", "total_cost"] >= ordered.loc["Baseline (P50)", "total_cost"]),
        ("Compound Shock is the worst outcome",
         ordered.loc["Compound Shock", "total_cost"] >= results_df["total_cost"].max() - 1e-6),
    ]
    print("\nMonotonicity checks:")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    # --- cost-ratio sensitivity sweep (Q7) ---
    print("\n--- Cost-ratio sweep ---")
    ratios = [1, 2, 3, 5, 8, 10, 15]
    sweep = []
    for r in ratios:
        cfg = dict(DEFAULT_CONFIG)
        cfg["backlog_penalty_per_piece"] = cfg["overtime_cost_per_min"] * r
        cfg["backlog_penalty_per_min"] = cfg["overtime_cost_per_min"] * r
        res = run_scenario(forecast_df, base_stats, f"ratio_{r}x", {}, cfg)
        sweep.append({"ratio": r, **res})

    sweep_df = pd.DataFrame(sweep)
    print(sweep_df[["ratio", "total_overtime", "total_backlog", "total_cost"]].to_string(index=False))

    sweep_df["backlog_drop"] = sweep_df["total_backlog"].diff()
    if sweep_df["backlog_drop"].notna().any():
        tipping_idx = sweep_df["backlog_drop"].idxmin()
        tipping_point = sweep_df.loc[tipping_idx, "ratio"]
        print(f"\nOvertime becomes the dominant strategy once backlog costs ~{tipping_point}x "
              f"overtime cost (largest single-step drop in total backlog).")

    # --- charts ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].bar(results_df["scenario"], results_df["cost_vs_baseline_pct"], color="#B25B25")
        axes[0].set_title("Cost Impact by Scenario (% vs. Baseline)")
        axes[0].tick_params(axis="x", rotation=45)

        axes[1].plot(sweep_df["ratio"], sweep_df["total_overtime"], marker="o", label="Overtime (min)")
        axes[1].plot(sweep_df["ratio"], sweep_df["total_backlog"], marker="o", label="Backlog (units)")
        axes[1].set_xlabel("Backlog : Overtime cost ratio")
        axes[1].set_title("Overtime vs. Backlog Trade-off")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(f"{output_dir}/scenario_analysis.png", dpi=150)
        plt.close(fig)
        print(f"\nSaved chart to {output_dir}/scenario_analysis.png")
    except ImportError:
        print("(matplotlib not available -- skipped charts)")

    results_df.to_csv(f"{output_dir}/phase7_scenarios.csv", index=False)
    sweep_df.to_csv(f"{output_dir}/phase7_cost_ratio_sweep.csv", index=False)
    print(f"Saved scenario table and sweep table to {output_dir}/")
