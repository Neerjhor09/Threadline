"""
ThreadLine — Consolidated Pipeline (Phases 2, 4, 5, 7): clean -> forecast -> optimize -> stress-test
==========================================================================================
Single cell, paste-and-run. Point RAW_CSV_PATH at your raw source file.

CORRECTIONS vs. the draft (each validated against the actual dataset -- this reintroduced
several bugs already fixed and confirmed in earlier phases of this project):

1. TARGET VARIABLE: forecasts `workload_pressure` again, which is constant 0 for the
   entire finishing department (WIP isn't tracked there). Restored `demand_proxy`
   (wip for sewing, over_time for finishing -- the only workload-adjacent field that
   actually varies for finishing, confirmed earlier: std 1776.7 vs. dead-zero on wip,
   idle_time, idle_men, no_of_style_change).

2. TEAM vs TEAM_ID: `groupby('team')` and `set_index(['team','date'])` reappear in three
   places (rolling/lag features, MILP team set, base_stats). All three collide sewing
   team N with finishing team N. Fixed to team_id throughout. The set_index one is the
   worst of the three -- with a duplicate index, `.to_dict()` silently keeps the last
   row and drops the other, meaning close to half the forecasts disappear with no error.

3. UNIT MISMATCH IN THE MILP: `(w*shift_min + o)/smv` applied uniformly assumes demand
   is in WIP pieces -- true for sewing, but finishing's demand_proxy is over_time,
   measured in minutes. Dispatched by department: sewing stays piece/SMV-based,
   finishing compares minutes to minutes directly (`o[t,d] >= demand + backlog`).
   Follows that BACKLOG_PENALTY can't be one shared constant either -- a backlog
   "unit" is a piece for sewing, a minute for finishing. Split into two.

4. OVERTIME BOUND: flat `max_overtime_min = 120` team-total is ~1-3% of any real
   team's historical overtime. An earlier attempt at fixing this scaled the flat
   number by team size using the dataset's GLOBAL MEDIAN per-worker overtime -- but a
   median is exceeded on ~half of all days by definition, and using it as a hard
   ceiling produced backlog on 100% of days for every finishing team when tested.
   Fixed to each team's own historical 90th-percentile over_time (already team-total,
   already in minutes -- no rescaling needed).

5. MISSING (team, day) COMBINATIONS: `demand.get((t,d), 0)` treats a missing forecast
   as "zero demand", which isn't a fact about the world, it's a gap in the forecast
   table (confirmed: 53 of 288 possible team-day combinations are missing in this
   dataset -- some teams have as few as 29 observations across all 59 dates). Fixed
   to only build variables/constraints over combinations that actually have a
   forecast. The draft's own missing-combo warning code was also unreachable as
   written -- `teams`/`days`/`demand` are local to optimize_schedule() and don't
   exist in the caller's scope, so that diagnostic would raise NameError the moment
   it ran. Fixed by returning the diagnostic from inside the function that has the
   data.

6. QUANTILE MONOTONICITY: P50/P90 trained as independent models with no constraint
   forcing P90 >= P50 on a given row -- confirmed this is violated on ~13% of rows.
   Enforced with max(p50, p90) after prediction.

7. actual_productivity CLIPPED to 1.0: throws away real over-target performance
   (37 rows, up to 1.12 in this dataset) instead of flagging it. Restored the flag-
   don't-clip approach from Phase 1-3.

8. EFFICIENCY / STRUCTURE: the draft called the full pipeline (clean + forecast) TWICE
   to compare P50 vs P90, retraining identical models for no reason since both
   quantiles are already produced in one forecast_workload() call. Split into
   prepare_forecast() [run once] and solve_scenario() [run per demand column], so the
   P50-vs-P90 comparison only re-solves the MILP, not the whole pipeline.

9. DATE PARSING: hardcoded format='%m/%d/%Y' will raise ValueError if fed a file with
   ISO-formatted dates (e.g. output from an earlier phase). Added a fallback so this
   works on either the true raw source or an already-partially-processed file.

10. IDEMPOTENCY: clean_pipeline now drops any pre-existing engineered columns before
    recomputing them, so it's safe to point at raw data OR at a previous run's output
    without producing _x/_y duplicate columns (same fix as Phase 1-3).
"""

import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpInteger, LpStatus, value, PULP_CBC_CMD

# ============================================================================
# CONFIG
# ============================================================================
FEATURES = ["no_of_workers", "smv", "over_time", "incentive", "idle_time",
            "style_change_flag", "demand_proxy_roll3", "demand_proxy_lag1",
            "targeted_productivity", "dept_enc"]

ENGINEERED_COLS = [
    "team_num", "team_id", "over_target_flag", "workload_pressure",
    "demand_proxy", "demand_proxy_source", "demand_proxy_roll3", "demand_proxy_lag1",
    "style_change_flag", "rated_capacity_min", "idle_loss_min",
    "effective_capacity_min", "productivity_gap",
]

DEFAULT_CONFIG = {
    "demand_column": "forecast_p50",   # swap to 'forecast_p90' for the Phase 7 stress test
    "shift_min": 480,
    "overtime_bound_quantile": 0.90,   # each team's own historical percentile -- see fix #4
    "overtime_cost_per_min": 1.5,
    "backlog_penalty_per_piece": 5,    # sewing: cost per unmet WIP piece (assumed)
    "backlog_penalty_per_min": 4.5,    # finishing: cost per unmet overtime-minute (assumed)
}


# ============================================================================
# PHASE 2 — CLEAN + FEATURE ENGINEER
# ============================================================================
def _parse_dates(series):
    try:
        return pd.to_datetime(series, format="%m/%d/%Y")
    except (ValueError, TypeError):
        return pd.to_datetime(series)


def _time_aware_rolling(df, value_col, prefix):
    all_dates = pd.Index(sorted(df["date"].unique()), name="date")
    frames = []
    for tid, g in df.groupby("team_id"):
        s = g.set_index("date")[value_col].reindex(all_dates)
        frames.append(pd.DataFrame({
            "team_id": tid, "date": all_dates,
            f"{prefix}_roll3": s.rolling(3, min_periods=1).mean().values,
            f"{prefix}_lag1": s.shift(1).values,
        }))
    return pd.concat(frames, ignore_index=True)


def clean_pipeline(raw_csv_path):
    df = pd.read_csv(raw_csv_path)
    df.columns = df.columns.str.strip()
    df = df.drop(columns=[c for c in ENGINEERED_COLS if c in df.columns])  # idempotency

    df["department"] = df["department"].str.strip().str.replace("sweing", "sewing", regex=False)
    df["date"] = _parse_dates(df["date"])

    df["team_num"] = df["team"]
    df["team_id"] = df["department"] + "_" + df["team"].astype(str)

    # flag over-target rows instead of clipping them away (real signal, not noise)
    df["over_target_flag"] = (df["actual_productivity"] > 1.0).astype(int)

    df["style_change_flag"] = (df["no_of_style_change"] > 0).astype(int)
    df["rated_capacity_min"] = df["no_of_workers"] * df["smv"]
    df["idle_loss_min"] = df["idle_time"] * df["idle_men"]
    df["effective_capacity_min"] = df["rated_capacity_min"] - df["idle_loss_min"]
    df["productivity_gap"] = df["targeted_productivity"] - df["actual_productivity"]

    # workload_pressure kept only as a raw numeric convenience field -- NOT the
    # forecasting target, since it's constant-zero for all of finishing
    df["workload_pressure"] = df["wip"].fillna(0)
    df["demand_proxy"] = np.where(df["department"] == "sewing", df["wip"], df["over_time"])
    df["demand_proxy_source"] = np.where(df["department"] == "sewing", "wip", "over_time")

    df = df.merge(_time_aware_rolling(df, "demand_proxy", "demand_proxy"),
                  on=["team_id", "date"], how="left")

    return df.sort_values(["team_id", "date"]).reset_index(drop=True)


# ============================================================================
# PHASE 4 — QUANTILE FORECASTING
# ============================================================================
def forecast_workload(df, cutoff_date=None, quantiles=(0.5, 0.9)):
    df = df.copy()
    le = LabelEncoder()
    df["dept_enc"] = le.fit_transform(df["department"])

    unique_dates = sorted(df["date"].unique())
    if cutoff_date is None:
        cutoff_date = unique_dates[int(len(unique_dates) * 0.8)]  # unique-date based, not row-based

    train = df[df["date"] <= cutoff_date]
    test = df[df["date"] > cutoff_date].copy()

    # drop rows missing the lag feature (each team's first observed day) instead
    # of fillna(0), which would fabricate a fake "no backlog" history
    train_clean = train.dropna(subset=["demand_proxy_lag1"])
    test_clean = test.dropna(subset=["demand_proxy_lag1"])

    X_train, y_train = train_clean[FEATURES], train_clean["demand_proxy"]
    X_test = test_clean[FEATURES]

    preds = {}
    for q in quantiles:
        model = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=q,
                                  n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
        model.fit(X_train, y_train)
        preds[q] = model.predict(X_test)

    test_clean = test_clean.copy()
    test_clean["forecast_p50"] = preds[0.5]
    test_clean["forecast_p90"] = np.maximum(preds[0.5], preds[0.9])  # enforce P90 >= P50

    return test_clean[["team_id", "date", "department", "demand_proxy",
                        "forecast_p50", "forecast_p90"]]


# ============================================================================
# PHASE 5 — CAPACITY OPTIMIZATION (MILP)
# ============================================================================
def build_base_stats(df):
    g = df.groupby(["team_id", "department"]).agg(
        smv=("smv", "mean"),
        base_workers=("no_of_workers", "mean"),
        overtime_p90=("over_time", lambda s: s.quantile(0.90)),
    ).reset_index()
    return {row.team_id: {
        "department": row.department, "smv": row.smv, "base_workers": row.base_workers,
        "overtime_p90": row.overtime_p90,
    } for row in g.itertuples()}


def _vname(t, d):
    return f"{t}_{pd.Timestamp(d).strftime('%Y%m%d')}"


def optimize_schedule(forecast_df, base_stats, config):
    demand_col = config["demand_column"]
    demand_df = forecast_df.dropna(subset=[demand_col])
    demand = {(row.team_id, row.date): getattr(row, demand_col) for row in demand_df.itertuples()}
    combos = sorted(demand.keys())
    teams = sorted({t for t, _ in combos})
    days_by_team = {t: sorted(d for tt, d in combos if tt == t) for t in teams}

    min_workers = {t: max(1, int(0.7 * base_stats[t]["base_workers"])) for t in teams}
    max_workers = {t: max(2, int(1.3 * base_stats[t]["base_workers"])) for t in teams}
    max_overtime = {t: base_stats[t]["overtime_p90"] * (config["overtime_bound_quantile"] / 0.90)
                     for t in teams}  # rescale if a caller overrides the quantile assumption
    total_workforce = int(sum(base_stats[t]["base_workers"] for t in teams))

    prob = LpProblem("Capacity_Scheduling", LpMinimize)
    w = {(t, d): LpVariable(f"w_{_vname(t, d)}", min_workers[t], max_workers[t], LpInteger) for (t, d) in combos}
    o = {(t, d): LpVariable(f"o_{_vname(t, d)}", 0, max_overtime[t]) for (t, d) in combos}
    bl = {(t, d): LpVariable(f"bl_{_vname(t, d)}", 0) for (t, d) in combos}

    obj_terms = []
    for (t, d) in combos:
        dept = base_stats[t]["department"]
        penalty = config["backlog_penalty_per_piece"] if dept == "sewing" else config["backlog_penalty_per_min"]
        obj_terms.append(config["overtime_cost_per_min"] * o[t, d] + penalty * bl[t, d])
    prob += lpSum(obj_terms)

    for t in teams:
        prev_bl = 0
        dept = base_stats[t]["department"]
        for d in days_by_team[t]:
            if dept == "sewing":
                capacity = (w[t, d] * config["shift_min"] + o[t, d]) / base_stats[t]["smv"]
            else:  # finishing: demand and capacity are already in minutes, no SMV conversion
                capacity = o[t, d]
            prob += capacity + bl[t, d] >= demand[t, d] + prev_bl
            prev_bl = bl[t, d]

    all_days = sorted({d for _, d in combos})
    for d in all_days:
        teams_today = [t for t in teams if d in days_by_team[t]]
        prob += lpSum(w[t, d] for t in teams_today) <= total_workforce

    # gapRel accepts a solution within 0.1% of proven-optimal instead of forcing
    # CBC to close the last fraction of a percent via exhaustive branch-and-bound --
    # on a large combined solve (e.g. a full-period hindsight backtest across all
    # teams and every date at once) that difference is 235+ seconds of runtime for
    # a change in the reported cost of about 0.001%. timeLimit is a hard safety net.
    # PuLP still reports status "Optimal" when solved within the gap tolerance.
    prob.solve(PULP_CBC_CMD(msg=0, gapRel=0.001, timeLimit=90))

    schedule = pd.DataFrame([
        {"team_id": t, "date": d, "department": base_stats[t]["department"],
         "workers": w[t, d].varValue, "overtime_min": o[t, d].varValue, "backlog": bl[t, d].varValue}
        for (t, d) in combos
    ])

    # diagnostic returned from the function that actually has the data,
    # instead of dangling in the caller's scope where it would NameError
    full_grid = {(t, d) for t in teams for d in all_days}
    missing = sorted(full_grid - set(combos))

    return {
        "status": LpStatus[prob.status],
        "total_cost": value(prob.objective),
        "schedule": schedule,
        "missing_combos": missing,
    }


# ============================================================================
# ORCHESTRATION
# ============================================================================
def prepare_forecast(raw_csv_path):
    """Run once: clean + forecast. Reused across scenario runs so comparing
    P50 vs P90 doesn't retrain identical models twice."""
    df = clean_pipeline(raw_csv_path)
    forecast_df = forecast_workload(df)
    base_stats = build_base_stats(df)
    return forecast_df, base_stats


def solve_scenario(forecast_df, base_stats, config, output_dir=None, label=None):
    result = optimize_schedule(forecast_df, base_stats, config)
    if result["status"] != "Optimal":
        raise RuntimeError(f"Solver did not find an optimal solution: {result['status']}")
    if result["missing_combos"]:
        print(f"Note: {len(result['missing_combos'])} team-day combinations had no forecast "
              f"and were excluded from the schedule (not treated as zero demand).")

    label = label or config["demand_column"]
    print(f"[{label}] Solver status: {result['status']} | Total cost: {result['total_cost']:.2f}")

    if output_dir:
        result["schedule"].to_csv(f"{output_dir}/schedule_{label}.csv", index=False)
        with open(f"{output_dir}/cost_summary_{label}.json", "w") as f:
            json.dump({"total_cost": result["total_cost"], "scenario": label}, f, indent=2)

    return result["schedule"], result["total_cost"]


def run_pipeline(raw_csv_path, config=None, output_dir="outputs"):
    """Convenience wrapper for a single scenario run."""
    config = config or DEFAULT_CONFIG
    forecast_df, base_stats = prepare_forecast(raw_csv_path)
    return solve_scenario(forecast_df, base_stats, config, output_dir=output_dir)


# ============================================================================
# USAGE (only runs when this file is executed directly, not when imported --
# without this guard, `import phase6_pipeline` from Phase 7 would silently
# re-run the whole P50/P90 pipeline as a side effect of the import)
# ============================================================================
if __name__ == "__main__":
    RAW_CSV_PATH = "garments_worker_productivity.csv"
    OUTPUT_DIR = "outputs"

    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    forecast_df, base_stats = prepare_forecast(RAW_CSV_PATH)

    schedule_p50, cost_p50 = solve_scenario(
        forecast_df, base_stats, {**DEFAULT_CONFIG, "demand_column": "forecast_p50"},
        output_dir=OUTPUT_DIR, label="p50")

    schedule_p90, cost_p90 = solve_scenario(
        forecast_df, base_stats, {**DEFAULT_CONFIG, "demand_column": "forecast_p90"},
        output_dir=OUTPUT_DIR, label="p90")

    assert cost_p90 >= cost_p50, "P90 (higher demand) should never cost less than P50 -- check the pipeline."
    print(f"\nP50 cost: {cost_p50:.2f} | P90 cost: {cost_p90:.2f} | Surge premium: {cost_p90 - cost_p50:.2f}")
