"""
ThreadLine — Report Figure Set
==========================================================================================
Generates the full set of static figures for the Word report / portfolio write-up.

Run:
    python generate_figures.py path/to/garments_worker_productivity.csv [output_dir]

Requires phase6_pipeline.py, phase7_scenarios.py, phase9_validation.py in the same
directory -- every number in these figures comes from those already-validated
functions, nothing is recomputed with different logic or a different train/test
split. (The standalone phase4_forecasting.py / phase5_optimization.py scripts used
a slightly different holdout window than phase6_pipeline.py's forecast_workload --
using two different splits across figures in the same report would produce numbers
that don't quite agree with each other, so everything here is standardized on the
phase6/7/9 pipeline, which is what Phase 8's dashboard also runs on.)

Figures produced:
  01_team_volatility.png          -- CV of demand_proxy by team, sewing vs finishing
  02_correlation_heatmaps.png     -- driver correlations, sewing vs finishing side by side
  03_demand_distributions.png     -- demand_proxy histograms by department
  04_model_comparison.png         -- Naive vs XGBoost vs LSTM (MAPE, RMSE)
  05_forecast_fan_charts.png      -- actual vs P50/P90, one sewing + one finishing team
  06_feature_importance.png       -- XGBoost gain-based feature importance
  07_backlog_persistence.png      -- % of days with backlog, by team
  08_actual_vs_hindsight_cost.png -- real historical cost vs perfect-hindsight optimizer
  09_worker_allocation.png        -- optimized worker allocation over time, by team
  10_scenario_and_sweep.png       -- Phase 7's two charts, sweep chart now annotated
                                     with the cost-ratio tipping point
  11_three_layer_validation.png   -- Phase 9's B (ceiling + matched) vs C, and A separately
  12_e2e_savings_by_fold.png      -- end-to-end savings per backtest fold
  13_walkforward_mape_by_fold.png -- walk-forward MAPE per fold (shows the outlier fold)
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error

from phase6_pipeline import (
    clean_pipeline, forecast_workload, build_base_stats, optimize_schedule, DEFAULT_CONFIG, FEATURES,
)
from phase7_scenarios import run_scenario
from phase9_validation import (
    actual_historical_cost, hindsight_backtest, walk_forward_forecast_eval, end_to_end_backtest,
)

NAVY = "#1C2B39"
AMBER = "#B25B25"
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "font.size": 10})


def _dept_colors(depts):
    return [NAVY if d == "sewing" else AMBER for d in depts]


def _dept_legend():
    return [Patch(color=NAVY, label="Sewing"), Patch(color=AMBER, label="Finishing")]


# ============================================================================
# 01 — Team volatility ranking
# ============================================================================
def fig_team_volatility(df, output_dir):
    vol = df.groupby(["department", "team_id"])["demand_proxy"].agg(["mean", "std"]).reset_index()
    vol["cv"] = vol["std"] / vol["mean"].replace(0, np.nan)
    vol = vol.sort_values("cv", ascending=False)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(vol["team_id"], vol["cv"], color=_dept_colors(vol["department"]))
    ax.set_xticks(range(len(vol)))
    ax.set_xticklabels(vol["team_id"], rotation=75, ha="right")
    ax.set_ylabel("Coefficient of Variation (demand_proxy)")
    ax.set_title("Team Workload Volatility Ranking")
    ax.legend(handles=_dept_legend())
    plt.tight_layout()
    plt.savefig(f"{output_dir}/01_team_volatility.png", dpi=150)
    plt.close(fig)


# ============================================================================
# 02 — Correlation heatmaps, sewing vs finishing
# ============================================================================
def fig_correlation_heatmaps(df, output_dir):
    cols = ["demand_proxy", "no_of_workers", "smv", "over_time", "incentive",
            "idle_time", "style_change_flag", "targeted_productivity", "actual_productivity"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, dept in zip(axes, ["sewing", "finishing"]):
        corr = df[df["department"] == dept][cols].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
                    cbar=True, annot_kws={"size": 7})
        ax.set_title(f"Driver Correlations — {dept}")
    fig.text(0.5, -0.02,
              "Blank cells in the finishing panel = zero variance in that field "
              "for finishing (correlation is undefined for a constant), not missing data.",
              ha="center", fontsize=8, style="italic")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/02_correlation_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# 03 — Demand signal distributions by department
# ============================================================================
def fig_demand_distributions(df, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, dept, color in zip(axes, ["sewing", "finishing"], [NAVY, AMBER]):
        vals = df[df["department"] == dept]["demand_proxy"]
        ax.hist(vals, bins=30, color=color, alpha=0.85)
        label = "WIP (pieces)" if dept == "sewing" else "Overtime (minutes)"
        ax.set_title(f"{dept} — demand_proxy = {label}")
        ax.set_xlabel(label)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/03_demand_distributions.png", dpi=150)
    plt.close(fig)


# ============================================================================
# 04 — Model comparison: Naive vs XGBoost vs LSTM
# ============================================================================
def compute_model_comparison(df):
    """Fits naive/XGBoost/LSTM once and returns everything downstream figures
    need -- the XGBoost model itself is reused by fig_feature_importance
    instead of being refit from scratch a second time."""
    from sklearn.preprocessing import LabelEncoder
    df = df.copy()
    df["dept_enc"] = LabelEncoder().fit_transform(df["department"])

    dates = sorted(df["date"].unique())
    cutoff = dates[-12]
    train = df[df["date"] < cutoff]
    test = df[df["date"] >= cutoff].copy()

    full_sorted = df.sort_values(["team_id", "date"])
    lagged = full_sorted.assign(naive_pred=full_sorted.groupby("team_id")["demand_proxy"].shift(1))
    naive_eval = test.merge(lagged[["team_id", "date", "naive_pred"]], on=["team_id", "date"]).dropna(subset=["naive_pred"])
    naive_mask = naive_eval["demand_proxy"] > 0
    naive_mape = (naive_eval.loc[naive_mask, "demand_proxy"] - naive_eval.loc[naive_mask, "naive_pred"]).abs().div(
        naive_eval.loc[naive_mask, "demand_proxy"]).mean()
    naive_rmse = np.sqrt(mean_squared_error(naive_eval["demand_proxy"], naive_eval["naive_pred"]))

    train_clean = train.dropna(subset=["demand_proxy_lag1"])
    test_clean = test.dropna(subset=["demand_proxy_lag1"])
    X_train, y_train = train_clean[FEATURES], train_clean["demand_proxy"]
    X_test, y_test = test_clean[FEATURES], test_clean["demand_proxy"]

    model = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42)
    model.fit(X_train, y_train)
    xgb_pred = model.predict(X_test)
    xgb_mask = y_test > 0
    xgb_mape = (y_test[xgb_mask] - xgb_pred[xgb_mask]).abs().div(y_test[xgb_mask]).mean()
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))

    from sklearn.preprocessing import StandardScaler
    from tensorflow import keras
    from tensorflow.keras import layers
    import tensorflow as tf
    tf.random.set_seed(42)
    WINDOW = 3
    tr_X, tr_y, te_X, te_y = [], [], [], []
    for _, g in df.sort_values("date").groupby("team_id"):
        vals, dts = g["demand_proxy"].values, g["date"].values
        for i in range(len(vals) - WINDOW):
            w, target, target_date = vals[i:i + WINDOW], vals[i + WINDOW], dts[i + WINDOW]
            (te_X if pd.Timestamp(target_date) >= pd.Timestamp(cutoff) else tr_X).append(w)
            (te_y if pd.Timestamp(target_date) >= pd.Timestamp(cutoff) else tr_y).append(target)
    tr_X, tr_y, te_X, te_y = map(np.array, (tr_X, tr_y, te_X, te_y))
    scaler = StandardScaler().fit(tr_y.reshape(-1, 1))
    tr_X_s = scaler.transform(tr_X.reshape(-1, 1)).reshape(tr_X.shape)
    te_X_s = scaler.transform(te_X.reshape(-1, 1)).reshape(te_X.shape)
    tr_y_s = scaler.transform(tr_y.reshape(-1, 1)).ravel()
    lstm = keras.Sequential([keras.Input(shape=(WINDOW, 1)), layers.LSTM(16, activation="tanh"), layers.Dense(1)])
    lstm.compile(optimizer="adam", loss="mse")
    lstm.fit(tr_X_s.reshape(-1, WINDOW, 1), tr_y_s, epochs=50, verbose=0)
    lstm_pred = scaler.inverse_transform(lstm.predict(te_X_s.reshape(-1, WINDOW, 1), verbose=0)).ravel()
    lstm_mask = te_y > 0
    lstm_mape = np.mean(np.abs(te_y[lstm_mask] - lstm_pred[lstm_mask]) / te_y[lstm_mask])
    lstm_rmse = np.sqrt(mean_squared_error(te_y, lstm_pred))

    return {
        "models": ["Naive", "XGBoost", "LSTM"],
        "mapes": [naive_mape, xgb_mape, lstm_mape],
        "rmses": [naive_rmse, xgb_rmse, lstm_rmse],
        "xgb_model": model,  # reused by fig_feature_importance
        "train_clean": train_clean,
    }


def fig_model_comparison(comparison, output_dir):
    models, mapes, rmses = comparison["models"], comparison["mapes"], comparison["rmses"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(models, mapes, color=[AMBER, NAVY, "#8C8C8C"])
    axes[0].set_title("Forecast MAPE by Model (lower is better)")
    for i, v in enumerate(mapes):
        axes[0].text(i, v + 0.01, f"{v:.1%}", ha="center")
    axes[1].bar(models, rmses, color=[AMBER, NAVY, "#8C8C8C"])
    axes[1].set_title("Forecast RMSE by Model (lower is better)")
    for i, v in enumerate(rmses):
        axes[1].text(i, v + 10, f"{v:.0f}", ha="center")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/04_model_comparison.png", dpi=150)
    plt.close(fig)


# ============================================================================
# 05 — Forecast fan charts: one sewing team + one finishing team
# ============================================================================
def fig_forecast_fan(forecast_df, output_dir):
    sewing_team = sorted(forecast_df[forecast_df["department"] == "sewing"]["team_id"].unique())[0]
    finishing_team = sorted(forecast_df[forecast_df["department"] == "finishing"]["team_id"].unique())[0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, team, label in zip(axes, [sewing_team, finishing_team], ["WIP (pieces)", "Overtime (minutes)"]):
        g = forecast_df[forecast_df["team_id"] == team].sort_values("date")
        ax.plot(g["date"], g["demand_proxy"], label=f"Actual {label}", color=NAVY)
        ax.plot(g["date"], g["forecast_p50"], label="Forecast (P50)", color=AMBER, linestyle="--")
        ax.fill_between(g["date"], g["forecast_p50"], g["forecast_p90"], alpha=0.15, color=AMBER, label="P50–P90")
        ax.set_title(f"{team} — {label}")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/05_forecast_fan_charts.png", dpi=150)
    plt.close(fig)


# ============================================================================
# 06 — Feature importance
# ============================================================================
def fig_feature_importance(comparison, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    xgb.plot_importance(comparison["xgb_model"], importance_type="gain", ax=ax,
                         title="XGBoost Feature Importance (gain)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/06_feature_importance.png", dpi=150)
    plt.close(fig)


# ============================================================================
# 07 — Backlog persistence by team
# ============================================================================
def fig_backlog_persistence(schedule_df, output_dir):
    persist = (schedule_df.groupby(["department", "team_id"])["backlog"]
               .apply(lambda s: (s > 1e-3).mean()).reset_index(name="pct_days_backlogged")
               .sort_values("pct_days_backlogged", ascending=False))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(persist["team_id"], persist["pct_days_backlogged"] * 100, color=_dept_colors(persist["department"]))
    ax.set_xticks(range(len(persist)))
    ax.set_xticklabels(persist["team_id"], rotation=75, ha="right")
    ax.set_ylabel("% of days with nonzero backlog (under the optimized schedule)")
    ax.set_title("Backlog Persistence by Team")
    ax.legend(handles=_dept_legend())
    plt.tight_layout()
    plt.savefig(f"{output_dir}/07_backlog_persistence.png", dpi=150)
    plt.close(fig)


# ============================================================================
# 08 — Actual historical cost vs. perfect-hindsight optimizer (Phase 9's real baseline)
# ============================================================================
def compute_validation(df):
    """Computes every number the Phase 9 figures need, exactly once. The three
    figures below (actual-vs-hindsight, three-layer summary, e2e-by-fold,
    walk-forward-by-fold) all read from this single result instead of each
    independently re-running walk-forward/end-to-end backtests, which retrain
    forecasting models per fold and are expensive to duplicate three times over."""
    _, hist_full = actual_historical_cost(df, DEFAULT_CONFIG)
    hindsight_full = hindsight_backtest(df, DEFAULT_CONFIG)
    b_ceiling = (1 - hindsight_full / hist_full) * 100

    dates_all = sorted(df["date"].unique())
    fold_starts = np.linspace(int(len(dates_all) * 0.5), len(dates_all) - 3 - 1, 5).astype(int)
    backtest_days = sorted(set(d for start in fold_starts
                                for d in dates_all[start + 1: min(start + 3, len(dates_all) - 1) + 1]))
    df_matched = df[df["date"].isin(backtest_days)]
    _, hist_matched = actual_historical_cost(df_matched, DEFAULT_CONFIG)
    hindsight_matched = hindsight_backtest(df_matched, DEFAULT_CONFIG)
    b_matched = (1 - hindsight_matched / hist_matched) * 100

    e2e = end_to_end_backtest(df)
    e2e_valid = e2e[e2e["status"] == "Optimal"].copy()
    e2e_valid["savings_pct"] = (1 - e2e_valid["realized_cost"] / e2e_valid["actual_cost"]) * 100

    wf = walk_forward_forecast_eval(df)

    return {
        "hist_full": hist_full, "hindsight_full": hindsight_full, "b_ceiling": b_ceiling,
        "b_matched": b_matched, "e2e_valid": e2e_valid, "wf": wf,
    }


def fig_actual_vs_hindsight(validation, output_dir):
    hist_cost, hindsight_cost, savings_pct = (
        validation["hist_full"], validation["hindsight_full"], validation["b_ceiling"])

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(["Actual Historical\nStaffing", "Hindsight-Optimal\nSchedule"],
                   [hist_cost, hindsight_cost], color=[AMBER, NAVY])
    for b, v in zip(bars, [hist_cost, hindsight_cost]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}", ha="center", va="bottom")
    ax.set_ylabel("Total Cost (full 59-day period)")
    ax.set_title(f"Actual vs. Hindsight-Optimal Cost — {savings_pct:.1f}% savings potential")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/08_actual_vs_hindsight_cost.png", dpi=150)
    plt.close(fig)


# ============================================================================
# 09 — Worker allocation over time
# ============================================================================
def fig_worker_allocation(schedule_df, output_dir):
    pivot = schedule_df.pivot(index="date", columns="team_id", values="workers")
    fig, ax = plt.subplots(figsize=(12, 5))
    pivot.plot(ax=ax, legend=False, linewidth=1, alpha=0.8)
    ax.set_title("Optimized Worker Allocation Over Time (all 24 teams)")
    ax.set_ylabel("Workers assigned")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/09_worker_allocation.png", dpi=150)
    plt.close(fig)


# ============================================================================
# 10 — Scenario impact + cost-ratio sweep (with tipping point annotated)
# ============================================================================
def fig_scenario_and_sweep(forecast_df, base_stats, output_dir):
    scenarios = [
        ("Baseline (P50)", {}),
        ("Demand Surge (P90)", {"demand_column": "forecast_p90"}),
        ("Labor Shortage (10%)", {"workforce_shock_pct": 0.10}),
        ("Compound Shock", {"demand_column": "forecast_p90",
                             "downtime_teams": {sorted(base_stats.keys())[0]: 0.20}}),
    ]
    results = [run_scenario(forecast_df, base_stats, name, scen) for name, scen in scenarios]
    results_df = pd.DataFrame(results)
    baseline_cost = results_df.loc[results_df["scenario"] == "Baseline (P50)", "total_cost"].values[0]
    results_df["cost_vs_baseline_pct"] = (results_df["total_cost"] / baseline_cost - 1) * 100

    ratios = [1, 2, 3, 5, 8, 10, 15]
    sweep = []
    for r in ratios:
        cfg = dict(DEFAULT_CONFIG)
        cfg["backlog_penalty_per_piece"] = cfg["overtime_cost_per_min"] * r
        cfg["backlog_penalty_per_min"] = cfg["overtime_cost_per_min"] * r
        sweep.append({"ratio": r, **run_scenario(forecast_df, base_stats, f"ratio_{r}x", {}, cfg)})
    sweep_df = pd.DataFrame(sweep)
    sweep_df["backlog_drop"] = sweep_df["total_backlog"].diff()
    tipping_point = sweep_df.loc[sweep_df["backlog_drop"].idxmin(), "ratio"] if sweep_df["backlog_drop"].notna().any() else None

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(results_df["scenario"], results_df["cost_vs_baseline_pct"], color=AMBER)
    axes[0].set_title("Cost Impact by Scenario (% vs. Baseline)")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].plot(sweep_df["ratio"], sweep_df["total_overtime"], marker="o", label="Overtime (min)", color=NAVY)
    axes[1].plot(sweep_df["ratio"], sweep_df["total_backlog"], marker="o", label="Backlog (units)", color=AMBER)
    if tipping_point is not None:
        axes[1].axvline(tipping_point, color="gray", linestyle=":", linewidth=1.5)
        axes[1].text(tipping_point, axes[1].get_ylim()[1] * 0.9, f" ~{tipping_point}x tipping point",
                     fontsize=8, color="gray")
    axes[1].set_xlabel("Backlog : Overtime cost ratio")
    axes[1].set_title("Overtime vs. Backlog Trade-off")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/10_scenario_and_sweep.png", dpi=150)
    plt.close(fig)


# ============================================================================
# 11 — Three-layer validation summary (Phase 9)
# ============================================================================
def fig_three_layer_validation(validation, output_dir):
    b_ceiling, b_matched = validation["b_ceiling"], validation["b_matched"]
    c_mean = validation["e2e_valid"]["savings_pct"].mean()
    a_mape = validation["wf"]["mape"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = ["B — Full-period\nceiling", "B — Scope-matched\nto backtest", "C — End-to-end\n(realistic)"]
    vals = [b_ceiling, b_matched, c_mean]
    bars = axes[0].bar(labels, vals, color=[NAVY, "#5B7A96", AMBER])
    for b, v in zip(bars, vals):
        axes[0].text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}%", ha="center", va="bottom")
    axes[0].set_title("Savings: Hindsight Ceiling vs. Realistic (Layer B vs. C)")
    axes[0].set_ylabel("% cost savings vs. actual historical staffing")

    axes[1].bar(["A — Walk-forward\nforecast MAPE"], [a_mape * 100], color=AMBER)
    axes[1].text(0, a_mape * 100, f"{a_mape:.1%}", ha="center", va="bottom")
    axes[1].set_title("Forecast Accuracy (Layer A)")
    axes[1].set_ylabel("Mean Absolute Percentage Error")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/11_three_layer_validation.png", dpi=150)
    plt.close(fig)


# ============================================================================
# 12 — End-to-end savings by fold
# ============================================================================
def fig_e2e_by_fold(validation, output_dir):
    e2e_valid = validation["e2e_valid"]
    labels = [pd.Timestamp(c).strftime("%b %d") for c in e2e_valid["cutoff"]]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, e2e_valid["savings_pct"], color=NAVY)
    ax.axhline(e2e_valid["savings_pct"].mean(), color=AMBER, linestyle="--",
               label=f"Mean: {e2e_valid['savings_pct'].mean():.1f}%")
    ax.set_ylabel("End-to-end savings (%)")
    ax.set_title("End-to-End Realistic Savings by Backtest Fold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/12_e2e_savings_by_fold.png", dpi=150)
    plt.close(fig)


# ============================================================================
# 13 — Walk-forward MAPE by fold
# ============================================================================
def fig_walkforward_mape(validation, output_dir):
    wf = validation["wf"]
    labels = [pd.Timestamp(c).strftime("%b %d") for c in wf["cutoff"]]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, wf["mape"] * 100, color=AMBER)
    ax.axhline(wf["mape"].mean() * 100, color=NAVY, linestyle="--",
               label=f"Mean: {wf['mape'].mean():.1%}")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Walk-Forward Forecast MAPE by Fold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/13_walkforward_mape_by_fold.png", dpi=150)
    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    raw_path = sys.argv[1] if len(sys.argv) > 1 else "garments_worker_productivity.csv"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "figures"
    os.makedirs(output_dir, exist_ok=True)

    print("Preparing pipeline (clean, forecast, base stats, baseline schedule)...")
    df = clean_pipeline(raw_path)
    forecast_df = forecast_workload(df)
    base_stats = build_base_stats(df)
    baseline_result = optimize_schedule(forecast_df, base_stats, DEFAULT_CONFIG)
    schedule_df = baseline_result["schedule"]

    print("Computing model comparison (Naive/XGBoost/LSTM) -- shared by figs 04 and 06...")
    comparison = compute_model_comparison(df)

    print("Computing Phase 9 validation (walk-forward + end-to-end backtests) -- "
          "shared by figs 08, 11, 12, 13, run once instead of three times...")
    validation = compute_validation(df)

    steps = [
        ("01 team volatility", lambda: fig_team_volatility(df, output_dir)),
        ("02 correlation heatmaps", lambda: fig_correlation_heatmaps(df, output_dir)),
        ("03 demand distributions", lambda: fig_demand_distributions(df, output_dir)),
        ("04 model comparison", lambda: fig_model_comparison(comparison, output_dir)),
        ("05 forecast fan charts", lambda: fig_forecast_fan(forecast_df, output_dir)),
        ("06 feature importance", lambda: fig_feature_importance(comparison, output_dir)),
        ("07 backlog persistence", lambda: fig_backlog_persistence(schedule_df, output_dir)),
        ("08 actual vs hindsight cost", lambda: fig_actual_vs_hindsight(validation, output_dir)),
        ("09 worker allocation", lambda: fig_worker_allocation(schedule_df, output_dir)),
        ("10 scenario and sweep", lambda: fig_scenario_and_sweep(forecast_df, base_stats, output_dir)),
        ("11 three-layer validation", lambda: fig_three_layer_validation(validation, output_dir)),
        ("12 e2e savings by fold", lambda: fig_e2e_by_fold(validation, output_dir)),
        ("13 walk-forward MAPE by fold", lambda: fig_walkforward_mape(validation, output_dir)),
    ]

    for name, fn in steps:
        print(f"Generating {name}...")
        fn()

    print(f"\nAll figures saved to {output_dir}/")
