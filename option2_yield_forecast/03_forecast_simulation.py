# Databricks notebook source
# Option 2 — Notebook 03: 60/90-Day Forecast Simulation
#
# For each county × crop:
#   1. Generate N=1000 weather scenarios via Monte Carlo (baseline + stress cases)
#   2. Feed each scenario through the trained XGBoost model
#   3. Compute yield distribution: median, P10, P25, P75, P90
#   4. Quantify "yield drop risk" relative to historical average
#   5. Persist forecast table to Delta Lake

# COMMAND ----------

import logging
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    ALERT_THRESHOLDS,
    CROPS,
    FEATURE_DELTA,
    FOCUS_STATES,
    FORECAST_DELTA,
    MODEL_DIR,
    N_SCENARIOS,
    QUANTILE_LEVELS,
    SCENARIOS_DELTA,
    TRAIN_YEARS,
)
from utils import get_spark, read_delta, write_delta
from utils.feature_engineering import get_feature_cols
from utils.weather_forecast_api import fetch_forecast_scenarios

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

TARGET_YEAR = 2025   # forecast year (change as needed)
HORIZONS = [60, 90]

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Load feature data and trained models

# COMMAND ----------

spark = get_spark()
feature_df = read_delta(FEATURE_DELTA, spark=spark)
feat_cols = get_feature_cols(feature_df)

# Load XGBoost models
models = {}
for crop in CROPS:
    model_path = MODEL_DIR / f"xgb_{crop.lower()}.pkl"
    if model_path.exists():
        models[crop] = joblib.load(model_path)
        log.info("Loaded model: %s", model_path)
    else:
        log.warning("Model not found for %s — run notebook 02 first.", crop)

# Historical baselines per county
baselines = (
    feature_df[feature_df["year"].isin(TRAIN_YEARS)]
    .groupby(["fips", "commodity_name"])["yield_bu_ac"]
    .agg(["mean", "std"])
    .reset_index()
    .rename(columns={"mean": "hist_mean", "std": "hist_std"})
)

# County lat/lon lookup
coords = (
    feature_df.dropna(subset=["lat", "lon"])
    .drop_duplicates("fips")[["fips", "lat", "lon", "county_name", "state_abbr"]]
)

# Focus on key states to keep runtime reasonable (remove filter for full run)
focus_fips = feature_df[feature_df["state_abbr"].isin(FOCUS_STATES)]["fips"].unique()
log.info("Focus counties: %d", len(focus_fips))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Per-county yield distribution via Monte Carlo

# COMMAND ----------

def build_scenario_features(
    weather_scenarios_df: pd.DataFrame,
    county_row: pd.Series,
    crop: str,
    feat_cols: list[str],
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge weather scenarios with county fixed-effect features to build
    a feature matrix that the model can score.
    """
    # Get latest available county fixed effects
    county_feats = feature_df[
        (feature_df["fips"] == county_row["fips"]) &
        (feature_df["commodity_name"] == crop)
    ].sort_values("year").iloc[-1]

    n = len(weather_scenarios_df)
    X = pd.DataFrame(index=range(n))

    for col in feat_cols:
        if col in weather_scenarios_df.columns:
            X[col] = weather_scenarios_df[col].values
        elif col in county_feats.index:
            X[col] = float(county_feats[col]) if not pd.isna(county_feats[col]) else 0.0
        else:
            X[col] = 0.0

    # Advance year-trend feature by 1-3 years
    if "year_trend" in X.columns:
        current_trend = county_feats.get("year_trend", 0)
        X["year_trend"] = current_trend + (TARGET_YEAR - 2022)

    return X[feat_cols].fillna(0)


all_forecasts = []
SCENARIO_NAMES = ["baseline", "drought", "excess_rain", "heat_wave", "ideal"]

for fips in focus_fips[:30]:   # Limit to 30 counties for hackathon demo; remove limit for full run
    loc = coords[coords["fips"] == fips]
    if loc.empty:
        continue
    loc = loc.iloc[0]

    try:
        # Generate weather scenarios for all defined stress cases
        scenario_dict = fetch_forecast_scenarios(
            lat=loc["lat"], lon=loc["lon"],
            target_year=TARGET_YEAR,
            horizon_days=max(HORIZONS),
            n_scenarios=N_SCENARIOS // len(SCENARIO_NAMES),
        )
    except Exception as exc:
        log.warning("Scenario fetch failed for %s: %s", fips, exc)
        continue

    for crop in CROPS:
        if crop not in models:
            continue
        model = models[crop]

        county_bl = baselines[
            (baselines["fips"] == fips) & (baselines["commodity_name"] == crop)
        ]
        if county_bl.empty:
            continue
        hist_mean = float(county_bl["hist_mean"].iloc[0])
        hist_std = float(county_bl["hist_std"].iloc[0])

        for scenario_name, weather_df in scenario_dict.items():
            X = build_scenario_features(weather_df, loc, crop, feat_cols, feature_df)
            preds = model.predict(X.values)

            p10, p25, p50, p75, p90 = np.percentile(preds, [10, 25, 50, 75, 90])
            drop_pct = 100 * (hist_mean - p50) / (hist_mean + 1e-6)

            all_forecasts.append(
                {
                    "fips": fips,
                    "county_name": loc.get("county_name", ""),
                    "state_abbr": loc.get("state_abbr", ""),
                    "lat": loc["lat"],
                    "lon": loc["lon"],
                    "commodity_name": crop,
                    "forecast_year": TARGET_YEAR,
                    "scenario": scenario_name,
                    "yield_p10": round(p10, 1),
                    "yield_p25": round(p25, 1),
                    "yield_p50": round(p50, 1),
                    "yield_p75": round(p75, 1),
                    "yield_p90": round(p90, 1),
                    "hist_mean": round(hist_mean, 1),
                    "hist_std": round(hist_std, 1),
                    "drop_pct_vs_hist": round(drop_pct, 1),
                    "weather_tmax_mean": round(weather_df["tmax_mean_c"].mean(), 1),
                    "weather_precip_total": round(weather_df["precip_total_mm"].mean(), 0),
                    "weather_gdd": round(weather_df["gdd_base10"].mean(), 0),
                    "weather_heat_stress_days": round(weather_df["heat_stress_days"].mean(), 1),
                }
            )

forecast_df = pd.DataFrame(all_forecasts)
print(f"Forecast records: {len(forecast_df):,}")
print(forecast_df.groupby(["commodity_name", "scenario"])["yield_p50"].mean().round(1).to_string())

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Risk classification

# COMMAND ----------

def classify_risk(drop_pct: float) -> str:
    if drop_pct < ALERT_THRESHOLDS["yield_drop_pct"]["warning"]:
        return "NORMAL"
    elif drop_pct < ALERT_THRESHOLDS["yield_drop_pct"]["alert"]:
        return "WARNING"
    elif drop_pct < ALERT_THRESHOLDS["yield_drop_pct"]["critical"]:
        return "ALERT"
    else:
        return "CRITICAL"

forecast_df["risk_level"] = forecast_df["drop_pct_vs_hist"].apply(classify_risk)

print("\nRisk distribution (baseline scenario):")
baseline_fc = forecast_df[forecast_df["scenario"] == "baseline"]
print(baseline_fc.groupby(["commodity_name", "risk_level"]).size().unstack(fill_value=0))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Uncertainty quantification summary

# COMMAND ----------

baseline_corn = forecast_df[
    (forecast_df["scenario"] == "baseline") & (forecast_df["commodity_name"] == "Corn")
]

if not baseline_corn.empty:
    print("\n=== Corn Forecast Uncertainty (Baseline Scenario) ===")
    print(f"Median yield P50: {baseline_corn['yield_p50'].mean():.1f} bu/ac")
    print(f"P10 (pessimistic): {baseline_corn['yield_p10'].mean():.1f} bu/ac")
    print(f"P90 (optimistic):  {baseline_corn['yield_p90'].mean():.1f} bu/ac")
    print(f"Mean uncertainty band (P90-P10): {(baseline_corn['yield_p90'] - baseline_corn['yield_p10']).mean():.1f} bu/ac")

    drought_corn = forecast_df[
        (forecast_df["scenario"] == "drought") & (forecast_df["commodity_name"] == "Corn")
    ]
    if not drought_corn.empty:
        drop = baseline_corn["yield_p50"].mean() - drought_corn["yield_p50"].mean()
        drop_pct = 100 * drop / (baseline_corn["yield_p50"].mean() + 1e-6)
        print(f"\nDrought scenario yield drop: {drop:.1f} bu/ac ({drop_pct:.1f}%)")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Persist forecast table

# COMMAND ----------

write_delta(forecast_df, FORECAST_DELTA, spark=spark, partition_by=["commodity_name", "scenario"])

if spark is not None:
    try:
        from config import DATABRICKS_FORECAST_TABLE
        spark.sql(
            f"""
            CREATE TABLE IF NOT EXISTS {DATABRICKS_FORECAST_TABLE}
            USING DELTA LOCATION '{FORECAST_DELTA}'
            """
        )
        print(f"Registered: {DATABRICKS_FORECAST_TABLE}")
    except Exception as exc:
        log.warning("Unity Catalog skipped: %s", exc)

print("\n=== Notebook 03 complete — proceed to 04_uncertainty_viz.py ===")
