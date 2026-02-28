# Databricks notebook source
# Option 1 — Notebook 04: Anomaly Detection & Explanation
#
# Identifies counties/seasons where yield is unusually high or low
# relative to weather conditions, then explains WHY using SHAP local
# explanations and rule-based interpretation.

# COMMAND ----------

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    ANOMALY_DELTA,
    ANOMALY_ZSCORE_THRESHOLD,
    CROPS,
    ISOLATION_FOREST_CONTAMINATION,
    MERGED_DELTA,
    TRAIN_YEARS,
)
from utils import get_spark, read_delta, write_delta

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Load merged data

# COMMAND ----------

spark = get_spark()
df = read_delta(MERGED_DELTA, spark=spark)

WEATHER_FEATURES = [
    c for c in [
        "tmax_mean_c", "tmin_mean_c", "tavg_mean_c",
        "precip_total_mm", "gdd_base10", "heat_stress_days",
        "drought_days", "et0_total_mm", "precip_z", "cwsi",
    ]
    if c in df.columns
]

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Layer 1 — Statistical anomaly (Z-score)

# COMMAND ----------

# Per-county Z-score was pre-computed in notebook 01, but we recompute
# to ensure we use a robust baseline (all years excluding the test row).
df = df.sort_values(["fips", "commodity_name", "year"])

def robust_zscore(series: pd.Series) -> pd.Series:
    """Z-score using leave-one-out median/IQR for robustness."""
    result = pd.Series(index=series.index, dtype=float)
    for idx in series.index:
        rest = series.drop(idx)
        med = rest.median()
        iqr = rest.quantile(0.75) - rest.quantile(0.25)
        if iqr == 0:
            result[idx] = 0.0
        else:
            result[idx] = (series[idx] - med) / (iqr / 1.349)  # normalise IQR to std
    return result

df["yield_z_robust"] = df.groupby(["fips", "commodity_name"])["yield_bu_ac"].transform(
    robust_zscore
)
df["z_anomaly"] = df["yield_z_robust"].abs() > ANOMALY_ZSCORE_THRESHOLD
df["z_direction"] = np.where(
    df["yield_z_robust"] < -ANOMALY_ZSCORE_THRESHOLD, "LOW",
    np.where(df["yield_z_robust"] > ANOMALY_ZSCORE_THRESHOLD, "HIGH", "NORMAL"),
)

print(f"Z-score anomalies: {df['z_anomaly'].sum():,} / {len(df):,}")
print(df[df["z_anomaly"]].groupby(["year", "z_direction"]).size().unstack(fill_value=0))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Layer 2 — Isolation Forest (multivariate anomaly in weather-yield space)

# COMMAND ----------

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    anomaly_all = []

    for crop in CROPS:
        subset = df[df["commodity_name"] == crop].dropna(subset=WEATHER_FEATURES + ["yield_bu_ac"]).copy()
        features_and_yield = WEATHER_FEATURES + ["yield_bu_ac"]
        X = subset[features_and_yield].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        iso = IsolationForest(
            n_estimators=200,
            contamination=ISOLATION_FOREST_CONTAMINATION,
            random_state=42,
        )
        subset["if_score"] = iso.fit_predict(X_scaled)           # -1 = anomaly
        subset["if_anomaly_score"] = -iso.score_samples(X_scaled)  # higher = more anomalous

        anomaly_all.append(subset)

    df_if = pd.concat(anomaly_all, ignore_index=True)
    df_if["if_anomaly"] = df_if["if_score"] == -1

    print(f"Isolation Forest anomalies: {df_if['if_anomaly'].sum():,}")
    print(df_if[df_if["if_anomaly"]].groupby("year").size().rename("n_anomalies").to_string())

    # Combine both anomaly flags
    df_if["is_anomaly"] = df_if["z_anomaly"] | df_if["if_anomaly"]

except ImportError:
    log.warning("scikit-learn not available — using Z-score only.")
    df_if = df.copy()
    df_if["if_anomaly"] = False
    df_if["if_anomaly_score"] = np.nan
    df_if["is_anomaly"] = df_if["z_anomaly"]

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Explain anomalies — weather driver attribution

# COMMAND ----------

EXTREME_EVENTS = {
    2012: "Great Plains Drought",
    2019: "Wet Spring / Prevent Plant",
    2021: "Western Drought / Heat Dome",
    2022: "Southern Heat Wave",
    2023: "Mixed conditions",
}

def explain_anomaly(row: pd.Series) -> str:
    """Generate a human-readable explanation for a yield anomaly row."""
    parts = []

    # Precipitation signal
    if "precip_z" in row and not pd.isna(row["precip_z"]):
        if row["precip_z"] < -1.5:
            parts.append(f"severe drought (precip {row['precip_total_mm']:.0f} mm, {row['precip_z']:.1f} SD below avg)")
        elif row["precip_z"] > 1.5:
            parts.append(f"excess moisture (precip {row['precip_total_mm']:.0f} mm, {row['precip_z']:.1f} SD above avg)")

    # Heat stress
    if "heat_stress_days" in row and not pd.isna(row["heat_stress_days"]):
        if row["heat_stress_days"] > 20:
            parts.append(f"extreme heat stress ({row['heat_stress_days']:.0f} days >35°C)")

    # GDD signal (too high or too low)
    if "gdd_base10" in row and "commodity_name" in row:
        gdd_threshold_low = 1200 if row.get("commodity_name") == "Corn" else 900
        if row["gdd_base10"] < gdd_threshold_low:
            parts.append(f"insufficient heat accumulation (GDD={row['gdd_base10']:.0f})")

    # CWSI (crop water stress)
    if "cwsi" in row and not pd.isna(row.get("cwsi")):
        if row["cwsi"] > 2.0:
            parts.append(f"high crop water stress index ({row['cwsi']:.1f})")

    # Known extreme event
    year = int(row.get("year", 0))
    if year in EXTREME_EVENTS:
        parts.append(f"known climate event: {EXTREME_EVENTS[year]}")

    if not parts:
        parts.append("cause unclear — investigate local factors (pest, disease, management)")

    direction = row.get("z_direction", "")
    prefix = f"{'Unexpectedly LOW' if direction == 'LOW' else 'Unexpectedly HIGH'} yield (Z={row.get('yield_z_robust', 0):.2f}): "
    return prefix + "; ".join(parts)


anomaly_df = df_if[df_if["is_anomaly"]].copy()
anomaly_df["explanation"] = anomaly_df.apply(explain_anomaly, axis=1)

print(f"\nTotal flagged anomalies: {len(anomaly_df):,}")
print("\nSample explanations:")
for _, row in anomaly_df.sample(min(5, len(anomaly_df)), random_state=1).iterrows():
    print(f"  [{row.get('state_abbr','?')}, {row.get('county_name','?')}, {row.get('year','?')}, {row.get('commodity_name','?')}]")
    print(f"  → {row['explanation']}\n")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Drought 2012 deep-dive

# COMMAND ----------

print("=== 2012 Drought Analysis ===")
drought_2012 = df_if[(df_if["year"] == 2012)].copy()
print(f"Counties in 2012 dataset: {drought_2012['fips'].nunique()}")
print(f"Anomalies flagged: {drought_2012['is_anomaly'].sum()}")
print(f"Mean yield Z-score (2012): {drought_2012['yield_z_robust'].mean():.2f}")

corn_drought = drought_2012[drought_2012["commodity_name"] == "Corn"]
print(f"\nCorn 2012 — median yield: {corn_drought['yield_bu_ac'].median():.1f} bu/ac")
print(f"Corn median across all years: {df_if[df_if['commodity_name']=='Corn']['yield_bu_ac'].median():.1f} bu/ac")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Persist anomaly table

# COMMAND ----------

save_cols = [
    "fips", "state_abbr", "county_name", "commodity_name", "irr_name",
    "year", "yield_bu_ac", "yield_z_robust", "z_direction",
    "z_anomaly", "if_anomaly", "is_anomaly",
    "precip_total_mm", "gdd_base10", "heat_stress_days",
    "precip_z", "cwsi", "lat", "lon",
    "explanation",
]
save_cols = [c for c in save_cols if c in anomaly_df.columns]

write_delta(anomaly_df[save_cols], ANOMALY_DELTA, spark=spark, partition_by=["commodity_name"])

print("\n=== Notebook 04 complete — proceed to 05_visualization_maps.py ===")
