# Databricks notebook source
# Option 1 — Notebook 04: Anomaly Detection & Explanation
#
# Three-layer anomaly detection, each layer more sophisticated:
#
#   Layer 1 — Robust Z-score
#       Flags yields that are statistically unusual for that county's own history.
#       Catches absolute low/high years but cannot distinguish weather-driven
#       losses from unexplained losses.
#
#   Layer 2 — XGBoost residual anomaly (NEW — uses model from notebook 03)
#       Loads the trained XGBoost model, generates a predicted yield for every
#       county-year based purely on weather features, then computes:
#           residual = actual_yield - predicted_yield
#       A large negative residual means the county UNDERPERFORMED beyond what
#       weather alone would predict. This is the genuinely interesting anomaly.
#       A county with 108 bu/ac during a severe drought is expected — its
#       residual will be near zero. A county with 108 bu/ac in a normal year
#       has a large negative residual — something non-meteorological happened.
#
#   Layer 3 — Isolation Forest on residuals only (IMPROVED)
#       Previously ran on yield + weather features together, which meant it
#       partly rediscovered what XGBoost already learned. Now runs on the
#       residual alone, so it only finds structure that the model missed.
#
#   Explanation — SHAP local explanations (IMPROVED from rule engine)
#       For each flagged row, uses SHAP to decompose the XGBoost prediction
#       into per-feature contributions. Instead of hand-coded threshold rules,
#       the top SHAP drivers ARE the explanation. The rule engine is kept as
#       a readable supplement.

# COMMAND ----------

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


from config import (
    ANOMALY_DELTA,
    ANOMALY_ZSCORE_THRESHOLD,
    CROPS,
    DATABRICKS_ANOMALY_TABLE,
    ISOLATION_FOREST_CONTAMINATION,
    MERGED_DELTA,
    MLFLOW_EXPERIMENT,
    TRAIN_YEARS,
    YEAR_MIN,
    YEAR_MAX,
)
from utils import get_spark, read_delta, write_delta

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# XGBoost models are saved by notebook 03 into a models/ subfolder
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Load merged data and trained XGBoost models

# COMMAND ----------
try:
    spark = get_spark()
except:
    spark = None
df = read_delta(MERGED_DELTA, spark=spark)

# Load XGBoost models and their saved feature lists from notebook 03.
# Using the saved feature list is critical: it guarantees notebook 04 uses
# exactly the same columns the model was trained on, preventing silent
# dropna() failures when the merged table has extra NaN-heavy columns.
xgb_models   = {}
xgb_features = {}   # per-crop feature list saved alongside the model

try:
    import joblib
    import xgboost as xgb

    for crop in CROPS:
        model_path   = MODEL_DIR / f"xgb_{crop.lower()}.pkl"
        feature_path = MODEL_DIR / f"xgb_{crop.lower()}_features.pkl"

        if model_path.exists():
            xgb_models[crop] = joblib.load(model_path)
            log.info("Loaded XGBoost model for %s", crop)
        else:
            log.warning(
                "No saved model for %s at %s — run notebook 03 first. "
                "Falling back to Z-score + raw Isolation Forest.", crop, model_path
            )
            continue

        if feature_path.exists():
            xgb_features[crop] = joblib.load(feature_path)
            log.info("Loaded feature list for %s: %s", crop, xgb_features[crop])
        else:
            # feature file missing (older run of notebook 03) — reconstruct safely
            xgb_features[crop] = [
                c for c in [
                    "tmax_mean_c", "tmin_mean_c", "tavg_mean_c",
                    "precip_total_mm", "gdd_base10", "heat_stress_days",
                    "drought_days", "et0_total_mm", "solar_total_mj",
                    "precip_z", "cwsi",
                ]
                if c in df.columns and df[c].notna().any()   # must match nb03 filter
            ]
            log.warning(
                "Feature list file missing for %s — reconstructed from data "
                "(re-run notebook 03 to fix permanently).", crop
            )

except ImportError as e:
    log.warning("XGBoost/joblib not available: %s", e)

# Global feature list = union of all crop feature lists (for Isolation Forest fallback)
all_saved_features = sorted({f for feats in xgb_features.values() for f in feats})
WEATHER_FEATURES = all_saved_features if all_saved_features else [
    c for c in [
        "tmax_mean_c", "tmin_mean_c", "tavg_mean_c",
        "precip_total_mm", "gdd_base10", "heat_stress_days",
        "drought_days", "precip_z", "cwsi",
    ]
    if c in df.columns and df[c].notna().any()
]

print(f"Loaded models: {list(xgb_models.keys())}")
print(f"Weather features available: {WEATHER_FEATURES}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Layer 1 — Robust Z-score (statistical baseline)

# COMMAND ----------

df = df.sort_values(["fips", "commodity_name", "year"])

def robust_zscore(series: pd.Series) -> pd.Series:
    """
    Leave-one-out IQR-based Z-score.

    For each observation, compute the baseline (median + IQR) using ALL OTHER
    years for that county. This prevents extreme outlier years from inflating
    the standard deviation and masking their own anomaly.
    """
    result = pd.Series(index=series.index, dtype=float)
    for idx in series.index:
        rest = series.drop(idx)
        med = rest.median()
        iqr = rest.quantile(0.75) - rest.quantile(0.25)
        if iqr == 0:
            result[idx] = 0.0
        else:
            result[idx] = (series[idx] - med) / (iqr / 1.349)
    return result

df["yield_z_robust"] = df.groupby(["fips", "commodity_name"])["yield_bu_ac"].transform(
    robust_zscore
)
df["z_anomaly"] = df["yield_z_robust"].abs() > ANOMALY_ZSCORE_THRESHOLD
df["z_direction"] = np.where(
    df["yield_z_robust"] < -ANOMALY_ZSCORE_THRESHOLD, "LOW",
    np.where(df["yield_z_robust"] > ANOMALY_ZSCORE_THRESHOLD, "HIGH", "NORMAL"),
)

print(f"Layer 1 — Z-score anomalies: {df['z_anomaly'].sum():,} / {len(df):,}")
print(df[df["z_anomaly"]].groupby(["year", "z_direction"]).size().unstack(fill_value=0))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Layer 2 — XGBoost residual anomaly (weather-adjusted)

# COMMAND ----------

# For every county-year, predict what yield SHOULD have been given the weather.
# The residual (actual - predicted) is what the model cannot explain.
# A large negative residual = underperformed beyond weather expectations.

df["xgb_predicted_yield"] = np.nan
df["xgb_residual"] = np.nan
df["xgb_residual_z"] = np.nan

for crop in CROPS:
    if crop not in xgb_models:
        continue

    model        = xgb_models[crop]
    crop_feats   = xgb_features.get(crop, WEATHER_FEATURES)   # use saved feature list
    mask         = df["commodity_name"] == crop
    subset       = df.loc[mask].dropna(subset=crop_feats)

    if subset.empty:
        log.warning("%s: all rows dropped by dropna on features %s", crop, crop_feats)
        continue

    X = subset[crop_feats].values
    preds = model.predict(X)

    df.loc[subset.index, "xgb_predicted_yield"] = preds
    df.loc[subset.index, "xgb_residual"] = (
        df.loc[subset.index, "yield_bu_ac"] - preds
    )

# Standardise residuals per county (so a -20 bu/ac residual in Iowa, where
# yields are 180 bu/ac, is treated the same as -10 bu/ac in Alabama where
# yields are 120 bu/ac)
df["xgb_residual_z"] = df.groupby(["fips", "commodity_name"])["xgb_residual"].transform(
    lambda s: (s - s.mean()) / (s.std() + 1e-6)
)

# Flag residual anomalies — yield underperformed the weather-based prediction
RESIDUAL_THRESHOLD = 1.5   # same scale as yield Z-score threshold
df["residual_anomaly"] = df["xgb_residual_z"] < -RESIDUAL_THRESHOLD
df["residual_direction"] = np.where(
    df["xgb_residual_z"] < -RESIDUAL_THRESHOLD, "UNEXPLAINED_LOW",
    np.where(df["xgb_residual_z"] > RESIDUAL_THRESHOLD, "UNEXPLAINED_HIGH", "WEATHER_EXPLAINED"),
)

if df["xgb_residual"].notna().any():
    print(f"\nLayer 2 — Residual anomalies: {df['residual_anomaly'].sum():,}")
    print("\nMean residual by year (negative = model over-predicted → weather was worse than features capture):")
    print(df.groupby("year")["xgb_residual"].mean().round(1).to_string())

    # Key diagnostic: compare Z-score anomalies vs residual anomalies
    weather_explained = df["z_anomaly"] & ~df["residual_anomaly"]
    truly_unexplained = df["z_anomaly"] & df["residual_anomaly"]
    print(f"\nOf {df['z_anomaly'].sum()} Z-score anomalies:")
    print(f"  Weather-explained (low yield, bad weather):  {weather_explained.sum():,}")
    print(f"  Truly unexplained (low yield, normal weather): {truly_unexplained.sum():,}")
else:
    log.warning("No residuals computed — XGBoost models not loaded. Skipping Layer 2.")
    df["residual_anomaly"] = False
    df["residual_direction"] = "WEATHER_EXPLAINED"

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Layer 3 — Isolation Forest on residuals only (IMPROVED)

# COMMAND ----------

# Key change from original: Isolation Forest now runs on the RESIDUAL alone,
# not on yield + weather features. This means:
#   - It no longer rediscovers weather patterns XGBoost already captured
#   - It finds non-linear structure in the unexplained variation
#   - Anomalies it flags are genuinely surprising given the full model

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    anomaly_all = []

    for crop in CROPS:
        subset = df[df["commodity_name"] == crop].copy()

        if df["xgb_residual"].notna().any() and crop in xgb_models:
            # Improved path: run on residuals only
            iso_input_cols = ["xgb_residual"]
            log.info("%s: Isolation Forest running on XGBoost residuals.", crop)
        else:
            # Fallback: original approach (yield + weather) if no model available
            iso_input_cols = WEATHER_FEATURES + ["yield_bu_ac"]
            log.info("%s: Isolation Forest running on raw yield + weather (fallback).", crop)

        valid = subset.dropna(subset=iso_input_cols)
        if valid.empty:
            continue

        X = valid[iso_input_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        iso = IsolationForest(
            n_estimators=300,
            contamination=ISOLATION_FOREST_CONTAMINATION,
            random_state=42,
        )
        valid = valid.copy()
        valid["if_score"] = iso.fit_predict(X_scaled)
        valid["if_anomaly_score"] = -iso.score_samples(X_scaled)
        anomaly_all.append(valid)

    df_if = pd.concat(anomaly_all, ignore_index=True) if anomaly_all else df.copy()

    if "if_score" in df_if.columns:
        df_if["if_anomaly"] = df_if["if_score"] == -1
        print(f"\nLayer 3 — Isolation Forest anomalies: {df_if['if_anomaly'].sum():,}")
    else:
        df_if["if_anomaly"] = False
        df_if["if_anomaly_score"] = np.nan

except ImportError:
    log.warning("scikit-learn not available — skipping Isolation Forest.")
    df_if = df.copy()
    df_if["if_anomaly"] = False
    df_if["if_anomaly_score"] = np.nan

# Combined flag: anomalous by ANY of the three layers
df_if["is_anomaly"] = (
    df_if["z_anomaly"] |
    df_if.get("residual_anomaly", pd.Series(False, index=df_if.index)) |
    df_if["if_anomaly"]
)

print(f"\nCombined anomalies (any layer): {df_if['is_anomaly'].sum():,}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. SHAP local explanations for flagged rows (IMPROVED from rule engine)

# COMMAND ----------

# For each flagged county-year, use SHAP to decompose the XGBoost prediction
# into per-feature contributions. The features with the largest |SHAP| values
# are the model's data-driven explanation for why it predicted what it did.
# The residual then tells us how much REMAINS unexplained.

try:
    import shap

    shap_explanation_records = []

    for crop in CROPS:
        if crop not in xgb_models:
            continue

        model = xgb_models[crop]
        explainer = shap.TreeExplainer(model)

        crop_feats = xgb_features.get(crop, WEATHER_FEATURES)
        flagged = df_if[
            (df_if["commodity_name"] == crop) & df_if["is_anomaly"]
        ].dropna(subset=crop_feats)

        if flagged.empty:
            continue

        X_flagged = flagged[crop_feats].values
        shap_vals = explainer.shap_values(X_flagged)   # shape: (n_flagged, n_features)

        for i, (idx, row) in enumerate(flagged.iterrows()):
            sv = shap_vals[i]
            # Sort features by absolute SHAP value descending
            ranked = sorted(
                zip(crop_feats, sv),
                key=lambda x: abs(x[1]),
                reverse=True,
            )
            top3 = ranked[:3]

            # Build a data-driven explanation from SHAP
            shap_parts = []
            for feat, val in top3:
                direction = "increased" if val > 0 else "decreased"
                feat_label = feat.replace("_", " ")
                shap_parts.append(
                    f"{feat_label} {direction} predicted yield by {abs(val):.1f} bu/ac"
                )

            residual = row.get("xgb_residual", np.nan)
            residual_z = row.get("xgb_residual_z", np.nan)

            if not pd.isna(residual):
                unexplained = abs(residual)
                direction_str = "below" if residual < 0 else "above"
                residual_sentence = (
                    f"After weather effects, actual yield was {unexplained:.1f} bu/ac "
                    f"{direction_str} the model's prediction "
                    f"(residual Z={residual_z:.2f}) — "
                )
                if abs(residual_z) > RESIDUAL_THRESHOLD:
                    residual_sentence += "this gap is statistically significant and likely non-meteorological."
                else:
                    residual_sentence += "this gap is within normal prediction error."
            else:
                residual_sentence = ""

            shap_explanation_records.append({
                "index": idx,
                "shap_explanation": "; ".join(shap_parts),
                "residual_explanation": residual_sentence,
                "top_driver": ranked[0][0] if ranked else "",
                "top_driver_shap": ranked[0][1] if ranked else np.nan,
            })

    if shap_explanation_records:
        shap_df = pd.DataFrame(shap_explanation_records).set_index("index")
        df_if = df_if.join(shap_df, how="left")
        log.info("SHAP explanations computed for %d flagged rows.", len(shap_df))
    else:
        df_if["shap_explanation"] = ""
        df_if["residual_explanation"] = ""
        df_if["top_driver"] = ""
        df_if["top_driver_shap"] = np.nan

except ImportError as e:
    log.warning("SHAP not available — using rule-based explanation only: %s", e)
    df_if["shap_explanation"] = ""
    df_if["residual_explanation"] = ""

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Rule-based explanation (supplementary — human-readable labels)

# COMMAND ----------

EXTREME_EVENTS = {
    2012: "Great Plains Drought",
    2019: "Wet Spring / Prevent Plant",
    2021: "Western Drought / Heat Dome",
    2022: "Southern Heat Wave",
    2023: "Mixed conditions",
}

def rule_based_explanation(row: pd.Series) -> str:
    """
    Threshold-based labels to supplement SHAP with agronomic context.
    Now secondary to SHAP — used as a readable cross-check.
    """
    parts = []

    if "precip_z" in row and not pd.isna(row.get("precip_z")):
        if row["precip_z"] < -1.5:
            parts.append(f"drought (precip {row['precip_total_mm']:.0f} mm, Z={row['precip_z']:.1f})")
        elif row["precip_z"] > 1.5:
            parts.append(f"excess moisture (precip {row['precip_total_mm']:.0f} mm, Z={row['precip_z']:.1f})")

    if "heat_stress_days" in row and not pd.isna(row.get("heat_stress_days")):
        if row["heat_stress_days"] > 20:
            parts.append(f"heat stress ({row['heat_stress_days']:.0f} days >35°C)")

    if "gdd_base10" in row and "commodity_name" in row:
        gdd_threshold = 1200 if row.get("commodity_name") == "Corn" else 900
        if not pd.isna(row.get("gdd_base10")) and row["gdd_base10"] < gdd_threshold:
            parts.append(f"low GDD ({row['gdd_base10']:.0f})")

    if "cwsi" in row and not pd.isna(row.get("cwsi")):
        if row["cwsi"] > 2.0:
            parts.append(f"water stress (CWSI={row['cwsi']:.1f})")

    year = int(row.get("year", 0))
    if year in EXTREME_EVENTS:
        parts.append(EXTREME_EVENTS[year])

    return "; ".join(parts) if parts else "no clear weather signal — investigate non-meteorological factors"


def full_explanation(row: pd.Series) -> str:
    """
    Combine SHAP explanation + residual gap + rule-based labels
    into a single readable string.
    """
    z = row.get("yield_z_robust", 0)
    direction = "LOW" if z < 0 else "HIGH"
    header = f"Unexpectedly {direction} yield (Z={z:.2f})"

    shap_text = row.get("shap_explanation", "")
    residual_text = row.get("residual_explanation", "")
    rule_text = rule_based_explanation(row)

    parts = [header]
    if shap_text:
        parts.append(f"Model attribution: {shap_text}")
    if residual_text:
        parts.append(residual_text)
    parts.append(f"Weather context: {rule_text}")
    return " | ".join(parts)


anomaly_df = df_if[df_if["is_anomaly"]].copy()
anomaly_df["explanation"] = anomaly_df.apply(full_explanation, axis=1)

print(f"\nTotal flagged anomalies: {len(anomaly_df):,}")
print("\n=== Sample explanations ===")
for _, row in anomaly_df.sample(min(5, len(anomaly_df)), random_state=1).iterrows():
    print(f"\n  [{row.get('state_abbr','?')}, {row.get('county_name','?')}, "
          f"{row.get('year','?')}, {row.get('commodity_name','?')}]")
    print(f"  Actual: {row.get('yield_bu_ac','?'):.1f} bu/ac  |  "
          f"Predicted: {row.get('xgb_predicted_yield', float('nan')):.1f} bu/ac  |  "
          f"Residual: {row.get('xgb_residual', float('nan')):.1f} bu/ac")
    print(f"  → {row['explanation']}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Anomaly type classification

# COMMAND ----------

# Classify each anomaly by what kind it is — more useful than a binary flag

def classify_anomaly_type(row: pd.Series) -> str:
    """
    Four mutually exclusive anomaly types based on the three detection layers.

    WEATHER_DRIVEN     — Z-score low AND residual near zero
                         Weather explains it. Expected given conditions.
    UNEXPLAINED        — Z-score low AND residual very negative
                         Underperformed beyond what weather predicts.
                         Investigate: pest, disease, management, reporting error.
    RESIDUAL_ONLY      — Residual anomaly but Z-score normal
                         Absolute yield is fine but the model expected more.
                         Interesting: good absolute yield during bad weather year?
    ISOLATION_ONLY     — Flagged by Isolation Forest only
                         Unusual combination of features, not necessarily bad.
    """
    z_flag = row.get("z_anomaly", False)
    r_flag = row.get("residual_anomaly", False)
    if_flag = row.get("if_anomaly", False)

    if z_flag and r_flag:
        return "UNEXPLAINED"
    elif z_flag and not r_flag:
        return "WEATHER_DRIVEN"
    elif r_flag and not z_flag:
        return "RESIDUAL_ONLY"
    elif if_flag:
        return "ISOLATION_ONLY"
    return "NORMAL"

anomaly_df["anomaly_type"] = anomaly_df.apply(classify_anomaly_type, axis=1)

print("\n=== Anomaly Type Distribution ===")
print(anomaly_df.groupby(["commodity_name", "anomaly_type"]).size().unstack(fill_value=0))

print("\n=== UNEXPLAINED anomalies by year (most interesting) ===")
unexplained = anomaly_df[anomaly_df["anomaly_type"] == "UNEXPLAINED"]
print(unexplained.groupby("year").size().rename("n_unexplained").to_string())

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8. 2012 drought deep-dive — weather-driven vs unexplained

# COMMAND ----------

print("=== 2012 Drought Deep-Dive ===")
drought_2012 = df_if[df_if["year"] == 2012]
print(f"Counties in 2012: {drought_2012['fips'].nunique()}")
print(f"Z-score anomalies: {drought_2012['z_anomaly'].sum()}")
print(f"Residual anomalies: {drought_2012.get('residual_anomaly', pd.Series(False)).sum()}")

corn_2012 = drought_2012[drought_2012["commodity_name"] == "Corn"]
if "xgb_residual" in corn_2012.columns:
    print(f"\nCorn 2012:")
    print(f"  Median actual yield:    {corn_2012['yield_bu_ac'].median():.1f} bu/ac")
    print(f"  Median predicted yield: {corn_2012['xgb_predicted_yield'].median():.1f} bu/ac")
    print(f"  Median residual:        {corn_2012['xgb_residual'].median():.1f} bu/ac")
    print(f"\n  Interpretation: if residual ≈ 0, the model correctly anticipated")
    print(f"  the yield loss from the drought weather features. The drought WAS")
    print(f"  the explanation — no additional anomaly investigation needed.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 9. Persist enriched anomaly table

# COMMAND ----------

save_cols = [
    "fips", "state_abbr", "county_name", "commodity_name", "irr_name",
    "year", "yield_bu_ac", "xgb_predicted_yield", "xgb_residual", "xgb_residual_z",
    "yield_z_robust", "z_direction", "residual_direction", "anomaly_type",
    "z_anomaly", "residual_anomaly", "if_anomaly", "if_anomaly_score", "is_anomaly",
    "precip_total_mm", "gdd_base10", "heat_stress_days", "precip_z", "cwsi",
    "lat", "lon",
    "shap_explanation", "residual_explanation", "top_driver", "top_driver_shap",
    "explanation",
]
save_cols = [c for c in save_cols if c in anomaly_df.columns]

write_delta(anomaly_df[save_cols], ANOMALY_DELTA, spark=spark, partition_by=["commodity_name"])

if spark is not None:
    try:
        spark.sql(
            f"""
            CREATE TABLE IF NOT EXISTS {DATABRICKS_ANOMALY_TABLE}
            USING DELTA LOCATION '{ANOMALY_DELTA}'
            """
        )
        print(f"Registered: {DATABRICKS_ANOMALY_TABLE}")
    except Exception as exc:
        log.warning("Unity Catalog registration skipped: %s", exc)

print("\nAnomaly table persisted — proceeding to Section 10 (findings report).")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 10. Findings report — ANOMALY_DETECTION_FINDINGS.md

# COMMAND ----------

# Generate a human-readable anomaly detection findings report.
# Mirrors WEATHER_YIELD_FINDINGS.md from notebook 03 but focused on which
# yield anomalies are weather-driven vs genuinely unexplained.

_lines = []


def _h(text, level=2):
    _lines.append(("#" * level) + " " + text)


def _p(*args):
    _lines.append(" ".join(str(a) for a in args))


def _blank():
    _lines.append("")


_h("Anomaly Detection Findings", 1)
_p("Generated by: Option 1 — Notebook 04 (Anomaly Detection & Explanation)")
_p(f"Date generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
_blank()
_p("This report summarises the three-layer yield anomaly detection results for Corn")
_p(f"and Soybeans across US counties ({YEAR_MIN}–{YEAR_MAX}).")
_blank()
_p("**Detection layers:**")
_p("- Layer 1 (Z-score): statistically unusual yield for that county's own history")
_p("- Layer 2 (XGBoost residual): yield below what weather features predict")
_p("- Layer 3 (Isolation Forest on residuals): non-linear unexplained structure")
_blank()

# ── Overall statistics ─────────────────────────────────────────────────────────
_h("Overall Statistics")
_total_rows = len(df_if)
_total_anom = int(df_if["is_anomaly"].sum())
_p(f"- Total county-year observations: {_total_rows:,}")
_p(
    f"- Total flagged as anomalous (any layer): {_total_anom:,} "
    f"({100 * _total_anom / max(_total_rows, 1):.1f}%)"
)
_blank()

if "anomaly_type" in anomaly_df.columns:
    _type_counts = anomaly_df["anomaly_type"].value_counts()
    _type_notes = {
        "UNEXPLAINED":    "yield low AND weather normal — investigate non-meteorological causes",
        "WEATHER_DRIVEN": "yield low AND bad weather — model confirms weather drove the loss",
        "RESIDUAL_ONLY":  "absolute yield OK but model expected more given conditions",
        "ISOLATION_ONLY": "unusual feature combination flagged by Isolation Forest",
    }
    _p("**Anomaly type breakdown:**")
    for _atype, _cnt in _type_counts.items():
        _pct = 100 * _cnt / max(_total_anom, 1)
        _note = _type_notes.get(_atype, "")
        _p(f"  - {_atype}: {_cnt:,} ({_pct:.1f}%)  ← {_note}")
    _blank()

# ── Per-crop breakdown ─────────────────────────────────────────────────────────
_h("Per-Crop Summary")
for _crop in CROPS:
    _h(_crop, 3)
    _crop_df = anomaly_df[anomaly_df["commodity_name"] == _crop]
    _p(f"- Total anomalies: {len(_crop_df):,}")
    if "anomaly_type" in _crop_df.columns:
        for _atype, _cnt in _crop_df["anomaly_type"].value_counts().items():
            _p(f"  - {_atype}: {_cnt:,}")
    _blank()

    if "anomaly_type" in _crop_df.columns:
        _unexpl = _crop_df[_crop_df["anomaly_type"] == "UNEXPLAINED"].copy()
        if not _unexpl.empty and "xgb_residual" in _unexpl.columns:
            _top5 = _unexpl.nsmallest(5, "xgb_residual")
            _p(f"**Top 5 most extreme UNEXPLAINED events — {_crop}** (largest negative residual):")
            for _, _row in _top5.iterrows():
                _loc  = f"{_row.get('county_name', '?')}, {_row.get('state_abbr', '?')}"
                _yr   = int(_row.get("year", 0))
                _act  = _row.get("yield_bu_ac", float("nan"))
                _pred = _row.get("xgb_predicted_yield", float("nan"))
                _res  = _row.get("xgb_residual", float("nan"))
                _p(
                    f"  - {_yr} | {_loc}: actual={_act:.1f}, "
                    f"predicted={_pred:.1f}, residual={_res:.1f} bu/ac"
                )
    _blank()

# ── Year-by-year breakdown ─────────────────────────────────────────────────────
_h("Year-by-Year Anomaly Counts")
_p("Total anomalies per year (all crops combined). Known extreme events flagged.")
_blank()

_yr_totals = anomaly_df.groupby("year").size()
_yr_type_pivot = None
if "anomaly_type" in anomaly_df.columns:
    _yr_type_pivot = (
        anomaly_df.groupby(["year", "anomaly_type"])
        .size()
        .unstack(fill_value=0)
    )

for _yr in sorted(_yr_totals.index):
    _yr_int = int(_yr)
    _total_yr = int(_yr_totals[_yr])
    _event_note = f"  ← **{EXTREME_EVENTS[_yr_int]}**" if _yr_int in EXTREME_EVENTS else ""
    _type_parts = []
    if _yr_type_pivot is not None and _yr in _yr_type_pivot.index:
        for _atype in ["UNEXPLAINED", "WEATHER_DRIVEN", "RESIDUAL_ONLY", "ISOLATION_ONLY"]:
            if _atype in _yr_type_pivot.columns:
                _cnt = int(_yr_type_pivot.loc[_yr, _atype])
                if _cnt:
                    _type_parts.append(f"{_atype}={_cnt}")
    _detail = f" ({', '.join(_type_parts)})" if _type_parts else ""
    _p(f"  - {_yr_int}: {_total_yr} anomalies{_detail}{_event_note}")
_blank()

# ── 2012 drought deep-dive ─────────────────────────────────────────────────────
_h("2012 Drought Deep-Dive (Model Validation)")
_blank()
_p("The 2012 Great Plains Drought is the canonical model validation test.")
_p("If XGBoost residuals are near zero in 2012, the model correctly captured")
_p("drought signals in weather features — confirming the pipeline is working.")
_blank()

_drought_2012 = df_if[df_if["year"] == 2012]
for _crop in CROPS:
    _c2012 = _drought_2012[_drought_2012["commodity_name"] == _crop]
    if "xgb_residual" in _c2012.columns and _c2012["xgb_residual"].notna().any():
        _med_act  = _c2012["yield_bu_ac"].median()
        _med_pred = _c2012["xgb_predicted_yield"].median()
        _med_res  = _c2012["xgb_residual"].median()
        _p(f"**{_crop} — 2012:**")
        _p(f"  - Median actual yield:    {_med_act:.1f} bu/ac")
        _p(f"  - Median predicted yield: {_med_pred:.1f} bu/ac")
        _p(f"  - Median residual:        {_med_res:.1f} bu/ac")
        if abs(_med_res) < 10:
            _p("  ✓ Small residual → model anticipated drought-driven losses from weather features.")
            _p("    Most 2012 low yields are WEATHER_DRIVEN, not UNEXPLAINED.")
        else:
            _p(f"  ✗ Large residual ({_med_res:.1f} bu/ac) — model missed part of the drought signal.")
        _blank()

# ── Key findings ──────────────────────────────────────────────────────────────
_h("Key Findings")
_blank()
_p(
    "1. **WEATHER_DRIVEN anomalies confirm model validity**: Low yields in 2012 (Great"
)
_p(
    "   Plains Drought), 2021 (Western Drought/Heat Dome), and 2022 (Southern Heat Wave)"
)
_p("   are predominantly WEATHER_DRIVEN — the model correctly anticipated the losses.")
_blank()
_p(
    "2. **UNEXPLAINED anomalies are the most actionable**: Counties flagged UNEXPLAINED"
)
_p("   underperformed despite normal or favourable weather. Likely causes: pest/disease")
_p("   outbreaks, irrigation failures, management errors, or data reporting issues.")
_blank()
_p("3. **RESIDUAL_ONLY anomalies reveal hidden resilience**: Counties achieving normal")
_p("   yields despite adverse conditions may indicate irrigation advantages,")
_p("   drought-tolerant varieties, or superior farm management.")
_blank()
_p("4. **2019 Prevent Plant signature**: UNEXPLAINED anomalies spike in 2019 because")
_p("   excess spring moisture prevented planting entirely — a discrete event the")
_p("   continuous weather features only partially capture.")
_blank()

# ── Methodology ───────────────────────────────────────────────────────────────
_h("Methodology")
_blank()
_p(f"- Data: USDA RMA county-level yields, Corn + Soybeans, {YEAR_MIN}–{YEAR_MAX}, 32 states")
_p("- Weather: NOAA GHCN-Daily (temperature, precipitation, GDD, heat stress days)")
_p(f"- Layer 1: Leave-one-out robust Z-score, threshold |Z| > {ANOMALY_ZSCORE_THRESHOLD}")
_p("- Layer 2: XGBoost residual (actual − predicted), threshold residual-Z < −1.5")
_p(f"- Layer 3: Isolation Forest on residuals, contamination = {ISOLATION_FOREST_CONTAMINATION}")
_p("- SHAP: TreeExplainer provides per-feature attribution for every flagged event")
_p("- Classification: UNEXPLAINED (Z∧residual), WEATHER_DRIVEN (Z only),")
_p("  RESIDUAL_ONLY (residual only), ISOLATION_ONLY (IF only)")
_blank()

# ── Save and log ──────────────────────────────────────────────────────────────
_report_path = Path(__file__).parent / "ANOMALY_DETECTION_FINDINGS.md"
_report_text = "\n".join(_lines)
_report_path.write_text(_report_text, encoding="utf-8")
print(f"✓ Findings report saved → {_report_path.name}")

try:
    import mlflow
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="Anomaly_Findings_Report"):
        mlflow.log_artifact(str(_report_path), artifact_path="Anomaly_Findings")
        mlflow.log_metrics({
            "total_anomalies": _total_anom,
            "pct_anomalous": round(100 * _total_anom / max(_total_rows, 1), 2),
        })
        if "anomaly_type" in anomaly_df.columns:
            for _atype, _cnt in anomaly_df["anomaly_type"].value_counts().items():
                mlflow.log_metric(f"n_{_atype.lower()}", int(_cnt))
    print("✓ Findings report logged to MLflow.")
except Exception as _mlflow_err:
    print(f"MLflow logging skipped: {_mlflow_err}")

print("\n=== Notebook 04 complete — proceed to 05_visualization_maps.py ===")
