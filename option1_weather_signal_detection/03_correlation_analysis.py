# Databricks notebook source
# Option 1 — Notebook 03: Weather-Yield Correlation Analysis
#
# 1. Merges yield + weather data → merged Delta table
# 2. Pearson / Spearman / MI correlations  → correlations Delta table   [A]
# 3. XGBoost + SHAP feature importance     → importance Delta table      [A]
# 4. MLflow: metrics + SHAP charts + CSV artifacts                       [B]

# COMMAND ----------

import logging
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless — works on both local and Databricks
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CORR_DELTA,
    CROPS,
    DATABRICKS_CORR_TABLE,
    DATABRICKS_IMPORTANCE_TABLE,
    DATABRICKS_MERGED_TABLE,
    IMPORTANCE_DELTA,
    MERGED_DELTA,
    MLFLOW_EXPERIMENT,
    TRAIN_YEARS,
    WEATHER_DELTA,
    YIELD_DELTA,
)
from utils import get_spark, read_delta, try_mlflow_log, write_delta

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Load and merge yield + weather

# COMMAND ----------

try:
    spark = get_spark()
except Exception:
    spark = None

yield_df   = read_delta(YIELD_DELTA,   spark=spark)
weather_df = read_delta(WEATHER_DELTA, spark=spark)

merged = yield_df.merge(weather_df, on=["fips", "year"], how="inner")
print(f"Merged rows : {len(merged):,}")
print(f"Crops       : {merged['commodity_name'].unique()}")

write_delta(merged, MERGED_DELTA, spark=spark, partition_by=["commodity_name"])

if spark is not None:
    try:
        spark.sql(
            f"CREATE TABLE IF NOT EXISTS {DATABRICKS_MERGED_TABLE} "
            f"USING DELTA LOCATION '{MERGED_DELTA}'"
        )
    except Exception:
        pass

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Correlation matrix (Pearson + Spearman)

# COMMAND ----------

WEATHER_FEATURES = [
    "tmax_mean_c", "tmin_mean_c", "tavg_mean_c",
    "precip_total_mm", "gdd_base10", "heat_stress_days",
    "drought_days", "et0_total_mm", "solar_total_mj",
    "precip_z", "cwsi",
]
# Drop columns absent or entirely NaN (e.g. et0/solar not in GHCN)
WEATHER_FEATURES = [
    c for c in WEATHER_FEATURES
    if c in merged.columns and merged[c].notna().any()
]
print(f"Active weather features: {WEATHER_FEATURES}")

corr_results = []
for crop in CROPS:
    subset = merged[merged["commodity_name"] == crop].dropna(subset=["yield_bu_ac"])
    for feat in WEATHER_FEATURES:
        vals = subset[["yield_bu_ac", feat]].dropna()
        if len(vals) < 20:
            continue
        r_p, p_p = stats.pearsonr(vals["yield_bu_ac"], vals[feat])
        r_s, p_s = stats.spearmanr(vals["yield_bu_ac"], vals[feat])
        corr_results.append({
            "crop":        crop,
            "feature":     feat,
            "pearson_r":   round(r_p, 3),
            "pearson_p":   round(p_p, 4),
            "spearman_r":  round(r_s, 3),
            "spearman_p":  round(p_s, 4),
            "abs_pearson": abs(r_p),
        })

corr_df = pd.DataFrame(corr_results)
if corr_df.empty:
    print("WARNING: No correlations computed.")
    print(f"  CROPS config   : {CROPS}")
    print(f"  Crops in data  : {merged['commodity_name'].unique().tolist()}")
    print(f"  Active features: {WEATHER_FEATURES}")
else:
    corr_df = corr_df.sort_values(["crop", "abs_pearson"], ascending=[True, False])
    print("\n=== Top correlations ===")
    print(corr_df.to_string(index=False))

# A: persist correlations to Delta
write_delta(corr_df, CORR_DELTA, spark=spark)
if spark is not None:
    try:
        spark.sql(
            f"CREATE TABLE IF NOT EXISTS {DATABRICKS_CORR_TABLE} "
            f"USING DELTA LOCATION '{CORR_DELTA}'"
        )
        print(f"Registered: {DATABRICKS_CORR_TABLE}")
    except Exception as exc:
        log.warning("Correlations table registration skipped: %s", exc)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. XGBoost feature importance + SHAP

# COMMAND ----------

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

importance_records = []

try:
    import joblib
    import mlflow
    import mlflow.xgboost
    import shap
    import xgboost as xgb
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import cross_val_score

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    for crop in CROPS:
        subset = (
            merged[
                (merged["commodity_name"] == crop)
                & (merged["year"].isin(TRAIN_YEARS))
            ]
            .dropna(subset=WEATHER_FEATURES + ["yield_bu_ac"])
        )
        X = subset[WEATHER_FEATURES].values
        y = subset["yield_bu_ac"].values

        model = xgb.XGBRegressor(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method="hist",
        )

        cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
        model.fit(X, y)

        # Save model for notebook 04 residual-based anomaly detection
        joblib.dump(model, MODEL_DIR / f"xgb_{crop.lower()}.pkl")
        log.info("Saved model: models/xgb_%s.pkl", crop.lower())

        rmse   = np.sqrt(mean_squared_error(y, model.predict(X)))
        r2     = r2_score(y, model.predict(X))

        # SHAP
        explainer     = shap.TreeExplainer(model)
        shap_vals     = explainer.shap_values(X)
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)

        for feat, imp, shap_imp in zip(
            WEATHER_FEATURES, model.feature_importances_, mean_abs_shap
        ):
            importance_records.append({
                "crop":           crop,
                "feature":        feat,
                "xgb_importance": round(float(imp), 4),
                "shap_mean_abs":  round(float(shap_imp), 4),
            })

        # ── B: MLflow run ────────────────────────────────────────────────────
        with mlflow.start_run(run_name=f"XGBoost_{crop}_Feature_Importance"):
            mlflow.log_params({"crop": crop, "n_features": len(WEATHER_FEATURES)})
            mlflow.log_metrics({
                "train_rmse":  round(rmse, 2),
                "train_r2":    round(r2, 3),
                "cv_r2_mean":  round(float(cv_scores.mean()), 3),
                "cv_r2_std":   round(float(cv_scores.std()),  3),
            })
            mlflow.xgboost.log_model(model, f"model_{crop.lower()}")

            # B: SHAP horizontal bar chart artifact
            feat_imp = pd.DataFrame({
                "feature":       WEATHER_FEATURES,
                "shap_mean_abs": mean_abs_shap,
            }).sort_values("shap_mean_abs")

            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ["#d73027" if v < 0 else "#4575b4"
                      for v in corr_df[corr_df["crop"] == crop]
                        .set_index("feature")
                        .reindex(feat_imp["feature"])["pearson_r"]
                        .fillna(0)]
            ax.barh(feat_imp["feature"], feat_imp["shap_mean_abs"],
                    color=colors, edgecolor="white")
            ax.axvline(feat_imp["shap_mean_abs"].mean(),
                       color="black", linestyle="--", linewidth=0.8,
                       label="mean importance")
            ax.set_xlabel("Mean |SHAP value| (bu/ac)")
            ax.set_title(f"{crop} — Weather Feature Importance (SHAP)\n"
                         f"blue = positive yield correlation, red = negative")
            ax.legend(fontsize=8)
            plt.tight_layout()
            mlflow.log_figure(fig, f"shap_importance_{crop.lower()}.png")
            plt.close(fig)

            # B: Pearson correlation bar chart artifact
            crop_corr = (
                corr_df[corr_df["crop"] == crop]
                .sort_values("pearson_r")
            )
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            bar_colors = ["#d73027" if v < 0 else "#4575b4"
                          for v in crop_corr["pearson_r"]]
            ax2.barh(crop_corr["feature"], crop_corr["pearson_r"],
                     color=bar_colors, edgecolor="white")
            ax2.axvline(0, color="black", linewidth=0.8)
            ax2.set_xlabel("Pearson r with yield (bu/ac)")
            ax2.set_title(f"{crop} — Weather-Yield Pearson Correlations")
            plt.tight_layout()
            mlflow.log_figure(fig2, f"pearson_corr_{crop.lower()}.png")
            plt.close(fig2)

            # B: CSV artifacts — full tables for this crop
            mlflow.log_text(
                corr_df[corr_df["crop"] == crop].to_csv(index=False),
                f"correlations_{crop.lower()}.csv",
            )
            mlflow.log_text(
                feat_imp.sort_values("shap_mean_abs", ascending=False)
                        .to_csv(index=False),
                f"importance_{crop.lower()}.csv",
            )

        print(
            f"\n{crop}: RMSE={rmse:.1f} bu/ac | "
            f"CV R²={cv_scores.mean():.3f} ± {cv_scores.std():.3f}"
        )

except ImportError as e:
    log.warning("XGBoost/SHAP not available: %s", e)
    importance_records = [
        {"crop": r["crop"], "feature": r["feature"],
         "xgb_importance": r["abs_pearson"], "shap_mean_abs": r["abs_pearson"]}
        for _, r in corr_df.iterrows()
    ]

importance_df = pd.DataFrame(importance_records).sort_values(
    ["crop", "shap_mean_abs"], ascending=[True, False]
)
print("\n=== Feature Importance (SHAP) ===")
print(importance_df.to_string(index=False))

# A: persist importance to Delta
write_delta(importance_df, IMPORTANCE_DELTA, spark=spark)
if spark is not None:
    try:
        spark.sql(
            f"CREATE TABLE IF NOT EXISTS {DATABRICKS_IMPORTANCE_TABLE} "
            f"USING DELTA LOCATION '{IMPORTANCE_DELTA}'"
        )
        print(f"Registered: {DATABRICKS_IMPORTANCE_TABLE}")
    except Exception as exc:
        log.warning("Importance table registration skipped: %s", exc)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Key insight summary

# COMMAND ----------

top_features = (
    importance_df.groupby("crop")
    .apply(lambda g: g.nlargest(3, "shap_mean_abs")["feature"].tolist())
)

print("\n=== Key Weather Drivers ===")
for crop, feats in top_features.items():
    print(f"  {crop}: {', '.join(feats)}")

print("\n=== Known Extreme Events Correlation Check ===")
if "yield_zscore" in merged.columns:
    drought_2012 = merged[merged["year"] == 2012].groupby("commodity_name")["yield_zscore"].mean().round(2)
    hot_2022     = merged[merged["year"] == 2022].groupby("commodity_name")["yield_zscore"].mean().round(2)
    print("2012 drought — mean yield Z-score:", drought_2012.to_dict())
    print("2022 heat   — mean yield Z-score:", hot_2022.to_dict())

print("\n=== Notebook 03 complete — proceed to 04_anomaly_detection.py ===")
