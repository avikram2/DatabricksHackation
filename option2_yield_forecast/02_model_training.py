# Databricks notebook source
# Option 2 — Notebook 02: Model Training & MLflow Tracking
#
# Trains three model families per crop:
#   A. XGBoost (primary — best accuracy)
#   B. LightGBM (fast, similar accuracy)
#   C. Quantile Regression Forest (for native uncertainty intervals)
#
# All experiments logged to MLflow. Best model registered in Model Registry.
# Validation on 2023; test on 2024.

# COMMAND ----------

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CROPS,
    DATABRICKS_FEATURE_TABLE,
    FEATURE_DELTA,
    MLFLOW_EXPERIMENT,
    MODEL_DIR,
    QUANTILE_LEVELS,
    TEST_YEARS,
    TRAIN_YEARS,
    VAL_YEARS,
)
from utils import get_spark, read_delta, try_mlflow_log
from utils.feature_engineering import get_feature_cols

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Load features

# COMMAND ----------

spark = get_spark()
feature_df = read_delta(FEATURE_DELTA, spark=spark)
feat_cols = get_feature_cols(feature_df)
print(f"Features: {feat_cols}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Helpers — metrics + cross-validation

# COMMAND ----------

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate(y_true, y_pred, label="") -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100)
    print(f"  {label:20s} RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.3f}  MAPE={mape:.1f}%")
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. XGBoost + SHAP (primary model)

# COMMAND ----------

try:
    import mlflow
    import mlflow.xgboost
    import joblib
    import shap
    import xgboost as xgb
    from sklearn.model_selection import KFold, cross_val_score

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    best_models = {}

    for crop in CROPS:
        print(f"\n{'='*60}")
        print(f"Training XGBoost for: {crop}")

        df = feature_df[feature_df["commodity_name"] == crop].dropna(
            subset=feat_cols + ["yield_bu_ac"]
        )

        train = df[df["year"].isin(TRAIN_YEARS)]
        val = df[df["year"].isin(VAL_YEARS)]
        test = df[df["year"].isin(TEST_YEARS)]

        X_train, y_train = train[feat_cols].values, train["yield_bu_ac"].values
        X_val, y_val = val[feat_cols].values, val["yield_bu_ac"].values
        X_test, y_test = test[feat_cols].values, test["yield_bu_ac"].values

        xgb_params = dict(
            n_estimators=800,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=5,
            random_state=42,
            tree_method="hist",
            early_stopping_rounds=50,
            eval_metric="rmse",
        )

        model = xgb.XGBRegressor(**xgb_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        with mlflow.start_run(run_name=f"XGBoost_{crop}") as run:
            train_metrics = evaluate(y_train, model.predict(X_train), "Train")
            val_metrics = evaluate(y_val, model.predict(X_val), "Validation (2023)")
            if len(y_test) > 0:
                test_metrics = evaluate(y_test, model.predict(X_test), "Test (2024)")
            else:
                test_metrics = {}

            mlflow.log_params({**xgb_params, "crop": crop, "n_features": len(feat_cols)})
            mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
            mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
            mlflow.xgboost.log_model(model, "xgb_model")

            # SHAP feature importance
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_train)
            mean_shap = pd.Series(
                np.abs(shap_vals).mean(axis=0),
                index=feat_cols,
            ).sort_values(ascending=False)
            print(f"\n  Top features (SHAP):\n{mean_shap.head(8).to_string()}")

            run_id = run.info.run_id

        # Save locally too
        model_path = MODEL_DIR / f"xgb_{crop.lower()}.pkl"
        joblib.dump(model, model_path)
        best_models[crop] = {"model": model, "run_id": run_id, "feat_cols": feat_cols}

except ImportError as e:
    log.warning("XGBoost/MLflow not available: %s", e)
    best_models = {}

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Quantile Regression Forest (uncertainty intervals)

# COMMAND ----------

try:
    from quantile_forest import RandomForestQuantileRegressor

    qrf_models = {}

    for crop in CROPS:
        df = feature_df[feature_df["commodity_name"] == crop].dropna(
            subset=feat_cols + ["yield_bu_ac"]
        )
        train = df[df["year"].isin(TRAIN_YEARS)]
        val = df[df["year"].isin(VAL_YEARS)]

        X_train, y_train = train[feat_cols].values, train["yield_bu_ac"].values
        X_val, y_val = val[feat_cols].values, val["yield_bu_ac"].values

        qrf = RandomForestQuantileRegressor(
            n_estimators=300, max_depth=10, min_samples_leaf=5, random_state=42
        )
        qrf.fit(X_train, y_train)
        qrf_models[crop] = qrf

        # Prediction intervals on validation
        q10 = qrf.predict(X_val, quantiles=0.10)
        q50 = qrf.predict(X_val, quantiles=0.50)
        q90 = qrf.predict(X_val, quantiles=0.90)
        coverage = np.mean((y_val >= q10) & (y_val <= q90))
        interval_width = np.mean(q90 - q10)
        print(f"\n{crop} QRF — 80% interval coverage: {coverage:.2%}, mean width: {interval_width:.1f} bu/ac")

        qrf_path = MODEL_DIR / f"qrf_{crop.lower()}.pkl"
        import joblib
        joblib.dump(qrf, qrf_path)

except ImportError:
    log.warning("quantile-forest not installed. Using XGBoost quantile objectives instead.")
    qrf_models = {}

    # Fallback: train separate XGBoost models per quantile level
    try:
        import xgboost as xgb
        import joblib

        qrf_models_fallback = {}
        for crop in CROPS:
            df = feature_df[feature_df["commodity_name"] == crop].dropna(
                subset=feat_cols + ["yield_bu_ac"]
            )
            train = df[df["year"].isin(TRAIN_YEARS)]
            X_train, y_train = train[feat_cols].values, train["yield_bu_ac"].values

            crop_quantile_models = {}
            for q in QUANTILE_LEVELS:
                qmodel = xgb.XGBRegressor(
                    objective="reg:quantileerror",
                    quantile_alpha=q,
                    n_estimators=300,
                    max_depth=5,
                    learning_rate=0.05,
                    random_state=42,
                    tree_method="hist",
                )
                qmodel.fit(X_train, y_train)
                crop_quantile_models[q] = qmodel

            qrf_models[crop] = crop_quantile_models
            joblib.dump(crop_quantile_models, MODEL_DIR / f"quantile_models_{crop.lower()}.pkl")

    except Exception as e2:
        log.warning("Quantile model fallback also failed: %s", e2)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. LightGBM (fast baseline comparison)

# COMMAND ----------

try:
    import lightgbm as lgb
    import joblib

    lgb_models = {}

    for crop in CROPS:
        df = feature_df[feature_df["commodity_name"] == crop].dropna(
            subset=feat_cols + ["yield_bu_ac"]
        )
        train = df[df["year"].isin(TRAIN_YEARS)]
        val = df[df["year"].isin(VAL_YEARS)]

        X_train, y_train = train[feat_cols], train["yield_bu_ac"]
        X_val, y_val = val[feat_cols], val["yield_bu_ac"]

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.03,
            "num_leaves": 63,
            "min_child_samples": 10,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
        }
        lgb_model = lgb.train(
            params, lgb_train,
            num_boost_round=500,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
        )
        lgb_models[crop] = lgb_model
        val_pred = lgb_model.predict(X_val)
        evaluate(y_val, val_pred, f"LightGBM {crop} Val")

        joblib.dump(lgb_model, MODEL_DIR / f"lgb_{crop.lower()}.pkl")

except ImportError as e:
    log.warning("LightGBM not available: %s", e)
    lgb_models = {}

print("\n=== Notebook 02 complete — proceed to 03_forecast_simulation.py ===")
