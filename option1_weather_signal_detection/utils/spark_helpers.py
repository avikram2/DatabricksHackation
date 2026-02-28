"""
Databricks / Spark helper utilities.

All functions degrade gracefully when running outside Databricks:
  - get_spark()      returns a local SparkSession (or None if PySpark unavailable)
  - write_delta()    falls back to CSV/parquet if Delta not available
  - read_delta()     falls back to pandas read_csv/parquet
  - try_mlflow_log() is a no-op when MLflow tracking is not configured
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Spark session
# ---------------------------------------------------------------------------

def get_spark(app_name: str = "YieldWeatherAnalysis"):
    """
    Return a SparkSession.

    On Databricks the pre-created `spark` variable is returned via dbutils.
    Locally a minimal session is created (or None if PySpark is not installed).
    """
    # Check for Databricks environment
    try:
        # In Databricks notebooks `spark` is a built-in global
        import IPython
        shell = IPython.get_ipython()
        if shell is not None:
            user_ns = shell.user_ns
            if "spark" in user_ns:
                return user_ns["spark"]
    except ImportError:
        pass

    try:
        from pyspark.sql import SparkSession

        spark = (
            SparkSession.builder.appName(app_name)
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config(
                "spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog",
            )
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("WARN")
        return spark
    except ImportError:
        logger.info("PySpark not available — running in pandas-only mode.")
        return None


# ---------------------------------------------------------------------------
# Delta Lake I/O
# ---------------------------------------------------------------------------

def write_delta(
    df: pd.DataFrame,
    path: str,
    spark=None,
    mode: str = "overwrite",
    partition_by: list[str] | None = None,
):
    """
    Write a pandas DataFrame to Delta format via Spark, or CSV fallback.
    """
    if spark is not None:
        sdf = spark.createDataFrame(df)
        writer = sdf.write.format("delta").mode(mode)
        if partition_by:
            writer = writer.partitionBy(*partition_by)
        writer.save(path)
        logger.info("Written Delta table to %s", path)
    else:
        fallback = Path(path).with_suffix(".csv")
        df.to_csv(fallback, index=False)
        logger.info("Spark unavailable — written CSV to %s", fallback)


def read_delta(path: str, spark=None) -> pd.DataFrame:
    """
    Read a Delta table as a pandas DataFrame, or CSV fallback.
    """
    if spark is not None:
        try:
            return spark.read.format("delta").load(path).toPandas()
        except Exception as exc:
            logger.warning("Delta read failed (%s), trying CSV: %s", path, exc)

    fallback = Path(path).with_suffix(".csv")
    if fallback.exists():
        return pd.read_csv(fallback)
    raise FileNotFoundError(f"No Delta or CSV found at {path}")


# ---------------------------------------------------------------------------
# MLflow helpers
# ---------------------------------------------------------------------------

def try_mlflow_log(metrics: dict[str, Any] = None, params: dict[str, Any] = None, tags: dict[str, str] = None):
    """
    Log to MLflow if it is configured; silently skip otherwise.
    """
    try:
        import mlflow

        if metrics:
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        if params:
            mlflow.log_params({k: str(v) for k, v in params.items()})
        if tags:
            mlflow.set_tags(tags)
    except Exception as exc:
        logger.debug("MLflow logging skipped: %s", exc)


def register_model(model, model_name: str, run_id: str | None = None):
    """Register a model in the MLflow Model Registry."""
    try:
        import mlflow

        model_uri = f"runs:/{run_id}/model" if run_id else "models:/latest"
        mlflow.register_model(model_uri, model_name)
        logger.info("Registered model '%s'", model_name)
    except Exception as exc:
        logger.debug("MLflow model registration skipped: %s", exc)
