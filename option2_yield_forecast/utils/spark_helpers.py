"""Databricks/Spark helpers (identical interface to option1 version)."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def get_spark(app_name: str = "YieldForecast"):
    try:
        import IPython
        shell = IPython.get_ipython()
        if shell is not None and "spark" in shell.user_ns:
            return shell.user_ns["spark"]
    except ImportError:
        pass
    try:
        from pyspark.sql import SparkSession
        spark = (
            SparkSession.builder.appName(app_name)
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog",
                    "org.apache.spark.sql.delta.catalog.DeltaCatalog")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("WARN")
        return spark
    except ImportError:
        logger.info("PySpark not available — pandas-only mode.")
        return None


def write_delta(df: pd.DataFrame, path: str, spark=None, mode: str = "overwrite",
                partition_by: list[str] | None = None):
    if spark is not None:
        sdf = spark.createDataFrame(df)
        writer = sdf.write.format("delta").mode(mode)
        if partition_by:
            writer = writer.partitionBy(*partition_by)
        writer.save(path)
    else:
        fallback = Path(path).with_suffix(".csv")
        df.to_csv(fallback, index=False)
        logger.info("CSV fallback: %s", fallback)


def read_delta(path: str, spark=None) -> pd.DataFrame:
    if spark is not None:
        try:
            return spark.read.format("delta").load(path).toPandas()
        except Exception:
            pass
    fallback = Path(path).with_suffix(".csv")
    if fallback.exists():
        return pd.read_csv(fallback)
    raise FileNotFoundError(f"No data at {path}")


def try_mlflow_log(metrics: dict[str, Any] = None, params: dict[str, Any] = None,
                   tags: dict[str, str] = None):
    try:
        import mlflow
        if metrics:
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        if params:
            mlflow.log_params({k: str(v) for k, v in params.items()})
        if tags:
            mlflow.set_tags(tags)
    except Exception as exc:
        logger.debug("MLflow skipped: %s", exc)
