# Databricks notebook source
# Option 2 — Notebook 01: Feature Engineering
#
# Builds the supervised feature matrix used for model training.
# Features come from three sources:
#   1. Yield history lags and rolling averages (from the RMA CSV)
#   2. County fixed effects (baseline yield, trend, volatility)
#   3. Growing-season weather aggregates (from Option 1 weather table,
#      or re-fetched here if not available)
#
# Output: Delta table of (fips, commodity, year, features..., yield_bu_ac)

# COMMAND ----------

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CENTROID_CACHE,
    CROPS,
    DATABRICKS_FEATURE_TABLE,
    FEATURE_DELTA,
    TRAIN_YEARS,
    YIELD_CSV,
    YIELD_DELTA,
)
from utils import get_spark, write_delta
from utils.feature_engineering import (
    build_feature_matrix,
    get_feature_cols,
)

WEATHER_FETCH_DELAY_S = 0.15  # polite rate limit for Open-Meteo free API

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Load yield data

# COMMAND ----------

# Re-use the ingestion logic from Option 1
def load_yield_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    str_cols = df.select_dtypes("object").columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())
    df = df.rename(columns={
        "Commodity Code": "commodity_code",
        "Commodity Name": "commodity_name",
        "State Code": "state_code",
        "State Name": "state_name",
        "State Abbreviation": "state_abbr",
        "County Code": "county_code",
        "County Name": "county_name",
        "Irrigation Practice Code": "irr_code",
        "Irrigation Practice Name": "irr_name",
        "Yield Year": "year",
        "Yield Amount": "yield_bu_ac",
    })
    df["year"] = df["year"].astype(int)
    df["yield_bu_ac"] = pd.to_numeric(df["yield_bu_ac"], errors="coerce")
    df["state_code"] = df["state_code"].astype(str).str.zfill(2)
    df["county_code"] = df["county_code"].astype(str).str.zfill(3)
    df["fips"] = df["state_code"] + df["county_code"]
    return df

import os
dbfs_path = "/dbfs/FileStore/hackathon/RMACountyYieldsReport-399.csv"
yield_df = load_yield_csv(dbfs_path if os.path.exists(dbfs_path) else YIELD_CSV)

# Attach county centroids for spatial joins
from utils.county_coords import get_county_centroids, build_fips_lookup
yield_df = build_fips_lookup(yield_df)

print(f"Yield rows: {len(yield_df):,}, Counties: {yield_df['fips'].nunique()}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Load or fetch weather data

# COMMAND ----------

# Try loading from Option 1's Delta table first; fall back to fetching
import importlib, sys as _sys
option1_weather_delta = str(Path(__file__).parent.parent / "option1_weather_signal_detection" / "delta" / "weather")
option1_weather_csv = Path(option1_weather_delta).with_suffix(".csv")

spark = get_spark()

if option1_weather_csv.exists():
    log.info("Re-using weather data from Option 1 pipeline.")
    weather_df = pd.read_csv(option1_weather_csv)
else:
    log.info("Fetching weather data fresh (this may take ~10 min) …")
    from utils.weather_api import fetch_weather_batch  # re-use option1 util
    coords = yield_df.dropna(subset=["lat", "lon"]).drop_duplicates("fips")[["fips", "lat", "lon"]]
    years = sorted(yield_df["year"].unique().tolist())
    weather_df = fetch_weather_batch(coords, years, delay_s=0.15)

print(f"Weather rows: {len(weather_df):,}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Build full feature matrix

# COMMAND ----------

feature_df = build_feature_matrix(yield_df, weather_df)

feat_cols = get_feature_cols(feature_df)
print(f"\nFeature columns ({len(feat_cols)}): {feat_cols}")
print(f"\nFeature matrix shape: {feature_df[feat_cols + ['yield_bu_ac']].shape}")
print(f"Missing values:\n{feature_df[feat_cols].isna().sum()[lambda s: s > 0]}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Train/val/test split preview

# COMMAND ----------

from config import TEST_YEARS, TRAIN_YEARS, VAL_YEARS

train_mask = feature_df["year"].isin(TRAIN_YEARS) & feature_df["yield_bu_ac"].notna()
val_mask = feature_df["year"].isin(VAL_YEARS) & feature_df["yield_bu_ac"].notna()
test_mask = feature_df["year"].isin(TEST_YEARS) & feature_df["yield_bu_ac"].notna()

print(f"Train rows: {train_mask.sum():,}")
print(f"Val rows:   {val_mask.sum():,}")
print(f"Test rows:  {test_mask.sum():,}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Persist feature table

# COMMAND ----------

write_delta(feature_df, FEATURE_DELTA, spark=spark, partition_by=["commodity_name", "year"])

if spark is not None:
    try:
        spark.sql(
            f"""
            CREATE TABLE IF NOT EXISTS {DATABRICKS_FEATURE_TABLE}
            USING DELTA LOCATION '{FEATURE_DELTA}'
            """
        )
        print(f"Registered: {DATABRICKS_FEATURE_TABLE}")
    except Exception as exc:
        log.warning("Unity Catalog skipped: %s", exc)

print("\n=== Notebook 01 complete — proceed to 02_model_training.py ===")
