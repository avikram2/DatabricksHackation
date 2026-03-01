# Databricks notebook source
# Option 1 — Notebook 01: Data Ingestion & Delta Lake Setup
#
# Reads the RMA County Yields CSV, cleans it, and writes it to a
# Delta Lake table (or local CSV when running outside Databricks).
#
# Run order: 01 → 02 → 03 → 04 → 05

# COMMAND ----------

import logging
import os
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

try:
    sys.path.insert(0, str(Path(__file__).parent))
except:
    pass

from config import (
    CENTROID_CACHE,
    CROPS,
    DATABRICKS_CATALOG,
    DATABRICKS_SCHEMA,
    DATABRICKS_YIELD_TABLE,
    YIELD_CSV,
    YIELD_DELTA,
)
from utils import build_fips_lookup, get_spark, write_delta

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Load & clean yield data

# COMMAND ----------

def load_yield_csv(path) -> pd.DataFrame:
    """Read and normalise the RMA County Yields CSV."""
    df = pd.read_csv(path)
    # Strip whitespace from all string columns
    str_cols = df.select_dtypes("object").columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())

    # Rename to snake_case for convenience
    df = df.rename(
        columns={
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
        }
    )

    # Type coercion
    df["year"] = df["year"].astype(int)
    df["yield_bu_ac"] = pd.to_numeric(df["yield_bu_ac"], errors="coerce")
    df["state_code"] = df["state_code"].astype(str).str.zfill(2)
    df["county_code"] = df["county_code"].astype(str).str.zfill(3)
    df["fips"] = df["state_code"] + df["county_code"]

    log.info("Loaded %d rows x %d columns", *df.shape)
    return df


# COMMAND ----------
# -- Try loading from Databricks DBFS first, then local path
try:
    # On Databricks: upload the CSV to DBFS and reference it here
    dbfs_path = "/dbfs/FileStore/hackathon/RMACountyYieldsReport-399.csv"
    if os.path.exists(dbfs_path):
        yield_df = load_yield_csv(dbfs_path)
    else:
        yield_df = load_yield_csv(YIELD_CSV)
except Exception as e:
    log.error("Could not load yield CSV: %s", e)
    raise

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Basic quality checks

# COMMAND ----------

print("=== Yield Data Overview ===")
print(f"Rows: {len(yield_df):,}")
print(f"Crops: {yield_df['commodity_name'].unique().tolist()}")
print(f"Years: {yield_df['year'].min()} – {yield_df['year'].max()}")
print(f"States: {yield_df['state_abbr'].nunique()}")
print(f"Counties: {yield_df['fips'].nunique()}")
print(f"Missing yield values: {yield_df['yield_bu_ac'].isna().sum()}")
print()
print(yield_df.groupby(["commodity_name", "irr_name"])["yield_bu_ac"].describe().round(1))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Attach county centroids (lat/lon)

# COMMAND ----------

yield_df = build_fips_lookup(yield_df)

# Keep only rows with valid coordinates (needed for weather fetch)
has_coords = yield_df["lat"].notna()
log.info(
    "Rows with lat/lon: %d / %d (%.1f%%)",
    has_coords.sum(),
    len(yield_df),
    100 * has_coords.mean(),
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Compute per-county historical statistics (baseline for anomaly detection)

# COMMAND ----------

stats = (
    yield_df.groupby(["fips", "commodity_name", "irr_name"])["yield_bu_ac"]
    .agg(["mean", "std", "min", "max", "count"])
    .reset_index()
    .rename(columns={"mean": "yield_mean", "std": "yield_std", "count": "n_years"})
)

# Z-score for each county-year observation
yield_df = yield_df.merge(stats, on=["fips", "commodity_name", "irr_name"], how="left")
yield_df["yield_zscore"] = (
    (yield_df["yield_bu_ac"] - yield_df["yield_mean"]) / yield_df["yield_std"]
).round(3)

print(yield_df[["fips", "county_name", "commodity_name", "year", "yield_bu_ac", "yield_zscore"]].head(10))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Persist to Delta Lake

# COMMAND ----------
try:
    spark = get_spark()
except:
    spark = None
    logger.info("No Spark")

# Write yield data
write_delta(yield_df, YIELD_DELTA, spark=spark, partition_by=["commodity_name", "year"])

# Also register as a Unity Catalog table when running on Databricks
if spark is not None:
    try:
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {DATABRICKS_CATALOG}.{DATABRICKS_SCHEMA}")
        spark.sql(
            f"""
            CREATE TABLE IF NOT EXISTS {DATABRICKS_YIELD_TABLE}
            USING DELTA
            LOCATION '{YIELD_DELTA}'
            """
        )
        print(f"Registered Unity Catalog table: {DATABRICKS_YIELD_TABLE}")
    except Exception as exc:
        log.warning("Unity Catalog registration skipped: %s", exc)

print("\n=== Notebook 01 complete — proceed to 02_fetch_weather.py ===")
