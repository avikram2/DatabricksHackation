# Databricks notebook source
# Option 1 — Notebook 02: Fetch Historical Weather Data
#
# For each unique county (FIPS) + year in the yield dataset, fetches
# growing-season (Apr–Oct) weather from the Open-Meteo archive API.
# Results are stored in a Delta table for use by notebooks 03–05.
#
# NOTE: With ~2,300 unique county-years this takes ~10 min on a free API.
#       On Databricks you can parallelise with spark.udf or Pandas UDFs.

# COMMAND ----------

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATABRICKS_WEATHER_TABLE,
    WEATHER_CACHE,
    WEATHER_DELTA,
    WEATHER_FETCH_DELAY_S,
    YEAR_MIN,
    YEAR_MAX,
    YIELD_DELTA,
)
from utils import fetch_weather_batch, get_spark, read_delta, write_delta

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Load yield data to get the list of (fips, lat, lon) pairs

# COMMAND ----------
try:
    spark = get_spark()
except:
    spark = None
yield_df = read_delta(YIELD_DELTA, spark=spark)

# Unique counties with valid coordinates
coords = (
    yield_df.dropna(subset=["lat", "lon"])
    .drop_duplicates("fips")[["fips", "lat", "lon", "county_name", "state_abbr"]]
)
years = list(range(YEAR_MIN, YEAR_MAX + 1))
print(f"Counties to fetch: {len(coords)}")
print(f"Years: {years[0]} – {years[-1]}")
print(f"Total API calls: {len(coords) * len(years)}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Parallelise with Spark (Databricks) or serial fallback (local)

# COMMAND ----------

if spark is not None:
    # -----------------------------------------------------------------------
    # Databricks path: use a Pandas UDF to distribute weather fetching
    # -----------------------------------------------------------------------
    from pyspark.sql.functions import col, pandas_udf, PandasUDFType
    from pyspark.sql.types import (
        DoubleType,
        IntegerType,
        StringType,
        StructField,
        StructType,
    )
    import sys

    # Import inside UDF scope to avoid serialisation issues
    WEATHER_SCHEMA = StructType(
        [
            StructField("fips", StringType()),
            StructField("year", IntegerType()),
            StructField("tmax_mean_c", DoubleType()),
            StructField("tmin_mean_c", DoubleType()),
            StructField("tavg_mean_c", DoubleType()),
            StructField("precip_total_mm", DoubleType()),
            StructField("gdd_base10", DoubleType()),
            StructField("heat_stress_days", DoubleType()),
            StructField("drought_days", DoubleType()),
            StructField("et0_total_mm", DoubleType()),
            StructField("solar_total_mj", DoubleType()),
            StructField("wind_max_mean_ms", DoubleType()),
        ]
    )

    # Explode coords × years into a Spark DF for distributed processing
    import itertools
    combos = pd.DataFrame(
        [(r.fips, r.lat, r.lon, y) for _, r in coords.iterrows() for y in years],
        columns=["fips", "lat", "lon", "year"],
    )
    combos_sdf = spark.createDataFrame(combos)

    @pandas_udf(WEATHER_SCHEMA, PandasUDFType.GROUPED_MAP)
    def fetch_weather_udf(pdf: pd.DataFrame) -> pd.DataFrame:
        import sys, os
        sys.path.insert(0, "/dbfs/FileStore/hackathon")
        from utils.weather_api import fetch_growing_season_weather
        import time

        results = []
        for _, row in pdf.iterrows():
            rec = fetch_growing_season_weather(row["lat"], row["lon"], int(row["year"]))
            if rec:
                rec["fips"] = row["fips"]
                results.append(rec)
            time.sleep(0.1)
        return pd.DataFrame(results) if results else pd.DataFrame(columns=WEATHER_SCHEMA.fieldNames())

    weather_sdf = combos_sdf.groupby("fips", "year").apply(fetch_weather_udf)
    weather_df = weather_sdf.toPandas()

else:
    # -----------------------------------------------------------------------
    # Local path: parallel fetch with disk cache
    # 6 workers × 0.15 s delay → ~20 req/s (≈ 6-8× faster than serial).
    # Already-fetched (fips, year) pairs are skipped on re-runs.
    # -----------------------------------------------------------------------
    log.info("Running parallel weather fetch (local mode) …")
    weather_df = fetch_weather_batch(
        coords,
        years,
        delay_s=WEATHER_FETCH_DELAY_S,
        cache_path=WEATHER_CACHE,
    )

print(f"Weather records fetched: {len(weather_df)}")
print(weather_df.head())

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Quality checks & imputation

# COMMAND ----------

print("\nMissing values per weather column:")
print(weather_df.isna().sum())

# Forward-fill any sporadic missing values per county (linear interpolation)
weather_df = weather_df.sort_values(["fips", "year"])
weather_cols = [
    "tmax_mean_c", "tmin_mean_c", "tavg_mean_c",
    "precip_total_mm", "gdd_base10", "heat_stress_days",
    "drought_days", "et0_total_mm",
]
for col in weather_cols:
    if col in weather_df.columns:
        weather_df[col] = weather_df.groupby("fips")[col].transform(
            lambda s: s.interpolate(method="linear").ffill().bfill()
        )

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Derived drought / excess-moisture indices

# COMMAND ----------

# Standardised Precipitation Index (SPI) proxy per county
weather_df["precip_z"] = weather_df.groupby("fips")["precip_total_mm"].transform(
    lambda s: (s - s.mean()) / s.std()
)

# Crop Water Stress Index (simple proxy: ET0 / precipitation)
weather_df["cwsi"] = (weather_df["et0_total_mm"] / weather_df["precip_total_mm"].clip(lower=1)).round(3)

# Degree-days above 35 C relative to GDD (heat fraction)
if "heat_stress_days" in weather_df.columns and "gdd_base10" in weather_df.columns:
    max_gdd = weather_df.groupby("fips")["gdd_base10"].transform("max").clip(lower=1)
    weather_df["heat_fraction"] = (weather_df["heat_stress_days"] / max_gdd).round(4)

print(weather_df.describe().round(2))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Persist to Delta Lake

# COMMAND ----------

write_delta(weather_df, WEATHER_DELTA, spark=spark, partition_by=["year"])

if spark is not None:
    try:
        spark.sql(
            f"""
            CREATE TABLE IF NOT EXISTS {DATABRICKS_WEATHER_TABLE}
            USING DELTA
            LOCATION '{WEATHER_DELTA}'
            """
        )
        print(f"Registered: {DATABRICKS_WEATHER_TABLE}")
    except Exception as exc:
        log.warning("Unity Catalog registration skipped: %s", exc)

print("\n=== Notebook 02 complete — proceed to 03_correlation_analysis.py ===")
