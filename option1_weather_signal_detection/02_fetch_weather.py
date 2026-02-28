# Databricks notebook source
# Option 1 — Notebook 02: Fetch Historical Weather Data
#
# Weather source: NOAA GHCN-Daily local bulk files (no API, no rate limits).
#
# Download the 16 required files once with:
#   python option1_weather_signal_detection/download_ghcn.py
#
# Files are saved to: option1_weather_signal_detection/data/ghcn/

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
    YEAR_MAX,
    YEAR_MIN,
    YIELD_DELTA,
)
from utils import get_spark, read_delta, write_delta
from utils.local_weather_loader import build_weather_from_ghcn, check_ghcn_files

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)
log = logger

# Directory containing ghcnd-stations.txt and YYYY.csv.gz files
GHCN_DIR = Path(__file__).parent / "data" / "ghcn"

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Load yield data — get (fips, lat, lon) pairs

# COMMAND ----------
try:
    spark = get_spark()
except Exception:
    spark = None
    logger.info("No Spark session — running locally.")

yield_df = read_delta(YIELD_DELTA, spark=spark)

coords = (
    yield_df.dropna(subset=["lat", "lon"])
    .drop_duplicates("fips")[["fips", "lat", "lon", "county_name", "state_abbr"]]
)
years = list(range(YEAR_MIN, YEAR_MAX + 1))

print(f"Counties : {len(coords)}")
print(f"Years    : {years[0]} – {years[-1]}")
print(f"Pairs    : {len(coords) * len(years):,}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Fetch weather — NOAA GHCN-Daily local files

# COMMAND ----------

# Show which files are present / missing before starting
check_ghcn_files(years, GHCN_DIR)

ghcn_cache = str(Path(WEATHER_CACHE).with_name("weather_ghcn_cache.csv"))
log.info("Building weather from GHCN-Daily (dir: %s, cache: %s) …", GHCN_DIR, ghcn_cache)

weather_df = build_weather_from_ghcn(
    coords,
    years,
    ghcn_dir=GHCN_DIR,
    cache_path=ghcn_cache,
)

if weather_df is None or weather_df.empty:
    raise RuntimeError(
        "GHCN-Daily returned no data.\n"
        "Download the bulk files first:\n"
        "  python option1_weather_signal_detection/download_ghcn.py\n\n"
        "Or check which files are missing:\n"
        "  python option1_weather_signal_detection/download_ghcn.py --check"
    )

log.info("GHCN-Daily: %d (county, year) records loaded.", len(weather_df))
print(f"\nWeather records: {len(weather_df):,}")
print(weather_df.head(3))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Quality checks & imputation

# COMMAND ----------

print("\nMissing values per weather column:")
print(weather_df.isna().sum())

weather_df = weather_df.sort_values(["fips", "year"])
interp_cols = [
    "tmax_mean_c", "tmin_mean_c", "tavg_mean_c",
    "precip_total_mm", "gdd_base10", "heat_stress_days",
    "drought_days",
]
for col in interp_cols:
    if col in weather_df.columns:
        weather_df[col] = weather_df.groupby("fips")[col].transform(
            lambda s: s.interpolate(method="linear").ffill().bfill()
        )

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Derived indices

# COMMAND ----------

weather_df["precip_z"] = weather_df.groupby("fips")["precip_total_mm"].transform(
    lambda s: (s - s.mean()) / (s.std() + 1e-6)
)

# CWSI: Crop Water Stress Index — Hargreaves ET0 estimate since GHCN has no radiation
# ET0_hargreaves ≈ 0.0023 × (Tmax - Tmin)^0.5 × (Tavg + 17.8) × Ra
# Ra ≈ 30 MJ/m²/day mean; ~214 growing-season days
weather_df["et0_estimate_mm"] = (
    0.0023
    * (weather_df["tmax_mean_c"] - weather_df["tmin_mean_c"]).clip(lower=0) ** 0.5
    * (weather_df["tavg_mean_c"] + 17.8)
    * 30 * 214
)
weather_df["cwsi"] = (
    weather_df["et0_estimate_mm"] / weather_df["precip_total_mm"].clip(lower=1)
).round(3)

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
            USING DELTA LOCATION '{WEATHER_DELTA}'
            """
        )
        print(f"Registered: {DATABRICKS_WEATHER_TABLE}")
    except Exception as exc:
        log.warning("Unity Catalog registration skipped: %s", exc)

print("\n=== Notebook 02 complete — proceed to 03_correlation_analysis.py ===")
