# Databricks notebook source
# Option 1 — Notebook 02: Fetch Historical Weather Data
#
# Weather source priority (automatic fallback):
#
#   1. DATABRICKS MARKETPLACE — NOAA GSOD (best for Databricks)
#      No API calls. Already a Delta table in the workspace.
#      Add from Marketplace: search "NOAA Global Surface Summary of Day"
#
#   2. NASA POWER API (best for local / when NOAA not available)
#      Free, no rate limits, no API key. ~2-5s per request but 4 parallel
#      workers. Specifically designed for agricultural meteorology.
#      URL: https://power.larc.nasa.gov/api/temporal/daily/point
#
#   3. Open-Meteo archive API (fallback)
#      Free, no key. Has a per-hour rate limit (~10k req/hour).
#      Full disk cache + resume on re-run — hitting the limit just means
#      re-running later to fetch the remainder.
#
# Set WEATHER_SOURCE below to force a specific source, or leave as "auto".

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
    YEAR_MAX,
    YEAR_MIN,
    YIELD_DELTA,
)
from utils import fetch_weather_batch, get_spark, read_delta, write_delta

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)
log = logger
# ── Source selection ──────────────────────────────────────────────────────────
# "auto"        : try NOAA GSOD → NASA POWER → Open-Meteo in that order
# "noaa_gsod"   : Databricks Marketplace NOAA data only
# "nasa_power"  : NASA POWER API only (good default when not on Databricks)
# "open_meteo"  : Open-Meteo archive API only
WEATHER_SOURCE = "auto"

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Load yield data — get (fips, lat, lon) pairs

# COMMAND ----------
try:
    spark = get_spark()
except:
    spark = None
    logger.info("No spark")
yield_df = read_delta(YIELD_DELTA, spark=spark)

coords = (
    yield_df.dropna(subset=["lat", "lon"])
    .drop_duplicates("fips")[["fips", "lat", "lon", "county_name", "state_abbr"]]
)
years = list(range(YEAR_MIN, YEAR_MAX + 1))

print(f"Counties: {len(coords)}")
print(f"Years:    {years[0]} – {years[-1]}")
print(f"Pairs:    {len(coords) * len(years):,}")
print(f"Source:   {WEATHER_SOURCE}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Source 1 — NOAA GSOD via Databricks Marketplace

# COMMAND ----------

weather_df = None

if WEATHER_SOURCE in ("auto", "noaa_gsod") and spark is not None:
    try:
        from utils.databricks_weather import build_weather_from_gsod

        log.info("Attempting NOAA GSOD from Databricks Marketplace …")
        weather_df = build_weather_from_gsod(spark, coords, years)

        if weather_df is not None and not weather_df.empty:
            log.info(
                "NOAA GSOD: loaded %d (county, year) records — no API calls needed.",
                len(weather_df),
            )
        else:
            log.warning(
                "NOAA GSOD returned no data. "
                "Add the dataset from Databricks Marketplace and re-run, "
                "or proceed with API fallback."
            )
            weather_df = None
    except Exception as exc:
        log.warning("NOAA GSOD failed: %s — falling back to API.", exc)
        weather_df = None

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Source 2 — NASA POWER API (no rate limits, no key)

# COMMAND ----------

if weather_df is None and WEATHER_SOURCE in ("auto", "nasa_power"):
    try:
        from utils.nasa_power_api import fetch_weather_batch_nasa

        nasa_cache = str(Path(WEATHER_CACHE).with_name("weather_nasa_cache.csv"))
        log.info(
            "Fetching from NASA POWER API "
            "(cache: %s — re-run resumes automatically) …", nasa_cache
        )
        weather_df = fetch_weather_batch_nasa(
            coords,
            years,
            delay_s=0.05,           # NASA recommends < 30 req/s; 4 workers × 0.05s ≈ 80 req/s peak → stay safe
            cache_path=nasa_cache,
        )

        if weather_df is not None and not weather_df.empty:
            log.info("NASA POWER: %d records fetched/loaded.", len(weather_df))
        else:
            log.warning("NASA POWER returned no data — falling back to Open-Meteo.")
            weather_df = None
    except Exception as exc:
        log.warning("NASA POWER failed: %s — falling back to Open-Meteo.", exc)
        weather_df = None

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Source 3 — Open-Meteo archive API (cached, resumable)

# COMMAND ----------

if weather_df is None and WEATHER_SOURCE in ("auto", "open_meteo"):
    log.info(
        "Fetching from Open-Meteo archive API "
        "(cache: %s — re-run resumes from where it left off) …",
        WEATHER_CACHE,
    )

    if spark is not None:
        # ── Databricks: distribute via Pandas UDF ───────────────────────────
        from pyspark.sql.types import (
            DoubleType, IntegerType, StringType, StructField, StructType,
        )
        from pyspark.sql.functions import pandas_udf, PandasUDFType
        import time

        WEATHER_SCHEMA = StructType([
            StructField("fips",             StringType()),
            StructField("year",             IntegerType()),
            StructField("tmax_mean_c",      DoubleType()),
            StructField("tmin_mean_c",      DoubleType()),
            StructField("tavg_mean_c",      DoubleType()),
            StructField("precip_total_mm",  DoubleType()),
            StructField("gdd_base10",       DoubleType()),
            StructField("heat_stress_days", DoubleType()),
            StructField("drought_days",     DoubleType()),
            StructField("et0_total_mm",     DoubleType()),
            StructField("solar_total_mj",   DoubleType()),
            StructField("wind_max_mean_ms", DoubleType()),
        ])

        combos = pd.DataFrame(
            [(r.fips, r.lat, r.lon, y) for _, r in coords.iterrows() for y in years],
            columns=["fips", "lat", "lon", "year"],
        )
        combos_sdf = spark.createDataFrame(combos)

        @pandas_udf(WEATHER_SCHEMA, PandasUDFType.GROUPED_MAP)
        def fetch_weather_udf(pdf: pd.DataFrame) -> pd.DataFrame:
            import sys, time
            sys.path.insert(0, "/dbfs/FileStore/hackathon")
            from utils.weather_api import fetch_growing_season_weather

            results = []
            for _, row in pdf.iterrows():
                rec = fetch_growing_season_weather(row["lat"], row["lon"], int(row["year"]))
                if rec:
                    rec["fips"] = row["fips"]
                    results.append(rec)
                time.sleep(0.15)
            return (
                pd.DataFrame(results)
                if results
                else pd.DataFrame(columns=WEATHER_SCHEMA.fieldNames())
            )

        weather_sdf = combos_sdf.groupby("fips", "year").apply(fetch_weather_udf)
        weather_df = weather_sdf.toPandas()

    else:
        # ── Local: parallel threaded fetch with disk cache ──────────────────
        weather_df = fetch_weather_batch(
            coords,
            years,
            delay_s=WEATHER_FETCH_DELAY_S,
            cache_path=WEATHER_CACHE,
            max_workers=6,
        )

if weather_df is None or weather_df.empty:
    raise RuntimeError(
        "All weather data sources failed.\n"
        "Options:\n"
        "  1. Add NOAA GSOD from Databricks Marketplace and set WEATHER_SOURCE='noaa_gsod'\n"
        "  2. Wait for Open-Meteo rate limit to reset (1 hour) and re-run — cache resumes\n"
        "  3. Set WEATHER_SOURCE='nasa_power' to use the NASA API instead"
    )

print(f"\nWeather source used:    {weather_df.get('source', pd.Series(['unknown'])).mode()[0] if 'source' in weather_df.columns else 'see logs'}")
print(f"Weather records loaded: {len(weather_df):,}")
print(weather_df.head(3))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Quality checks & imputation

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
# MAGIC ## 6. Derived indices

# COMMAND ----------

weather_df["precip_z"] = weather_df.groupby("fips")["precip_total_mm"].transform(
    lambda s: (s - s.mean()) / (s.std() + 1e-6)
)

if "et0_total_mm" in weather_df.columns:
    weather_df["cwsi"] = (
        weather_df["et0_total_mm"] / weather_df["precip_total_mm"].clip(lower=1)
    ).round(3)
else:
    # CWSI proxy when ET0 is unavailable (NASA POWER / NOAA GSOD path)
    # Use temperature-based Hargreaves ET0 estimate:
    #   ET0_hargreaves ≈ 0.0023 × (Tmax - Tmin)^0.5 × (Tavg + 17.8) × Ra
    # Ra (extraterrestrial radiation) approximated as constant 30 MJ/m²/day
    if "tmax_mean_c" in weather_df.columns and "tmin_mean_c" in weather_df.columns:
        weather_df["et0_estimate_mm"] = (
            0.0023
            * (weather_df["tmax_mean_c"] - weather_df["tmin_mean_c"]).clip(lower=0) ** 0.5
            * (weather_df["tavg_mean_c"] + 17.8)
            * 30 * 214      # Ra × ~214 growing season days
        )
        weather_df["cwsi"] = (
            weather_df["et0_estimate_mm"] / weather_df["precip_total_mm"].clip(lower=1)
        ).round(3)
    else:
        weather_df["cwsi"] = float("nan")

if "heat_stress_days" in weather_df.columns and "gdd_base10" in weather_df.columns:
    max_gdd = weather_df.groupby("fips")["gdd_base10"].transform("max").clip(lower=1)
    weather_df["heat_fraction"] = (weather_df["heat_stress_days"] / max_gdd).round(4)

print(weather_df.describe().round(2))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Persist to Delta Lake

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
