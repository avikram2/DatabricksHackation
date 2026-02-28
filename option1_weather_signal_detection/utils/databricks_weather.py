"""
Databricks Marketplace weather data loader.

Uses NOAA GSOD (Global Surface Summary of Day) which is available as a
free Delta Share on Databricks Marketplace — no API calls, no rate limits,
already in columnar Delta format.

How to connect on Databricks:
  1. Databricks UI → Marketplace → search "NOAA Global Surface Summary"
  2. Click "Get instant access" → creates a catalog (e.g. "noaa_gsod")
  3. Run this module — it reads directly from that catalog via Spark SQL

Alternatively, Databricks sample datasets include a subset:
  spark.read.table("samples.noaa.gsod")   ← available on all workspaces

NOAA GSOD schema (key columns):
  stn       — WMO station ID
  year      — 4-digit year
  mo        — month (01-12)
  da        — day (01-31)
  temp      — mean daily temperature (°F)
  max       — max temperature (°F)
  min       — min temperature (°F)
  prcp      — precipitation (inches)
  wdsp      — wind speed (knots)
  stp       — station pressure (mbar)
  lat, lon  — station coordinates

We map stations to counties using nearest-neighbour join on lat/lon,
then aggregate daily → growing-season annual values.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Databricks catalog options ────────────────────────────────────────────────
# Try these in order until one works on the current workspace.
GSOD_TABLE_CANDIDATES = [
    "samples.noaa.gsod",                    # built-in sample (all workspaces)
    "noaa_gsod.noaa.gsod_stations_data",    # Marketplace Delta Share name
    "noaa.gsod.gsod",                       # alternative share name
]

GROWING_MONTHS = list(range(4, 11))        # April (4) through October (10)
F_TO_C = lambda f: (f - 32) * 5 / 9       # Fahrenheit → Celsius
IN_TO_MM = 25.4                            # inches → mm


def _find_gsod_table(spark) -> str | None:
    """Return the first GSOD table name that exists on this workspace."""
    for table in GSOD_TABLE_CANDIDATES:
        try:
            spark.sql(f"SELECT 1 FROM {table} LIMIT 1")
            logger.info("Found NOAA GSOD table: %s", table)
            return table
        except Exception:
            continue
    return None


def load_gsod_growing_season(spark, years: list[int]) -> pd.DataFrame | None:
    """
    Load NOAA GSOD growing-season daily records from Databricks.

    Returns a DataFrame with columns:
        stn, year, lat, lon, tmax_c, tmin_c, prcp_mm, wdsp_ms
    One row per station-day during April-October for the requested years.
    """
    table = _find_gsod_table(spark)
    if table is None:
        logger.warning(
            "No NOAA GSOD table found. "
            "Add it from Databricks Marketplace or use NASA POWER instead."
        )
        return None

    years_str = ", ".join(str(y) for y in years)
    months_str = ", ".join(str(m) for m in GROWING_MONTHS)

    logger.info("Querying %s for years %s …", table, years_str)
    query = f"""
        SELECT
            stn,
            year,
            mo  AS month,
            da  AS day,
            lat,
            lon,
            -- Convert units: °F → °C, 99.9/9999.9 are NOAA missing-value sentinels
            CASE WHEN max  < 9000 THEN (max  - 32) * 5 / 9 ELSE NULL END AS tmax_c,
            CASE WHEN min  < 9000 THEN (min  - 32) * 5 / 9 ELSE NULL END AS tmin_c,
            CASE WHEN temp < 9000 THEN (temp - 32) * 5 / 9 ELSE NULL END AS tavg_c,
            -- Precipitation: 99.99 = missing; 0 = trace; actual in inches → mm
            CASE WHEN prcp < 99   THEN prcp * 25.4            ELSE 0    END AS prcp_mm,
            -- Wind speed: 999.9 = missing; knots → m/s
            CASE WHEN wdsp < 999  THEN wdsp * 0.514444        ELSE NULL END AS wdsp_ms
        FROM {table}
        WHERE year IN ({years_str})
          AND mo   IN ({months_str})
          AND lat IS NOT NULL
          AND lon IS NOT NULL
    """

    try:
        sdf = spark.sql(query)
        df = sdf.toPandas()
        logger.info("Loaded %d GSOD daily records.", len(df))
        return df
    except Exception as exc:
        logger.error("GSOD query failed: %s", exc)
        return None


def map_stations_to_counties(
    gsod_df: pd.DataFrame,
    coords_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign each NOAA weather station to the nearest county centroid.

    Uses a fast vectorised nearest-neighbour search (no scipy needed).
    coords_df must have columns: fips, lat, lon.

    Returns gsod_df with a 'fips' column added.
    """
    stations = gsod_df[["stn", "lat", "lon"]].drop_duplicates("stn")
    county_lats = coords_df["lat"].values
    county_lons = coords_df["lon"].values
    county_fips = coords_df["fips"].values

    fips_for_station = {}
    for _, srow in stations.iterrows():
        # Euclidean distance in degrees (good enough for nearest-neighbour)
        dlat = county_lats - srow["lat"]
        dlon = county_lons - srow["lon"]
        dist = dlat**2 + dlon**2
        nearest_idx = int(np.argmin(dist))
        fips_for_station[srow["stn"]] = county_fips[nearest_idx]

    gsod_df = gsod_df.copy()
    gsod_df["fips"] = gsod_df["stn"].map(fips_for_station)
    return gsod_df


def aggregate_to_season(gsod_county: pd.DataFrame) -> dict[str, float]:
    """
    Aggregate a county's daily GSOD rows to growing-season annual values.
    Same output dict structure as the Open-Meteo / NASA POWER versions.
    """
    if gsod_county.empty:
        return {}

    tmax = gsod_county["tmax_c"].dropna()
    tmin = gsod_county["tmin_c"].dropna()
    prcp = gsod_county["prcp_mm"].fillna(0)
    wind = gsod_county["wdsp_ms"].dropna()

    if len(tmax) == 0 or len(tmin) == 0:
        return {}

    tavg = (tmax + tmin) / 2
    gdd  = (tavg - 10.0).clip(lower=0).sum()

    return {
        "tmax_mean_c":      float(tmax.mean()),
        "tmin_mean_c":      float(tmin.mean()),
        "tavg_mean_c":      float(tavg.mean()),
        "precip_total_mm":  float(prcp.sum()),
        "gdd_base10":       float(gdd),
        "heat_stress_days": int((tmax > 35).sum()),
        "drought_days":     int((prcp < 0.5).sum()),
        "wind_mean_ms":     float(wind.mean()) if len(wind) else float("nan"),
        "et0_total_mm":     float("nan"),   # not in GSOD
        "solar_total_mj":   float("nan"),   # not in GSOD
    }


def build_weather_from_gsod(
    spark,
    coords_df: pd.DataFrame,
    years: list[int],
) -> pd.DataFrame | None:
    """
    Full pipeline: load GSOD → map stations to counties → aggregate to seasons.

    coords_df must have columns: fips, lat, lon.
    Returns a DataFrame with one row per (fips, year) — same schema as
    fetch_weather_batch() output so notebook 02 can use it interchangeably.
    """
    # 1. Load raw GSOD data
    gsod_df = load_gsod_growing_season(spark, years)
    if gsod_df is None or gsod_df.empty:
        return None

    # 2. Map stations → counties
    gsod_df = map_stations_to_counties(gsod_df, coords_df)

    # 3. Aggregate per (fips, year)
    records = []
    for (fips, year), group in gsod_df.groupby(["fips", "year"]):
        agg = aggregate_to_season(group)
        if agg:
            # Add lat/lon from county centroid
            loc = coords_df[coords_df["fips"] == fips]
            if not loc.empty:
                agg["lat"] = float(loc.iloc[0]["lat"])
                agg["lon"] = float(loc.iloc[0]["lon"])
            agg["fips"] = str(fips)
            agg["year"] = int(year)
            records.append(agg)

    if not records:
        logger.warning("No aggregated records produced from GSOD.")
        return None

    result = pd.DataFrame(records)
    logger.info(
        "GSOD pipeline complete: %d (county, year) records for %d counties.",
        len(result), result["fips"].nunique(),
    )
    return result
