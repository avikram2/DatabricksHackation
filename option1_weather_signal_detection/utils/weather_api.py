"""
Open-Meteo Historical Weather API client.

Fetches growing-season (April-October) aggregated weather variables for
a given latitude/longitude and year range. No API key required.

Variables returned per year:
  - tmax_mean_c         Mean of daily Tmax (C) during growing season
  - tmin_mean_c         Mean of daily Tmin (C) during growing season
  - precip_total_mm     Total precipitation (mm) during growing season
  - gdd_base10          Growing Degree Days (base 10 C) accumulated
  - heat_stress_days    Days with Tmax > 35 C (95 F) during growing season
  - drought_days        Days with precip < 0.5 mm (proxy dry days)
  - et0_total_mm        Total reference evapotranspiration
  - solar_total_mj      Total surface solar radiation
  - wind_max_mean_ms    Mean of daily wind gusts
"""

import logging
import time
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
GROWING_SEASON_START = "04-01"
GROWING_SEASON_END = "10-31"

DAILY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "et0_fao_evapotranspiration",
    "shortwave_radiation_sum",
    "windgusts_10m_max",
]


def _compute_season_aggregates(daily: pd.DataFrame) -> dict[str, float]:
    """Compute annual growing-season aggregates from a daily DataFrame."""
    if daily.empty:
        return {}

    tavg = (daily["temperature_2m_max"] + daily["temperature_2m_min"]) / 2.0

    # GDD base 10 C (corn/soy standard)
    gdd = (tavg - 10.0).clip(lower=0).sum()

    return {
        "tmax_mean_c": daily["temperature_2m_max"].mean(),
        "tmin_mean_c": daily["temperature_2m_min"].mean(),
        "tavg_mean_c": tavg.mean(),
        "precip_total_mm": daily["precipitation_sum"].sum(),
        "gdd_base10": gdd,
        "heat_stress_days": (daily["temperature_2m_max"] > 35).sum(),
        "drought_days": (daily["precipitation_sum"] < 0.5).sum(),
        "et0_total_mm": daily.get("et0_fao_evapotranspiration", pd.Series([0])).sum(),
        "solar_total_mj": daily.get("shortwave_radiation_sum", pd.Series([0])).sum(),
        "wind_max_mean_ms": daily.get("windgusts_10m_max", pd.Series([0])).mean(),
    }


def fetch_growing_season_weather(
    lat: float,
    lon: float,
    year: int,
    retries: int = 3,
    backoff: float = 2.0,
) -> dict[str, Any] | None:
    """
    Fetch one year of growing-season weather for a county centroid.

    Returns a dict of aggregated weather features, or None on failure.
    """
    start = f"{year}-{GROWING_SEASON_START}"
    end = f"{year}-{GROWING_SEASON_END}"

    params = {
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "start_date": start,
        "end_date": end,
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "America/Chicago",
    }

    for attempt in range(retries):
        try:
            resp = requests.get(ARCHIVE_URL, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            daily_raw = data.get("daily", {})
            if not daily_raw or "time" not in daily_raw:
                return None
            daily_df = pd.DataFrame(daily_raw)
            daily_df["time"] = pd.to_datetime(daily_df["time"])
            agg = _compute_season_aggregates(daily_df)
            agg.update({"lat": lat, "lon": lon, "year": year})
            return agg
        except requests.exceptions.RequestException as exc:
            if attempt < retries - 1:
                wait = backoff * (2**attempt)
                logger.debug("Retry %d for (%s,%s,%d): %s", attempt + 1, lat, lon, year, exc)
                time.sleep(wait)
            else:
                logger.warning("Failed weather fetch (%s,%s,%d): %s", lat, lon, year, exc)
                return None


def fetch_weather_batch(
    coords_df: pd.DataFrame,
    years: list[int],
    delay_s: float = 0.15,
) -> pd.DataFrame:
    """
    Fetch weather for all (fips, year) combinations.

    coords_df must have columns: fips, lat, lon
    Returns a long DataFrame with one row per (fips, year).
    """
    records = []
    unique_locations = coords_df.drop_duplicates("fips")[["fips", "lat", "lon"]]
    total = len(unique_locations) * len(years)
    done = 0

    for _, loc in unique_locations.iterrows():
        if pd.isna(loc["lat"]) or pd.isna(loc["lon"]):
            continue
        for year in years:
            result = fetch_growing_season_weather(loc["lat"], loc["lon"], year)
            if result:
                result["fips"] = loc["fips"]
                records.append(result)
            done += 1
            if done % 50 == 0:
                pct = 100 * done / total
                logger.info("Weather fetch progress: %d/%d (%.1f%%)", done, total, pct)
            time.sleep(delay_s)

    return pd.DataFrame(records)
