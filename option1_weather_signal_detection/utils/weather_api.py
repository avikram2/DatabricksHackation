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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
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
    Handles HTTP 429 (rate limit) by reading the Retry-After header and
    sleeping before retrying — 429s do not consume a retry slot.
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

            # 429: respect the Retry-After header before the next attempt.
            # Note: this does advance the attempt counter, so 3 consecutive
            # 429 responses will exhaust retries and return None.
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 60))
                logger.warning(
                    "Rate limited (429) for (%s,%s,%d) — sleeping %ds",
                    lat, lon, year, retry_after,
                )
                time.sleep(retry_after)
                continue   # retry same attempt index

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


def _load_weather_cache(cache_path: str | None) -> pd.DataFrame:
    """Load previously fetched weather records from disk, or return empty DataFrame."""
    if not cache_path:
        return pd.DataFrame()
    try:
        df = pd.read_csv(cache_path, dtype={"fips": str})
        logger.info("Loaded %d cached weather records from %s", len(df), cache_path)
        return df
    except FileNotFoundError:
        return pd.DataFrame()


def _save_weather_cache(records: list[dict], cache_path: str | None) -> None:
    """Write the full list of records to the cache CSV."""
    if not cache_path or not records:
        return
    pd.DataFrame(records).to_csv(cache_path, index=False)


def fetch_weather_batch(
    coords_df: pd.DataFrame,
    years: list[int],
    delay_s: float = 0.3,
    cache_path: str | None = None,
    max_workers: int = 6,
) -> pd.DataFrame:
    """
    Fetch weather for all (fips, year) combinations, with disk cache and
    parallel I/O via ThreadPoolExecutor.

    On the first run every (fips, year) is fetched and written to cache_path.
    On subsequent runs, rows already present in the cache are skipped entirely —
    only new (fips, year) pairs hit the network.

    Throughput: max_workers=6 with delay_s=0.3 per thread → ~20 req/s, a
    respectful rate for the Open-Meteo free tier.

    coords_df must have columns: fips, lat, lon
    Returns a DataFrame with one row per (fips, year).
    """
    # --- 1. Load existing cache -------------------------------------------------
    cached_df = _load_weather_cache(cache_path)
    if not cached_df.empty:
        done_keys: set[tuple] = set(zip(cached_df["fips"], cached_df["year"].astype(int)))
    else:
        done_keys = set()

    all_records: list[dict] = [] if cached_df.empty else cached_df.to_dict("records")

    # --- 2. Build the todo list -------------------------------------------------
    unique_locs = coords_df.drop_duplicates("fips")[["fips", "lat", "lon"]]
    todo = [
        (str(row["fips"]), float(row["lat"]), float(row["lon"]), int(year))
        for _, row in unique_locs.iterrows()
        for year in years
        if not pd.isna(row["lat"]) and (str(row["fips"]), int(year)) not in done_keys
    ]

    if not todo:
        logger.info("All %d records already cached — skipping network fetch.", len(all_records))
        return cached_df

    logger.info(
        "Fetching %d (fips, year) pairs using %d workers "
        "(%.0f cached, %.0f skipped).",
        len(todo), max_workers, len(all_records), len(done_keys),
    )

    # --- 3. Parallel fetch with periodic cache saves ----------------------------
    lock = threading.Lock()
    completed = [0]

    def _fetch_one(fips: str, lat: float, lon: float, year: int) -> dict | None:
        time.sleep(delay_s)   # polite per-thread rate limiting
        result = fetch_growing_season_weather(lat, lon, year)
        if result:
            result["fips"] = fips
        return result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_one, fips, lat, lon, year): (fips, year)
            for fips, lat, lon, year in todo
        }
        for future in as_completed(futures):
            result = future.result()
            with lock:
                if result:
                    all_records.append(result)
                completed[0] += 1
                if completed[0] % 200 == 0:
                    pct = 100 * completed[0] / len(todo)
                    logger.info(
                        "Weather fetch: %d/%d (%.1f%%) — saving cache …",
                        completed[0], len(todo), pct,
                    )
                    _save_weather_cache(all_records, cache_path)

    # --- 4. Final cache save ----------------------------------------------------
    _save_weather_cache(all_records, cache_path)
    logger.info(
        "Fetch complete. Total records: %d (fetched %d new, skipped %d cached).",
        len(all_records), completed[0], len(done_keys),
    )
    return pd.DataFrame(all_records)
