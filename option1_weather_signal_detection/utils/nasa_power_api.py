"""
NASA POWER (Prediction of Worldwide Energy Resources) API client.

Designed specifically for agricultural meteorology. Key advantages over
Open-Meteo free tier:
  - No published rate limit (NASA recommends < 30 req/s — far more generous)
  - No API key required
  - Specifically curated for agro-met applications (community="AG")
  - Same variables available: Tmax, Tmin, precipitation, solar, humidity
  - Global coverage, 0.5° × 0.5° grid (~50 km resolution)

Endpoint: https://power.larc.nasa.gov/api/temporal/daily/point

NASA POWER variable names → our internal names:
  T2M_MAX     → tmax_mean_c       (2m air temp max, °C)
  T2M_MIN     → tmin_mean_c       (2m air temp min, °C)
  PRECTOTCORR → precip_total_mm   (bias-corrected precipitation, mm/day)
  ALLSKY_SFC_SW_DWN → solar_total_mj  (all-sky surface shortwave, MJ/m²/day)
  RH2M        → rh_mean_pct       (relative humidity at 2m, %)
  WS2M        → wind_mean_ms      (wind speed at 2m, m/s)
"""

from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# NASA POWER parameter names for the AG (agriculture) community
NASA_PARAMS = [
    "T2M_MAX",          # Daily max 2m temperature (°C)
    "T2M_MIN",          # Daily min 2m temperature (°C)
    "PRECTOTCORR",      # Bias-corrected precipitation (mm/day)
    "ALLSKY_SFC_SW_DWN",# Surface solar radiation (MJ/m²/day)
    "RH2M",             # Relative humidity at 2m (%)
    "WS2M",             # Wind speed at 2m (m/s)
]

GROWING_SEASON_START = "0401"   # MMDD format for NASA API date strings
GROWING_SEASON_END   = "1031"


def _compute_aggregates(daily: pd.DataFrame) -> dict[str, float]:
    """
    Compute growing-season agronomic aggregates from a NASA POWER daily DataFrame.
    Identical logic to the Open-Meteo version so downstream code is API-agnostic.
    """
    if daily.empty:
        return {}

    tmax = daily.get("T2M_MAX", pd.Series(dtype=float))
    tmin = daily.get("T2M_MIN", pd.Series(dtype=float))
    prec = daily.get("PRECTOTCORR", pd.Series(dtype=float)).clip(lower=0)
    solar = daily.get("ALLSKY_SFC_SW_DWN", pd.Series(dtype=float))

    tavg = (tmax + tmin) / 2.0
    gdd  = (tavg - 10.0).clip(lower=0).sum()

    return {
        "tmax_mean_c":      float(tmax.mean()),
        "tmin_mean_c":      float(tmin.mean()),
        "tavg_mean_c":      float(tavg.mean()),
        "precip_total_mm":  float(prec.sum()),
        "gdd_base10":       float(gdd),
        "heat_stress_days": int((tmax > 35).sum()),
        "drought_days":     int((prec < 0.5).sum()),
        "solar_total_mj":   float(solar.sum()),
        "rh_mean_pct":      float(daily.get("RH2M",  pd.Series([0])).mean()),
        "wind_mean_ms":     float(daily.get("WS2M",  pd.Series([0])).mean()),
        # et0 not directly available from POWER AG community — set to nan
        "et0_total_mm":     float("nan"),
    }


def fetch_growing_season_weather_nasa(
    lat: float,
    lon: float,
    year: int,
    retries: int = 3,
    backoff: float = 3.0,
) -> dict[str, Any] | None:
    """
    Fetch one year of growing-season weather from NASA POWER.

    Returns the same dict structure as the Open-Meteo version so callers
    can swap APIs without changing any downstream code.
    """
    start = f"{year}{GROWING_SEASON_START}"   # e.g. "20120401"
    end   = f"{year}{GROWING_SEASON_END}"     # e.g. "20121031"

    params = {
        "parameters": ",".join(NASA_PARAMS),
        "community":  "AG",
        "longitude":  round(lon, 4),
        "latitude":   round(lat, 4),
        "start":      start,
        "end":        end,
        "format":     "JSON",
    }

    for attempt in range(retries):
        try:
            resp = requests.get(NASA_POWER_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            # Navigate NASA POWER JSON structure
            props = data.get("properties", {}).get("parameter", {})
            if not props:
                logger.warning("Empty NASA POWER response for (%s,%s,%d)", lat, lon, year)
                return None

            # Build daily DataFrame — columns = parameter names, index = date strings
            daily = pd.DataFrame(props)           # shape: (n_days, n_params)
            daily.index = pd.to_datetime(daily.index, format="%Y%m%d")

            # NASA uses -999 as fill value for missing data
            daily = daily.replace(-999.0, float("nan"))

            agg = _compute_aggregates(daily)
            agg.update({"lat": lat, "lon": lon, "year": year})
            return agg

        except requests.exceptions.RequestException as exc:
            wait = backoff * (2 ** attempt)
            logger.warning(
                "NASA POWER fetch attempt %d failed (%s,%s,%d): %s — retrying in %.0fs",
                attempt + 1, lat, lon, year, exc, wait,
            )
            time.sleep(wait)

    logger.error("All retries exhausted for NASA POWER (%s,%s,%d)", lat, lon, year)
    return None


def fetch_weather_batch_nasa(
    coords_df: pd.DataFrame,
    years: list[int],
    delay_s: float = 0.1,
    cache_path: str | None = None,
) -> pd.DataFrame:
    """
    Fetch weather for all (fips, year) combinations using NASA POWER.

    Identical signature to fetch_weather_batch() in weather_api.py so
    notebook 02 can call either with a one-line switch.

    NASA POWER is slower per-request (~2–5s) but has no rate limiting,
    so we use a modest delay and no parallel threads (NASA recommends
    serial or low-concurrency access for the free API).
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Load cache
    cached_df = pd.DataFrame()
    done_keys: set[tuple] = set()
    all_records: list[dict] = []

    if cache_path:
        try:
            cached_df = pd.read_csv(cache_path, dtype={"fips": str})
            done_keys = set(zip(cached_df["fips"], cached_df["year"].astype(int)))
            all_records = cached_df.to_dict("records")
            logger.info("Loaded %d cached NASA POWER records.", len(cached_df))
        except FileNotFoundError:
            pass

    unique_locs = coords_df.drop_duplicates("fips")[["fips", "lat", "lon"]]
    todo = [
        (str(row["fips"]), float(row["lat"]), float(row["lon"]), int(year))
        for _, row in unique_locs.iterrows()
        for year in years
        if not pd.isna(row["lat"]) and (str(row["fips"]), int(year)) not in done_keys
    ]

    if not todo:
        logger.info("All records cached — no NASA POWER calls needed.")
        return cached_df if not cached_df.empty else pd.DataFrame(all_records)

    logger.info(
        "Fetching %d (fips, year) pairs from NASA POWER "
        "(%.0f already cached).", len(todo), len(done_keys)
    )

    lock = threading.Lock()
    completed = [0]

    def _fetch_one(fips, lat, lon, year):
        time.sleep(delay_s)
        result = fetch_growing_season_weather_nasa(lat, lon, year)
        if result:
            result["fips"] = fips
        return result

    # NASA recommends low concurrency — use 4 workers max
    max_workers = min(4, len(todo))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_one, *args): args
            for args in todo
        }
        for future in as_completed(futures):
            result = future.result()
            with lock:
                if result:
                    all_records.append(result)
                completed[0] += 1
                if completed[0] % 100 == 0:
                    pct = 100 * completed[0] / len(todo)
                    logger.info("NASA POWER fetch: %d/%d (%.1f%%)", completed[0], len(todo), pct)
                    if cache_path:
                        pd.DataFrame(all_records).to_csv(cache_path, index=False)

    if cache_path and all_records:
        pd.DataFrame(all_records).to_csv(cache_path, index=False)

    return pd.DataFrame(all_records)
