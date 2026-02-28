"""
Open-Meteo Forecast API client for 60/90-day yield simulation.

Strategy:
  - Open-Meteo free tier provides 16-day deterministic forecast.
  - For 60/90-day horizons we combine:
      1. 16-day deterministic forecast (high confidence near term)
      2. Historical climate normals ± perturbation for days 17-90
         (Monte Carlo weather scenarios)
  - Each scenario produces one yield estimate; the ensemble gives
    a probability distribution with uncertainty bands.
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "et0_fao_evapotranspiration",
    "shortwave_radiation_sum",
    "windgusts_10m_max",
]


# ---------------------------------------------------------------------------
# 16-day deterministic forecast
# ---------------------------------------------------------------------------

def fetch_16day_forecast(lat: float, lon: float) -> pd.DataFrame | None:
    """Fetch Open-Meteo 16-day daily forecast for a county centroid."""
    params = {
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "daily": ",".join(DAILY_VARS),
        "timezone": "America/Chicago",
        "forecast_days": 16,
    }
    try:
        resp = requests.get(FORECAST_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data["daily"])
        df["time"] = pd.to_datetime(df["time"])
        return df
    except Exception as exc:
        logger.warning("Forecast fetch failed (%s, %s): %s", lat, lon, exc)
        return None


# ---------------------------------------------------------------------------
# Historical climate normals (day-of-year statistics)
# ---------------------------------------------------------------------------

def fetch_historical_normals(
    lat: float,
    lon: float,
    reference_years: list[int] = None,
) -> dict[int, dict]:
    """
    Compute day-of-year climate normals from historical data.

    Returns: {doy: {"tmax_mean", "tmax_std", "tmin_mean", "tmin_std",
                     "precip_mean", "precip_std"}}
    """
    if reference_years is None:
        reference_years = list(range(2010, 2023))

    doy_records: dict[int, list] = {}

    for year in reference_years:
        start = f"{year}-04-01"
        end = f"{year}-10-31"
        params = {
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "start_date": start,
            "end_date": end,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
            "timezone": "America/Chicago",
        }
        try:
            resp = requests.get(ARCHIVE_URL, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            df = pd.DataFrame(data["daily"])
            df["time"] = pd.to_datetime(df["time"])
            df["doy"] = df["time"].dt.dayofyear
            for _, row in df.iterrows():
                doy = int(row["doy"])
                if doy not in doy_records:
                    doy_records[doy] = []
                doy_records[doy].append(
                    (row["temperature_2m_max"], row["temperature_2m_min"], row["precipitation_sum"])
                )
            time.sleep(0.1)
        except Exception as exc:
            logger.debug("Historical fetch %d failed: %s", year, exc)

    normals = {}
    for doy, vals in doy_records.items():
        arr = np.array(vals)
        normals[doy] = {
            "tmax_mean": float(np.nanmean(arr[:, 0])),
            "tmax_std": float(np.nanstd(arr[:, 0])),
            "tmin_mean": float(np.nanmean(arr[:, 1])),
            "tmin_std": float(np.nanstd(arr[:, 1])),
            "precip_mean": float(np.nanmean(arr[:, 2])),
            "precip_std": float(np.nanstd(arr[:, 2])),
        }
    return normals


# ---------------------------------------------------------------------------
# Monte Carlo scenario generator
# ---------------------------------------------------------------------------

def generate_weather_scenarios(
    lat: float,
    lon: float,
    target_year: int,
    horizon_days: int = 90,
    n_scenarios: int = 500,
    climate_shift_c: float = 0.0,
    precip_scale: float = 1.0,
    rng: np.random.Generator = None,
) -> pd.DataFrame:
    """
    Generate N synthetic growing-season weather scenarios.

    Steps:
    1. Fetch 16-day deterministic forecast (used as first ~16 days).
    2. Use climate normals for remaining days, sampling with noise.
    3. Aggregate each scenario to season-level features.
    4. Return DataFrame with shape (N, n_weather_features).

    climate_shift_c : temperature delta to apply (°C) — for scenario analysis
    precip_scale    : multiplicative factor on precipitation
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)

    # Get 16-day forecast
    forecast_df = fetch_16day_forecast(lat, lon)
    today = date.today()
    season_start = date(target_year, 4, 1)
    season_end = date(target_year, 10, 31)

    # Get climate normals for the full season
    normals = fetch_historical_normals(lat, lon)

    # Date range for the full growing season
    all_dates = pd.date_range(str(season_start), str(season_end), freq="D")

    scenarios = []

    for _ in range(n_scenarios):
        tmax_list, tmin_list, precip_list = [], [], []

        for dt in all_dates:
            doy = dt.dayofyear
            in_forecast = (
                forecast_df is not None
                and dt.date() >= today
                and dt.date() <= today + timedelta(days=15)
            )

            if in_forecast:
                row = forecast_df[forecast_df["time"].dt.date == dt.date()]
                if not row.empty:
                    tmax = float(row["temperature_2m_max"].iloc[0]) + climate_shift_c
                    tmin = float(row["temperature_2m_min"].iloc[0]) + climate_shift_c
                    prec = float(row["precipitation_sum"].iloc[0]) * precip_scale
                    tmax_list.append(tmax)
                    tmin_list.append(tmin)
                    precip_list.append(max(0, prec))
                    continue

            # Use climate normal + random noise
            norm = normals.get(doy)
            if norm is None:
                # Fall back to adjacent DOY
                norm = normals.get(doy - 1, {"tmax_mean": 25, "tmax_std": 3,
                                              "tmin_mean": 15, "tmin_std": 3,
                                              "precip_mean": 3, "precip_std": 5})

            tmax = rng.normal(norm["tmax_mean"] + climate_shift_c, max(norm["tmax_std"], 0.5))
            tmin = rng.normal(norm["tmin_mean"] + climate_shift_c, max(norm["tmin_std"], 0.5))
            prec_raw = rng.normal(norm["precip_mean"] * precip_scale, max(norm["precip_std"], 0.1))
            prec = max(0, prec_raw)

            tmax_list.append(tmax)
            tmin_list.append(tmin)
            precip_list.append(prec)

        tmax_arr = np.array(tmax_list)
        tmin_arr = np.array(tmin_list)
        prec_arr = np.array(precip_list)
        tavg_arr = (tmax_arr + tmin_arr) / 2

        gdd = np.maximum(0, tavg_arr - 10).sum()
        scenarios.append(
            {
                "tmax_mean_c": tmax_arr.mean(),
                "tmin_mean_c": tmin_arr.mean(),
                "tavg_mean_c": tavg_arr.mean(),
                "precip_total_mm": prec_arr.sum(),
                "gdd_base10": gdd,
                "heat_stress_days": (tmax_arr > 35).sum(),
                "drought_days": (prec_arr < 0.5).sum(),
                "climate_shift_c": climate_shift_c,
                "precip_scale": precip_scale,
            }
        )

    return pd.DataFrame(scenarios)


def fetch_forecast_scenarios(
    lat: float,
    lon: float,
    target_year: int,
    horizon_days: int = 90,
    n_scenarios: int = 500,
) -> dict[str, pd.DataFrame]:
    """
    Return a dict of DataFrames: baseline + stress scenarios.

    Keys: "baseline", "drought", "excess_rain", "heat_wave", "ideal"
    """
    rng = np.random.default_rng(seed=2024)
    return {
        "baseline": generate_weather_scenarios(lat, lon, target_year, horizon_days, n_scenarios, rng=rng),
        "drought": generate_weather_scenarios(lat, lon, target_year, horizon_days, n_scenarios,
                                               precip_scale=0.5, climate_shift_c=1.5, rng=rng),
        "excess_rain": generate_weather_scenarios(lat, lon, target_year, horizon_days, n_scenarios,
                                                   precip_scale=2.0, rng=rng),
        "heat_wave": generate_weather_scenarios(lat, lon, target_year, horizon_days, n_scenarios,
                                                 climate_shift_c=3.0, rng=rng),
        "ideal": generate_weather_scenarios(lat, lon, target_year, horizon_days, n_scenarios,
                                             climate_shift_c=-0.5, precip_scale=1.1, rng=rng),
    }
