"""
Feature engineering for the 60/90-day yield forecast model.

Produces a feature matrix with:
  - Historical yield lags (1, 2, 3 years)
  - Rolling-mean yields (3-year, 5-year)
  - County-level fixed effects (mean yield, trend coefficient)
  - Growing-season weather aggregates (from Open-Meteo)
  - Derived agro-meteorological indices (GDD, heat stress, SPI proxy)
  - Time features (year trend, decade encoding)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core agronomic calculations
# ---------------------------------------------------------------------------

def compute_gdd(tmax_c: pd.Series, tmin_c: pd.Series, base_c: float = 10.0) -> pd.Series:
    """
    Growing Degree Days (GDD) accumulated over the growing season.

    Standard formula (daily): GDD = max(0, (Tmax + Tmin)/2 - base)
    Capped: Tmax clipped to 30°C for corn (avoid negative GDD at extreme heat).
    """
    tmax_capped = tmax_c.clip(upper=30.0)
    tavg = (tmax_capped + tmin_c) / 2.0
    return (tavg - base_c).clip(lower=0)


def compute_stress_days(tmax_c: pd.Series, threshold_c: float = 35.0) -> pd.Series:
    """Count days where Tmax exceeds the heat-stress threshold."""
    return (tmax_c > threshold_c).astype(int)


def compute_spi_proxy(precip_series: pd.Series) -> pd.Series:
    """
    Standardised Precipitation Index proxy.

    Uses the county's own historical distribution as baseline.
    A simple Z-score of log-transformed precipitation.
    """
    log_p = np.log1p(precip_series)
    return (log_p - log_p.mean()) / (log_p.std() + 1e-6)


# ---------------------------------------------------------------------------
# Lag / rolling features
# ---------------------------------------------------------------------------

def add_lag_features(
    df: pd.DataFrame,
    value_col: str = "yield_bu_ac",
    group_cols: list[str] = None,
    lags: list[int] = None,
) -> pd.DataFrame:
    """
    Add lagged yield columns within each county-crop group.

    E.g. lag_1 = previous year's yield, lag_2 = two years prior.
    """
    if group_cols is None:
        group_cols = ["fips", "commodity_name"]
    if lags is None:
        lags = [1, 2, 3]

    df = df.sort_values(group_cols + ["year"])
    for lag in lags:
        col_name = f"yield_lag_{lag}"
        df[col_name] = df.groupby(group_cols)[value_col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    value_col: str = "yield_bu_ac",
    group_cols: list[str] = None,
    windows: list[int] = None,
) -> pd.DataFrame:
    """
    Add rolling-mean yield columns (trailing windows, excluding current year).
    """
    if group_cols is None:
        group_cols = ["fips", "commodity_name"]
    if windows is None:
        windows = [3, 5]

    df = df.sort_values(group_cols + ["year"])
    for w in windows:
        col_name = f"yield_roll{w}y"
        df[col_name] = (
            df.groupby(group_cols)[value_col]
            .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        )
    return df


# ---------------------------------------------------------------------------
# County fixed effects
# ---------------------------------------------------------------------------

def add_county_effects(
    df: pd.DataFrame,
    value_col: str = "yield_bu_ac",
    group_cols: list[str] = None,
) -> pd.DataFrame:
    """
    Add county-level baseline features:
      - county_mean_yield : long-run average
      - county_yield_trend: slope of OLS trend over available years
      - county_yield_cv   : coefficient of variation (volatility)
    """
    if group_cols is None:
        group_cols = ["fips", "commodity_name"]

    def county_trend(series: pd.Series) -> float:
        years = np.arange(len(series))
        if len(series) < 3:
            return 0.0
        slope, *_ = stats.linregress(years, series.values)
        return float(slope)

    grouped = df.groupby(group_cols)[value_col]
    df["county_mean_yield"] = grouped.transform("mean")
    df["county_yield_cv"] = grouped.transform(lambda s: s.std() / (s.mean() + 1e-6))
    df["county_yield_trend"] = grouped.transform(county_trend)
    return df


# ---------------------------------------------------------------------------
# Full feature matrix builder
# ---------------------------------------------------------------------------

def build_feature_matrix(
    yield_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    lag_years: list[int] = None,
    rolling_windows: list[int] = None,
) -> pd.DataFrame:
    """
    Merge yield + weather and build the full supervised feature matrix.

    Parameters
    ----------
    yield_df : cleaned yield DataFrame (output of notebook 01)
    weather_df : annual growing-season weather aggregates per (fips, year)

    Returns
    -------
    feature_df : one row per (fips, commodity_name, year), all features, target=yield_bu_ac
    """
    if lag_years is None:
        lag_years = [1, 2, 3]
    if rolling_windows is None:
        rolling_windows = [3, 5]

    df = yield_df.merge(weather_df, on=["fips", "year"], how="inner")

    # Derived weather features
    if "tmax_mean_c" in df.columns and "tmin_mean_c" in df.columns:
        # Note: these are season *averages*, so we approximate daily accumulation
        # by multiplying by ~214 growing-season days and averaging
        df["gdd_base10_approx"] = df.get("gdd_base10", compute_gdd(df["tmax_mean_c"], df["tmin_mean_c"]) * 214)

    if "precip_total_mm" in df.columns:
        df["spi_proxy"] = df.groupby(["fips", "commodity_name"])["precip_total_mm"].transform(compute_spi_proxy)

    # Lag + rolling yield features
    df = add_lag_features(df, lags=lag_years)
    df = add_rolling_features(df, windows=rolling_windows)
    df = add_county_effects(df)

    # Time trend
    df["year_trend"] = df["year"] - df["year"].min()
    df["decade"] = ((df["year"] - 2010) // 10).astype(int)

    # Irrigation flag
    if "irr_code" in df.columns:
        df["is_irrigated"] = (df["irr_code"].astype(str).str.strip() != "003").astype(int)

    logger.info(
        "Feature matrix: %d rows, %d columns. Target: yield_bu_ac",
        len(df), df.shape[1],
    )
    return df


# ---------------------------------------------------------------------------
# Feature column selector
# ---------------------------------------------------------------------------

WEATHER_FEATURE_COLS = [
    "tmax_mean_c", "tmin_mean_c", "tavg_mean_c",
    "precip_total_mm", "gdd_base10", "heat_stress_days",
    "drought_days", "et0_total_mm", "solar_total_mj",
    "precip_z", "cwsi", "spi_proxy",
]

YIELD_FEATURE_COLS = [
    "yield_lag_1", "yield_lag_2", "yield_lag_3",
    "yield_roll3y", "yield_roll5y",
    "county_mean_yield", "county_yield_cv", "county_yield_trend",
]

TIME_FEATURE_COLS = ["year_trend", "decade"]

CATEGORICAL_FEATURE_COLS = ["is_irrigated"]


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return available feature columns for the model."""
    candidates = (
        WEATHER_FEATURE_COLS + YIELD_FEATURE_COLS +
        TIME_FEATURE_COLS + CATEGORICAL_FEATURE_COLS
    )
    return [c for c in candidates if c in df.columns]
