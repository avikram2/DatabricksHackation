"""
County centroid lookup utilities.

Downloads 2020 US Census county population centroids and builds a FIPS-keyed
lookup table compatible with the RMA yield dataset (State Code + County Code).

Problem: the RMA data has county names and FIPS codes but no coordinates. We need lat/lon to call the weather API.

Solution: Download the US Census Bureau's 2020 county population centroid file — 
a free 300 KB CSV with exact lat/lon for every US county. 
We build a fips key (state_fips + county_fips) 
that matches exactly what RMA uses (already zero-padded), 
then left-join it onto the yield dataframe. 
A hard-coded fallback covers the 12 biggest corn-belt counties in case the
Census URL is unreachable in the demo environment.
"""

import io
import logging
import time

import pandas as pd
import requests

logger = logging.getLogger(__name__)

CENSUS_CENTROID_URL = (
    "https://www2.census.gov/geo/docs/reference/cenpop2020/county/CenPop2020_Mean_CO.txt"
)

# Fallback: hard-coded centroids for the top corn/soy producing counties
# (used when Census URL is unavailable)
FALLBACK_CENTROIDS = {
    # Iowa
    "19013": (42.08, -93.54),  # Boone
    "19153": (41.65, -93.60),  # Polk
    "19085": (42.47, -92.32),  # Howard
    # Illinois
    "17001": (40.16, -88.52),  # Adams
    "17019": (40.56, -88.87),  # Champaign
    # Indiana
    "18039": (40.20, -86.85),  # Fountain
    # Nebraska
    "31055": (41.55, -97.59),  # Dodge
    # Minnesota
    "27049": (44.30, -93.96),  # Faribault
    # Ohio
    "39047": (40.70, -83.59),  # Fayette
    # Kansas
    "20035": (38.39, -95.70),  # Cherokee
}


def get_county_centroids(cache_path: str | None = None) -> pd.DataFrame:
    """
    Return a DataFrame with columns: fips, lat, lon, county_name, state_name.

    Tries Census URL first; falls back to a small hard-coded table.
    Optionally caches the result locally.
    """
    if cache_path:
        try:
            df = pd.read_csv(cache_path)
            if {"fips", "lat", "lon"}.issubset(df.columns):
                logger.info("Loaded county centroids from cache: %s", cache_path)
                return df
        except FileNotFoundError:
            pass

    df = _download_census_centroids()
    if df is not None and cache_path:
        df.to_csv(cache_path, index=False)
    if df is None:
        df = _fallback_centroids()
    return df


def _download_census_centroids() -> pd.DataFrame | None:
    """Download and parse the Census county centroid file."""
    try:
        logger.info("Downloading county centroids from Census Bureau …")
        resp = requests.get(CENSUS_CENTROID_URL, timeout=30)
        resp.raise_for_status()
        # Census CSVs are sometimes saved with a UTF-8 BOM (\xef\xbb\xbf).
        # Reading from bytes with encoding="utf-8-sig" strips it automatically,
        # preventing the first column name from arriving as "Ï»¿STATEFP".
        raw = pd.read_csv(io.BytesIO(resp.content), encoding="utf-8-sig")
        raw.columns = raw.columns.str.strip().str.upper()
        # Columns: STATEFP, COUNTYFP, COUNAME, STNAME, POPULATION, LATITUDE, LONGITUDE
        logger.debug("Census centroid columns: %s", list(raw.columns))
        df = pd.DataFrame(
            {
                "fips": (
                    raw["STATEFP"].astype(str).str.zfill(2)
                    + raw["COUNTYFP"].astype(str).str.zfill(3)
                ),
                "state_fips": raw["STATEFP"].astype(str).str.zfill(2),
                "county_fips": raw["COUNTYFP"].astype(str).str.zfill(3),
                "lat": raw["LATITUDE"].astype(float),
                "lon": raw["LONGITUDE"].astype(float),
                "county_name": raw["COUNAME"].str.strip(),
                "state_name": raw["STNAME"].str.strip(),
            }
        )
        logger.info("Downloaded %d county centroids.", len(df))
        return df
    except Exception as exc:
        logger.warning("Could not download Census centroids: %s", exc)
        return None


def _fallback_centroids() -> pd.DataFrame:
    """Return the hard-coded fallback table."""
    rows = [
        {"fips": k, "lat": v[0], "lon": v[1], "county_name": "", "state_name": ""}
        for k, v in FALLBACK_CENTROIDS.items()
    ]
    df = pd.DataFrame(rows)
    df["state_fips"] = df["fips"].str[:2]
    df["county_fips"] = df["fips"].str[2:]
    return df


def build_fips_lookup(yield_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join yield_df with centroid data on the 'fips' column.

    yield_df must already have a 'fips' column (5-digit zero-padded string).
    load_yield_csv() in notebook 01 builds it from state_code + county_code,
    so no reconstruction is needed here.
    Returns yield_df with lat/lon columns appended.
    """
    centroids = get_county_centroids()

    if "fips" not in yield_df.columns:
        raise KeyError(
            "yield_df has no 'fips' column. "
            "Call load_yield_csv() before build_fips_lookup()."
        )

    merged = yield_df.merge(
        centroids[["fips", "lat", "lon"]],
        on="fips",
        how="left",
    )
    missing = merged["lat"].isna().sum()
    if missing:
        logger.warning("%d rows missing lat/lon after centroid join.", missing)
    return merged
