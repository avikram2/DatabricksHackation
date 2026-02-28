"""
NOAA GHCN-Daily local file loader.

No API calls, no rate limits. Reads the bulk CSV files downloaded directly
from NOAA's servers and aggregates them to growing-season county-year records.

Download instructions
─────────────────────
1. Station metadata (one file, ~10 MB, fixed-width text):
     https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt

2. One yearly file per year needed (CSV.gz, ~90 MB each):
     https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_year/2010.csv.gz
     https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_year/2011.csv.gz
     ...
     https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_year/2024.csv.gz

Place all files in:
     option1_weather_signal_detection/data/ghcn/

GHCN-Daily yearly file format (CSV, no header):
    STATION, DATE, ELEMENT, DATA_VALUE, M_FLAG, Q_FLAG, S_FLAG, OBS_TIME

    STATION   — 11-char WMO/NOAA ID. US stations start with "US"
    DATE      — YYYYMMDD integer
    ELEMENT   — TMAX, TMIN, PRCP, SNOW, AWND, etc.
    DATA_VALUE— integer in tenths:
                  TMAX/TMIN in tenths of °C  (e.g. 256 = 25.6 °C)
                  PRCP in tenths of mm       (e.g. 15 = 1.5 mm)
    Q_FLAG    — quality flag; non-blank = suspect reading → drop

Station metadata file (fixed-width, no header):
    Cols  1-11  : Station ID
    Cols 13-20  : Latitude
    Cols 22-30  : Longitude
    Cols 32-37  : Elevation (m)
    Cols 39-40  : State (US only)
    Cols 42-71  : Station name
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GHCN_DIR_DEFAULT = Path(__file__).parent.parent / "data" / "ghcn"
STATIONS_FILE    = "ghcnd-stations.txt"
GROWING_MONTHS   = set(range(4, 11))   # April–October

# GHCN element codes we care about
ELEMENTS = {"TMAX", "TMIN", "PRCP", "AWND"}

# US state abbreviation → FIPS state code (for filtering to relevant states)
STATE_FIPS = {
    "AL": "01", "AR": "05", "CO": "08", "DE": "10", "IA": "19",
    "IL": "17", "IN": "18", "KS": "20", "KY": "21", "LA": "22",
    "MD": "24", "MI": "26", "MN": "27", "MO": "29", "MS": "28",
    "NC": "37", "ND": "38", "NE": "31", "NJ": "34", "NY": "36",
    "OH": "39", "OK": "40", "PA": "42", "SC": "45", "SD": "46",
    "TN": "47", "TX": "48", "VA": "51", "VT": "50", "WI": "55",
    "WV": "54", "WY": "56",
}
TARGET_STATES = set(STATE_FIPS.keys())


# ── Station metadata ──────────────────────────────────────────────────────────

def load_stations(ghcn_dir: Path = GHCN_DIR_DEFAULT) -> pd.DataFrame:
    """
    Parse ghcnd-stations.txt and return US stations as a DataFrame.

    Columns: station_id, lat, lon, elevation, state, name
    """
    path = ghcn_dir / STATIONS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Station file not found: {path}\n"
            f"Download from: https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt"
        )

    rows = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if len(line) < 40:
                continue
            station_id = line[0:11].strip()
            if not station_id.startswith("US"):
                continue                         # keep only US stations
            try:
                lat   = float(line[12:20])
                lon   = float(line[21:30])
                state = line[38:40].strip()
            except ValueError:
                continue
            if state not in TARGET_STATES:
                continue
            rows.append({
                "station_id": station_id,
                "lat":        lat,
                "lon":        lon,
                "state":      state,
                "name":       line[41:71].strip(),
            })

    df = pd.DataFrame(rows)
    logger.info("Loaded %d US stations in target states.", len(df))
    return df


# ── Nearest-neighbour station → county mapping ────────────────────────────────

def map_stations_to_counties(
    stations_df: pd.DataFrame,
    coords_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign each weather station to its nearest county centroid.

    coords_df must have columns: fips, lat, lon.
    Returns stations_df with a 'fips' column appended.
    """
    county_lats = coords_df["lat"].values
    county_lons = coords_df["lon"].values
    county_fips = coords_df["fips"].values

    fips_list = []
    for _, srow in stations_df.iterrows():
        dlat = county_lats - srow["lat"]
        dlon = county_lons - srow["lon"]
        nearest = int(np.argmin(dlat**2 + dlon**2))
        fips_list.append(county_fips[nearest])

    stations_df = stations_df.copy()
    stations_df["fips"] = fips_list
    return stations_df


# ── Yearly file loader ────────────────────────────────────────────────────────

def _load_year_file(year: int, ghcn_dir: Path, valid_stations: set[str]) -> pd.DataFrame:
    """
    Read one GHCN yearly CSV.gz and return growing-season records for valid stations.

    Drops rows with non-blank Q_FLAG (suspect quality).
    Converts DATA_VALUE from tenths to actual units.
    """
    gz_path  = ghcn_dir / f"{year}.csv.gz"
    csv_path = ghcn_dir / f"{year}.csv"

    if gz_path.exists():
        path = gz_path
    elif csv_path.exists():
        path = csv_path
    else:
        logger.warning(
            "GHCN file not found for %d: %s\n"
            "Download from: https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_year/%d.csv.gz",
            year, gz_path, year,
        )
        return pd.DataFrame()

    logger.info("Reading %s …", path.name)

    # Read in chunks to keep RAM usage low (files can be 400 MB+ uncompressed)
    COLS = ["station_id", "date", "element", "data_value", "m_flag", "q_flag", "s_flag", "obs_time"]
    chunks = []
    for chunk in pd.read_csv(
        path,
        names=COLS,
        dtype={"station_id": str, "date": str, "element": str,
               "data_value": float, "q_flag": str},
        chunksize=500_000,
        compression="infer",
    ):
        # Filter early to reduce memory
        mask = (
            chunk["station_id"].isin(valid_stations)
            & chunk["element"].isin(ELEMENTS)
            & chunk["q_flag"].isna()          # blank Q_FLAG = passed QC
        )
        sub = chunk.loc[mask, ["station_id", "date", "element", "data_value"]].copy()
        if sub.empty:
            continue

        sub["date"] = pd.to_datetime(sub["date"], format="%Y%m%d", errors="coerce")
        sub = sub.dropna(subset=["date", "data_value"])
        sub = sub[sub["date"].dt.month.isin(GROWING_MONTHS)]
        if sub.empty:
            continue

        # Convert tenths → actual units
        temp_mask = sub["element"].isin({"TMAX", "TMIN"})
        sub.loc[temp_mask, "data_value"] /= 10.0          # tenths °C → °C
        prcp_mask = sub["element"] == "PRCP"
        sub.loc[prcp_mask, "data_value"] /= 10.0          # tenths mm → mm
        wind_mask = sub["element"] == "AWND"
        sub.loc[wind_mask, "data_value"] /= 10.0          # tenths m/s → m/s

        # Drop implausible values
        sub = sub[sub["data_value"].between(-60, 60) | (sub["element"] == "PRCP")]
        sub = sub[sub["data_value"].between(0, 2000) | (sub["element"] != "PRCP")]

        chunks.append(sub)

    if not chunks:
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)
    df["year"] = year
    logger.info("  %d: %d growing-season records for %d stations.", year, len(df), df["station_id"].nunique())
    return df


# ── Season-level aggregation ──────────────────────────────────────────────────

def _aggregate_station_year(grp: pd.DataFrame) -> dict:
    """Aggregate one station's growing-season daily records to season totals."""
    result = {}

    tmax = grp.loc[grp["element"] == "TMAX", "data_value"].dropna()
    tmin = grp.loc[grp["element"] == "TMIN", "data_value"].dropna()
    prcp = grp.loc[grp["element"] == "PRCP", "data_value"].fillna(0)
    wind = grp.loc[grp["element"] == "AWND", "data_value"].dropna()

    if len(tmax) < 30 or len(tmin) < 30:    # need at least 30 days of temp data
        return {}

    tavg = (tmax.values + tmin.values[:len(tmax)]) / 2
    gdd  = np.maximum(0, tavg - 10).sum()

    result = {
        "tmax_mean_c":      round(float(tmax.mean()), 2),
        "tmin_mean_c":      round(float(tmin.mean()), 2),
        "tavg_mean_c":      round(float(tavg.mean()), 2),
        "precip_total_mm":  round(float(prcp.sum()), 1),
        "gdd_base10":       round(float(gdd), 1),
        "heat_stress_days": int((tmax > 35).sum()),
        "drought_days":     int((prcp < 0.5).sum()),
        "wind_mean_ms":     round(float(wind.mean()), 2) if len(wind) > 0 else float("nan"),
        "et0_total_mm":     float("nan"),   # not in GHCN-Daily
        "solar_total_mj":   float("nan"),   # not in GHCN-Daily
    }
    return result


def _aggregate_to_county_year(
    daily_df: pd.DataFrame,
    station_fips: dict[str, str],
) -> list[dict]:
    """
    Average multiple stations within the same county, then return one record
    per (fips, year).
    """
    daily_df = daily_df.copy()
    daily_df["fips"] = daily_df["station_id"].map(station_fips)
    daily_df = daily_df.dropna(subset=["fips"])

    records = []
    for (fips, year), county_grp in daily_df.groupby(["fips", "year"]):
        # Aggregate each station separately, then average across stations
        station_aggs = []
        for _, stn_grp in county_grp.groupby("station_id"):
            agg = _aggregate_station_year(stn_grp)
            if agg:
                station_aggs.append(agg)

        if not station_aggs:
            continue

        # Mean across stations for the county
        agg_df = pd.DataFrame(station_aggs)
        county_agg = agg_df.mean(numeric_only=True).to_dict()
        county_agg["fips"] = str(fips)
        county_agg["year"] = int(year)
        records.append(county_agg)

    return records


# ── Public entry point ────────────────────────────────────────────────────────

def build_weather_from_ghcn(
    coords_df: pd.DataFrame,
    years: list[int],
    ghcn_dir: Path | str = GHCN_DIR_DEFAULT,
    cache_path: str | None = None,
) -> pd.DataFrame:
    """
    Full pipeline: load GHCN files → map stations → aggregate → return DataFrame.

    coords_df : columns fips, lat, lon  (county centroids)
    years     : list of years to process
    ghcn_dir  : directory containing ghcnd-stations.txt and YYYY.csv.gz files
    cache_path: if provided, load from cache if it exists and save after processing

    Returns a DataFrame with one row per (fips, year), same schema as the
    API-based fetch functions so notebook 02 can use it interchangeably.
    """
    ghcn_dir = Path(ghcn_dir)

    # Check cache first
    if cache_path:
        cache = Path(cache_path)
        if cache.exists():
            cached = pd.read_csv(cache, dtype={"fips": str})
            cached_years = set(cached["year"].astype(int).unique())
            missing_years = [y for y in years if y not in cached_years]
            if not missing_years:
                logger.info("All years found in GHCN cache (%s).", cache_path)
                return cached
            logger.info(
                "GHCN cache has %d years; fetching remaining: %s",
                len(cached_years), missing_years,
            )
            years = missing_years
        else:
            cached = pd.DataFrame()
    else:
        cached = pd.DataFrame()

    # 1. Load station metadata and map to counties
    try:
        stations_df = load_stations(ghcn_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        return pd.DataFrame()

    stations_df = map_stations_to_counties(stations_df, coords_df)
    station_fips = dict(zip(stations_df["station_id"], stations_df["fips"]))
    valid_stations = set(station_fips.keys())
    logger.info("Mapped %d stations to %d counties.", len(valid_stations), len(set(station_fips.values())))

    # 2. Process each year
    all_records = []
    missing_files = []

    for year in years:
        daily_df = _load_year_file(year, ghcn_dir, valid_stations)
        if daily_df.empty:
            missing_files.append(year)
            continue
        records = _aggregate_to_county_year(daily_df, station_fips)
        all_records.extend(records)
        logger.info("Year %d: %d county records.", year, len(records))

    if missing_files:
        logger.warning(
            "Missing GHCN files for years: %s\n"
            "Download with:\n%s",
            missing_files,
            "\n".join(
                f"  https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_year/{y}.csv.gz"
                for y in missing_files
            ),
        )

    if not all_records:
        logger.error("No records produced. Check that GHCN files are in: %s", ghcn_dir)
        return pd.DataFrame()

    result = pd.DataFrame(all_records)

    # Combine with cached data if partial
    if not cached.empty:
        result = pd.concat([cached, result], ignore_index=True)
        result = result.drop_duplicates(subset=["fips", "year"])

    if cache_path:
        result.to_csv(cache_path, index=False)
        logger.info("Saved GHCN results to cache: %s", cache_path)

    logger.info(
        "GHCN pipeline complete: %d records, %d counties, %d years.",
        len(result), result["fips"].nunique(), result["year"].nunique(),
    )
    return result


def check_ghcn_files(
    years: list[int],
    ghcn_dir: Path | str = GHCN_DIR_DEFAULT,
) -> None:
    """
    Print a download checklist showing which files are present and which are missing.
    Run this before build_weather_from_ghcn() to verify your downloads.
    """
    ghcn_dir = Path(ghcn_dir)
    print(f"\nGHCN-Daily file check — directory: {ghcn_dir.resolve()}")
    print("-" * 60)

    stations_ok = (ghcn_dir / STATIONS_FILE).exists()
    print(f"  {'✓' if stations_ok else '✗'} ghcnd-stations.txt", end="")
    if not stations_ok:
        print(f"\n    → Download: https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt")
    else:
        print()

    for year in years:
        gz  = (ghcn_dir / f"{year}.csv.gz").exists()
        csv = (ghcn_dir / f"{year}.csv").exists()
        ok  = gz or csv
        fmt = ".csv.gz" if gz else ".csv" if csv else ".csv.gz (MISSING)"
        print(f"  {'✓' if ok else '✗'} {year}{fmt}", end="")
        if not ok:
            print(f"\n    → Download: https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_year/{year}.csv.gz")
        else:
            print()

    print()
