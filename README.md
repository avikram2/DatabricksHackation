# Databricks Hackathon — Crop Yield Intelligence Platform

County-level Corn & Soybean yield analysis (2010-2024, 32 US states) built on Databricks + NOAA weather APIs.

## Solution

| Option | Directory | Description |
|--------|-----------|-------------|
| 1 | `option1_weather_signal_detection/` | Correlate historical weather with yield, flag anomalies, explain outliers |

## Data Source
- **Yields**: USDA RMA County Yields Report (Corn + Soybeans, 34,712 rows)
- **Weather**: Open-Meteo Historical API (free, no API key required)
- **County Centroids**: US Census Bureau 2020 county population centroids

## Databricks Architecture

```
Databricks Workspace
  Delta Lake     -- persistent yield + weather tables (bronze/silver/gold)
  MLflow         -- model experiment tracking and registry
  Databricks SQL -- interactive dashboards
  Notebooks      -- 5-stage Python pipeline
```

## Quick Start

```bash
pip install -r option1_weather_signal_detection/requirements.txt
```

Upload each `0N_*.py` file to Databricks as a notebook and run in order (01 to 05).
All notebooks also run locally -- Spark/Delta calls gracefully fall back to pandas.

## Key Findings
- 2012 drought: Z-score < -2 in 80%+ of Corn Belt counties, most anomalous year in dataset
- Top signals: growing-season precipitation, Growing Degree Days, heat-stress days (>95F)
- 2022 collapse: July heat waves drove southern-tier yield losses of 15-25%

