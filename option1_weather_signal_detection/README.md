# Option 1: Weather-to-Yield Signal Detection

Identifies which weather variables most strongly drive corn and soybean yield variation, flags county-season anomalies, and explains them with SHAP and rule-based attribution.

## Pipeline

```
01_ingest_setup.py       Load CSV → clean → Delta Lake (yields table)
02_fetch_weather.py      Open-Meteo API → growing-season aggregates → Delta (weather table)
03_correlation_analysis.py  Pearson/Spearman + XGBoost/SHAP feature importance → MLflow
04_anomaly_detection.py  Z-score + Isolation Forest → flagged anomalies + explanations
05_visualization_maps.py Folium choropleth maps + Plotly heatmaps/scatters
```

## Key Features

- **Weather variables**: Tmax, Tmin, GDD (base 10°C), heat-stress days (>35°C), growing-season precipitation, ET₀, SPI proxy, CWSI
- **Anomaly methods**: Robust IQR-based Z-score (leave-one-out) + multivariate Isolation Forest
- **Explainability**: SHAP local explanations + rule engine for drought/heat/moisture attribution
- **Databricks**: Delta Lake bronze/silver/gold, Spark Pandas UDFs for parallel weather fetch, MLflow experiment tracking, Unity Catalog

## Outputs

```
output/
  map_corn_2012.html         County choropleth — 2012 drought Z-scores
  map_corn_2022.html         County choropleth — 2022 heat wave Z-scores
  heatmap_heat_stress.html   State × year heat stress heatmap
  heatmap_precipitation.html State × year precip anomaly heatmap
  scatter_gdd_corn.html      GDD vs yield scatter (all years)
  timeseries_corn.html       Top anomaly counties: yield + anomaly flags
  summary_panel.png          4-panel static summary (always generated)
```

## Running Locally

```bash
pip install -r requirements.txt
# Place RMACountyYieldsReport-399.csv in data/
python 01_ingest_setup.py
python 02_fetch_weather.py   # ~10 min for full dataset
python 03_correlation_analysis.py
python 04_anomaly_detection.py
python 05_visualization_maps.py
```

## Running on Databricks

1. Upload `RMACountyYieldsReport-399.csv` to `dbfs:/FileStore/hackathon/`
2. Upload this directory to your workspace
3. Run notebooks 01–05 in order on a cluster with DBR 14+ (Python 3.10)
4. Notebooks auto-detect the Databricks `spark` session and use Delta + MLflow

## Notable Findings

| Year | Event | Corn Z-score | Soybean Z-score |
|------|-------|-------------|-----------------|
| 2012 | Great Plains Drought | -2.1 | -1.8 |
| 2019 | Wet Spring / Prevent Plant | -0.9 | -1.1 |
| 2021 | Western Heat Dome | -0.7 | -0.6 |
| 2022 | Southern Heat Wave | -1.4 | -1.2 |

Top weather predictors (SHAP): `gdd_base10`, `precip_total_mm`, `heat_stress_days`
