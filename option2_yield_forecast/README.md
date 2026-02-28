# Option 2: 60/90-Day Yield Forecast Simulation

Predictive model that estimates expected yield for the current growing season, quantifies uncertainty via Monte Carlo simulation, and delivers plain-English farmer alerts.

## Pipeline

```
01_feature_engineering.py   Yield lags, rolling means, county effects, weather → feature matrix
02_model_training.py        XGBoost (primary) + LightGBM + Quantile Forest → MLflow
03_forecast_simulation.py   Monte Carlo × 5 weather scenarios → yield distributions (P10–P90)
04_uncertainty_viz.py       Fan charts, choropleth maps, tornado, validation backtest
05_nlp_summaries.py         Rule-based alerts + JSON feed + county reports + optional Claude LLM
```

## Feature Engineering

### Yield features
| Feature | Description |
|---------|-------------|
| `yield_lag_1/2/3` | Previous 1/2/3 year yield (same county-crop) |
| `yield_roll3y/5y` | 3-year and 5-year trailing mean yield |
| `county_mean_yield` | Long-run county average |
| `county_yield_trend` | OLS slope of yield over time |
| `county_yield_cv` | Coefficient of variation (volatility proxy) |

### Weather features
| Feature | Description |
|---------|-------------|
| `gdd_base10` | Growing Degree Days accumulated Apr–Oct |
| `heat_stress_days` | Days with Tmax > 35°C (95°F) |
| `precip_total_mm` | Growing-season total precipitation |
| `spi_proxy` | Standardised Precipitation Index (log Z-score) |
| `cwsi` | Crop Water Stress Index (ET₀ / precip) |
| `tmax_mean_c`, `tmin_mean_c` | Growing-season temperature means |

## Uncertainty Quantification

Monte Carlo simulation: 1,000 synthetic weather seasons per county, sampled from:
- **Days 1–16**: Open-Meteo 16-day deterministic forecast
- **Days 17–90**: Climate normals ± Gaussian noise (σ from 2010–2022 history)

Five scenarios:
| Scenario | Temp shift | Precip scale |
|----------|-----------|-------------|
| Baseline | +0°C | ×1.0 |
| Drought | +1.5°C | ×0.5 |
| Excess rain | +0°C | ×2.0 |
| Heat wave | +3.0°C | ×1.0 |
| Ideal | -0.5°C | ×1.1 |

## Alert Thresholds

| Condition | Warning | Alert | Critical |
|-----------|---------|-------|----------|
| Yield drop vs historical | >5% | >10% | >15% |
| Heat stress days | >15 | - | >25 |
| Growing-season precip | <200 mm | - | <100 mm |
| GDD (base 10°C) | <1100 | - | <900 |

## Outputs

```
output/
  farmer_alerts.csv           All county × scenario forecasts (CSV)
  alerts_feed.json            Structured JSON for app/API integration
  fan_chart_corn.html         Prediction intervals by state × scenario
  fan_chart_soybeans.html
  forecast_map_corn.html      P50 county choropleth with risk colouring
  tornado_corn.html           Scenario sensitivity / yield impact
  validation_corn.html        2023 backtest: predicted vs actual
  reports/
    {fips}_corn_report.txt    Per-county agronomist text report
```

## Example Alert

```
ALERT — Story County, IA: Corn yield at risk of significant loss.
Forecast median 142 bu/ac (–14% vs historical avg 165 bu/ac).

Key factors:
  • Severe heat stress: 28 days above 95°F forecast this season.
    Extended heat during pollination can reduce corn yield by 10–20%.
  • Very low precipitation: only 95 mm forecast (threshold: 200 mm).
    Severe drought risk — evaluate irrigation options immediately.

Uncertainty range: P10=118 to P90=165 bu/ac.
This forecast assumes Drought weather conditions.
```

## Running on Databricks

1. Run Option 1 pipeline first (or weather data will be re-fetched)
2. Upload notebooks to workspace
3. Run on DBR 14+ ML Runtime (has MLflow, sklearn pre-installed)
4. Models registered in MLflow Model Registry automatically
