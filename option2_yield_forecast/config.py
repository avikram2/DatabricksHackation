"""
Shared configuration for Option 2: 60/90-Day Yield Forecast.
"""

from pathlib import Path

ROOT = Path(__file__).parent

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
YIELD_CSV = ROOT / "data" / "RMACountyYieldsReport-399.csv"

DELTA_ROOT = ROOT / "delta"
YIELD_DELTA = str(DELTA_ROOT / "yields")
FEATURE_DELTA = str(DELTA_ROOT / "features")
FORECAST_DELTA = str(DELTA_ROOT / "forecasts")
SCENARIOS_DELTA = str(DELTA_ROOT / "scenarios")

CENTROID_CACHE = str(ROOT / "data" / "county_centroids.csv")
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "output"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# MLflow
MLFLOW_EXPERIMENT = "/Hackathon/YieldForecast"

# ---------------------------------------------------------------------------
# Databricks Unity Catalog
# ---------------------------------------------------------------------------
DATABRICKS_CATALOG = "hackathon"
DATABRICKS_SCHEMA = "yield_forecast"
DATABRICKS_FEATURE_TABLE = f"{DATABRICKS_CATALOG}.{DATABRICKS_SCHEMA}.features"
DATABRICKS_FORECAST_TABLE = f"{DATABRICKS_CATALOG}.{DATABRICKS_SCHEMA}.forecasts"

# ---------------------------------------------------------------------------
# Analysis parameters
# ---------------------------------------------------------------------------
CROPS = ["Corn", "Soybeans"]

YEAR_MIN = 2010
YEAR_MAX = 2024
TRAIN_YEARS = list(range(2010, 2023))   # 2010-2022 training
VAL_YEARS = [2023]                       # 2023 validation
TEST_YEARS = [2024]                      # 2024 "holdout" / forecast target

# Current forecast horizon
FORECAST_HORIZONS = [60, 90]            # days from June 1 into growing season

# Uncertainty quantification
QUANTILE_LEVELS = [0.10, 0.25, 0.50, 0.75, 0.90]

# Monte Carlo scenarios per county
N_SCENARIOS = 1000

# Feature engineering
LAG_YEARS = [1, 2, 3]                   # previous year yields as features
ROLLING_WINDOWS = [3, 5]               # rolling-mean yield windows

# Focus states (by corn/soy production volume)
FOCUS_STATES = ["IA", "IL", "IN", "MN", "OH", "NE", "KS", "MO", "SD", "ND"]

# Open-Meteo forecast endpoint
FORECAST_DAYS = 16                      # free tier maximum

# Natural-language alert thresholds
ALERT_THRESHOLDS = {
    "heat_stress_days": {"high": 15, "critical": 25},
    "precip_total_mm": {"low": 200, "very_low": 100},
    "gdd_base10": {"low": 1100, "critical": 900},
    "yield_drop_pct": {"warning": 5, "alert": 10, "critical": 15},
}
