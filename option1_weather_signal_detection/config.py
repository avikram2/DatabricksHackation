"""
Shared configuration for Option 1: Weather-to-Yield Signal Detection.

Edit these paths to match your Databricks workspace / local environment.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent

# Local copy of the RMA yield CSV (place it alongside this file, or set full path)
YIELD_CSV = ROOT / "data" / "RMACountyYieldsReport-399.csv"

# Delta Lake / parquet paths (relative or DBFS paths on Databricks)
# On Databricks use something like: "dbfs:/hackathon/yields"
DELTA_ROOT = ROOT / "delta"
YIELD_DELTA = str(DELTA_ROOT / "yields")
WEATHER_DELTA = str(DELTA_ROOT / "weather")
MERGED_DELTA = str(DELTA_ROOT / "merged")
ANOMALY_DELTA = str(DELTA_ROOT / "anomalies")

# Cache for county centroids (avoid re-downloading every run)
CENTROID_CACHE = str(ROOT / "data" / "county_centroids.csv")

# MLflow experiment name
MLFLOW_EXPERIMENT = "/Hackathon/WeatherYieldSignals"

# ---------------------------------------------------------------------------
# Analysis parameters
# ---------------------------------------------------------------------------

# Crops to analyse
CROPS = ["Corn", "Soybeans"]

# Year range in the dataset
YEAR_MIN = 2010
YEAR_MAX = 2024

# Years used for correlation / model training
TRAIN_YEARS = list(range(2010, 2023))
TEST_YEARS = [2023, 2024]

# Anomaly Z-score threshold
ANOMALY_ZSCORE_THRESHOLD = 1.5   # |z| > this => flagged

# Isolation Forest contamination
ISOLATION_FOREST_CONTAMINATION = 0.08   # ~8% of samples expected to be anomalous

# Top-N states to focus on for visualisation (by data volume)
FOCUS_STATES = ["IA", "IL", "IN", "MN", "OH", "NE", "KS", "MO", "SD", "ND"]

# Open-Meteo fetch delay (seconds between requests — be polite to the free API)
WEATHER_FETCH_DELAY_S = 0.15

# ---------------------------------------------------------------------------
# Databricks-specific (ignored when running locally)
# ---------------------------------------------------------------------------
DATABRICKS_CATALOG = "hackathon"
DATABRICKS_SCHEMA = "crop_yields"
DATABRICKS_YIELD_TABLE = f"{DATABRICKS_CATALOG}.{DATABRICKS_SCHEMA}.yields"
DATABRICKS_WEATHER_TABLE = f"{DATABRICKS_CATALOG}.{DATABRICKS_SCHEMA}.weather"
DATABRICKS_MERGED_TABLE = f"{DATABRICKS_CATALOG}.{DATABRICKS_SCHEMA}.merged"
