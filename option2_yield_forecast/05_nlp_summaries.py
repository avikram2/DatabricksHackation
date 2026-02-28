# Databricks notebook source
# Option 2 — Notebook 05: Natural Language Summaries for Farmers
#
# Generates:
#   1. Rule-based NL alerts (always available, no LLM required)
#   2. Structured JSON alert feed (API-ready)
#   3. County-level agronomist report
#   4. Optional: Claude/OpenAI LLM enrichment (if API key configured)
#
# Output: console + ./output/farmer_alerts.csv + ./output/reports/*.txt

# COMMAND ----------

import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import ALERT_THRESHOLDS, FORECAST_DELTA, OUTPUT_DIR
from utils import get_spark, read_delta

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

REPORTS_DIR = OUTPUT_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Load forecast data

# COMMAND ----------

spark = get_spark()
forecast_df = read_delta(FORECAST_DELTA, spark=spark)
print(f"Forecast rows loaded: {len(forecast_df):,}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Rule-based NL alert engine

# COMMAND ----------

CROP_UNITS = {"Corn": "bu/ac", "Soybeans": "bu/ac"}

HEAT_STRESS_THRESHOLDS = ALERT_THRESHOLDS["heat_stress_days"]
PRECIP_THRESHOLDS = ALERT_THRESHOLDS["precip_total_mm"]
GDD_THRESHOLDS = ALERT_THRESHOLDS["gdd_base10"]
DROP_THRESHOLDS = ALERT_THRESHOLDS["yield_drop_pct"]


def generate_alert_message(row: pd.Series) -> str:
    """
    Generate a plain-English alert message for a forecast row.

    The message is written for a grower/agronomist audience.
    """
    county = row.get("county_name", "Unknown County")
    state = row.get("state_abbr", "")
    crop = row.get("commodity_name", "crop")
    scenario = row.get("scenario", "baseline").replace("_", " ").title()
    p50 = row.get("yield_p50", 0)
    p10 = row.get("yield_p10", 0)
    p90 = row.get("yield_p90", 0)
    hist = row.get("hist_mean", 0)
    drop_pct = row.get("drop_pct_vs_hist", 0)
    heat = row.get("weather_heat_stress_days", 0)
    precip = row.get("weather_precip_total", 0)
    gdd = row.get("weather_gdd", 0)
    risk = row.get("risk_level", "NORMAL")

    unit = CROP_UNITS.get(crop, "bu/ac")

    # Opening sentence
    if risk == "NORMAL":
        headline = (
            f"{county}, {state}: {crop} yield looks on track. "
            f"Expected {p50:.0f} {unit} (historical avg: {hist:.0f})."
        )
    elif risk == "WARNING":
        headline = (
            f"ATTENTION — {county}, {state}: {crop} yield may be below average. "
            f"Forecast median {p50:.0f} {unit}, down {drop_pct:.0f}% from your {hist:.0f} {unit} historical average."
        )
    elif risk == "ALERT":
        headline = (
            f"ALERT — {county}, {state}: {crop} yield at risk of significant loss. "
            f"Forecast median {p50:.0f} {unit} (–{drop_pct:.0f}% vs historical avg {hist:.0f} {unit})."
        )
    else:  # CRITICAL
        headline = (
            f"CRITICAL ALERT — {county}, {state}: {crop} facing severe yield loss. "
            f"Forecast median {p50:.0f} {unit} represents a {drop_pct:.0f}% decline vs {hist:.0f} {unit} average."
        )

    # Weather detail bullets
    bullets = []

    if heat >= HEAT_STRESS_THRESHOLDS["critical"]:
        bullets.append(
            f"Severe heat stress: {heat:.0f} days above 95°F forecast this season. "
            "Extended heat during pollination can reduce corn yield by 10–20%."
        )
    elif heat >= HEAT_STRESS_THRESHOLDS["high"]:
        bullets.append(
            f"Elevated heat stress: {heat:.0f} days above 95°F expected. "
            "Monitor closely; consider scouting for pollen viability issues."
        )

    if precip <= PRECIP_THRESHOLDS["very_low"]:
        bullets.append(
            f"Very low precipitation: only {precip:.0f} mm forecast for the growing season "
            f"(threshold for concern: {PRECIP_THRESHOLDS['low']} mm). "
            "Severe drought risk — evaluate irrigation options immediately."
        )
    elif precip <= PRECIP_THRESHOLDS["low"]:
        bullets.append(
            f"Below-normal precipitation: {precip:.0f} mm expected. "
            "Below-average moisture may limit yield potential. Consider conservation practices."
        )
    elif precip > 600:
        bullets.append(
            f"Above-average precipitation: {precip:.0f} mm forecast. "
            "Risk of excess moisture, potential for delayed planting, disease pressure, and N leaching."
        )

    if gdd <= GDD_THRESHOLDS["critical"]:
        bullets.append(
            f"Insufficient heat accumulation: {gdd:.0f} GDD (base 10°C) forecast. "
            "Crops may not reach maturity — evaluate earlier maturity varieties."
        )
    elif gdd <= GDD_THRESHOLDS["low"]:
        bullets.append(
            f"Below-normal heat accumulation: {gdd:.0f} GDD forecast. "
            "Potential for late maturity; watch for early frost risk."
        )

    # Uncertainty range
    uncertainty = (
        f"Uncertainty range: P10={p10:.0f} to P90={p90:.0f} {unit} "
        f"(80% of simulated scenarios fall within this band)."
    )

    # Scenario context
    scen_context = f"This forecast assumes {scenario} weather conditions."

    parts = [headline, ""]
    if bullets:
        parts += ["Key factors:"] + [f"  • {b}" for b in bullets] + [""]
    parts += [uncertainty, scen_context]

    return "\n".join(parts)


# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Generate alerts for all counties × scenarios

# COMMAND ----------

forecast_df["alert_message"] = forecast_df.apply(generate_alert_message, axis=1)

# Print sample alerts
print("=" * 70)
print("SAMPLE FARMER ALERTS")
print("=" * 70)
sample = forecast_df[
    (forecast_df["scenario"] == "drought") & (forecast_df["commodity_name"] == "Corn")
].nlargest(3, "drop_pct_vs_hist")

for _, row in sample.iterrows():
    print(row["alert_message"])
    print("-" * 70)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Structured JSON alert feed (for app/API integration)

# COMMAND ----------

def row_to_alert_json(row: pd.Series) -> dict[str, Any]:
    """Convert forecast row to a structured alert JSON object."""
    return {
        "alert_id": f"{row['fips']}_{row['commodity_name']}_{row['forecast_year']}_{row['scenario']}",
        "location": {
            "fips": row["fips"],
            "county": row.get("county_name", ""),
            "state": row.get("state_abbr", ""),
            "lat": row.get("lat"),
            "lon": row.get("lon"),
        },
        "crop": row["commodity_name"],
        "forecast_year": int(row["forecast_year"]),
        "scenario": row["scenario"],
        "risk_level": row["risk_level"],
        "yield_forecast": {
            "p10_bu_ac": row["yield_p10"],
            "p25_bu_ac": row["yield_p25"],
            "p50_bu_ac": row["yield_p50"],
            "p75_bu_ac": row["yield_p75"],
            "p90_bu_ac": row["yield_p90"],
        },
        "historical_baseline": {
            "mean_bu_ac": row["hist_mean"],
            "std_bu_ac": row["hist_std"],
        },
        "change_vs_baseline_pct": row["drop_pct_vs_hist"],
        "weather_summary": {
            "tmax_mean_c": row.get("weather_tmax_mean"),
            "precip_total_mm": row.get("weather_precip_total"),
            "gdd_base10": row.get("weather_gdd"),
            "heat_stress_days": row.get("weather_heat_stress_days"),
        },
        "message": row["alert_message"],
    }

alerts_json = [row_to_alert_json(r) for _, r in forecast_df.iterrows()]
json_path = OUTPUT_DIR / "alerts_feed.json"
with open(json_path, "w") as f:
    json.dump(alerts_json, f, indent=2, default=str)
print(f"JSON alert feed saved: {json_path} ({len(alerts_json)} alerts)")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Per-county agronomist report

# COMMAND ----------

def write_county_report(county_fc: pd.DataFrame, county_name: str, state: str, crop: str):
    """Write a structured text report for an agronomist."""
    fips = county_fc["fips"].iloc[0] if "fips" in county_fc.columns else "unknown"
    report_lines = [
        f"{'='*60}",
        f"AGRONOMIST YIELD FORECAST REPORT",
        f"{'='*60}",
        f"County:        {county_name}, {state}",
        f"Crop:          {crop}",
        f"Forecast Year: {county_fc['forecast_year'].iloc[0]}",
        f"Generated:     {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"{'='*60}",
        "",
        "SCENARIO COMPARISON",
        f"{'Scenario':<20} {'P10':>8} {'P50':>8} {'P75':>8} {'P90':>8} {'Drop%':>8} {'Risk':>10}",
        "-"*60,
    ]
    for _, row in county_fc.iterrows():
        report_lines.append(
            f"{row['scenario']:<20} {row['yield_p10']:>8.1f} {row['yield_p50']:>8.1f} "
            f"{row['yield_p75']:>8.1f} {row['yield_p90']:>8.1f} "
            f"{row['drop_pct_vs_hist']:>7.1f}% {row['risk_level']:>10}"
        )

    report_lines += [
        "",
        "BASELINE WEATHER SUMMARY",
        "-"*60,
    ]
    baseline = county_fc[county_fc["scenario"] == "baseline"]
    if not baseline.empty:
        b = baseline.iloc[0]
        report_lines += [
            f"  Max Temp (avg): {b.get('weather_tmax_mean', 'N/A')} °C",
            f"  Precipitation:  {b.get('weather_precip_total', 'N/A')} mm",
            f"  GDD (base 10):  {b.get('weather_gdd', 'N/A')}",
            f"  Heat Stress:    {b.get('weather_heat_stress_days', 'N/A')} days >35°C",
        ]

    report_lines += [
        "",
        "DETAILED ALERTS",
        "-"*60,
    ]
    for _, row in county_fc.iterrows():
        if row["risk_level"] != "NORMAL":
            report_lines.append(row["alert_message"])
            report_lines.append("")

    return "\n".join(report_lines)


# Write reports for unique counties
for fips in forecast_df["fips"].unique()[:10]:   # cap at 10 for demo
    for crop in ["Corn", "Soybeans"]:
        county_fc = forecast_df[
            (forecast_df["fips"] == fips) & (forecast_df["commodity_name"] == crop)
        ]
        if county_fc.empty:
            continue
        county_name = county_fc["county_name"].iloc[0] if "county_name" in county_fc.columns else fips
        state = county_fc["state_abbr"].iloc[0] if "state_abbr" in county_fc.columns else ""
        report = write_county_report(county_fc, county_name, state, crop)
        rpath = REPORTS_DIR / f"{fips}_{crop.lower()}_report.txt"
        rpath.write_text(report)

print(f"County reports written to: {REPORTS_DIR.resolve()}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. CSV export (for dashboards / Excel)

# COMMAND ----------

csv_cols = [
    "fips", "county_name", "state_abbr", "commodity_name", "forecast_year",
    "scenario", "risk_level",
    "yield_p10", "yield_p25", "yield_p50", "yield_p75", "yield_p90",
    "hist_mean", "drop_pct_vs_hist",
    "weather_tmax_mean", "weather_precip_total", "weather_gdd", "weather_heat_stress_days",
]
csv_cols = [c for c in csv_cols if c in forecast_df.columns]
forecast_df[csv_cols].to_csv(OUTPUT_DIR / "farmer_alerts.csv", index=False)
print(f"Alerts CSV saved: {OUTPUT_DIR / 'farmer_alerts.csv'}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Optional: LLM enrichment via Claude API

# COMMAND ----------

# Uncomment and set ANTHROPIC_API_KEY environment variable to enable
#
# import os, anthropic
#
# client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
#
# def enrich_with_llm(alert_text: str, county: str, crop: str) -> str:
#     prompt = f"""
#     You are an agricultural advisor. A grower in {county} is growing {crop}.
#     Here is a yield forecast summary:
#
#     {alert_text}
#
#     Write a concise, actionable 2-3 sentence recommendation for this grower.
#     Use plain language. Focus on what they should do NOW, not just risks.
#     """
#     message = client.messages.create(
#         model="claude-opus-4-6",
#         max_tokens=200,
#         messages=[{"role": "user", "content": prompt}],
#     )
#     return message.content[0].text
#
# # Apply to high-risk rows
# critical = forecast_df[forecast_df["risk_level"].isin(["ALERT", "CRITICAL"])]
# critical["llm_recommendation"] = critical.apply(
#     lambda r: enrich_with_llm(r["alert_message"], r.get("county_name",""), r["commodity_name"]),
#     axis=1
# )

print("\n=== Notebook 05 complete — Option 2 pipeline finished! ===")
print(f"All outputs in: {OUTPUT_DIR.resolve()}")
print("""
Key outputs:
  output/farmer_alerts.csv      — machine-readable alert table
  output/alerts_feed.json       — structured JSON for app integration
  output/fan_chart_corn.html    — interactive yield prediction intervals
  output/forecast_map_corn.html — county choropleth with risk colours
  output/tornado_corn.html      — scenario sensitivity chart
  output/validation_corn.html   — 2023 backtest scatter
  output/reports/*.txt          — county agronomist reports
""")
