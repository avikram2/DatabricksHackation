# Databricks notebook source
# Option 1 — Notebook 05: Visualization & Interactive Maps
#
# Produces:
#   1. County choropleth map of yield anomalies (Folium + GeoJSON)
#   2. Heatmap: weather features vs year by state (Plotly)
#   3. Scatter: weather signal vs yield with anomaly highlights
#   4. Time-series: county-level yield + weather overlaid
#   5. 2012 drought and 2022 heat-wave county maps
#
# Output files written to ./output/ (or displayed inline in Databricks)

# COMMAND ----------

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


from config import ANOMALY_DELTA, FOCUS_STATES, MERGED_DELTA
from utils import get_spark, read_delta

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Load data

# COMMAND ----------

spark = get_spark()
merged = read_delta(MERGED_DELTA, spark=spark)
anomalies = read_delta(ANOMALY_DELTA, spark=spark)

print(f"Merged rows: {len(merged):,}")
print(f"Anomaly rows: {len(anomalies):,}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Choropleth map — Yield anomaly Z-score by county (Folium)

# COMMAND ----------

try:
    import folium
    import requests
    from folium.plugins import HeatMap

    # Fetch US county GeoJSON (simplified, ~2 MB)
    COUNTY_GEOJSON_URL = (
        "https://raw.githubusercontent.com/plotly/datasets/master/"
        "geojson-counties-fips.json"
    )

    def make_anomaly_map(year: int, crop: str = "Corn") -> folium.Map:
        """Create a choropleth map of yield Z-score anomalies for a given year."""
        subset = merged[
            (merged["year"] == year) & (merged["commodity_name"] == crop)
        ][["fips", "yield_z_robust", "yield_bu_ac", "county_name", "state_abbr"]].copy()

        if "yield_z_robust" not in subset.columns and "yield_zscore" in subset.columns:
            subset["yield_z_robust"] = subset["yield_zscore"]

        subset = subset.dropna(subset=["yield_z_robust"])

        try:
            resp = requests.get(COUNTY_GEOJSON_URL, timeout=30)
            geojson = resp.json()
        except Exception as exc:
            log.warning("Could not fetch GeoJSON: %s", exc)
            return None

        m = folium.Map(location=[39.5, -98.35], zoom_start=4, tiles="CartoDB positron")

        folium.Choropleth(
            geo_data=geojson,
            name=f"{crop} Yield Z-Score {year}",
            data=subset,
            columns=["fips", "yield_z_robust"],
            key_on="feature.id",
            fill_color="RdYlGn",
            fill_opacity=0.75,
            line_opacity=0.1,
            legend_name=f"{crop} Yield Z-Score ({year})",
            nan_fill_color="white",
        ).add_to(m)

        # Tooltip layer
        style_fn = lambda x: {
            "fillOpacity": 0.0,
            "weight": 0,
        }
        fips_to_info = subset.set_index("fips")[["yield_bu_ac", "yield_z_robust", "county_name", "state_abbr"]].to_dict("index")

        for feature in geojson["features"]:
            fips = feature["id"]
            if fips in fips_to_info:
                info = fips_to_info[fips]
                folium.GeoJson(
                    feature,
                    style_function=style_fn,
                    tooltip=folium.Tooltip(
                        f"{info['county_name']}, {info['state_abbr']}<br>"
                        f"Yield: {info['yield_bu_ac']:.1f} bu/ac<br>"
                        f"Z-score: {info['yield_z_robust']:.2f}"
                    ),
                ).add_to(m)

        folium.LayerControl().add_to(m)
        return m

    # Generate maps for key years
    for yr in [2012, 2021, 2022, 2024]:
        for crop in ["Corn", "Soybeans"]:
            m = make_anomaly_map(yr, crop)
            if m is not None:
                fname = OUTPUT_DIR / f"map_{crop.lower()}_{yr}.html"
                m.save(str(fname))
                log.info("Saved: %s", fname)

    print("Choropleth maps saved to ./output/")

except ImportError as e:
    log.warning("folium not installed — skipping map: %s", e)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Plotly heatmap — Weather feature trends by year × state

# COMMAND ----------

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    focus = merged[merged["state_abbr"].isin(FOCUS_STATES)].copy()

    # --- Heatmap: mean heat stress days by state × year ---
    heat_pivot = (
        focus[focus["commodity_name"] == "Corn"]
        .groupby(["state_abbr", "year"])["heat_stress_days"]
        .mean()
        .unstack("year")
        .fillna(0)
    )
    fig_heat = px.imshow(
        heat_pivot,
        color_continuous_scale="Reds",
        title="Mean Heat Stress Days (Tmax > 35°C) — Corn Growing Season",
        labels={"x": "Year", "y": "State", "color": "Days"},
        aspect="auto",
    )
    fig_heat.write_html(str(OUTPUT_DIR / "heatmap_heat_stress.html"))

    # --- Heatmap: precipitation Z-score by state × year ---
    if "precip_z" in focus.columns:
        precip_pivot = (
            focus[focus["commodity_name"] == "Corn"]
            .groupby(["state_abbr", "year"])["precip_z"]
            .mean()
            .unstack("year")
        )
        fig_precip = px.imshow(
            precip_pivot,
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            title="Precipitation Anomaly (Z-score) — Corn Growing Season",
            labels={"x": "Year", "y": "State", "color": "Precip Z"},
            aspect="auto",
        )
        fig_precip.write_html(str(OUTPUT_DIR / "heatmap_precipitation.html"))

    # --- Scatter: GDD vs Yield coloured by year ---
    for crop in ["Corn", "Soybeans"]:
        crop_data = focus[focus["commodity_name"] == crop].dropna(
            subset=["gdd_base10", "yield_bu_ac"]
        )
        fig_scatter = px.scatter(
            crop_data,
            x="gdd_base10",
            y="yield_bu_ac",
            color="year",
            hover_data=["county_name", "state_abbr", "year"],
            trendline="ols",
            title=f"{crop}: Growing Degree Days vs Yield",
            labels={"gdd_base10": "GDD (Base 10°C)", "yield_bu_ac": "Yield (bu/ac)"},
            color_continuous_scale="Viridis",
        )
        fig_scatter.write_html(str(OUTPUT_DIR / f"scatter_gdd_{crop.lower()}.html"))

    print("Plotly charts saved to ./output/")

except ImportError as e:
    log.warning("plotly not installed — skipping Plotly charts: %s", e)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Time-series: county yield vs precipitation

# COMMAND ----------

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Pick representative counties (top anomaly counties)
    if len(anomalies) > 0:
        top_counties = (
            anomalies.groupby("fips")["is_anomaly"].sum()
            .nlargest(6)
            .index.tolist()
        )
    else:
        top_counties = merged["fips"].value_counts().head(6).index.tolist()

    for crop in ["Corn", "Soybeans"]:
        fig = make_subplots(
            rows=len(top_counties), cols=1,
            shared_xaxes=True,
            subplot_titles=[
                f"{merged[merged['fips']==f]['county_name'].iloc[0]}, "
                f"{merged[merged['fips']==f]['state_abbr'].iloc[0]}"
                for f in top_counties if len(merged[merged["fips"] == f]) > 0
            ],
        )

        for i, fips in enumerate(top_counties):
            county_data = merged[
                (merged["fips"] == fips) & (merged["commodity_name"] == crop)
            ].sort_values("year")
            if county_data.empty:
                continue

            # Yield line
            fig.add_trace(
                go.Scatter(
                    x=county_data["year"], y=county_data["yield_bu_ac"],
                    name="Yield (bu/ac)", line=dict(color="steelblue"),
                    showlegend=(i == 0),
                ),
                row=i + 1, col=1,
            )

            # Mark anomalies
            anom = county_data[county_data.get("z_anomaly", pd.Series(False, index=county_data.index))]
            if "z_direction" in county_data.columns:
                anom = county_data[county_data["z_direction"] != "NORMAL"]
                fig.add_trace(
                    go.Scatter(
                        x=anom["year"], y=anom["yield_bu_ac"],
                        mode="markers",
                        marker=dict(color="red", size=10, symbol="x"),
                        name="Anomaly",
                        showlegend=(i == 0),
                    ),
                    row=i + 1, col=1,
                )

        fig.update_layout(
            height=200 * len(top_counties),
            title_text=f"{crop} — Yield Time Series with Anomaly Flags",
        )
        fig.write_html(str(OUTPUT_DIR / f"timeseries_{crop.lower()}.html"))

    print("Time-series charts saved.")

except ImportError as e:
    log.warning("plotly not installed: %s", e)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Matplotlib static summary (always available)

# COMMAND ----------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Weather-to-Yield Signal Detection — Summary", fontsize=14, fontweight="bold")

corn = merged[merged["commodity_name"] == "Corn"]

# Panel A: National average corn yield over time
ax = axes[0, 0]
yr_mean = corn.groupby("year")["yield_bu_ac"].agg(["mean", "std"]).reset_index()
ax.fill_between(yr_mean["year"], yr_mean["mean"] - yr_mean["std"], yr_mean["mean"] + yr_mean["std"], alpha=0.2, color="steelblue")
ax.plot(yr_mean["year"], yr_mean["mean"], "o-", color="steelblue", linewidth=2)
ax.axvline(2012, color="red", linestyle="--", label="2012 drought")
ax.axvline(2022, color="orange", linestyle="--", label="2022 heat")
ax.set_title("A. National Average Corn Yield")
ax.set_ylabel("Yield (bu/ac)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Panel B: Correlation bar chart
if "precip_total_mm" in corn.columns:
    weather_feats = ["precip_total_mm", "gdd_base10", "heat_stress_days", "tavg_mean_c", "drought_days"]
    weather_feats = [f for f in weather_feats if f in corn.columns]
    corrs = {f: corn[["yield_bu_ac", f]].dropna().corr().iloc[0, 1] for f in weather_feats}
    labels, vals = zip(*sorted(corrs.items(), key=lambda x: x[1]))
    colors = ["tomato" if v < 0 else "mediumseagreen" for v in vals]
    ax = axes[0, 1]
    ax.barh(labels, vals, color=colors)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_title("B. Weather–Yield Pearson Correlation (Corn)")
    ax.set_xlabel("Pearson r")
    ax.grid(alpha=0.3)
else:
    axes[0, 1].text(0.5, 0.5, "Weather data not yet loaded", ha="center")

# Panel C: Anomaly count by year
if "is_anomaly" in merged.columns:
    anom_yr = merged.groupby("year")["is_anomaly"].sum().reset_index()
    axes[1, 0].bar(anom_yr["year"], anom_yr["is_anomaly"], color="coral")
    axes[1, 0].set_title("C. Anomaly Count by Year")
    axes[1, 0].set_ylabel("# Flagged Counties")
    axes[1, 0].grid(alpha=0.3)
else:
    # Show yield distribution by year as proxy
    corn.boxplot(column="yield_bu_ac", by="year", ax=axes[1, 0], rot=45)
    axes[1, 0].set_title("C. Corn Yield Distribution by Year")

# Panel D: Scatter precip vs yield (all years)
if "precip_total_mm" in corn.columns:
    sample = corn.dropna(subset=["precip_total_mm", "yield_bu_ac"]).sample(min(500, len(corn)), random_state=42)
    axes[1, 1].scatter(sample["precip_total_mm"], sample["yield_bu_ac"], alpha=0.3, s=8, color="steelblue")
    axes[1, 1].set_title("D. Growing-Season Precip vs Corn Yield")
    axes[1, 1].set_xlabel("Precipitation (mm)")
    axes[1, 1].set_ylabel("Yield (bu/ac)")
    axes[1, 1].grid(alpha=0.3)
else:
    axes[1, 1].text(0.5, 0.5, "Weather data not yet loaded", ha="center")

plt.tight_layout()
out_path = OUTPUT_DIR / "summary_panel.png"
plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
plt.close()
print(f"Summary panel saved: {out_path}")

print("\n=== Notebook 05 complete — Option 1 pipeline finished! ===")
print(f"All outputs in: {OUTPUT_DIR.resolve()}")
