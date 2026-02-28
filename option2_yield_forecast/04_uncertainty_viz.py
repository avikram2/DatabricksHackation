# Databricks notebook source
# Option 2 — Notebook 04: Uncertainty Visualization
#
# Produces:
#   1. Fan chart — yield prediction intervals by scenario × county
#   2. Map — P50 forecast with risk-level colouring (Folium choropleth)
#   3. Scenario comparison — violin + box plots per state
#   4. Sensitivity tornado chart — which weather variable drives variance most
#   5. Validation backtest — predicted vs actual 2023 with PI coverage

# COMMAND ----------

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import FEATURE_DELTA, FORECAST_DELTA, FOCUS_STATES, OUTPUT_DIR, TRAIN_YEARS
from utils import get_spark, read_delta

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Load forecast data

# COMMAND ----------

spark = get_spark()
forecast_df = read_delta(FORECAST_DELTA, spark=spark)
feature_df = read_delta(FEATURE_DELTA, spark=spark)
print(f"Forecast rows: {len(forecast_df):,}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Fan chart — prediction intervals by scenario

# COMMAND ----------

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    SCENARIO_COLOURS = {
        "baseline": "#2196F3",
        "drought": "#F44336",
        "excess_rain": "#4CAF50",
        "heat_wave": "#FF9800",
        "ideal": "#9C27B0",
    }

    for crop in ["Corn", "Soybeans"]:
        crop_fc = forecast_df[forecast_df["commodity_name"] == crop]
        if crop_fc.empty:
            continue

        # Aggregate to state level for clearer fan chart
        state_fc = (
            crop_fc.groupby(["state_abbr", "scenario"])[
                ["yield_p10", "yield_p25", "yield_p50", "yield_p75", "yield_p90", "hist_mean"]
            ]
            .mean()
            .reset_index()
        )

        fig = go.Figure()

        for scenario, grp in state_fc.groupby("scenario"):
            colour = SCENARIO_COLOURS.get(scenario, "#607D8B")
            states = grp["state_abbr"].tolist()
            p50 = grp["yield_p50"].tolist()
            p10 = grp["yield_p10"].tolist()
            p90 = grp["yield_p90"].tolist()
            p25 = grp["yield_p25"].tolist()
            p75 = grp["yield_p75"].tolist()

            # 80% interval band
            fig.add_trace(
                go.Scatter(
                    x=states + states[::-1],
                    y=p10 + p90[::-1],
                    fill="toself",
                    fillcolor=f"rgba{tuple(int(colour.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.15,)}",
                    line=dict(color="rgba(0,0,0,0)"),
                    name=f"{scenario} 80% PI",
                    showlegend=False,
                )
            )
            # 50% interval band
            fig.add_trace(
                go.Scatter(
                    x=states + states[::-1],
                    y=p25 + p75[::-1],
                    fill="toself",
                    fillcolor=f"rgba{tuple(int(colour.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.30,)}",
                    line=dict(color="rgba(0,0,0,0)"),
                    name=f"{scenario} 50% PI",
                    showlegend=False,
                )
            )
            # Median line
            fig.add_trace(
                go.Scatter(
                    x=states, y=p50,
                    mode="lines+markers",
                    line=dict(color=colour, width=2),
                    name=scenario.replace("_", " ").title(),
                )
            )

        # Historical mean reference
        hist = state_fc[state_fc["scenario"] == "baseline"].groupby("state_abbr")["hist_mean"].first()
        fig.add_trace(
            go.Scatter(
                x=hist.index.tolist(), y=hist.values.tolist(),
                mode="lines",
                line=dict(color="black", width=1.5, dash="dot"),
                name="Historical Mean",
            )
        )

        fig.update_layout(
            title=f"{crop} — Forecast Yield by State with Uncertainty Bands",
            xaxis_title="State",
            yaxis_title="Yield (bu/ac)",
            legend_title="Scenario",
            height=600,
        )
        fig.write_html(str(OUTPUT_DIR / f"fan_chart_{crop.lower()}.html"))
        log.info("Saved fan chart for %s", crop)

    print("Fan charts saved.")

except ImportError as e:
    log.warning("plotly not available: %s", e)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Choropleth map — Forecast P50 with risk colouring

# COMMAND ----------

try:
    import folium
    import requests

    COUNTY_GEOJSON_URL = (
        "https://raw.githubusercontent.com/plotly/datasets/master/"
        "geojson-counties-fips.json"
    )

    risk_colours = {
        "NORMAL": "#43A047",
        "WARNING": "#FDD835",
        "ALERT": "#FB8C00",
        "CRITICAL": "#E53935",
    }

    for crop in ["Corn", "Soybeans"]:
        baseline = forecast_df[
            (forecast_df["scenario"] == "baseline") & (forecast_df["commodity_name"] == crop)
        ][["fips", "yield_p50", "drop_pct_vs_hist", "risk_level", "county_name", "state_abbr"]].copy()

        if baseline.empty:
            continue

        try:
            resp = requests.get(COUNTY_GEOJSON_URL, timeout=30)
            geojson = resp.json()
        except Exception:
            log.warning("GeoJSON not available — skipping map.")
            break

        m = folium.Map(location=[39.5, -98.35], zoom_start=4, tiles="CartoDB positron")

        folium.Choropleth(
            geo_data=geojson,
            name=f"{crop} Forecast Yield P50",
            data=baseline,
            columns=["fips", "yield_p50"],
            key_on="feature.id",
            fill_color="YlGn",
            fill_opacity=0.7,
            line_opacity=0.1,
            legend_name=f"{crop} Forecast P50 (bu/ac)",
        ).add_to(m)

        # Risk-level markers
        for _, row in baseline.iterrows():
            if row["risk_level"] == "NORMAL":
                continue
            county_loc = forecast_df[forecast_df["fips"] == row["fips"]][["lat", "lon"]]
            if county_loc.empty:
                continue
            lat_v = county_loc.iloc[0]["lat"]
            lon_v = county_loc.iloc[0]["lon"]
            folium.CircleMarker(
                location=[lat_v, lon_v],
                radius=5,
                color=risk_colours.get(row["risk_level"], "gray"),
                fill=True,
                fill_opacity=0.7,
                tooltip=(
                    f"{row.get('county_name','?')}, {row.get('state_abbr','?')}<br>"
                    f"P50: {row['yield_p50']:.1f} bu/ac<br>"
                    f"Drop: {row['drop_pct_vs_hist']:.1f}%<br>"
                    f"Risk: {row['risk_level']}"
                ),
            ).add_to(m)

        fname = OUTPUT_DIR / f"forecast_map_{crop.lower()}.html"
        m.save(str(fname))
        log.info("Saved forecast map: %s", fname)

    print("Forecast maps saved.")

except ImportError as e:
    log.warning("folium not available: %s", e)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Sensitivity tornado chart — yield variance attribution

# COMMAND ----------

try:
    import plotly.graph_objects as go

    for crop in ["Corn", "Soybeans"]:
        crop_fc = forecast_df[forecast_df["commodity_name"] == crop]
        if crop_fc.empty:
            continue

        # Compute mean P50 per scenario vs baseline
        baseline_p50 = crop_fc[crop_fc["scenario"] == "baseline"]["yield_p50"].mean()
        scenario_deltas = []
        for scen in ["drought", "heat_wave", "excess_rain", "ideal"]:
            scen_p50 = crop_fc[crop_fc["scenario"] == scen]["yield_p50"].mean()
            delta = scen_p50 - baseline_p50
            scenario_deltas.append((scen.replace("_", " ").title(), delta))

        scenario_deltas.sort(key=lambda x: x[1])
        labels = [s[0] for s in scenario_deltas]
        deltas = [s[1] for s in scenario_deltas]
        colours = ["#F44336" if d < 0 else "#4CAF50" for d in deltas]

        fig = go.Figure(
            go.Bar(
                y=labels, x=deltas, orientation="h",
                marker_color=colours,
                text=[f"{d:+.1f} bu/ac" for d in deltas],
                textposition="outside",
            )
        )
        fig.add_vline(x=0, line_color="black", line_width=1)
        fig.update_layout(
            title=f"{crop} — Yield Sensitivity to Weather Scenarios<br>(vs. Baseline Forecast)",
            xaxis_title="Change in Median Yield (bu/ac)",
            height=400,
        )
        fig.write_html(str(OUTPUT_DIR / f"tornado_{crop.lower()}.html"))

    print("Tornado charts saved.")

except ImportError as e:
    log.warning("plotly not available: %s", e)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Validation backtest — predicted vs actual 2023

# COMMAND ----------

try:
    import joblib
    import plotly.graph_objects as go
    from utils.feature_engineering import get_feature_cols

    feat_cols = get_feature_cols(feature_df)

    for crop in ["Corn", "Soybeans"]:
        model_path = Path(__file__).parent / "models" / f"xgb_{crop.lower()}.pkl"
        if not model_path.exists():
            continue
        model = joblib.load(model_path)

        val_df = feature_df[
            (feature_df["commodity_name"] == crop) & (feature_df["year"] == 2023)
        ].dropna(subset=feat_cols + ["yield_bu_ac"])

        if val_df.empty:
            continue

        X_val = val_df[feat_cols].fillna(0).values
        y_val = val_df["yield_bu_ac"].values
        y_pred = model.predict(X_val)
        residuals = y_val - y_pred

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_val, y=y_pred,
            mode="markers", marker=dict(color="#2196F3", opacity=0.5, size=6),
            name="County",
            hovertext=val_df["county_name"].astype(str) + ", " + val_df["state_abbr"].astype(str),
        ))
        # 45-degree perfect-fit line
        mn, mx = min(y_val.min(), y_pred.min()), max(y_val.max(), y_pred.max())
        fig.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx],
            mode="lines", line=dict(color="red", dash="dash"),
            name="Perfect fit",
        ))
        rmse = float(np.sqrt(np.mean(residuals**2)))
        r2 = float(1 - np.var(residuals) / np.var(y_val))
        fig.update_layout(
            title=f"{crop} 2023 Validation — Predicted vs Actual<br>RMSE={rmse:.1f} bu/ac | R²={r2:.3f}",
            xaxis_title="Actual Yield (bu/ac)",
            yaxis_title="Predicted Yield (bu/ac)",
            height=550,
        )
        fig.write_html(str(OUTPUT_DIR / f"validation_{crop.lower()}.html"))
        log.info("%s 2023 validation: RMSE=%.1f, R²=%.3f", crop, rmse, r2)

    print("Validation charts saved.")

except Exception as e:
    log.warning("Validation chart failed: %s", e)

print(f"\n=== Notebook 04 complete — outputs in {OUTPUT_DIR.resolve()} ===")
print("Proceed to 05_nlp_summaries.py")
