# Databricks notebook source
# Option 1 — Notebook 06: Create AI/BI Dashboard
#
# Creates a Databricks Lakeview (AI/BI) dashboard with four pages:
#   1. Feature Importance  — SHAP importance by crop
#   2. Correlations        — Pearson / Spearman heatmap
#   3. Yield Trends        — actual yield over time, coloured by crop
#   4. Anomalies           — anomaly events table + map
#
# Requires: notebooks 01–04 to have been run (Delta tables must exist).
# Run this notebook ONCE to create the dashboard; it is idempotent
# (re-running updates the spec in-place rather than creating duplicates).

# COMMAND ----------

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATABRICKS_CATALOG,
    DATABRICKS_SCHEMA,
    DATABRICKS_CORR_TABLE,
    DATABRICKS_IMPORTANCE_TABLE,
    DATABRICKS_MERGED_TABLE,
)

# Full table references
ANOMALY_TABLE = f"{DATABRICKS_CATALOG}.{DATABRICKS_SCHEMA}.anomalies"

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Verify tables exist

# COMMAND ----------

tables = {
    "merged":      DATABRICKS_MERGED_TABLE,
    "correlations": DATABRICKS_CORR_TABLE,
    "importance":  DATABRICKS_IMPORTANCE_TABLE,
    "anomalies":   ANOMALY_TABLE,
}

missing = []
for name, tbl in tables.items():
    try:
        spark.sql(f"SELECT 1 FROM {tbl} LIMIT 1")
        print(f"  ✓ {tbl}")
    except Exception:
        print(f"  ✗ {tbl}  — run notebooks 01–04 first")
        missing.append(name)

if missing:
    raise RuntimeError(
        f"Missing tables: {missing}\n"
        "Run notebooks 01–04 before creating the dashboard."
    )

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Build dashboard spec

# COMMAND ----------

DASHBOARD_NAME = "Weather-to-Yield Signal Detection"

# ── Datasets ──────────────────────────────────────────────────────────────────
datasets = [
    {
        "name": "ds_importance",
        "displayName": "Feature Importance",
        "query": (
            f"SELECT crop, feature, shap_mean_abs, xgb_importance "
            f"FROM {DATABRICKS_IMPORTANCE_TABLE} "
            f"ORDER BY crop, shap_mean_abs DESC"
        ),
    },
    {
        "name": "ds_correlations",
        "displayName": "Correlations",
        "query": (
            f"SELECT crop, feature, pearson_r, spearman_r, abs_pearson "
            f"FROM {DATABRICKS_CORR_TABLE} "
            f"ORDER BY crop, abs_pearson DESC"
        ),
    },
    {
        "name": "ds_yield_trend",
        "displayName": "Yield Trends",
        "query": (
            f"SELECT year, commodity_name, "
            f"  ROUND(AVG(yield_bu_ac), 1) AS avg_yield_bu_ac, "
            f"  ROUND(AVG(gdd_base10), 0)  AS avg_gdd, "
            f"  ROUND(AVG(heat_stress_days), 1) AS avg_heat_days "
            f"FROM {DATABRICKS_MERGED_TABLE} "
            f"GROUP BY year, commodity_name "
            f"ORDER BY year"
        ),
    },
    {
        "name": "ds_anomalies",
        "displayName": "Anomalies",
        "query": (
            f"SELECT year, fips, county_name, state_abbr, commodity_name, "
            f"  yield_bu_ac, xgb_predicted_yield, xgb_residual, "
            f"  zscore_yield, anomaly_type, explanation "
            f"FROM {ANOMALY_TABLE} "
            f"WHERE is_anomaly = TRUE "
            f"ORDER BY year DESC, ABS(xgb_residual) DESC "
            f"LIMIT 500"
        ),
    },
]

# ── Widget helpers ─────────────────────────────────────────────────────────────
def _bar(name, title, dataset, x_field, y_field, color_field=None,
         x_label=None, y_label=None, position=None):
    encodings = {
        "x": {"fieldName": x_field, "scale": {"type": "quantitative"},
              "displayName": x_label or x_field},
        "y": {"fieldName": y_field, "scale": {"type": "categorical"},
              "displayName": y_label or y_field},
    }
    if color_field:
        encodings["color"] = {"fieldName": color_field,
                               "scale": {"type": "categorical"}}
    return {
        "widget": {
            "name": name,
            "queries": [{
                "name": "q",
                "query": {
                    "datasetName": dataset,
                    "fields": [
                        {"name": f} for f in
                        ([x_field, y_field] + ([color_field] if color_field else []))
                    ],
                    "disaggregated": False,
                },
            }],
            "spec": {
                "version": 2,
                "widgetType": "bar",
                "encodings": encodings,
                "frame": {"title": title, "showTitle": True},
            },
        },
        "position": position or {"x": 0, "y": 0, "width": 3, "height": 8},
    }


def _line(name, title, dataset, x_field, y_field, color_field=None,
          x_label=None, y_label=None, position=None):
    encodings = {
        "x": {"fieldName": x_field, "scale": {"type": "quantitative"},
              "displayName": x_label or x_field},
        "y": {"fieldName": y_field, "scale": {"type": "quantitative"},
              "displayName": y_label or y_field},
    }
    if color_field:
        encodings["color"] = {"fieldName": color_field,
                               "scale": {"type": "categorical"}}
    return {
        "widget": {
            "name": name,
            "queries": [{
                "name": "q",
                "query": {
                    "datasetName": dataset,
                    "fields": [
                        {"name": f} for f in
                        ([x_field, y_field] + ([color_field] if color_field else []))
                    ],
                    "disaggregated": False,
                },
            }],
            "spec": {
                "version": 2,
                "widgetType": "line",
                "encodings": encodings,
                "frame": {"title": title, "showTitle": True},
            },
        },
        "position": position or {"x": 0, "y": 0, "width": 6, "height": 6},
    }


def _table(name, title, dataset, fields, position=None):
    return {
        "widget": {
            "name": name,
            "queries": [{
                "name": "q",
                "query": {
                    "datasetName": dataset,
                    "fields": [{"name": f} for f in fields],
                    "disaggregated": True,
                },
            }],
            "spec": {
                "version": 2,
                "widgetType": "table",
                "encodings": {
                    "columns": [{"fieldName": f, "displayName": f} for f in fields],
                },
                "frame": {"title": title, "showTitle": True},
            },
        },
        "position": position or {"x": 0, "y": 0, "width": 6, "height": 8},
    }


# ── Pages ──────────────────────────────────────────────────────────────────────
pages = [
    # Page 1 — Feature Importance
    {
        "name": "page_importance",
        "displayName": "Feature Importance",
        "layout": [
            _bar(
                name="shap_corn",
                title="Corn — SHAP Feature Importance",
                dataset="ds_importance",
                x_field="shap_mean_abs", x_label="Mean |SHAP| (bu/ac)",
                y_field="feature",
                color_field="crop",
                position={"x": 0, "y": 0, "width": 3, "height": 8},
            ),
            _bar(
                name="shap_both",
                title="Corn vs Soybeans — Importance Comparison",
                dataset="ds_importance",
                x_field="shap_mean_abs", x_label="Mean |SHAP| (bu/ac)",
                y_field="feature",
                color_field="crop",
                position={"x": 3, "y": 0, "width": 3, "height": 8},
            ),
        ],
    },
    # Page 2 — Correlations
    {
        "name": "page_correlations",
        "displayName": "Correlations",
        "layout": [
            _bar(
                name="pearson_corn",
                title="Pearson r — Weather vs Yield",
                dataset="ds_correlations",
                x_field="pearson_r", x_label="Pearson r",
                y_field="feature",
                color_field="crop",
                position={"x": 0, "y": 0, "width": 3, "height": 8},
            ),
            _bar(
                name="spearman_corn",
                title="Spearman r — Weather vs Yield",
                dataset="ds_correlations",
                x_field="spearman_r", x_label="Spearman r",
                y_field="feature",
                color_field="crop",
                position={"x": 3, "y": 0, "width": 3, "height": 8},
            ),
        ],
    },
    # Page 3 — Yield Trends
    {
        "name": "page_trends",
        "displayName": "Yield Trends",
        "layout": [
            _line(
                name="yield_trend",
                title="Average Yield by Year",
                dataset="ds_yield_trend",
                x_field="year",        x_label="Year",
                y_field="avg_yield_bu_ac", y_label="Avg Yield (bu/ac)",
                color_field="commodity_name",
                position={"x": 0, "y": 0, "width": 4, "height": 6},
            ),
            _line(
                name="gdd_trend",
                title="Growing Degree Days by Year",
                dataset="ds_yield_trend",
                x_field="year",     x_label="Year",
                y_field="avg_gdd",  y_label="Avg GDD (base 10°C)",
                color_field="commodity_name",
                position={"x": 4, "y": 0, "width": 2, "height": 6},
            ),
            _line(
                name="heat_trend",
                title="Heat Stress Days by Year",
                dataset="ds_yield_trend",
                x_field="year",           x_label="Year",
                y_field="avg_heat_days",  y_label="Avg Heat Stress Days",
                color_field="commodity_name",
                position={"x": 0, "y": 6, "width": 3, "height": 6},
            ),
        ],
    },
    # Page 4 — Anomalies
    {
        "name": "page_anomalies",
        "displayName": "Anomalies",
        "layout": [
            _table(
                name="anomaly_table",
                title="Detected Yield Anomalies",
                dataset="ds_anomalies",
                fields=[
                    "year", "state_abbr", "county_name", "commodity_name",
                    "yield_bu_ac", "xgb_predicted_yield", "xgb_residual",
                    "anomaly_type", "explanation",
                ],
                position={"x": 0, "y": 0, "width": 6, "height": 10},
            ),
            _bar(
                name="anomaly_by_year",
                title="Anomaly Count by Year",
                dataset="ds_anomalies",
                x_field="year",            x_label="Year",
                y_field="xgb_residual",    y_label="Residual (bu/ac)",
                color_field="anomaly_type",
                position={"x": 0, "y": 10, "width": 3, "height": 6},
            ),
            _bar(
                name="anomaly_by_state",
                title="Anomaly Residual by State",
                dataset="ds_anomalies",
                x_field="xgb_residual",   x_label="XGB Residual (bu/ac)",
                y_field="state_abbr",     y_label="State",
                color_field="commodity_name",
                position={"x": 3, "y": 10, "width": 3, "height": 6},
            ),
        ],
    },
]

dashboard_spec = {"pages": pages, "datasets": datasets}

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Create or update dashboard via REST API

# COMMAND ----------

import requests

# Credentials available automatically inside a Databricks notebook
ctx   = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
host  = ctx.apiUrl().get()
token = ctx.apiToken().get()

headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
base    = f"{host}/api/2.0/lakeview/dashboards"

serialized = json.dumps(dashboard_spec)

# Check if dashboard already exists (avoid duplicates on re-run)
existing_id = None
resp = requests.get(base, headers=headers)
if resp.ok:
    for d in resp.json().get("dashboards", []):
        if d.get("display_name") == DASHBOARD_NAME:
            existing_id = d["dashboard_id"]
            break

if existing_id:
    # Update existing
    resp = requests.patch(
        f"{base}/{existing_id}",
        headers=headers,
        json={"display_name": DASHBOARD_NAME,
              "serialized_dashboard": serialized},
    )
    action = "Updated"
else:
    # Create new
    resp = requests.post(
        base,
        headers=headers,
        json={"display_name": DASHBOARD_NAME,
              "serialized_dashboard": serialized},
    )
    action = "Created"

if resp.ok:
    dashboard_id  = resp.json()["dashboard_id"]
    dashboard_url = f"{host}/dashboardsv3/{dashboard_id}"
    print(f"\n✓ {action} dashboard: {DASHBOARD_NAME}")
    print(f"  URL: {dashboard_url}")
    print(f"\nTo publish it (make it viewable without a cluster):")
    print(f"  Open the URL above → click 'Publish' in the top-right corner.")
else:
    print(f"✗ API call failed ({resp.status_code}): {resp.text}")
    print("\nManual fallback — create the dashboard via the UI:")
    print("  1. Left sidebar → Dashboards → Create dashboard")
    print("  2. Add these datasets:")
    for ds in datasets:
        print(f"\n     [{ds['displayName']}]")
        print(f"     {ds['query']}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Publish (make readable without a cluster)

# COMMAND ----------

if resp.ok:
    pub = requests.post(
        f"{base}/{dashboard_id}/published",
        headers=headers,
        json={"embed_credentials": False, "warehouse_id": ""},
    )
    if pub.ok:
        print(f"✓ Dashboard published — shareable at:\n  {dashboard_url}")
        print("\n  Anyone in the workspace can view it without needing a cluster.")
        print("  It runs on a SQL Warehouse instead.")
    else:
        # warehouse_id required in some workspaces — list available ones and retry
        wh_resp = requests.get(f"{host}/api/2.0/sql/warehouses", headers=headers)
        if wh_resp.ok:
            warehouses = wh_resp.json().get("warehouses", [])
            if warehouses:
                wh_id = warehouses[0]["id"]
                pub2  = requests.post(
                    f"{base}/{dashboard_id}/published",
                    headers=headers,
                    json={"embed_credentials": False, "warehouse_id": wh_id},
                )
                if pub2.ok:
                    print(f"✓ Published using warehouse: {warehouses[0]['name']}")
                    print(f"  URL: {dashboard_url}")
                else:
                    print(f"Publishing requires manual step — open {dashboard_url}")
                    print(f"and click 'Publish' in the top-right corner.")
            else:
                print("No SQL Warehouses found. Create one in Compute → SQL Warehouses,")
                print(f"then publish the dashboard manually at:\n  {dashboard_url}")
        else:
            print(f"Could not list warehouses. Publish manually at:\n  {dashboard_url}")
