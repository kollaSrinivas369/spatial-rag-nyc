"""
spatial_engine.py
─────────────────
Executes structured spatial query plans against the NYC neighborhood GeoDataFrame.

Uses GeoPandas for attribute queries and Shapely for spatial operations.
All distance calculations reproject to EPSG:32618 (UTM Zone 18N, meters)
so that distance_km values translate correctly.

The engine accepts a dict query plan (produced by llm_bridge.parse_query)
and returns a result dict with matching features, overlays, and context.
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon
import numpy as np
from typing import Optional

# UTM Zone 18N — meters, covers all of NYC
METRIC_CRS = "EPSG:32618"
# WGS 84 (lat/lon)
WGS84_CRS = "EPSG:4326"


# ── Data loading ──────────────────────────────────────────────────────


def load_data(parquet_path: str) -> gpd.GeoDataFrame:
    """Load the GeoParquet file and prepare it for queries."""
    gdf = gpd.read_parquet(parquet_path)
    # Only keep residential neighborhoods (ntatype '0') that have hydrants
    gdf = gdf[gdf["ntatype"] == "0"].copy()
    gdf["area_km2"] = gdf["area_km2"].round(2)
    gdf["hydrants_per_km2"] = gdf["hydrants_per_km2"].round(1)
    # Ensure WGS84
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(WGS84_CRS)
    return gdf


# ── Query plan execution ─────────────────────────────────────────────


def execute_query_plan(gdf: gpd.GeoDataFrame, plan: dict) -> dict:
    """
    Execute a structured query plan against the GeoDataFrame.

    Args:
        gdf: Full neighborhood GeoDataFrame (WGS84).
        plan: Structured query plan dict with optional keys:
              spatial_op, filters, sort, limit, aggregate.

    Returns:
        dict with:
            results        – GeoDataFrame of matching features
            answer_context – metadata for answer generation
            overlay_geoms  – Shapely geometries to render (buffers, circles)
            center_point   – optional (lat, lon) to center the map
            aggregation    – optional aggregation result
    """
    result = gdf.copy()
    overlay_geoms: list = []
    center_point: Optional[tuple] = None
    answer_context: dict = {}

    # ── 1. Spatial operation (defines the working set) ────────────────
    spatial_op = plan.get("spatial_op")
    if spatial_op:
        op_type = spatial_op.get("type")

        if op_type == "nearest":
            lat, lon = spatial_op["lat"], spatial_op["lon"]
            n = spatial_op.get("n", 5)
            center_point = (lat, lon)
            result = _nearest(result, lat, lon, n)
            answer_context["spatial_desc"] = (
                f"nearest {n} neighborhoods to ({lat:.4f}, {lon:.4f})"
            )

        elif op_type == "within_distance":
            lat, lon = spatial_op["lat"], spatial_op["lon"]
            distance_km = spatial_op["distance_km"]
            center_point = (lat, lon)
            result, buffer_geom = _within_distance(result, lat, lon, distance_km)
            overlay_geoms.append(buffer_geom)
            answer_context["spatial_desc"] = (
                f"within {distance_km} km of ({lat:.4f}, {lon:.4f})"
            )

        elif op_type == "within_polygon":
            polygon_coords = spatial_op["polygon"]  # [[lon, lat], ...]
            result, poly_geom = _within_polygon(result, polygon_coords)
            overlay_geoms.append(poly_geom)
            answer_context["spatial_desc"] = "within drawn polygon"

        elif op_type == "buffer":
            neighborhood_name = spatial_op["neighborhood"]
            distance_km = spatial_op["distance_km"]
            result, buffer_geom, center = _buffer_neighborhood(
                gdf, result, neighborhood_name, distance_km
            )
            if buffer_geom:
                overlay_geoms.append(buffer_geom)
            if center:
                center_point = center
            answer_context["spatial_desc"] = (
                f"within {distance_km} km of {neighborhood_name}"
            )

    # ── 2. Attribute filters ──────────────────────────────────────────
    for f in plan.get("filters") or []:
        result = _apply_filter(result, f)

    # ── 3. Sort ───────────────────────────────────────────────────────
    sort_spec = plan.get("sort")
    if sort_spec:
        col = sort_spec["by"]
        ascending = sort_spec.get("order", "desc") == "asc"
        if col in result.columns:
            result = result.sort_values(col, ascending=ascending)

    # ── 4. Limit ──────────────────────────────────────────────────────
    limit = plan.get("limit")
    if limit and limit.get("n"):
        result = result.head(limit["n"])

    # ── 5. Aggregate ──────────────────────────────────────────────────
    agg_result = None
    aggregate = plan.get("aggregate")
    if aggregate:
        agg_result = _aggregate(result, aggregate)
        answer_context["aggregation"] = agg_result

    return {
        "results": result,
        "answer_context": answer_context,
        "overlay_geoms": overlay_geoms,
        "center_point": center_point,
        "aggregation": agg_result,
    }


# ── Spatial operations ────────────────────────────────────────────────


def _nearest(
    gdf: gpd.GeoDataFrame, lat: float, lon: float, n: int
) -> gpd.GeoDataFrame:
    """Find N nearest neighborhoods to a point (by centroid distance)."""
    point = Point(lon, lat)
    gdf_metric = gdf.to_crs(METRIC_CRS)
    point_metric = (
        gpd.GeoSeries([point], crs=WGS84_CRS).to_crs(METRIC_CRS).iloc[0]
    )

    gdf = gdf.copy()
    gdf["_dist_m"] = gdf_metric.geometry.centroid.distance(point_metric)
    gdf["distance_km"] = (gdf["_dist_m"] / 1000).round(2)
    gdf = gdf.sort_values("_dist_m").head(n).drop(columns=["_dist_m"])
    return gdf


def _within_distance(
    gdf: gpd.GeoDataFrame, lat: float, lon: float, distance_km: float
):
    """Find neighborhoods whose geometry intersects a buffer circle."""
    point = Point(lon, lat)
    gdf_metric = gdf.to_crs(METRIC_CRS)
    point_metric = (
        gpd.GeoSeries([point], crs=WGS84_CRS).to_crs(METRIC_CRS).iloc[0]
    )

    buffer_metric = point_metric.buffer(distance_km * 1000)
    # Convert buffer back to WGS84 for map overlay
    buffer_wgs84 = (
        gpd.GeoSeries([buffer_metric], crs=METRIC_CRS).to_crs(WGS84_CRS).iloc[0]
    )

    mask = gdf_metric.geometry.intersects(buffer_metric)
    result = gdf[mask].copy()
    result["distance_km"] = (
        gdf_metric[mask].geometry.centroid.distance(point_metric) / 1000
    ).round(2)
    result = result.sort_values("distance_km")

    return result, buffer_wgs84


def _within_polygon(gdf: gpd.GeoDataFrame, polygon_coords: list):
    """Find neighborhoods intersecting a user-drawn polygon."""
    poly = Polygon(polygon_coords)  # [[lon, lat], ...]
    mask = gdf.geometry.intersects(poly)
    return gdf[mask].copy(), poly


def _buffer_neighborhood(
    full_gdf: gpd.GeoDataFrame,
    gdf: gpd.GeoDataFrame,
    neighborhood_name: str,
    distance_km: float,
):
    """Find neighborhoods within distance_km of a named neighborhood."""
    # Exact match first, then fuzzy
    source = full_gdf[
        full_gdf["ntaname"].str.lower() == neighborhood_name.lower()
    ]
    if len(source) == 0:
        source = full_gdf[
            full_gdf["ntaname"]
            .str.lower()
            .str.contains(neighborhood_name.lower(), na=False)
        ]
    if len(source) == 0:
        return gdf.head(0), None, None  # empty result

    source_geom = source.geometry.iloc[0]
    source_metric = (
        gpd.GeoSeries([source_geom], crs=WGS84_CRS).to_crs(METRIC_CRS).iloc[0]
    )

    buffer_metric = source_metric.buffer(distance_km * 1000)
    buffer_wgs84 = (
        gpd.GeoSeries([buffer_metric], crs=METRIC_CRS).to_crs(WGS84_CRS).iloc[0]
    )

    gdf_metric = gdf.to_crs(METRIC_CRS)
    mask = gdf_metric.geometry.intersects(buffer_metric)
    result = gdf[mask].copy()

    center_lat = source_geom.centroid.y
    center_lon = source_geom.centroid.x

    return result, buffer_wgs84, (center_lat, center_lon)


# ── Attribute filtering ──────────────────────────────────────────────


def _apply_filter(gdf: gpd.GeoDataFrame, filter_spec: dict) -> gpd.GeoDataFrame:
    """Apply a single attribute filter."""
    col = filter_spec.get("column")
    if not col or col not in gdf.columns:
        return gdf

    op = filter_spec.get("op", "eq")
    value = filter_spec.get("value")

    ops = {
        "eq": lambda s, v: s == v,
        "neq": lambda s, v: s != v,
        "gt": lambda s, v: s > v,
        "gte": lambda s, v: s >= v,
        "lt": lambda s, v: s < v,
        "lte": lambda s, v: s <= v,
        "in": lambda s, v: s.isin(v),
        "between": lambda s, v: (s >= v[0]) & (s <= v[1]),
        "contains": lambda s, v: s.str.lower().str.contains(
            str(v).lower(), na=False
        ),
    }

    fn = ops.get(op)
    if fn:
        return gdf[fn(gdf[col], value)]
    return gdf


# ── Aggregation ──────────────────────────────────────────────────────


def _aggregate(gdf: gpd.GeoDataFrame, agg_spec: dict):
    """Compute aggregate statistics, optionally grouped."""
    col = agg_spec.get("column", "hydrants_per_km2")
    op = agg_spec.get("op", "mean")
    group_by = agg_spec.get("group_by")

    if col not in gdf.columns:
        return {"error": f"Column '{col}' not found"}

    series = gdf.groupby(group_by)[col] if (group_by and group_by in gdf.columns) else gdf[col]

    agg_fns = {
        "sum": lambda s: s.sum(),
        "mean": lambda s: s.mean(),
        "min": lambda s: s.min(),
        "max": lambda s: s.max(),
        "count": lambda s: s.count() if hasattr(s, "count") else len(s),
        "std": lambda s: s.std(),
    }
    result = agg_fns.get(op, agg_fns["mean"])(series)

    if isinstance(result, pd.Series):
        return result.round(2).to_dict()
    return round(float(result), 2)
