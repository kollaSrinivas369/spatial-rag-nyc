"""
map_utils.py
────────────
Folium map rendering for the SpatialRAG app.

Provides choropleth maps with:
  - YlOrRd color ramp (matching lessons 2.5 / 3.3)
  - Optional highlighted result subset
  - Overlay geometries (buffer circles, drawn polygons)
  - Folium Draw plugin for interactive drawing
  - Legend HTML generation
"""

import folium
from folium.plugins import Draw
import geopandas as gpd
import numpy as np
from shapely.geometry import mapping
from typing import Optional


# ── Color scheme (step breaks matching the legend) ────────────────────

COLOR_BREAKS = [
    (0, 100, "#ffffcc"),
    (100, 150, "#fed976"),
    (150, 200, "#fd8d3c"),
    (200, 300, "#e31a1c"),
    (300, float("inf"), "#800026"),
]


def _get_color(density: float) -> str:
    """Map a density value to a YlOrRd step color."""
    for low, high, color in COLOR_BREAKS:
        if low <= density < high:
            return color
    return COLOR_BREAKS[-1][2]


# ── Map creation ─────────────────────────────────────────────────────


def create_base_map(
    gdf: gpd.GeoDataFrame,
    highlight_gdf: Optional[gpd.GeoDataFrame] = None,
    center: Optional[tuple] = None,
    zoom: int = 11,
    overlay_geoms: Optional[list] = None,
    point: Optional[tuple] = None,
    enable_draw: bool = False,
    drawn_features: Optional[list] = None,
) -> folium.Map:
    """
    Build an interactive Folium choropleth map.

    Args:
        gdf:              Full neighborhood GeoDataFrame (background).
        highlight_gdf:    Subset to highlight with bold outlines.
        center:           (lat, lon) to center on.
        zoom:             Initial zoom level.
        overlay_geoms:    Shapely geometries to overlay (buffers, etc.).
        point:            (lat, lon) for a query-center marker.
        enable_draw:      Add Folium Draw tools.
        drawn_features:   Previously drawn GeoJSON features to persist
                          visually across Streamlit reruns.
    """
    # ── Center ────────────────────────────────────────────────────────
    if center:
        center_lat, center_lon = center
    else:
        bounds = gdf.total_bounds
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles="CartoDB positron",
    )

    # ── Background choropleth ─────────────────────────────────────────
    dim = highlight_gdf is not None and len(highlight_gdf) > 0

    for _, row in gdf.iterrows():
        color = _get_color(row["hydrants_per_km2"])
        opacity = 0.25 if dim else 0.7

        geo_json = gpd.GeoSeries([row["geometry"]]).__geo_interface__

        tooltip = None
        if not dim:
            tooltip = folium.Tooltip(
                f"<b>{row['ntaname']}</b><br>"
                f"{row['boroname']}<br>"
                f"Density: {row['hydrants_per_km2']} / km²"
            )

        folium.GeoJson(
            geo_json,
            style_function=lambda feature, c=color, o=opacity: {
                "fillColor": c,
                "color": "#999" if o < 0.5 else "#333",
                "weight": 0.3 if o < 0.5 else 0.5,
                "fillOpacity": o,
            },
            tooltip=tooltip,
        ).add_to(m)

    # ── Highlighted results ───────────────────────────────────────────
    if dim:
        for _, row in highlight_gdf.iterrows():
            color = _get_color(row["hydrants_per_km2"])
            geo_json = gpd.GeoSeries([row["geometry"]]).__geo_interface__

            tip = (
                f"<b>{row['ntaname']}</b><br>"
                f"{row['boroname']}<br>"
                f"Hydrants: {row['hydrant_count']}<br>"
                f"Area: {row['area_km2']} km²<br>"
                f"Density: {row['hydrants_per_km2']} / km²"
            )
            if "distance_km" in row.index:
                dist = row.get("distance_km")
                if dist is not None and not (isinstance(dist, float) and np.isnan(dist)):
                    tip += f"<br>Distance: {dist} km"

            folium.GeoJson(
                geo_json,
                style_function=lambda feature, c=color: {
                    "fillColor": c,
                    "color": "#000",
                    "weight": 2,
                    "fillOpacity": 0.8,
                },
                tooltip=folium.Tooltip(tip),
            ).add_to(m)

    # ── Overlay geometries (buffers, circles) ─────────────────────────
    if overlay_geoms:
        for geom in overlay_geoms:
            folium.GeoJson(
                mapping(geom),
                style_function=lambda feature: {
                    "fillColor": "#3388ff",
                    "color": "#3388ff",
                    "weight": 2,
                    "fillOpacity": 0.1,
                    "dashArray": "5, 5",
                },
                interactive=False,
            ).add_to(m)

    # ── Query-center marker ───────────────────────────────────────────
    if point:
        folium.Marker(
            location=[point[0], point[1]],
            icon=folium.Icon(color="red", icon="info-sign"),
            popup="Query center",
        ).add_to(m)

    # ── Re-render previously drawn shapes (persist across reruns) ─────
    if drawn_features:
        for feat in drawn_features:
            if feat and feat.get("geometry"):
                folium.GeoJson(
                    feat,
                    style_function=lambda feature: {
                        "fillColor": "#ff7800",
                        "color": "#ff7800",
                        "weight": 2,
                        "fillOpacity": 0.15,
                    },
                    interactive=False,
                ).add_to(m)

    # ── Draw tools ────────────────────────────────────────────────────
    if enable_draw:
        Draw(
            draw_options={
                "polyline": False,
                "rectangle": True,
                "polygon": True,
                "circle": True,
                "marker": True,
                "circlemarker": False,
            },
            edit_options={"edit": False},
        ).add_to(m)

    # ── Fit bounds to highlighted results ─────────────────────────────
    if dim:
        bounds = highlight_gdf.total_bounds
        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    return m


# ── Legend ────────────────────────────────────────────────────────────


def create_legend_html() -> str:
    """Generate an HTML legend matching the YlOrRd color ramp."""
    items = [
        ("#ffffcc", "< 100"),
        ("#fed976", "100 – 150"),
        ("#fd8d3c", "150 – 200"),
        ("#e31a1c", "200 – 300"),
        ("#800026", "300+"),
    ]

    html = '<div style="font-size:13px;"><b>Hydrants per km²</b><br>'
    for color, label in items:
        html += (
            f'<div style="display:flex;align-items:center;margin:2px 0;">'
            f'<div style="width:18px;height:12px;background:{color};'
            f'border:1px solid #ccc;margin-right:6px;border-radius:2px;"></div>'
            f"{label}</div>"
        )
    html += "</div>"
    return html
