"""
llm_bridge.py
─────────────
Bridges natural language queries to structured query plans via OpenAI's GPT models.

Uses OpenAI's Structured Outputs (with a Pydantic response_format) so the
spatial engine always receives structurally valid input — no preamble text or
markdown fences to strip.

Includes a known_places lookup for common NYC landmarks so the LLM can
resolve place names to lat/lon without an external geocoder.
"""

import json
from typing import Optional, List, Any, Union

from pydantic import BaseModel, Field
from openai import OpenAI
import geopandas as gpd


# ── Known NYC landmarks ──────────────────────────────────────────────
# Lets the LLM resolve place names without an external geocoder.
# For anything not listed, Gemini estimates coords from training data
# (or geopy/Nominatim can be added as a fallback later).

KNOWN_PLACES = {
    "times square": {"lat": 40.7580, "lon": -73.9855},
    "central park": {"lat": 40.7829, "lon": -73.9654},
    "empire state building": {"lat": 40.7484, "lon": -73.9857},
    "statue of liberty": {"lat": 40.6892, "lon": -74.0445},
    "brooklyn bridge": {"lat": 40.7061, "lon": -73.9969},
    "one world trade center": {"lat": 40.7127, "lon": -74.0134},
    "yankee stadium": {"lat": 40.8296, "lon": -73.9262},
    "citi field": {"lat": 40.7571, "lon": -73.8458},
    "jfk airport": {"lat": 40.6413, "lon": -73.7781},
    "laguardia airport": {"lat": 40.7769, "lon": -73.8740},
    "grand central terminal": {"lat": 40.7527, "lon": -73.9772},
    "penn station": {"lat": 40.7506, "lon": -73.9935},
    "coney island": {"lat": 40.5749, "lon": -73.9708},
    "prospect park": {"lat": 40.6602, "lon": -73.9690},
    "wall street": {"lat": 40.7074, "lon": -74.0113},
    "columbia university": {"lat": 40.8075, "lon": -73.9626},
    "nyu": {"lat": 40.7295, "lon": -73.9965},
    "madison square garden": {"lat": 40.7505, "lon": -73.9934},
    "rockefeller center": {"lat": 40.7587, "lon": -73.9787},
    "battery park": {"lat": 40.7033, "lon": -74.0170},
    "flushing meadows": {"lat": 40.7400, "lon": -73.8407},
    "bronx zoo": {"lat": 40.8506, "lon": -73.8770},
    "staten island ferry": {"lat": 40.6433, "lon": -74.0735},
}


# ── Pydantic models for structured Gemini output ─────────────────────


class SpatialOp(BaseModel):
    """A single spatial operation that defines the working set."""

    type: str = Field(
        description="One of: nearest, within_distance, within_polygon, buffer"
    )
    lat: Optional[float] = Field(
        default=None, description="Latitude of center point"
    )
    lon: Optional[float] = Field(
        default=None, description="Longitude of center point"
    )
    n: Optional[int] = Field(
        default=None, description="Number of results (for nearest)"
    )
    distance_km: Optional[float] = Field(
        default=None, description="Distance in kilometers"
    )
    neighborhood: Optional[str] = Field(
        default=None, description="Neighborhood name (for buffer op)"
    )
    polygon: Optional[List[List[float]]] = Field(
        default=None, description="Polygon coords as [[lon,lat],...]"
    )


class FilterSpec(BaseModel):
    """An attribute filter applied after the spatial op."""

    column: str = Field(description="Column name to filter on")
    op: str = Field(
        description="One of: eq, neq, gt, gte, lt, lte, in, between, contains"
    )
    value: Union[str, float, List[str], List[float]] = Field(
        description="Filter value (scalar, list of values for 'in', [low, high] for 'between')"
    )


class SortSpec(BaseModel):
    by: str = Field(description="Column to sort by")
    order: str = Field(default="desc", description="asc or desc")


class LimitSpec(BaseModel):
    n: int = Field(description="Maximum number of results")


class AggregateSpec(BaseModel):
    column: str = Field(description="Column to aggregate")
    op: str = Field(description="One of: sum, mean, min, max, count, std")
    group_by: Optional[str] = Field(
        default=None, description="Column to group by (usually boroname)"
    )


class QueryPlan(BaseModel):
    """Structured query plan produced by Gemini."""

    spatial_op: Optional[SpatialOp] = Field(default=None)
    filters: Optional[List[FilterSpec]] = Field(default=None)
    sort: Optional[SortSpec] = Field(default=None)
    limit: Optional[LimitSpec] = Field(default=None)
    aggregate: Optional[AggregateSpec] = Field(default=None)
    explanation: str = Field(
        description="Brief explanation of what this query plan does"
    )


# ── System prompt builder ────────────────────────────────────────────


def _build_system_prompt(gdf: gpd.GeoDataFrame) -> str:
    """Build a system prompt containing the data schema, value ranges,
    known places, operation catalog, and worked examples."""

    boroughs = sorted(gdf["boroname"].unique().tolist())
    neighborhoods_sample = sorted(gdf["ntaname"].unique().tolist())[:10]
    density_range = (
        float(gdf["hydrants_per_km2"].min()),
        float(gdf["hydrants_per_km2"].max()),
    )
    count_range = (
        int(gdf["hydrant_count"].min()),
        int(gdf["hydrant_count"].max()),
    )
    area_range = (
        float(gdf["area_km2"].min()),
        float(gdf["area_km2"].max()),
    )

    places_block = "\n".join(
        f"  - {name}: lat={c['lat']}, lon={c['lon']}"
        for name, c in sorted(KNOWN_PLACES.items())
    )

    return f"""You are a spatial query planner for NYC fire hydrant density data.
Your ONLY job is to convert a natural language question into a JSON query plan.

## Data Schema
{len(gdf)} NYC neighborhoods with columns:
- ntaname (str): Neighborhood name.  Examples: {', '.join(neighborhoods_sample)}…
- boroname (str): Borough.  Values: {', '.join(boroughs)}
- hydrant_count (int): Fire hydrants.  Range: {count_range[0]}–{count_range[1]}
- area_km2 (float): Area in km².  Range: {area_range[0]}–{area_range[1]}
- hydrants_per_km2 (float): Density.  Range: {density_range[0]}–{density_range[1]}

## Known NYC Landmarks (use these for place-name queries)
{places_block}

## Available Operations

### spatial_op  (at most ONE per query)
| type             | required params                         |
|------------------|-----------------------------------------|
| nearest          | lat, lon, n                             |
| within_distance  | lat, lon, distance_km                   |
| buffer           | neighborhood (str), distance_km         |
| within_polygon   | polygon ([[lon,lat],...])                |

### filters  (list, applied after spatial_op)
column: one of [ntaname, boroname, hydrant_count, area_km2, hydrants_per_km2]
op: eq | neq | gt | gte | lt | lte | in | between | contains
value: scalar, list (for "in"), or [low, high] (for "between")

### sort
by: column name, order: "asc" | "desc"

### limit
n: max rows

### aggregate
column, op (sum|mean|min|max|count|std), optional group_by

## Rules
1. When a user mentions a place name, look it up in the Known Landmarks table
   and use its lat/lon.  If the place is NOT listed, estimate coordinates.
2. For "top N" queries → sort + limit.
3. For "compare A vs B" → aggregate with group_by and an appropriate filter.
   (There is no dedicated compare operation.)
4. Only include fields that are needed.  Omit null/empty fields.
5. Always include an explanation field.

## Examples

Q: "Which neighborhood has the highest hydrant density?"
A: {{"sort":{{"by":"hydrants_per_km2","order":"desc"}},"limit":{{"n":1}},"explanation":"Find the single densest neighborhood."}}

Q: "Show me neighborhoods within 3 km of Times Square"
A: {{"spatial_op":{{"type":"within_distance","lat":40.758,"lon":-73.9855,"distance_km":3}},"explanation":"Neighborhoods within 3 km of Times Square."}}

Q: "Average hydrant density in Brooklyn"
A: {{"filters":[{{"column":"boroname","op":"eq","value":"Brooklyn"}}],"aggregate":{{"column":"hydrants_per_km2","op":"mean"}},"explanation":"Mean density for Brooklyn neighborhoods."}}

Q: "Compare hydrant counts across boroughs"
A: {{"aggregate":{{"column":"hydrant_count","op":"sum","group_by":"boroname"}},"explanation":"Total hydrants by borough."}}

Q: "Top 5 least dense areas in the Bronx"
A: {{"filters":[{{"column":"boroname","op":"eq","value":"Bronx"}}],"sort":{{"by":"hydrants_per_km2","order":"asc"}},"limit":{{"n":5}},"explanation":"Five lowest-density Bronx neighborhoods."}}

Q: "What neighborhoods are near Gramercy?"
A: {{"spatial_op":{{"type":"buffer","neighborhood":"Gramercy","distance_km":2}},"explanation":"Neighborhoods within 2 km of Gramercy."}}
"""


# ── Public API ────────────────────────────────────────────────────────


def create_client(api_key: str) -> OpenAI:
    """Create an OpenAI client from an API key."""
    return OpenAI(api_key=api_key)


def parse_query(
    client: OpenAI, question: str, gdf: gpd.GeoDataFrame
) -> QueryPlan:
    """Convert a natural language question into a validated QueryPlan.

    Uses OpenAI's structured outputs via response_format.
    """
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _build_system_prompt(gdf)},
            {"role": "user", "content": question}
        ],
        response_format=QueryPlan,
        temperature=0,
    )
    return completion.choices[0].message.parsed


def generate_answer(
    client: OpenAI,
    question: str,
    results: gpd.GeoDataFrame,
    query_plan: QueryPlan,
    aggregation: Any = None,
) -> str:
    """Generate a concise natural-language answer from query results."""

    if len(results) == 0:
        return (
            "No neighborhoods matched your query. "
            "Try broadening your search criteria."
        )

    # Build a compact text table of the results
    display_cols = [
        "ntaname", "boroname", "hydrant_count", "area_km2", "hydrants_per_km2",
    ]
    if "distance_km" in results.columns:
        display_cols.append("distance_km")
    available = [c for c in display_cols if c in results.columns]
    table_text = results[available].head(20).to_string(index=False)

    agg_text = ""
    if aggregation is not None:
        agg_text = (
            f"\n\nAggregation result: "
            f"{json.dumps(aggregation) if isinstance(aggregation, dict) else aggregation}"
        )

    prompt = f"""Answer the user's question based on these query results.

Question: {question}

Query plan: {query_plan.explanation}

Results ({len(results)} neighborhoods):
{table_text}{agg_text}

Rules:
- Be concise but informative (2-4 sentences).
- Cite specific numbers from the data.
- If there are aggregation results, highlight them.
- Use neighborhood and borough names from the data.
- Do NOT mention the query plan or technical details.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return response.choices[0].message.content
