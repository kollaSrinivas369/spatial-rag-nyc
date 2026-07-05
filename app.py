"""
app.py — SpatialRAG
────────────────────
Natural language spatial queries for NYC fire hydrant density data.

Ask questions in plain English → OpenAI GPT converts them to structured
query plans → GeoPandas/Shapely executes the spatial analysis →
results appear as highlighted neighborhoods on an interactive map.

Architecture note on drawing:
  st_folium triggers a Streamlit rerun on every drawing interaction.
  We handle this by:
  1. Using returned_objects=["all_drawings"] to capture drawing data
  2. NEVER auto-executing any query on drawing events
  3. Only executing queries when the user explicitly types in chat_input
     or clicks an example button
  4. When drawn shapes exist, auto-injecting them as spatial filters

Usage:
    1. Copy .env.example to .env and set your OPENAI_API_KEY
    2. pip install -r requirements.txt
    3. streamlit run app.py
"""

import streamlit as st
import json
import os
import traceback
from pathlib import Path

from dotenv import load_dotenv
from streamlit_folium import st_folium

from spatial_engine import load_data, execute_query_plan
from llm_bridge import create_client, parse_query, generate_answer
from map_utils import create_base_map, create_legend_html

# ── Load .env file ────────────────────────────────────────────────────
load_dotenv()

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="SpatialRAG — NYC Hydrant Density",
    page_icon="🗺️",
    layout="wide",
)

# ── Cached data loading ──────────────────────────────────────────────


@st.cache_data
def get_data():
    return load_data(str(Path(__file__).parent / "data" / "data.parquet"))


data = get_data()

# ── Session state initialisation ─────────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "drawn_features" not in st.session_state:
    st.session_state.drawn_features = []

if "last_query_result" not in st.session_state:
    st.session_state.last_query_result = None

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🗺️ SpatialRAG")
    st.caption("Natural-language spatial queries for NYC hydrant data")

    st.divider()

    # API key — prefer .env / env var, fall back to text input
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        st.success("✓ OpenAI API key loaded from environment")
    else:
        st.warning(
            "Set `OPENAI_API_KEY` in a `.env` file "
            "or as an environment variable."
        )
        api_key = st.text_input(
            "Or paste your key here:", type="password", key="api_key_input"
        )

    st.divider()

    # Colour legend
    st.markdown(create_legend_html(), unsafe_allow_html=True)

    st.divider()

    # Data summary
    st.markdown(
        f"**{len(data)}** neighborhoods · **5** boroughs\n\n"
        f"Density range: **{data['hydrants_per_km2'].min()}** – "
        f"**{data['hydrants_per_km2'].max()}** / km²"
    )

    st.divider()

    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.last_query_result = None
        st.session_state.drawn_features = []
        st.rerun()

    st.markdown("---")
    st.caption(
        "**Data:** [NTA Neighborhoods]"
        "(https://data.cityofnewyork.us/City-Government/"
        "2020-Neighborhood-Tabulation-Areas-NTAs-/9nt8-h7nd) "
        "& [Hydrants]"
        "(https://data.cityofnewyork.us/Environment/Hydrants/5bgh-vtsn)"
    )

# ── Main layout: Chat (left) | Map (right) ───────────────────────────
chat_col, map_col = st.columns([3, 2])

# ══════════════════════════════════════════════════════════════════════
# RIGHT COLUMN: Interactive map
# ══════════════════════════════════════════════════════════════════════
with map_col:
    st.subheader("🗺️ Map")

    last_result = st.session_state.last_query_result

    if last_result:
        m = create_base_map(
            data,
            highlight_gdf=last_result.get("results"),
            center=last_result.get("center_point"),
            overlay_geoms=last_result.get("overlay_geoms"),
            point=last_result.get("center_point"),
            enable_draw=True,
            drawn_features=st.session_state.drawn_features,
        )
    else:
        m = create_base_map(
            data,
            enable_draw=True,
            drawn_features=st.session_state.drawn_features,
        )

    # st_folium returns data on every interaction (draw, click, zoom).
    # We ONLY use it to silently capture drawings — never to trigger queries.
    map_data = st_folium(
        m,
        use_container_width=True,
        height=550,
        key="main_map",
        returned_objects=["all_drawings"],
    )

    # Silently capture drawings into session state.
    # This runs on every rerun but NEVER triggers a query.
    if map_data is not None:
        drawings = map_data.get("all_drawings")
        if drawings is not None and len(drawings) > 0:
            st.session_state.drawn_features = drawings

    # Show drawing status
    if st.session_state.drawn_features:
        n = len(st.session_state.drawn_features)
        st.info(
            f"📐 **{n} shape(s) captured.** "
            "Type your question in the chat — drawn shapes will "
            "automatically be used as a spatial filter."
        )
        if st.button("🗑️ Clear drawings", key="clear_draw_btn"):
            st.session_state.drawn_features = []
            st.rerun()

# ══════════════════════════════════════════════════════════════════════
# LEFT COLUMN: Chat interface
# ══════════════════════════════════════════════════════════════════════
with chat_col:
    st.subheader("💬 Ask about NYC hydrant data")

    # Example queries
    examples = [
        "Which neighborhood has the highest hydrant density?",
        "Top 5 densest neighborhoods in Manhattan",
        "What neighborhoods are within 3 km of Times Square?",
        "Total hydrant count per borough",
        "Least covered areas in Brooklyn",
        "What neighborhoods are near Gramercy?",
    ]

    st.markdown("**Try an example:**")
    ex_cols = st.columns(2)
    selected_example = None
    for i, ex in enumerate(examples):
        if ex_cols[i % 2].button(ex, key=f"ex_{i}", use_container_width=True):
            selected_example = ex

    st.divider()

    # ── Render chat history ───────────────────────────────────────────
    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            st.write(entry["answer"])
            if entry.get("query_plan"):
                with st.expander("🔍 Query plan"):
                    st.json(entry["query_plan"])
            if entry.get("result_count") is not None:
                st.caption(f"📊 {entry['result_count']} neighborhoods matched")

    # ── Chat input ────────────────────────────────────────────────────
    # This is the ONLY way a query gets executed — explicit user action.
    user_input = st.chat_input("Ask a question about NYC hydrant density…")

    # Determine which query to run (user input takes priority).
    # Note: NO pending_query mechanism. Only explicit user actions trigger queries.
    query = user_input or selected_example

    # ── Execute query ─────────────────────────────────────────────────
    if query and api_key:
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing your question…"):
                try:
                    client = create_client(api_key)

                    # Parse natural language → structured query plan
                    plan = parse_query(client, query, data)
                    plan_dict = plan.model_dump(exclude_none=True)

                    # Check if a valid, populated spatial op exists in the plan
                    has_spatial_op = False
                    if plan.spatial_op:
                        op = plan.spatial_op
                        if op.polygon or (op.lat is not None and op.lon is not None) or op.neighborhood:
                            has_spatial_op = True

                    # ── Inject drawn geometry as spatial filter ────────
                    # If the user has drawn shapes AND the LLM didn't
                    # already produce a spatial op, inject the drawing.
                    if st.session_state.drawn_features and not has_spatial_op:
                        last_drawing = st.session_state.drawn_features[-1]
                        geom = last_drawing.get("geometry", {})
                        geom_type = geom.get("type", "")

                        if geom_type == "Polygon":
                            plan_dict["spatial_op"] = {
                                "type": "within_polygon",
                                "polygon": geom["coordinates"][0],
                            }
                        elif geom_type == "Point":
                            # Circle drawn: geometry is Point + radius in properties
                            radius_m = last_drawing.get("properties", {}).get("radius", 2000.0)
                            plan_dict["spatial_op"] = {
                                "type": "within_distance",
                                "lat": geom["coordinates"][1],
                                "lon": geom["coordinates"][0],
                                "distance_km": radius_m / 1000.0,
                            }

                    # Execute the spatial query plan
                    result = execute_query_plan(data, plan_dict)
                    results_gdf = result["results"]

                    # Generate a human-readable answer
                    answer = generate_answer(
                        client,
                        query,
                        results_gdf,
                        plan,
                        aggregation=result.get("aggregation"),
                    )

                    st.write(answer)

                    with st.expander("🔍 Query plan"):
                        st.json(plan_dict)

                    st.caption(
                        f"📊 {len(results_gdf)} neighborhoods matched"
                    )

                    # Persist to session state
                    st.session_state.chat_history.append(
                        {
                            "question": query,
                            "answer": answer,
                            "query_plan": plan_dict,
                            "result_count": len(results_gdf),
                        }
                    )

                    st.session_state.last_query_result = {
                        "results": results_gdf,
                        "center_point": result.get("center_point"),
                        "overlay_geoms": result.get("overlay_geoms", []),
                    }

                    # Rerun so the map column picks up new results
                    st.rerun()

                except Exception as exc:
                    st.error(f"Error: {exc}")
                    with st.expander("Full traceback"):
                        st.code(traceback.format_exc())

    elif query and not api_key:
        st.warning(
            "Please set your OpenAI API key in a `.env` file or the sidebar."
        )
