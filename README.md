# 🗺️ SpatialRAG — NYC Hydrant Density Analyzer

SpatialRAG is an interactive **spatial retrieval-augmented generation (RAG)** application built with Streamlit, GeoPandas, and LLM(OpenAI). It allows users to ask complex, natural-language questions about New York City neighborhood fire hydrant density, translates those questions into structured spatial query plans, runs the analysis using high-performance vector/geometric operations, and displays the results on an interactive Folium map.

---

## 🚀 Key Features

*   **Natural Language to Spatial Query Plans:** Ask questions in plain English (e.g. *"Top 5 densest neighborhoods in Manhattan"*, *"What neighborhoods are within 3 km of Times Square?"*). The app parses the query into a structured JSON plan (identifying spatial bounds, attribute filters, sorting, and aggregation).
*   **Interactive Leaflet Map with Drawing Tools:** Use built-in drawing tools (polygons, circles, rectangles) to draw a region of interest directly on the map.
*   **Dynamic Geometric Constraints:** When a shape is drawn on the map, it automatically constrains any text-based query to that specific geographic boundary (e.g. *"least covered areas in the area I drew"*).
*   **Local Spatial Query Engine:** Computes geometric intersections, spatial buffers, and distances locally using **GeoPandas** and **Shapely** (with metric reprojection via EPSG:32618 for accurate physical distance queries in kilometers).
*   **Beautiful Visualizations:** Color-coded neighborhood choropleth map matching the official NYC hydrant density range, complete with rich hover tooltips and dynamic zooming.

---

## 🛠️ Technology Stack

*   **Frontend UI:** [Streamlit](https://streamlit.io/)
*   **Map Rendering:** [Folium](https://python-visualization.github.io/folium/) & [streamlit-folium](https://github.com/randyzwitch/streamlit-folium)
*   **Geospatial Processing:** [GeoPandas](https://geopandas.org/) & [Shapely](https://shapely.readthedocs.io/)
*   **LLM Gateway:** [OpenAI Python SDK](https://github.com/openai/openai-python) (using Structured Outputs via Pydantic model formats)

---

## 📋 Installation & Local Setup

### Prerequisites

*   Python 3.10+
*   An OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### Step-by-Step Guide

1.  **Clone this repository** (or download the files):
    ```bash
    git clone https://github.com/kollaSrinivas369/spatial-rag-nyc.git
    cd spatial_rag
    ```

2.  **Create and activate a virtual environment**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables**:
    Copy the example `.env` file and replace the placeholder with your actual OpenAI API key:
    ```bash
    cp .env.example .env
    ```
    Open `.env` and fill in your key:
    ```env
    OPENAI_API_KEY=YourActualOpenAIKeyHere...
    ```

5.  **Run the application**:
    ```bash
    streamlit run app.py
    ```

---

## 💡 Example Queries to Try

*   *Which neighborhood has the highest hydrant density?*
*   *Top 5 densest neighborhoods in Manhattan*
*   *What neighborhoods are within 3 km of Times Square?*
*   *Total hydrant count per borough*
*   *Least covered areas in Brooklyn*
*   *What neighborhoods are near Gramercy?*
*   **Drawing test:** Draw a polygon or circle over Brooklyn, then type: *"least covered areas"* or *"what is the average density?"*.

---

## 📁 Repository Structure

```text
spatial_rag/
├── app.py              # Main Streamlit app containing UI and event loops
├── spatial_engine.py   # Geospatial execution engine (distances, nearest, intersections)
├── llm_bridge.py       # OpenAI API integration & structured JSON query plan translation
├── map_utils.py        # Map generation and rendering helper functions
├── requirements.txt    # Application package dependencies
├── .gitignore          # Files to exclude from Git control
├── .env.example        # Environment variable template
└── data/
    └── data.parquet    # Neighborhood geometry and hydrant density source dataset
```
