import streamlit as st
import json
import networkx as nx
import plotly.graph_objects as go
import pandas as pd

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NPM Blast Radius Explorer",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* dark background */
    .stApp { background-color: #0d1117; }
    section[data-testid="stSidebar"] { background-color: #161b22; }

    /* metric cards */
    div[data-testid="metric-container"] {
        background: #161b22;
        border: 1px solid #58a6ff;
        border-radius: 10px;
        padding: 12px 16px;
    }

    /* metric label */
    div[data-testid="metric-container"] label {
        color: #ffffff !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        text-transform: none !important;
        letter-spacing: 0px !important;
    }

    /* metric value */
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 28px !important;
        font-weight: 700 !important;
    }

    /* title */
    h1 { color: #58a6ff !important; letter-spacing: -1px; }
    h2, h3 { color: #ffffff !important; }

    /* all general text white */
    .stApp, .stApp p, .stApp span, .stApp div,
    .stApp label, .stApp li, .stApp a,
    [data-testid="stText"], [data-testid="stMarkdown"] p,
    [data-testid="stCaptionContainer"] p,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div { color: #ffffff !important; }

    /* selectbox and widgets */
    [data-testid="stSelectbox"] label,
    [data-testid="stMetric"] label { color: #ffffff !important; }

    /* dataframe text */
    [data-testid="stDataFrame"] * { color: #ffffff !important; }

    /* expander header */
    [data-testid="stExpander"] summary p { color: #ffffff !important; }

    /* tag pills */
    .pill {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 12px;
        margin: 2px;
        font-weight: 600;
    }
    .pill-red   { background:#ff4444; color:white; }
    .pill-orange{ background:#ff8800; color:white; }
    .pill-green { background:#28a745; color:white; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    with open("data/packages.json") as f:
        return json.load(f)

@st.cache_data
def build_graph(pkg_json):
    packages = json.loads(pkg_json)
    G = nx.DiGraph()
    for name, data in packages.items():
        G.add_node(name,
                   version=data.get("version", ""),
                   description=data.get("description", ""),
                   weekly_downloads=data.get("weekly_downloads", 0))
        for dep in data.get("dependencies", []):
            if dep in packages:
                # edge means: `name` depends on `dep`
                G.add_edge(name, dep)
    return G

# ── Graph helpers ──────────────────────────────────────────────────────────────
def get_blast_radius(G, package):
    """All packages that would break if `package` disappeared."""
    rev = G.reverse()
    try:
        return nx.descendants(rev, package)
    except Exception:
        return set()

def get_direct_dependents(G, package):
    """Packages that directly depend on `package`."""
    rev = G.reverse()
    return set(rev.successors(package))

# ── Plotly network ─────────────────────────────────────────────────────────────
def build_figure(G, packages, selected, blast_set, direct_set):
    pos = nx.spring_layout(G, seed=42, k=1.8)

    try:
        pagerank = nx.pagerank(G, alpha=0.85)
    except Exception:
        pagerank = {n: 1 / max(len(G.nodes), 1) for n in G.nodes}

    # ── edges + arrows ──
    # Each arrow annotation: tail = u (the package that depends),
    #                        tip  = v (the package being depended on)
    # Reading the arrow: "u depends on v"  →  if v disappears, u breaks.
    edge_x, edge_y = [], []
    annotations = []

    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        # arrow points FROM u TO v  (u depends on v)
        annotations.append(dict(
            x=x1, y=y1,          # arrowhead sits at v (the dependency)
            ax=x0, ay=y0,        # arrow tail starts at u (the dependent)
            xref="x", yref="y",
            axref="x", ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.2,
            arrowwidth=1,
            arrowcolor="#30363d",
            opacity=0.6,
        ))

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=0.6, color="#30363d"),
        hoverinfo="none",
    )

    # ── highlighted edges: selected → its dependencies (orange) ──
    hi_edge_x, hi_edge_y = [], []
    for dep in G.successors(selected):
        x0, y0 = pos[selected]
        x1, y1 = pos[dep]
        hi_edge_x += [x0, x1, None]
        hi_edge_y += [y0, y1, None]
        annotations.append(dict(
            x=x1, y=y1,
            ax=x0, ay=y0,
            xref="x", yref="y",
            axref="x", ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor="#ff8800",
            opacity=1.0,
        ))

    hi_edge_trace = go.Scatter(
        x=hi_edge_x, y=hi_edge_y,
        mode="lines",
        line=dict(width=2, color="#ff8800"),
        hoverinfo="none",
    )

    # ── highlighted edges: direct dependents → selected (red) ──
    in_edge_x, in_edge_y = [], []
    for dependent in G.predecessors(selected):
        x0, y0 = pos[dependent]
        x1, y1 = pos[selected]
        in_edge_x += [x0, x1, None]
        in_edge_y += [y0, y1, None]
        annotations.append(dict(
            x=x1, y=y1,
            ax=x0, ay=y0,
            xref="x", yref="y",
            axref="x", ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor="#ff4444",
            opacity=1.0,
        ))

    in_edge_trace = go.Scatter(
        x=in_edge_x, y=in_edge_y,
        mode="lines",
        line=dict(width=2, color="#ff4444"),
        hoverinfo="none",
    )

    # ── nodes ──
    node_x, node_y, colors, sizes, hover_texts, labels = [], [], [], [], [], []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        pr   = pagerank.get(node, 0)
        data = packages.get(node, {})
        dl   = data.get("weekly_downloads", 0)
        desc = (data.get("description") or "")[:70]
        ndeps = G.out_degree(node)
        ndependents = G.in_degree(node)        # in reverse = who depends on it

        hover_texts.append(
            f"<b>{node}</b> v{data.get('version','?')}<br>"
            f"Downloads/week : {dl:,}<br>"
            f"Depends on     : {ndeps} packages<br>"
            f"Depended on by : {ndependents} packages<br>"
            f"Centrality     : {pr:.4f}<br>"
            f"<i>{desc}</i>"
        )
        labels.append(node)

        # size by centrality
        sizes.append(12 + pr * 600)

        # color priority: selected > blast > direct > normal
        if node == selected:
            colors.append("#ff4444")        # red
        elif node in direct_set:
            colors.append("#ff8800")        # orange — direct dependents
        elif node in blast_set:
            colors.append("#ffd700")        # gold  — indirect dependents
        else:
            # shade by criticality
            if pr > 0.04:
                colors.append("#58a6ff")    # blue  — highly central
            else:
                colors.append("#3fb950")    # green — safe leaf

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=labels,
        textposition="top center",
        textfont=dict(size=8, color="#8b949e"),
        hovertext=hover_texts,
        hoverinfo="text",
        marker=dict(
            size=sizes,
            color=colors,
            line=dict(width=1, color="#0d1117"),
        ),
    )

    fig = go.Figure(
        data=[edge_trace, hi_edge_trace, in_edge_trace, node_trace],
        layout=go.Layout(
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            margin=dict(l=0, r=0, t=10, b=0),
            height=560,
            showlegend=False,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            annotations=annotations,
        ),
    )
    return fig

# ── App layout ─────────────────────────────────────────────────────────────────
st.title("NPM Blast Radius Explorer")
st.caption(
    "Pick any npm package and instantly see which packages across the ecosystem "
    "would **break** if it were deleted — just like the left-pad incident of 2016."
)

packages = load_data()
G        = build_graph(json.dumps(packages))

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Select a Package")
    pkg_names = sorted(packages.keys())
    default   = pkg_names.index("lodash") if "lodash" in pkg_names else 0
    selected  = st.selectbox("Package", pkg_names, index=default)

    st.divider()

    st.markdown("### Legend")
    st.markdown("""
    <span class='pill pill-red'>   Selected   </span><br>
    <span class='pill pill-orange'>Direct dependents</span><br>
    <span style='background:#ffd700;color:#000;display:inline-block;
                 padding:2px 10px;border-radius:20px;font-size:12px;
                 font-weight:600;margin:2px;'>Indirect dependents</span><br>
    <span style='background:#58a6ff;color:white;display:inline-block;
                 padding:2px 10px;border-radius:20px;font-size:12px;
                 font-weight:600;margin:2px;'>Critical hub</span><br>
    <span class='pill pill-green'>  Safe leaf  </span>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### Ecosystem Stats")
    st.metric("Packages tracked", len(packages))
    st.metric("Dependency edges", G.number_of_edges())

# ── Compute metrics ────────────────────────────────────────────────────────────
blast_set  = get_blast_radius(G, selected)
direct_set = get_direct_dependents(G, selected)

pkg_data     = packages.get(selected, {})
downloads    = pkg_data.get("weekly_downloads", 0)
direct_deps  = list(G.successors(selected))          # what selected depends on
blast_count  = len(blast_set)
direct_count = len(direct_set)

# ── Top metrics row ────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Package",           selected)
c2.metric("Blast Radius",      blast_count,
          help="Total packages that break (direct + indirect) if this is deleted")
c3.metric("Direct Dependents", direct_count,
          help="Packages that directly list this as a dependency")
c4.metric("Downloads / week",  f"{downloads:,}")

# ── Risk banner ────────────────────────────────────────────────────────────────
if blast_count > 15:
    st.error(
        f"**CRITICAL** — Deleting `{selected}` would cascade through "
        f"**{blast_count} packages** in this ecosystem. "
        f"This is a left-pad level threat."
    )
elif blast_count > 5:
    st.warning(
        f"**HIGH RISK** — `{selected}` is depended on by **{blast_count} packages**. "
        f"Removal would cause significant breakage."
    )
elif blast_count > 0:
    st.info(
        f"**MODERATE** — `{selected}` affects **{blast_count} packages** if removed."
    )
else:
    st.success(
        f"**LOW RISK** — No tracked packages depend on `{selected}`. Safe to remove."
    )

# ── Network graph ──────────────────────────────────────────────────────────────
st.subheader("Dependency Network")

fig = build_figure(G, packages, selected, blast_set, direct_set)
st.plotly_chart(fig, use_container_width=True)

# ── Two-column breakdown ───────────────────────────────────────────────────────
left, right = st.columns(2)

with left:
    st.subheader(f"What `{selected}` depends on")
    if direct_deps:
        dep_rows = []
        for d in direct_deps:
            dl = packages.get(d, {}).get("weekly_downloads", 0)
            dep_rows.append({"Package": d, "Downloads/week": dl})
        df_deps = pd.DataFrame(dep_rows).sort_values("Downloads/week", ascending=False)
        st.dataframe(df_deps, use_container_width=True, hide_index=True)
    else:
        st.success("This package has no tracked dependencies — it's a leaf node.")

with right:
    st.subheader(f"What breaks if `{selected}` is deleted")
    if blast_set:
        rows = []
        for pkg in blast_set:
            dl   = packages.get(pkg, {}).get("weekly_downloads", 0)
            desc = (packages.get(pkg, {}).get("description") or "")[:55]
            kind = "Direct" if pkg in direct_set else "Indirect"
            rows.append({
                "Package": pkg,
                "Impact":  kind,
                "Downloads/week": dl,
                "Description": desc,
            })
        df_blast = (
            pd.DataFrame(rows)
            .sort_values(["Impact", "Downloads/week"], ascending=[True, False])
        )
        st.dataframe(df_blast, use_container_width=True, hide_index=True)
    else:
        st.success("No tracked packages would break. This package is a safe leaf.")

# ── Description card ──────────────────────────────────────────────────────────
with st.expander(f"About `{selected}`"):
    st.markdown(f"**Version:** `{pkg_data.get('version', 'unknown')}`")
    st.markdown(f"**Description:** {pkg_data.get('description', 'N/A')}")
    st.markdown(f"**Weekly Downloads:** {downloads:,}")
    st.markdown(f"**Direct Dependencies:** {', '.join(direct_deps) if direct_deps else 'None'}")

# ── Write-up section ──────────────────────────────────────────────────────────
with st.expander("Project Write-up (Assignment 5)"):
    st.markdown("""
## 1. What Question Is Being Answered?

**"If a popular npm package were suddenly deleted, which other packages across the JavaScript ecosystem would break - and how far would the damage spread?"**

This question is directly inspired by the real-world **left-pad incident of 2016**, in which an 11-line npm package was unpublished by its author. Because thousands of widely-used projects - including React and Babel - depended on it either directly or indirectly, the deletion caused a massive cascading failure across the JavaScript ecosystem worldwide.

This also resonates deeply with my own professional experience. In my company, we have a large number of internally created packages built over the years for various purposes. Over time, due to reasons such as code cleanup, version upgrades, technology changes, or deprecating outdated functionality, we periodically plan to remove or retire some of these packages. What sounds straightforward on paper has caused real problems in practice - on multiple occasions, deleting what seemed like an unused or outdated dependency package unexpectedly broke other parts of the system that were silently depending on it.

I never had a proper opportunity to address this problem systematically. This assignment gave me exactly that chance. I wanted to simulate a similar scenario using public npm packages and build a visual tool that could help any development team answer the question: **"Is it safe to delete this package?"** before actually doing it.

The goal is that this visualization could serve as a practical reference for development teams anywhere - not just for npm packages, but as a conceptual model for understanding dependency risk in any software ecosystem.

**Limitation:** For this assignment, the visualization is scoped to 50 carefully selected npm packages. While sufficient to demonstrate the concept, it does not represent the full npm ecosystem of over 2 million packages. In a real-world scenario, the blast radius numbers would be significantly larger.

---

## 2. Design Rationale

### Visual Encodings
- **Node color** - encodes role: red (selected), orange (direct dependent), gold (indirect dependent), blue (critical hub by PageRank), green (safe leaf)
- **Node size** - encodes ecosystem centrality via PageRank. Larger = more influential
- **Arrow direction** - shows who depends on whom. Arrow from A to B means "A depends on B"
- **Edge highlighting** - orange arrows for outgoing dependencies, red arrows for incoming dependents

### Interaction Techniques
- **Dynamic query** - dropdown instantly recomputes blast radius for any package
- **Details-on-demand** - hover tooltips show version, downloads, dependency counts, and centrality
- **Brushing** - affected nodes are highlighted automatically in the network
- **Multi-view coordination** - selecting a package updates the graph, KPI metrics, risk banner, and both tables simultaneously

### Alternatives Considered
- **Treemap** - discarded: cannot show propagation paths or directional dependencies
- **Sankey diagram** - discarded: becomes unreadable with 50+ nodes
- **Adjacency matrix** - discarded: non-intuitive for general users
- **Force-directed graph** - chosen for its natural representation of network topology

---

## 3. Data Sources

### How the Data Was Collected
The dataset was not downloaded from a pre-existing file. It was collected by running a custom Python script (`fetch_data.py`) from the command line, which queried two of npm's publicly available APIs:

- **Step 1** - 50 popular npm packages were manually curated based on prior work experience and cross-referenced with npm's most depended-upon list (npmjs.com/browse/depended) and npm Trends (npmtrends.com)
- **Step 2** - npm Registry API queried for version, description, and dependencies: `registry.npmjs.org/{package}/latest`
- **Step 3** - npm Downloads API queried for weekly download count: `api.npmjs.org/downloads/point/last-week/{package}`
- **Step 4** - All data saved locally to `data/packages.json`

### References
- npm Registry API - https://registry.npmjs.org
- npm Downloads API - https://api.npmjs.org
- npm Most Depended Upon - https://www.npmjs.com/browse/depended
- npm Trends - https://npmtrends.com
- Stack Overflow Developer Survey - https://survey.stackoverflow.co
- The left-pad incident (2016) - Azer Koculu

---

## 4. Development Commentary

**Total time: approximately 20-22 hours**

- **Idea Brainstorming - ~2 hrs:** Explored multiple ideas (customer churn, e-commerce) before settling on this topic, driven by real work experience with dependency issues
- **Data Pipeline - ~4 hrs:** Setting up API calls, finalizing the 50-package list, handling scoped packages like @babel/core, saving to JSON
- **Data Cleaning - ~2 hrs:** Filtered API responses to extract only the 5 needed fields: name, version, description, dependencies, weekly downloads
- **Graph Logic - ~6 hrs:** Learning NetworkX from scratch, implementing directed graph, reverse-graph traversal for blast radius, PageRank centrality
- **UI Layout - ~2 hrs:** Streamlit dashboard, dark theme, KPI cards, risk banners, multi-view coordination
- **Arrow Annotation - ~1 hr:** Dual-direction arrowheads using Plotly annotations
- **Testing - ~2 hrs:** Validated JSON data against dashboard to confirm dependency relationships matched
- **Write-up - ~2 hrs**

**What took the most time:** Learning NetworkX from scratch was the biggest challenge. The API work, color encoding decisions, and tooltip design also required multiple iterations.
    """)
