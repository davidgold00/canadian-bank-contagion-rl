import math
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="Systemic Bank Network", layout="wide")

BANKS = ["RY.TO", "TD.TO", "BMO.TO", "BNS.TO", "CM.TO", "NA.TO"]
BANK_NAMES = {
    "RY.TO": "Royal Bank of Canada",
    "TD.TO": "Toronto-Dominion Bank",
    "BMO.TO": "Bank of Montreal",
    "BNS.TO": "Bank of Nova Scotia",
    "CM.TO": "CIBC",
    "NA.TO": "National Bank",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@st.cache_data
def load_prices() -> pd.DataFrame:
    root = repo_root()
    candidates = [
        root / "data" / "processed" / "prices.csv",
        root / "data" / "raw" / "market_prices.csv",
        root / "data" / "sample" / "market_prices.csv",
    ]

    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            date_col = "date" if "date" in df.columns else "Date"
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.rename(columns={date_col: "date"}).set_index("date").sort_index()
            return df[[c for c in df.columns if c in BANKS or c in ["XFN.TO", "XIU.TO", "^VIX", "CL=F", "CADUSD=X"]]]

    raise FileNotFoundError("No price file found. Run scripts/download_data.py first.")


def rolling_corr_matrix(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    returns = prices[BANKS].pct_change()
    corr = returns.tail(window).corr()
    return corr.fillna(0)


def tail_corr_matrix(prices: pd.DataFrame, window: int, stress_quantile: float = 0.20) -> pd.DataFrame:
    returns = prices[BANKS].pct_change()
    recent = returns.tail(window).dropna()

    if "XFN.TO" in prices.columns:
        market_ret = prices["XFN.TO"].pct_change().reindex(recent.index)
        stress_days = market_ret <= market_ret.quantile(stress_quantile)
        recent = recent.loc[stress_days]

    if len(recent) < 10:
        recent = returns.tail(window).dropna()

    return recent.corr().fillna(0)


def build_network(adj: pd.DataFrame, threshold: float) -> nx.Graph:
    graph = nx.Graph()

    for bank in BANKS:
        graph.add_node(bank)

    for i, source in enumerate(BANKS):
        for target in BANKS[i + 1:]:
            weight = float(adj.loc[source, target])
            if abs(weight) >= threshold:
                graph.add_edge(source, target, weight=weight, abs_weight=abs(weight))

    return graph


def calculate_node_stress(prices: pd.DataFrame) -> pd.Series:
    returns = prices[BANKS].pct_change()
    vol = returns.tail(21).std() * np.sqrt(252)

    recent_drawdown = {}
    for bank in BANKS:
        px = prices[bank].dropna()
        recent_drawdown[bank] = px.iloc[-1] / px.tail(63).max() - 1 if len(px) > 63 else 0

    drawdown = pd.Series(recent_drawdown)
    vol_score = 100 * (vol - vol.min()) / (vol.max() - vol.min() + 1e-9)
    dd_score = 100 * ((-drawdown) - (-drawdown).min()) / ((-drawdown).max() - (-drawdown).min() + 1e-9)

    return (0.60 * vol_score + 0.40 * dd_score).fillna(50).clip(0, 100)


def network_layout(graph: nx.Graph) -> dict:
    if graph.number_of_edges() == 0:
        return nx.circular_layout(graph)
    return nx.spring_layout(graph, seed=42, weight="abs_weight")


def plot_network(graph: nx.Graph, node_stress: pd.Series, title: str) -> go.Figure:
    pos = network_layout(graph)

    edge_x, edge_y = [], []
    edge_text = []

    for source, target, data in graph.edges(data=True):
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_text.append(f"{source} ↔ {target}: {data['weight']:.2f}")

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1.5, color="rgba(80,80,80,0.45)"),
        hoverinfo="none",
        mode="lines",
    )

    node_x, node_y, node_size, node_color, node_text = [], [], [], [], []

    centrality = nx.eigenvector_centrality_numpy(graph) if graph.number_of_edges() > 0 else {n: 0 for n in graph.nodes}

    for node in graph.nodes:
        x, y = pos[node]
        stress = float(node_stress.get(node, 50))
        importance = float(centrality.get(node, 0))

        node_x.append(x)
        node_y.append(y)
        node_size.append(28 + 45 * importance)
        node_color.append(stress)
        node_text.append(
            f"<b>{node}</b><br>"
            f"{BANK_NAMES[node]}<br>"
            f"Node stress: {stress:.1f}/100<br>"
            f"Eigenvector centrality: {importance:.2f}<br>"
            f"Interpretation: larger nodes are more systemically central."
        )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=list(graph.nodes),
        textposition="top center",
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(
            size=node_size,
            color=node_color,
            colorscale="RdYlGn_r",
            showscale=True,
            colorbar=dict(title="Node<br>Stress"),
            line=dict(width=2, color="white"),
        ),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=title,
        height=650,
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
    )
    return fig


def graph_metrics(graph: nx.Graph, adj: pd.DataFrame, node_stress: pd.Series) -> pd.DataFrame:
    degree = nx.degree_centrality(graph)
    between = nx.betweenness_centrality(graph, weight="abs_weight") if graph.number_of_edges() else {n: 0 for n in graph.nodes}
    eigen = nx.eigenvector_centrality_numpy(graph) if graph.number_of_edges() else {n: 0 for n in graph.nodes}
    clustering = nx.clustering(graph, weight="abs_weight") if graph.number_of_edges() else {n: 0 for n in graph.nodes}

    rows = []
    for bank in BANKS:
        connected_edges = [abs(adj.loc[bank, other]) for other in BANKS if other != bank]
        rows.append(
            {
                "Bank": bank,
                "Full Name": BANK_NAMES[bank],
                "Node Stress": round(float(node_stress.get(bank, 50)), 2),
                "Degree Centrality": round(float(degree.get(bank, 0)), 3),
                "Eigenvector Centrality": round(float(eigen.get(bank, 0)), 3),
                "Betweenness Centrality": round(float(between.get(bank, 0)), 3),
                "Clustering": round(float(clustering.get(bank, 0)), 3),
                "Avg Absolute Link Strength": round(float(np.mean(connected_edges)), 3),
            }
        )

    return pd.DataFrame(rows).sort_values(["Eigenvector Centrality", "Node Stress"], ascending=False)


st.title("Systemic Bank Network")

st.info(
    """
    This page converts the Big Six Canadian banks into a dynamic financial network.
    Each bank is a node. Edges represent stress-transmission channels such as return correlation
    or tail co-movement. This makes the project more than a stock dashboard: it becomes a simplified
    systemic-risk map.
    """
)

prices = load_prices().dropna(how="all")

left, right = st.columns([0.72, 0.28])

with right:
    st.subheader("Network Controls")
    edge_type = st.selectbox(
        "Edge definition",
        ["Rolling correlation", "Tail-stress correlation"],
        help="Rolling correlation shows normal co-movement. Tail-stress correlation only looks at bad market days.",
    )
    window = st.slider("Lookback window", 21, 252, 63, step=21)
    threshold = st.slider(
        "Minimum edge strength",
        0.00,
        0.95,
        0.35,
        step=0.05,
        help="Higher thresholds only show the strongest contagion channels.",
    )

    st.markdown("### How to interpret")
    st.markdown(
        """
        - **Large node** = systemically central bank.
        - **Red node** = elevated recent stress.
        - **Thick/high edge value** = stronger co-movement.
        - **Dense network** = diversification is weakening.
        """
    )

if edge_type == "Rolling correlation":
    adj = rolling_corr_matrix(prices, window)
else:
    adj = tail_corr_matrix(prices, window)

node_stress = calculate_node_stress(prices)
graph = build_network(adj, threshold)

avg_corr = float(adj.where(~np.eye(len(adj), dtype=bool)).stack().mean())
density = nx.density(graph)
largest_eigenvalue = float(np.linalg.eigvalsh(adj.fillna(0).values).max())
most_stressed = node_stress.sort_values(ascending=False).index[0]

metric_cols = st.columns(4)
metric_cols[0].metric("Average Bank Correlation", f"{avg_corr:.2f}")
metric_cols[1].metric("Network Density", f"{density:.2f}")
metric_cols[2].metric("Largest Corr Eigenvalue", f"{largest_eigenvalue:.2f}")
metric_cols[3].metric("Most Stressed Bank", most_stressed)

st.plotly_chart(
    plot_network(graph, node_stress, f"{edge_type} Network | Window={window}d | Threshold={threshold:.2f}"),
    use_container_width=True,
)

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Centrality Rankings",
        "Correlation Heatmap",
        "Transmission Edges",
        "Methodology",
    ]
)

with tab1:
    st.subheader("Systemic Importance Rankings")
    st.markdown(
        """
        This table explains which banks are most important in the network.

        - **Degree centrality**: how many strong links a bank has.
        - **Eigenvector centrality**: whether a bank is connected to other important banks.
        - **Betweenness centrality**: whether a bank acts as a bridge in the network.
        - **Node stress**: recent volatility and drawdown pressure.
        """
    )

    metrics_df = graph_metrics(graph, adj, node_stress)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    leader = metrics_df.iloc[0]
    st.success(
        f"""
        Current interpretation: **{leader['Bank']}** ranks highest by systemic importance.
        In a stress event, this name may matter more because it is tightly connected to the rest
        of the Big Six network.
        """
    )

with tab2:
    st.subheader("Bank-to-Bank Co-Movement Matrix")
    st.markdown(
        """
        This heatmap shows the raw relationship behind the network. Darker / stronger values mean
        two banks have moved more similarly over the selected window. In crises, these values often rise,
        which reduces diversification.
        """
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=adj.values,
            x=adj.columns,
            y=adj.index,
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Correlation"),
            text=np.round(adj.values, 2),
            texttemplate="%{text}",
        )
    )
    fig.update_layout(height=550, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Strongest Transmission Channels")
    rows = []
    for i, source in enumerate(BANKS):
        for target in BANKS[i + 1:]:
            rows.append(
                {
                    "Source": source,
                    "Target": target,
                    "Edge Weight": round(float(adj.loc[source, target]), 3),
                    "Abs Weight": round(abs(float(adj.loc[source, target])), 3),
                    "Interpretation": "Strong contagion channel"
                    if abs(float(adj.loc[source, target])) >= threshold
                    else "Below display threshold",
                }
            )

    edge_df = pd.DataFrame(rows).sort_values("Abs Weight", ascending=False)
    st.dataframe(edge_df, use_container_width=True, hide_index=True)

    strongest = edge_df.iloc[0]
    st.warning(
        f"""
        Strongest current channel: **{strongest['Source']} ↔ {strongest['Target']}**
        with edge weight **{strongest['Edge Weight']}**. If one of these names is hit by a shock,
        this pair is the first place to monitor for spillover.
        """
    )

with tab4:
    st.subheader("Why this page is valuable")
    st.markdown(
        """
        A normal portfolio dashboard treats each bank as a separate investment.
        A systemic-risk dashboard treats banks as an interconnected system.

        This matters because during stress:

        1. Correlations rise.
        2. Diversification weakens.
        3. ETF ownership can transmit selling pressure.
        4. Common macro exposures hit multiple banks simultaneously.
        5. The most connected banks can become risk amplifiers.

        This page is designed to show those relationships visually and quantitatively.
        """
    )

    st.markdown("### Senior quant extension ideas")
    st.markdown(
        """
        - Replace correlation with dynamic conditional correlation models.
        - Add ETF ownership overlap as a second edge layer.
        - Add balance-sheet similarity from annual reports.
        - Use Granger-style lead-lag edges.
        - Estimate conditional tail dependence instead of ordinary correlation.
        - Store daily adjacency matrices for temporal graph neural networks.
        """
    )