from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="Systemic Bank Network", layout="wide")

BANKS = ["RY.TO", "TD.TO", "BMO.TO", "BNS.TO", "CM.TO", "NA.TO"]


def root() -> Path:
    return Path(__file__).resolve().parents[3]


@st.cache_data
def load_prices():
    for path in [
        root() / "data" / "raw" / "market_prices.csv",
        root() / "data" / "sample" / "market_prices.csv",
    ]:
        if path.exists():
            df = pd.read_csv(path)
            date_col = "date" if "date" in df.columns else "Date"
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.rename(columns={date_col: "date"}).set_index("date").sort_index()
            return df[[c for c in BANKS if c in df.columns]].dropna()

    st.error("No market price data found.")
    st.stop()


def build_adj(prices, window, edge_type):
    rets = prices.pct_change().dropna()

    if edge_type == "Tail-stress correlation":
        market = rets.mean(axis=1)
        rets = rets.loc[market <= market.quantile(0.20)]

    recent = rets.tail(window)
    corr = recent.corr().fillna(0)
    return corr


def build_graph(adj, threshold):
    g = nx.Graph()

    for b in BANKS:
        g.add_node(b)

    for i, a in enumerate(BANKS):
        for b in BANKS[i + 1:]:
            w = float(adj.loc[a, b])
            if abs(w) >= threshold:
                g.add_edge(a, b, weight=w, abs_weight=abs(w))

    return g


def node_stress(prices):
    rets = prices.pct_change()
    vol = rets.tail(21).std() * np.sqrt(252)
    dd = prices.iloc[-1] / prices.tail(63).max() - 1

    vol_score = 100 * (vol - vol.min()) / (vol.max() - vol.min() + 1e-9)
    dd_score = 100 * ((-dd) - (-dd).min()) / ((-dd).max() - (-dd).min() + 1e-9)

    return (0.6 * vol_score + 0.4 * dd_score).clip(0, 100)


def network_fig(g, stress):
    pos = nx.spring_layout(g, seed=11, weight="abs_weight") if g.edges else nx.circular_layout(g)

    edge_x, edge_y = [], []
    for a, b in g.edges:
        x0, y0 = pos[a]
        x1, y1 = pos[b]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    centrality = nx.eigenvector_centrality_numpy(g) if g.number_of_edges() else {n: 0 for n in g.nodes}

    node_x, node_y, sizes, colors, labels, hover = [], [], [], [], [], []

    for n in g.nodes:
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        labels.append(n)
        colors.append(float(stress[n]))
        sizes.append(35 + 65 * centrality.get(n, 0))
        hover.append(
            f"<b>{n}</b><br>"
            f"Node stress: {stress[n]:.1f}/100<br>"
            f"Systemic centrality: {centrality.get(n, 0):.2f}<br>"
            f"Larger node = more connected to important banks"
        )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=2, color="rgba(90,90,90,0.35)"),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=labels,
            textposition="top center",
            hovertext=hover,
            hoverinfo="text",
            marker=dict(
                size=sizes,
                color=colors,
                colorscale="RdYlGn_r",
                cmin=0,
                cmax=100,
                showscale=True,
                colorbar=dict(title="Stress"),
                line=dict(width=2, color="white"),
            ),
        )
    )

    fig.update_layout(
        height=620,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
        plot_bgcolor="white",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def edge_table(adj):
    rows = []
    for i, a in enumerate(BANKS):
        for b in BANKS[i + 1:]:
            rows.append(
                {
                    "Bank Pair": f"{a} ↔ {b}",
                    "Correlation": float(adj.loc[a, b]),
                    "Strength": abs(float(adj.loc[a, b])),
                    "Interpretation": "Major contagion channel"
                    if abs(float(adj.loc[a, b])) >= 0.70
                    else "Moderate channel"
                    if abs(float(adj.loc[a, b])) >= 0.40
                    else "Weak channel",
                }
            )
    return pd.DataFrame(rows).sort_values("Strength", ascending=False)


def centrality_table(g, stress):
    degree = nx.degree_centrality(g)
    eigen = nx.eigenvector_centrality_numpy(g) if g.number_of_edges() else {n: 0 for n in g.nodes}
    between = nx.betweenness_centrality(g, weight="abs_weight") if g.number_of_edges() else {n: 0 for n in g.nodes}

    rows = []
    for b in BANKS:
        rows.append(
            {
                "Bank": b,
                "Stress Score": stress[b],
                "Degree Centrality": degree.get(b, 0),
                "Eigenvector Centrality": eigen.get(b, 0),
                "Betweenness Centrality": between.get(b, 0),
                "Plain-English Meaning": "Core systemic node"
                if eigen.get(b, 0) > np.mean(list(eigen.values()))
                else "Peripheral / lower systemic role",
            }
        )
    return pd.DataFrame(rows).sort_values("Eigenvector Centrality", ascending=False)


prices = load_prices()

st.title("Systemic Bank Network")

st.markdown(
    """
    ### Executive readout

    This page answers: **if stress hits one Canadian bank, which other banks are most likely to move with it?**

    The network is built from recent Big Six return relationships.  
    It is designed to show **contagion channels**, not just individual stock performance.
    """
)

with st.container():
    c1, c2, c3 = st.columns([1, 1, 1])

    with c1:
        edge_type = st.selectbox("Edge definition", ["Rolling correlation", "Tail-stress correlation"])

    with c2:
        window = st.selectbox("Lookback window", [21, 63, 126, 252], index=1)

    with c3:
        threshold = st.slider("Minimum edge strength", 0.0, 0.95, 0.35, 0.05)

adj = build_adj(prices, window, edge_type)
stress = node_stress(prices)
g = build_graph(adj, threshold)

avg_corr = adj.where(~np.eye(len(adj), dtype=bool)).stack().mean()
density = nx.density(g)
largest_eigen = np.linalg.eigvalsh(adj.values).max()
most_stressed = stress.sort_values(ascending=False).index[0]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Average Bank Correlation", f"{avg_corr:.2f}")
m2.metric("Network Density", f"{density:.2f}")
m3.metric("Largest Correlation Eigenvalue", f"{largest_eigen:.2f}")
m4.metric("Most Stressed Bank", most_stressed)

st.markdown("---")

left, right = st.columns([0.62, 0.38])

with left:
    st.subheader("Contagion Network Map")
    st.plotly_chart(network_fig(g, stress), use_container_width=True)

with right:
    st.subheader("How to read this")
    st.markdown(
        """
        **Nodes** are banks.  
        **Edges** are strong co-movement links.  
        **Redder nodes** are under more recent stress.  
        **Larger nodes** are more systemically central.

        A dense red network means the sector is behaving like one crowded trade.
        """
    )

    if density > 0.75:
        st.error("High network density: diversification across Big Six banks is currently weak.")
    elif density > 0.40:
        st.warning("Moderate density: several contagion channels are active.")
    else:
        st.success("Low density: bank-specific diversification is relatively stronger.")

    st.markdown("### Current interpretation")
    st.write(
        f"""
        The current average correlation is **{avg_corr:.2f}** and the most stressed bank is
        **{most_stressed}**. The largest eigenvalue is **{largest_eigen:.2f}**, which measures
        whether one common systemic factor is dominating bank returns.
        """
    )

tab1, tab2, tab3, tab4 = st.tabs(
    ["Strongest Channels", "Systemic Rankings", "Correlation Matrix", "Methodology"]
)

with tab1:
    st.subheader("Strongest Bank-to-Bank Contagion Channels")
    st.markdown(
        """
        This table replaces a meaningless spreadsheet with a ranked list of the most important
        transmission channels. These are the relationships to watch during a Canadian bank selloff.
        """
    )
    edges = edge_table(adj)
    display = edges.copy()
    display["Correlation"] = display["Correlation"].map(lambda x: f"{x:.2f}")
    display["Strength"] = display["Strength"].map(lambda x: f"{x:.2f}")
    st.dataframe(display, use_container_width=True, hide_index=True)

    top = edges.iloc[0]
    st.warning(
        f"Most important channel right now: **{top['Bank Pair']}** with correlation **{top['Correlation']:.2f}**."
    )

with tab2:
    st.subheader("Systemic Importance Ranking")
    st.markdown(
        """
        This ranking identifies which banks are most central in the network.
        A central bank is not necessarily the riskiest stock, but it may matter more for contagion.
        """
    )
    rankings = centrality_table(g, stress)
    display = rankings.copy()
    for col in ["Stress Score", "Degree Centrality", "Eigenvector Centrality", "Betweenness Centrality"]:
        display[col] = display[col].map(lambda x: f"{x:.3f}")
    st.dataframe(display, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Raw Correlation Matrix")
    st.markdown(
        """
        This is the numerical input behind the graph. Values close to 1 mean two banks are moving together.
        During systemic stress, these values often rise across the whole matrix.
        """
    )

    fig = go.Figure(
        go.Heatmap(
            z=adj.values,
            x=adj.columns,
            y=adj.index,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            text=np.round(adj.values, 2),
            texttemplate="%{text}",
            colorbar=dict(title="Correlation"),
        )
    )
    fig.update_layout(height=520, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Methodology")
    st.markdown(
        """
        ### What this page does

        It converts bank returns into a graph:

        - each bank is a node,
        - correlations become edges,
        - recent volatility and drawdown become node stress,
        - centrality measures identify systemic importance.

        ### Why this is valuable

        Traditional dashboards show six separate bank stocks.
        This page shows whether the Big Six are acting like one connected risk cluster.

        ### Senior quant extensions

        - Add ETF ownership overlap as a second edge layer.
        - Add balance-sheet similarity using bank annual reports.
        - Replace rolling correlation with dynamic conditional correlation.
        - Add tail dependence using downside-only returns.
        - Use the graph as input to a temporal GNN.
        - Store daily adjacency matrices for walk-forward contagion modeling.
        """
    )