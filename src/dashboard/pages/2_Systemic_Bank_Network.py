import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.dashboard.insight_utils import (
    BANKS,
    BANK_CONTEXT,
    BANK_NAMES,
    bank_stress_snapshot,
    latest_valid_date,
    load_features,
    load_prices,
)
from src.dashboard.ui_components import analyst_header, apply_dashboard_style, insight_card


st.set_page_config(page_title="Systemic Bank Network", layout="wide")
apply_dashboard_style()

prices = load_prices()
features = load_features()
bank_prices = prices[[b for b in BANKS if b in prices.columns]].dropna()


def build_adj(prices_df: pd.DataFrame, window: int, edge_type: str) -> pd.DataFrame:
    returns = prices_df.pct_change().dropna()
    if edge_type == "Tail-stress correlation":
        sector_return = returns.mean(axis=1)
        returns = returns.loc[sector_return <= sector_return.quantile(0.20)]
    recent = returns.tail(window)
    return recent.corr().fillna(0)


def build_graph(adj: pd.DataFrame, threshold: float) -> nx.Graph:
    graph = nx.Graph()
    for bank in BANKS:
        graph.add_node(bank)
    for i, source in enumerate(BANKS):
        for target in BANKS[i + 1 :]:
            if source in adj.index and target in adj.columns:
                weight = float(adj.loc[source, target])
                if abs(weight) >= threshold:
                    graph.add_edge(source, target, weight=weight, abs_weight=abs(weight))
    return graph


def centrality(graph: nx.Graph) -> dict:
    if graph.number_of_edges() == 0:
        return {bank: 0.0 for bank in BANKS}
    values = nx.degree_centrality(graph)
    max_value = max(values.values()) or 1
    return {bank: values.get(bank, 0) / max_value for bank in BANKS}


def network_fig(graph: nx.Graph, stress: pd.DataFrame) -> go.Figure:
    pos = nx.spring_layout(graph, seed=19, weight="abs_weight") if graph.number_of_edges() else nx.circular_layout(graph)
    node_stress = stress.set_index("Bank")["Node Stress"]
    cent = centrality(graph)

    edge_x, edge_y = [], []
    for source, target in graph.edges:
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x, node_y, labels, colors, sizes, hover = [], [], [], [], [], []
    for bank in graph.nodes:
        x, y = pos[bank]
        node_x.append(x)
        node_y.append(y)
        labels.append(bank)
        stress_value = float(node_stress.get(bank, 50))
        colors.append(stress_value)
        sizes.append(32 + 44 * cent.get(bank, 0))
        hover.append(
            f"<b>{bank} - {BANK_NAMES[bank]}</b><br>"
            f"Node stress: {stress_value:.1f}/100<br>"
            f"Network centrality: {cent.get(bank, 0):.2f}<br>"
            f"{BANK_CONTEXT[bank]}"
        )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=2, color="rgba(70, 80, 95, 0.35)"),
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
                cmin=0,
                cmax=100,
                colorscale="RdYlGn_r",
                showscale=True,
                colorbar=dict(title="Stress"),
                line=dict(width=2, color="white"),
            ),
        )
    )
    fig.update_layout(
        height=620,
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


def edge_table(adj: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, source in enumerate(BANKS):
        for target in BANKS[i + 1 :]:
            if source in adj.index and target in adj.columns:
                corr = float(adj.loc[source, target])
                strength = abs(corr)
                rows.append(
                    {
                        "Bank Pair": f"{source} - {target}",
                        "Correlation": corr,
                        "Strength": strength,
                        "Business Meaning": "Primary contagion channel"
                        if strength >= 0.70
                        else "Meaningful co-movement"
                        if strength >= 0.45
                        else "Lower current channel",
                    }
                )
    return pd.DataFrame(rows).sort_values("Strength", ascending=False)


analyst_header(
    "Systemic Bank Network",
    "See whether the Big Six are diversifying each other or moving as one risk cluster.",
    date_text=latest_valid_date(prices),
    source_text="Edges from rolling return relationships",
)

st.markdown(
    """
    A bank can be individually healthy and still matter systemically if it is tightly connected
    to the rest of the sector. This page turns bank returns into a network so the contagion
    question becomes visible: where could stress travel next?
    """
)

if bank_prices.empty:
    st.error("No bank price data found.")
    st.stop()

c1, c2, c3 = st.columns(3)
with c1:
    edge_type = st.selectbox("Relationship Type", ["Rolling correlation", "Tail-stress correlation"])
with c2:
    window = st.selectbox("Lookback Window", [21, 63, 126, 252], index=1)
with c3:
    threshold = st.slider("Minimum Link Strength", 0.0, 0.95, 0.35, 0.05)

adj = build_adj(bank_prices, window, edge_type)
graph = build_graph(adj, threshold)
stress = bank_stress_snapshot(features)
edges = edge_table(adj)

mask = ~np.eye(len(adj), dtype=bool)
avg_corr = float(adj.where(mask).stack().mean())
density = nx.density(graph)
largest_eigen = float(np.linalg.eigvalsh(adj.values).max())
central_bank = max(centrality(graph), key=centrality(graph).get)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Average Correlation", f"{avg_corr:.2f}")
m2.metric("Network Density", f"{density:.2f}")
m3.metric("Largest Eigenvalue", f"{largest_eigen:.2f}", help="Higher values mean one common bank factor dominates returns.")
m4.metric("Most Central Bank", central_bank)

left, right = st.columns([0.64, 0.36])
with left:
    st.plotly_chart(network_fig(graph, stress), use_container_width=True)
with right:
    if density > 0.70:
        insight_card(
            "Crowded Sector Signal",
            "Most banks are linked at the selected threshold. In this regime, owning several bank stocks may not provide much diversification.",
            status="danger",
        )
    elif density > 0.35:
        insight_card(
            "Partial Contagion Signal",
            "Several links are active. Watch whether the network becomes denser as volatility rises.",
            status="warning",
        )
    else:
        insight_card(
            "Contained Network Signal",
            "The network is less dense at this threshold. Bank-specific diversification is more credible right now.",
            status="success",
        )
    st.markdown(
        f"""
        **Plain-English takeaway:** average correlation is **{avg_corr:.2f}** and the largest eigenvalue is
        **{largest_eigen:.2f}**. When both climb, the sector behaves less like six separate businesses
        and more like one macro-financial trade.
        """
    )

tab1, tab2, tab3, tab4 = st.tabs(["Contagion Channels", "Systemic Ranking", "Correlation Matrix", "Business Use"])

with tab1:
    st.subheader("Strongest Bank-to-Bank Channels")
    show = edges.copy()
    show["Correlation"] = show["Correlation"].map(lambda x: f"{x:.2f}")
    show["Strength"] = show["Strength"].map(lambda x: f"{x:.2f}")
    st.dataframe(show, use_container_width=True, hide_index=True)
    top = edges.iloc[0]
    st.warning(
        f"The strongest current channel is {top['Bank Pair']} with correlation {top['Correlation']:.2f}. "
        "That pair deserves extra attention in a sector selloff."
    )

with tab2:
    st.subheader("Systemic Importance and Stress")
    cent = centrality(graph)
    ranking = stress.copy()
    ranking["Network Centrality"] = ranking["Bank"].map(cent)
    ranking["Systemic Interpretation"] = ranking.apply(
        lambda r: "High attention: central and stressed"
        if r["Network Centrality"] >= 0.65 and r["Node Stress"] >= 60
        else "Central transmission node"
        if r["Network Centrality"] >= 0.65
        else "Stress contributor"
        if r["Node Stress"] >= 60
        else "Lower current systemic role",
        axis=1,
    )
    show = ranking[
        ["Bank", "Name", "Node Stress", "Network Centrality", "Action Readout", "Systemic Interpretation"]
    ].copy()
    show["Node Stress"] = show["Node Stress"].map(lambda x: f"{x:.1f}/100")
    show["Network Centrality"] = show["Network Centrality"].map(lambda x: f"{x:.2f}")
    st.dataframe(show, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Correlation Matrix")
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
    fig.update_layout(height=560, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("How to Use the Network")
    st.markdown(
        """
        - **Portfolio manager:** reduce concentration when the network is dense and red.
        - **Risk manager:** stress test the most central bank even if it is not the worst performer.
        - **Economic analyst:** rising density suggests bank equity investors are pricing a shared macro problem.
        - **ML engineer:** centrality and edge strength can become model features for stress prediction or RL state design.
        """
    )
