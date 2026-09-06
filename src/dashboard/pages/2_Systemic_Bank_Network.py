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
from src.dashboard.ui_components import PALETTE, PLOTLY_TEMPLATE, analyst_header, apply_dashboard_style, decision_callout, decision_memo, insight_card, page_intro


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
        plot_bgcolor=PALETTE["bg"],
        paper_bgcolor=PALETTE["surface"],
        font=dict(color=PALETTE["ink"]),
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

page_intro(
    why=(
        "When the Big Six banks move together, owning several of them provides much less protection than it appears. "
        "This network map shows how tightly the banks are linked — a dense, red network means a selloff in one bank "
        "is very likely to pull the others down too."
    ),
    how=(
        "Each circle (node) is a bank. Larger nodes are more central to the network. Red colour means higher market stress. "
        "Lines between banks mean they move together. Use the controls below to change the lookback period and minimum link strength."
    ),
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

decision_memo(
    "Network Decision Memo",
    [
        {
            "Observation": f"Network density {density:.2f}",
            "Decision Implication": (
                "Sector concentration is the primary risk; multiple bank holdings can behave like one position."
                if density > 0.70
                else "Some name diversification remains useful, but central nodes still deserve tighter limits."
                if density > 0.35
                else "Name diversification is credible at the selected threshold."
            ),
            "Monitoring Trigger": "Tighten exposure limits when density and average correlation rise together.",
        },
        {
            "Observation": f"Largest eigenvalue {largest_eigen:.2f}",
            "Decision Implication": "A larger common factor means bank equity risk is being priced as one macro trade.",
            "Monitoring Trigger": "Escalate if eigenvalue rises while the financials ETF is in drawdown.",
        },
        {
            "Observation": f"Most central bank: {central_bank}",
            "Decision Implication": "Stress-test this name even if it is not currently the weakest performer.",
            "Monitoring Trigger": "If centrality combines with node stress above 60, prioritize it in hedging and exposure reviews.",
        },
    ],
    tone="danger" if density > 0.70 else "warning" if density > 0.35 else "success",
)

left, right = st.columns([0.64, 0.36])
with left:
    st.plotly_chart(network_fig(graph, stress), width="stretch")
with right:
    if density > 0.70:
        insight_card(
            "High Contagion Risk — Banks Moving as One",
            "Most banks are tightly linked right now. Holding several bank stocks provides much less diversification than usual. "
            "A problem at one bank is very likely to drag the others down.",
            status="danger",
        )
        decision_callout(
            plain_english="Dense networks mean the whole sector behaves like a single concentrated trade, not six separate companies.",
            action="Reduce total bank sector exposure or tighten position sizes. Don't rely on diversification across bank names.",
            tone="danger",
        )
    elif density > 0.35:
        insight_card(
            "Partial Contagion — Some Concentration Risk",
            "Several banks are moving together, but not all. Diversification still works to a degree, but monitor whether the network tightens further.",
            status="warning",
        )
        decision_callout(
            plain_english="Some banks are clustered; a stress event could spread but is not guaranteed to hit all names equally.",
            action="Avoid adding new concentration in the most-linked banks. Consider trimming the highest-stress node.",
            tone="warning",
        )
    else:
        insight_card(
            "Contained Network — Diversification is Working",
            "Banks are currently behaving more independently at this threshold. Holding a spread of bank names is more effective at reducing risk.",
            status="success",
        )
        decision_callout(
            plain_english="Banks are responding to their own news more than a shared macro signal. The network is not a primary risk right now.",
            action="Bank-level diversification is credible. Focus on fundamentals and individual bank stress scores.",
            tone="success",
        )
    st.markdown(
        f"**Network summary:** Average bank correlation is **{avg_corr:.2f}** and the largest eigenvalue is "
        f"**{largest_eigen:.2f}**. When the correlation is above 0.80 and the eigenvalue is rising, the sector "
        "is behaving like one concentrated trade rather than six separate businesses.",
        unsafe_allow_html=False,
    )

tab1, tab2, tab3, tab4 = st.tabs(["Contagion Channels", "Systemic Ranking", "Correlation Matrix", "Business Use"])

with tab1:
    st.subheader("Strongest Bank-to-Bank Channels")
    show = edges.copy()
    show["Correlation"] = show["Correlation"].map(lambda x: f"{x:.2f}")
    show["Strength"] = show["Strength"].map(lambda x: f"{x:.2f}")
    st.dataframe(show, width="stretch", hide_index=True)
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
        ["Bank", "Name", "Node Stress", "Network Centrality", "Risk response", "Systemic Interpretation"]
    ].copy()
    show["Node Stress"] = show["Node Stress"].map(lambda x: f"{x:.1f}/100")
    show["Network Centrality"] = show["Network Centrality"].map(lambda x: f"{x:.2f}")
    st.dataframe(show, width="stretch", hide_index=True)

    c_left, c_right = st.columns(2)
    with c_left:
        centrality_rank = ranking.sort_values("Network Centrality", ascending=True)
        fig = go.Figure(go.Bar(
            x=centrality_rank["Network Centrality"],
            y=centrality_rank["Bank"],
            orientation="h",
            text=[f"{x:.2f}" for x in centrality_rank["Network Centrality"]],
            textposition="auto",
            marker_color=PALETTE["blue"],
        ))
        fig.update_layout(
            title="Systemic Centrality — Which Bank Is Most Connected?",
            xaxis_title="Relative centrality (higher = more connected to peers)",
            **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
            height=390,
        )
        st.plotly_chart(fig, width="stretch")
    with c_right:
        stress_rank = ranking.sort_values("Node Stress", ascending=True)
        fig = go.Figure(go.Bar(
            x=stress_rank["Node Stress"],
            y=stress_rank["Bank"],
            orientation="h",
            text=[f"{x:.1f}" for x in stress_rank["Node Stress"]],
            textposition="auto",
            marker=dict(
                color=stress_rank["Node Stress"],
                colorscale=[[0, PALETTE["green"]], [0.5, PALETTE["amber"]], [1.0, PALETTE["red"]]],
                cmin=0, cmax=100, showscale=False,
            ),
        ))
        fig.update_layout(
            title="Node Stress — Which Bank Is Under Most Pressure?",
            xaxis_title="Stress score: 0 = low stress, 100 = maximum stress",
            **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
            height=390,
        )
        st.plotly_chart(fig, width="stretch")

with tab3:
    st.subheader("Correlation Matrix")
    st.markdown(
        "Each cell shows how closely two banks' daily returns move together. "
        "A value near **1.0** (red) means they almost always move in the same direction. "
        "Near **0** means they move independently. Near **−1** (blue) means they move opposite each other."
    )
    fig = go.Figure(go.Heatmap(
        z=adj.values,
        x=adj.columns,
        y=adj.index,
        zmin=-1,
        zmax=1,
        colorscale="RdBu",
        text=np.round(adj.values, 2),
        texttemplate="%{text}",
        colorbar=dict(title="Correlation", tickfont=dict(color=PALETTE["muted"])),
    ))
    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        height=560,
    )
    st.plotly_chart(fig, width="stretch")
    decision_callout(
        plain_english="A matrix full of high positive correlations (dark red) means all banks are being driven by the same factors — making diversification across banks less effective.",
        action="When most pairs exceed 0.75, treat the whole bank allocation as one concentrated position and size accordingly.",
        tone="info",
    )

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
