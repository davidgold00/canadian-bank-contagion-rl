from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="Stress Testing Lab", layout="wide")

BANKS = ["RY.TO", "TD.TO", "BMO.TO", "BNS.TO", "CM.TO", "NA.TO"]

SCENARIOS = {
    "Housing Crisis": {
        "description": "Mortgage arrears rise, housing prices weaken, domestic credit risk increases.",
        "base_shocks": {"RY.TO": 35, "TD.TO": 35, "BMO.TO": 30, "BNS.TO": 30, "CM.TO": 45, "NA.TO": 32},
        "macro": ["Mortgage arrears up", "Housing price index down", "Credit provisions up", "Domestic bank stress"],
    },
    "Oil Crash": {
        "description": "Oil falls sharply, CAD weakens, Western Canada credit exposure deteriorates.",
        "base_shocks": {"RY.TO": 20, "TD.TO": 18, "BMO.TO": 26, "BNS.TO": 25, "CM.TO": 22, "NA.TO": 18},
        "macro": ["Oil -30%", "CAD weakens", "Energy credit risk up", "TSX pressure"],
    },
    "Liquidity Squeeze": {
        "description": "VIX spikes, ETF selling pressure rises, correlations jump across banks.",
        "base_shocks": {"RY.TO": 40, "TD.TO": 38, "BMO.TO": 36, "BNS.TO": 36, "CM.TO": 38, "NA.TO": 34},
        "macro": ["VIX spike", "Funding stress", "Correlation shock", "ETF outflows"],
    },
    "Yield Curve Inversion": {
        "description": "Net interest margin pressure and recession concern hit bank valuations.",
        "base_shocks": {"RY.TO": 24, "TD.TO": 24, "BMO.TO": 22, "BNS.TO": 22, "CM.TO": 28, "NA.TO": 20},
        "macro": ["10Y-2Y slope falls", "NIM pressure", "Recession probability up", "Credit spreads widen"],
    },
    "Global Risk-Off": {
        "description": "Global equities sell off, CAD weakens, financial stocks de-risk together.",
        "base_shocks": {"RY.TO": 32, "TD.TO": 32, "BMO.TO": 30, "BNS.TO": 31, "CM.TO": 33, "NA.TO": 29},
        "macro": ["SPX down", "TSX down", "CAD down", "VIX up"],
    },
    "Bank-Specific Shock": {
        "description": "One bank suffers an idiosyncratic drawdown and stress propagates through the network.",
        "base_shocks": {"RY.TO": 0, "TD.TO": 0, "BMO.TO": 0, "BNS.TO": 0, "CM.TO": 0, "NA.TO": 0},
        "macro": ["Idiosyncratic event", "Peer read-through", "Correlation channel", "ETF channel"],
    },
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@st.cache_data
def load_prices() -> pd.DataFrame:
    root = repo_root()
    candidates = [
        root / "data" / "processed" / "prices.csv",
        root / "data" / "sample" / "market_prices.csv",
    ]

    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            date_col = "date" if "date" in df.columns else "Date"
            df[date_col] = pd.to_datetime(df[date_col])
            return df.rename(columns={date_col: "date"}).set_index("date").sort_index()

    st.error("No price data found. Run scripts/download_data.py first.")
    st.stop()


def adjacency_matrix(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    returns = prices[BANKS].pct_change().tail(window)
    corr = returns.corr().fillna(0).clip(lower=0)
    np.fill_diagonal(corr.values, 0)
    row_sums = corr.sum(axis=1).replace(0, 1)
    return corr.div(row_sums, axis=0)


def propagate_stress(
    initial: pd.Series,
    adjacency: pd.DataFrame,
    propagation_strength: float,
    decay: float,
    steps: int,
) -> pd.DataFrame:
    stress = initial.copy().astype(float)
    rows = [{"Step": 0, **stress.to_dict()}]

    for step in range(1, steps + 1):
        network_effect = adjacency.T.dot(stress)
        stress = decay * stress + propagation_strength * network_effect
        stress = stress.clip(0, 100)
        rows.append({"Step": step, **stress.to_dict()})

    return pd.DataFrame(rows).set_index("Step")


def recommended_response(final_stress: pd.Series) -> str:
    avg = final_stress.mean()
    max_bank = final_stress.idxmax()

    if avg >= 70:
        return f"Severe stress: reduce bank exposure materially, increase cash/hedges, monitor {max_bank}."
    if avg >= 50:
        return f"High stress: underweight most stressed names, reduce concentration, consider partial cash allocation. Watch {max_bank}."
    if avg >= 30:
        return f"Moderate stress: rebalance toward lower-stress banks and avoid increasing concentration in {max_bank}."
    return "Contained stress: maintain diversified allocation, but continue monitoring correlation and volatility."


def portfolio_impact(final_stress: pd.Series, portfolio_weights: pd.Series) -> pd.DataFrame:
    expected_loss = -0.0015 * final_stress * portfolio_weights * 100
    contribution = expected_loss / expected_loss.sum() if expected_loss.sum() != 0 else expected_loss

    return pd.DataFrame(
        {
            "Bank": BANKS,
            "Portfolio Weight": [portfolio_weights.get(b, 0) for b in BANKS],
            "Final Stress": [final_stress.get(b, 0) for b in BANKS],
            "Expected Loss Contribution": [expected_loss.get(b, 0) for b in BANKS],
            "Loss Share": [contribution.get(b, 0) for b in BANKS],
        }
    ).sort_values("Expected Loss Contribution")


def plot_stress_paths(paths: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for bank in BANKS:
        fig.add_trace(go.Scatter(x=paths.index, y=paths[bank], mode="lines+markers", name=bank))

    fig.update_layout(
        title="Shock Propagation by Bank",
        xaxis_title="Propagation step",
        yaxis_title="Stress score, 0–100",
        height=460,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def plot_final_bar(final_stress: pd.Series) -> go.Figure:
    final_stress = final_stress.sort_values()
    fig = go.Figure(
        go.Bar(
            x=final_stress.values,
            y=final_stress.index,
            orientation="h",
            text=[f"{v:.1f}" for v in final_stress.values],
            textposition="auto",
        )
    )
    fig.update_layout(
        title="Final Bank Stress Ranking",
        xaxis_title="Stress score",
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def plot_network_final(adjacency: pd.DataFrame, final_stress: pd.Series) -> go.Figure:
    graph = nx.Graph()

    for bank in BANKS:
        graph.add_node(bank)

    for i, source in enumerate(BANKS):
        for target in BANKS[i + 1:]:
            weight = float((adjacency.loc[source, target] + adjacency.loc[target, source]) / 2)
            if weight > 0.08:
                graph.add_edge(source, target, weight=weight)

    pos = nx.spring_layout(graph, seed=7, weight="weight") if graph.number_of_edges() else nx.circular_layout(graph)

    edge_x, edge_y = [], []
    for source, target in graph.edges():
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x, node_y, colors, sizes, labels = [], [], [], [], []
    for bank in graph.nodes:
        x, y = pos[bank]
        stress = float(final_stress[bank])
        node_x.append(x)
        node_y.append(y)
        colors.append(stress)
        sizes.append(25 + stress / 2)
        labels.append(bank)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1, color="rgba(80,80,80,0.4)")))
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=labels,
            textposition="top center",
            marker=dict(
                size=sizes,
                color=colors,
                colorscale="RdYlGn_r",
                showscale=True,
                colorbar=dict(title="Stress"),
                line=dict(width=2, color="white"),
            ),
        )
    )
    fig.update_layout(
        title="Post-Shock Network Stress Map",
        height=460,
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


st.title("Stress Testing Lab")

st.info(
    """
    This page lets you run simplified institutional stress tests.
    Choose a macro-financial scenario, shock the banks, propagate stress through the network,
    and estimate which banks and portfolio positions are most exposed.
    """
)

prices = load_prices()
adj = adjacency_matrix(prices, 126)

left, right = st.columns([0.70, 0.30])

with right:
    st.subheader("Scenario Controls")

    scenario_name = st.selectbox("Scenario", list(SCENARIOS.keys()))
    scenario = SCENARIOS[scenario_name]

    severity = st.slider("Shock severity multiplier", 0.25, 3.00, 1.00, step=0.25)
    steps = st.slider("Propagation steps", 1, 12, 5)
    propagation_strength = st.slider("Network propagation strength", 0.00, 1.00, 0.45, step=0.05)
    decay = st.slider("Stress persistence / decay", 0.00, 1.00, 0.70, step=0.05)

    if scenario_name == "Bank-Specific Shock":
        shocked_bank = st.selectbox("Primary shocked bank", BANKS)
        bank_shock = st.slider("Initial bank shock", 10, 100, 70)
    else:
        shocked_bank = None
        bank_shock = None

    st.markdown("### Macro shock narrative")
    st.write(scenario["description"])
    for item in scenario["macro"]:
        st.markdown(f"- {item}")

initial = pd.Series(scenario["base_shocks"], dtype=float)

if scenario_name == "Bank-Specific Shock":
    initial[:] = 0
    initial[shocked_bank] = bank_shock

initial = (initial * severity).clip(0, 100)
paths = propagate_stress(initial, adj, propagation_strength, decay, steps)
final_stress = paths.iloc[-1]

avg_final = final_stress.mean()
max_bank = final_stress.idxmax()
max_stress = final_stress.max()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Scenario", scenario_name)
m2.metric("Average Final Stress", f"{avg_final:.1f}/100")
m3.metric("Most Stressed Bank", max_bank)
m4.metric("Peak Bank Stress", f"{max_stress:.1f}/100")

st.warning(recommended_response(final_stress))

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Propagation Chart",
        "Portfolio Impact",
        "Network View",
        "Stress Test Explanation",
    ]
)

with tab1:
    st.subheader("How the Shock Spreads")
    st.markdown(
        """
        Step 0 is the initial exogenous shock. Later steps show how stress propagates through the
        bank network based on the correlation-derived adjacency matrix.
        """
    )

    st.plotly_chart(plot_stress_paths(paths), use_container_width=True)
    st.plotly_chart(plot_final_bar(final_stress), use_container_width=True)
    st.dataframe(paths.round(2), use_container_width=True)

with tab2:
    st.subheader("Portfolio Loss Contribution")
    st.markdown(
        """
        This table maps scenario stress into a simplified portfolio impact.
        It shows which banks contribute most to estimated portfolio loss under the scenario.
        """
    )

    default_weights = pd.Series(1 / len(BANKS), index=BANKS)

    st.markdown("### Portfolio Weights")
    cols = st.columns(3)
    weights = {}
    for i, bank in enumerate(BANKS):
        with cols[i % 3]:
            weights[bank] = st.slider(f"{bank} weight", 0.0, 1.0, float(default_weights[bank]), step=0.01)

    portfolio_weights = pd.Series(weights)
    if portfolio_weights.sum() > 0:
        portfolio_weights = portfolio_weights / portfolio_weights.sum()

    impact = portfolio_impact(final_stress, portfolio_weights)
    st.dataframe(impact.round(4), use_container_width=True, hide_index=True)

    fig = go.Figure(
        go.Bar(
            x=impact["Expected Loss Contribution"],
            y=impact["Bank"],
            orientation="h",
            text=[f"{v:.2f}" for v in impact["Expected Loss Contribution"]],
            textposition="auto",
        )
    )
    fig.update_layout(
        title="Estimated Loss Contribution by Bank",
        xaxis_title="Simplified expected loss contribution",
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Post-Shock Network")
    st.markdown(
        """
        Larger and redder nodes represent banks with higher final scenario stress.
        This is useful for explaining contagion visually to non-technical stakeholders.
        """
    )

    st.plotly_chart(plot_network_final(adj, final_stress), use_container_width=True)

with tab4:
    st.subheader("How to Interpret the Stress Test")
    st.markdown(
        """
        This is a simplified version of a bank stress-testing workflow.

        ### Inputs

        - initial scenario shock,
        - network adjacency matrix,
        - propagation strength,
        - stress persistence,
        - portfolio weights.

        ### Propagation logic

        Stress at the next step is a combination of:

        - remaining stress from the previous step,
        - new stress transmitted from connected banks.

        ### What this page demonstrates

        - graph-based contagion modeling,
        - scenario analysis,
        - bank-specific vulnerability,
        - systemic-risk visualization,
        - portfolio-level exposure attribution.

        ### How to make this more advanced

        - Add OSFI capital ratio data.
        - Add actual mortgage and CRE exposure by bank.
        - Add ETF holdings overlap edges.
        - Calibrate propagation strength using historical stress periods.
        - Estimate nonlinear shock transmission with a graph neural network.
        """
    )