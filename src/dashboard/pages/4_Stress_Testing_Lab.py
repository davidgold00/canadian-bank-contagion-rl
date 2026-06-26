"""
Stress Testing Lab — macro scenario propagation through the bank network,
portfolio loss attribution, Monte Carlo loss distribution, and post-shock allocation.
"""

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.dashboard.insight_utils import latest_valid_date
from src.dashboard.ui_components import (
    PALETTE,
    PLOTLY_TEMPLATE,
    analyst_header,
    apply_dashboard_style,
    decision_callout,
    decision_memo,
    insight_card,
    interpretation_box,
    page_intro,
)

st.set_page_config(page_title="Stress Testing Lab", layout="wide")
apply_dashboard_style()

BANKS = ["RY.TO", "TD.TO", "BMO.TO", "BNS.TO", "CM.TO", "NA.TO"]

BANK_SHOCK_PCT = {
    "Housing Crisis":        {"RY.TO": -0.12, "TD.TO": -0.12, "BMO.TO": -0.10, "BNS.TO": -0.10, "CM.TO": -0.16, "NA.TO": -0.11},
    "Oil Crash":             {"RY.TO": -0.07, "TD.TO": -0.06, "BMO.TO": -0.09, "BNS.TO": -0.08, "CM.TO": -0.07, "NA.TO": -0.06},
    "Liquidity Squeeze":     {"RY.TO": -0.14, "TD.TO": -0.13, "BMO.TO": -0.12, "BNS.TO": -0.12, "CM.TO": -0.14, "NA.TO": -0.11},
    "Yield Curve Inversion": {"RY.TO": -0.08, "TD.TO": -0.08, "BMO.TO": -0.07, "BNS.TO": -0.07, "CM.TO": -0.09, "NA.TO": -0.06},
    "Global Risk-Off":       {"RY.TO": -0.11, "TD.TO": -0.11, "BMO.TO": -0.10, "BNS.TO": -0.10, "CM.TO": -0.11, "NA.TO": -0.10},
    "Bank-Specific Shock":   {"RY.TO": 0.0,   "TD.TO": 0.0,   "BMO.TO": 0.0,  "BNS.TO": 0.0,  "CM.TO": 0.0,  "NA.TO": 0.0},
}

SCENARIOS = {
    "Housing Crisis": {
        "description": "Mortgage arrears rise, housing prices weaken, domestic credit risk increases.",
        "base_shocks": {"RY.TO": 35, "TD.TO": 35, "BMO.TO": 30, "BNS.TO": 30, "CM.TO": 45, "NA.TO": 32},
        "macro": ["Mortgage arrears ↑", "Housing price index ↓", "Credit provisions ↑", "Domestic bank stress"],
        "key_risk": "CIBC (CM.TO) — highest domestic mortgage sensitivity; expect the largest idiosyncratic drawdown.",
    },
    "Oil Crash": {
        "description": "Oil falls sharply, CAD weakens, Western Canada credit exposure deteriorates.",
        "base_shocks": {"RY.TO": 20, "TD.TO": 18, "BMO.TO": 26, "BNS.TO": 25, "CM.TO": 22, "NA.TO": 18},
        "macro": ["Oil -30%+", "CAD weakens", "Energy credit risk ↑", "TSX pressure"],
        "key_risk": "BMO and BNS — both carry above-average Western Canada commercial credit exposure.",
    },
    "Liquidity Squeeze": {
        "description": "VIX spikes, ETF selling pressure rises, correlations jump across all banks.",
        "base_shocks": {"RY.TO": 40, "TD.TO": 38, "BMO.TO": 36, "BNS.TO": 36, "CM.TO": 38, "NA.TO": 34},
        "macro": ["VIX spike", "Funding stress", "Correlation shock", "ETF outflows"],
        "key_risk": "All banks affected uniformly — ETF-driven selling removes bank-specific differentiation.",
    },
    "Yield Curve Inversion": {
        "description": "Net interest margin pressure and recession concern hit bank valuations.",
        "base_shocks": {"RY.TO": 24, "TD.TO": 24, "BMO.TO": 22, "BNS.TO": 22, "CM.TO": 28, "NA.TO": 20},
        "macro": ["10Y−2Y slope ↓", "NIM pressure", "Recession probability ↑", "Credit spreads widen"],
        "key_risk": "CIBC most rate-sensitive on domestic consumer lending; flat curve hits margin hardest.",
    },
    "Global Risk-Off": {
        "description": "Global equities sell off, CAD weakens, financial stocks de-risk together.",
        "base_shocks": {"RY.TO": 32, "TD.TO": 32, "BMO.TO": 30, "BNS.TO": 31, "CM.TO": 33, "NA.TO": 29},
        "macro": ["SPX ↓", "TSX ↓", "CAD ↓", "VIX ↑"],
        "key_risk": "TD (US retail exposure) and BNS (EM/global exposure) may diverge from peers.",
    },
    "Bank-Specific Shock": {
        "description": "One bank suffers an idiosyncratic event; stress propagates via the network.",
        "base_shocks": {"RY.TO": 0, "TD.TO": 0, "BMO.TO": 0, "BNS.TO": 0, "CM.TO": 0, "NA.TO": 0},
        "macro": ["Idiosyncratic event", "Peer read-through", "Correlation channel", "ETF selling"],
        "key_risk": "Central banks (high degree) transmit to more peers. RY is systemically largest.",
    },
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@st.cache_data
def load_prices() -> pd.DataFrame:
    root = repo_root()
    for path in [root / "data/processed/prices.csv", root / "data/sample/market_prices.csv"]:
        if path.exists():
            df = pd.read_csv(path)
            date_col = "date" if "date" in df.columns else "Date"
            df[date_col] = pd.to_datetime(df[date_col])
            return df.rename(columns={date_col: "date"}).set_index("date").sort_index()
    st.error("No price data found. Run scripts/download_data.py first.")
    st.stop()


def adjacency_matrix(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    returns = prices[[b for b in BANKS if b in prices]].pct_change().tail(window)
    corr = returns.corr().fillna(0).clip(lower=0)
    vals = corr.to_numpy(copy=True)
    np.fill_diagonal(vals, 0)
    corr = pd.DataFrame(vals, index=corr.index, columns=corr.columns)
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


def monte_carlo_portfolio_loss(
    bank_shocks_pct: dict[str, float],
    portfolio_weights: pd.Series,
    n_simulations: int = 2000,
    shock_vol: float = 0.30,
    rng_seed: int = 42,
) -> pd.Series:
    """
    Monte Carlo distribution of 1-day portfolio P&L.
    Base shocks are the mean; shock_vol adds cross-bank noise.
    """
    rng = np.random.default_rng(rng_seed)
    banks = list(bank_shocks_pct.keys())
    means = np.array([bank_shocks_pct.get(b, 0.0) for b in banks])
    vols = np.abs(means) * shock_vol + 0.01
    noise = rng.normal(0, 1, (n_simulations, len(banks)))
    sim_shocks = means + noise * vols
    weights_vec = np.array([float(portfolio_weights.get(b, 0.0)) for b in banks])
    return pd.Series(sim_shocks @ weights_vec)


def recommended_response(final_stress: pd.Series) -> tuple[str, str]:
    avg = final_stress.mean()
    max_bank = final_stress.idxmax()
    if avg >= 70:
        return (f"Severe: reduce bank exposure materially. Prioritise exiting {max_bank}.", "danger")
    if avg >= 50:
        return (f"High: underweight most stressed names, partial cash. Watch {max_bank}.", "warning")
    if avg >= 30:
        return (f"Moderate: rebalance toward lower-stress banks. Avoid adding {max_bank}.", "info")
    return ("Contained: maintain diversification, continue monitoring.", "success")


prices = load_prices()
adj = adjacency_matrix(prices, 126)

analyst_header(
    "Stress Testing Lab",
    "Turn a macro shock into bank stress, contagion propagation, portfolio loss distribution, and post-shock action.",
    date_text=latest_valid_date(prices),
    source_text="Scenario shocks · correlation network propagation · Monte Carlo loss distribution",
)

page_intro(
    why=(
        "Stress tests force a concrete question: if housing, oil, liquidity, rates, or global risk off breaks badly, "
        "where does damage appear first, how fast does it spread across banks, "
        "and how large would your portfolio loss be? The goal is to find this out <em>before</em> it happens."
    ),
    how=(
        "Pick a scenario and a severity from the controls on the right. "
        "Then read the <b>Propagation</b> tab to see which banks get hit hardest, "
        "and the <b>Portfolio P&L</b> tab to estimate your loss using your own weights."
    ),
)

# ── Controls ──────────────────────────────────────────────────────────────────
ctrl_left, ctrl_right = st.columns([0.70, 0.30])

with ctrl_right:
    st.subheader("Scenario Controls")
    scenario_name = st.selectbox("Scenario", list(SCENARIOS.keys()))
    scenario = SCENARIOS[scenario_name]
    severity = st.slider("Shock severity ×", 0.25, 3.00, 1.00, 0.25)
    steps = st.slider("Propagation steps", 1, 12, 5)
    propagation_strength = st.slider("Network propagation strength", 0.00, 1.00, 0.45, 0.05)
    decay = st.slider("Stress persistence / decay", 0.00, 1.00, 0.70, 0.05)

    if scenario_name == "Bank-Specific Shock":
        shocked_bank = st.selectbox("Primary shocked bank", BANKS)
        bank_shock_level = st.slider("Initial shock (stress score)", 10, 100, 70)
    else:
        shocked_bank = None
        bank_shock_level = None

    st.markdown("### Macro Narrative")
    st.markdown(f"*{scenario['description']}*")
    for item in scenario["macro"]:
        st.markdown(f"- {item}")
    insight_card("Key Risk Focus", scenario["key_risk"], status="warning")

initial = pd.Series(scenario["base_shocks"], dtype=float)
if scenario_name == "Bank-Specific Shock" and shocked_bank:
    initial[:] = 0
    initial[shocked_bank] = bank_shock_level

initial = (initial * severity).clip(0, 100)
paths = propagate_stress(initial, adj, propagation_strength, decay, steps)
final_stress = paths.iloc[-1]

avg_final = final_stress.mean()
max_bank = final_stress.idxmax()
max_stress = final_stress.max()

# ── KPI row ──────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Scenario", scenario_name)
k2.metric("Avg Final Stress", f"{avg_final:.1f}/100")
k3.metric("Most Stressed Bank", max_bank)
k4.metric("Peak Bank Stress", f"{max_stress:.1f}/100")

response_text, response_tone = recommended_response(final_stress)
insight_card("Scenario Recommendation", response_text, status=response_tone)
decision_callout(
    plain_english=(
        f"Under the <b>{scenario_name}</b> scenario at ×{severity:.1f} severity, average bank stress reaches "
        f"<b>{avg_final:.1f}/100</b>. The most exposed bank is <b>{max_bank}</b> at {max_stress:.1f}/100."
    ),
    action=response_text,
    tone=response_tone,
)

decision_memo(
    "Scenario Decision Memo",
    [
        {
            "Observation": f"{scenario_name} at ×{severity:.1f}",
            "Decision Implication": "Use the scenario as a pre-trade veto: a good allocation should survive the selected stress before capital is added.",
            "Monitoring Trigger": "Re-run after changing severity, propagation strength, or portfolio weights.",
        },
        {
            "Observation": f"Peak exposed bank: {max_bank} ({max_stress:.1f}/100)",
            "Decision Implication": "This name should receive the tightest limit or hedge in this scenario.",
            "Monitoring Trigger": "Escalate if it is also a top current holding or high network-centrality node.",
        },
        {
            "Observation": f"Average final stress {avg_final:.1f}/100",
            "Decision Implication": response_text,
            "Monitoring Trigger": "If average stress exceeds 50, require a smaller bank budget or higher cash buffer.",
        },
    ],
    tone=response_tone,
)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Propagation",
    "Portfolio P&L",
    "Monte Carlo Loss",
    "Post-Shock Network",
    "Methodology",
])

# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("How the Shock Propagates Through the Network")

    fig_prop = go.Figure()
    colors_banks = [PALETTE["blue"], PALETTE["green"], PALETTE["amber"], PALETTE["red"], PALETTE["teal"], "#e040fb"]
    for i, bank in enumerate(BANKS):
        if bank in paths.columns:
            fig_prop.add_trace(go.Scatter(
                x=paths.index, y=paths[bank],
                mode="lines+markers", name=bank,
                line=dict(color=colors_banks[i % len(colors_banks)], width=2.5),
                marker=dict(size=7),
            ))
    fig_prop.add_hline(y=70, line_dash="dash", line_color=PALETTE["red"], annotation_text="Severe threshold")
    fig_prop.add_hline(y=40, line_dash="dot", line_color=PALETTE["amber"], annotation_text="Moderate threshold")
    fig_prop.update_layout(
        title="Shock Propagation by Bank — Step 0 is the Exogenous Shock",
        xaxis_title="Propagation step",
        yaxis_title="Stress score (0–100)",
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        height=460,
    )
    st.plotly_chart(fig_prop, use_container_width=True)

    ordered_stress = final_stress.sort_values(ascending=True)
    fig_final = go.Figure(go.Bar(
        x=ordered_stress.values,
        y=ordered_stress.index,
        orientation="h",
        text=[f"{v:.1f}" for v in ordered_stress.values],
        textposition="auto",
        marker=dict(
            color=ordered_stress.values,
            colorscale=[[0, PALETTE["green"]], [0.5, PALETTE["amber"]], [1.0, PALETTE["red"]]],
            cmin=0, cmax=100,
            showscale=False,
        ),
    ))
    fig_final.update_layout(
        title="Final Bank Stress Ranking",
        xaxis_title="Stress score (0–100)",
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        height=380,
    )
    st.plotly_chart(fig_final, use_container_width=True)

    scenario_table = pd.DataFrame({
        "Bank": BANKS,
        "Initial Shock": [f"{initial.get(b, 0):.1f}/100" for b in BANKS],
        "Final Stress": [f"{final_stress.get(b, 0):.1f}/100" for b in BANKS],
        "Incremental": [f"{final_stress.get(b, 0) - initial.get(b, 0):+.1f}" for b in BANKS],
    })
    st.dataframe(scenario_table, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Portfolio Loss Attribution Under Scenario")
    st.markdown(
        "Enter your portfolio weights, then see which banks drive the most estimated P&L loss "
        "under the scenario's assumed bank-level drawdowns."
    )

    bank_shock_pct = {k: v * severity for k, v in BANK_SHOCK_PCT.get(scenario_name, {}).items()}
    if scenario_name == "Bank-Specific Shock" and shocked_bank:
        for b in BANKS:
            bank_shock_pct[b] = -0.04 if b != shocked_bank else -0.18 * severity

    default_wts = {b: round(1.0 / len(BANKS), 3) for b in BANKS}
    wt_cols = st.columns(3)
    custom_weights = {}
    for i, bank in enumerate(BANKS):
        with wt_cols[i % 3]:
            custom_weights[bank] = st.slider(
                f"{bank}", 0.0, 1.0, float(default_wts[bank]), 0.01, key=f"stress_wt_{bank}"
            )
    pw = pd.Series(custom_weights)
    if pw.sum() > 0:
        pw = pw / pw.sum()

    pnl_rows = []
    for bank in BANKS:
        shock = bank_shock_pct.get(bank, 0.0)
        w = float(pw.get(bank, 0.0))
        pnl_rows.append({
            "Bank": bank,
            "Weight": w,
            "Scenario Shock": shock,
            "P&L Contribution": w * shock,
        })
    pnl_df = pd.DataFrame(pnl_rows).sort_values("P&L Contribution")
    total_pnl = pnl_df["P&L Contribution"].sum()
    tone = "danger" if total_pnl < -0.04 else "warning" if total_pnl < -0.02 else "success"
    insight_card(
        f"Total Estimated Portfolio P&L: {total_pnl:+.2%}",
        f"Largest loss contributor: {pnl_df.iloc[0]['Bank']} ({pnl_df.iloc[0]['P&L Contribution']:+.2%}). "
        "Based on per-bank drawdown assumptions calibrated to historical stress episodes. "
        "Actual losses depend on execution timing, secondary price effects, and non-bank positions.",
        status=tone,
    )

    fig_pnl = go.Figure(go.Bar(
        x=pnl_df["Bank"], y=pnl_df["P&L Contribution"],
        text=[f"{p:+.2%}" for p in pnl_df["P&L Contribution"]],
        textposition="outside",
        marker_color=[PALETTE["red"] if p < 0 else PALETTE["green"] for p in pnl_df["P&L Contribution"]],
    ))
    fig_pnl.add_hline(y=0, line_color=PALETTE["border"])
    fig_pnl.update_layout(
        title=f"{scenario_name} — Estimated 1-Day P&L by Bank",
        yaxis_title="Portfolio P&L contribution",
        yaxis_tickformat=".1%",
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        height=400,
    )
    st.plotly_chart(fig_pnl, use_container_width=True)

    pnl_display = pnl_df.copy()
    pnl_display["Weight"] = pnl_display["Weight"].map(lambda x: f"{x:.1%}")
    pnl_display["Scenario Shock"] = pnl_display["Scenario Shock"].map(lambda x: f"{x:+.1%}")
    pnl_display["P&L Contribution"] = pnl_display["P&L Contribution"].map(lambda x: f"{x:+.2%}")
    st.dataframe(pnl_display, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Monte Carlo Loss Distribution")
    st.markdown(
        "The base shocks have estimation uncertainty. This simulation draws 2,000 scenarios "
        "by adding cross-bank noise to the base shock, showing the distribution of possible portfolio losses."
    )

    mc_losses = monte_carlo_portfolio_loss(bank_shock_pct, pw, n_simulations=2000)
    var_95 = float(np.percentile(mc_losses, 5))
    cvar_95 = float(mc_losses[mc_losses <= var_95].mean())
    mean_loss = float(mc_losses.mean())
    worst = float(mc_losses.min())
    best = float(mc_losses.max())

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Mean Scenario P&L", f"{mean_loss:.2%}")
    mc2.metric("VaR 95%", f"{var_95:.2%}", help="5th percentile of simulated P&L (value-at-risk)")
    mc3.metric("CVaR 95%", f"{cvar_95:.2%}", help="Average P&L in worst 5% of simulations")
    mc4.metric("Worst / Best", f"{worst:.2%} / {best:.2%}")

    fig_mc = go.Figure()
    fig_mc.add_trace(go.Histogram(
        x=mc_losses.values,
        nbinsx=60,
        marker_color=PALETTE["blue"],
        opacity=0.75,
        name="Simulated P&L",
    ))
    fig_mc.add_vline(x=var_95, line_color=PALETTE["amber"], line_dash="dash",
                     annotation_text=f"VaR 95% = {var_95:.2%}", annotation_position="top right")
    fig_mc.add_vline(x=cvar_95, line_color=PALETTE["red"], line_dash="dot",
                     annotation_text=f"CVaR 95% = {cvar_95:.2%}", annotation_position="top left")
    fig_mc.add_vline(x=mean_loss, line_color=PALETTE["green"], line_dash="solid",
                     annotation_text=f"Mean = {mean_loss:.2%}", annotation_position="top right")
    fig_mc.update_layout(
        title=f"Monte Carlo P&L Distribution — {scenario_name} (2,000 simulations)",
        xaxis_title="1-Day portfolio P&L",
        yaxis_title="Frequency",
        xaxis_tickformat=".1%",
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        height=460,
        showlegend=False,
    )
    st.plotly_chart(fig_mc, use_container_width=True)

    insight_card(
        "How to Use the MC Distribution",
        f"The distribution has mean {mean_loss:.2%} and CVaR 95% of {cvar_95:.2%}. "
        f"On 5% of days in this stress environment, the model expects losses exceeding {abs(var_95):.2%}. "
        "If this exceeds your risk tolerance, reduce bank exposure or add cash. "
        "Monte Carlo adds realistic shock dispersion around the scenario mean — it is not a forecast.",
        status="warning" if cvar_95 < -0.04 else "info",
    )

# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Post-Shock Network Stress Map")
    graph = nx.Graph()
    graph.add_nodes_from(BANKS)
    for i, source in enumerate(BANKS):
        for target in BANKS[i + 1:]:
            weight = float((adj.loc[source, target] + adj.loc[target, source]) / 2)
            if weight > 0.08:
                graph.add_edge(source, target, weight=weight)

    pos = nx.spring_layout(graph, seed=7, weight="weight") if graph.number_of_edges() else nx.circular_layout(graph)

    edge_x, edge_y, edge_weights = [], [], []
    for s, t, data in graph.edges(data=True):
        x0, y0 = pos[s]; x1, y1 = pos[t]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
        edge_weights.append(data.get("weight", 0.1))

    node_x, node_y, sizes, colors, labels, hovers = [], [], [], [], [], []
    for bank in graph.nodes:
        x, y = pos[bank]
        stress_val = float(final_stress.get(bank, 50))
        deg = nx.degree_centrality(graph).get(bank, 0)
        node_x.append(x); node_y.append(y)
        sizes.append(28 + stress_val * 0.6 + deg * 30)
        colors.append(stress_val)
        labels.append(bank)
        hovers.append(f"<b>{bank}</b><br>Final stress: {stress_val:.1f}/100<br>Network degree: {deg:.2f}")

    fig_net = go.Figure()
    fig_net.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=1.2, color="rgba(120,145,166,0.3)"),
        hoverinfo="skip",
    ))
    fig_net.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=labels, textposition="top center",
        hovertext=hovers, hoverinfo="text",
        marker=dict(
            size=sizes,
            color=colors,
            colorscale=[[0, PALETTE["green"]], [0.5, PALETTE["amber"]], [1.0, PALETTE["red"]]],
            cmin=0, cmax=100,
            showscale=True,
            colorbar=dict(title="Stress", tickfont=dict(color=PALETTE["muted"])),
            line=dict(width=2, color=PALETTE["surface"]),
        ),
    ))
    fig_net.update_layout(
        title="Post-Shock Contagion Network (node size = stress + centrality)",
        showlegend=False,
        **{k: v for k, v in PLOTLY_TEMPLATE["layout"].to_plotly_json().items() if k not in ("xaxis", "yaxis")},
        height=520,
    )
    fig_net.update_xaxes(visible=False)
    fig_net.update_yaxes(visible=False)
    st.plotly_chart(fig_net, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Stress Testing Methodology")
    interpretation_box("Propagation Model", [
        "Step 0 = exogenous shock set by scenario and severity.",
        "Each step: stress_{t+1} = decay × stress_t + propagation_strength × A^T × stress_t",
        "A is the row-normalised correlation-derived adjacency matrix (off-diagonal, clipped to [0,1]).",
        "High propagation_strength + low decay = fast contagion, large step-over-step increase.",
        "Low propagation_strength + high decay = contained stress, slowly fades.",
    ])
    interpretation_box("P&L Attribution", [
        "Each bank's P&L contribution = weight × scenario drawdown assumption.",
        "Drawdowns are calibrated to historical Canadian bank stress episodes by scenario type.",
        "Monte Carlo adds bank-specific shock dispersion (±30% of the base shock magnitude).",
        "Total portfolio P&L is the weighted sum of individual bank contributions.",
    ])
    interpretation_box("How to Act on This", [
        "If total estimated P&L exceeds your loss tolerance, reduce bank exposure before the scenario materialises.",
        "Focus reductions on the highest P&L contributor, not necessarily the most stressed bank.",
        "Use the CVaR Lab to solve for optimal weights that minimise tail loss under this scenario.",
        "Compare pre- and post-scenario allocations to find a portfolio that survives the stress.",
        "Rerun with different propagation strength to stress-test your assumptions about contagion speed.",
    ])

st.caption(
    "Simplified scenario model. Not a regulatory stress test. "
    "Shock assumptions are approximate; actual outcomes depend on timing, correlation regimes, and liquidity."
)
