"""
CVaR Optimization Lab — production-style portfolio construction with
graph-adjusted covariance, tail-risk optimization, Monte Carlo simulation,
and explicit investment recommendations.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.dashboard.components import format_percent, load_price_data, load_processed_dataset
from src.dashboard.insight_utils import latest_valid_date
from src.dashboard.ui_components import (
    PALETTE,
    PLOTLY_TEMPLATE,
    analyst_header,
    apply_dashboard_style,
    insight_card,
    interpretation_box,
)
from src.portfolio.allocation_policy import BANKS
from src.portfolio.cvar_optimizer import (
    efficient_frontier,
    monte_carlo_cvar,
    optimize_cvar_portfolio,
    optimizer_tables,
    parametric_cvar,
    risk_decomposition,
    stress_scenario_loss,
)
from src.portfolio.portfolio_constraints import PortfolioConstraints

st.set_page_config(page_title="CVaR Optimization Lab", layout="wide")
apply_dashboard_style()


@st.cache_data(show_spinner="Solving graph-adjusted CVaR portfolio…")
def run_optimizer(
    confidence_level, lookback_window, covariance_method, risk_aversion,
    cvar_penalty, volatility_penalty, turnover_penalty, contagion_penalty,
    graph_penalty_strength, max_single_name_weight, max_bank_exposure,
    min_cash_weight, max_cash_weight,
):
    prices = load_price_data()
    features = load_processed_dataset()
    constraints = PortfolioConstraints(
        max_single_name_weight=max_single_name_weight,
        max_bank_exposure=max_bank_exposure,
        min_cash_weight=min_cash_weight,
        max_cash_weight=max_cash_weight,
    )
    result = optimize_cvar_portfolio(
        prices_history=prices,
        features_history=features,
        confidence_level=confidence_level,
        lookback_window=lookback_window,
        covariance_method=covariance_method,
        risk_aversion=risk_aversion,
        cvar_penalty=cvar_penalty,
        volatility_penalty=volatility_penalty,
        turnover_penalty=turnover_penalty,
        contagion_penalty=contagion_penalty,
        graph_penalty_strength=graph_penalty_strength,
        constraints=constraints,
    )
    frontier = efficient_frontier(
        prices, features,
        confidence_level=confidence_level,
        lookback_window=lookback_window,
        constraints=constraints,
    )
    return result, frontier


prices = load_price_data()
features = load_processed_dataset()

analyst_header(
    "CVaR Optimization Lab",
    "Production-style portfolio construction — graph-adjusted covariance, tail-risk constraints, Monte Carlo validation.",
    date_text=latest_valid_date(features),
    source_text="Ledoit-Wolf covariance · graph contagion penalty · SLSQP optimizer",
)

insight_card(
    "Why CVaR Instead of Mean-Variance",
    "Variance treats upside and downside volatility symmetrically. CVaR (Conditional Value at Risk / Expected Shortfall) "
    "directly minimizes the average loss in the left tail of the return distribution — the question risk committees "
    "and institutional allocators actually care about. The graph adjustment inflates effective covariance when banks "
    "are highly correlated or systemically central, preventing the optimizer from taking false diversification credit.",
    status="info",
)

# ── Sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Optimization Controls")
    confidence_level = st.selectbox("CVaR confidence level", [0.95, 0.99], index=0, format_func=lambda x: f"{x:.0%}")
    lookback_window = st.selectbox("Lookback window (days)", [63, 126, 252, 504], index=2)
    covariance_method = st.selectbox("Covariance estimator", ["ledoit_wolf", "ewma", "sample"])

    st.markdown("#### Objective Penalty Weights")
    risk_aversion      = st.slider("Risk aversion (λ_risk)", 1.0, 20.0, 7.0, 0.5)
    cvar_penalty       = st.slider("CVaR penalty (λ_cvar)", 1.0, 20.0, 9.0, 0.5)
    volatility_penalty = st.slider("Volatility penalty (λ_vol)", 0.0, 5.0, 1.0, 0.25)
    turnover_penalty   = st.slider("Turnover penalty (λ_turn)", 0.0, 2.0, 0.20, 0.05)
    contagion_penalty  = st.slider("Contagion exposure penalty (λ_ctg)", 0.0, 3.0, 1.00, 0.05)
    graph_penalty_strength = st.slider("Graph covariance inflation", 0.0, 1.50, 0.40, 0.05)

    st.markdown("#### Portfolio Constraints")
    max_single_name_weight = st.slider("Max single-name weight", 0.05, 0.40, 0.20, 0.01)
    max_bank_exposure      = st.slider("Max Big Six exposure",   0.20, 1.00, 0.72, 0.02)
    min_cash_weight        = st.slider("Min cash allocation",    0.00, 0.40, 0.05, 0.01)
    max_cash_weight        = st.slider("Max cash allocation",    0.05, 0.90, 0.55, 0.05)

result, frontier = run_optimizer(
    confidence_level, lookback_window, covariance_method, risk_aversion,
    cvar_penalty, volatility_penalty, turnover_penalty, contagion_penalty,
    graph_penalty_strength, max_single_name_weight, max_bank_exposure,
    min_cash_weight, max_cash_weight,
)

weights_table, penalty_table, diagnostics_table = optimizer_tables(result)
diag = result.diagnostics

# ── Summary metrics row ───────────────────────────────────────────────────────
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Expected Return", format_percent(diag["expected_return"]))
m2.metric(f"Historical CVaR ({confidence_level:.0%})", format_percent(diag["historical_cvar"]))
m3.metric("Annualized Volatility", format_percent(diag["annualized_volatility"]))
m4.metric("Cash Weight", format_percent(result.weights.get("cash", 0.0)))
m5.metric("Financial Exposure", format_percent(result.weights.reindex(BANKS + ["XFN.TO"]).fillna(0.0).sum()))
m6.metric("Solver Status", str(diag["status"]).split(":")[0].capitalize())

# ── Parametric vs. Historical CVaR check ─────────────────────────────────────
param_cvar = parametric_cvar(result.weights, result.adjusted_covariance, result.expected_returns, confidence_level)
mc_result = monte_carlo_cvar(
    weights=result.weights,
    covariance=result.adjusted_covariance,
    expected_returns=result.expected_returns,
    n_simulations=5000,
    confidence_level=confidence_level,
)

cvr1, cvr2, cvr3 = st.columns(3)
cvr1.metric("Historical CVaR", format_percent(diag["historical_cvar"]), help="Average loss in worst (1−α)% of observed return days")
cvr2.metric("Parametric CVaR (Gaussian)", format_percent(param_cvar), help="Analytical CVaR assuming normally distributed returns")
cvr3.metric("Monte Carlo CVaR (5 000 sims)", format_percent(mc_result["mc_cvar"]), help="Cholesky-decomposed simulation CVaR")

st.divider()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Portfolio Weights",
    "Efficient Frontier",
    "Risk Decomposition",
    "Covariance",
    "Stress Test",
    "Methodology & Raw Data",
])

# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    c1, c2 = st.columns([0.56, 0.44])
    with c1:
        ordered = result.weights.sort_values()
        weight_colors = [
            PALETTE["blue"] if a in BANKS else PALETTE["teal"] if a == "XFN.TO" else PALETTE["muted"]
            for a in ordered.index
        ]
        fig_w = go.Figure(go.Bar(
            x=ordered.values, y=ordered.index,
            orientation="h",
            text=[f"{x:.1%}" for x in ordered.values],
            textposition="auto",
            marker_color=weight_colors,
        ))
        fig_w.update_layout(
            title="CVaR-Optimal Portfolio Weights",
            xaxis_title="Weight",
            xaxis_tickformat=".0%",
            **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
            height=460,
        )
        st.plotly_chart(fig_w, use_container_width=True)

    with c2:
        rc = result.risk_contributions.sort_values("Contagion Contribution", ascending=True)
        fig_ctg = go.Figure(go.Bar(
            x=rc["Contagion Contribution"],
            y=rc["Asset"],
            orientation="h",
            text=[format_percent(x) for x in rc["Contagion Contribution"]],
            textposition="auto",
            marker_color=PALETTE["red"],
        ))
        fig_ctg.update_layout(
            title="Contagion-Aware Risk Budget",
            xaxis_title="Contagion contribution",
            xaxis_tickformat=".0%",
            **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
            height=460,
        )
        st.plotly_chart(fig_ctg, use_container_width=True)

    insight_card(
        "How to Read the Weights",
        f"Cash weight of {result.weights.get('cash', 0):.1%} is driven by the contagion penalty when the "
        f"graph density ({diag['graph_density']:.2f}) and systemic stress are elevated. "
        "Single-name concentration is capped at the max-single-name constraint; financial-sector "
        "exposure is bounded by the max-bank-exposure constraint. The optimizer does not predict returns — "
        "it minimizes tail loss while satisfying the risk budget.",
        status="teal",
    )

    # Actionable recommendation
    bank_w = result.weights.reindex(BANKS).fillna(0.0)
    top_bank = bank_w.idxmax()
    bot_bank = bank_w.idxmin()
    insight_card(
        "Investment Recommendation",
        f"Highest CVaR-optimal allocation: **{top_bank}** at {bank_w[top_bank]:.1%}. "
        f"Lowest: **{bot_bank}** at {bank_w[bot_bank]:.1%}. "
        f"Total bank exposure {bank_w.sum():.1%} vs. constraint ceiling {max_bank_exposure:.0%}. "
        f"Cash position of {result.weights.get('cash', 0):.1%} reflects current tail-risk penalty weight.",
        status="info",
    )

# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    fig_frontier = go.Figure(go.Scatter(
        x=frontier["CVaR"],
        y=frontier["Expected Return"],
        mode="markers+lines+text",
        text=[f"λ={x}" for x in frontier["Risk Aversion"]],
        textposition="top right",
        marker=dict(
            size=14,
            color=frontier["Cash"],
            colorscale="Blues",
            colorbar=dict(title="Cash Wt", tickfont=dict(color=PALETTE["muted"])),
            line=dict(width=1.5, color=PALETTE["border"]),
        ),
        line=dict(color=PALETTE["blue"], width=1.5),
    ))
    # Mark current portfolio
    fig_frontier.add_trace(go.Scatter(
        x=[diag["historical_cvar"]],
        y=[diag["expected_return"]],
        mode="markers+text",
        text=["  ← Current"],
        textposition="middle right",
        marker=dict(size=18, color=PALETTE["amber"], symbol="star"),
        name="Current portfolio",
    ))
    fig_frontier.update_layout(
        title="CVaR Efficient Frontier (Expected Return vs. Tail Risk)",
        xaxis_title="Historical CVaR (95%)",
        yaxis_title="Expected annual return",
        xaxis_tickformat=".1%",
        yaxis_tickformat=".1%",
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        height=500,
    )
    st.plotly_chart(fig_frontier, use_container_width=True)

    st.dataframe(frontier, use_container_width=True, hide_index=True)
    insight_card(
        "Reading the Frontier",
        "Each point represents a different risk-aversion setting. Moving left (lower CVaR) means accepting lower expected return "
        "for stronger downside protection. The star marks the current optimizer output. Cash weight (bubble shade) rises "
        "as risk aversion increases, indicating the optimizer is parking into the defensive asset.",
        status="info",
    )

# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    rd = risk_decomposition(result.weights, result.adjusted_covariance)
    c1, c2 = st.columns(2)

    with c1:
        rc_vol = result.risk_contributions.sort_values("Volatility Contribution", ascending=True)
        fig_vol = go.Figure(go.Bar(
            x=rc_vol["Volatility Contribution"],
            y=rc_vol["Asset"],
            orientation="h",
            text=[format_percent(x) for x in rc_vol["Volatility Contribution"]],
            textposition="auto",
            marker_color=PALETTE["blue"],
        ))
        fig_vol.update_layout(
            title="Marginal Volatility Contribution",
            xaxis_title="Fraction of portfolio vol",
            xaxis_tickformat=".0%",
            **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
            height=420,
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    with c2:
        rc_cvar = result.risk_contributions.sort_values("CVaR Contribution", ascending=True)
        fig_cvar_c = go.Figure(go.Bar(
            x=rc_cvar["CVaR Contribution"],
            y=rc_cvar["Asset"],
            orientation="h",
            text=[format_percent(x) for x in rc_cvar["CVaR Contribution"]],
            textposition="auto",
            marker_color=PALETTE["red"],
        ))
        fig_cvar_c.update_layout(
            title="Contribution to Portfolio CVaR (95%)",
            xaxis_title="Share of tail risk",
            xaxis_tickformat=".0%",
            **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
            height=420,
        )
        st.plotly_chart(fig_cvar_c, use_container_width=True)

    # Full risk decomp table
    rd_display = rd.copy()
    for col in ["Weight", "Standalone Vol", "Marginal Risk", "Component Risk", "% of Portfolio Risk"]:
        rd_display[col] = rd_display[col].map(lambda x: f"{x:.2%}")
    st.dataframe(rd_display, use_container_width=True, hide_index=True)

    interpretation_box("Risk Budget Interpretation", [
        "Component risk = weight × marginal contribution. Sums to total portfolio volatility.",
        "Assets with high component risk relative to weight are the biggest risk contributors per dollar invested.",
        "CVaR contribution reflects the average loss each asset causes on the worst days.",
        "If CVaR contribution > volatility contribution for a bank, it has heavier left-tail skew — worth hedging.",
        f"Current largest risk contributor: {rd.iloc[0]['Asset']} ({rd.iloc[0]['% of Portfolio Risk']:.1%} of portfolio risk)",
    ])

# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    c1, c2 = st.columns(2)
    with c1:
        base_cov = result.base_covariance
        fig_base = go.Figure(go.Heatmap(
            z=base_cov.values, x=base_cov.columns, y=base_cov.index,
            colorscale="RdBu_r", colorbar=dict(title="Cov", tickfont=dict(color=PALETTE["muted"])),
        ))
        fig_base.update_layout(title="Shrinkage Covariance (Ledoit-Wolf)", **PLOTLY_TEMPLATE["layout"].to_plotly_json(), height=500)
        st.plotly_chart(fig_base, use_container_width=True)

    with c2:
        adj_cov = result.adjusted_covariance
        fig_adj = go.Figure(go.Heatmap(
            z=adj_cov.values, x=adj_cov.columns, y=adj_cov.index,
            colorscale="RdBu_r", colorbar=dict(title="Adj Cov", tickfont=dict(color=PALETTE["muted"])),
        ))
        fig_adj.update_layout(title="Graph-Adjusted Effective Covariance", **PLOTLY_TEMPLATE["layout"].to_plotly_json(), height=500)
        st.plotly_chart(fig_adj, use_container_width=True)

    insight_card(
        "Why the Graph Adjustment Matters",
        f"Graph density: {diag['graph_density']:.2f} · Average correlation: {diag['average_correlation']:.2f} · "
        f"Largest eigenvalue: {diag['largest_eigenvalue']:.2f}. "
        "The adjusted covariance inflates bank-to-bank risk when the sector network is dense or concentrated. "
        "This prevents the optimizer from treating correlated banks as genuine diversifiers.",
        status="warning" if diag['graph_density'] > 0.65 else "info",
    )
    st.dataframe(penalty_table, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Scenario Stress Test on Optimal Portfolio")
    scenario_map = {
        "Housing Crisis":        {"RY.TO": -0.12, "TD.TO": -0.12, "BMO.TO": -0.10, "BNS.TO": -0.10, "CM.TO": -0.16, "NA.TO": -0.11, "XFN.TO": -0.10, "cash": 0.0},
        "Oil Crash":             {"RY.TO": -0.07, "TD.TO": -0.06, "BMO.TO": -0.09, "BNS.TO": -0.08, "CM.TO": -0.07, "NA.TO": -0.06, "XFN.TO": -0.06, "cash": 0.0},
        "Liquidity Squeeze":     {"RY.TO": -0.14, "TD.TO": -0.13, "BMO.TO": -0.12, "BNS.TO": -0.12, "CM.TO": -0.14, "NA.TO": -0.11, "XFN.TO": -0.13, "cash": 0.0},
        "Yield Curve Inversion": {"RY.TO": -0.08, "TD.TO": -0.08, "BMO.TO": -0.07, "BNS.TO": -0.07, "CM.TO": -0.09, "NA.TO": -0.06, "XFN.TO": -0.07, "cash": 0.0},
        "Global Risk-Off":       {"RY.TO": -0.11, "TD.TO": -0.11, "BMO.TO": -0.10, "BNS.TO": -0.10, "CM.TO": -0.11, "NA.TO": -0.10, "XFN.TO": -0.11, "cash": 0.0},
    }

    col1, col2 = st.columns([0.4, 0.6])
    with col1:
        chosen_scenario = st.selectbox("Scenario", list(scenario_map.keys()))
        sev = st.slider("Severity multiplier", 0.5, 3.0, 1.0, 0.25)

    shocks = {k: v * sev for k, v in scenario_map[chosen_scenario].items()}
    loss_info = stress_scenario_loss(result.weights, shocks)
    total_loss = loss_info["total_loss"]
    loss_by_asset = loss_info["loss_by_asset"]

    loss_df = pd.DataFrame([
        {"Asset": k, "Weight": float(result.weights.get(k, 0)), "Shock": shocks.get(k, 0), "P&L": v}
        for k, v in loss_by_asset.items()
    ]).sort_values("P&L")

    tone = "danger" if total_loss < -0.04 else "warning" if total_loss < -0.02 else "success"
    insight_card(
        f"{chosen_scenario} (×{sev:.1f}) — Total Portfolio Impact: {total_loss:+.2%}",
        f"Largest contributor: {loss_info['largest_contributor']} "
        f"({loss_by_asset.get(loss_info['largest_contributor'], 0):+.2%}). "
        f"Cash of {result.weights.get('cash', 0):.1%} absorbs {result.weights.get('cash', 0):.1%} "
        "of notional from default shock. CVaR optimization reduces tail exposure vs. an equal-weight portfolio.",
        status=tone,
    )

    fig_loss = go.Figure(go.Bar(
        x=loss_df["Asset"],
        y=loss_df["P&L"],
        text=[f"{x:+.2%}" for x in loss_df["P&L"]],
        textposition="outside",
        marker_color=[PALETTE["red"] if p < 0 else PALETTE["green"] for p in loss_df["P&L"]],
    ))
    fig_loss.add_hline(y=0, line_color=PALETTE["border"])
    fig_loss.update_layout(
        title=f"P&L Contribution by Asset — {chosen_scenario}",
        yaxis_title="Estimated 1-day P&L",
        yaxis_tickformat=".1%",
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        height=400,
    )
    st.plotly_chart(fig_loss, use_container_width=True)

    loss_display = loss_df.copy()
    loss_display["Weight"] = loss_display["Weight"].map(lambda x: f"{x:.1%}")
    loss_display["Shock"] = loss_display["Shock"].map(lambda x: f"{x:+.1%}")
    loss_display["P&L"] = loss_display["P&L"].map(lambda x: f"{x:+.2%}")
    st.dataframe(loss_display, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("CVaR Optimization Objective")
    st.markdown(
        r"""
        The optimizer minimizes a composite risk-adjusted objective:

        $$\min_w \quad -\mu^\top w + \lambda_{\text{cvr}} \cdot CVaR_\alpha(w) + \lambda_{\text{vol}} \cdot \sqrt{w^\top \Sigma_g w}
        + \lambda_{\text{ctg}} \cdot C_g^\top w + \lambda_{\text{turn}} \cdot \|w - w_{\text{prev}}\|^2$$

        **subject to:**
        - $\sum_i w_i = 1$ (fully invested)
        - $w_i \geq 0$ (long-only)
        - $w_i \leq w_{\max}$ (single-name cap)
        - $\sum_{i \in \text{banks}} w_i \leq w_{\text{bank}}$ (sector concentration cap)
        - $w_{\min}^{\text{cash}} \leq w_{\text{cash}} \leq w_{\max}^{\text{cash}}$

        where $\Sigma_g$ is the graph-adjusted covariance (inflated by centrality and graph density)
        and $C_g$ is the per-asset contagion centrality vector.
        """,
        unsafe_allow_html=False,
    )

    st.subheader("Three CVaR Estimates Compared")
    cvar_compare = pd.DataFrame([
        {"Method": "Historical (empirical)", "CVaR": format_percent(diag["historical_cvar"]),
         "Assumption": "Actual observed tail events", "Best For": "Realistic backtesting"},
        {"Method": "Parametric (Gaussian)", "CVaR": format_percent(param_cvar),
         "Assumption": "Normal return distribution", "Best For": "Stress-testing fat-tail assumption"},
        {"Method": "Monte Carlo (5,000 sims)", "CVaR": format_percent(mc_result["mc_cvar"]),
         "Assumption": "Cholesky multivariate normal", "Best For": "Scenario sampling, non-additive structures"},
    ])
    st.dataframe(cvar_compare, use_container_width=True, hide_index=True)

    st.subheader("Raw Diagnostics")
    st.dataframe(diagnostics_table, use_container_width=True, hide_index=True)
    st.dataframe(result.constraint_diagnostics, use_container_width=True, hide_index=True)

    with st.expander("Raw covariance matrices"):
        st.dataframe(result.base_covariance.round(6), use_container_width=True)
        st.dataframe(result.adjusted_covariance.round(6), use_container_width=True)

    st.warning(
        "Limitations: Historical CVaR depends on the lookback window and the quality/completeness of the return series. "
        "Parametric CVaR understates tail risk when returns are non-normal. Monte Carlo assumes multivariate normality. "
        "None of these replace independent regulatory risk models or professional portfolio management."
    )
