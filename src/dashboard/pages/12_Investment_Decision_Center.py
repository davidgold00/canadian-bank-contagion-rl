"""
Investment Decision Center — the single page where all risk signals converge
into explicit, auditable portfolio recommendations.

This is the answer to: "What should I actually do with this information?"
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.dashboard.insight_utils import (
    BANKS,
    latest,
    latest_valid_date,
    load_features,
    load_macro,
    load_prices,
    risk_regime,
)
from src.dashboard.investment_signals import (
    compute_bank_signals,
    compute_market_positioning,
    compute_portfolio_recommendations,
    compute_sensitivity_analysis,
)
from src.dashboard.ui_components import (
    PLOTLY_TEMPLATE,
    action_list,
    analyst_header,
    apply_dashboard_style,
    decision_callout,
    decision_memo,
    insight_card,
    page_intro,
    regime_banner,
    signal_table,
)
from src.portfolio.cvar_optimizer import (
    monte_carlo_cvar,
    optimize_cvar_portfolio,
    risk_decomposition,
)
from src.portfolio.portfolio_constraints import PortfolioConstraints

st.set_page_config(page_title="Investment Decision Center", layout="wide")
apply_dashboard_style()


@st.cache_data(ttl=3600, show_spinner="Loading data…")
def _load():
    features = load_features()
    prices = load_prices()
    macro = load_macro()
    return features, prices, macro


@st.cache_data(ttl=3600, show_spinner="Running CVaR optimization…")
def _run_cvar(lookback: int, confidence: float, risk_aversion: float):
    prices = load_prices()
    features = load_features()
    constraints = PortfolioConstraints(
        max_single_name_weight=0.20,
        max_bank_exposure=0.72,
        min_cash_weight=0.05,
        max_cash_weight=0.55,
    )
    return optimize_cvar_portfolio(
        prices_history=prices,
        features_history=features,
        confidence_level=confidence,
        lookback_window=lookback,
        risk_aversion=risk_aversion,
        cvar_penalty=9.0,
        contagion_penalty=1.0,
        constraints=constraints,
    )


features, prices, macro = _load()

score = float(latest(features, "contagion_risk_score", 50.0))
regime = risk_regime(score)
signals = compute_bank_signals(features, prices, macro)
positioning = compute_market_positioning(features, macro, score)
recs = compute_portfolio_recommendations(signals, score)

analyst_header(
    "Investment Decision Center",
    "From contagion signals to explicit portfolio actions — all risk reads converged into one page.",
    date_text=latest_valid_date(features),
    source_text="Multi-factor signals + CVaR optimization + Monte Carlo stress",
)

page_intro(
    why=(
        "Every other page in this dashboard produces evidence. This page converts that evidence into decisions. "
        "It answers: which banks to buy, hold, or reduce; how much cash to hold; what the CVaR-optimal allocation looks like; "
        "and how much you would lose under a stress scenario."
    ),
    how=(
        "Start with the regime banner below. Then go to <b>Bank Signals</b> for per-bank calls, "
        "<b>Rebalance Plan</b> for explicit weight changes, and <b>Risk Stress</b> to validate the plan "
        "against adverse scenarios before acting."
    ),
)

# ── Top regime banner ────────────────────────────────────────────────────────
regime_banner(regime["label"], regime["summary"], score, regime["tone"])

# ── Key metrics row ──────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Contagion Score", f"{score:.1f}/100", help="Composite systemic stress. <30 Low · 30–60 Moderate · 60–80 High · 80+ Severe")
m2.metric("Suggested Bank Budget", positioning["total_bank_budget"], help="Total allocation to Canadian bank equities")
m3.metric("Cash Guidance", positioning["cash_guidance"].split(".")[0], help="Defensive cash range for the current regime")
avg_corr = latest(features, "avg_pairwise_corr_63d", float("nan"))
m4.metric("Avg Bank Correlation", f"{avg_corr:.2f}" if pd.notna(avg_corr) else "N/A", help="High correlation = diversification within banks is failing")
buys = (signals["Signal"] == "BUY").sum()
reduces = (signals["Signal"] == "REDUCE").sum()
m5.metric("Signal Mix", f"{buys}B · {6 - buys - reduces}H · {reduces}R", help="Buy / Hold / Reduce count across the Big Six")

decision_callout(
    plain_english=(
        f"The system is in a <b>{regime['label']} risk regime</b> (score: {score:.1f}/100). "
        f"Average bank correlation is <b>{avg_corr:.2f}</b> — "
        + ("diversification across banks is significantly reduced." if avg_corr > 0.75 else "banks are still providing some diversification benefit.")
        + f" The model signals <b>{buys} BUY</b>, <b>{6 - buys - reduces} HOLD</b>, and <b>{reduces} REDUCE</b> across the Big Six."
    ),
    action=regime["actions"][0] + " See the Bank Signals and Rebalance Plan tabs for specific names and weight changes.",
    tone=regime["tone"],
)

highest_stress = signals.sort_values("Node Stress", ascending=False).iloc[0]
strongest_signal = signals.sort_values("Composite Score", ascending=False).iloc[0]
largest_rebalance = recs.iloc[recs["Delta"].abs().argmax()]
decision_memo(
    "Portfolio Decision Memo",
    [
        {
            "Observation": f"{regime['label']} regime at {score:.1f}/100",
            "Decision Implication": positioning["sector_bias"],
            "Monitoring Trigger": "Change the bank budget if the score crosses the next regime band or reverses for 10+ trading days.",
        },
        {
            "Observation": f"Average bank correlation {avg_corr:.2f}",
            "Decision Implication": (
                "Treat multiple bank holdings as one shared risk bucket; name diversification is weak."
                if pd.notna(avg_corr) and avg_corr > 0.75
                else "Name diversification still has some value; focus on relative stress ranking."
            ),
            "Monitoring Trigger": "Escalate if correlation moves above 0.80 while drawdowns widen.",
        },
        {
            "Observation": f"Highest stress: {highest_stress['Bank']} at {highest_stress['Node Stress']:.1f}/100",
            "Decision Implication": "Use this as the first trim, hedge, or due-diligence candidate before cutting lower-stress names.",
            "Monitoring Trigger": "Review if node stress stays above 70 or becomes the largest P&L contributor in stress tests.",
        },
        {
            "Observation": f"Strongest signal: {strongest_signal['Bank']} at {strongest_signal['Composite Score']:.1f}/100",
            "Decision Implication": "A positive tilt needs both signal strength and room in the risk budget; do not add if the regime is deteriorating.",
            "Monitoring Trigger": "Confirm score leadership persists after the next weekly rebalance window.",
        },
        {
            "Observation": f"Largest rebalance: {largest_rebalance['Bank']} {largest_rebalance['Delta']:+.1%}",
            "Decision Implication": largest_rebalance["Reason"],
            "Monitoring Trigger": "Act only if the required trade exceeds internal tolerance after costs, tax, and liquidity checks.",
        },
    ],
    tone=regime["tone"],
)

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Bank Signals",
    "Rebalance Plan",
    "Portfolio Construction",
    "Risk Stress",
    "Sensitivity",
])

# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Bank-by-Bank Investment Signals")
    st.markdown(
        "Signals are generated from a multi-factor model combining cross-sectional momentum, "
        "node stress (inverted), mean reversion, and macro tailwinds. "
        "Weights shift toward stress protection as the contagion regime worsens."
    )

    signal_table(signals)

    st.divider()
    st.subheader("Factor Decomposition by Bank")
    factor_cols = ["Bank", "Momentum", "Stress (inverted)", "Mean Reversion", "Macro Tailwind", "Composite Score"]
    factor_df = signals[factor_cols].copy()

    fig = go.Figure()
    factors = ["Momentum", "Stress (inverted)", "Mean Reversion", "Macro Tailwind"]
    colors = ["#1e88e5", "#00c853", "#ffb300", "#00bcd4"]
    for factor, color in zip(factors, colors):
        fig.add_trace(go.Bar(
            name=factor,
            x=factor_df["Bank"],
            y=factor_df[factor],
            marker_color=color,
        ))
    fig.update_layout(
        barmode="group",
        title="Factor Scores by Bank (0–100, higher = more attractive)",
        yaxis_title="Score",
        yaxis_range=[0, 100],
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        height=430,
    )
    st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("Rationale by Bank")
    for _, row in signals.iterrows():
        col1, col2 = st.columns([0.22, 0.78])
        sig = str(row["Signal"])
        tone = "success" if sig == "BUY" else ("danger" if sig == "REDUCE" else "warning")
        with col1:
            insight_card(
                f"{row['Bank']} — {sig}",
                f"Conviction: {'★' * int(row['Conviction'])}  |  Score: {row['Composite Score']:.1f}/100",
                status=tone,
            )
        with col2:
            insight_card(
                row["Name"],
                f"{row['Rationale']}  <br><span style='color:#7a91a6;font-size:0.82rem'>{row['Economic Context']}</span>",
                status="info",
            )

# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Rebalance Action Plan")
    st.markdown(
        "This table shows the gap between an equal-weight baseline and the model's "
        "signal-derived target weights. Actions are flagged only when the delta exceeds 3pp. "
        "Always apply judgment, liquidity constraints, and tax considerations before trading."
    )

    high_priority = recs[recs["Priority"] == "High"]
    med_priority = recs[recs["Priority"] == "Medium"]
    holds = recs[recs["Action"] == "HOLD"]

    if not high_priority.empty:
        st.markdown("#### High Priority Actions")
        for _, row in high_priority.iterrows():
            tone = "success" if row["Action"] == "BUY" else "danger"
            delta_str = f"{row['Delta']:+.1%}"
            insight_card(
                f"{row['Bank']} — {row['Action']}  |  Delta: {delta_str}  →  Target: {row['Target Weight']:.1%}",
                row["Reason"],
                status=tone,
            )

    if not med_priority.empty:
        st.markdown("#### Medium Priority Actions")
        for _, row in med_priority.iterrows():
            tone = "success" if row["Action"] == "BUY" else "danger"
            insight_card(
                f"{row['Bank']} — {row['Action']}  |  Delta: {row['Delta']:+.1%}  →  Target: {row['Target Weight']:.1%}",
                row["Reason"],
                status=tone,
            )

    if not holds.empty:
        st.markdown("#### Hold / No Action")
        hold_display = holds[["Bank", "Current Weight", "Target Weight", "Delta"]].copy()
        for col in ["Current Weight", "Target Weight"]:
            hold_display[col] = hold_display[col].map(lambda x: f"{x:.1%}")
        hold_display["Delta"] = hold_display["Delta"].map(lambda x: f"{x:+.1%}")
        st.dataframe(hold_display, width="stretch", hide_index=True)

    st.divider()
    st.subheader("Positioning Guidance")
    col1, col2 = st.columns(2)
    with col1:
        insight_card("Sector Bias", positioning["sector_bias"], status=regime["tone"])
        st.markdown("**Key Risks Identified**")
        action_list("Current Risk Factors", positioning["key_risks"])
    with col2:
        insight_card("Opportunity", positioning["opportunity"], status="teal")
        insight_card(
            "Cash / Defensive Allocation",
            positioning["cash_guidance"],
            status="info" if score < 60 else "warning",
        )

    fig_weights = go.Figure(go.Bar(
        x=recs["Bank"],
        y=recs["Target Weight"],
        text=[f"{x:.1%}" for x in recs["Target Weight"]],
        textposition="outside",
        marker_color=[
            "#00c853" if r == "BUY" else "#f44336" if r == "REDUCE" else "#1e88e5"
            for r in [signals.set_index("Bank")["Signal"].get(b, "HOLD") for b in recs["Bank"]]
        ],
    ))
    fig_weights.add_hline(
        y=1 / len(BANKS),
        line_dash="dash",
        line_color="#7a91a6",
        annotation_text="Equal weight",
        annotation_position="top right",
    )
    fig_weights.update_layout(
        title="Signal-Derived Target Weights vs Equal-Weight Baseline",
        yaxis_title="Allocation weight",
        yaxis_tickformat=".0%",
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        height=400,
    )
    st.plotly_chart(fig_weights, width="stretch")

# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("CVaR Portfolio Construction")
    st.markdown(
        "This section runs the graph-adjusted CVaR optimizer at the current regime score "
        "and shows how the governed optimizer's weights compare to the signal-derived targets."
    )

    c_left, c_right = st.columns([0.3, 0.7])
    with c_left:
        risk_aversion = st.slider("Risk Aversion", 2.0, 16.0, 7.0, 0.5)
        confidence_level = st.selectbox("CVaR Confidence", [0.95, 0.99], 0, format_func=lambda x: f"{x:.0%}")
        lookback = st.selectbox("Lookback (days)", [63, 126, 252], 1)

    cvar_result = _run_cvar(lookback, confidence_level, risk_aversion)

    diag = cvar_result.diagnostics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expected Return", f"{diag['expected_return']:.1%}")
    c2.metric("Annual Vol", f"{diag['annualized_volatility']:.1%}")
    c3.metric(f"Historical CVaR ({confidence_level:.0%})", f"{diag['historical_cvar']:.1%}")
    c4.metric("Status", str(diag["status"]).capitalize())

    # Monte Carlo CVaR
    if not prices.empty:
        risky_assets = [a for a in cvar_result.weights.index if a != "cash"]
        mc = monte_carlo_cvar(
            weights=cvar_result.weights,
            covariance=cvar_result.adjusted_covariance,
            expected_returns=cvar_result.expected_returns,
            n_simulations=5000,
            confidence_level=confidence_level,
        )
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Monte Carlo CVaR", f"{mc['mc_cvar']:.1%}", help="5,000-simulation Cholesky Monte Carlo CVaR")
        mc2.metric("Monte Carlo VaR", f"{mc['mc_var']:.1%}")
        mc3.metric("MC Expected Return", f"{mc['mc_expected_return']:.1%}")
        mc4.metric("MC Annualized Vol", f"{mc['mc_volatility']:.1%}")
        insight_card(
            "Historical vs. Monte Carlo CVaR",
            f"Historical CVaR ({diag['historical_cvar']:.1%}) vs. Monte Carlo ({mc['mc_cvar']:.1%}) — "
            "large divergence signals either non-normality in historical returns or sparse tail data. "
            "Parametric CVaR assumes Gaussian returns; historical CVaR captures actual tail events.",
            status="info",
        )

    # Side-by-side weights comparison
    st.subheader("CVaR Weights vs Signal Targets")
    cvar_w = cvar_result.weights.reindex(BANKS).fillna(0.0)
    signal_w = signals.set_index("Bank")["Target Weight"].reindex(BANKS).fillna(1 / len(BANKS))

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(name="CVaR Optimizer", x=BANKS, y=cvar_w.values, marker_color="#1e88e5"))
    fig_comp.add_trace(go.Bar(name="Signal Target", x=BANKS, y=signal_w.values, marker_color="#00bcd4"))
    fig_comp.update_layout(
        barmode="group",
        title="CVaR Optimizer vs Signal-Derived Weights",
        yaxis_title="Weight",
        yaxis_tickformat=".0%",
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        height=400,
    )
    st.plotly_chart(fig_comp, width="stretch")

    # Risk decomposition
    st.subheader("Risk Decomposition (Component Risk)")
    rd = risk_decomposition(cvar_result.weights, cvar_result.adjusted_covariance)
    rd_display = rd.copy()
    for col in ["Weight", "Standalone Vol", "Marginal Risk", "Component Risk", "% of Portfolio Risk"]:
        rd_display[col] = rd_display[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
    st.dataframe(rd_display, width="stretch", hide_index=True)

    fig_rc = go.Figure(go.Bar(
        x=rd["% of Portfolio Risk"],
        y=rd["Asset"],
        orientation="h",
        text=[f"{x:.1%}" for x in rd["% of Portfolio Risk"]],
        textposition="auto",
        marker_color="#f44336",
    ))
    fig_rc.update_layout(
        title="% of Total Portfolio Risk by Asset",
        xaxis_title="Share of portfolio risk",
        xaxis_tickformat=".0%",
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        height=400,
    )
    st.plotly_chart(fig_rc, width="stretch")

# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Stress Scenario P&L Attribution")
    st.markdown(
        "Map a macro shock directly to estimated portfolio P&L using signal-derived target weights. "
        "Results are for illustrative purposes — actual outcomes depend on execution, correlations, and timing."
    )

    scenario_name = st.selectbox("Scenario", [
        "Housing Crisis", "Oil Crash", "Liquidity Squeeze",
        "Yield Curve Inversion", "Global Risk-Off",
    ])
    severity = st.slider("Severity multiplier", 0.5, 2.5, 1.0, 0.25)

    SCENARIO_SHOCKS: dict[str, dict[str, float]] = {
        "Housing Crisis":        {"RY.TO": -0.12, "TD.TO": -0.12, "BMO.TO": -0.10, "BNS.TO": -0.10, "CM.TO": -0.16, "NA.TO": -0.11},
        "Oil Crash":             {"RY.TO": -0.07, "TD.TO": -0.06, "BMO.TO": -0.09, "BNS.TO": -0.08, "CM.TO": -0.07, "NA.TO": -0.06},
        "Liquidity Squeeze":     {"RY.TO": -0.14, "TD.TO": -0.13, "BMO.TO": -0.12, "BNS.TO": -0.12, "CM.TO": -0.14, "NA.TO": -0.11},
        "Yield Curve Inversion": {"RY.TO": -0.08, "TD.TO": -0.08, "BMO.TO": -0.07, "BNS.TO": -0.07, "CM.TO": -0.09, "NA.TO": -0.06},
        "Global Risk-Off":       {"RY.TO": -0.11, "TD.TO": -0.11, "BMO.TO": -0.10, "BNS.TO": -0.10, "CM.TO": -0.11, "NA.TO": -0.10},
    }

    base_shocks = {k: v * severity for k, v in SCENARIO_SHOCKS[scenario_name].items()}

    target_w = signals.set_index("Bank")["Target Weight"]
    cvar_w_banks = cvar_result.weights.reindex(BANKS).fillna(0.0)
    eq_w = pd.Series(1 / len(BANKS), index=BANKS)

    rows_stress = []
    for bank in BANKS:
        shock = base_shocks.get(bank, 0.0)
        for label, w in [("Signal Target", target_w), ("CVaR Optimizer", cvar_w_banks), ("Equal Weight", eq_w)]:
            rows_stress.append({
                "Strategy": label,
                "Bank": bank,
                "Shock": shock,
                "Weight": float(w.get(bank, 0.0)),
                "P&L Contribution": float(w.get(bank, 0.0)) * shock,
            })

    stress_df = pd.DataFrame(rows_stress)
    total_pnl = stress_df.groupby("Strategy")["P&L Contribution"].sum().reset_index()
    total_pnl.columns = ["Strategy", "Total Portfolio P&L"]

    pnl_colors = {"Signal Target": "#1e88e5", "CVaR Optimizer": "#00c853", "Equal Weight": "#ffb300"}
    fig_pnl = go.Figure()
    for strat in ["Signal Target", "CVaR Optimizer", "Equal Weight"]:
        sub = stress_df[stress_df["Strategy"] == strat]
        fig_pnl.add_trace(go.Bar(
            name=strat,
            x=sub["Bank"],
            y=sub["P&L Contribution"],
            marker_color=pnl_colors.get(strat, "#7a91a6"),
        ))
    fig_pnl.update_layout(
        barmode="group",
        title=f"{scenario_name} (×{severity:.1f}) — Estimated 1-Day P&L by Bank",
        yaxis_title="Estimated portfolio contribution",
        yaxis_tickformat=".1%",
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        height=420,
    )
    st.plotly_chart(fig_pnl, width="stretch")

    total_pnl["Total Portfolio P&L"] = total_pnl["Total Portfolio P&L"].map(lambda x: f"{x:+.2%}")
    st.dataframe(total_pnl, width="stretch", hide_index=True)

    total_signal = float((target_w * pd.Series(base_shocks)).sum())
    tone = "danger" if total_signal < -0.05 else "warning" if total_signal < -0.02 else "success"
    insight_card(
        f"Signal Portfolio: Estimated Loss = {total_signal:.2%}",
        f"Under the {scenario_name} scenario at ×{severity:.1f} severity, "
        f"signal-derived weights generate an estimated {total_signal:.2%} 1-day P&L. "
        "Actual portfolio impact depends on execution, hedges, and non-bank positions.",
        status=tone,
    )

# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Sensitivity Analysis: Targets Across Score Scenarios")
    st.markdown(
        "Shows how target weights shift if the contagion score changes. "
        "Use this to understand how robust today's positioning is to a deterioration or improvement in market conditions."
    )

    sensitivity = compute_sensitivity_analysis(features, macro)

    fig_sens = go.Figure()
    colors_banks = ["#1e88e5", "#00c853", "#ffb300", "#f44336", "#00bcd4", "#e040fb"]
    for i, bank in enumerate(BANKS):
        if bank in sensitivity.columns:
            fig_sens.add_trace(go.Scatter(
                x=sensitivity.index,
                y=sensitivity[bank],
                mode="lines+markers",
                name=bank,
                line=dict(color=colors_banks[i % len(colors_banks)], width=2),
                marker=dict(size=7),
            ))
    fig_sens.update_layout(
        title="Target Weight Sensitivity to Contagion Score",
        xaxis_title="Contagion Score Scenario",
        yaxis_title="Target Weight",
        yaxis_tickformat=".0%",
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        height=450,
    )
    st.plotly_chart(fig_sens, width="stretch")

    display_sens = sensitivity.copy()
    for col in display_sens.columns:
        display_sens[col] = display_sens[col].map(lambda x: f"{x:.1%}")
    st.dataframe(display_sens, width="stretch")

    insight_card(
        "How to Use This Table",
        "Each row represents a stress scenario (score level). If the score rises from current levels, "
        "target weights shift toward lower-stress banks and cash increases. If conditions improve, "
        "the model tilts more toward momentum leaders. Use this to stress-test your conviction "
        "before deciding to hold or rebalance.",
        status="info",
    )

st.caption(
    "Educational research only. Not investment advice. No real trades are placed. "
    "All signals, weights, and P&L estimates are model outputs from synthetic or delayed data."
)
