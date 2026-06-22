"""
Executive Market Overview — entry-point page for daily regime readout,
investment signals, and economic context.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.dashboard.insight_utils import (
    BANK_NAMES,
    BANKS,
    bank_stress_snapshot,
    latest,
    latest_valid_date,
    load_features,
    load_macro,
    load_prices,
    pct,
    previous,
    risk_regime,
    signed_num,
    signed_pct,
    strongest_drivers,
)
from src.dashboard.investment_signals import (
    compute_bank_signals,
    compute_market_positioning,
)
from src.dashboard.ui_components import (
    PALETTE,
    PLOTLY_TEMPLATE,
    action_list,
    analyst_header,
    apply_dashboard_style,
    insight_card,
    regime_banner,
    signal_table,
)


st.set_page_config(page_title="Market Overview", layout="wide")
apply_dashboard_style()


@st.cache_data(ttl=3600, show_spinner="Loading market data…")
def _load():
    return load_features(), load_prices(), load_macro()


features, prices, macro = _load()

score = latest(features, "contagion_risk_score", 50.0)
regime = risk_regime(score)
score_delta = score - previous(features, "contagion_risk_score", 21)

sector_ret_21d = latest(features, "XFN.TO_ret_21d")
sector_drawdown = latest(features, "XFN.TO_drawdown_63d")
avg_corr = latest(features, "avg_pairwise_corr_63d")
avg_vol = latest(features, "avg_bank_vol_21d")
vix = latest(features, "VIX_level")
oil_21d = latest(features, "CL=F_ret_21d")
cad_21d = latest(features, "CADUSD=X_ret_21d")
yield_slope = latest(features, "slope_10y_2y")

bank_perf = bank_stress_snapshot(features)
best_bank = bank_perf.sort_values("21D Return", ascending=False).iloc[0]
worst_bank = bank_perf.sort_values("21D Return", ascending=True).iloc[0]
return_dispersion = float(bank_perf["21D Return"].std()) if len(bank_perf) else float("nan")

signals = compute_bank_signals(features, prices, macro)
positioning = compute_market_positioning(features, macro, score)

analyst_header(
    "Executive Market Overview",
    "Today's Canadian bank risk regime — translated into decisions.",
    date_text=latest_valid_date(features),
    source_text="Live market + Bank of Canada yields + model signals",
)

regime_banner(regime["label"], regime["summary"], score, regime["tone"])

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Contagion Score", f"{score:.1f}/100", delta=signed_num(score_delta, 1), help="0–100 composite systemic stress indicator")
m2.metric("XFN 21D Return", signed_pct(sector_ret_21d), help="Financial sector ETF 21-day return")
m3.metric("XFN 63D Drawdown", pct(sector_drawdown), help="Max drawdown of financial sector ETF over 63 days")
m4.metric("Avg Bank Vol", pct(avg_vol), help="Annualized 21-day volatility averaged across the Big Six")
m5.metric("Bank Correlation", f"{avg_corr:.2f}" if pd.notna(avg_corr) else "N/A", help="Higher correlation = diversification across banks is less effective")
m6.metric("VIX", f"{vix:.1f}" if pd.notna(vix) else "N/A", help="Global implied volatility proxy")

s1, s2, s3, s4, s5, s6 = st.columns(6)
s1.metric("Best 21D Bank", best_bank["Bank"], delta=signed_pct(best_bank["21D Return"]))
s2.metric("Worst 21D Bank", worst_bank["Bank"], delta=signed_pct(worst_bank["21D Return"]))
s3.metric("Return Dispersion", pct(return_dispersion), help="Cross-sectional spread of Big Six 21-day returns")
s4.metric("Oil 21D Return", signed_pct(oil_21d))
s5.metric("CAD 21D Return", signed_pct(cad_21d))
s6.metric("Yield Curve Slope", f"{yield_slope:.2f}%" if pd.notna(yield_slope) else "N/A", help="10Y minus 2Y Government of Canada yield spread")

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Executive Readout",
    "Investment Signals",
    "Bank Risk Map",
    "Economic Backdrop",
    "Data Dictionary",
])

# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    left, right = st.columns([0.62, 0.38])

    with left:
        score_series = features["contagion_risk_score"].dropna()
        fig_score = go.Figure()
        fig_score.add_trace(go.Scatter(
            x=score_series.index, y=score_series,
            mode="lines", name="Contagion Risk",
            line=dict(color=PALETTE["blue"], width=2.5),
        ))
        fig_score.add_hrect(y0=0,  y1=30,  fillcolor="#00c853", opacity=0.06, line_width=0)
        fig_score.add_hrect(y0=30, y1=60,  fillcolor="#ffb300", opacity=0.06, line_width=0)
        fig_score.add_hrect(y0=60, y1=80,  fillcolor="#ff6f00", opacity=0.06, line_width=0)
        fig_score.add_hrect(y0=80, y1=100, fillcolor="#f44336", opacity=0.08, line_width=0)
        fig_score.add_hline(y=30, line_dash="dot", line_color="#00c853", opacity=0.5)
        fig_score.add_hline(y=60, line_dash="dot", line_color="#ffb300", opacity=0.5)
        fig_score.add_hline(y=80, line_dash="dot", line_color="#f44336", opacity=0.5)
        fig_score.add_annotation(x=score_series.index[-1], y=score, text=f"  {score:.1f}",
                                  showarrow=False, font=dict(color=PALETTE["blue_light"], size=13))
        fig_score.update_layout(
            title="Systemic Contagion Risk Score (0–100)",
            yaxis_title="Risk score",
            **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
            height=430,
        )
        st.plotly_chart(fig_score, use_container_width=True)

    with right:
        action_list("Decision Actions Now", regime["actions"])
        st.divider()
        insight_card("Sector Bias", positioning["sector_bias"], status=regime["tone"])
        insight_card("Cash / Defensive Guidance", positioning["cash_guidance"].split(".")[0], status="info")

    drivers = strongest_drivers(features).head(6)
    st.subheader("Current Stress Driver Percentiles")
    st.markdown(
        "Each driver is ranked against its own full history. "
        "A high percentile means it is unusually elevated today."
    )

    fig_drivers = go.Figure(go.Bar(
        x=drivers["Stress Percentile"],
        y=drivers["Driver"],
        orientation="h",
        text=[f"{x:.0%}" for x in drivers["Stress Percentile"]],
        textposition="auto",
        marker=dict(
            color=drivers["Stress Percentile"],
            colorscale=[[0, PALETTE["green"]], [0.5, PALETTE["amber"]], [1.0, PALETTE["red"]]],
            cmin=0, cmax=1,
            showscale=False,
        ),
    ))
    fig_drivers.update_layout(
        title="Top Stress Driver Percentiles (vs. Own History)",
        xaxis_title="Historical percentile",
        xaxis_tickformat=".0%",
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        height=380,
    )
    st.plotly_chart(fig_drivers, use_container_width=True)

    show = drivers[["Driver", "Latest", "Stress Percentile", "Status", "Why it matters"]].copy()
    show["Stress Percentile"] = show["Stress Percentile"].map(lambda x: f"{x:.0%}" if pd.notna(x) else "N/A")
    st.dataframe(show, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Investment Signals — Big Six Banks")
    st.markdown(
        "Multi-factor composite scores combining cross-sectional momentum, node stress (inverted), "
        "mean-reversion potential, and macro tailwinds. "
        "Regime-conditioned: in High/Severe regimes, stress protection dominates momentum."
    )

    signal_table(signals)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        composite_fig = go.Figure(go.Bar(
            x=signals["Bank"],
            y=signals["Composite Score"],
            text=[f"{s:.1f}" for s in signals["Composite Score"]],
            textposition="outside",
            marker=dict(
                color=signals["Composite Score"],
                colorscale=[[0, PALETTE["red"]], [0.5, PALETTE["amber"]], [1.0, PALETTE["green"]]],
                cmin=0, cmax=100,
                showscale=False,
            ),
        ))
        composite_fig.add_hline(y=50, line_dash="dash", line_color=PALETTE["muted"],
                                 annotation_text="Neutral", annotation_position="top right")
        composite_fig.update_layout(
            title="Composite Attractiveness Score (0–100)",
            yaxis_title="Score (higher = more attractive long)",
            yaxis_range=[0, 110],
            **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
            height=370,
        )
        st.plotly_chart(composite_fig, use_container_width=True)

    with col2:
        target_w = signals.set_index("Bank")["Target Weight"]
        eq_w = 1.0 / len(BANKS)
        deltas = target_w - eq_w

        colors_delta = [PALETTE["green"] if d > 0.005 else PALETTE["red"] if d < -0.005 else PALETTE["muted"]
                        for d in deltas]
        fig_delta = go.Figure(go.Bar(
            x=deltas.index,
            y=deltas.values,
            text=[f"{d:+.1%}" for d in deltas.values],
            textposition="outside",
            marker_color=colors_delta,
        ))
        fig_delta.add_hline(y=0, line_color=PALETTE["muted"])
        fig_delta.update_layout(
            title="Signal Tilt vs Equal-Weight Baseline",
            yaxis_title="Weight delta",
            yaxis_tickformat="+.0%",
            **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
            height=370,
        )
        st.plotly_chart(fig_delta, use_container_width=True)

    action_list("Key Risks Flagged by the Model", positioning["key_risks"])

# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Big Six Bank Performance & Stress")

    scatter = go.Figure(go.Scatter(
        x=bank_perf["63D Drawdown"],
        y=bank_perf["21D Volatility"],
        mode="markers+text",
        text=bank_perf["Bank"],
        textposition="top center",
        marker=dict(
            size=22,
            color=bank_perf["Node Stress"],
            colorscale=[[0, PALETTE["green"]], [0.5, PALETTE["amber"]], [1.0, PALETTE["red"]]],
            cmin=0, cmax=100,
            colorbar=dict(title="Node Stress", tickfont=dict(color=PALETTE["muted"])),
            line=dict(width=2, color=PALETTE["surface"]),
        ),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Drawdown: %{x:.1%}<br>"
            "Volatility: %{y:.1%}<extra></extra>"
        ),
    ))
    scatter.update_layout(
        title="Banks: Volatility vs Drawdown (colored by Node Stress)",
        xaxis_title="63D Drawdown",
        yaxis_title="21D Annualized Volatility",
        xaxis_tickformat=".1%",
        yaxis_tickformat=".1%",
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        height=440,
    )
    st.plotly_chart(scatter, use_container_width=True)

    returns_fig = go.Figure(go.Bar(
        x=bank_perf["Bank"],
        y=bank_perf["21D Return"],
        text=[signed_pct(x) for x in bank_perf["21D Return"]],
        textposition="outside",
        marker=dict(
            color=bank_perf["21D Return"],
            colorscale=[[0, PALETTE["red"]], [0.5, PALETTE["muted"]], [1.0, PALETTE["green"]]],
            showscale=False,
        ),
    ))
    returns_fig.add_hline(y=0, line_color=PALETTE["border"])
    returns_fig.update_layout(
        title="21-Day Return by Bank",
        yaxis_tickformat=".1%",
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        height=380,
    )
    st.plotly_chart(returns_fig, use_container_width=True)

    display = bank_perf[[
        "Bank", "Name", "21D Return", "21D Volatility", "63D Drawdown", "Node Stress", "Action Readout", "Economic Lens"
    ]].copy()
    for col in ["21D Return", "21D Volatility", "63D Drawdown"]:
        display[col] = display[col].map(lambda x: pct(x) if pd.notna(x) else "N/A")
    display["Node Stress"] = display["Node Stress"].map(lambda x: f"{x:.1f}/100")
    st.dataframe(display, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Macro Context for Canadian Banks")
    c1, c2 = st.columns(2)

    with c1:
        if not macro.empty:
            fig_rates = go.Figure()
            for col, label, color in [
                ("policy_rate", "Policy Rate", PALETTE["blue"]),
                ("ca_2y", "2Y Yield", PALETTE["green"]),
                ("ca_10y", "10Y Yield", PALETTE["amber"]),
            ]:
                if col in macro:
                    fig_rates.add_trace(go.Scatter(
                        x=macro.index, y=macro[col], mode="lines", name=label,
                        line=dict(color=color, width=2),
                    ))
            fig_rates.update_layout(
                title="Canadian Rate Backdrop",
                yaxis_title="Percent (%)",
                **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
                height=420,
            )
            st.plotly_chart(fig_rates, use_container_width=True)

    with c2:
        if not prices.empty:
            cols_price = [c for c in ["XFN.TO", "XIU.TO", "CADUSD=X", "CL=F"] if c in prices]
            norm = prices[cols_price].tail(252).ffill()
            norm = norm.divide(norm.iloc[0]).mul(100)
            fig_ctx = go.Figure()
            colors_ctx = [PALETTE["blue"], PALETTE["green"], PALETTE["amber"], PALETTE["teal"]]
            for i, col in enumerate(cols_price):
                fig_ctx.add_trace(go.Scatter(
                    x=norm.index, y=norm[col], mode="lines", name=col,
                    line=dict(color=colors_ctx[i % len(colors_ctx)], width=2),
                ))
            fig_ctx.update_layout(
                title="One-Year Macro Context (Indexed to 100)",
                yaxis_title="Indexed to 100",
                **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
                height=420,
            )
            st.plotly_chart(fig_ctx, use_container_width=True)

    insight_card(
        "Economic Transmission",
        "Canadian banks sit between households, housing, businesses, markets, and policy rates. "
        "When yields move quickly, equity drawdowns deepen, and banks start moving together, "
        "the signal is no longer just about six stocks — it is a read on credit availability "
        "and economic confidence across the country.",
        status="teal",
    )

    if not macro.empty:
        macro_cols = [c for c in ["policy_rate", "ca_2y", "ca_5y", "ca_10y", "slope_10y_2y"] if c in macro.columns]
        latest_macro = macro[macro_cols].dropna(how="all").tail(1).T.reset_index()
        latest_macro.columns = ["Macro Field", "Latest Value"]
        latest_macro["Latest Value"] = latest_macro["Latest Value"].map(lambda x: f"{x:.3f}%")
        st.dataframe(latest_macro, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Fields Used on This Page")
    fields = [
        {"Field": "contagion_risk_score", "Meaning": "Composite 0–100 systemic stress score", "Investment Use": "Sets regime, defense posture, and bank budget"},
        {"Field": "avg_pairwise_corr_63d", "Meaning": "Average rolling bank-to-bank correlation (63D)", "Investment Use": "When >0.8, bank diversification is failing — reduce total bank exposure"},
        {"Field": "avg_bank_vol_21d", "Meaning": "Annualized 21D vol averaged across Big Six", "Investment Use": "Rising vol → tighten position sizing; peak vol → potential mean-reversion entry"},
        {"Field": "XFN.TO_ret_21d / drawdown_63d", "Meaning": "Sector ETF recent performance", "Investment Use": "Distinguishes bank-specific from sector-wide stress"},
        {"Field": "VIX_level", "Meaning": "Global volatility proxy", "Investment Use": "High VIX → international risk-off bleeds into Canadian financials"},
        {"Field": "slope_10y_2y", "Meaning": "Yield-curve slope proxy (10Y−2Y GoC)", "Investment Use": "Flat/inverted slope → NIM compression, recession signal → reduce bank longs"},
        {"Field": "CL=F_ret_21d", "Meaning": "WTI crude oil 21D return", "Investment Use": "Oil weakness pressures Canadian credit, CAD, and energy-linked bank loans"},
        {"Field": "CADUSD=X_ret_21d", "Meaning": "Canadian dollar 21D return", "Investment Use": "CAD weakness can signal capital outflow or weaker Canadian growth"},
        {"Field": "bank node stress", "Meaning": "Bank-level stress score (vol + drawdown + beta)", "Investment Use": "Rank which holding to trim or hedge first"},
        {"Field": "investment signals", "Meaning": "Multi-factor composite score per bank", "Investment Use": "Explicit BUY/HOLD/REDUCE signals with conviction and target weights"},
    ]
    st.dataframe(pd.DataFrame(fields), use_container_width=True, hide_index=True)

st.caption(
    "Educational research dashboard. Simulated paper portfolio only. "
    "Not investment advice, not a trading bot, and not a regulatory stress-testing model."
)
