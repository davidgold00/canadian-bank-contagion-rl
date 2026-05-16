import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.dashboard.insight_utils import (
    BANKS,
    BANK_NAMES,
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
from src.dashboard.ui_components import action_list, analyst_header, apply_dashboard_style, insight_card


st.set_page_config(page_title="Market Overview", layout="wide")
apply_dashboard_style()

features = load_features()
prices = load_prices()
macro = load_macro()

score = latest(features, "contagion_risk_score", 50)
regime = risk_regime(score)
score_delta = score - previous(features, "contagion_risk_score", 21)

analyst_header(
    "Executive Market Overview",
    "Today's Canadian bank risk regime, translated into business decisions.",
    date_text=latest_valid_date(features),
    source_text="Live market and rate data when available",
)

st.markdown(
    """
    Start here for the board-level readout. The goal is to separate normal bank-stock noise
    from conditions that can matter for the Canadian economy: tighter funding, weaker housing
    confidence, falling bank equity, and rising correlation across the sector.
    """
)

sector_ret_21d = latest(features, "XFN.TO_ret_21d")
sector_drawdown = latest(features, "XFN.TO_drawdown_63d")
avg_corr = latest(features, "avg_pairwise_corr_63d")
vix = latest(features, "VIX_level")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Risk Regime", regime["label"], help=regime["summary"])
m2.metric("Contagion Score", f"{score:.1f}/100", delta=signed_num(score_delta, 1))
m3.metric("XFN 21D Return", signed_pct(sector_ret_21d))
m4.metric("XFN 63D Drawdown", pct(sector_drawdown))
m5.metric("Bank Correlation", f"{avg_corr:.2f}", help="Higher correlation means bank diversification is less reliable.")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Executive Readout", "Bank Risk Map", "Economic Transmission", "Data Dictionary"]
)

with tab1:
    left, right = st.columns([0.62, 0.38])

    with left:
        score_series = features["contagion_risk_score"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=score_series.index, y=score_series, mode="lines", name="Score"))
        fig.add_hline(y=30, line_dash="dot", annotation_text="Low")
        fig.add_hline(y=60, line_dash="dot", annotation_text="High")
        fig.add_hline(y=80, line_dash="dot", annotation_text="Severe")
        fig.update_layout(
            title="Systemic Risk Score",
            yaxis_title="0-100 score",
            height=420,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        insight_card(f"{regime['label']} Risk", regime["summary"], status=regime["tone"])
        action_list("What to Do Now", regime["actions"])

    drivers = strongest_drivers(features).head(5)
    st.subheader("Why the Score Is Here")
    show = drivers.copy()
    show["Stress Percentile"] = show["Stress Percentile"].map(lambda x: f"{x:.0%}" if x == x else "N/A")
    st.dataframe(show, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Big Six Recent Performance")
    rows = []
    for bank in BANKS:
        rows.append(
            {
                "Bank": bank,
                "Name": BANK_NAMES[bank],
                "5D Return": latest(features, f"{bank}_ret_5d"),
                "21D Return": latest(features, f"{bank}_ret_21d"),
                "21D Volatility": latest(features, f"{bank}_vol_21d"),
                "63D Drawdown": latest(features, f"{bank}_drawdown_63d"),
            }
        )
    bank_perf = bank_stress_snapshot(features)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=bank_perf["Bank"],
            y=bank_perf["21D Return"],
            text=[signed_pct(x) for x in bank_perf["21D Return"]],
            textposition="auto",
            marker_color="#1d5f8f",
            name="21D return",
        )
    )
    fig.update_layout(
        title="21-Day Return by Bank",
        yaxis_tickformat=".1%",
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    display = bank_perf[
        ["Bank", "Name", "21D Return", "21D Volatility", "63D Drawdown", "Node Stress", "Action Readout", "Economic Lens"]
    ].copy()
    for col in ["21D Return", "21D Volatility", "63D Drawdown"]:
        display[col] = display[col].map(lambda x: pct(x) if x == x else "N/A")
    display["Node Stress"] = display["Node Stress"].map(lambda x: f"{x:.1f}/100")
    st.dataframe(display, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("How Market Stress Reaches the Canadian Economy")
    c1, c2 = st.columns(2)

    with c1:
        if not macro.empty:
            fig = go.Figure()
            for col, label in [("policy_rate", "Policy rate"), ("ca_2y", "2Y yield"), ("ca_10y", "10Y yield")]:
                if col in macro:
                    fig.add_trace(go.Scatter(x=macro.index, y=macro[col], mode="lines", name=label))
            fig.update_layout(
                title="Canadian Rate Backdrop",
                yaxis_title="Percent",
                height=410,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        if not prices.empty:
            cols = [c for c in ["XFN.TO", "XIU.TO", "^GSPTSE", "CADUSD=X", "CL=F"] if c in prices]
            normalized = prices[cols].tail(252).ffill()
            normalized = normalized.divide(normalized.iloc[0]).mul(100)
            fig = go.Figure()
            for col in cols:
                fig.add_trace(go.Scatter(x=normalized.index, y=normalized[col], mode="lines", name=col))
            fig.update_layout(
                title="One-Year Macro Market Context",
                yaxis_title="Indexed to 100",
                height=410,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

    insight_card(
        "Plain-English Link",
        "Canadian banks sit between households, housing, businesses, markets, and policy rates. "
        "When yields move quickly, equity drawdowns deepen, and banks start moving together, the signal is no longer just about six stocks; "
        "it is a read on credit availability and economic confidence.",
    )

with tab4:
    st.subheader("Fields Used on This Page")
    st.dataframe(
        [
            {
                "Field": "contagion_risk_score",
                "Meaning": "Composite 0-100 stress score.",
                "Action Use": "Sets monitoring intensity and defensive posture.",
            },
            {
                "Field": "avg_pairwise_corr_63d",
                "Meaning": "Average rolling bank-to-bank correlation.",
                "Action Use": "Shows when diversification across banks is failing.",
            },
            {
                "Field": "XFN.TO_ret_21d / drawdown_63d",
                "Meaning": "Recent Canadian financial-sector performance.",
                "Action Use": "Separates bank-specific moves from sector-wide stress.",
            },
            {
                "Field": "policy_rate, ca_2y, ca_10y",
                "Meaning": "Bank of Canada policy and bond-market rate backdrop.",
                "Action Use": "Explains funding, mortgage, and valuation pressure.",
            },
            {
                "Field": "bank node stress",
                "Meaning": "Bank-level volatility, drawdown, and sector beta summary.",
                "Action Use": "Ranks which holdings to trim, hedge, or investigate first.",
            },
        ],
        use_container_width=True,
        hide_index=True,
    )
