import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.dashboard.insight_utils import (
    BANKS,
    bank_stress_snapshot,
    latest,
    latest_valid_date,
    load_features,
    pct,
    percentile_rank,
    previous,
    risk_regime,
    strongest_drivers,
)
from src.dashboard.ui_components import action_list, analyst_header, apply_dashboard_style, insight_card


st.set_page_config(page_title="Contagion Risk Score", layout="wide")
apply_dashboard_style()

features = load_features()


def component_scores(df: pd.DataFrame) -> pd.DataFrame:
    candidates = {
        "Bank volatility": ("avg_bank_vol_21d", 1, "How jumpy bank returns are."),
        "Bank correlation": ("avg_pairwise_corr_63d", 1, "Whether banks are moving together."),
        "Financials drawdown": ("XFN.TO_drawdown_63d", -1, "How far the sector ETF has fallen from a recent high."),
        "Global volatility": ("VIX_level", 1, "Global risk appetite and liquidity pressure."),
        "Volatility spike": ("VIX_chg_5d", 1, "How quickly fear is rising."),
        "Yield curve pressure": ("slope_10y_2y", -1, "Curve flattening or inversion pressure."),
        "Oil shock": ("CL=F_ret_21d", -1, "Canada-linked macro and credit sentiment."),
        "CAD pressure": ("CADUSD=X_ret_21d", -1, "Currency weakness as macro stress proxy."),
    }
    out = pd.DataFrame(index=df.index)
    labels = {}
    for label, (col, sign, meaning) in candidates.items():
        if col in df:
            s = sign * df[col]
            out[label] = 100 * s.rank(pct=True)
            labels[label] = meaning
    if "contagion_risk_score" in df:
        out["Composite score"] = df["contagion_risk_score"]
        labels["Composite score"] = "The dashboard's overall 0-100 stress score."
    return out.ffill().fillna(50).clip(0, 100), labels


def gauge(score: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "/100"},
            title={"text": "Composite Contagion Risk"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#18212f"},
                "steps": [
                    {"range": [0, 30], "color": "#e9f7ef"},
                    {"range": [30, 60], "color": "#fff4df"},
                    {"range": [60, 80], "color": "#fdebd3"},
                    {"range": [80, 100], "color": "#fdecec"},
                ],
            },
        )
    )
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=20))
    return fig


components, labels = component_scores(features)
score_series = features["contagion_risk_score"] if "contagion_risk_score" in features else components.mean(axis=1)
score = float(score_series.dropna().iloc[-1])
regime = risk_regime(score)
score_pct = percentile_rank(score_series, score)
score_5d = score - previous(features, "contagion_risk_score", 5, default=np.nan)
score_21d = score - previous(features, "contagion_risk_score", 21, default=np.nan)

analyst_header(
    "Contagion Risk Score",
    "A 0-100 answer to: are Canadian bank stress signals clustering?",
    date_text=latest_valid_date(features),
    source_text="Composite of market, network, and macro stress features",
)

st.markdown(
    """
    The score is a risk dashboard signal, not a crash forecast. It becomes important when several
    indicators agree: volatility rises, correlations tighten, financials draw down, and macro
    stress makes bank earnings or credit quality less forgiving.
    """
)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Latest Score", f"{score:.1f}/100")
c2.metric("Regime", regime["label"], help=regime["summary"])
c3.metric("Historical Percentile", f"{score_pct:.0%}" if score_pct == score_pct else "N/A")
c4.metric("5D Change", f"{score_5d:+.1f}" if score_5d == score_5d else "N/A")
c5.metric("21D Change", f"{score_21d:+.1f}" if score_21d == score_21d else "N/A")
c6.metric("Latest Date", latest_valid_date(features))

left, right = st.columns([0.40, 0.60])
with left:
    st.plotly_chart(gauge(score), use_container_width=True)
with right:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=score_series.index, y=score_series, mode="lines", name="Score"))
    fig.add_hline(y=30, line_dash="dot", annotation_text="Low")
    fig.add_hline(y=60, line_dash="dot", annotation_text="High")
    fig.add_hline(y=80, line_dash="dot", annotation_text="Severe")
    fig.update_layout(
        title="Score History",
        yaxis_title="0-100",
        height=330,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

insight_card(f"Interpretation: {regime['label']} Risk", regime["summary"], status=regime["tone"])

tab1, tab2, tab3, tab4 = st.tabs(["Driver Decomposition", "Bank Contributors", "Regime Map", "Decision Rules"])

with tab1:
    st.subheader("What Is Driving the Score Today?")
    latest_components = components.iloc[-1].sort_values(ascending=True)
    fig = go.Figure(
        go.Bar(
            x=latest_components.values,
            y=latest_components.index,
            orientation="h",
            text=[f"{v:.1f}" for v in latest_components.values],
            textposition="auto",
            marker_color="#1d5f8f",
        )
    )
    fig.update_layout(
        title="Current Component Scores",
        xaxis_title="Stress contribution, 0-100",
        height=520,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    drivers = strongest_drivers(features)
    show = drivers.copy()
    show["Stress Percentile"] = show["Stress Percentile"].map(lambda x: f"{x:.0%}" if x == x else "N/A")
    st.dataframe(show, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Bank-Level Contributors")
    bank_table = bank_stress_snapshot(features)
    show = bank_table[
        ["Bank", "Name", "Node Stress", "21D Return", "21D Volatility", "63D Drawdown", "Beta to XFN", "Action Readout"]
    ].copy()
    for col in ["21D Return", "21D Volatility", "63D Drawdown"]:
        show[col] = show[col].map(lambda x: pct(x) if x == x else "N/A")
    show["Node Stress"] = show["Node Stress"].map(lambda x: f"{x:.1f}/100")
    show["Beta to XFN"] = show["Beta to XFN"].map(lambda x: f"{x:.2f}" if x == x else "N/A")
    st.dataframe(show, use_container_width=True, hide_index=True)

    leader = bank_table.iloc[0]
    insight_card(
        "Highest Attention Name",
        f"{leader['Bank']} has the highest current node stress at {leader['Node Stress']:.1f}/100. "
        "That means its recent market behavior is more stressed than peers; it is not a claim about bank solvency.",
        status="warning" if leader["Node Stress"] < 70 else "danger",
    )

with tab3:
    st.subheader("Stress Breadth Over the Last Six Months")
    recent = components.tail(126).T
    fig = go.Figure(
        go.Heatmap(
            z=recent.values,
            x=recent.columns,
            y=recent.index,
            zmin=0,
            zmax=100,
            colorscale="RdYlGn_r",
            colorbar=dict(title="Stress"),
        )
    )
    fig.update_layout(height=560, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        "Broad horizontal red/orange bands mean stress is coming from several channels at once. "
        "A single red cell is a warning, but broad stress is what makes contagion more plausible."
    )

with tab4:
    st.subheader("Decision Rules")
    action_list("Current Regime Actions", regime["actions"])
    st.dataframe(
        [
            {
                "Score Band": "0-30",
                "Label": "Low",
                "Business Meaning": "Normal bank-market noise.",
                "Portfolio Posture": "Diversified bank exposure can be evaluated mainly on fundamentals.",
            },
            {
                "Score Band": "30-60",
                "Label": "Moderate",
                "Business Meaning": "Some stress channels are active.",
                "Portfolio Posture": "Avoid adding concentration; monitor correlations and drawdowns.",
            },
            {
                "Score Band": "60-80",
                "Label": "High",
                "Business Meaning": "Systemic pressure is elevated.",
                "Portfolio Posture": "Trim high-stress names, raise liquidity, run scenario tests.",
            },
            {
                "Score Band": "80-100",
                "Label": "Severe",
                "Business Meaning": "Multiple stress channels are flashing.",
                "Portfolio Posture": "Defensive allocation dominates until breadth improves.",
            },
        ],
        use_container_width=True,
        hide_index=True,
    )
