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
from src.dashboard.ui_components import PALETTE, PLOTLY_TEMPLATE, action_list, analyst_header, apply_dashboard_style, decision_callout, decision_memo, insight_card, page_intro


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
    needle_color = (
        PALETTE["red"] if score >= 80
        else PALETTE["amber_muted"] if score >= 60
        else PALETTE["amber"] if score >= 30
        else PALETTE["green"]
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "/100", "font": {"color": needle_color, "size": 36, "family": "JetBrains Mono, monospace"}},
        title={"text": "Composite Contagion Risk", "font": {"color": PALETTE["ink"], "size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": PALETTE["muted"], "tickfont": {"color": PALETTE["muted"]}},
            "bar": {"color": needle_color, "thickness": 0.25},
            "bgcolor": PALETTE["card"],
            "bordercolor": PALETTE["border"],
            "steps": [
                {"range": [0, 30],  "color": "#0d2318"},
                {"range": [30, 60], "color": "#1f1700"},
                {"range": [60, 80], "color": "#1f1000"},
                {"range": [80, 100],"color": "#1f0a08"},
            ],
            "threshold": {"line": {"color": PALETTE["red"], "width": 3}, "thickness": 0.75, "value": 80},
        },
    ))
    fig.update_layout(
        height=320,
        paper_bgcolor=PALETTE["surface"],
        plot_bgcolor=PALETTE["surface"],
        font=dict(color=PALETTE["ink"]),
        margin=dict(l=20, r=20, t=50, b=20),
    )
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
    "A 0–100 answer to: are Canadian bank stress signals clustering?",
    date_text=latest_valid_date(features),
    source_text="Composite of market, network, and macro stress features",
)

page_intro(
    why=(
        "The score answers one question: are multiple stress indicators rising <em>together</em>? "
        "When volatility, correlations, drawdowns, and macro pressure all climb at the same time, "
        "the risk of contagion across banks is highest."
    ),
    how=(
        "Read the gauge and the 5D/21D change to know whether risk is rising or falling. "
        "Scroll down to see which specific signals are driving the score, and what the current regime means for portfolio decisions."
    ),
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
    st.plotly_chart(gauge(score), width="stretch")
with right:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=score_series.index, y=score_series, mode="lines", name="Contagion Score",
        line=dict(color=PALETTE["blue"], width=2),
    ))
    fig.add_hrect(y0=0,  y1=30,  fillcolor="#00c853", opacity=0.05, line_width=0)
    fig.add_hrect(y0=30, y1=60,  fillcolor="#ffb300", opacity=0.05, line_width=0)
    fig.add_hrect(y0=60, y1=80,  fillcolor="#ff6f00", opacity=0.06, line_width=0)
    fig.add_hrect(y0=80, y1=100, fillcolor="#f44336", opacity=0.07, line_width=0)
    fig.add_hline(y=30, line_dash="dot", line_color="#00c853", opacity=0.6,
                  annotation_text="Low→Moderate", annotation_font_color="#00c853", annotation_position="top left")
    fig.add_hline(y=60, line_dash="dot", line_color="#ffb300", opacity=0.6,
                  annotation_text="Moderate→High", annotation_font_color="#ffb300", annotation_position="top left")
    fig.add_hline(y=80, line_dash="dot", line_color="#f44336", opacity=0.6,
                  annotation_text="High→Severe", annotation_font_color="#f44336", annotation_position="top left")
    fig.update_layout(
        title="Score History — How Has Risk Changed Over Time?",
        yaxis_title="0 = no stress, 100 = maximum stress",
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        height=330,
    )
    st.plotly_chart(fig, width="stretch")

insight_card(f"Interpretation: {regime['label']} Risk", regime["summary"], status=regime["tone"])

drivers_now = strongest_drivers(features)
top_driver = drivers_now.iloc[0] if not drivers_now.empty else None
memo_rows = [
    {
        "Observation": f"Score percentile {score_pct:.0%}" if score_pct == score_pct else "Score percentile unavailable",
        "Decision Implication": "Percentile tells you whether today is ordinary noise or a historically unusual stress state.",
        "Monitoring Trigger": "Treat readings above the 75th percentile as a constraint on new bank exposure.",
    },
    {
        "Observation": f"5D / 21D score change: {score_5d:+.1f} / {score_21d:+.1f}",
        "Decision Implication": "Direction matters as much as level; rising risk should reduce tolerance for new concentration.",
        "Monitoring Trigger": "Escalate when both short and medium windows are positive.",
    },
]
if top_driver is not None:
    memo_rows.append(
        {
            "Observation": f"Top driver: {top_driver['Driver']}",
            "Decision Implication": top_driver["Why it matters"],
            "Monitoring Trigger": "Use the top driver to choose the first scenario and hedge assumption.",
        }
    )
decision_memo("Score Decision Memo", memo_rows, tone=regime["tone"])

tab1, tab2, tab3, tab4 = st.tabs(["Driver Decomposition", "Bank Contributors", "Regime Map", "Decision Rules"])

with tab1:
    st.subheader("What Is Driving the Score Today?")
    st.markdown(
        "Each bar shows one stress driver ranked against its own history. "
        "A score of **100** means it is at its most stressed level ever recorded. "
        "A score of **50** is normal. Focus on the bars furthest to the right."
    )
    latest_components = components.iloc[-1].sort_values(ascending=True)
    fig = go.Figure(go.Bar(
        x=latest_components.values,
        y=latest_components.index,
        orientation="h",
        text=[f"{v:.1f}" for v in latest_components.values],
        textposition="auto",
        marker=dict(
            color=latest_components.values,
            colorscale=[[0, PALETTE["green"]], [0.5, PALETTE["amber"]], [1.0, PALETTE["red"]]],
            cmin=0, cmax=100, showscale=False,
        ),
    ))
    fig.update_layout(
        title="Stress Component Scores — 0 = Calm, 100 = Maximum Stress",
        xaxis_title="Percentile vs. own history (100 = highest stress ever seen for this driver)",
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        height=520,
    )
    st.plotly_chart(fig, width="stretch")
    decision_callout(
        plain_english="Drivers with scores above 75 are at their most stressed quartile historically. When several are above 75 simultaneously, the composite score rises rapidly.",
        action="Focus attention on the top 2–3 drivers. Those are the specific channels most likely to cause portfolio losses if conditions worsen.",
        tone="info",
    )

    drivers = drivers_now
    show = drivers.copy()
    show["Stress Percentile"] = show["Stress Percentile"].map(lambda x: f"{x:.0%}" if x == x else "N/A")
    st.dataframe(show, width="stretch", hide_index=True)

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
    st.dataframe(show, width="stretch", hide_index=True)

    leader = bank_table.iloc[0]
    insight_card(
        "Highest Attention Name",
        f"{leader['Bank']} has the highest current node stress at {leader['Node Stress']:.1f}/100. "
        "That means its recent market behavior is more stressed than peers; it is not a claim about bank solvency.",
        status="warning" if leader["Node Stress"] < 70 else "danger",
    )

with tab3:
    st.subheader("Stress Breadth Over the Last Six Months")
    st.markdown(
        "Each row is a stress driver; each column is a trading day. **Red** = that driver was highly stressed that day. "
        "Wide horizontal red bands mean stress was hitting multiple channels simultaneously — that is when contagion risk is highest."
    )
    recent = components.tail(126).T
    fig = go.Figure(go.Heatmap(
        z=recent.values,
        x=recent.columns,
        y=recent.index,
        zmin=0,
        zmax=100,
        colorscale="RdYlGn_r",
        colorbar=dict(title="Stress Level", tickfont=dict(color=PALETTE["muted"])),
    ))
    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        height=560,
    )
    st.plotly_chart(fig, width="stretch")
    decision_callout(
        plain_english="A single red row means one channel is stressed. Multiple red rows on the same day means stress is widespread — the most dangerous signal for contagion.",
        action="When 3 or more drivers are simultaneously in the red, treat the regime as elevated even if the composite score hasn't peaked yet.",
        tone="warning",
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
        width="stretch",
        hide_index=True,
    )
