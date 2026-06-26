import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.dashboard.insight_utils import (  # noqa: E402
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
    strongest_drivers,
)
from src.dashboard.ui_components import (  # noqa: E402
    PALETTE,
    PLOTLY_TEMPLATE,
    action_list,
    analyst_header,
    apply_dashboard_style,
    business_value_panel,
    decision_memo,
    insight_card,
    mandate_fit_table,
    page_intro,
)


st.set_page_config(
    page_title="Canadian Bank Contagion Command Center",
    layout="wide",
)
apply_dashboard_style()

features = load_features()
prices = load_prices()
macro = load_macro()

score = latest(features, "contagion_risk_score", 50)
regime = risk_regime(score)
score_21d_ago = previous(features, "contagion_risk_score", 21)
score_delta = score - score_21d_ago if score_21d_ago == score_21d_ago else None

analyst_header(
    "Canadian Bank Contagion Command Center",
    "A plain-English risk dashboard for the Big Six banks, the Canadian economy, and portfolio action.",
    date_text=latest_valid_date(features),
    source_text="Yahoo Finance prices + Bank of Canada yields + reproducible fallbacks",
)

page_intro(
    why=(
        "Canada's Big Six banks are deeply interconnected — when one is under stress, the others often follow. "
        "This dashboard tracks when that risk is rising, which banks are most exposed, and what a portfolio should do about it."
    ),
    how=(
        "Use the sidebar to navigate to any page. Start here for today's risk regime, then go to "
        "<b>Executive Market Overview</b> for signals and to <b>Investment Decision Center</b> for explicit portfolio actions."
    ),
)

business_value_panel(
    title="Executive Value Proposition",
    intro=(
        "This command center is not trying to prove that an active model always beats Nasdaq or ZEB. "
        "It gives a business a disciplined process for monitoring Canadian bank contagion risk, explaining exposure changes, "
        "and deciding when passive risk is acceptable versus when downside controls matter more."
    ),
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Contagion Risk", f"{score:.1f}/100", delta=signed_num(score_delta, 1) if score_delta is not None else None)
m2.metric("Regime", regime["label"], help=regime["summary"])
m3.metric("Avg Bank 21D Vol", pct(latest(features, "avg_bank_vol_21d")))
m4.metric("Avg Bank Correlation", f"{latest(features, 'avg_pairwise_corr_63d'):.2f}")

left, right = st.columns([0.62, 0.38])

with left:
    score_series = features["contagion_risk_score"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=score_series.index, y=score_series, mode="lines", name="Contagion risk",
        line=dict(color=PALETTE["blue"], width=2.5),
    ))
    fig.add_hrect(y0=0,  y1=30,  fillcolor="#00c853", opacity=0.06, line_width=0)
    fig.add_hrect(y0=30, y1=60,  fillcolor="#ffb300", opacity=0.06, line_width=0)
    fig.add_hrect(y0=60, y1=80,  fillcolor="#ff6f00", opacity=0.07, line_width=0)
    fig.add_hrect(y0=80, y1=100, fillcolor="#f44336", opacity=0.08, line_width=0)
    fig.add_hline(y=30, line_dash="dot", line_color="#00c853", opacity=0.5, annotation_text="Low / Moderate", annotation_font_color="#00c853")
    fig.add_hline(y=60, line_dash="dot", line_color="#ffb300", opacity=0.5, annotation_text="Moderate / High", annotation_font_color="#ffb300")
    fig.add_hline(y=80, line_dash="dot", line_color="#f44336", opacity=0.5, annotation_text="High / Severe", annotation_font_color="#f44336")
    fig.update_layout(
        title="Systemic Contagion Risk Score — History Since 2012",
        yaxis_title="Risk score (0 = no stress, 100 = maximum stress)",
        height=430,
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    insight_card(
        f"Current Readout: {regime['label']} Risk",
        regime["summary"],
        status=regime["tone"],
    )
    action_list("Immediate Decisions", regime["actions"])

drivers_now = strongest_drivers(features).head(6)
bank_table_now = bank_stress_snapshot(features)
top_driver = drivers_now.iloc[0] if not drivers_now.empty else None
top_bank = bank_table_now.iloc[0] if not bank_table_now.empty else None
decision_rows = [
    {
        "Observation": f"Contagion score {score:.1f}/100 ({regime['label']})",
        "Decision Implication": "Set the starting risk budget before looking at single-bank opportunities.",
        "Monitoring Trigger": "Revisit bank exposure when the score crosses 30, 60, or 80.",
    },
    {
        "Observation": f"Average bank correlation {latest(features, 'avg_pairwise_corr_63d'):.2f}",
        "Decision Implication": "High correlation turns several bank positions into one common factor trade.",
        "Monitoring Trigger": "If correlation rises with drawdowns, trim sector concentration before name selection.",
    },
]
if top_driver is not None:
    decision_rows.append(
        {
            "Observation": f"Top driver: {top_driver['Driver']} ({top_driver['Stress Percentile']:.0%})",
            "Decision Implication": top_driver["Why it matters"],
            "Monitoring Trigger": "Use this channel as the first scenario to stress before adding exposure.",
        }
    )
if top_bank is not None:
    decision_rows.append(
        {
            "Observation": f"Highest bank stress: {top_bank['Bank']} ({top_bank['Node Stress']:.1f}/100)",
            "Decision Implication": "This is the first due-diligence, trim, or hedge candidate in a sector selloff.",
            "Monitoring Trigger": "Escalate if it remains the highest-stress node for multiple weekly checks.",
        }
    )
decision_memo("Opening Decision Memo", decision_rows, tone=regime["tone"])

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "What Matters Now",
        "Bank-by-Bank View",
        "Economic Backdrop",
        "How the Pages Fit Together",
    ]
)

with tab1:
    drivers = strongest_drivers(features).head(6)
    st.subheader("Top Current Risk Drivers")
    st.markdown(
        "The table ranks today's stress drivers by where they sit versus their own history. "
        "A high percentile means that feature is unusually stressful."
    )
    show = drivers.copy()
    show["Stress Percentile"] = show["Stress Percentile"].map(lambda x: f"{x:.0%}" if x == x else "N/A")
    st.dataframe(show, use_container_width=True, hide_index=True)

    fig = go.Figure(
        go.Bar(
            x=drivers["Stress Percentile"],
            y=drivers["Driver"],
            orientation="h",
            text=[f"{x:.0%}" for x in drivers["Stress Percentile"]],
            marker_color="#1d5f8f",
        )
    )
    fig.update_layout(
        title="Stress Driver Percentiles",
        xaxis_tickformat=".0%",
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    bank_table = bank_stress_snapshot(features)
    st.subheader("Which Bank Needs the Most Attention?")
    st.markdown(
        "This is not a solvency ranking. It is a market-stress ranking built from volatility, drawdown, and sector beta."
    )
    show = bank_table.copy()
    for col in ["21D Return", "5D Return", "21D Volatility", "63D Drawdown"]:
        show[col] = show[col].map(lambda x: pct(x) if x == x else "N/A")
    show["Beta to XFN"] = show["Beta to XFN"].map(lambda x: f"{x:.2f}" if x == x else "N/A")
    show["Node Stress"] = show["Node Stress"].map(lambda x: f"{x:.1f}/100")
    st.dataframe(show, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Macro Context for Canadian Banks")
    if macro.empty:
        st.warning("No macro data is available.")
    else:
        cols = [c for c in ["policy_rate", "ca_2y", "ca_5y", "ca_10y", "slope_10y_2y"] if c in macro.columns]
        latest_macro = macro[cols].dropna(how="all").tail(1).T.reset_index()
        latest_macro.columns = ["Macro Field", "Latest Value"]
        st.dataframe(latest_macro, use_container_width=True, hide_index=True)

        fig = go.Figure()
        for col in [c for c in ["policy_rate", "ca_2y", "ca_10y"] if c in macro.columns]:
            fig.add_trace(go.Scatter(x=macro.index, y=macro[col], mode="lines", name=col))
        fig.update_layout(
            title="Canadian Rate Backdrop",
            yaxis_title="Percent",
            height=420,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        insight_card(
            "Economic Interpretation",
            "Banks benefit from credit growth and stable funding. A very flat curve can pressure lending margins, "
            "while rapid rate moves can stress borrowers and reprice bank equity risk.",
        )

with tab4:
    st.subheader("Dashboard Flow")
    st.markdown(
        """
        1. **Market Overview** explains today's broad risk regime.
        2. **Systemic Bank Network** shows whether the Big Six are moving as one connected cluster.
        3. **Contagion Risk Score** decomposes the 0-100 score into understandable drivers.
        4. **Stress Testing Lab** turns scenarios into bank and portfolio impacts.
        5. **RL Portfolio Agent** connects the risk signal to allocation behavior.
        6. **Model Validation** checks whether the ML layer has out-of-sample signal.
        7. **Performance Tracker** turns daily recommendations into a simulated paper portfolio with trades, holdings, costs, and benchmarks.
        8. **Data Catalog** explains every CSV and shows what each file contributes.
        9. **CVaR Optimization Lab** is the production-style graph-aware portfolio construction engine.
        10. **CVaR Paper Fund** simulates the optimized portfolio through time.
        11. **RL vs CVaR Comparison** studies when experimental RL or governed optimization behaves better.
        """
    )
    st.subheader("Mandate Fit")
    mandate_fit_table()

st.caption(
    "Educational research dashboard. Simulated paper portfolio only. Not investment advice, "
    "not a trading bot, and not a regulatory stress-testing model."
)
