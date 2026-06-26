import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.dashboard.insight_utils import latest, latest_valid_date, load_features, risk_regime  # noqa: E402
from src.dashboard.ui_components import (  # noqa: E402
    analyst_header,
    apply_dashboard_style,
    business_value_panel,
    decision_callout,
    decision_memo,
    insight_card,
    mandate_fit_table,
    page_intro,
    plain_english_expander,
)


st.set_page_config(page_title="Business Value Center", layout="wide")
apply_dashboard_style()

features = load_features()
score = latest(features, "contagion_risk_score", 50.0)
regime = risk_regime(score)
avg_corr = latest(features, "avg_pairwise_corr_63d", float("nan"))

analyst_header(
    "Business Value Center",
    "Why the project is useful even when passive indexes outperform the model over some periods.",
    date_text=latest_valid_date(features),
    source_text="Executive framing for portfolio, risk, and model-governance users",
)

page_intro(
    why=(
        "A passive index beating the model over a 10-year window is important evidence, but it is not the whole business question. "
        "This project is most valuable as a Canadian bank risk-governance and decision-support system: it explains concentration, "
        "stress transmission, tail risk, and the portfolio actions that follow."
    ),
    how=(
        "Use this page to separate <b>return-chasing questions</b> from <b>risk-management questions</b>. "
        "If the mandate is maximum growth, Nasdaq may be the right benchmark. If the mandate is Canadian financial exposure "
        "with explainable downside controls, this system answers a different and more business-relevant problem."
    ),
)

top = st.columns(4)
top[0].metric("Current Risk Regime", regime["label"])
top[1].metric("Contagion Score", f"{score:.1f}/100")
top[2].metric("Bank Correlation", f"{avg_corr:.2f}" if pd.notna(avg_corr) else "N/A")
top[3].metric("Primary Use", "Risk governance")

business_value_panel(
    intro=(
        "The model should not be sold as a universal return maximizer. Its stronger business case is that it converts "
        "Canadian bank market stress into a repeatable governance process: monitor, explain, stress test, constrain, and document."
    ),
)

decision_callout(
    plain_english=(
        "If Nasdaq or ZEB outperforms over 10 years, that means passive exposure was the better total-return choice for that period. "
        "It does <b>not</b> prove that a bank-risk system has no value. A business still needs to know when bank holdings are becoming "
        "one concentrated macro bet, how large a drawdown could be under stress, and what action should be taken before risk limits break."
    ),
    action=(
        "Position the project as a decision-support and governance layer for Canadian financial exposure. "
        "Judge it on clarity, risk controls, drawdown behavior, stress readiness, and explainability alongside return."
    ),
    tone="teal",
)

decision_memo(
    "Executive Value Memo",
    [
        {
            "Observation": "Passive benchmarks can outperform the model over long bull-market windows.",
            "Decision Implication": "Do not pitch the system as a generic growth-index replacement.",
            "Monitoring Trigger": "If the business objective is only maximum return, prefer transparent passive benchmarks.",
        },
        {
            "Observation": "Canadian bank holdings can become highly correlated during stress.",
            "Decision Implication": "A portfolio can look diversified by name count while carrying one shared sector risk.",
            "Monitoring Trigger": "Escalate when average bank correlation rises above 0.75 and drawdowns widen.",
        },
        {
            "Observation": "The dashboard produces explanations, triggers, and trade reasons.",
            "Decision Implication": "This makes the system usable for committees, audits, and client conversations.",
            "Monitoring Trigger": "Reject recommendations that cannot be explained in business language.",
        },
        {
            "Observation": "CVaR and stress tests focus on bad outcomes, not just average returns.",
            "Decision Implication": "The project is valuable when downside tolerance, concentration limits, and governance matter.",
            "Monitoring Trigger": "Use CVaR, max drawdown, and stress P&L beside ending value before approving allocation changes.",
        },
    ],
    tone="teal",
)

tab1, tab2, tab3, tab4 = st.tabs(["Mandate Fit", "Business Workflows", "Simplified System Map", "Improvement Roadmap"])

with tab1:
    st.subheader("Use the Right Benchmark for the Right Question")
    mandate_fit_table()
    insight_card(
        "The Core Product Claim",
        "This is not 'AI beats Nasdaq.' The stronger claim is: 'This gives a Canadian-bank portfolio process better risk awareness, "
        "clearer explanations, repeatable stress testing, and more disciplined exposure controls.'",
        status="info",
    )

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Evaluation Lens": "Return",
                    "Metric": "Ending value / cumulative return",
                    "What Good Looks Like": "Competitive with passive bank-sector benchmarks after costs.",
                },
                {
                    "Evaluation Lens": "Risk efficiency",
                    "Metric": "Sharpe, Sortino, Calmar",
                    "What Good Looks Like": "Similar or better return per unit of risk than bank-sector alternatives.",
                },
                {
                    "Evaluation Lens": "Downside control",
                    "Metric": "Max drawdown, rolling CVaR, stress P&L",
                    "What Good Looks Like": "Shallower losses during Canadian bank stress regimes.",
                },
                {
                    "Evaluation Lens": "Governance",
                    "Metric": "Decision memo, triggers, explainability",
                    "What Good Looks Like": "A committee can understand why exposure changed and what would reverse the decision.",
                },
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

with tab2:
    st.subheader("Who Would Use This and Why")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Business User": "Portfolio manager",
                    "Workflow": "Check regime, compare bank stress, decide whether to add, hold, trim, or hedge.",
                    "Business Value": "Avoids treating all Big Six exposure as equally attractive.",
                },
                {
                    "Business User": "Risk committee",
                    "Workflow": "Review CVaR, max drawdown, stress scenario P&L, and monitoring triggers.",
                    "Business Value": "Creates an auditable record for concentration and tail-risk decisions.",
                },
                {
                    "Business User": "Research analyst",
                    "Workflow": "Study drivers, feature importance, correlation regimes, and model validation.",
                    "Business Value": "Turns model outputs into testable investment hypotheses.",
                },
                {
                    "Business User": "Advisor or client-facing team",
                    "Workflow": "Explain why a bank sleeve is being de-risked or why cash is temporarily higher.",
                    "Business Value": "Supports clear communication without requiring technical ML knowledge.",
                },
                {
                    "Business User": "Model governance",
                    "Workflow": "Inspect assumptions, limitations, raw data, no-lookahead design, and validation diagnostics.",
                    "Business Value": "Separates explainable decision support from black-box trading claims.",
                },
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

with tab3:
    st.subheader("The System in Plain English")
    st.markdown(
        """
        1. **Watch the market:** gather bank prices, sector ETFs, macro indicators, VIX, CAD, oil, and rates.
        2. **Translate noise into signals:** calculate returns, volatility, drawdowns, beta, and correlations.
        3. **Measure contagion:** detect when banks are moving together and diversification is weakening.
        4. **Explain the regime:** convert the evidence into a 0-100 risk score and business-language driver table.
        5. **Stress the portfolio:** ask how housing, oil, liquidity, rates, or global risk-off scenarios would hurt holdings.
        6. **Constrain the action:** use CVaR, graph risk, cash limits, and single-name caps before recommending weights.
        7. **Compare honestly:** challenge the model against passive indexes, bank ETFs, and individual banks.
        8. **Document the decision:** show what changed, why it matters, and what monitoring trigger should prompt the next review.
        """
    )
    plain_english_expander(
        "Optional Glossary for Business Users",
        [
            ("Mandate", "The actual job the portfolio is supposed to do, such as growth, income, capital preservation, or risk control."),
            ("Risk budget", "The amount of volatility, drawdown, or tail loss the business is willing to tolerate."),
            ("Benchmark", "The passive alternative the model must justify itself against."),
            ("Governance", "The process for explaining, approving, monitoring, and reversing investment decisions."),
            ("Tail risk", "Large adverse moves that happen rarely but matter most to capital preservation."),
        ],
    )

with tab4:
    st.subheader("What Would Make This More Valuable to a Business")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Priority": "1",
                    "Enhancement": "Mandate-specific backtests",
                    "Why It Matters": "Compare against bank-sector, income, low-volatility, and capital-preservation mandates separately.",
                },
                {
                    "Priority": "2",
                    "Enhancement": "Explicit risk-limit policy",
                    "Why It Matters": "Connect score bands to formal exposure, cash, and stop-review thresholds.",
                },
                {
                    "Priority": "3",
                    "Enhancement": "Client-ready PDF decision memo",
                    "Why It Matters": "Turns the dashboard into a repeatable governance artifact.",
                },
                {
                    "Priority": "4",
                    "Enhancement": "Richer credit and balance-sheet data",
                    "Why It Matters": "Public equity data is useful, but real bank risk decisions need credit quality, capital, liquidity, and funding data.",
                },
                {
                    "Priority": "5",
                    "Enhancement": "Model challenger framework",
                    "Why It Matters": "Compare CVaR against passive, low-vol, momentum, equal-risk, and discretionary rules before production use.",
                },
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
