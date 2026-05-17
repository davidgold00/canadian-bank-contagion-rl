import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.dashboard.insight_utils import latest_valid_date, load_features
from src.dashboard.ui_components import analyst_header, apply_dashboard_style, insight_card


st.set_page_config(page_title="About | Canadian Bank Contagion RL", layout="wide")
apply_dashboard_style()

features = load_features()

analyst_header(
    "About the Canadian Bank Contagion Simulator",
    "A financial-engineering project that turns bank, macro, and network data into decisions.",
    date_text=latest_valid_date(features),
    source_text="Big Six banks, XFN/XIU, CAD, oil, gold, TSX, VIX, and BoC yields",
)

st.markdown(
    """
    Canada has a concentrated banking system. The Big Six banks touch mortgages, business credit,
    capital markets, consumer deposits, ETFs, pension portfolios, and the broader TSX. When they
    sell off together, the signal is bigger than a stock chart: it can point to tighter credit,
    housing pressure, weaker investor confidence, and more fragile diversification.
    """
)

insight_card(
    "Core Question",
    "When stress rises in Canadian financial markets, how might it spread across the Big Six banks, and how should exposure adapt?",
)

tab1, tab2, tab3, tab4 = st.tabs(["System Map", "Economic Meaning", "Dashboard Pages", "Limitations"])

with tab1:
    st.subheader("How the System Works")
    st.markdown(
        """
        1. **Data:** prices, rates, currency, commodities, volatility, housing templates, CDS templates, ETF holdings templates.
        2. **Features:** returns, rolling volatility, drawdowns, XFN beta, bank correlations, yield-curve slope, macro changes.
        3. **Network:** each bank is a node; correlations and stress relationships become edges.
        4. **Risk Score:** a 0-100 monitor that rises when stress channels cluster.
        5. **Scenarios:** macro shocks propagate through the bank network.
        6. **Portfolio Agent:** a defensive allocation policy moves between banks, ETFs, and cash.
        7. **Performance Tracker:** a simulated paper fund records target weights, trades, holdings, P&L, costs, and benchmarks.
        8. **Validation:** chronological tests check whether the ML layer has out-of-sample signal.
        """
    )

with tab2:
    st.subheader("Why the Data Matters")
    st.dataframe(
        [
            {
                "Signal": "Bank returns and drawdowns",
                "Economic Meaning": "Equity investors are marking down future earnings, credit risk, or confidence.",
                "Decision Use": "Identify whether pressure is bank-specific or sector-wide.",
            },
            {
                "Signal": "Bank correlation",
                "Economic Meaning": "The market is pricing shared macro risk rather than idiosyncratic stories.",
                "Decision Use": "Reduce false confidence in diversification.",
            },
            {
                "Signal": "Yield curve and policy rate",
                "Economic Meaning": "Rate levels affect funding costs, mortgage demand, net interest margins, and credit quality.",
                "Decision Use": "Explain whether stress is tied to the Canadian rate cycle.",
            },
            {
                "Signal": "CAD, oil, TSX, VIX",
                "Economic Meaning": "Canada-sensitive macro and global risk appetite indicators.",
                "Decision Use": "Separate domestic bank pressure from global risk-off pressure.",
            },
            {
                "Signal": "ETF holdings and CDS templates",
                "Economic Meaning": "Potential forced-selling and credit-spread channels not fully visible in stock returns.",
                "Decision Use": "Extend the model with ownership overlap and market-implied credit risk.",
            },
        ],
        use_container_width=True,
        hide_index=True,
    )

with tab3:
    st.subheader("What Each Page Answers")
    st.markdown(
        """
        - **Market Overview:** What is the current risk regime and what should a decision-maker do today?
        - **Systemic Bank Network:** Which banks are central contagion channels?
        - **Contagion Risk Score:** Which features are pushing systemic risk higher?
        - **Stress Testing Lab:** What happens under housing, oil, liquidity, rate, global, or bank-specific shocks?
        - **RL Portfolio Agent:** How would a risk-aware allocation change across banks, ETFs, and cash?
        - **Model Validation:** Is the ML layer learning useful future stress signal?
        - **Performance Tracker:** If the model gave daily weights, what would a fake-money portfolio have held and traded?
        - **Data Catalog:** What does every CSV mean, and what chart explains it?
        """
    )

with tab4:
    st.subheader("What This Is Not")
    st.warning(
        """
        This is educational financial-engineering research. It is not investment advice, not a
        trading bot, not a regulatory stress-testing model, not a solvency assessment of any
        Canadian bank, and not a promise that historical relationships will hold in the future.
        The Performance Tracker is a simulated paper portfolio only; no real trades are placed.
        """
    )
    st.markdown(
        """
        The most important model risks are public-data limitations, proxy features for credit stress,
        simplified stress propagation, backtest overfitting, transaction-cost assumptions, and regime
        changes that historical data cannot fully anticipate.
        """
    )
