from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent
PAGES = ROOT / "pages"

st.set_page_config(
    page_title="Northern Signal",
    page_icon="◆",
    layout="wide",
)

navigation = st.navigation(
    {
        "Overview": [
            st.Page(PAGES / "0_Overview.py", title="Current conditions", icon=":material/home:", default=True),
            st.Page(PAGES / "1_Executive_Market_Overview.py", title="Market evidence", icon=":material/query_stats:"),
        ],
        "Risk": [
            st.Page(PAGES / "2_Systemic_Bank_Network.py", title="Network", icon=":material/hub:"),
            st.Page(PAGES / "3_Contagion_Risk_Score.py", title="Composite score", icon=":material/speed:"),
        ],
        "Scenarios": [
            st.Page(PAGES / "4_Stress_Testing_Lab.py", title="Scenario analysis", icon=":material/experiment:"),
        ],
        "Models": [
            st.Page(PAGES / "5_RL_Portfolio_Agent.py", title="RL strategy", icon=":material/psychology:"),
            st.Page(PAGES / "9_CVaR_Optimization_Lab.py", title="CVaR strategy", icon=":material/tune:"),
            st.Page(PAGES / "11_RL_vs_CVaR_Comparison.py", title="Comparison", icon=":material/compare_arrows:"),
            st.Page(PAGES / "6_Model_Validation.py", title="Validation", icon=":material/fact_check:"),
        ],
        "Decision": [
            st.Page(PAGES / "12_Investment_Decision_Center.py", title="Portfolio decision", icon=":material/assignment_turned_in:"),
        ],
        "Performance": [
            st.Page(PAGES / "7_Performance_Tracker.py", title="Strategy performance", icon=":material/monitoring:"),
            st.Page(PAGES / "10_CVaR_Paper_Fund.py", title="Paper portfolio", icon=":material/account_balance_wallet:"),
        ],
        "Research": [
            st.Page(PAGES / "8_Data_Catalog.py", title="Data catalog", icon=":material/database:"),
            st.Page(PAGES / "0_About.py", title="Methodology & limitations", icon=":material/menu_book:"),
            st.Page(PAGES / "13_Business_Value_Center.py", title="Decision context", icon=":material/domain:"),
        ],
    },
    position="sidebar",
    expanded=True,
)

navigation.run()
