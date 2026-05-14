import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import streamlit as st, pandas as pd
from pathlib import Path
from src.data.build_dataset import build_dataset
st.set_page_config(page_title='Canadian Bank Contagion Simulator Using Graph Reinforcement Learning', layout='wide')
st.title('Canadian Bank Contagion Simulator Using Graph Reinforcement Learning')
st.info("""
This dashboard answers one core question:

When stress rises in Canadian financial markets, how might risk spread across the Big Six banks, and how should portfolio exposure adapt?

Use the sidebar pages to move from market conditions → bank network risk → contagion score → stress scenarios → RL portfolio response.
""")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "What this monitors",
        "Canadian bank contagion",
        help="A systemic-risk view of the Big Six Canadian banks using market, macro, and graph features."
    )

with col2:
    st.metric(
        "Main risk signal",
        "0–100 contagion score",
        help="Higher values mean bank stress, correlation, drawdown, and macro-risk features are elevated."
    )

with col3:
    st.metric(
        "Decision layer",
        "RL allocation agent",
        help="A reinforcement learning agent attempts to reduce exposure during stress while preserving upside."
    )
if not Path('data/processed/model_dataset.csv').exists(): build_dataset()
df=pd.read_csv('data/processed/model_dataset.csv',parse_dates=['date'],index_col='date')
st.metric('Latest Contagion Risk Score', f"{df['contagion_risk_score'].iloc[-1]:.1f}/100")
st.line_chart(df['contagion_risk_score'])
st.caption('Educational research dashboard. Not investment advice.')
