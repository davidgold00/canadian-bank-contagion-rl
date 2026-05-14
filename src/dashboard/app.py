import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import streamlit as st, pandas as pd
from pathlib import Path
from src.data.build_dataset import build_dataset
st.set_page_config(page_title='Canadian Bank Contagion Simulator Using Graph Reinforcement Learning', layout='wide')
st.title('Canadian Bank Contagion Simulator Using Graph Reinforcement Learning')
if not Path('data/processed/model_dataset.csv').exists(): build_dataset()
df=pd.read_csv('data/processed/model_dataset.csv',parse_dates=['date'],index_col='date')
st.metric('Latest Contagion Risk Score', f"{df['contagion_risk_score'].iloc[-1]:.1f}/100")
st.line_chart(df['contagion_risk_score'])
st.caption('Educational research dashboard. Not investment advice.')
