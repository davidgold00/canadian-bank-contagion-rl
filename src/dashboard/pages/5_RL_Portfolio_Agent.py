import streamlit as st, pandas as pd
st.title('5 RL Portfolio Agent')
st.info('Run python scripts/build_features.py first. This page uses processed sample or live data when available.')
try:
    df=pd.read_csv('data/processed/model_dataset.csv',parse_dates=['date'],index_col='date')
    st.dataframe(df.tail(20))
except Exception as e:
    st.warning(str(e))
