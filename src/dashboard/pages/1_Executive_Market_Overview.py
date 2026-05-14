import streamlit as st, pandas as pd
st.title('1 Market Overview')
st.info("""
This page gives the macro and market context. It shows whether Canadian bank equities, XFN, oil, CAD, volatility, and yield indicators are moving into a stress regime.
""")
st.info('Run python scripts/build_features.py first. This page uses processed sample or live data when available.')
try:
    df=pd.read_csv('data/processed/model_dataset.csv',parse_dates=['date'],index_col='date')
    st.dataframe(df.tail(20))
except Exception as e:
    st.warning(str(e))
