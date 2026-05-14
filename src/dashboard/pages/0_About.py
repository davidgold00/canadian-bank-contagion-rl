import streamlit as st

st.set_page_config(page_title="About | Canadian Bank Contagion RL", layout="wide")

st.title("Canadian Bank Contagion Simulator Using Graph Reinforcement Learning")

st.markdown("""
## What is this project?

This dashboard models how financial stress could spread across the Big Six Canadian banks:

- Royal Bank of Canada
- TD Bank
- Bank of Montreal
- Scotiabank
- CIBC
- National Bank

It combines market data, macro stress features, dynamic bank correlation networks, systemic-risk scoring, supervised machine learning, stress scenario simulation, and reinforcement learning portfolio allocation.

The goal is not to predict the future perfectly. The goal is to demonstrate how a risk team, quant team, or ML engineer could build a prototype system for monitoring Canadian bank contagion risk.
""")

st.markdown("---")

st.header("Why this matters")

st.markdown("""
Canada is unusually exposed to a small number of large banks. These banks are affected by common macro risks:

- housing stress
- mortgage arrears
- yield curve shocks
- oil prices
- CAD/USD movements
- equity drawdowns
- volatility spikes
- credit stress
- ETF ownership concentration

When stress rises, banks may become more correlated. Diversification can break down. A portfolio that looks balanced in normal markets may become concentrated during crisis conditions.

This project tries to answer:

> When Canadian bank stress increases, which banks are most vulnerable, how does stress propagate, and how should portfolio exposure adapt?
""")

st.markdown("---")

st.header("How the system works")

st.markdown("""
### 1. Data Layer

Collects or generates data for:

- Big Six Canadian bank prices
- XFN Canadian financials ETF
- broad Canadian equity ETF
- oil
- gold
- CAD/USD
- VIX
- yield curve proxies
- housing and mortgage stress proxies

If live data fails, the system falls back to sample data so the dashboard still runs.

### 2. Feature Layer

Creates financial features such as:

- daily, weekly, and monthly returns
- rolling volatility
- drawdowns
- beta to XFN
- beta to the market
- distance from 52-week high
- rolling correlations
- yield curve stress
- volatility stress

### 3. Graph Layer

Each bank is treated as a node in a network.

Edges represent relationships such as:

- return correlation
- tail-risk co-movement
- ETF ownership overlap
- balance-sheet similarity proxies
- shock transmission strength

This lets us view Canadian banks as an interconnected financial system rather than six isolated stocks.

### 4. Contagion Risk Layer

The system creates a composite contagion risk score from 0 to 100.

Higher values mean:

- bank volatility is elevated
- correlations are rising
- market drawdowns are worsening
- credit stress proxies are increasing
- macro stress indicators are deteriorating

### 5. Machine Learning Layer

Supervised models estimate the probability of future stress events.

These models are not meant to be perfect trading signals. They show whether historical stress features contain predictive information.

### 6. Reinforcement Learning Layer

The RL agent learns how to allocate across:

- Canadian bank stocks
- XFN
- broad market ETF
- cash

The agent is rewarded for:

- earning returns
- reducing drawdowns
- lowering exposure during contagion periods
- avoiding excessive turnover
- preserving upside after stress falls
""")

st.markdown("---")

st.header("What each dashboard page means")

st.subheader("Market Overview")
st.markdown("""
Shows the broad financial environment.

Use this page to answer:

- Are Canadian banks rising or falling?
- Is XFN under pressure?
- Are volatility and macro stress rising?
- Is the contagion score elevated?

This is the executive summary page.
""")

st.subheader("Bank Network")
st.markdown("""
Shows the Big Six banks as a financial network.

Use this page to answer:

- Which banks are most connected?
- Which banks may transmit stress to others?
- Are correlations becoming concentrated?
- Is one bank becoming systemically important?

This is the graph ML / systemic-risk page.
""")

st.subheader("Contagion Risk")
st.markdown("""
Explains the current systemic-risk score.

Use this page to answer:

- Is risk low, moderate, or high?
- Which features are driving risk?
- Are stress indicators rising together?
- Which banks have elevated node stress?

This is the risk-monitoring page.
""")

st.subheader("Stress Scenarios")
st.markdown("""
Lets you simulate hypothetical shocks.

Examples:

- housing crisis
- oil crash
- liquidity squeeze
- yield curve inversion
- global risk-off event
- bank-specific shock

Use this page to answer:

- What happens if a shock hits one bank?
- How does that shock propagate?
- Which banks become most stressed?
- What portfolio response would be defensive?

This is the stress-testing page.
""")

st.subheader("RL Portfolio Agent")
st.markdown("""
Shows how the reinforcement learning allocation agent behaves.

Use this page to answer:

- Does the agent reduce bank exposure during stress?
- Does it move into cash defensively?
- Does it outperform equal-weight banks?
- Does it reduce drawdowns?
- What allocation would it recommend today?

This is the quant portfolio management page.
""")

st.subheader("Model Evaluation")
st.markdown("""
Shows whether the models are useful.

Use this page to answer:

- Did the supervised model predict stress better than random?
- How did the RL agent compare to benchmarks?
- What were Sharpe, drawdown, and stress-period returns?
- Are results robust or fragile?

This is the validation and credibility page.
""")

st.markdown("---")

st.header("How to explain this project in interviews")

st.markdown("""
A strong interview explanation:

> I built a Canadian bank systemic-risk simulator that models the Big Six banks as a dynamic financial graph. It ingests market and macro data, engineers stress features, builds rolling correlation and contagion networks, creates a composite systemic-risk score, trains supervised models to predict stress events, and uses reinforcement learning to allocate a portfolio defensively under rising contagion risk.

Key technical talking points:

- time-series feature engineering
- graph-based financial networks
- dynamic adjacency matrices
- systemic-risk scoring
- stress scenario simulation
- leakage-aware train/test splitting
- reinforcement learning with custom Gymnasium environment
- Streamlit dashboard for interpretability
- fallback data pipeline for reproducibility
""")

st.warning("""
Educational research only. This is not investment advice, not a trading system, and not a production bank risk model.
""")
