import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.dashboard.components import disclaimer_box, format_currency, format_percent, load_price_data, load_processed_dataset  # noqa: E402
from src.dashboard.insight_utils import latest_valid_date  # noqa: E402
from src.dashboard.ui_components import PALETTE, PLOTLY_TEMPLATE, analyst_header, apply_dashboard_style, decision_callout, insight_card, page_intro  # noqa: E402
from src.portfolio.paper_trader import CVaRPaperPortfolioSimulator, PaperPortfolioSimulator  # noqa: E402
from src.portfolio.performance_metrics import drawdown_series, performance_summary, rolling_cvar, rolling_sharpe  # noqa: E402


st.set_page_config(page_title="CVaR Paper Fund", layout="wide")
apply_dashboard_style()


@st.cache_data(show_spinner="Simulating CVaR paper fund...")
def run_cvar_paper_fund(
    initial_capital,
    transaction_cost_bps,
    rebalance_threshold,
    start_date,
    confidence_level,
    lookback_window,
    rebalance_frequency,
    max_single_name_weight,
    max_bank_exposure,
    min_cash_weight,
    max_cash_weight,
    risk_aversion,
    turnover_penalty,
    contagion_penalty,
):
    prices = load_price_data()
    features = load_processed_dataset()
    cvar_sim = CVaRPaperPortfolioSimulator(prices, features, use_trained_model=False)
    cvar = cvar_sim.simulate_cvar(
        initial_capital=initial_capital,
        transaction_cost_bps=transaction_cost_bps,
        rebalance_threshold=rebalance_threshold,
        start_date=start_date,
        confidence_level=confidence_level,
        lookback_window=lookback_window,
        rebalance_frequency=rebalance_frequency,
        max_single_name_weight=max_single_name_weight,
        max_bank_exposure=max_bank_exposure,
        min_cash_weight=min_cash_weight,
        max_cash_weight=max_cash_weight,
        risk_aversion=risk_aversion,
        turnover_penalty=turnover_penalty,
        contagion_penalty=contagion_penalty,
    )
    rl = PaperPortfolioSimulator(prices, features, use_trained_model=True).simulate(
        initial_capital=initial_capital,
        transaction_cost_bps=transaction_cost_bps,
        rebalance_threshold=rebalance_threshold,
        start_date=start_date,
        max_single_name_weight=max_single_name_weight,
        max_bank_exposure=max_bank_exposure,
    )
    benchmarks = cvar.benchmarks.copy()
    benchmarks["RL research baseline"] = rl.ledger["portfolio_value"].reindex(benchmarks.index).ffill()
    return cvar, benchmarks


def value_chart(ledger: pd.DataFrame, benchmarks: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ledger.index, y=ledger["portfolio_value"], mode="lines", name="CVaR paper fund", line=dict(width=3, color="#1d5f8f")))
    for col in benchmarks.columns:
        fig.add_trace(go.Scatter(x=benchmarks.index, y=benchmarks[col], mode="lines", name=col))
    fig.update_layout(title="CVaR Paper Fund vs Benchmarks", xaxis_title="Date", yaxis_title="Portfolio value", height=500)
    return fig


def line_chart(series: dict[str, pd.Series], title: str, yaxis: str, tickformat=None) -> go.Figure:
    fig = go.Figure()
    for name, values in series.items():
        fig.add_trace(go.Scatter(x=values.index, y=values, mode="lines", name=name))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title=yaxis, height=420)
    if tickformat:
        fig.update_yaxes(tickformat=tickformat)
    return fig


def allocation_chart(weights: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col in weights.columns:
        fig.add_trace(go.Scatter(x=weights.index, y=weights[col], mode="lines", stackgroup="one", name=col))
    fig.update_layout(title="CVaR Allocation Through Time", yaxis_title="Weight", yaxis_tickformat=".0%", height=520)
    return fig


def format_holdings(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["asset", "shares", "latest_price", "market_value", "weight", "unrealized_pnl"]].copy()
    out.columns = ["Asset", "Shares", "Latest Price", "Market Value", "Weight", "Unrealized P&L"]
    out["Shares"] = out["Shares"].map(lambda x: f"{x:,.4f}")
    for col in ["Latest Price", "Market Value", "Unrealized P&L"]:
        out[col] = out[col].map(lambda x: format_currency(x, 2))
    out["Weight"] = out["Weight"].map(lambda x: format_percent(x, 2))
    return out


def format_trades(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.sort_values("date", ascending=False).copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out = out[["date", "asset", "action", "shares", "price", "notional", "transaction_cost", "reason"]]
    out.columns = ["Date", "Asset", "Action", "Shares", "Price", "Notional", "Transaction Cost", "Reason"]
    out["Shares"] = out["Shares"].map(lambda x: f"{x:,.4f}")
    for col in ["Price", "Notional", "Transaction Cost"]:
        out[col] = out[col].map(lambda x: format_currency(x, 2))
    return out


prices = load_price_data()
features = load_processed_dataset()

analyst_header(
    "CVaR Paper Fund",
    "Simulated paper portfolio that follows the graph-adjusted CVaR optimizer through time.",
    date_text=latest_valid_date(features),
    source_text="No broker connection; simulated trades and holdings only",
)

page_intro(
    why=(
        "The CVaR Optimization Lab shows you <em>today's</em> optimal allocation. "
        "This page runs the optimizer every rebalance period over years of history — "
        "showing whether the CVaR approach consistently outperforms simple benchmarks over time, "
        "especially during periods of bank stress."
    ),
    how=(
        "Compare the blue portfolio line against the benchmark lines. "
        "Focus on the <b>drawdown chart</b>: a good tail-risk strategy should show shallower drawdowns "
        "during stress periods (2015, 2020, 2022) even if it gives up some return in calm markets."
    ),
)
disclaimer_box()

with st.sidebar:
    st.header("Simulation Controls")
    initial_capital = st.number_input("Initial capital", min_value=10_000.0, max_value=10_000_000.0, value=100_000.0, step=10_000.0)
    transaction_cost_bps = st.slider("Transaction cost, bps", 0.0, 50.0, 5.0, step=0.5)
    rebalance_threshold = st.slider("Rebalance threshold", 0.0, 0.10, 0.01, step=0.005, format="%.3f")
    confidence_level = st.selectbox("CVaR confidence level", [0.95, 0.99], format_func=lambda x: f"{x:.0%}")
    lookback_window = st.selectbox("Lookback window", [63, 126, 252], index=1)
    rebalance_frequency = st.selectbox("Rebalance frequency, trading days", [1, 5, 10, 21], index=2)
    max_single_name_weight = st.slider("Max single-name weight", 0.05, 0.35, 0.20, step=0.01)
    max_bank_exposure = st.slider("Max Big Six exposure", 0.20, 1.00, 0.70, step=0.05)
    min_cash_weight = st.slider("Minimum cash", 0.00, 0.40, 0.05, step=0.01)
    max_cash_weight = st.slider("Maximum cash", 0.05, 0.90, 0.60, step=0.05)
    risk_aversion = st.slider("Risk aversion", 1.0, 20.0, 7.0, step=0.5)
    turnover_penalty = st.slider("Turnover penalty", 0.0, 2.0, 0.25, step=0.05)
    contagion_penalty = st.slider("Contagion penalty", 0.0, 3.0, 0.90, step=0.05)
    default_start = prices.index[max(252, int(len(prices) * 0.70))].date()
    start_date = st.date_input("Start date", value=default_start, min_value=prices.index.min().date(), max_value=prices.index.max().date())
    show_full_ledger = st.checkbox("Show full ledger", value=False)

result, benchmarks = run_cvar_paper_fund(
    float(initial_capital),
    float(transaction_cost_bps),
    float(rebalance_threshold),
    str(start_date),
    float(confidence_level),
    int(lookback_window),
    int(rebalance_frequency),
    float(max_single_name_weight),
    float(max_bank_exposure),
    float(min_cash_weight),
    float(max_cash_weight),
    float(risk_aversion),
    float(turnover_penalty),
    float(contagion_penalty),
)

ledger = result.ledger
summary = performance_summary(ledger["portfolio_value"], ledger["daily_return"], ledger["turnover"], ledger["transaction_costs"])
latest = ledger.iloc[-1]

cols = st.columns(6)
cols[0].metric("Current Value", format_currency(summary["ending_value"]))
cols[1].metric("Cumulative Return", format_percent(summary["cumulative_return"]))
cols[2].metric("Annualized Return", format_percent(summary["annualized_return"]))
cols[3].metric("Annualized Vol", format_percent(summary["annualized_volatility"]))
cols[4].metric("Sharpe", f"{summary['sharpe_ratio']:.2f}")
cols[5].metric("Max Drawdown", format_percent(summary["max_drawdown"]))

cols2 = st.columns(6)
cols2[0].metric("Sortino", f"{summary['sortino_ratio']:.2f}")
cols2[1].metric("Realized CVaR", format_percent(summary["conditional_value_at_risk"]))
cols2[2].metric("Avg Turnover", format_percent(summary["average_daily_turnover"]))
cols2[3].metric("Transaction Costs", format_currency(summary["total_transaction_costs"], 2))
cols2[4].metric("Cash Weight", format_percent(latest["cash_weight"]))
cols2[5].metric("Avg Bank Exposure", format_percent(ledger["bank_exposure"].mean()))

decision_callout(
    plain_english=(
        f"The CVaR fund returned <b>{format_percent(summary['cumulative_return'])}</b> cumulatively "
        f"with a Sharpe of <b>{summary['sharpe_ratio']:.2f}</b> and a worst drawdown of "
        f"<b>{format_percent(summary['max_drawdown'])}</b>. "
        f"Realized tail risk (CVaR) was <b>{format_percent(summary['conditional_value_at_risk'])}</b>. "
        "Compare these against the XFN and XIU benchmark lines in the chart below."
    ),
    action=(
        "If the CVaR fund shows shallower drawdowns than the XFN benchmark during stress periods, "
        "the tail-risk approach is delivering its intended benefit. "
        "If not, consider increasing the CVaR penalty or risk aversion in the sidebar controls."
    ),
    tone="success" if summary["sharpe_ratio"] > 0.4 else "warning",
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Risk Analytics", "Allocation & Turnover", "Trades & Holdings", "Methodology"])

with tab1:
    st.plotly_chart(value_chart(ledger, benchmarks), use_container_width=True)
    pnl = ledger["portfolio_value"] - float(initial_capital)
    st.plotly_chart(line_chart({"Cumulative P&L": pnl}, "Cumulative Simulated P&L", "CAD"), use_container_width=True)

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        drawdowns = {"CVaR paper fund": drawdown_series(ledger["portfolio_value"])}
        for col in benchmarks.columns:
            drawdowns[col] = drawdown_series(benchmarks[col])
        st.plotly_chart(line_chart(drawdowns, "Drawdown vs Benchmarks", "Drawdown", ".0%"), use_container_width=True)
        st.plotly_chart(line_chart({"Rolling CVaR": ledger["realized_cvar_63d"]}, "Rolling 63D CVaR", "CVaR", ".1%"), use_container_width=True)
    with c2:
        st.plotly_chart(line_chart({"Rolling volatility": ledger["realized_volatility_63d"]}, "Rolling 63D Volatility", "Volatility", ".1%"), use_container_width=True)
        st.plotly_chart(line_chart({"Rolling Sharpe": rolling_sharpe(ledger["daily_return"], 63)}, "Rolling 63D Sharpe", "Sharpe"), use_container_width=True)

with tab3:
    st.plotly_chart(allocation_chart(result.weights), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(line_chart({"Bank exposure": ledger["bank_exposure"], "Cash": ledger["cash_weight"]}, "Bank Exposure and Cash Weight", "Weight", ".0%"), use_container_width=True)
    with c2:
        st.plotly_chart(line_chart({"Turnover": ledger["turnover"], "Graph density": ledger["graph_density"]}, "Turnover and Graph Density", "Level", ".1%"), use_container_width=True)

with tab4:
    st.subheader("Current Holdings")
    st.dataframe(format_holdings(result.current_holdings), use_container_width=True, hide_index=True)
    st.subheader("Recent Trades")
    st.dataframe(format_trades(result.trades.tail(75)), use_container_width=True, hide_index=True)
    with st.expander("Daily portfolio ledger"):
        ledger_display = ledger.reset_index().copy()
        ledger_display["date"] = pd.to_datetime(ledger_display["date"]).dt.date
        st.dataframe(ledger_display if show_full_ledger else ledger_display.tail(120), use_container_width=True, hide_index=True)

with tab5:
    st.markdown(
        """
        **Daily workflow**

        1. Observe only market and feature history available up to the rebalance date.
        2. Estimate shrinkage covariance from the trailing window.
        3. Build the bank correlation graph and calculate centrality, density, average correlation, and node stress.
        4. Inflate effective covariance when graph contagion is elevated.
        5. Solve a constrained long-only CVaR optimization.
        6. Simulate paper trades, transaction costs, cash, shares, holdings, and daily P&L.

        Transaction costs matter because a portfolio that looks good before costs may be uninvestable after turnover.
        Turnover matters because institutional risk systems must distinguish true signal from noisy rebalancing.
        """
    )
    st.warning("Simulated paper portfolio only. No real orders are sent, and past simulated performance does not imply future returns.")
