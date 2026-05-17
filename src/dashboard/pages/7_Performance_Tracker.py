import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.dashboard.components import (  # noqa: E402
    disclaimer_box,
    format_currency,
    format_percent,
    load_price_data,
    load_processed_dataset,
)
from src.dashboard.insight_utils import BANKS, latest_valid_date  # noqa: E402
from src.dashboard.ui_components import analyst_header, apply_dashboard_style, insight_card  # noqa: E402
from src.portfolio.paper_trader import PaperPortfolioSimulator  # noqa: E402
from src.portfolio.performance_metrics import drawdown_series, performance_summary  # noqa: E402


st.set_page_config(page_title="Performance Tracker", layout="wide")
apply_dashboard_style()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@st.cache_data(show_spinner="Running simulated paper portfolio...")
def run_simulation(
    initial_capital: float,
    transaction_cost_bps: float,
    rebalance_threshold: float,
    start_date: str,
    max_single_name_weight: float,
    max_bank_exposure: float,
    defensive_cash_sensitivity: float,
    use_trained_model: bool,
) -> tuple:
    prices = load_price_data()
    features = load_processed_dataset()
    simulator = PaperPortfolioSimulator(
        prices=prices,
        features=features,
        use_trained_model=use_trained_model,
        model_path=repo_root() / "artifacts" / "rl" / "ppo_model.zip",
    )
    result = simulator.simulate(
        initial_capital=initial_capital,
        transaction_cost_bps=transaction_cost_bps,
        rebalance_threshold=rebalance_threshold,
        start_date=start_date,
        max_single_name_weight=max_single_name_weight,
        max_bank_exposure=max_bank_exposure,
        defensive_cash_sensitivity=defensive_cash_sensitivity,
    )
    return result.ledger, result.trades, result.holdings, result.weights, result.benchmarks, result.current_holdings, result.policy_source


def plot_value(ledger: pd.DataFrame, benchmarks: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ledger.index,
            y=ledger["portfolio_value"],
            mode="lines",
            name="Model paper portfolio",
            line=dict(width=3, color="#1d5f8f"),
        )
    )
    for col in benchmarks.columns:
        fig.add_trace(go.Scatter(x=benchmarks.index, y=benchmarks[col], mode="lines", name=col, line=dict(width=1.6)))
    fig.update_layout(
        title="Simulated Portfolio Value vs Benchmarks",
        xaxis_title="Date",
        yaxis_title="Portfolio value, CAD",
        height=500,
        margin=dict(l=20, r=20, t=55, b=25),
    )
    return fig


def plot_pnl(ledger: pd.DataFrame) -> go.Figure:
    pnl = ledger["portfolio_value"] - ledger["portfolio_value"].iloc[0]
    fig = go.Figure(go.Scatter(x=ledger.index, y=pnl, mode="lines", name="Cumulative P&L", line=dict(color="#1f7a5a")))
    fig.update_layout(
        title="Cumulative Simulated P&L",
        xaxis_title="Date",
        yaxis_title="Profit / loss, CAD",
        height=420,
        margin=dict(l=20, r=20, t=55, b=25),
    )
    return fig


def plot_drawdown(ledger: pd.DataFrame, benchmarks: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ledger.index, y=drawdown_series(ledger["portfolio_value"]), mode="lines", name="Model paper portfolio"))
    for col in benchmarks.columns:
        fig.add_trace(go.Scatter(x=benchmarks.index, y=drawdown_series(benchmarks[col]), mode="lines", name=col))
    fig.update_layout(
        title="Drawdown Comparison",
        xaxis_title="Date",
        yaxis_title="Drawdown",
        yaxis_tickformat=".0%",
        height=420,
        margin=dict(l=20, r=20, t=55, b=25),
    )
    return fig


def plot_daily_pnl_hist(ledger: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Histogram(x=ledger["daily_pnl"], nbinsx=60, marker_color="#1d5f8f", name="Daily P&L"))
    fig.update_layout(
        title="Daily Simulated P&L Distribution",
        xaxis_title="Daily P&L, CAD",
        yaxis_title="Number of days",
        height=420,
        margin=dict(l=20, r=20, t=55, b=25),
    )
    return fig


def plot_current_allocation(weights: pd.DataFrame) -> go.Figure:
    latest = weights.iloc[-1].sort_values()
    fig = go.Figure(
        go.Bar(
            x=latest.values,
            y=latest.index,
            orientation="h",
            text=[format_percent(v) for v in latest.values],
            textposition="auto",
            marker_color="#1d5f8f",
        )
    )
    fig.update_layout(
        title="Current Simulated Allocation",
        xaxis_title="Portfolio weight",
        xaxis_tickformat=".0%",
        height=430,
        margin=dict(l=20, r=20, t=55, b=25),
    )
    return fig


def plot_weight_history(weights: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col in weights.columns:
        fig.add_trace(go.Scatter(x=weights.index, y=weights[col], mode="lines", stackgroup="one", name=col))
    fig.update_layout(
        title="Allocation Through Time",
        xaxis_title="Date",
        yaxis_title="Weight",
        yaxis_tickformat=".0%",
        height=520,
        margin=dict(l=20, r=20, t=55, b=25),
    )
    return fig


def plot_cash_vs_risk(ledger: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ledger.index, y=ledger["contagion_risk_score"], name="Contagion risk", yaxis="y1"))
    fig.add_trace(go.Scatter(x=ledger.index, y=ledger["cash_weight"], name="Cash weight", yaxis="y2"))
    fig.update_layout(
        title="Cash Weight vs Contagion Risk",
        xaxis_title="Date",
        yaxis=dict(title="Risk score, 0-100"),
        yaxis2=dict(title="Cash weight", overlaying="y", side="right", tickformat=".0%"),
        height=430,
        margin=dict(l=20, r=20, t=55, b=25),
    )
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
    "Performance Tracker",
    "A simulated paper fund that follows the model's daily allocation recommendations.",
    date_text=latest_valid_date(prices),
    source_text="Paper portfolio, transaction-cost model, and benchmark comparison",
)
disclaimer_box()

if prices.empty or features.empty:
    st.error("Price or feature data is missing. Run scripts/download_data.py and scripts/build_features.py first.")
    st.stop()

ppo_exists = (repo_root() / "artifacts" / "rl" / "ppo_model.zip").exists()

with st.sidebar:
    st.header("Paper Fund Controls")
    initial_capital = st.number_input("Initial capital", min_value=10_000.0, max_value=10_000_000.0, value=100_000.0, step=10_000.0)
    transaction_cost_bps = st.slider("Transaction cost, bps", 0.0, 50.0, 5.0, step=0.5)
    rebalance_threshold = st.slider("Rebalance threshold", 0.0, 0.10, 0.01, step=0.005, format="%.3f")
    max_single_name_weight = st.slider("Max single-name bank weight", 0.05, 0.50, 0.22, step=0.01)
    max_bank_exposure = st.slider("Max total Big Six exposure", 0.10, 1.00, 0.80, step=0.05)
    defensive_cash_sensitivity = st.slider("Defensive cash sensitivity", 0.25, 2.00, 1.00, step=0.05)
    default_start = prices.index[max(63, int(len(prices) * 0.55))].date()
    start_date = st.date_input("Start date", value=default_start, min_value=prices.index.min().date(), max_value=prices.index.max().date())
    use_trained_model = st.checkbox("Use trained PPO model when available", value=ppo_exists)
    show_full_ledger = st.checkbox("Show full daily ledger", value=False)

ledger, trades, holdings, weights, benchmarks, current_holdings, policy_source = run_simulation(
    initial_capital=float(initial_capital),
    transaction_cost_bps=float(transaction_cost_bps),
    rebalance_threshold=float(rebalance_threshold),
    start_date=str(start_date),
    max_single_name_weight=float(max_single_name_weight),
    max_bank_exposure=float(max_bank_exposure),
    defensive_cash_sensitivity=float(defensive_cash_sensitivity),
    use_trained_model=bool(use_trained_model),
)

summary = performance_summary(
    ledger["portfolio_value"],
    ledger["daily_return"],
    ledger["turnover"],
    ledger["transaction_costs"],
)
latest = ledger.iloc[-1]
cum_pnl = summary["ending_value"] - float(initial_capital)
num_trades = int(ledger["number_of_trades"].sum())

top = st.columns(4)
top[0].metric("Starting Capital", format_currency(initial_capital))
top[1].metric("Current Paper Value", format_currency(summary["ending_value"]), delta=format_currency(cum_pnl))
top[2].metric("Cumulative Return", format_percent(summary["cumulative_return"]))
top[3].metric("Policy Source", policy_source)

row2 = st.columns(4)
row2[0].metric("Annualized Return", format_percent(summary["annualized_return"]))
row2[1].metric("Annualized Volatility", format_percent(summary["annualized_volatility"]))
row2[2].metric("Sharpe Ratio", f"{summary['sharpe_ratio']:.2f}")
row2[3].metric("Max Drawdown", format_percent(summary["max_drawdown"]))

row3 = st.columns(4)
row3[0].metric("Transaction Costs", format_currency(summary["total_transaction_costs"], 2))
row3[1].metric("Number of Trades", f"{num_trades:,}")
row3[2].metric("Current Cash Weight", format_percent(latest["cash_weight"]))
row3[3].metric("Current Bank Exposure", format_percent(latest["bank_exposure"]))

if policy_source == "trained PPO model":
    insight_card(
        "Allocation Engine",
        "The paper fund is using the saved PPO model artifact, with long-only and concentration guardrails applied before trades are simulated.",
        status="success",
    )
elif ppo_exists and use_trained_model:
    note = ledger["policy_note"].dropna().iloc[-1] if "policy_note" in ledger else "PPO unavailable."
    insight_card(
        "Allocation Engine",
        f"A PPO artifact exists, but this run used the interpretable fallback policy. Note: {note}",
        status="warning",
    )
else:
    insight_card(
        "Allocation Engine",
        "This run uses the interpretable stress-aware fallback policy. It increases cash when systemic risk rises and tilts bank exposure away from high-stress nodes.",
        status="info",
    )

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Analysis", "Trades & Holdings", "Methodology", "Raw Data"])

with tab1:
    left, right = st.columns([0.62, 0.38])
    with left:
        st.plotly_chart(plot_value(ledger, benchmarks), use_container_width=True)
    with right:
        st.plotly_chart(plot_current_allocation(weights), use_container_width=True)
    st.plotly_chart(plot_pnl(ledger), use_container_width=True)

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_drawdown(ledger, benchmarks), use_container_width=True)
        st.plotly_chart(plot_daily_pnl_hist(ledger), use_container_width=True)
    with c2:
        st.plotly_chart(plot_cash_vs_risk(ledger), use_container_width=True)
        st.plotly_chart(plot_weight_history(weights), use_container_width=True)

    benchmark_final = benchmarks.iloc[-1].sort_values(ascending=False).rename("Ending Value").reset_index()
    benchmark_final.columns = ["Benchmark", "Ending Value"]
    benchmark_final["Ending Value"] = benchmark_final["Ending Value"].map(lambda x: format_currency(x, 2))
    st.dataframe(benchmark_final, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Current Holdings")
    st.markdown(
        "Holdings are paper positions only. They are calculated from simulated daily rebalances, prices, transaction costs, and target weights."
    )
    st.dataframe(format_holdings(current_holdings), use_container_width=True, hide_index=True)

    st.subheader("Recent Simulated Trades")
    recent_trades = format_trades(trades.head(0) if trades.empty else trades.tail(50))
    st.dataframe(recent_trades, use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Daily Process and Leakage Controls")
    st.markdown(
        """
        1. Observe prices, features, contagion score, and bank stress available up to the current date.
        2. Generate target weights using the trained PPO policy when usable; otherwise use the transparent stress-aware fallback.
        3. Compare target weights with current paper holdings.
        4. Generate simulated buy and sell trades when the rebalance threshold is exceeded.
        5. Apply transaction costs, update cash, update shares, and store the trade ledger.
        6. Measure next-day P&L using holdings established on the prior day.

        The simulator does not use future prices or future features to decide today's target weights.
        """
    )
    st.info(
        "Interpretation: this is portfolio intelligence, not broker execution. The output is useful for studying whether the risk model would have reduced drawdowns, raised cash during stress, or concentrated too much in one bank."
    )
    st.markdown(
        """
        **Limitations**

        - No market-impact model, tax model, borrow constraints, or intraday execution.
        - Bank financial statements, OSFI capital data, and true liquidity metrics are not fully modeled.
        - Historical paper performance is not evidence that the policy will work in the future.
        - The PPO artifact, when used, is still a research model and remains subject to validation.
        """
    )

with tab5:
    st.subheader("Daily Portfolio Ledger")
    ledger_display = ledger.reset_index().copy()
    ledger_display["date"] = pd.to_datetime(ledger_display["date"]).dt.date
    if show_full_ledger:
        st.dataframe(ledger_display, use_container_width=True, hide_index=True)
    else:
        st.dataframe(ledger_display.tail(100), use_container_width=True, hide_index=True)

    with st.expander("Full trade ledger"):
        st.dataframe(format_trades(trades), use_container_width=True, hide_index=True)
