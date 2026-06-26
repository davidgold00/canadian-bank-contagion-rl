import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.dashboard.components import format_currency, format_percent, load_price_data, load_processed_dataset  # noqa: E402
from src.dashboard.insight_utils import latest_valid_date  # noqa: E402
from src.dashboard.ui_components import PALETTE, PLOTLY_TEMPLATE, analyst_header, apply_dashboard_style, decision_callout, decision_memo, insight_card, page_intro  # noqa: E402
from src.portfolio.paper_trader import CVaRPaperPortfolioSimulator, PaperPortfolioSimulator  # noqa: E402
from src.portfolio.performance_metrics import drawdown_series, performance_summary, rolling_cvar, rolling_sharpe  # noqa: E402


st.set_page_config(page_title="RL vs CVaR Comparison", layout="wide")
apply_dashboard_style()


@st.cache_data(show_spinner="Running comparative research backtest...")
def run_comparison(
    initial_capital,
    transaction_cost_bps,
    rebalance_threshold,
    start_date,
    confidence_level,
    lookback_window,
    rebalance_frequency,
):
    prices = load_price_data()
    features = load_processed_dataset()
    cvar = CVaRPaperPortfolioSimulator(prices, features, use_trained_model=False).simulate_cvar(
        initial_capital=initial_capital,
        transaction_cost_bps=transaction_cost_bps,
        rebalance_threshold=rebalance_threshold,
        start_date=start_date,
        confidence_level=confidence_level,
        lookback_window=lookback_window,
        rebalance_frequency=rebalance_frequency,
    )
    rl = PaperPortfolioSimulator(prices, features, use_trained_model=True).simulate(
        initial_capital=initial_capital,
        transaction_cost_bps=transaction_cost_bps,
        rebalance_threshold=rebalance_threshold,
        start_date=start_date,
        max_single_name_weight=0.22,
        max_bank_exposure=0.80,
    )
    return cvar, rl


def comparison_metrics(cvar, rl) -> pd.DataFrame:
    rows = []
    for name, result in [("CVaR optimizer", cvar), ("RL research baseline", rl)]:
        summary = performance_summary(
            result.ledger["portfolio_value"],
            result.ledger["daily_return"],
            result.ledger["turnover"],
            result.ledger["transaction_costs"],
        )
        rows.append(
            {
                "Strategy": name,
                "Ending Value": summary["ending_value"],
                "Cumulative Return": summary["cumulative_return"],
                "Annualized Return": summary["annualized_return"],
                "Annualized Volatility": summary["annualized_volatility"],
                "Sharpe": summary["sharpe_ratio"],
                "Sortino": summary["sortino_ratio"],
                "CVaR": summary["conditional_value_at_risk"],
                "Max Drawdown": summary["max_drawdown"],
                "Avg Turnover": summary["average_daily_turnover"],
                "Transaction Costs": summary["total_transaction_costs"],
                "Average Bank Exposure": result.ledger["bank_exposure"].mean(),
                "Average Cash": result.ledger["cash_weight"].mean(),
            }
        )
    return pd.DataFrame(rows)


def line_chart(series: dict[str, pd.Series], title: str, yaxis: str, tickformat=None) -> go.Figure:
    fig = go.Figure()
    for name, values in series.items():
        fig.add_trace(go.Scatter(x=values.index, y=values, mode="lines", name=name))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title=yaxis, height=430)
    if tickformat:
        fig.update_yaxes(tickformat=tickformat)
    return fig


def allocation_heatmap(weights: pd.DataFrame, title: str) -> go.Figure:
    sample = weights.resample("ME").last().tail(36)
    fig = go.Figure(
        go.Heatmap(
            z=sample.T.values,
            x=sample.index,
            y=sample.columns,
            zmin=0,
            zmax=max(0.4, float(sample.max().max())),
            colorscale="Blues",
            colorbar=dict(title="Weight"),
        )
    )
    fig.update_layout(title=title, height=420, margin=dict(l=20, r=20, t=55, b=20))
    return fig


def format_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Ending Value"] = out["Ending Value"].map(lambda x: format_currency(x, 0))
    out["Transaction Costs"] = out["Transaction Costs"].map(lambda x: format_currency(x, 2))
    for col in [
        "Cumulative Return",
        "Annualized Return",
        "Annualized Volatility",
        "CVaR",
        "Max Drawdown",
        "Avg Turnover",
        "Average Bank Exposure",
        "Average Cash",
    ]:
        out[col] = out[col].map(lambda x: format_percent(x, 2))
    out["Sharpe"] = out["Sharpe"].map(lambda x: f"{x:.2f}")
    out["Sortino"] = out["Sortino"].map(lambda x: f"{x:.2f}")
    return out


prices = load_price_data()
features = load_processed_dataset()

analyst_header(
    "RL vs CVaR Comparative Analytics",
    "A quant research comparison between experimental RL allocation and governed CVaR optimization.",
    date_text=latest_valid_date(features),
    source_text="Same data, same capital, same transaction-cost assumptions",
)

page_intro(
    why=(
        "Two allocation approaches compete head-to-head on the same data: "
        "the <b>RL policy</b> learns from market states and adjusts behavior dynamically; "
        "the <b>CVaR optimizer</b> solves a constrained mathematical problem minimizing tail losses. "
        "Knowing which performs better — and when — tells you which to trust in practice."
    ),
    how=(
        "The key question is not just 'which earned more?' but 'which managed risk better during stress periods?' "
        "Look for differences in drawdown depth during the worst market periods. "
        "Use the regime breakdown tab to see if one approach is clearly superior in stress vs. calm regimes."
    ),
)

with st.sidebar:
    st.header("Comparison Controls")
    initial_capital = st.number_input("Initial capital", min_value=10_000.0, max_value=10_000_000.0, value=100_000.0, step=10_000.0)
    transaction_cost_bps = st.slider("Transaction cost, bps", 0.0, 50.0, 5.0, step=0.5)
    rebalance_threshold = st.slider("Rebalance threshold", 0.0, 0.10, 0.01, step=0.005, format="%.3f")
    confidence_level = st.selectbox("CVaR confidence level", [0.95, 0.99], format_func=lambda x: f"{x:.0%}")
    lookback_window = st.selectbox("CVaR lookback window", [63, 126, 252], index=1)
    rebalance_frequency = st.selectbox("CVaR rebalance frequency", [1, 5, 10, 21], index=2)
    default_start = prices.index[max(252, int(len(prices) * 0.70))].date()
    start_date = st.date_input("Start date", value=default_start, min_value=prices.index.min().date(), max_value=prices.index.max().date())

cvar, rl = run_comparison(
    float(initial_capital),
    float(transaction_cost_bps),
    float(rebalance_threshold),
    str(start_date),
    float(confidence_level),
    int(lookback_window),
    int(rebalance_frequency),
)

metrics = comparison_metrics(cvar, rl)
winner_sharpe = metrics.sort_values("Sharpe", ascending=False).iloc[0]["Strategy"]
winner_cvar = metrics.sort_values("CVaR", ascending=True).iloc[0]["Strategy"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Best Sharpe", winner_sharpe)
c2.metric("Lowest CVaR", winner_cvar)
c3.metric("CVaR Ending Value", format_currency(metrics.loc[metrics["Strategy"] == "CVaR optimizer", "Ending Value"].iloc[0]))
c4.metric("RL Ending Value", format_currency(metrics.loc[metrics["Strategy"] == "RL research baseline", "Ending Value"].iloc[0]))

cvar_sharpe = metrics.loc[metrics["Strategy"] == "CVaR optimizer", "Sharpe"].iloc[0]
rl_sharpe = metrics.loc[metrics["Strategy"] == "RL research baseline", "Sharpe"].iloc[0]
decision_callout(
    plain_english=(
        f"The <b>{winner_sharpe}</b> achieved a higher Sharpe ratio, meaning better return per unit of risk taken. "
        f"The <b>{winner_cvar}</b> had the lower CVaR, meaning smaller average losses in the worst scenarios. "
        "CVaR Sharpe: {:.2f} | RL Sharpe: {:.2f}".format(cvar_sharpe, rl_sharpe)
    ),
    action=(
        "For a governed portfolio: use CVaR as the primary allocator with constraints. "
        "Use RL signals as a research overlay or sanity check. "
        "If both agree on a bank (both overweight or both underweight), the conviction is higher."
    ),
    tone="teal",
)

decision_memo(
    "Allocator Choice Memo",
    [
        {
            "Observation": f"Best Sharpe: {winner_sharpe}",
            "Decision Implication": "Sharpe winner is the better candidate when return efficiency is the primary objective.",
            "Monitoring Trigger": "Confirm the result is stable across start dates and transaction-cost assumptions.",
        },
        {
            "Observation": f"Lowest CVaR: {winner_cvar}",
            "Decision Implication": "CVaR winner is the better candidate when downside containment is the primary objective.",
            "Monitoring Trigger": "Use this allocator for stress regimes unless it materially sacrifices drawdown or liquidity.",
        },
        {
            "Observation": f"CVaR Sharpe {cvar_sharpe:.2f}; RL Sharpe {rl_sharpe:.2f}",
            "Decision Implication": "If the two allocators agree on exposure, conviction rises; if they diverge, use CVaR as the governed baseline.",
            "Monitoring Trigger": "Investigate divergence before changing real-world policy limits.",
        },
    ],
    tone="teal",
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Performance", "Risk Regimes", "Allocation Stability", "Research Interpretation", "Raw Diagnostics"])

with tab1:
    st.dataframe(format_metrics(metrics), use_container_width=True, hide_index=True)
    st.plotly_chart(
        line_chart(
            {
                "CVaR optimizer": cvar.ledger["portfolio_value"],
                "RL research baseline": rl.ledger["portfolio_value"],
            },
            "Equity Curve Comparison",
            "Portfolio value",
        ),
        use_container_width=True,
    )
    c_left, c_right = st.columns(2)
    with c_left:
        st.plotly_chart(
            line_chart(
                {
                    "CVaR optimizer": drawdown_series(cvar.ledger["portfolio_value"]),
                    "RL research baseline": drawdown_series(rl.ledger["portfolio_value"]),
                },
                "Drawdown Comparison",
                "Drawdown",
                ".0%",
            ),
            use_container_width=True,
        )
    with c_right:
        st.plotly_chart(
            line_chart(
                {
                    "CVaR optimizer": rolling_cvar(cvar.ledger["daily_return"], 63),
                    "RL research baseline": rolling_cvar(rl.ledger["daily_return"], 63),
                },
                "Rolling 63D CVaR",
                "CVaR",
                ".1%",
            ),
            use_container_width=True,
        )

with tab2:
    stress_threshold = features["contagion_risk_score"].reindex(cvar.ledger.index).quantile(0.90)
    stress_days = features["contagion_risk_score"].reindex(cvar.ledger.index) >= stress_threshold
    rows = []
    for name, ledger in [("CVaR optimizer", cvar.ledger), ("RL research baseline", rl.ledger)]:
        rows.append(
            {
                "Strategy": name,
                "Avg Stress-Day Return": ledger.loc[stress_days, "daily_return"].mean(),
                "Worst Stress-Day Return": ledger.loc[stress_days, "daily_return"].min(),
                "Stress-Day Volatility": ledger.loc[stress_days, "daily_return"].std(),
                "Avg Stress-Day Cash": ledger.loc[stress_days, "cash_weight"].mean(),
                "Avg Stress-Day Bank Exposure": ledger.loc[stress_days, "bank_exposure"].mean(),
            }
        )
    stress_table = pd.DataFrame(rows)
    display = stress_table.copy()
    for col in display.columns:
        if col != "Strategy":
            display[col] = display[col].map(lambda x: format_percent(x, 2))
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.plotly_chart(
        line_chart(
            {
                "CVaR bank exposure": cvar.ledger["bank_exposure"],
                "RL bank exposure": rl.ledger["bank_exposure"],
                "CVaR cash": cvar.ledger["cash_weight"],
                "RL cash": rl.ledger["cash_weight"],
            },
            "Rolling Bank Exposure and Cash Posture",
            "Weight",
            ".0%",
        ),
        use_container_width=True,
    )

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(allocation_heatmap(cvar.weights, "CVaR Allocation Heatmap"), use_container_width=True)
        st.plotly_chart(line_chart({"CVaR turnover": cvar.ledger["turnover"]}, "CVaR Turnover", "Turnover", ".0%"), use_container_width=True)
    with c2:
        st.plotly_chart(allocation_heatmap(rl.weights, "RL Allocation Heatmap"), use_container_width=True)
        st.plotly_chart(line_chart({"RL turnover": rl.ledger["turnover"]}, "RL Turnover", "Turnover", ".0%"), use_container_width=True)

with tab4:
    st.subheader("Institutional Interpretation")
    st.markdown(
        """
        **Potential strengths of RL**

        - Adaptive behavior and nonlinear policy learning.
        - Can react to state combinations that linear rules may miss.
        - Useful as a research challenger and stress-behavior benchmark.

        **Potential weaknesses of RL**

        - Harder to validate and explain.
        - Sensitive to reward design and training regime.
        - Can overfit simulated environments.

        **Potential strengths of CVaR optimization**

        - Directly controls tail risk.
        - Easier to govern, audit, and explain to a risk committee.
        - Constraints map naturally to portfolio policy limits.
        - Graph-adjusted covariance makes systemic concentration explicit.

        **Potential weaknesses of CVaR**

        - Depends on covariance and tail estimates.
        - May react slowly if the lookback window is stale.
        - Can become conservative when stress signals remain elevated.
        """
    )

with tab5:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("CVaR Ledger")
        st.dataframe(cvar.ledger.tail(100), use_container_width=True)
    with c2:
        st.subheader("RL Ledger")
        st.dataframe(rl.ledger.tail(100), use_container_width=True)
