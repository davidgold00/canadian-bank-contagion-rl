import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.dashboard.benchmarking import (  # noqa: E402
    BROAD_MARKET_BENCHMARKS,
    CANADIAN_BANK_BENCHMARKS,
    build_benchmark_values,
)
from src.dashboard.components import format_currency, format_percent, load_price_data, load_processed_dataset  # noqa: E402
from src.dashboard.insight_utils import latest_valid_date  # noqa: E402
from src.dashboard.ui_components import PALETTE, PLOTLY_TEMPLATE, analyst_header, apply_dashboard_style, business_value_panel, decision_callout, decision_memo, insight_card, mandate_fit_table, page_intro, plain_english_expander  # noqa: E402
from src.portfolio.paper_trader import CVaRPaperPortfolioSimulator, PaperPortfolioSimulator  # noqa: E402
from src.portfolio.performance_metrics import drawdown_series, performance_summary, rolling_cvar  # noqa: E402


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


def _metric_row(
    name: str,
    values: pd.Series,
    returns: pd.Series | None = None,
    turnover: pd.Series | None = None,
    costs: pd.Series | None = None,
    strategy_type: str = "Model",
    average_bank_exposure: float | None = None,
    average_cash: float | None = None,
) -> dict[str, float | str | None]:
    summary = performance_summary(values, returns, turnover, costs)
    return {
        "Strategy": name,
        "Type": strategy_type,
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
        "Average Bank Exposure": average_bank_exposure,
        "Average Cash": average_cash,
    }


def comparison_metrics(cvar, rl, benchmarks: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = [
        _metric_row(
            "CVaR optimizer",
            cvar.ledger["portfolio_value"],
            cvar.ledger["daily_return"],
            cvar.ledger["turnover"],
            cvar.ledger["transaction_costs"],
            average_bank_exposure=float(cvar.ledger["bank_exposure"].mean()),
            average_cash=float(cvar.ledger["cash_weight"].mean()),
        ),
        _metric_row(
            "RL research baseline",
            rl.ledger["portfolio_value"],
            rl.ledger["daily_return"],
            rl.ledger["turnover"],
            rl.ledger["transaction_costs"],
            average_bank_exposure=float(rl.ledger["bank_exposure"].mean()),
            average_cash=float(rl.ledger["cash_weight"].mean()),
        ),
    ]

    if benchmarks is not None:
        for label in benchmarks.columns:
            values = benchmarks[label].dropna()
            if len(values) < 2:
                continue
            rows.append(
                _metric_row(
                    label,
                    values,
                    values.pct_change().fillna(0.0),
                    strategy_type="Buy-and-hold benchmark",
                )
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
    out["Sharpe"] = out["Sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    out["Sortino"] = out["Sortino"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
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
    st.divider()
    st.markdown("### Benchmark Toggles")
    st.caption("Check a box to add that buy-and-hold comparison to the table and time-series charts.")
    selected_benchmark_labels = []
    for spec in BROAD_MARKET_BENCHMARKS:
        if st.checkbox(spec.label, value=spec.default_shown, help=spec.description):
            selected_benchmark_labels.append(spec.label)
    st.markdown("#### Big Six Banks")
    for spec in CANADIAN_BANK_BENCHMARKS:
        if st.checkbox(spec.label, value=spec.default_shown, help=spec.description):
            selected_benchmark_labels.append(spec.label)

cvar, rl = run_comparison(
    float(initial_capital),
    float(transaction_cost_bps),
    float(rebalance_threshold),
    str(start_date),
    float(confidence_level),
    int(lookback_window),
    int(rebalance_frequency),
)

selected_benchmarks, missing_benchmarks = build_benchmark_values(
    prices,
    selected_benchmark_labels,
    float(initial_capital),
    cvar.ledger.index,
)

metrics = comparison_metrics(cvar, rl, selected_benchmarks)
winner_sharpe = metrics.sort_values("Sharpe", ascending=False).iloc[0]["Strategy"]
winner_cvar = metrics.sort_values("CVaR", ascending=True).iloc[0]["Strategy"]
winner_ending = metrics.sort_values("Ending Value", ascending=False).iloc[0]["Strategy"]
winner_drawdown = metrics.sort_values("Max Drawdown", ascending=False).iloc[0]["Strategy"]
model_metrics = metrics[metrics["Type"] == "Model"]
benchmark_metrics = metrics[metrics["Type"] == "Buy-and-hold benchmark"]
best_model_return = model_metrics.sort_values("Ending Value", ascending=False).iloc[0]
best_benchmark_return = benchmark_metrics.sort_values("Ending Value", ascending=False).iloc[0] if not benchmark_metrics.empty else None

c1, c2, c3, c4 = st.columns(4)
c1.metric("Best Ending Value", winner_ending)
c2.metric("Best Sharpe", winner_sharpe)
c3.metric("Lowest CVaR", winner_cvar)
c4.metric("Smallest Drawdown", winner_drawdown)

if missing_benchmarks:
    st.warning(
        "Missing benchmark price history for: "
        + ", ".join(missing_benchmarks)
        + ". Refresh data with `python scripts/download_data.py` and `python scripts/build_features.py` to populate unavailable tickers."
    )

plain_english_expander(
    "Optional Guide: What This Comparison Proves",
    [
        ("Buy-and-hold benchmark", "A passive investment that starts with the same capital and simply holds one index, ETF, or bank stock through time."),
        ("Ending value", "The most direct wealth comparison: which path turned the same starting capital into the most dollars."),
        ("Sharpe", "Return per unit of volatility. A higher number means the strategy was paid more for each unit of risk."),
        ("CVaR", "Average loss on the worst days. Lower is better because the tail losses were smaller."),
        ("Max drawdown", "The deepest peak-to-trough loss. Closer to zero is better because the investor had to tolerate less pain."),
    ],
)

business_value_panel(
    title="How to Judge This Page",
    intro=(
        "This comparison is intentionally uncomfortable: it should show when passive Nasdaq, ZEB, or single-bank exposure wins. "
        "That benchmark pressure makes the project stronger because it prevents overclaiming. The business value is deciding "
        "whether active risk control, explanation, and drawdown management are worth the return trade-off."
    ),
    points=[
        (
            "Return Truth",
            "If a passive benchmark wins on ending value, say so directly and treat it as the growth benchmark.",
            "Honesty",
        ),
        (
            "Risk Trade-Off",
            "Then ask whether the model reduced drawdown, CVaR, concentration, or stress exposure enough to justify lower return.",
            "Mandate",
        ),
        (
            "Canadian Bank Sleeve",
            "Use ZEB and the Big Six as the most relevant challengers for a Canadian bank mandate.",
            "Fit",
        ),
        (
            "Governance Value",
            "Use the explanations, triggers, and ledgers to show why exposure changed and when it should be reviewed.",
            "Process",
        ),
    ],
)

cvar_sharpe = metrics.loc[metrics["Strategy"] == "CVaR optimizer", "Sharpe"].iloc[0]
rl_sharpe = metrics.loc[metrics["Strategy"] == "RL research baseline", "Sharpe"].iloc[0]
active_benchmark_text = ", ".join(selected_benchmarks.columns) if not selected_benchmarks.empty else "no external benchmarks selected"
decision_callout(
    plain_english=(
        f"The <b>{winner_sharpe}</b> achieved a higher Sharpe ratio, meaning better return per unit of risk taken. "
        f"The <b>{winner_cvar}</b> had the lower CVaR, meaning smaller average losses in the worst scenarios. "
        "The benchmark set currently shown is: <b>{}</b>. "
        "CVaR Sharpe: {:.2f} | RL Sharpe: {:.2f}".format(active_benchmark_text, cvar_sharpe, rl_sharpe)
    ),
    action=(
        "Use this page as the fairness check: the models only matter if they beat simple buy-and-hold alternatives "
        "after accounting for risk, drawdowns, and transaction costs. If a passive benchmark wins on both return "
        "and drawdown, the model needs stronger evidence before real capital should trust it."
    ),
    tone="teal",
)

if best_benchmark_return is not None and best_benchmark_return["Ending Value"] > best_model_return["Ending Value"]:
    return_gap = best_benchmark_return["Ending Value"] / best_model_return["Ending Value"] - 1
    decision_callout(
        plain_english=(
            f"The strongest passive benchmark, <b>{best_benchmark_return['Strategy']}</b>, finished "
            f"<b>{return_gap:.1%}</b> ahead of the strongest model result over this selected window. "
            "That is a real result, not something to hide. It means the model did not win the simple growth contest."
        ),
        action=(
            "Use the rest of the page to decide whether the model still earns a role as a risk-control overlay: "
            "lower tail loss, smaller drawdown, clearer exposure rules, or better stress behavior. "
            "If it does not improve those mandate-specific outcomes, passive exposure is the better answer for that mandate."
        ),
        tone="warning",
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
            "Observation": f"Best ending value: {winner_ending}",
            "Decision Implication": "This is the plain wealth comparison against passive alternatives with the same starting capital.",
            "Monitoring Trigger": "If a passive benchmark keeps winning, inspect whether the active model is over-constrained or over-trading.",
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
    with st.expander("How to interpret this table for a business user", expanded=False):
        mandate_fit_table()
    equity_series = {
        "CVaR optimizer": cvar.ledger["portfolio_value"],
        "RL research baseline": rl.ledger["portfolio_value"],
    }
    equity_series.update({label: selected_benchmarks[label] for label in selected_benchmarks.columns})
    st.plotly_chart(
        line_chart(
            equity_series,
            "Equity Curve Comparison",
            "Portfolio value",
        ),
        use_container_width=True,
    )
    c_left, c_right = st.columns(2)
    with c_left:
        drawdown_series_map = {name: drawdown_series(values) for name, values in equity_series.items()}
        st.plotly_chart(
            line_chart(
                drawdown_series_map,
                "Drawdown Comparison",
                "Drawdown",
                ".0%",
            ),
            use_container_width=True,
        )
    with c_right:
        rolling_cvar_map = {
            name: rolling_cvar(pd.Series(values).pct_change().fillna(0.0), 63)
            for name, values in equity_series.items()
        }
        st.plotly_chart(
            line_chart(
                rolling_cvar_map,
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
