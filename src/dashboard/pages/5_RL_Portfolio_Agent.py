from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="RL Portfolio Agent", layout="wide")

BANKS = ["RY.TO", "TD.TO", "BMO.TO", "BNS.TO", "CM.TO", "NA.TO"]
TRADABLES = BANKS + ["XFN.TO", "XIU.TO", "cash"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    root = repo_root()

    prices_path = root / "data" / "processed" / "prices.csv"
    sample_prices_path = root / "data" / "sample" / "market_prices.csv"
    dataset_path = root / "data" / "processed" / "model_dataset.csv"

    path = prices_path if prices_path.exists() else sample_prices_path

    prices = pd.read_csv(path)
    date_col = "date" if "date" in prices.columns else "Date"
    prices[date_col] = pd.to_datetime(prices[date_col])
    prices = prices.rename(columns={date_col: "date"}).set_index("date").sort_index()

    if dataset_path.exists():
        features = pd.read_csv(dataset_path)
        date_col = "date" if "date" in features.columns else features.columns[0]
        features[date_col] = pd.to_datetime(features[date_col])
        features = features.rename(columns={date_col: "date"}).set_index("date").sort_index()
    else:
        features = pd.DataFrame(index=prices.index)
        features["contagion_risk_score"] = 50

    return prices, features


def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1


def annualized_return(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0
    total = equity.iloc[-1] / equity.iloc[0] - 1
    years = len(equity) / 252
    return (1 + total) ** (1 / years) - 1 if years > 0 else 0


def annualized_vol(returns: pd.Series) -> float:
    return returns.std() * np.sqrt(252)


def sharpe(returns: pd.Series) -> float:
    vol = annualized_vol(returns)
    return annualized_return((1 + returns.fillna(0)).cumprod()) / vol if vol > 0 else 0


def max_drawdown(equity: pd.Series) -> float:
    return float(drawdown(equity).min())


def bank_node_stress(features: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=features.index)

    for bank in BANKS:
        components = []

        for col, sign in [
            (f"{bank}_vol_21d", 1),
            (f"{bank}_drawdown_63d", -1),
            (f"{bank}_beta_xfn_63d", 1),
        ]:
            if col in features:
                s = sign * features[col]
                components.append(100 * s.rank(pct=True))

        if components:
            out[bank] = pd.concat(components, axis=1).mean(axis=1)
        else:
            out[bank] = 50

    return out.ffill().fillna(50).clip(0, 100)


def generate_policy_weights(prices: pd.DataFrame, features: pd.DataFrame, mode: str) -> pd.DataFrame:
    idx = prices.index.intersection(features.index)
    prices = prices.reindex(idx)
    features = features.reindex(idx)

    risk = features["contagion_risk_score"] if "contagion_risk_score" in features else pd.Series(50, index=idx)
    stress = bank_node_stress(features).reindex(idx).ffill().fillna(50)

    returns_21d = prices[BANKS].pct_change(21).reindex(idx)
    vol_21d = prices[BANKS].pct_change().rolling(21).std().reindex(idx)

    weights = pd.DataFrame(0.0, index=idx, columns=TRADABLES)

    for date in idx:
        r = float(risk.loc[date])

        if mode == "RL-style defensive":
            cash_weight = np.clip((r - 35) / 65, 0.05, 0.75)
            xfn_weight = np.clip((60 - r) / 100, 0.00, 0.25)
            bank_budget = 1 - cash_weight - xfn_weight

            bank_scores = 100 - stress.loc[date]
            bank_scores = bank_scores.clip(lower=1)
            bank_alloc = bank_scores / bank_scores.sum()

            weights.loc[date, BANKS] = bank_budget * bank_alloc
            weights.loc[date, "XFN.TO"] = xfn_weight
            weights.loc[date, "cash"] = cash_weight

        elif mode == "Momentum":
            mom = returns_21d.loc[date].fillna(0).clip(lower=0)
            if mom.sum() == 0:
                weights.loc[date, BANKS] = 1 / len(BANKS)
            else:
                weights.loc[date, BANKS] = mom / mom.sum()

        elif mode == "Low volatility":
            inv_vol = 1 / vol_21d.loc[date].replace(0, np.nan)
            inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan).fillna(inv_vol.median())
            weights.loc[date, BANKS] = inv_vol / inv_vol.sum()

        else:
            weights.loc[date, BANKS] = 1 / len(BANKS)

    return weights.ffill().fillna(0)


def backtest(prices: pd.DataFrame, weights: pd.DataFrame, transaction_cost_bps: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    asset_returns = prices[[c for c in weights.columns if c != "cash"]].pct_change().reindex(weights.index).fillna(0)
    asset_returns["cash"] = 0.0

    shifted_weights = weights.shift(1).fillna(weights.iloc[0])
    turnover = weights.diff().abs().sum(axis=1).fillna(0)
    transaction_cost = turnover * transaction_cost_bps / 10000

    portfolio_returns = (shifted_weights * asset_returns[weights.columns]).sum(axis=1) - transaction_cost
    equity = (1 + portfolio_returns).cumprod()

    return equity, portfolio_returns, turnover


def metrics_table(results: dict[str, tuple[pd.Series, pd.Series, pd.Series]]) -> pd.DataFrame:
    rows = []

    for name, (equity, returns, turnover) in results.items():
        rows.append(
            {
                "Strategy": name,
                "Cumulative Return": equity.iloc[-1] - 1,
                "Annualized Return": annualized_return(equity),
                "Annualized Volatility": annualized_vol(returns),
                "Sharpe": sharpe(returns),
                "Max Drawdown": max_drawdown(equity),
                "Average Daily Turnover": turnover.mean(),
            }
        )

    return pd.DataFrame(rows).sort_values("Sharpe", ascending=False)


def plot_equity(results: dict[str, tuple[pd.Series, pd.Series, pd.Series]]) -> go.Figure:
    fig = go.Figure()

    for name, (equity, _, _) in results.items():
        fig.add_trace(go.Scatter(x=equity.index, y=equity, mode="lines", name=name))

    fig.update_layout(
        title="Strategy Equity Curves",
        yaxis_title="Growth of $1",
        height=500,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig


def plot_drawdowns(results: dict[str, tuple[pd.Series, pd.Series, pd.Series]]) -> go.Figure:
    fig = go.Figure()

    for name, (equity, _, _) in results.items():
        fig.add_trace(go.Scatter(x=equity.index, y=drawdown(equity), mode="lines", name=name))

    fig.update_layout(
        title="Drawdowns",
        yaxis_title="Drawdown",
        height=430,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig


st.title("RL Portfolio Agent")

st.info(
    """
    This page explains the portfolio decision layer. The agent observes stress indicators,
    bank node stress, correlations, drawdowns, and market features, then allocates across banks,
    ETFs, and cash. Its objective is not just return; it is return adjusted for drawdown,
    turnover, concentration, and contagion exposure.
    """
)

prices, features = load_data()

common_index = prices.index.intersection(features.index)
prices = prices.reindex(common_index)
features = features.reindex(common_index)

st.sidebar.header("RL Backtest Controls")
transaction_cost_bps = st.sidebar.slider("Transaction cost, bps", 0.0, 25.0, 5.0, step=1.0)
start_index = st.sidebar.slider("Backtest start percentile", 0, 80, 25, step=5)
start_date = common_index[int(len(common_index) * start_index / 100)]

prices_bt = prices.loc[start_date:]
features_bt = features.loc[start_date:]

strategies = {
    "RL-style defensive": generate_policy_weights(prices_bt, features_bt, "RL-style defensive"),
    "Equal-weight Big Six": generate_policy_weights(prices_bt, features_bt, "Equal"),
    "Momentum": generate_policy_weights(prices_bt, features_bt, "Momentum"),
    "Low volatility": generate_policy_weights(prices_bt, features_bt, "Low volatility"),
}

results = {
    name: backtest(prices_bt, weights, transaction_cost_bps)
    for name, weights in strategies.items()
}

metrics = metrics_table(results)

c1, c2, c3, c4 = st.columns(4)
best = metrics.iloc[0]
rl_metrics = metrics.loc[metrics["Strategy"] == "RL-style defensive"].iloc[0]

c1.metric("Best Sharpe Strategy", best["Strategy"])
c2.metric("RL Sharpe", f"{rl_metrics['Sharpe']:.2f}")
c3.metric("RL Max Drawdown", f"{rl_metrics['Max Drawdown']:.1%}")
c4.metric("RL Cumulative Return", f"{rl_metrics['Cumulative Return']:.1%}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Performance",
        "Current Allocation",
        "Allocation Through Time",
        "Stress Behavior",
        "How the Agent Thinks",
    ]
)

with tab1:
    st.subheader("RL Agent vs Benchmarks")
    st.markdown(
        """
        This section compares the defensive RL-style allocation to simple benchmark strategies.
        A strong risk-aware agent should not only chase returns; it should reduce drawdowns
        and reduce exposure during high-contagion periods.
        """
    )

    st.plotly_chart(plot_equity(results), use_container_width=True)
    st.plotly_chart(plot_drawdowns(results), use_container_width=True)

    display_metrics = metrics.copy()
    for col in ["Cumulative Return", "Annualized Return", "Annualized Volatility", "Max Drawdown", "Average Daily Turnover"]:
        display_metrics[col] = display_metrics[col].map(lambda x: f"{x:.2%}")
    display_metrics["Sharpe"] = display_metrics["Sharpe"].map(lambda x: f"{x:.2f}")

    st.dataframe(display_metrics, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Current Recommended Allocation")

    latest_weights = strategies["RL-style defensive"].iloc[-1].sort_values(ascending=False)
    latest_risk = float(features_bt["contagion_risk_score"].iloc[-1]) if "contagion_risk_score" in features_bt else 50

    fig = go.Figure(
        go.Bar(
            x=latest_weights.index,
            y=latest_weights.values,
            text=[f"{v:.1%}" for v in latest_weights.values],
            textposition="auto",
        )
    )
    fig.update_layout(
        title=f"Latest RL-Style Allocation | Contagion Risk={latest_risk:.1f}/100",
        yaxis_title="Portfolio weight",
        height=430,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    if latest_risk >= 70:
        explanation = "The agent is defensive because contagion risk is high. It should prefer cash and lower-stress banks."
    elif latest_risk >= 45:
        explanation = "The agent is balanced because risk is moderate. It keeps some bank exposure but avoids highly stressed names."
    else:
        explanation = "The agent is risk-on because contagion risk is contained. It can hold more bank exposure."

    st.success(explanation)

    st.dataframe(
        latest_weights.rename("Weight").reset_index().rename(columns={"index": "Asset"}),
        use_container_width=True,
        hide_index=True,
    )

with tab3:
    st.subheader("How Allocations Change Over Time")

    weights = strategies["RL-style defensive"]

    fig = go.Figure()
    for col in weights.columns:
        fig.add_trace(go.Scatter(x=weights.index, y=weights[col], mode="lines", stackgroup="one", name=col))

    fig.update_layout(
        title="RL-Style Portfolio Weights",
        yaxis_title="Weight",
        height=520,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        A credible risk-aware agent should visibly change behavior:
        it should hold more bank exposure when stress is low and increase cash or defensive exposure
        when contagion risk rises.
        """
    )

with tab4:
    st.subheader("Behavior During High-Stress Periods")

    risk = features_bt["contagion_risk_score"] if "contagion_risk_score" in features_bt else pd.Series(50, index=features_bt.index)
    high_stress = risk >= risk.quantile(0.90)

    comparison = []
    for name, (equity, returns, turnover) in results.items():
        comparison.append(
            {
                "Strategy": name,
                "Avg Return on Top 10% Stress Days": returns.loc[high_stress].mean(),
                "Vol on Top 10% Stress Days": returns.loc[high_stress].std(),
                "Worst Stress-Day Return": returns.loc[high_stress].min(),
                "Avg Turnover on Stress Days": turnover.loc[high_stress].mean(),
            }
        )

    stress_df = pd.DataFrame(comparison)
    display = stress_df.copy()
    for col in display.columns:
        if col != "Strategy":
            display[col] = display[col].map(lambda x: f"{x:.2%}")

    st.dataframe(display, use_container_width=True, hide_index=True)

    rl_cash = strategies["RL-style defensive"]["cash"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=risk.index, y=risk, name="Contagion risk", yaxis="y1"))
    fig.add_trace(go.Scatter(x=rl_cash.index, y=100 * rl_cash, name="RL cash weight", yaxis="y2"))
    fig.update_layout(
        title="Risk Score vs RL Cash Allocation",
        yaxis=dict(title="Risk score"),
        yaxis2=dict(title="Cash weight %", overlaying="y", side="right"),
        height=460,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.subheader("What the RL Agent Is Supposed to Learn")
    st.markdown(
        """
        The value of the RL layer is that it connects risk measurement to action.

        ### State

        The agent observes:

        - recent returns,
        - current portfolio weights,
        - bank node stress,
        - contagion risk score,
        - volatility,
        - drawdowns,
        - correlation pressure,
        - macro stress proxies.

        ### Action

        The agent chooses target portfolio weights across:

        - Big Six bank equities,
        - XFN,
        - XIU,
        - cash.

        ### Reward

        The reward balances:

        - portfolio return,
        - volatility penalty,
        - drawdown penalty,
        - turnover penalty,
        - concentration penalty,
        - contagion exposure penalty.

        ### What good behavior looks like

        - holds more bank exposure during calm regimes,
        - reduces exposure during rising systemic risk,
        - avoids concentrating in the most stressed bank,
        - re-risks after stress declines,
        - does not overtrade.
        """
    )

    st.warning(
        """
        The current dashboard includes an interpretable RL-style policy visualization.
        After training a saved PPO model, this page can be extended to load the actual model's predicted actions.
        """
    )