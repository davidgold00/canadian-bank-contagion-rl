import numpy as np
import pandas as pd

from src.portfolio.allocation_policy import BANKS
from src.portfolio.paper_trader import PaperPortfolioSimulator


def _sample_data(n=55):
    dates = pd.bdate_range("2024-01-01", periods=n)
    prices = pd.DataFrame(index=dates)
    for i, asset in enumerate(BANKS + ["XFN.TO", "XIU.TO"]):
        drift = 0.001 + i * 0.0001
        prices[asset] = 100 + i * 2 + np.cumprod(np.full(n, 1 + drift))

    features = pd.DataFrame(index=dates)
    features["contagion_risk_score"] = np.r_[np.full(n // 2, 35.0), np.full(n - n // 2, 75.0)]
    for j, bank in enumerate(BANKS):
        features[f"{bank}_vol_21d"] = 0.12 + j * 0.01
        features[f"{bank}_drawdown_63d"] = -0.02 - j * 0.01
        features[f"{bank}_beta_xfn_63d"] = 0.9 + j * 0.02
    return prices, features


def test_simulation_outputs_required_ledgers_and_benchmarks():
    prices, features = _sample_data()
    simulator = PaperPortfolioSimulator(prices, features, use_trained_model=False)
    result = simulator.simulate(
        initial_capital=100_000,
        transaction_cost_bps=5,
        rebalance_threshold=0.0,
        start_date=prices.index[22],
    )

    required_trade_cols = {
        "date",
        "asset",
        "action",
        "shares",
        "price",
        "notional",
        "transaction_cost",
        "reason",
    }
    required_ledger_cols = {
        "portfolio_value",
        "cash",
        "daily_return",
        "daily_pnl",
        "turnover",
        "transaction_costs",
        "contagion_risk_score",
    }

    assert required_trade_cols.issubset(result.trades.columns)
    assert required_ledger_cols.issubset(result.ledger.columns)
    assert {"Equal-weight Big Six", "XFN buy-and-hold", "XIU buy-and-hold", "Cash"}.issubset(result.benchmarks.columns)
    assert not result.current_holdings.empty


def test_long_only_cash_and_final_value_accounting():
    prices, features = _sample_data()
    simulator = PaperPortfolioSimulator(prices, features, use_trained_model=False)
    result = simulator.simulate(
        initial_capital=100_000,
        transaction_cost_bps=10,
        rebalance_threshold=0.0,
        start_date=prices.index[22],
    )

    assert (result.ledger["cash"] >= -1e-6).all()
    assert (result.holdings["shares"] >= -1e-9).all()
    assert result.ledger["transaction_costs"].sum() > 0

    final_value_from_holdings = result.current_holdings["market_value"].sum()
    final_value_from_ledger = result.ledger["portfolio_value"].iloc[-1]
    assert abs(final_value_from_holdings - final_value_from_ledger) < 1e-6


def test_transaction_costs_reduce_portfolio_value():
    prices, features = _sample_data()
    simulator = PaperPortfolioSimulator(prices, features, use_trained_model=False)
    no_cost = simulator.simulate(100_000, transaction_cost_bps=0, rebalance_threshold=0.0, start_date=prices.index[22])
    with_cost = simulator.simulate(100_000, transaction_cost_bps=25, rebalance_threshold=0.0, start_date=prices.index[22])

    assert with_cost.ledger["transaction_costs"].sum() > no_cost.ledger["transaction_costs"].sum()
    assert with_cost.ledger["portfolio_value"].iloc[-1] < no_cost.ledger["portfolio_value"].iloc[-1]


def test_simulation_records_no_future_observation_dates():
    prices, features = _sample_data()
    simulator = PaperPortfolioSimulator(prices, features, use_trained_model=False)
    result = simulator.simulate(start_date=prices.index[22])

    obs_dates = pd.to_datetime(result.ledger["allocation_observation_date"])
    assert (obs_dates <= result.ledger.index).all()
