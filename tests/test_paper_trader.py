import numpy as np
import pandas as pd

from src.portfolio.allocation_policy import BANKS
from src.portfolio.paper_trader import CVaRPaperPortfolioSimulator, PaperPortfolioSimulator


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


def test_cvar_paper_fund_generates_trade_ledger_without_lookahead():
    prices, features = _sample_data(n=95)
    simulator = CVaRPaperPortfolioSimulator(prices, features, use_trained_model=False)
    result = simulator.simulate_cvar(
        initial_capital=100_000,
        transaction_cost_bps=5,
        rebalance_threshold=0.0,
        start_date=prices.index[70],
        lookback_window=45,
        rebalance_frequency=3,
    )

    assert result.policy_source == "graph-adjusted CVaR optimizer"
    assert not result.trades.empty
    assert {"graph_density", "realized_cvar_63d", "stress_exposure"}.issubset(result.ledger.columns)
    assert (pd.to_datetime(result.ledger["allocation_observation_date"]) <= result.ledger.index).all()
    assert abs(result.current_holdings["market_value"].sum() - result.ledger["portfolio_value"].iloc[-1]) < 1e-6


def test_cvar_trades_only_on_rebalance_dates_and_records_true_observation_date():
    prices, features = _sample_data(n=95)
    simulator = CVaRPaperPortfolioSimulator(prices, features, use_trained_model=False)
    result = simulator.simulate_cvar(
        start_date=prices.index[70],
        lookback_window=45,
        rebalance_frequency=3,
        rebalance_threshold=0.0,
    )

    scheduled_dates = set(result.ledger.index[::3])
    assert set(pd.to_datetime(result.trades["date"]).unique()).issubset(scheduled_dates)

    expected_observation_dates = pd.Series(
        [result.ledger.index[(step // 3) * 3] for step in range(len(result.ledger))],
        index=result.ledger.index,
    )
    pd.testing.assert_series_equal(
        pd.to_datetime(result.ledger["allocation_observation_date"]),
        expected_observation_dates,
        check_names=False,
    )


def test_future_only_feature_values_do_not_change_prior_cvar_allocation():
    prices, features = _sample_data(n=95)
    cutoff = prices.index[70]
    future_start = prices.index[71]
    low_future = features.copy()
    high_future = features.copy()
    low_future.loc[:cutoff, :] = np.nan
    high_future.loc[:cutoff, :] = np.nan
    low_future.loc[future_start:, :] = 10.0
    high_future.loc[future_start:, :] = 90.0

    low = CVaRPaperPortfolioSimulator(prices, low_future, use_trained_model=False).simulate_cvar(
        start_date=cutoff,
        end_date=cutoff,
        lookback_window=45,
        rebalance_threshold=0.0,
    )
    high = CVaRPaperPortfolioSimulator(prices, high_future, use_trained_model=False).simulate_cvar(
        start_date=cutoff,
        end_date=cutoff,
        lookback_window=45,
        rebalance_threshold=0.0,
    )

    pd.testing.assert_frame_equal(low.weights, high.weights)
    pd.testing.assert_series_equal(low.ledger.iloc[0], high.ledger.iloc[0])


def test_simulation_waits_for_complete_prices_instead_of_backfilling():
    prices, features = _sample_data(n=95)
    first_valid = prices.index[12]
    prices.loc[: prices.index[11], "XIU.TO"] = np.nan
    simulator = PaperPortfolioSimulator(prices, features, use_trained_model=False)

    assert pd.isna(simulator.prices.loc[prices.index[0], "XIU.TO"])
    assert simulator._simulation_dates(None, None)[0] == first_valid


def test_reported_financial_exposure_includes_xfn_and_respects_policy_cap():
    prices, features = _sample_data(n=55)
    features["contagion_risk_score"] = 10.0
    simulator = PaperPortfolioSimulator(prices, features, use_trained_model=False)
    result = simulator.simulate(
        transaction_cost_bps=0,
        rebalance_threshold=0.0,
        start_date=prices.index[22],
        max_bank_exposure=0.70,
    )

    calculated = result.weights.reindex(columns=BANKS + ["XFN.TO"]).fillna(0.0).sum(axis=1)
    pd.testing.assert_series_equal(result.ledger["financial_exposure"], calculated, check_names=False)
    pd.testing.assert_series_equal(result.ledger["bank_exposure"], calculated, check_names=False)
    assert (result.ledger["financial_exposure"] <= 0.70 + 1e-9).all()
