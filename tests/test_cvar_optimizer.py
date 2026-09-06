import numpy as np
import pandas as pd

from src.portfolio.allocation_policy import BANKS
from src.portfolio.cvar_optimizer import historical_cvar, optimize_cvar_portfolio
from src.portfolio.portfolio_constraints import PortfolioConstraints


def _sample_data(n=140):
    dates = pd.bdate_range("2023-01-01", periods=n)
    prices = pd.DataFrame(index=dates)
    for i, asset in enumerate(BANKS + ["XFN.TO", "XIU.TO"]):
        daily = 1 + 0.0004 + i * 0.00003 + 0.002 * np.sin(np.linspace(0, 8, n) + i)
        prices[asset] = (100 + i) * np.cumprod(daily)
    features = pd.DataFrame(index=dates)
    features["contagion_risk_score"] = np.linspace(35, 75, n)
    for bank in BANKS:
        features[f"{bank}_vol_21d"] = np.linspace(0.12, 0.25, n)
        features[f"{bank}_drawdown_63d"] = np.linspace(-0.02, -0.12, n)
        features[f"{bank}_beta_xfn_63d"] = np.linspace(0.8, 1.3, n)
    return prices, features


def test_cvar_optimizer_weights_are_feasible():
    prices, features = _sample_data()
    constraints = PortfolioConstraints(max_single_name_weight=0.18, max_bank_exposure=0.65, min_cash_weight=0.05)
    result = optimize_cvar_portfolio(prices, features, constraints=constraints, lookback_window=100)

    assert abs(result.weights.sum() - 1.0) < 1e-6
    assert (result.weights >= -1e-8).all()
    assert result.weights.reindex(BANKS).fillna(0.0).sum() <= 0.65 + 1e-6
    assert result.weights.get("cash", 0.0) >= 0.05 - 1e-6
    assert result.diagnostics["historical_cvar"] >= 0


def test_defensive_cvar_is_not_above_equal_weight_cvar():
    prices, features = _sample_data()
    constraints = PortfolioConstraints(max_single_name_weight=0.16, max_bank_exposure=0.55, min_cash_weight=0.20, max_cash_weight=0.70)
    result = optimize_cvar_portfolio(
        prices,
        features,
        risk_aversion=10.0,
        cvar_penalty=12.0,
        constraints=constraints,
        lookback_window=100,
    )
    returns = prices[[c for c in result.weights.index if c != "cash"]].pct_change().dropna().tail(100)
    asset_returns = returns.copy()
    asset_returns["cash"] = 0.0
    asset_returns = asset_returns.reindex(columns=result.weights.index).fillna(0.0)
    opt_cvar = historical_cvar(asset_returns @ result.weights)

    equal = pd.Series(1 / len(result.weights), index=result.weights.index)
    equal_cvar = historical_cvar(asset_returns @ equal)
    assert opt_cvar <= equal_cvar + 0.01
