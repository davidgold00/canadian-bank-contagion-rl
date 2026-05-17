import numpy as np
import pandas as pd

from src.portfolio.allocation_policy import BANKS, CASH_ASSET, generate_model_allocation


def _sample_prices_features(n=40):
    dates = pd.bdate_range("2024-01-01", periods=n)
    prices = pd.DataFrame(index=dates)
    for i, asset in enumerate(BANKS + ["XFN.TO", "XIU.TO"]):
        prices[asset] = 100 + i * 3 + np.linspace(0, 8 + i, n)

    features = pd.DataFrame(index=dates)
    features["contagion_risk_score"] = np.linspace(25, 70, n)
    for bank in BANKS:
        features[f"{bank}_vol_21d"] = np.linspace(0.12, 0.22, n)
        features[f"{bank}_drawdown_63d"] = np.linspace(-0.02, -0.10, n)
        features[f"{bank}_beta_xfn_63d"] = np.linspace(0.8, 1.2, n)
    return prices, features


def test_fallback_weights_sum_to_one_and_are_long_only():
    prices, features = _sample_prices_features()
    result = generate_model_allocation(
        prices.iloc[:30],
        features.iloc[:30],
        use_trained_model=False,
        max_single_name_weight=0.20,
        max_bank_exposure=0.70,
    )

    assert abs(result.weights.sum() - 1.0) < 1e-9
    assert (result.weights >= 0).all()
    assert result.weights.loc[[b for b in BANKS if b in result.weights.index]].sum() <= 0.70 + 1e-9
    assert result.weights.loc[[b for b in BANKS if b in result.weights.index]].max() <= 0.20 + 1e-9
    assert CASH_ASSET in result.reasons


def test_allocation_uses_only_history_passed_to_it():
    prices, features = _sample_prices_features()
    history_prices = prices.iloc[:25]
    history_features = features.iloc[:25]

    future_shocked = features.copy()
    future_shocked.iloc[25:, future_shocked.columns.get_loc("contagion_risk_score")] = 99

    baseline = generate_model_allocation(history_prices, history_features, use_trained_model=False).weights
    future_changed = generate_model_allocation(history_prices, future_shocked.iloc[:25], use_trained_model=False).weights

    pd.testing.assert_series_equal(baseline, future_changed)
