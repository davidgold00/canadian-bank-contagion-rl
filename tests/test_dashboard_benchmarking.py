import pandas as pd

from src.dashboard.benchmarking import build_benchmark_values, default_benchmark_labels


def test_default_benchmarks_include_requested_market_yardsticks() -> None:
    assert default_benchmark_labels() == ["S&P 500", "Nasdaq", "Dow Jones", "ZEB.TO"]


def test_build_benchmark_values_normalizes_selected_price_series() -> None:
    dates = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.DataFrame(
        {
            "^GSPC": [5000.0, 5050.0, 5100.0],
            "ZEB.TO": [40.0, 39.0, 42.0],
            "RY.TO": [120.0, 123.0, 126.0],
        },
        index=dates,
    )

    values, missing = build_benchmark_values(
        prices,
        ["S&P 500", "ZEB.TO", "RY.TO", "Nasdaq"],
        initial_capital=100_000,
        target_index=dates,
    )

    assert missing == ["Nasdaq"]
    assert list(values.columns) == ["S&P 500", "ZEB.TO", "RY.TO"]
    assert values.loc[dates[0], "S&P 500"] == 100_000
    assert values.loc[dates[-1], "S&P 500"] == 102_000
    assert values.loc[dates[-1], "ZEB.TO"] == 105_000
    assert values.loc[dates[-1], "RY.TO"] == 105_000
