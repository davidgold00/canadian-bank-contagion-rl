from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class BenchmarkSpec:
    label: str
    tickers: tuple[str, ...]
    description: str
    default_shown: bool = False


BROAD_MARKET_BENCHMARKS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec(
        "S&P 500",
        ("^GSPC", "SPY"),
        "U.S. large-cap equities. A broad global risk-appetite benchmark.",
        True,
    ),
    BenchmarkSpec(
        "Nasdaq",
        ("^IXIC", "QQQ"),
        "Growth-heavy U.S. equities. Useful for comparing against a higher-beta equity benchmark.",
        True,
    ),
    BenchmarkSpec(
        "Dow Jones",
        ("^DJI", "DIA"),
        "Large blue-chip U.S. equities. Useful as a traditional equity-market yardstick.",
        True,
    ),
    BenchmarkSpec(
        "ZEB.TO",
        ("ZEB.TO",),
        "BMO Equal Weight Banks Index ETF. A practical Canadian bank-sector buy-and-hold proxy.",
        True,
    ),
)

CANADIAN_BANK_BENCHMARKS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec("RY.TO", ("RY.TO",), "Royal Bank of Canada buy-and-hold."),
    BenchmarkSpec("TD.TO", ("TD.TO",), "Toronto-Dominion Bank buy-and-hold."),
    BenchmarkSpec("BMO.TO", ("BMO.TO",), "Bank of Montreal buy-and-hold."),
    BenchmarkSpec("BNS.TO", ("BNS.TO",), "Bank of Nova Scotia buy-and-hold."),
    BenchmarkSpec("CM.TO", ("CM.TO",), "Canadian Imperial Bank of Commerce buy-and-hold."),
    BenchmarkSpec("NA.TO", ("NA.TO",), "National Bank of Canada buy-and-hold."),
)

BENCHMARK_SPECS: tuple[BenchmarkSpec, ...] = BROAD_MARKET_BENCHMARKS + CANADIAN_BANK_BENCHMARKS
BENCHMARK_BY_LABEL = {spec.label: spec for spec in BENCHMARK_SPECS}


def default_benchmark_labels() -> list[str]:
    return [spec.label for spec in BENCHMARK_SPECS if spec.default_shown]


def build_benchmark_values(
    prices: pd.DataFrame,
    selected_labels: Iterable[str],
    initial_capital: float,
    target_index: pd.Index | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Convert selected price series into buy-and-hold portfolio value curves."""
    if prices.empty:
        return pd.DataFrame(index=target_index), list(selected_labels)

    index = pd.DatetimeIndex(target_index) if target_index is not None else pd.DatetimeIndex(prices.index)
    values = pd.DataFrame(index=index)
    missing: list[str] = []

    for label in selected_labels:
        spec = BENCHMARK_BY_LABEL.get(label)
        if spec is None:
            missing.append(label)
            continue

        source = None
        for ticker in spec.tickers:
            if ticker in prices.columns:
                candidate = pd.to_numeric(prices[ticker], errors="coerce").dropna()
                if not candidate.empty:
                    source = candidate
                    break

        if source is None:
            missing.append(label)
            continue

        aligned = source.reindex(index, method="ffill")
        first_valid = aligned.dropna()
        if first_valid.empty or first_valid.iloc[0] == 0:
            missing.append(label)
            continue

        values[label] = aligned / first_valid.iloc[0] * float(initial_capital)

    return values, missing
