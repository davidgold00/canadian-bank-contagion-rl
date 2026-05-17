from __future__ import annotations

import math

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def _clean_returns(returns: pd.Series) -> pd.Series:
    return pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()


def cumulative_return(values: pd.Series) -> float:
    """Return total return for a portfolio value series."""
    values = pd.Series(values, dtype=float).dropna()
    if len(values) < 2 or values.iloc[0] == 0:
        return 0.0
    return float(values.iloc[-1] / values.iloc[0] - 1)


def annualized_return(values: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Compound total return to an annualized rate."""
    values = pd.Series(values, dtype=float).dropna()
    if len(values) < 2 or values.iloc[0] <= 0:
        return 0.0
    total = values.iloc[-1] / values.iloc[0]
    years = max((len(values) - 1) / periods_per_year, 1 / periods_per_year)
    if total <= 0:
        return -1.0
    return float(total ** (1 / years) - 1)


def annualized_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    returns = _clean_returns(returns)
    if returns.empty:
        return 0.0
    return float(returns.std(ddof=0) * math.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    returns = _clean_returns(returns)
    if returns.empty:
        return 0.0
    excess = returns - risk_free_rate / periods_per_year
    vol = excess.std(ddof=0)
    if vol == 0 or np.isnan(vol):
        return 0.0
    return float(excess.mean() / vol * math.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, target_return: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    returns = _clean_returns(returns)
    if returns.empty:
        return 0.0
    downside = returns[returns < target_return / periods_per_year]
    downside_std = downside.std(ddof=0)
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0
    return float((returns.mean() - target_return / periods_per_year) / downside_std * math.sqrt(periods_per_year))


def drawdown_series(values: pd.Series) -> pd.Series:
    values = pd.Series(values, dtype=float).dropna()
    if values.empty:
        return values
    return values / values.cummax() - 1


def max_drawdown(values: pd.Series) -> float:
    dd = drawdown_series(values)
    return float(dd.min()) if not dd.empty else 0.0


def calmar_ratio(values: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    mdd = abs(max_drawdown(values))
    if mdd == 0:
        return 0.0
    return float(annualized_return(values, periods_per_year) / mdd)


def average_turnover(turnover: pd.Series) -> float:
    turnover = pd.Series(turnover, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    return float(turnover.mean()) if not turnover.empty else 0.0


def performance_summary(
    values: pd.Series,
    returns: pd.Series | None = None,
    turnover: pd.Series | None = None,
    costs: pd.Series | None = None,
    periods_per_year: int = TRADING_DAYS,
) -> dict[str, float]:
    """Create a compact institutional-style performance summary."""
    values = pd.Series(values, dtype=float).dropna()
    if returns is None:
        returns = values.pct_change().fillna(0.0)
    returns = pd.Series(returns, dtype=float).fillna(0.0)
    turnover = pd.Series(turnover if turnover is not None else 0.0, index=values.index, dtype=float)
    costs = pd.Series(costs if costs is not None else 0.0, index=values.index, dtype=float)

    return {
        "ending_value": float(values.iloc[-1]) if len(values) else 0.0,
        "cumulative_return": cumulative_return(values),
        "annualized_return": annualized_return(values, periods_per_year),
        "annualized_volatility": annualized_volatility(returns, periods_per_year),
        "sharpe_ratio": sharpe_ratio(returns, periods_per_year=periods_per_year),
        "sortino_ratio": sortino_ratio(returns, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(values),
        "calmar_ratio": calmar_ratio(values, periods_per_year),
        "average_daily_turnover": average_turnover(turnover),
        "total_transaction_costs": float(costs.sum()),
    }


def benchmark_summary(benchmarks: pd.DataFrame, periods_per_year: int = TRADING_DAYS) -> pd.DataFrame:
    """Summarize benchmark value series using the same metrics as the simulated fund."""
    rows = []
    for name in benchmarks.columns:
        values = benchmarks[name].dropna()
        if values.empty:
            continue
        returns = values.pct_change().fillna(0.0)
        row = performance_summary(values, returns, periods_per_year=periods_per_year)
        row["Benchmark"] = name
        rows.append(row)
    return pd.DataFrame(rows)
