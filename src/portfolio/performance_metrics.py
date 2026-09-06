from __future__ import annotations

import math

import numpy as np
import pandas as pd


TRADING_DAYS = 252


# ── Additional institutional metrics ────────────────────────────────────────

def tracking_error(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Annualized standard deviation of active returns (portfolio − benchmark)."""
    p = _clean_returns(portfolio_returns)
    b = _clean_returns(benchmark_returns)
    common = p.index.intersection(b.index)
    if len(common) < 2:
        return 0.0
    active = p.loc[common] - b.loc[common]
    return float(active.std(ddof=1) * math.sqrt(periods_per_year))


def information_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Active return / tracking error. Measures skill per unit of active risk."""
    p = _clean_returns(portfolio_returns)
    b = _clean_returns(benchmark_returns)
    common = p.index.intersection(b.index)
    if len(common) < 2:
        return 0.0
    active = p.loc[common] - b.loc[common]
    te = float(active.std(ddof=1))
    if te == 0 or np.isnan(te):
        return 0.0
    return float(active.mean() / te * math.sqrt(periods_per_year))


def upside_capture(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """
    Upside capture ratio: how much of benchmark up-days the portfolio captures.
    >1.0 means the portfolio amplifies benchmark gains.
    """
    p = _clean_returns(portfolio_returns)
    b = _clean_returns(benchmark_returns)
    common = p.index.intersection(b.index)
    if len(common) < 2:
        return 1.0
    up_days = b.loc[common] > 0
    if up_days.sum() < 1:
        return 1.0
    port_up = p.loc[common][up_days]
    bench_up = b.loc[common][up_days]
    port_ann = (1 + port_up).prod() ** (periods_per_year / len(port_up)) - 1
    bench_ann = (1 + bench_up).prod() ** (periods_per_year / len(bench_up)) - 1
    return float(port_ann / bench_ann) if bench_ann != 0 else 1.0


def downside_capture(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """
    Downside capture ratio: how much of benchmark down-days the portfolio absorbs.
    <1.0 means the portfolio loses less than the benchmark on bad days.
    """
    p = _clean_returns(portfolio_returns)
    b = _clean_returns(benchmark_returns)
    common = p.index.intersection(b.index)
    if len(common) < 2:
        return 1.0
    down_days = b.loc[common] < 0
    if down_days.sum() < 1:
        return 1.0
    port_dn = p.loc[common][down_days]
    bench_dn = b.loc[common][down_days]
    port_ann = (1 + port_dn).prod() ** (periods_per_year / len(port_dn)) - 1
    bench_ann = (1 + bench_dn).prod() ** (periods_per_year / len(bench_dn)) - 1
    return float(port_ann / bench_ann) if bench_ann != 0 else 1.0


def win_rate(returns: pd.Series) -> float:
    """Fraction of days with positive return."""
    r = _clean_returns(returns)
    if r.empty:
        return 0.0
    return float((r > 0).mean())


def avg_win_loss_ratio(returns: pd.Series) -> float:
    """Average winning return divided by average losing return (positive number)."""
    r = _clean_returns(returns)
    wins = r[r > 0]
    losses = r[r < 0]
    if wins.empty or losses.empty:
        return 0.0
    return float(wins.mean() / abs(losses.mean()))


def ulcer_index(values: pd.Series) -> float:
    """
    Root-mean-square of percentage drawdowns.
    Lower is better; measures the depth and duration of drawdowns together.
    """
    dd = drawdown_series(values)
    if dd.empty:
        return 0.0
    return float(math.sqrt((dd ** 2).mean()))


def pain_ratio(values: pd.Series, returns: pd.Series | None = None, periods_per_year: int = TRADING_DAYS) -> float:
    """Annualized return divided by Ulcer Index. Higher is better."""
    ui = ulcer_index(values)
    if ui == 0:
        return 0.0
    ann_ret = annualized_return(values, periods_per_year)
    return float(ann_ret / ui)


def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    """
    Omega ratio: probability-weighted gains above threshold / losses below threshold.
    >1 means more mass above threshold than below.
    """
    r = _clean_returns(returns)
    if r.empty:
        return 1.0
    gains = (r - threshold).clip(lower=0).mean()
    losses = (threshold - r).clip(lower=0).mean()
    return float(gains / losses) if losses > 0 else float("inf")


def tail_ratio(returns: pd.Series, percentile: float = 0.05) -> float:
    """
    Absolute 95th percentile return / absolute 5th percentile loss.
    >1 means right tail is fatter than left tail (desirable).
    """
    r = _clean_returns(returns)
    if len(r) < 20:
        return 1.0
    right = abs(float(r.quantile(1 - percentile)))
    left = abs(float(r.quantile(percentile)))
    return float(right / left) if left > 0 else 1.0


def regime_performance(
    values: pd.Series,
    regime_series: pd.Series,
    regime_label: str,
) -> dict[str, float]:
    """
    Return annualized return and volatility for days where regime_series == regime_label.
    Useful for showing performance during stress vs. calm regimes.
    """
    common = values.index.intersection(regime_series.index)
    mask = regime_series.loc[common] == regime_label
    sub = values.loc[common][mask]
    if len(sub) < 5:
        return {"annualized_return": float("nan"), "annualized_volatility": float("nan"), "n_days": 0}
    sub_ret = sub.pct_change().dropna()
    return {
        "annualized_return": annualized_return(sub, TRADING_DAYS),
        "annualized_volatility": annualized_volatility(sub_ret, TRADING_DAYS),
        "n_days": int(len(sub)),
    }


def extended_performance_summary(
    values: pd.Series,
    returns: pd.Series | None = None,
    benchmark_values: pd.Series | None = None,
    turnover: pd.Series | None = None,
    costs: pd.Series | None = None,
    periods_per_year: int = TRADING_DAYS,
) -> dict[str, float]:
    """
    Full institutional performance summary including vs-benchmark metrics.
    Returns all keys from performance_summary plus:
    tracking_error, information_ratio, upside_capture, downside_capture,
    win_rate, avg_win_loss_ratio, ulcer_index, omega_ratio, tail_ratio.
    """
    base = performance_summary(values, returns, turnover, costs, periods_per_year)

    if returns is None:
        returns = values.pct_change().fillna(0.0)
    returns = pd.Series(returns, dtype=float).fillna(0.0)

    extra = {
        "win_rate": win_rate(returns),
        "avg_win_loss_ratio": avg_win_loss_ratio(returns),
        "ulcer_index": ulcer_index(values),
        "omega_ratio": omega_ratio(returns),
        "tail_ratio": tail_ratio(returns),
    }

    if benchmark_values is not None:
        bench_ret = pd.Series(benchmark_values, dtype=float).pct_change().fillna(0.0)
        extra["tracking_error"] = tracking_error(returns, bench_ret, periods_per_year)
        extra["information_ratio"] = information_ratio(returns, bench_ret, periods_per_year)
        extra["upside_capture"] = upside_capture(returns, bench_ret, periods_per_year)
        extra["downside_capture"] = downside_capture(returns, bench_ret, periods_per_year)

    return {**base, **extra}


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


def conditional_value_at_risk(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Historical CVaR / expected shortfall as a positive loss number."""
    returns = _clean_returns(returns)
    if returns.empty:
        return 0.0
    cutoff = returns.quantile(1 - confidence_level)
    tail = returns[returns <= cutoff]
    return float(-tail.mean()) if not tail.empty else 0.0


def rolling_cvar(returns: pd.Series, window: int = 63, confidence_level: float = 0.95) -> pd.Series:
    returns = pd.Series(returns, dtype=float).fillna(0.0)
    return returns.rolling(window).apply(lambda x: conditional_value_at_risk(pd.Series(x), confidence_level), raw=False)


def rolling_sharpe(returns: pd.Series, window: int = 63, periods_per_year: int = TRADING_DAYS) -> pd.Series:
    returns = pd.Series(returns, dtype=float).fillna(0.0)
    mean = returns.rolling(window).mean()
    vol = returns.rolling(window).std(ddof=0)
    return mean.div(vol.replace(0, np.nan)).mul(math.sqrt(periods_per_year)).fillna(0.0)


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
        "conditional_value_at_risk": conditional_value_at_risk(returns),
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
