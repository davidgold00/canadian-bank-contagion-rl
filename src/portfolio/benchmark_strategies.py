from __future__ import annotations

import pandas as pd

from .allocation_policy import BANKS
from .covariance import price_returns


def benchmark_values(prices: pd.DataFrame, initial_capital: float = 100_000.0) -> pd.DataFrame:
    """Build simple institutional comparison benchmarks."""
    returns = price_returns(prices).fillna(0.0)
    out = pd.DataFrame(index=returns.index)
    bank_cols = [bank for bank in BANKS if bank in returns.columns]
    if bank_cols:
        out["Equal-weight Big Six"] = initial_capital * (1 + returns[bank_cols].mean(axis=1)).cumprod()
    for asset, label in [("XFN.TO", "XFN buy-and-hold"), ("XIU.TO", "XIU buy-and-hold"), ("XIC.TO", "XIC buy-and-hold")]:
        if asset in returns.columns:
            out[label] = initial_capital * (1 + returns[asset]).cumprod()
    out["Cash"] = initial_capital
    return out


def low_volatility_weights(returns: pd.DataFrame) -> pd.Series:
    vol = returns.std().replace(0, pd.NA)
    inv = (1 / vol).replace([pd.NA, float("inf")], pd.NA).fillna(0.0)
    return inv / inv.sum() if inv.sum() else pd.Series(1 / len(returns.columns), index=returns.columns)
