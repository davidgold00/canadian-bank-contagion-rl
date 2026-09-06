from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


TRADING_DAYS = 252


def price_returns(prices: pd.DataFrame, assets: list[str] | None = None) -> pd.DataFrame:
    """Convert prices into clean daily returns."""
    view = prices.copy()
    if "date" in view.columns:
        view["date"] = pd.to_datetime(view["date"], errors="coerce")
        view = view.set_index("date")
    view.index = pd.to_datetime(view.index)
    view = view.sort_index()
    if assets is not None:
        view = view[[asset for asset in assets if asset in view.columns]]
    return view.apply(pd.to_numeric, errors="coerce").ffill().pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")


def ensure_psd(matrix: pd.DataFrame, epsilon: float = 1e-8) -> pd.DataFrame:
    """Project a symmetric covariance-like matrix to positive semidefinite form."""
    if matrix.empty:
        return matrix
    values = np.asarray(matrix, dtype=float)
    values = np.nan_to_num((values + values.T) / 2, nan=0.0, posinf=0.0, neginf=0.0)
    eigvals, eigvecs = np.linalg.eigh(values)
    eigvals = np.clip(eigvals, epsilon, None)
    psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    psd = (psd + psd.T) / 2
    return pd.DataFrame(psd, index=matrix.index, columns=matrix.columns)


def sample_covariance(returns: pd.DataFrame, annualize: bool = True) -> pd.DataFrame:
    cov = returns.dropna(how="all").cov().fillna(0.0)
    if annualize:
        cov *= TRADING_DAYS
    return ensure_psd(cov)


def ledoit_wolf_covariance(returns: pd.DataFrame, annualize: bool = True) -> pd.DataFrame:
    """Estimate a shrinkage covariance matrix robustly for small return samples."""
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if len(clean) < 3 or clean.shape[1] < 2:
        return sample_covariance(returns, annualize=annualize)
    estimator = LedoitWolf().fit(clean.values)
    cov = estimator.covariance_
    if annualize:
        cov *= TRADING_DAYS
    return ensure_psd(pd.DataFrame(cov, index=clean.columns, columns=clean.columns))


def exponentially_weighted_covariance(
    returns: pd.DataFrame,
    span: int = 63,
    annualize: bool = True,
) -> pd.DataFrame:
    """Estimate covariance with exponentially higher weight on recent observations."""
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if len(clean) < 3:
        return sample_covariance(returns, annualize=annualize)
    demeaned = clean - clean.ewm(span=span).mean()
    weights = np.exp(np.linspace(-1.0, 0.0, len(clean)))
    weights = weights / weights.sum()
    values = demeaned.values
    cov = values.T @ np.diag(weights) @ values
    if annualize:
        cov *= TRADING_DAYS
    return ensure_psd(pd.DataFrame(cov, index=clean.columns, columns=clean.columns))


def rolling_covariance(
    returns: pd.DataFrame,
    window: int = 126,
    method: str = "ledoit_wolf",
    annualize: bool = True,
) -> pd.DataFrame:
    """Estimate covariance using the trailing window only."""
    trailing = returns.tail(window)
    if method == "ewma":
        return exponentially_weighted_covariance(trailing, span=max(10, window // 2), annualize=annualize)
    if method == "sample":
        return sample_covariance(trailing, annualize=annualize)
    return ledoit_wolf_covariance(trailing, annualize=annualize)
