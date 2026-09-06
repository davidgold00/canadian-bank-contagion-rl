from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .allocation_policy import BANKS, CASH_ASSET, available_assets
from .covariance import ledoit_wolf_covariance, price_returns, rolling_covariance
from .graph_adjustment import (
    GraphRiskMetrics,
    centrality_penalty_table,
    contagion_adjusted_covariance,
    graph_risk_metrics,
)
from .portfolio_constraints import FINANCIAL_EXPOSURE_ASSETS, PortfolioConstraints, constraint_diagnostics, normalize_long_only


@dataclass(frozen=True)
class CVaROptimizationResult:
    weights: pd.Series
    expected_returns: pd.Series
    base_covariance: pd.DataFrame
    adjusted_covariance: pd.DataFrame
    graph_metrics: GraphRiskMetrics
    diagnostics: dict[str, float | str]
    risk_contributions: pd.DataFrame
    constraint_diagnostics: pd.DataFrame


def historical_cvar(portfolio_returns: pd.Series | np.ndarray, confidence_level: float = 0.95) -> float:
    """Historical expected shortfall / CVaR as a positive loss number."""
    returns = pd.Series(portfolio_returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty:
        return 0.0
    cutoff = returns.quantile(1.0 - confidence_level)
    tail = returns[returns <= cutoff]
    if tail.empty:
        return 0.0
    return float(-tail.mean())


def expected_return_estimate(returns: pd.DataFrame, method: str = "blend") -> pd.Series:
    """Estimate annualized expected returns using conservative trailing signals."""
    clean = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if clean.empty:
        return pd.Series(dtype=float)
    mean_63 = clean.tail(min(63, len(clean))).mean() * 252
    mean_252 = clean.tail(min(252, len(clean))).mean() * 252
    if method == "momentum":
        est = mean_63
    elif method == "long":
        est = mean_252
    else:
        est = 0.35 * mean_63 + 0.65 * mean_252
    if CASH_ASSET in est.index:
        est.loc[CASH_ASSET] = 0.0
    return est.clip(lower=-0.25, upper=0.35)


def _portfolio_volatility(weights: np.ndarray, covariance: np.ndarray) -> float:
    return float(np.sqrt(max(weights @ covariance @ weights, 0.0)))


def _cvar_contributions(asset_returns: pd.DataFrame, weights: pd.Series, confidence_level: float) -> pd.Series:
    portfolio_returns = asset_returns.reindex(columns=weights.index).fillna(0.0) @ weights
    cutoff = portfolio_returns.quantile(1 - confidence_level)
    tail = asset_returns.loc[portfolio_returns <= cutoff].reindex(columns=weights.index).fillna(0.0)
    if tail.empty:
        return pd.Series(0.0, index=weights.index)
    contribution = -weights * tail.mean()
    total = contribution.sum()
    return contribution / total if total > 0 else contribution


def risk_contribution_table(
    weights: pd.Series,
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    asset_returns: pd.DataFrame,
    graph_metrics: GraphRiskMetrics,
    confidence_level: float,
) -> pd.DataFrame:
    cov = covariance.reindex(index=weights.index, columns=weights.index).fillna(0.0)
    sigma_w = cov.values @ weights.values
    vol = _portfolio_volatility(weights.values, cov.values)
    vol_contrib = weights.values * sigma_w / vol if vol > 0 else np.zeros(len(weights))
    cvar_contrib = _cvar_contributions(asset_returns, weights, confidence_level).reindex(weights.index).fillna(0.0)
    centrality = graph_metrics.centrality.reindex(weights.index).fillna(0.0)
    stress = graph_metrics.node_stress.reindex(weights.index).fillna(0.0)

    out = pd.DataFrame(
        {
            "Asset": weights.index,
            "Weight": weights.values,
            "Expected Return Contribution": weights * expected_returns.reindex(weights.index).fillna(0.0),
            "Volatility Contribution": vol_contrib,
            "CVaR Contribution": cvar_contrib.values,
            "Centrality": centrality.values,
            "Node Stress": stress.values,
            "Contagion Contribution": weights.values * (0.65 * centrality.values + 0.35 * stress.values / 100.0),
        }
    )
    return out.sort_values("Weight", ascending=False)


def optimize_cvar_portfolio(
    prices_history: pd.DataFrame,
    features_history: pd.DataFrame,
    assets: list[str] | None = None,
    confidence_level: float = 0.95,
    lookback_window: int = 252,
    covariance_method: str = "ledoit_wolf",
    risk_aversion: float = 6.0,
    cvar_penalty: float = 8.0,
    volatility_penalty: float = 1.0,
    contagion_penalty: float = 0.8,
    turnover_penalty: float = 0.15,
    graph_penalty_strength: float = 0.40,
    constraints: PortfolioConstraints | None = None,
    previous_weights: pd.Series | None = None,
) -> CVaROptimizationResult:
    """Run long-only graph-adjusted CVaR optimization using data available to date."""
    constraints = constraints or PortfolioConstraints()
    chosen_assets = available_assets(prices_history, assets)
    risky_assets = [asset for asset in chosen_assets if asset != CASH_ASSET]
    returns = price_returns(prices_history, risky_assets).tail(lookback_window).fillna(0.0)
    if returns.empty:
        weights = pd.Series(0.0, index=chosen_assets)
        weights.loc[CASH_ASSET] = 1.0
        covariance = pd.DataFrame(0.0, index=chosen_assets, columns=chosen_assets)
        graph_metrics = graph_risk_metrics(pd.DataFrame(index=prices_history.index), features_history)
        return CVaROptimizationResult(
            weights=weights,
            expected_returns=pd.Series(0.0, index=chosen_assets),
            base_covariance=covariance,
            adjusted_covariance=covariance,
            graph_metrics=graph_metrics,
            diagnostics={"status": "insufficient_data"},
            risk_contributions=pd.DataFrame(),
            constraint_diagnostics=constraint_diagnostics(weights, constraints),
        )

    returns_with_cash = returns.copy()
    returns_with_cash[CASH_ASSET] = 0.0
    returns_with_cash = returns_with_cash.reindex(columns=chosen_assets).fillna(0.0)

    base_cov_risky = rolling_covariance(returns, window=min(lookback_window, len(returns)), method=covariance_method)
    base_cov = pd.DataFrame(0.0, index=chosen_assets, columns=chosen_assets)
    base_cov.loc[base_cov_risky.index, base_cov_risky.columns] = base_cov_risky

    graph_metrics = graph_risk_metrics(returns, features_history)
    risk_score = 50.0
    if "contagion_risk_score" in features_history:
        score = pd.to_numeric(features_history["contagion_risk_score"], errors="coerce").dropna()
        if len(score):
            risk_score = float(score.iloc[-1])
    adjusted_cov = contagion_adjusted_covariance(base_cov, graph_metrics, risk_score, graph_penalty_strength)
    expected_returns = expected_return_estimate(returns_with_cash).reindex(chosen_assets).fillna(0.0)

    prev = previous_weights.reindex(chosen_assets).fillna(0.0) if previous_weights is not None else pd.Series(0.0, index=chosen_assets)
    if prev.sum() <= 0:
        prev = pd.Series(1.0 / len(chosen_assets), index=chosen_assets)
    prev = prev / prev.sum()

    n = len(chosen_assets)
    x0 = normalize_long_only(prev, constraints).reindex(chosen_assets).fillna(0.0).values
    cov_values = adjusted_cov.reindex(index=chosen_assets, columns=chosen_assets).fillna(0.0).values
    returns_values = returns_with_cash[chosen_assets].values
    mu = expected_returns.values
    centrality = graph_metrics.centrality.reindex(chosen_assets).fillna(0.0).values
    stress = graph_metrics.node_stress.reindex(chosen_assets).fillna(0.0).values / 100.0
    contagion_vector = 0.65 * centrality + 0.35 * stress

    def objective(w: np.ndarray) -> float:
        port = returns_values @ w
        cvar = historical_cvar(port, confidence_level)
        vol = _portfolio_volatility(w, cov_values)
        turnover = np.sum((w - prev.values) ** 2)
        contagion = float(w @ contagion_vector)
        expected = float(w @ mu)
        return (
            -expected
            + risk_aversion * 0.01 * vol
            + cvar_penalty * cvar
            + volatility_penalty * vol**2
            + contagion_penalty * contagion
            + turnover_penalty * turnover
        )

    bounds = []
    for asset in chosen_assets:
        if asset == CASH_ASSET:
            bounds.append((constraints.min_cash_weight, constraints.max_cash_weight))
        elif asset in BANKS:
            bounds.append((0.0, constraints.max_single_name_weight))
        else:
            bounds.append((0.0, 1.0))

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bank_idx = [i for i, asset in enumerate(chosen_assets) if asset in FINANCIAL_EXPOSURE_ASSETS]
    if bank_idx:
        cons.append({"type": "ineq", "fun": lambda w, idx=bank_idx: constraints.max_bank_exposure - np.sum(w[idx])})

    result = minimize(objective, x0=x0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 400, "ftol": 1e-10})
    raw = pd.Series(result.x if result.success else x0, index=chosen_assets)
    weights = normalize_long_only(raw, constraints)
    portfolio_returns = returns_with_cash[chosen_assets] @ weights
    vol = _portfolio_volatility(weights.values, cov_values)
    cvar = historical_cvar(portfolio_returns, confidence_level)
    diagnostics = {
        "status": "optimal" if result.success else f"fallback: {result.message}",
        "objective": float(objective(weights.values)),
        "expected_return": float(weights @ expected_returns),
        "annualized_volatility": vol,
        "historical_cvar": cvar,
        "contagion_score": risk_score,
        "average_correlation": graph_metrics.average_correlation,
        "graph_density": graph_metrics.graph_density,
        "largest_eigenvalue": graph_metrics.largest_eigenvalue,
        "turnover_from_previous": float(np.abs(weights - prev).sum()),
    }
    contrib = risk_contribution_table(weights, expected_returns, adjusted_cov, returns_with_cash, graph_metrics, confidence_level)
    return CVaROptimizationResult(
        weights=weights,
        expected_returns=expected_returns,
        base_covariance=base_cov,
        adjusted_covariance=adjusted_cov,
        graph_metrics=graph_metrics,
        diagnostics=diagnostics,
        risk_contributions=contrib,
        constraint_diagnostics=constraint_diagnostics(weights, constraints),
    )


def parametric_cvar(
    weights: pd.Series,
    covariance: pd.DataFrame,
    expected_returns: pd.Series,
    confidence_level: float = 0.95,
) -> float:
    """
    Parametric (Gaussian) CVaR assuming normally distributed portfolio returns.

    CVaR_parametric = -mu_p + sigma_p * phi(z) / (1 - alpha)
    where phi(z) is the normal PDF at the z-alpha quantile.

    Useful as a fast sanity check vs. historical CVaR.
    """
    from scipy.stats import norm

    w = weights.reindex(covariance.index).fillna(0.0).values
    cov = covariance.values
    mu_p = float(weights.reindex(expected_returns.index).fillna(0.0) @ expected_returns.values)
    var_p = float(w @ cov @ w)
    sigma_p = math.sqrt(max(var_p, 0.0))
    if sigma_p == 0:
        return 0.0
    z = norm.ppf(1 - confidence_level)
    return float(-mu_p / 252 + sigma_p * norm.pdf(z) / (1 - confidence_level))


def monte_carlo_cvar(
    weights: pd.Series,
    covariance: pd.DataFrame,
    expected_returns: pd.Series,
    n_simulations: int = 10_000,
    horizon_days: int = 1,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> dict[str, float]:
    """
    Monte Carlo CVaR and VaR via Cholesky-decomposed multivariate normal simulation.

    Returns
    -------
    dict with keys: mc_cvar, mc_var, mc_expected_return, mc_volatility, n_simulations.
    """
    rng = np.random.default_rng(random_state)
    assets = weights.index
    w = weights.values
    cov_aligned = covariance.reindex(index=assets, columns=assets).fillna(0.0).values
    mu_aligned = expected_returns.reindex(assets).fillna(0.0).values / 252.0

    try:
        L = np.linalg.cholesky(cov_aligned + 1e-9 * np.eye(len(assets)))
    except np.linalg.LinAlgError:
        eigvals = np.linalg.eigvalsh(cov_aligned)
        shift = max(-eigvals.min() + 1e-6, 0)
        L = np.linalg.cholesky(cov_aligned + shift * np.eye(len(assets)))

    z = rng.standard_normal((n_simulations, len(assets)))
    daily_returns = mu_aligned + z @ L.T
    if horizon_days > 1:
        port_returns = np.prod(1 + daily_returns[:, :horizon_days] @ w, axis=1) - 1 if daily_returns.ndim == 3 else (daily_returns @ w) * horizon_days
    else:
        port_returns = daily_returns @ w

    cutoff = np.percentile(port_returns, (1 - confidence_level) * 100)
    tail = port_returns[port_returns <= cutoff]

    return {
        "mc_cvar": float(-tail.mean()) if len(tail) else 0.0,
        "mc_var": float(-cutoff),
        "mc_expected_return": float(port_returns.mean() * 252),
        "mc_volatility": float(port_returns.std() * math.sqrt(252)),
        "n_simulations": n_simulations,
    }


def stress_scenario_loss(
    weights: pd.Series,
    shock_returns: dict[str, float],
) -> dict[str, float]:
    """
    Compute approximate portfolio loss under a named stress scenario.

    Parameters
    ----------
    weights : portfolio weights indexed by asset ticker
    shock_returns : dict mapping asset ticker -> 1-day shock return (e.g. {"RY.TO": -0.08})

    Returns
    -------
    dict: total_loss, loss_by_asset (dict), largest_contributor
    """
    total_loss = 0.0
    loss_by_asset: dict[str, float] = {}
    for asset, w in weights.items():
        shock = shock_returns.get(asset, 0.0)
        contrib = float(w * shock)
        loss_by_asset[asset] = contrib
        total_loss += contrib

    if loss_by_asset:
        largest = min(loss_by_asset, key=loss_by_asset.get)  # most negative
    else:
        largest = ""

    return {
        "total_loss": total_loss,
        "loss_by_asset": loss_by_asset,
        "largest_contributor": largest,
    }


def risk_decomposition(
    weights: pd.Series,
    covariance: pd.DataFrame,
) -> pd.DataFrame:
    """
    Full marginal, component, and percentage risk decomposition.

    Returns a DataFrame with columns:
    Asset, Weight, Marginal Risk, Component Risk, % of Total Risk, Standalone Vol.
    """
    assets = weights.index
    w = weights.reindex(assets).fillna(0.0).values
    cov = covariance.reindex(index=assets, columns=assets).fillna(0.0).values
    port_var = float(w @ cov @ w)
    port_vol = math.sqrt(max(port_var, 0.0))

    sigma_w = cov @ w
    marginal_risk = sigma_w / port_vol if port_vol > 0 else np.zeros(len(w))
    component_risk = w * marginal_risk
    standalone_vol = np.sqrt(np.diag(cov))

    return pd.DataFrame({
        "Asset": assets,
        "Weight": w,
        "Standalone Vol": standalone_vol,
        "Marginal Risk": marginal_risk,
        "Component Risk": component_risk,
        "% of Portfolio Risk": component_risk / port_vol if port_vol > 0 else np.zeros(len(w)),
    }).sort_values("Component Risk", ascending=False)


def efficient_frontier(
    prices_history: pd.DataFrame,
    features_history: pd.DataFrame,
    assets: list[str] | None = None,
    risk_aversion_values: list[float] | None = None,
    confidence_level: float = 0.95,
    lookback_window: int = 252,
    constraints: PortfolioConstraints | None = None,
) -> pd.DataFrame:
    values = risk_aversion_values or [1, 2, 4, 6, 8, 10, 14, 18]
    rows = []
    previous = None
    for value in values:
        result = optimize_cvar_portfolio(
            prices_history=prices_history,
            features_history=features_history,
            assets=assets,
            confidence_level=confidence_level,
            lookback_window=lookback_window,
            risk_aversion=value,
            constraints=constraints,
            previous_weights=previous,
        )
        rows.append(
            {
                "Risk Aversion": value,
                "Expected Return": result.diagnostics["expected_return"],
                "Volatility": result.diagnostics["annualized_volatility"],
                "CVaR": result.diagnostics["historical_cvar"],
                "Bank Exposure": result.weights.reindex(BANKS).fillna(0.0).sum(),
                "Cash": result.weights.get(CASH_ASSET, 0.0),
            }
        )
        previous = result.weights
    return pd.DataFrame(rows)


def optimizer_tables(result: CVaROptimizationResult) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weights = result.risk_contributions.copy()
    weights["Expected Return"] = weights["Asset"].map(result.expected_returns)
    penalty = centrality_penalty_table(result.graph_metrics)
    diagnostics = pd.DataFrame([{"Metric": key, "Value": value} for key, value in result.diagnostics.items()])
    return weights, penalty, diagnostics
