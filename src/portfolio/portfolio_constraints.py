from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .allocation_policy import BANKS, CASH_ASSET, FINANCIAL_EXPOSURE_ASSETS


@dataclass(frozen=True)
class PortfolioConstraints:
    """Long-only institutional allocation guardrails."""

    max_single_name_weight: float = 0.22
    max_bank_exposure: float = 0.80
    min_cash_weight: float = 0.02
    max_cash_weight: float = 0.60
    long_only: bool = True


def _validate_constraints(constraints: PortfolioConstraints) -> None:
    values = {
        "max_single_name_weight": constraints.max_single_name_weight,
        "max_bank_exposure": constraints.max_bank_exposure,
        "min_cash_weight": constraints.min_cash_weight,
        "max_cash_weight": constraints.max_cash_weight,
    }
    for name, value in values.items():
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")
    if constraints.min_cash_weight > constraints.max_cash_weight:
        raise ValueError("min_cash_weight cannot exceed max_cash_weight.")


def _allocate_with_caps(
    weights: pd.Series,
    assets: list[str],
    amount: float,
    capacities: pd.Series,
) -> float:
    """Allocate `amount` proportionally without exceeding per-asset capacity."""
    remaining = float(amount)
    tolerance = 1e-12
    while remaining > tolerance:
        available = [asset for asset in assets if capacities.get(asset, 0.0) > tolerance]
        if not available:
            break
        basis = weights.reindex(available).clip(lower=0.0)
        if basis.sum() <= tolerance:
            basis = pd.Series(1.0, index=available)
        proposal = remaining * basis / basis.sum()
        addition = pd.concat([proposal, capacities.reindex(available)], axis=1).min(axis=1)
        added = float(addition.sum())
        if added <= tolerance:
            break
        weights.loc[available] += addition
        capacities.loc[available] -= addition
        remaining -= added
    return remaining


def normalize_long_only(weights: pd.Series, constraints: PortfolioConstraints) -> pd.Series:
    _validate_constraints(constraints)
    out = weights.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out = out.clip(lower=0.0)
    if CASH_ASSET not in out.index:
        out.loc[CASH_ASSET] = 0.0
    if out.sum() <= 0:
        out.loc[CASH_ASSET] = 1.0
    out = out / out.sum()

    for asset in [asset for asset in BANKS if asset in out.index]:
        if out.loc[asset] > constraints.max_single_name_weight:
            excess = out.loc[asset] - constraints.max_single_name_weight
            out.loc[asset] = constraints.max_single_name_weight
            out.loc[CASH_ASSET] += excess

    banks = [asset for asset in FINANCIAL_EXPOSURE_ASSETS if asset in out.index]
    bank_total = float(out.loc[banks].sum()) if banks else 0.0
    if bank_total > constraints.max_bank_exposure and bank_total > 0:
        released = bank_total - constraints.max_bank_exposure
        out.loc[banks] *= constraints.max_bank_exposure / bank_total
        out.loc[CASH_ASSET] += released

    if out.loc[CASH_ASSET] < constraints.min_cash_weight:
        shortfall = constraints.min_cash_weight - out.loc[CASH_ASSET]
        risky = [asset for asset in out.index if asset != CASH_ASSET]
        risky_sum = out.loc[risky].sum()
        if risky_sum > 0:
            out.loc[risky] *= max(risky_sum - shortfall, 0.0) / risky_sum
        out.loc[CASH_ASSET] = constraints.min_cash_weight

    if out.loc[CASH_ASSET] > constraints.max_cash_weight:
        excess = out.loc[CASH_ASSET] - constraints.max_cash_weight
        non_financial = [
            asset
            for asset in out.index
            if asset != CASH_ASSET and asset not in FINANCIAL_EXPOSURE_ASSETS
        ]
        if non_financial:
            basis = out.reindex(non_financial).clip(lower=0.0)
            if basis.sum() <= 1e-12:
                basis = pd.Series(1.0, index=non_financial)
            out.loc[non_financial] += excess * basis / basis.sum()
            excess = 0.0
        else:
            financial = [asset for asset in FINANCIAL_EXPOSURE_ASSETS if asset in out.index]
            financial_room = max(constraints.max_bank_exposure - float(out.reindex(financial).fillna(0.0).sum()), 0.0)
            if financial_room + 1e-12 < excess:
                raise ValueError("Portfolio constraints are infeasible for the available assets.")
            capacities = pd.Series(0.0, index=financial, dtype=float)
            for asset in financial:
                if asset in BANKS:
                    capacities.loc[asset] = max(constraints.max_single_name_weight - out.loc[asset], 0.0)
                else:
                    capacities.loc[asset] = financial_room
            excess = _allocate_with_caps(out, financial, excess, capacities)
        if excess > 1e-10:
            raise ValueError("Portfolio constraints are infeasible for the available assets.")
        out.loc[CASH_ASSET] = constraints.max_cash_weight

    out = out.clip(lower=0.0)
    out = out / out.sum()

    bank_assets = [asset for asset in BANKS if asset in out.index]
    financial_assets = [asset for asset in FINANCIAL_EXPOSURE_ASSETS if asset in out.index]
    feasible = (
        (not bank_assets or out.reindex(bank_assets).max() <= constraints.max_single_name_weight + 1e-8)
        and (not financial_assets or out.reindex(financial_assets).sum() <= constraints.max_bank_exposure + 1e-8)
        and constraints.min_cash_weight - 1e-8 <= out.loc[CASH_ASSET] <= constraints.max_cash_weight + 1e-8
    )
    if not feasible:
        raise ValueError("Portfolio constraints are infeasible for the available assets.")
    return out


def constraint_diagnostics(weights: pd.Series, constraints: PortfolioConstraints) -> pd.DataFrame:
    banks = [asset for asset in BANKS if asset in weights.index]
    financial_assets = [asset for asset in FINANCIAL_EXPOSURE_ASSETS if asset in weights.index]
    rows = [
        {
            "Constraint": "Weights sum to 1",
            "Value": float(weights.sum()),
            "Limit": 1.0,
            "Status": "Pass" if abs(weights.sum() - 1.0) < 1e-6 else "Review",
        },
        {
            "Constraint": "Long-only",
            "Value": float(weights.min()),
            "Limit": 0.0,
            "Status": "Pass" if weights.min() >= -1e-8 else "Breach",
        },
        {
            "Constraint": "Max single bank",
            "Value": float(weights.reindex(banks).fillna(0.0).max()) if banks else 0.0,
            "Limit": constraints.max_single_name_weight,
            "Status": "Pass" if not banks or weights.reindex(banks).fillna(0.0).max() <= constraints.max_single_name_weight + 1e-6 else "Breach",
        },
        {
            "Constraint": "Max total financial exposure",
            "Value": float(weights.reindex(financial_assets).fillna(0.0).sum()) if financial_assets else 0.0,
            "Limit": constraints.max_bank_exposure,
            "Status": "Pass"
            if not financial_assets
            or weights.reindex(financial_assets).fillna(0.0).sum() <= constraints.max_bank_exposure + 1e-6
            else "Breach",
        },
        {
            "Constraint": "Cash range",
            "Value": float(weights.get(CASH_ASSET, 0.0)),
            "Limit": f"{constraints.min_cash_weight:.1%} to {constraints.max_cash_weight:.1%}",
            "Status": "Pass"
            if constraints.min_cash_weight - 1e-6 <= weights.get(CASH_ASSET, 0.0) <= constraints.max_cash_weight + 1e-6
            else "Review",
        },
    ]
    return pd.DataFrame(rows)
