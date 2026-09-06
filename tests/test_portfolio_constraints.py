import pandas as pd
import pytest

from src.portfolio.portfolio_constraints import (
    PortfolioConstraints,
    constraint_diagnostics,
    normalize_long_only,
)


def test_projection_redistributes_excess_cash_without_rebreaching_bank_caps():
    constraints = PortfolioConstraints(
        max_single_name_weight=0.20,
        max_bank_exposure=0.70,
        min_cash_weight=0.05,
        max_cash_weight=0.60,
    )
    raw = pd.Series({"RY.TO": 0.10, "TD.TO": 0.0, "BMO.TO": 0.0, "cash": 0.90})

    projected = normalize_long_only(raw, constraints)

    assert abs(projected.sum() - 1.0) < 1e-9
    assert projected[["RY.TO", "TD.TO", "BMO.TO"]].max() <= 0.20 + 1e-9
    assert projected[["RY.TO", "TD.TO", "BMO.TO"]].sum() <= 0.70 + 1e-9
    assert constraints.min_cash_weight <= projected["cash"] <= constraints.max_cash_weight
    assert (constraint_diagnostics(projected, constraints)["Status"] == "Pass").all()


def test_xfn_is_limited_as_financial_exposure_not_mislabeled_as_single_bank():
    constraints = PortfolioConstraints(
        max_single_name_weight=0.20,
        max_bank_exposure=0.70,
        min_cash_weight=0.05,
        max_cash_weight=0.60,
    )
    raw = pd.Series({"XFN.TO": 0.90, "XIU.TO": 0.0, "cash": 0.10})

    projected = normalize_long_only(raw, constraints)
    diagnostics = constraint_diagnostics(projected, constraints).set_index("Constraint")

    assert projected["XFN.TO"] <= constraints.max_bank_exposure + 1e-9
    assert diagnostics.loc["Max single bank", "Value"] == 0.0
    assert diagnostics.loc["Max single bank", "Status"] == "Pass"
    assert diagnostics.loc["Max total financial exposure", "Status"] == "Pass"


def test_projection_rejects_constraints_that_available_assets_cannot_satisfy():
    constraints = PortfolioConstraints(
        max_single_name_weight=0.20,
        max_bank_exposure=0.20,
        min_cash_weight=0.05,
        max_cash_weight=0.60,
    )

    with pytest.raises(ValueError, match="infeasible"):
        normalize_long_only(pd.Series({"RY.TO": 0.0, "cash": 1.0}), constraints)
