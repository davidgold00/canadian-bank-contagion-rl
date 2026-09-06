from __future__ import annotations

import pandas as pd


def latest_contagion_score(features_history: pd.DataFrame) -> float:
    if "contagion_risk_score" not in features_history:
        return 50.0
    score = pd.to_numeric(features_history["contagion_risk_score"], errors="coerce").dropna()
    return float(score.iloc[-1]) if len(score) else 50.0


def regime_label(score: float) -> str:
    if score < 30:
        return "Low"
    if score < 60:
        return "Moderate"
    if score < 80:
        return "High"
    return "Severe"


def regime_constraints(score: float) -> dict[str, float]:
    """Regime-aware guardrails used by CVaR allocation pages."""
    if score >= 80:
        return {"min_cash_weight": 0.18, "max_bank_exposure": 0.45, "max_single_name_weight": 0.14}
    if score >= 60:
        return {"min_cash_weight": 0.10, "max_bank_exposure": 0.60, "max_single_name_weight": 0.18}
    if score >= 30:
        return {"min_cash_weight": 0.04, "max_bank_exposure": 0.75, "max_single_name_weight": 0.22}
    return {"min_cash_weight": 0.02, "max_bank_exposure": 0.85, "max_single_name_weight": 0.25}
