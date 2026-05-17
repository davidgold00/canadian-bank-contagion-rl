from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


BANKS = ["RY.TO", "TD.TO", "BMO.TO", "BNS.TO", "CM.TO", "NA.TO"]
ETF_CANDIDATES = ["XFN.TO", "XIU.TO", "XIC.TO"]
CASH_ASSET = "cash"
DEFAULT_TRADABLES = BANKS + ["XFN.TO", "XIU.TO", CASH_ASSET]


@dataclass(frozen=True)
class AllocationPolicyResult:
    """Target weights plus the provenance needed to audit the allocation."""

    weights: pd.Series
    policy_source: str
    reasons: dict[str, str]
    diagnostics: dict[str, Any]


def available_assets(prices: pd.DataFrame, assets: Sequence[str] | None = None) -> list[str]:
    """Return long-only tradables available in the current price panel plus cash."""
    if assets is None:
        chosen = [asset for asset in DEFAULT_TRADABLES if asset == CASH_ASSET or asset in prices.columns]
        if "XIU.TO" not in chosen and "XIC.TO" in prices.columns:
            chosen.insert(-1 if CASH_ASSET in chosen else len(chosen), "XIC.TO")
    else:
        chosen = [asset for asset in assets if asset == CASH_ASSET or asset in prices.columns]
    if CASH_ASSET not in chosen:
        chosen.append(CASH_ASSET)
    return list(dict.fromkeys(chosen))


def load_ppo_model(model_path: str | Path | None = None) -> tuple[Any | None, str]:
    """Load a saved PPO model when stable-baselines3 and the artifact are available."""
    path = Path(model_path or "artifacts/rl/ppo_model.zip")
    if not path.exists():
        return None, f"No trained PPO artifact found at {path}."
    try:
        from stable_baselines3 import PPO

        return PPO.load(path), f"Loaded trained PPO model from {path}."
    except Exception as exc:  # pragma: no cover - depends on optional runtime libraries
        return None, f"PPO artifact exists but could not be loaded: {exc}"


def bank_node_stress(features_history: pd.DataFrame, banks: Sequence[str] = BANKS) -> pd.Series:
    """Estimate bank stress from volatility, drawdown, and sector beta using history to date only."""
    if features_history.empty:
        return pd.Series(50.0, index=list(banks))

    rows: dict[str, float] = {}
    for bank in banks:
        components: list[float] = []
        for col, sign in [
            (f"{bank}_vol_21d", 1.0),
            (f"{bank}_drawdown_63d", -1.0),
            (f"{bank}_beta_xfn_63d", 1.0),
        ]:
            if col not in features_history:
                continue
            series = sign * pd.to_numeric(features_history[col], errors="coerce")
            latest = series.dropna().iloc[-1] if series.dropna().size else np.nan
            if pd.notna(latest):
                components.append(float((series.dropna() <= latest).mean() * 100))
        rows[bank] = float(np.nanmean(components)) if components else 50.0
    return pd.Series(rows).clip(0, 100).fillna(50.0)


def _latest_risk(features_history: pd.DataFrame) -> float:
    if "contagion_risk_score" not in features_history:
        return 50.0
    series = pd.to_numeric(features_history["contagion_risk_score"], errors="coerce").dropna()
    return float(series.iloc[-1]) if len(series) else 50.0


def _softmax(action: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(action), dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=5.0, neginf=-5.0)
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    total = exp.sum()
    return exp / total if total else np.ones_like(exp) / len(exp)


def _ppo_weights(
    ppo_model: Any,
    prices_history: pd.DataFrame,
    features_history: pd.DataFrame,
    assets: Sequence[str],
    current_weights: pd.Series | None,
    lookback: int = 21,
) -> tuple[pd.Series | None, str]:
    """Infer PPO weights using the same observation shape as the training environment."""
    if ppo_model is None or len(prices_history) < lookback + 1:
        return None, "PPO unavailable or insufficient lookback."

    returns = prices_history.reindex(columns=[a for a in assets if a != CASH_ASSET]).pct_change().fillna(0.0)
    returns[CASH_ASSET] = 0.0
    returns = returns.reindex(columns=assets).fillna(0.0)

    feature_view = features_history.reindex(prices_history.index).ffill().fillna(0.0)
    feature_cols = list(feature_view.columns[:40])
    f_values = feature_view[feature_cols].iloc[-1].to_numpy(dtype=float) if feature_cols else np.array([], dtype=float)

    if current_weights is None:
        current = pd.Series(1 / len(assets), index=assets, dtype=float)
    else:
        current = current_weights.reindex(assets).fillna(0.0).astype(float)
        current = current / current.sum() if current.sum() > 0 else pd.Series(1 / len(assets), index=assets)

    obs = np.concatenate(
        [
            returns[assets].iloc[-lookback:].to_numpy(dtype=float).flatten(),
            np.nan_to_num(f_values, nan=0.0, posinf=0.0, neginf=0.0),
            current.to_numpy(dtype=float),
        ]
    ).astype(np.float32)

    expected_shape = getattr(getattr(ppo_model, "observation_space", None), "shape", None)
    if expected_shape and int(np.prod(expected_shape)) != len(obs):
        return None, f"PPO observation shape mismatch: expected {expected_shape}, got {len(obs)}."

    try:
        action, _ = ppo_model.predict(obs, deterministic=True)
    except Exception as exc:  # pragma: no cover - depends on optional model runtime
        return None, f"PPO inference failed: {exc}"

    action = np.asarray(action, dtype=float).flatten()
    if len(action) != len(assets):
        return None, f"PPO action shape mismatch: expected {len(assets)}, got {len(action)}."
    return pd.Series(_softmax(action), index=assets), "trained PPO model"


def _fallback_weights(
    prices_history: pd.DataFrame,
    features_history: pd.DataFrame,
    assets: Sequence[str],
    max_bank_exposure: float,
    defensive_cash_sensitivity: float,
) -> tuple[pd.Series, dict[str, Any]]:
    """Transparent stress-aware policy used when a PPO policy is unavailable."""
    risk = _latest_risk(features_history)
    banks = [bank for bank in BANKS if bank in assets and bank in prices_history.columns]
    stress = bank_node_stress(features_history, banks)

    returns_21d = prices_history[banks].pct_change(21).iloc[-1] if banks and len(prices_history) > 21 else pd.Series(0.0, index=banks)
    vol_21d = prices_history[banks].pct_change().rolling(21).std().iloc[-1] if banks and len(prices_history) > 21 else pd.Series(0.02, index=banks)

    cash_weight = 0.04 + defensive_cash_sensitivity * max(risk - 30.0, 0.0) / 70.0 * 0.62
    cash_weight = float(np.clip(cash_weight, 0.04, 0.82))

    xfn_weight = 0.0
    if "XFN.TO" in assets:
        xfn_weight = float(np.clip((58.0 - risk) / 100.0, 0.0, 0.18))

    market_etf = "XIU.TO" if "XIU.TO" in assets else "XIC.TO" if "XIC.TO" in assets else None
    market_weight = 0.0
    if market_etf:
        market_weight = float(np.clip((50.0 - risk) / 140.0, 0.0, 0.10))

    remaining = max(1.0 - cash_weight - xfn_weight - market_weight, 0.0)
    bank_budget = min(remaining, max_bank_exposure)
    cash_weight += max(remaining - bank_budget, 0.0)

    weights = pd.Series(0.0, index=assets, dtype=float)
    if banks and bank_budget > 0:
        stress_score = (100.0 - stress.reindex(banks).fillna(50.0)).clip(lower=1.0) / 100.0
        momentum_score = returns_21d.reindex(banks).fillna(0.0).rank(pct=True).fillna(0.5).clip(lower=0.1)
        inv_vol = 1.0 / vol_21d.reindex(banks).replace(0, np.nan)
        inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan).fillna(inv_vol.median() if pd.notna(inv_vol.median()) else 1.0)
        inv_vol_score = inv_vol.rank(pct=True).fillna(0.5).clip(lower=0.1)
        raw = 0.58 * stress_score + 0.24 * momentum_score + 0.18 * inv_vol_score
        raw = raw.clip(lower=0.01)
        weights.loc[banks] = bank_budget * raw / raw.sum()

    if "XFN.TO" in weights:
        weights.loc["XFN.TO"] = xfn_weight
    if market_etf:
        weights.loc[market_etf] = market_weight
    weights.loc[CASH_ASSET] = cash_weight

    diagnostics = {
        "contagion_risk_score": risk,
        "bank_node_stress": stress.to_dict(),
        "defensive_cash_sensitivity": defensive_cash_sensitivity,
    }
    return weights, diagnostics


def enforce_allocation_constraints(
    weights: pd.Series,
    max_single_name_weight: float = 0.22,
    max_bank_exposure: float = 0.80,
    banks: Sequence[str] = BANKS,
) -> pd.Series:
    """Enforce long-only, cash-balanced, capped bank allocation constraints."""
    out = weights.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    if CASH_ASSET not in out.index:
        out.loc[CASH_ASSET] = 0.0
    if out.sum() == 0:
        out.loc[CASH_ASSET] = 1.0
    out = out / out.sum()

    bank_cols = [bank for bank in banks if bank in out.index]
    for bank in bank_cols:
        if out.loc[bank] > max_single_name_weight:
            excess = out.loc[bank] - max_single_name_weight
            out.loc[bank] = max_single_name_weight
            out.loc[CASH_ASSET] += excess

    bank_total = float(out.loc[bank_cols].sum()) if bank_cols else 0.0
    if bank_total > max_bank_exposure and bank_total > 0:
        scale = max_bank_exposure / bank_total
        released = bank_total - max_bank_exposure
        out.loc[bank_cols] *= scale
        out.loc[CASH_ASSET] += released

    out = out.clip(lower=0.0)
    total = float(out.sum())
    if total <= 0:
        out.loc[CASH_ASSET] = 1.0
        total = 1.0
    return out / total


def generate_trade_reasons(
    target_weights: pd.Series,
    current_weights: pd.Series | None,
    risk_score: float,
    bank_stress: pd.Series | None = None,
    threshold: float = 0.005,
) -> dict[str, str]:
    """Explain each material allocation decision in plain English."""
    current = current_weights.reindex(target_weights.index).fillna(0.0) if current_weights is not None else pd.Series(0.0, index=target_weights.index)
    stress = bank_stress if bank_stress is not None else pd.Series(50.0, index=BANKS)
    reasons: dict[str, str] = {}

    for asset, target in target_weights.items():
        previous = float(current.get(asset, 0.0))
        change = float(target - previous)
        if abs(change) < threshold:
            reasons[asset] = "Maintained exposure because the risk signal did not justify a material rebalance."
            continue
        if asset == CASH_ASSET:
            if change > 0 and risk_score >= 60:
                reasons[asset] = "Increased cash because contagion risk exceeded the high-risk threshold."
            elif change > 0:
                reasons[asset] = "Increased cash to preserve liquidity and respect portfolio guardrails."
            else:
                reasons[asset] = "Reduced cash because systemic risk eased or lower-stress assets had room in the budget."
            continue
        if asset in BANKS:
            asset_stress = float(stress.get(asset, 50.0))
            if change < 0 and asset_stress >= 65:
                reasons[asset] = "Rotated away from high node-stress bank."
            elif change > 0 and asset_stress <= 45:
                reasons[asset] = "Increased allocation to lower-stress bank."
            elif change < 0 and risk_score >= 60:
                reasons[asset] = "Reduced bank exposure because contagion risk exceeded high-risk threshold."
            elif change > 0:
                reasons[asset] = "Rebalanced toward target weight after stress-adjusted ranking improved."
            else:
                reasons[asset] = "Rebalanced down to control concentration and turnover."
        elif asset == "XFN.TO":
            reasons[asset] = "Adjusted sector ETF exposure as a liquid Canadian financials sleeve."
        else:
            reasons[asset] = "Adjusted broad-market ETF exposure as a benchmark-aware stabilizer."
    return reasons


def generate_model_allocation(
    prices_history: pd.DataFrame,
    features_history: pd.DataFrame,
    assets: Sequence[str] | None = None,
    current_weights: pd.Series | None = None,
    max_single_name_weight: float = 0.22,
    max_bank_exposure: float = 0.80,
    defensive_cash_sensitivity: float = 1.0,
    use_trained_model: bool = True,
    ppo_model: Any | None = None,
) -> AllocationPolicyResult:
    """Generate target weights using information available up to the latest date only."""
    prices_history = prices_history.sort_index().copy()
    features_history = features_history.sort_index().copy()
    chosen_assets = available_assets(prices_history, assets)
    risk = _latest_risk(features_history)
    stress = bank_node_stress(features_history, [b for b in BANKS if b in chosen_assets])

    ppo_note = "PPO not requested."
    raw_weights: pd.Series | None = None
    policy_source = "rule-based fallback policy"
    if use_trained_model and ppo_model is not None:
        raw_weights, ppo_note = _ppo_weights(ppo_model, prices_history, features_history, chosen_assets, current_weights)
        if raw_weights is not None:
            policy_source = "trained PPO model"

    if raw_weights is None:
        raw_weights, fallback_diagnostics = _fallback_weights(
            prices_history=prices_history,
            features_history=features_history,
            assets=chosen_assets,
            max_bank_exposure=max_bank_exposure,
            defensive_cash_sensitivity=defensive_cash_sensitivity,
        )
    else:
        fallback_diagnostics = {}

    weights = enforce_allocation_constraints(raw_weights.reindex(chosen_assets).fillna(0.0), max_single_name_weight, max_bank_exposure)
    reasons = generate_trade_reasons(weights, current_weights, risk, stress)
    diagnostics = {
        "contagion_risk_score": risk,
        "bank_node_stress": stress.to_dict(),
        "policy_source": policy_source,
        "ppo_note": ppo_note,
        "observation_date": prices_history.index[-1] if len(prices_history) else None,
        **fallback_diagnostics,
    }
    return AllocationPolicyResult(weights=weights, policy_source=policy_source, reasons=reasons, diagnostics=diagnostics)
