"""
Investment signal generation for Northern Signal.

Translates raw risk metrics into explicit portfolio decisions:
  - Multi-factor composite scores per bank (momentum, stress, mean-reversion, macro)
  - BUY / HOLD / REDUCE signals with conviction (1–5) and rationale
  - Regime-conditioned target weights with delta vs. a neutral baseline
  - Portfolio action table ready for display
  - Sensitivity analysis: what happens to targets if the contagion score rises
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.dashboard.insight_utils import (
    BANK_CONTEXT,
    BANK_NAMES,
    BANKS,
    latest,
    percentile_rank,
    risk_regime,
)

_NEUTRAL_WEIGHT = 1.0 / len(BANKS)

_MACRO_BULL_THRESHOLD = 0.55
_MACRO_BEAR_THRESHOLD = 0.40


def _safe_pctrank(series: pd.Series, value: float) -> float:
    """Return percentile rank, clipped to [0, 1], defaulting to 0.5 on failure."""
    try:
        return float(np.clip(percentile_rank(series, value), 0.0, 1.0))
    except Exception:
        return 0.5


def _momentum_score(bank: str, features: pd.DataFrame) -> float:
    """
    0–100. Blends relative return (vs sector) over 5D and 21D windows.
    Higher = better momentum vs the peer group.
    """
    ret_5d = latest(features, f"{bank}_ret_5d", np.nan)
    ret_21d = latest(features, f"{bank}_ret_21d", np.nan)
    sector_5d = latest(features, "XFN.TO_ret_5d", np.nan)
    sector_21d = latest(features, "XFN.TO_ret_21d", np.nan)

    rel_5d = (ret_5d - sector_5d) if pd.notna(ret_5d) and pd.notna(sector_5d) else 0.0
    rel_21d = (ret_21d - sector_21d) if pd.notna(ret_21d) and pd.notna(sector_21d) else 0.0

    # Percentile rank vs all banks (cross-sectional)
    bank_ret_5d_series = pd.Series({b: latest(features, f"{b}_ret_5d", np.nan) for b in BANKS}).dropna()
    bank_ret_21d_series = pd.Series({b: latest(features, f"{b}_ret_21d", np.nan) for b in BANKS}).dropna()

    rank_5d = _safe_pctrank(bank_ret_5d_series, ret_5d) if pd.notna(ret_5d) and len(bank_ret_5d_series) else 0.5
    rank_21d = _safe_pctrank(bank_ret_21d_series, ret_21d) if pd.notna(ret_21d) and len(bank_ret_21d_series) else 0.5

    # Blend: 40% 5D cross-sectional, 60% 21D cross-sectional, with a relative boost
    raw = 0.40 * rank_5d + 0.60 * rank_21d
    rel_boost = np.tanh(50 * rel_21d) * 0.05
    return float(np.clip((raw + rel_boost) * 100, 0, 100))


def _stress_score(bank: str, features: pd.DataFrame) -> float:
    """
    0–100. HIGH score = HIGH stress (bad for going long).
    Combines volatility, drawdown, and sector beta percentile ranks.
    """
    vol = latest(features, f"{bank}_vol_21d", np.nan)
    dd = latest(features, f"{bank}_drawdown_63d", np.nan)
    beta = latest(features, f"{bank}_beta_xfn_63d", np.nan)

    components = []
    for col, sign in [
        (f"{bank}_vol_21d", 1),
        (f"{bank}_drawdown_63d", -1),
        (f"{bank}_beta_xfn_63d", 1),
    ]:
        val = latest(features, col, np.nan)
        if pd.notna(val) and col in features:
            direction_series = sign * features[col].dropna()
            direction_val = sign * val
            components.append(_safe_pctrank(direction_series, direction_val))

    return float(np.nanmean(components) * 100) if components else 50.0


def _mean_reversion_score(bank: str, features: pd.DataFrame) -> float:
    """
    0–100. HIGH score = oversold / likely recovery candidate.
    Uses distance from 52-week high after a meaningful drawdown.
    Positive signal only when paired with stabilizing macro environment.
    """
    dd_col = f"{bank}_drawdown_63d"
    dist_col = f"{bank}_dist_52w_high"

    dd = latest(features, dd_col, 0.0)
    dist = latest(features, dist_col, 0.0)

    if dd_col not in features or dist_col not in features:
        return 50.0

    dd_pct = _safe_pctrank(-features[dd_col].dropna(), -dd)
    dist_pct = _safe_pctrank(-features[dist_col].dropna(), -dist)

    raw = 0.50 * dd_pct + 0.50 * dist_pct
    return float(np.clip(raw * 100, 0, 100))


def _macro_tailwind_score(features: pd.DataFrame, macro: pd.DataFrame) -> float:
    """
    0–100. HIGH score = macro environment favourable for Canadian bank longs.
    Combines yield-curve shape, VIX regime, oil, CAD.
    """
    slope = latest(features, "slope_10y_2y", np.nan)
    vix = latest(features, "VIX_level", np.nan)
    oil_21d = latest(features, "CL=F_ret_21d", np.nan)
    cad_21d = latest(features, "CADUSD=X_ret_21d", np.nan)

    components = []

    if pd.notna(slope) and "slope_10y_2y" in features:
        slope_pct = _safe_pctrank(features["slope_10y_2y"].dropna(), slope)
        components.append(slope_pct)

    if pd.notna(vix) and "VIX_level" in features:
        vix_pct = 1.0 - _safe_pctrank(features["VIX_level"].dropna(), vix)
        components.append(vix_pct)

    if pd.notna(oil_21d) and "CL=F_ret_21d" in features:
        oil_pct = _safe_pctrank(features["CL=F_ret_21d"].dropna(), oil_21d)
        components.append(oil_pct)

    if pd.notna(cad_21d) and "CADUSD=X_ret_21d" in features:
        cad_pct = _safe_pctrank(features["CADUSD=X_ret_21d"].dropna(), cad_21d)
        components.append(cad_pct)

    return float(np.nanmean(components) * 100) if components else 50.0


def _composite_score(
    bank: str,
    features: pd.DataFrame,
    macro: pd.DataFrame,
    regime_score: float,
) -> tuple[float, dict[str, float]]:
    """
    Weighted composite attractiveness score (0-100, HIGHER = more attractive long).
    Weights shift by regime: in Low risk, momentum matters more;
    in Elevated/High/Severe, stress protection (inverted stress) matters more.
    """
    mom = _momentum_score(bank, features)
    stress = _stress_score(bank, features)
    mr = _mean_reversion_score(bank, features)
    macro_tw = _macro_tailwind_score(features, macro)

    if regime_score < 30:
        w = dict(momentum=0.35, inv_stress=0.25, mean_reversion=0.15, macro=0.25)
    elif regime_score < 60:
        w = dict(momentum=0.25, inv_stress=0.35, mean_reversion=0.15, macro=0.25)
    elif regime_score < 80:
        w = dict(momentum=0.15, inv_stress=0.45, mean_reversion=0.20, macro=0.20)
    else:
        w = dict(momentum=0.10, inv_stress=0.55, mean_reversion=0.20, macro=0.15)

    composite = (
        w["momentum"] * mom
        + w["inv_stress"] * (100 - stress)
        + w["mean_reversion"] * mr
        + w["macro"] * macro_tw
    )

    components = {
        "Momentum": mom,
        "Stress (inverted)": 100 - stress,
        "Mean Reversion": mr,
        "Macro Tailwind": macro_tw,
        "Composite": composite,
    }
    return float(np.clip(composite, 0, 100)), components


def _conviction_level(score: float, percentile: float) -> int:
    """
    1–5 conviction scale.
    5 = highest conviction (extreme score deviating far from neutral).
    """
    deviation = abs(score - 50)
    if deviation >= 30 and percentile > 0.85:
        return 5
    if deviation >= 22 and percentile > 0.70:
        return 4
    if deviation >= 14:
        return 3
    if deviation >= 7:
        return 2
    return 1


def _signal_label(score: float, regime_score: float) -> str:
    """Map composite score to BUY / HOLD / REDUCE with regime overlay."""
    if regime_score >= 80:
        if score >= 68:
            return "HOLD"
        return "REDUCE"
    if score >= 65:
        return "BUY"
    if score >= 40:
        return "HOLD"
    return "REDUCE"


def _signal_rationale(
    bank: str,
    signal: str,
    components: dict[str, float],
    features: pd.DataFrame,
    regime_score: float,
) -> str:
    """Produce a 1-sentence human rationale for the signal."""
    top_factor = max(
        [("Momentum", components["Momentum"]),
         ("Stress protection", components["Stress (inverted)"]),
         ("Mean Reversion", components["Mean Reversion"]),
         ("Macro Tailwind", components["Macro Tailwind"])],
        key=lambda x: abs(x[1] - 50)
    )[0]

    node_stress = _stress_score(bank, features)
    vol = latest(features, f"{bank}_vol_21d", np.nan)

    if signal == "BUY":
        return (
            f"{top_factor} is the primary driver at {components[top_factor.replace('Stress protection', 'Stress (inverted)')]:.0f}/100; "
            f"node stress {node_stress:.0f}/100 and vol {vol:.1%} are manageable in the current regime."
        )
    if signal == "REDUCE":
        if regime_score >= 60:
            return (
                f"High regime score ({regime_score:.0f}/100) warrants reducing concentration; "
                f"node stress {node_stress:.0f}/100 and {top_factor} are the primary concerns."
            )
        return (
            f"Node stress {node_stress:.0f}/100 and below-median {top_factor} "
            f"({components[top_factor.replace('Stress protection', 'Stress (inverted)')]:.0f}/100) argue for trimming."
        )
    return (
        f"Mixed signals: {top_factor} at {components[top_factor.replace('Stress protection', 'Stress (inverted)')]:.0f}/100 "
        f"with node stress {node_stress:.0f}/100. No compelling edge to add or reduce."
    )


def _target_weight(
    bank: str,
    composite_score: float,
    regime_score: float,
    all_scores: dict[str, float],
) -> float:
    """
    Tilt from equal-weight baseline proportional to cross-sectional rank of composite score.
    Total bank budget shrinks as regime worsens.
    """
    if regime_score < 30:
        bank_budget = 0.75
        tilt_strength = 0.20
    elif regime_score < 60:
        bank_budget = 0.65
        tilt_strength = 0.15
    elif regime_score < 80:
        bank_budget = 0.50
        tilt_strength = 0.10
    else:
        bank_budget = 0.35
        tilt_strength = 0.06

    scores_series = pd.Series(all_scores)
    rank = float((scores_series <= composite_score).mean())

    per_bank_base = bank_budget / len(BANKS)
    tilt = tilt_strength * (rank - 0.5)
    raw = per_bank_base + tilt

    return float(np.clip(raw, 0.02, 0.35))


def compute_bank_signals(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    macro: pd.DataFrame,
) -> pd.DataFrame:
    """
    Core function: returns one row per bank with signals, scores, targets, and rationale.

    Columns returned
    ----------------
    Bank, Name, Signal, Conviction, Composite, Momentum, StressInverted,
    MeanReversion, MacroTailwind, NodeStress, 21DReturn, 21DVol, 63DDrawdown,
    BetaXFN, TargetWeight, NeutralWeight, WeightDelta, Rationale, EconomicContext
    """
    regime_score = latest(features, "contagion_risk_score", 50.0)

    all_composites: dict[str, float] = {}
    bank_data: list[dict] = []

    for bank in BANKS:
        composite, components = _composite_score(bank, features, macro, regime_score)
        all_composites[bank] = composite
        bank_data.append((bank, composite, components))

    rows = []
    for bank, composite, components in bank_data:
        target_w = _target_weight(bank, composite, regime_score, all_composites)
        composites_series = pd.Series(all_composites)
        comp_pctrank = float((composites_series <= composite).mean())
        conviction = _conviction_level(composite, comp_pctrank)
        signal = _signal_label(composite, regime_score)
        rationale = _signal_rationale(bank, signal, components, features, regime_score)

        rows.append({
            "Bank": bank,
            "Name": BANK_NAMES[bank],
            "Signal": signal,
            "Conviction": conviction,
            "Composite Score": composite,
            "Momentum": components["Momentum"],
            "Stress (inverted)": components["Stress (inverted)"],
            "Mean Reversion": components["Mean Reversion"],
            "Macro Tailwind": components["Macro Tailwind"],
            "Node Stress": _stress_score(bank, features),
            "21D Return": latest(features, f"{bank}_ret_21d", np.nan),
            "21D Vol": latest(features, f"{bank}_vol_21d", np.nan),
            "63D Drawdown": latest(features, f"{bank}_drawdown_63d", np.nan),
            "Beta to XFN": latest(features, f"{bank}_beta_xfn_63d", np.nan),
            "Target Weight": target_w,
            "Neutral Weight": _NEUTRAL_WEIGHT,
            "Weight Delta": target_w - _NEUTRAL_WEIGHT,
            "Rationale": rationale,
            "Economic Context": BANK_CONTEXT[bank],
        })

    return pd.DataFrame(rows).sort_values("Composite Score", ascending=False).reset_index(drop=True)


def compute_portfolio_recommendations(
    signals: pd.DataFrame,
    regime_score: float,
    current_weights: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Generate an explicit rebalance table.

    Returns columns: Bank, Current Weight, Target Weight, Action, Notional % Change,
    Priority, Action Reason.
    """
    target = signals.set_index("Bank")["Target Weight"]
    signal_map = signals.set_index("Bank")["Signal"]
    rationale_map = signals.set_index("Bank")["Rationale"]

    if current_weights is None:
        current = pd.Series(_NEUTRAL_WEIGHT, index=BANKS)
    else:
        current = current_weights.reindex(BANKS).fillna(_NEUTRAL_WEIGHT)

    rows = []
    for bank in BANKS:
        cur = float(current.get(bank, _NEUTRAL_WEIGHT))
        tgt = float(target.get(bank, _NEUTRAL_WEIGHT))
        delta = tgt - cur
        sig = signal_map.get(bank, "HOLD")

        if abs(delta) >= 0.03:
            action = "BUY" if delta > 0 else "SELL"
            priority = "High" if abs(delta) >= 0.06 else "Medium"
        else:
            action = "HOLD"
            priority = "Low"

        if action != "HOLD":
            reason = rationale_map.get(bank, "")
        else:
            reason = "Within tolerance — no rebalance required."

        rows.append({
            "Bank": bank,
            "Current Weight": cur,
            "Target Weight": tgt,
            "Delta": delta,
            "Action": action,
            "Priority": priority,
            "Reason": reason,
        })

    return pd.DataFrame(rows).sort_values("Delta", ascending=False).reset_index(drop=True)


def compute_sensitivity_analysis(
    features: pd.DataFrame,
    macro: pd.DataFrame,
    score_shocks: list[float] | None = None,
) -> pd.DataFrame:
    """
    Show how target weights change across hypothetical contagion score scenarios.

    Parameters
    ----------
    score_shocks : list of float
        Absolute contagion score values to test. Default: current ± 10, ± 20.
    """
    current_score = latest(features, "contagion_risk_score", 50.0)
    if score_shocks is None:
        score_shocks = [
            max(0, current_score - 20),
            max(0, current_score - 10),
            current_score,
            min(100, current_score + 10),
            min(100, current_score + 20),
        ]

    rows = []
    for shock in score_shocks:
        all_composites: dict[str, float] = {}
        for bank in BANKS:
            composite, _ = _composite_score(bank, features, macro, shock)
            all_composites[bank] = composite
        for bank in BANKS:
            tgt = _target_weight(bank, all_composites[bank], shock, all_composites)
            regime_label = risk_regime(shock)["label"]
            rows.append({
                "Score Scenario": f"{shock:.0f}/100  ({regime_label})",
                "Bank": bank,
                "Target Weight": tgt,
                "Regime": regime_label,
            })

    df = pd.DataFrame(rows)
    pivot = df.pivot(index="Score Scenario", columns="Bank", values="Target Weight")
    return pivot


def compute_market_positioning(
    features: pd.DataFrame,
    macro: pd.DataFrame,
    regime_score: float,
) -> dict:
    """
    Return a summary dict with macro-level positioning guidance.

    Keys: regime, total_bank_budget, cash_guidance, sector_bias, key_risks, opportunity.
    """
    regime = risk_regime(regime_score)

    vix = latest(features, "VIX_level", np.nan)
    slope = latest(features, "slope_10y_2y", np.nan)
    avg_corr = latest(features, "avg_pairwise_corr_63d", np.nan)
    oil_21d = latest(features, "CL=F_ret_21d", np.nan)
    cad_21d = latest(features, "CADUSD=X_ret_21d", np.nan)

    if regime_score < 30:
        total_bank_budget = "65–75%"
        cash_guidance = "5–10% minimum. No need to be heavily defensive."
        sector_bias = "Broad bank exposure acceptable. Slight tilt to momentum leaders."
        opportunity = "Normal diversification is working. Focus on bank-specific fundamental drivers."
    elif regime_score < 60:
        total_bank_budget = "50–65%"
        cash_guidance = "10–15%. Build defensive buffer for potential escalation."
        sector_bias = "Avoid highest-stress names. Prefer banks with lower volatility and drawdown."
        opportunity = "Active rotation from stress laggards to relative quality leaders possible."
    elif regime_score < 80:
        total_bank_budget = "35–50%"
        cash_guidance = "20–35%. Defensive positioning is justified by elevated tail risk."
        sector_bias = "Underweight banks broadly. Long XFN retains financial-sector exposure; reducing existing XFN lowers that exposure. Focus on lowest-node-stress banks."
        opportunity = "Sector ETF (XFN) as partial replacement reduces idiosyncratic concentration while keeping sector exposure."
    else:
        total_bank_budget = "20–35%"
        cash_guidance = "35–50%+. Multiple stress channels are active simultaneously."
        sector_bias = "Minimize bank concentration. Consider reducing existing financial-sector exposure within a long-only mandate."
        opportunity = "The benefit of a more defensive allocation remains mandate-dependent and unvalidated."

    key_risks = []
    if pd.notna(avg_corr) and avg_corr > 0.75:
        key_risks.append(f"High cross-bank correlation ({avg_corr:.2f}) — diversification within banks is failing.")
    if pd.notna(vix) and vix > 25:
        key_risks.append(f"Elevated VIX ({vix:.1f}) — global risk-off sentiment is an active headwind.")
    if pd.notna(slope) and slope < 0.5:
        key_risks.append(f"Flat/inverted yield curve (slope {slope:.2f}%) — NIM pressure and recession signal.")
    if pd.notna(oil_21d) and oil_21d < -0.05:
        key_risks.append(f"Oil weakness ({oil_21d:.1%} 21D) — pressures Canadian credit and macro sentiment.")
    if pd.notna(cad_21d) and cad_21d < -0.02:
        key_risks.append(f"CAD weakness ({cad_21d:.1%} 21D) — potential capital outflow or growth concern.")
    if not key_risks:
        key_risks = ["No acute macro tail risks identified at current readings."]

    return {
        "regime": regime["label"],
        "regime_tone": regime["tone"],
        "regime_summary": regime["summary"],
        "total_bank_budget": total_bank_budget,
        "cash_guidance": cash_guidance,
        "sector_bias": sector_bias,
        "key_risks": key_risks,
        "opportunity": opportunity,
        "contagion_score": regime_score,
    }


def signal_display_table(signals: pd.DataFrame) -> pd.DataFrame:
    """Formatted version of signals for display in the dashboard."""
    display = signals[[
        "Bank", "Name", "Signal", "Conviction",
        "Composite Score", "Node Stress",
        "21D Return", "21D Vol", "63D Drawdown",
        "Target Weight", "Weight Delta",
        "Rationale",
    ]].copy()

    display["Signal"] = display["Signal"].map({
        "BUY": "▲ BUY",
        "HOLD": "◆ HOLD",
        "REDUCE": "▼ REDUCE",
    })
    display["Conviction"] = display["Conviction"].map(lambda x: "★" * x + "☆" * (5 - x))
    display["Composite Score"] = display["Composite Score"].map(lambda x: f"{x:.1f}/100")
    display["Node Stress"] = display["Node Stress"].map(lambda x: f"{x:.1f}/100")
    display["21D Return"] = display["21D Return"].map(lambda x: f"{x:+.1%}" if pd.notna(x) else "N/A")
    display["21D Vol"] = display["21D Vol"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
    display["63D Drawdown"] = display["63D Drawdown"].map(lambda x: f"{x:+.1%}" if pd.notna(x) else "N/A")
    display["Target Weight"] = display["Target Weight"].map(lambda x: f"{x:.1%}")
    display["Weight Delta"] = display["Weight Delta"].map(lambda x: f"{x:+.1%}")

    return display


def rebalance_display_table(recs: pd.DataFrame) -> pd.DataFrame:
    """Formatted rebalance recommendations for display."""
    display = recs.copy()
    for col in ["Current Weight", "Target Weight", "Delta"]:
        display[col] = display[col].map(lambda x: f"{x:+.1%}" if "Delta" in col else f"{x:.1%}")
    return display
