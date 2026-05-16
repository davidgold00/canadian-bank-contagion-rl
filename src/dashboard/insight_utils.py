from pathlib import Path

import numpy as np
import pandas as pd


BANKS = ["RY.TO", "TD.TO", "BMO.TO", "BNS.TO", "CM.TO", "NA.TO"]

BANK_NAMES = {
    "RY.TO": "Royal Bank of Canada",
    "TD.TO": "Toronto-Dominion Bank",
    "BMO.TO": "Bank of Montreal",
    "BNS.TO": "Scotiabank",
    "CM.TO": "CIBC",
    "NA.TO": "National Bank",
}

BANK_CONTEXT = {
    "RY.TO": "largest diversified Canadian bank; often a proxy for system-wide confidence",
    "TD.TO": "large Canadian and US retail footprint; sensitive to North American credit cycles",
    "BMO.TO": "Canada/US bank with capital-markets and commercial-credit exposure",
    "BNS.TO": "more international exposure than peers; useful for global risk read-through",
    "CM.TO": "domestic consumer and mortgage sensitivity tends to matter in housing stress",
    "NA.TO": "Quebec-heavy franchise; smaller but can show regional sentiment shifts",
}

DATASET_EXPLANATIONS = {
    "data/sample/market_prices.csv": {
        "role": "Synthetic fallback market prices",
        "plain_english": "Lets the dashboard run when live downloads are unavailable. It should not be used as real market history.",
        "chart": "Price trend and return distribution by ticker.",
    },
    "data/sample/macro.csv": {
        "role": "Synthetic fallback macro and yield data",
        "plain_english": "Provides policy-rate and yield-curve shaped fields when live Bank of Canada data is unavailable.",
        "chart": "Yield curve slope and policy-rate trend.",
    },
    "data/templates/housing_stress_template.csv": {
        "role": "Manual housing stress input template",
        "plain_english": "A place to enter mortgage arrears, house-price, unemployment, and household-debt assumptions.",
        "chart": "Housing stress variables, normalized for comparability.",
    },
    "data/templates/cds_template.csv": {
        "role": "Manual credit-spread input template",
        "plain_english": "A template for 5-year bank CDS or credit-spread proxies. Wider spreads imply the market is demanding more compensation for bank credit risk.",
        "chart": "Credit-spread level by bank.",
    },
    "data/templates/etf_holdings_template.csv": {
        "role": "ETF ownership overlap input template",
        "plain_english": "Shows how much each bank is held in an ETF such as XFN. Shared ETF ownership can make selling pressure more synchronized.",
        "chart": "Holding weight by bank.",
    },
    "data/raw/market_prices.csv": {
        "role": "Live Yahoo Finance market prices",
        "plain_english": "The current market-data feed used for returns, volatility, drawdowns, correlations, and the dashboard's latest charts.",
        "chart": "Recent bank and macro-market price trends.",
    },
    "data/raw/boc_yields.csv": {
        "role": "Live Bank of Canada rates and yields",
        "plain_english": "Policy-rate, 2-year, 5-year, and 10-year Canadian yield history. This explains the macro rate backdrop for banks.",
        "chart": "Yield curve slope and rate levels.",
    },
    "data/processed/prices.csv": {
        "role": "Clean aligned price panel",
        "plain_english": "The price table used by pages and backtests after date parsing and incomplete-row cleanup.",
        "chart": "Recent normalized price performance.",
    },
    "data/processed/model_dataset.csv": {
        "role": "Feature model table",
        "plain_english": "The main analytical dataset. It combines returns, volatility, drawdown, beta, correlations, yield features, and the contagion score.",
        "chart": "Contagion score and the most important driver time series.",
    },
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_csv_date(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df
    date_col = "date" if "date" in df.columns else "Date" if "Date" in df.columns else None
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.rename(columns={date_col: "date"}).set_index("date").sort_index()
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (TypeError, ValueError):
            pass
    return df


def load_features() -> pd.DataFrame:
    path = repo_root() / "data" / "processed" / "model_dataset.csv"
    if not path.exists():
        from src.data.build_dataset import build_dataset

        build_dataset(str(path))
    return read_csv_date(path)


def load_prices() -> pd.DataFrame:
    root = repo_root()
    for rel in ["data/processed/prices.csv", "data/raw/market_prices.csv", "data/sample/market_prices.csv"]:
        path = root / rel
        if path.exists():
            return read_csv_date(path).dropna(how="all")
    return pd.DataFrame()


def load_macro() -> pd.DataFrame:
    root = repo_root()
    for rel in ["data/raw/boc_yields.csv", "data/sample/macro.csv"]:
        path = root / rel
        if path.exists():
            macro = read_csv_date(path)
            if {"ca_10y", "ca_2y"}.issubset(macro.columns) and "slope_10y_2y" not in macro.columns:
                macro["slope_10y_2y"] = macro["ca_10y"] - macro["ca_2y"]
            return macro
    return pd.DataFrame()


def latest_valid_date(df: pd.DataFrame) -> str:
    if df.empty:
        return "N/A"
    return str(pd.to_datetime(df.dropna(how="all").index[-1]).date())


def latest(df: pd.DataFrame, col: str, default=np.nan) -> float:
    if col not in df:
        return default
    s = df[col].dropna()
    return float(s.iloc[-1]) if len(s) else default


def previous(df: pd.DataFrame, col: str, periods=21, default=np.nan) -> float:
    if col not in df:
        return default
    s = df[col].dropna()
    if len(s) <= periods:
        return default
    return float(s.iloc[-periods - 1])


def pct(x, digits=1) -> str:
    return f"{x:.{digits}%}" if pd.notna(x) else "N/A"


def num(x, digits=2) -> str:
    return f"{x:.{digits}f}" if pd.notna(x) else "N/A"


def signed_pct(x, digits=1) -> str:
    if pd.isna(x):
        return "N/A"
    return f"{x:+.{digits}%}"


def signed_num(x, digits=1) -> str:
    if pd.isna(x):
        return "N/A"
    return f"{x:+.{digits}f}"


def percentile_rank(series: pd.Series, value: float) -> float:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty or pd.isna(value):
        return np.nan
    return float((clean <= value).mean())


def risk_regime(score: float) -> dict:
    if score < 30:
        return {
            "label": "Low",
            "tone": "success",
            "summary": "Bank stress is contained and diversification is likely still useful.",
            "actions": [
                "Maintain ordinary monitoring cadence.",
                "Avoid adding defensive hedges solely because of the score.",
                "Use bank-level stress to spot early divergence.",
            ],
        }
    if score < 60:
        return {
            "label": "Moderate",
            "tone": "warning",
            "summary": "Some stress channels are active, but the system is not in a broad crisis regime.",
            "actions": [
                "Check whether volatility, correlation, and drawdown are rising together.",
                "Reduce new concentration in the highest-stress bank.",
                "Prepare scenario tests before risk becomes urgent.",
            ],
        }
    if score < 80:
        return {
            "label": "High",
            "tone": "danger",
            "summary": "Systemic pressure is elevated; bank stocks may behave more like one crowded trade.",
            "actions": [
                "Cut overweight bank exposure or require a stronger return hurdle.",
                "Prefer lower-stress banks and more liquid instruments.",
                "Run housing, liquidity, and global risk-off scenarios.",
            ],
        }
    return {
        "label": "Severe",
        "tone": "danger",
        "summary": "Multiple stress channels are flashing at once. Defensive posture should dominate.",
        "actions": [
            "Materially reduce unhedged bank exposure.",
            "Raise cash or hedges until the score and correlations normalize.",
            "Escalate monitoring of funding, credit-spread, and housing indicators.",
        ],
    }


def bank_stress_snapshot(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bank in BANKS:
        vol = latest(features, f"{bank}_vol_21d")
        dd = latest(features, f"{bank}_drawdown_63d")
        beta = latest(features, f"{bank}_beta_xfn_63d")
        ret_21d = latest(features, f"{bank}_ret_21d")
        ret_5d = latest(features, f"{bank}_ret_5d")

        components = []
        for col, sign in [
            (f"{bank}_vol_21d", 1),
            (f"{bank}_drawdown_63d", -1),
            (f"{bank}_beta_xfn_63d", 1),
        ]:
            if col in features:
                value = latest(features, col)
                components.append(100 * percentile_rank(sign * features[col], sign * value))
        node_stress = float(np.nanmean(components)) if components else 50
        if node_stress >= 70:
            readout = "Reduce/hedge first"
        elif node_stress >= 45:
            readout = "Monitor closely"
        else:
            readout = "Lower relative stress"

        rows.append(
            {
                "Bank": bank,
                "Name": BANK_NAMES[bank],
                "21D Return": ret_21d,
                "5D Return": ret_5d,
                "21D Volatility": vol,
                "63D Drawdown": dd,
                "Beta to XFN": beta,
                "Node Stress": node_stress,
                "Action Readout": readout,
                "Economic Lens": BANK_CONTEXT[bank],
            }
        )
    return pd.DataFrame(rows).sort_values("Node Stress", ascending=False)


def strongest_drivers(features: pd.DataFrame) -> pd.DataFrame:
    candidates = {
        "Bank volatility": ("avg_bank_vol_21d", "Higher volatility raises portfolio risk before losses are fully visible."),
        "Bank correlation": ("avg_pairwise_corr_63d", "High correlation means Big Six diversification is weakening."),
        "VIX level": ("VIX_level", "Global volatility can force broad de-risking in financial equities."),
        "VIX 5D change": ("VIX_chg_5d", "A quick volatility jump often precedes liquidity pressure."),
        "XFN drawdown": ("XFN.TO_drawdown_63d", "Sector ETF weakness shows bank stress is broad, not isolated."),
        "Oil 21D return": ("CL=F_ret_21d", "Oil weakness can pressure Canada-linked credit and macro sentiment."),
        "CAD 21D return": ("CADUSD=X_ret_21d", "Currency weakness can reflect capital-flow or growth concerns."),
        "2Y yield": ("ca_2y", "Short yields reflect policy-rate expectations and funding conditions."),
        "10Y-2Y slope": ("slope_10y_2y", "A flatter or inverted curve can pressure lending margins and recession sentiment."),
    }
    rows = []
    for label, (col, meaning) in candidates.items():
        if col not in features:
            continue
        value = latest(features, col)
        direction_value = -value if "drawdown" in col or col == "slope_10y_2y" else value
        direction_series = -features[col] if "drawdown" in col or col == "slope_10y_2y" else features[col]
        percentile = percentile_rank(direction_series, direction_value)
        rows.append(
            {
                "Driver": label,
                "Latest": value,
                "Stress Percentile": percentile,
                "Status": "Elevated" if percentile >= 0.75 else "Benign" if percentile <= 0.25 else "Normal",
                "Why it matters": meaning,
            }
        )
    return pd.DataFrame(rows).sort_values("Stress Percentile", ascending=False)


def csv_inventory() -> pd.DataFrame:
    root = repo_root()
    rows = []
    for path in sorted((root / "data").rglob("*.csv")):
        rel = path.relative_to(root).as_posix()
        try:
            df = pd.read_csv(path, nrows=5)
            full = read_csv_date(path)
            rows.append(
                {
                    "CSV": rel,
                    "Rows": len(full),
                    "Columns": len(full.columns),
                    "Date Range": _date_range(full),
                    "Role": DATASET_EXPLANATIONS.get(rel, {}).get("role", "Project CSV"),
                    "Explanation": DATASET_EXPLANATIONS.get(rel, {}).get("plain_english", "Supporting data file used by the project."),
                    "Example Columns": ", ".join(list(df.columns[:6])),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "CSV": rel,
                    "Rows": None,
                    "Columns": None,
                    "Date Range": "Unreadable",
                    "Role": "Project CSV",
                    "Explanation": f"Could not profile file: {exc}",
                    "Example Columns": "",
                }
            )
    return pd.DataFrame(rows)


def _date_range(df: pd.DataFrame) -> str:
    if isinstance(df.index, pd.DatetimeIndex) and len(df.index):
        return f"{df.index.min().date()} to {df.index.max().date()}"
    return "No date index"


def normalize_to_100(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number]).dropna(how="all")
    if numeric.empty:
        return numeric
    start = numeric.ffill().bfill().iloc[0].replace(0, np.nan)
    return numeric.divide(start).mul(100)
