from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


def format_percent(value: float | int | None, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}%}"


def format_currency(value: float | int | None, digits: int = 0) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"${float(value):,.{digits}f}"


def metric_card(label: str, value: str, help_text: str | None = None, delta: str | None = None) -> None:
    st.metric(label, value, delta=delta, help=help_text)


def section_header(title: str, why_it_matters: str | None = None) -> None:
    st.subheader(title)
    if why_it_matters:
        st.caption(why_it_matters)


def explanation_box(title: str, body: str, tone: str = "info") -> None:
    styles = {
        "info": st.info,
        "success": st.success,
        "warning": st.warning,
        "danger": st.error,
    }
    styles.get(tone, st.info)(f"**{title}**\n\n{body}")


def dataframe_with_context(df: pd.DataFrame, context: str, hide_index: bool = True) -> None:
    st.caption(context)
    st.dataframe(df, use_container_width=True, hide_index=hide_index)


def disclaimer_box() -> None:
    st.warning(
        "Simulated paper portfolio only. No real trades are placed. This is not investment advice, "
        "not a trading bot, and not a forecast of future returns. Data may be proxied or synthetic, "
        "and transaction-cost/liquidity assumptions are simplified."
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@st.cache_data
def load_processed_dataset() -> pd.DataFrame:
    path = _repo_root() / "data" / "processed" / "model_dataset.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    date_col = "date" if "date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    return df.rename(columns={date_col: "date"}).set_index("date").sort_index()


@st.cache_data
def load_price_data() -> pd.DataFrame:
    root = _repo_root()
    for rel in ["data/processed/prices.csv", "data/raw/market_prices.csv", "data/sample/market_prices.csv"]:
        path = root / rel
        if path.exists():
            df = pd.read_csv(path)
            date_col = "date" if "date" in df.columns else "Date" if "Date" in df.columns else df.columns[0]
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            return df.rename(columns={date_col: "date"}).set_index("date").sort_index()
    return pd.DataFrame()


def calculate_strategy_metrics(values: pd.Series) -> dict[str, float]:
    from src.portfolio.performance_metrics import performance_summary

    returns = values.pct_change().fillna(0.0)
    return performance_summary(values, returns)
