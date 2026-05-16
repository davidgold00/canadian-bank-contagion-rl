import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.dashboard.insight_utils import (
    DATASET_EXPLANATIONS,
    csv_inventory,
    latest_valid_date,
    normalize_to_100,
    read_csv_date,
    repo_root,
)
from src.dashboard.ui_components import analyst_header, apply_dashboard_style, insight_card


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def chart_for_csv(selected: str, df: pd.DataFrame) -> go.Figure:
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No rows available", height=420)
        return fig

    numeric_cols = _numeric_columns(df)

    if "holding weight" in df.columns:
        chart_df = df.copy()
        fig = go.Figure(
            go.Bar(
                x=chart_df.get("holding ticker", chart_df.index),
                y=chart_df["holding weight"],
                text=[f"{x:.1%}" for x in chart_df["holding weight"]],
                textposition="auto",
            )
        )
        fig.update_layout(title="ETF Holding Weight by Bank", yaxis_tickformat=".0%", height=430)
        return fig

    if "cds_5y_bps" in df.columns:
        fig = go.Figure(go.Bar(x=df.get("bank", df.index), y=df["cds_5y_bps"], text=df["cds_5y_bps"], textposition="auto"))
        fig.update_layout(title="5-Year CDS / Credit Spread Proxy", yaxis_title="Basis points", height=430)
        return fig

    if "contagion_risk_score" in df.columns and isinstance(df.index, pd.DatetimeIndex):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["contagion_risk_score"], mode="lines", name="Contagion score"))
        fig.update_layout(title="Contagion Risk Score from Model Dataset", yaxis_title="0-100", height=430)
        return fig

    if isinstance(df.index, pd.DatetimeIndex) and numeric_cols:
        sample = df[numeric_cols].tail(252).dropna(how="all")
        if len(numeric_cols) > 8:
            numeric_cols = numeric_cols[:8]
            sample = sample[numeric_cols]
        normalized = normalize_to_100(sample)
        fig = go.Figure()
        for col in normalized.columns:
            fig.add_trace(go.Scatter(x=normalized.index, y=normalized[col], mode="lines", name=col))
        fig.update_layout(title="Recent Numeric Series Indexed to 100", yaxis_title="Index", height=430)
        return fig

    if numeric_cols:
        values = df[numeric_cols].iloc[0].dropna()
        fig = go.Figure(go.Bar(x=values.index, y=values.values, text=[f"{x:.2f}" for x in values.values], textposition="auto"))
        fig.update_layout(title="Numeric Values in First Row", height=430)
        return fig

    fig = go.Figure()
    fig.update_layout(title="No numeric chart available; use the table preview.", height=430)
    return fig


st.set_page_config(page_title="Data Catalog", layout="wide")
apply_dashboard_style()

inventory = csv_inventory()

analyst_header(
    "Data Catalog",
    "Every CSV explained, profiled, and connected to an analytical chart.",
    date_text=latest_valid_date(read_csv_date(repo_root() / "data" / "processed" / "model_dataset.csv"))
    if (repo_root() / "data" / "processed" / "model_dataset.csv").exists()
    else "sample data",
    source_text="Tracked templates plus generated raw/processed files when present",
)

st.markdown(
    """
    This page exists so the project does not feel like a folder of mysterious spreadsheets.
    Each CSV has a business role, a plain-English explanation, and a chart that shows what the file contributes.
    """
)

m1, m2, m3 = st.columns(3)
m1.metric("CSV Files Found", f"{len(inventory):,}")
m2.metric("Total Rows", f"{int(inventory['Rows'].fillna(0).sum()):,}")
m3.metric("Explained Files", f"{inventory['Explanation'].notna().sum():,}")

tab1, tab2, tab3 = st.tabs(["CSV Inventory", "Chart Explorer", "How Data Flows"])

with tab1:
    st.subheader("Inventory and Business Meaning")
    st.dataframe(inventory, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("CSV Chart Explorer")
    selected = st.selectbox("CSV", inventory["CSV"].tolist())
    path = repo_root() / selected
    meta = DATASET_EXPLANATIONS.get(selected, {})
    df = read_csv_date(path)

    insight_card(
        meta.get("role", "Project CSV"),
        meta.get("plain_english", "Supporting data file used by the project."),
    )

    c1, c2 = st.columns([0.65, 0.35])
    with c1:
        fig = chart_for_csv(selected, df)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("#### File Profile")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Field": "Rows", "Value": len(df)},
                    {"Field": "Columns", "Value": len(df.columns)},
                    {"Field": "Date Range", "Value": latest_valid_date(df) if isinstance(df.index, pd.DatetimeIndex) else "No date index"},
                    {"Field": "Chart Purpose", "Value": meta.get("chart", "Shows the numeric content of this CSV.")},
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("#### Preview")
    st.dataframe(df.head(25), use_container_width=True)

with tab3:
    st.subheader("From CSV to Insight")
    st.markdown(
        """
        1. `data/raw/market_prices.csv` and `data/raw/boc_yields.csv` are downloaded from live sources when network access is available.
        2. `data/processed/prices.csv` stores the cleaned aligned price panel used by dashboards and backtests.
        3. `data/processed/model_dataset.csv` combines price, macro, volatility, drawdown, beta, correlation, and contagion-score features.
        4. `data/templates/*.csv` files are analyst-controlled scenario inputs for housing stress, CDS spreads, and ETF ownership overlap.
        5. The dashboard pages translate those tables into decisions: risk regime, contagion channels, scenario losses, and allocation posture.
        """
    )
