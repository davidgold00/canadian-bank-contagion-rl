from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="Market Overview", layout="wide")

BANKS = ["RY.TO", "TD.TO", "BMO.TO", "BNS.TO", "CM.TO", "NA.TO"]


def root() -> Path:
    return Path(__file__).resolve().parents[3]


@st.cache_data
def load_data():
    processed = root() / "data" / "processed" / "model_dataset.csv"
    sample = root() / "data" / "sample" / "market_prices.csv"

    if processed.exists():
        df = pd.read_csv(processed)
        date_col = "date" if "date" in df.columns else df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col])
        return df.rename(columns={date_col: "date"}).set_index("date").sort_index(), "processed features"

    if sample.exists():
        df = pd.read_csv(sample)
        date_col = "date" if "date" in df.columns else "Date"
        df[date_col] = pd.to_datetime(df[date_col])
        px = df.rename(columns={date_col: "date"}).set_index("date").sort_index()
        out = pd.DataFrame(index=px.index)
        for col in px.columns:
            out[f"{col}_ret_1d"] = px[col].pct_change()
            out[f"{col}_ret_21d"] = px[col].pct_change(21)
            out[f"{col}_vol_21d"] = px[col].pct_change().rolling(21).std() * np.sqrt(252)
            out[f"{col}_drawdown_63d"] = px[col] / px[col].rolling(63, min_periods=10).max() - 1
        out["contagion_risk_score"] = 50
        return out.dropna(how="all"), "sample prices"

    st.error("No data found. Run scripts/download_data.py and scripts/build_features.py first.")
    st.stop()


def latest(df, col, default=np.nan):
    return float(df[col].dropna().iloc[-1]) if col in df and len(df[col].dropna()) else default


def risk_band(score):
    if score < 30:
        return "Low", "Stress is contained."
    if score < 60:
        return "Moderate", "Some risk signals are active."
    if score < 80:
        return "High", "Systemic pressure is elevated."
    return "Severe", "Multiple stress indicators are flashing."


def pct(x):
    return f"{x:.2%}" if pd.notna(x) else "N/A"


def num(x):
    return f"{x:.2f}" if pd.notna(x) else "N/A"


def plot_risk_score(df):
    score = df["contagion_risk_score"] if "contagion_risk_score" in df else pd.Series(50, index=df.index)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=score.index, y=score, mode="lines", name="Contagion Risk"))
    fig.add_hline(y=30, line_dash="dot", annotation_text="Low")
    fig.add_hline(y=60, line_dash="dot", annotation_text="High")
    fig.add_hline(y=80, line_dash="dot", annotation_text="Severe")
    fig.update_layout(
        title="Systemic Risk Monitor",
        height=380,
        yaxis_title="Risk score, 0–100",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def plot_bank_returns(df):
    rows = []
    for bank in BANKS:
        rows.append(
            {
                "Bank": bank,
                "1D": latest(df, f"{bank}_ret_1d"),
                "5D": latest(df, f"{bank}_ret_5d"),
                "21D": latest(df, f"{bank}_ret_21d"),
            }
        )
    table = pd.DataFrame(rows)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=table["Bank"], y=table["21D"], name="21D return"))
    fig.update_layout(
        title="Big Six 21-Day Return Snapshot",
        height=360,
        yaxis_tickformat=".1%",
        yaxis_title="Return",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig, table


def plot_vol_drawdown(df):
    rows = []
    for bank in BANKS:
        rows.append(
            {
                "Bank": bank,
                "21D Volatility": latest(df, f"{bank}_vol_21d"),
                "63D Drawdown": latest(df, f"{bank}_drawdown_63d"),
            }
        )
    table = pd.DataFrame(rows)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=table["21D Volatility"],
            y=table["63D Drawdown"],
            mode="markers+text",
            text=table["Bank"],
            textposition="top center",
            marker=dict(size=18),
        )
    )
    fig.update_layout(
        title="Risk Map: Volatility vs Drawdown",
        xaxis_title="21D annualized volatility",
        yaxis_title="63D drawdown",
        xaxis_tickformat=".1%",
        yaxis_tickformat=".1%",
        height=430,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig, table


df, source = load_data()

st.title("Market Overview")

score = latest(df, "contagion_risk_score", 50)
band, explanation = risk_band(score)

st.markdown(
    f"""
    ### Executive readout

    This page summarizes whether the Canadian bank complex is behaving normally or moving into a stress regime.

    **Current regime:** `{band}`  
    **Interpretation:** {explanation}

    The dashboard is using **{source}**.
    """
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Contagion Risk Score", f"{score:.1f}/100")
m2.metric("Risk Band", band)
m3.metric("Avg Bank 21D Vol", pct(latest(df, "avg_bank_vol_21d")))
m4.metric("Avg Bank Correlation", num(latest(df, "avg_pairwise_corr_63d")))

tab1, tab2, tab3, tab4 = st.tabs(
    ["Executive Dashboard", "Bank Risk Map", "Risk Drivers", "Data Dictionary"]
)

with tab1:
    left, right = st.columns(2)
    with left:
        st.plotly_chart(plot_risk_score(df), use_container_width=True)
    with right:
        fig, returns_table = plot_bank_returns(df)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("What this means")
    st.markdown(
        """
        - Rising **contagion risk** means stress signals are clustering.
        - Negative **21-day bank returns** indicate recent sector pressure.
        - High **correlation** means diversification across banks is less effective.
        - High **volatility** means portfolio risk is rising even if prices have not fully broken down.
        """
    )

with tab2:
    fig, risk_table = plot_vol_drawdown(df)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Bank-level interpretation")
    display = risk_table.copy()
    display["21D Volatility"] = display["21D Volatility"].map(pct)
    display["63D Drawdown"] = display["63D Drawdown"].map(pct)
    display["Interpretation"] = risk_table.apply(
        lambda r: "High-risk quadrant"
        if r["21D Volatility"] > risk_table["21D Volatility"].median()
        and r["63D Drawdown"] < risk_table["63D Drawdown"].median()
        else "Monitor"
        if r["21D Volatility"] > risk_table["21D Volatility"].median()
        else "Lower relative stress",
        axis=1,
    )
    st.dataframe(display, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Current risk drivers")

    driver_candidates = {
        "Bank volatility": "avg_bank_vol_21d",
        "Bank correlation": "avg_pairwise_corr_63d",
        "VIX level": "VIX_level",
        "VIX 5D change": "VIX_chg_5d",
        "XFN drawdown": "XFN.TO_drawdown_63d",
        "Oil 21D return": "CL=F_ret_21d",
        "CAD 21D return": "CADUSD=X_ret_21d",
        "Yield curve slope": "slope_10y_2y",
    }

    rows = []
    for label, col in driver_candidates.items():
        if col in df.columns:
            s = df[col].dropna()
            val = s.iloc[-1]
            percentile = (s <= val).mean()
            rows.append(
                {
                    "Driver": label,
                    "Latest Value": val,
                    "Historical Percentile": percentile,
                    "Meaning": "Elevated"
                    if percentile >= 0.75
                    else "Low"
                    if percentile <= 0.25
                    else "Normal range",
                }
            )

    drivers = pd.DataFrame(rows)

    fig = go.Figure(
        go.Bar(
            x=drivers["Historical Percentile"],
            y=drivers["Driver"],
            orientation="h",
            text=[f"{x:.0%}" for x in drivers["Historical Percentile"]],
            textposition="auto",
        )
    )
    fig.update_layout(
        title="Risk Driver Percentiles",
        xaxis_tickformat=".0%",
        height=430,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(drivers, use_container_width=True, hide_index=True)

with tab4:
    st.subheader("What the fields mean")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Feature": "ret_1d / ret_5d / ret_21d",
                    "Meaning": "Recent return over 1, 5, or 21 trading days.",
                    "Why it matters": "Captures short-term market pressure.",
                },
                {
                    "Feature": "vol_21d / vol_63d",
                    "Meaning": "Annualized rolling volatility.",
                    "Why it matters": "Higher volatility means risk is rising.",
                },
                {
                    "Feature": "drawdown_63d",
                    "Meaning": "Distance from recent 63-day high.",
                    "Why it matters": "Shows downside pressure.",
                },
                {
                    "Feature": "beta_xfn_63d",
                    "Meaning": "Sensitivity to Canadian financials ETF.",
                    "Why it matters": "Measures sector contagion exposure.",
                },
                {
                    "Feature": "avg_pairwise_corr_63d",
                    "Meaning": "Average bank-to-bank correlation.",
                    "Why it matters": "High correlation weakens diversification.",
                },
                {
                    "Feature": "contagion_risk_score",
                    "Meaning": "Composite systemic risk score from 0 to 100.",
                    "Why it matters": "Summarizes broad stress conditions.",
                },
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )