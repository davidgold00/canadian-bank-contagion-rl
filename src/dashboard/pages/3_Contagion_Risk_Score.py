from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="Contagion Risk Score", layout="wide")

BANKS = ["RY.TO", "TD.TO", "BMO.TO", "BNS.TO", "CM.TO", "NA.TO"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@st.cache_data
def load_dataset() -> pd.DataFrame:
    root = repo_root()
    path = root / "data" / "processed" / "model_dataset.csv"

    if not path.exists():
        st.error("model_dataset.csv not found. Run scripts/build_features.py first.")
        st.stop()

    df = pd.read_csv(path)
    date_col = "date" if "date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    return df.rename(columns={date_col: "date"}).set_index("date").sort_index()


def percentile_rank(series: pd.Series, value: float) -> float:
    clean = series.dropna()
    if len(clean) == 0:
        return 0.5
    return float((clean <= value).mean())


def risk_band(score: float) -> tuple[str, str]:
    if score < 30:
        return "Low", "Normal market conditions. Bank stress is relatively contained."
    if score < 60:
        return "Moderate", "Some stress indicators are active. Monitor correlations and drawdowns."
    if score < 80:
        return "High", "Systemic pressure is elevated. Diversification may be weakening."
    return "Severe", "Multiple stress channels are elevated. Defensive positioning may be warranted."


def zscore_percentile(s: pd.Series) -> pd.Series:
    s = s.replace([np.inf, -np.inf], np.nan)
    return 100 * s.rank(pct=True)


def find_col(df: pd.DataFrame, options: list[str]) -> str | None:
    for col in options:
        if col in df.columns:
            return col
    return None


def build_risk_components(df: pd.DataFrame) -> pd.DataFrame:
    components = pd.DataFrame(index=df.index)

    vol_col = find_col(df, ["avg_bank_vol_21d"])
    corr_col = find_col(df, ["avg_pairwise_corr_63d"])
    vix_col = find_col(df, ["VIX_level"])
    vix_change_col = find_col(df, ["VIX_chg_5d"])
    curve_col = find_col(df, ["slope_10y_2y", "10y_2y_slope"])
    oil_col = find_col(df, ["CL=F_ret_5d", "CL=F_ret_21d"])
    xfn_col = find_col(df, ["XFN.TO_drawdown_63d", "XFN.TO_ret_21d"])

    if vol_col:
        components["Bank volatility stress"] = zscore_percentile(df[vol_col])

    if corr_col:
        components["Correlation concentration"] = zscore_percentile(df[corr_col])

    if vix_col:
        components["Global volatility proxy"] = zscore_percentile(df[vix_col])

    if vix_change_col:
        components["Volatility spike"] = zscore_percentile(df[vix_change_col])

    if curve_col:
        components["Yield curve stress"] = zscore_percentile(-df[curve_col])

    if oil_col:
        components["Oil shock proxy"] = zscore_percentile(-df[oil_col])

    if xfn_col:
        components["Financials drawdown pressure"] = zscore_percentile(-df[xfn_col])

    if "contagion_risk_score" in df.columns:
        components["Composite model score"] = df["contagion_risk_score"]

    return components.ffill().fillna(50).clip(0, 100)


def build_bank_stress(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for bank in BANKS:
        vol_col = f"{bank}_vol_21d"
        dd_col = f"{bank}_drawdown_63d"
        beta_col = f"{bank}_beta_xfn_63d"
        ret_col = f"{bank}_ret_5d"

        vol = float(df[vol_col].iloc[-1]) if vol_col in df else np.nan
        dd = float(df[dd_col].iloc[-1]) if dd_col in df else np.nan
        beta = float(df[beta_col].iloc[-1]) if beta_col in df else np.nan
        ret = float(df[ret_col].iloc[-1]) if ret_col in df else np.nan

        score_parts = []
        if vol_col in df:
            score_parts.append(percentile_rank(df[vol_col], vol) * 100)
        if dd_col in df:
            score_parts.append(percentile_rank(-df[dd_col], -dd) * 100)
        if beta_col in df:
            score_parts.append(percentile_rank(df[beta_col], beta) * 100)

        node_score = float(np.mean(score_parts)) if score_parts else 50

        rows.append(
            {
                "Bank": bank,
                "Node Stress Score": round(node_score, 1),
                "21D Volatility": round(vol, 3) if not np.isnan(vol) else None,
                "63D Drawdown": round(dd, 3) if not np.isnan(dd) else None,
                "Beta to XFN": round(beta, 2) if not np.isnan(beta) else None,
                "5D Return": round(ret, 3) if not np.isnan(ret) else None,
                "Interpretation": "Elevated stress"
                if node_score >= 70
                else "Moderate stress"
                if node_score >= 40
                else "Contained stress",
            }
        )

    return pd.DataFrame(rows).sort_values("Node Stress Score", ascending=False)


def gauge(score: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "/100"},
            title={"text": "Composite Contagion Risk"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "black"},
                "steps": [
                    {"range": [0, 30], "color": "#d9f0d3"},
                    {"range": [30, 60], "color": "#ffffbf"},
                    {"range": [60, 80], "color": "#fdae61"},
                    {"range": [80, 100], "color": "#d7191c"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 80,
                },
            },
        )
    )
    fig.update_layout(height=330, margin=dict(l=20, r=20, t=60, b=20))
    return fig


st.title("Contagion Risk Score")

st.info(
    """
    This page turns many noisy stress signals into one interpretable risk score.
    The score is not a price forecast. It is a systemic-risk monitor: high values mean
    Canadian bank stress, volatility, correlation, and macro-risk features are clustering together.
    """
)

df = load_dataset()
components = build_risk_components(df)

score_series = df["contagion_risk_score"] if "contagion_risk_score" in df.columns else components.mean(axis=1)
current_score = float(score_series.iloc[-1])
band, band_text = risk_band(current_score)
pct = percentile_rank(score_series, current_score) * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest Risk Score", f"{current_score:.1f}/100", help="Composite systemic stress score.")
c2.metric("Risk Band", band, help=band_text)
c3.metric("Historical Percentile", f"{pct:.0f}%", help="How extreme today's score is versus its own history.")
c4.metric("Latest Date", str(score_series.index[-1].date()))

left, right = st.columns([0.42, 0.58])

with left:
    st.plotly_chart(gauge(current_score), use_container_width=True)
    st.warning(f"Current interpretation: **{band} risk** — {band_text}")

with right:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=score_series.index,
            y=score_series,
            mode="lines",
            name="Contagion risk score",
        )
    )
    fig.add_hline(y=30, line_dash="dot", annotation_text="Low/Moderate")
    fig.add_hline(y=60, line_dash="dot", annotation_text="Moderate/High")
    fig.add_hline(y=80, line_dash="dot", annotation_text="Severe threshold")
    fig.update_layout(
        title="Historical Contagion Risk Score",
        yaxis_title="Risk score",
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Risk Drivers",
        "Bank Stress Table",
        "Stress Regime Map",
        "How to Use This",
    ]
)

with tab1:
    st.subheader("Current Risk Driver Decomposition")
    st.markdown(
        """
        These components explain what is pushing the score higher or lower.
        A useful systemic-risk signal usually comes from several components rising together,
        not from one isolated spike.
        """
    )

    latest_components = components.iloc[-1].sort_values(ascending=True)

    fig = go.Figure(
        go.Bar(
            x=latest_components.values,
            y=latest_components.index,
            orientation="h",
            text=[f"{v:.1f}" for v in latest_components.values],
            textposition="auto",
        )
    )
    fig.update_layout(
        title="Current Component Scores",
        xaxis_title="Stress contribution, 0–100",
        height=520,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        latest_components.rename("Current Component Score").reset_index().rename(columns={"index": "Component"}),
        use_container_width=True,
        hide_index=True,
    )

with tab2:
    st.subheader("Bank-Level Node Stress")
    st.markdown(
        """
        This table identifies which individual banks are contributing most to systemic risk.
        It combines volatility, drawdown pressure, and sensitivity to XFN where available.
        """
    )

    bank_stress = build_bank_stress(df)
    st.dataframe(bank_stress, use_container_width=True, hide_index=True)

    leader = bank_stress.iloc[0]
    st.error(
        f"""
        Highest current node stress: **{leader['Bank']}** with score **{leader['Node Stress Score']}**.
        This does not mean the bank is impaired. It means its recent market behavior is more stressed
        relative to the other Big Six names.
        """
    )

with tab3:
    st.subheader("Risk Component Heatmap")
    st.markdown(
        """
        This heatmap shows whether stress is broad-based or isolated.
        Broad-based horizontal bands indicate systemic pressure.
        """
    )

    recent_components = components.tail(126).T

    fig = go.Figure(
        data=go.Heatmap(
            z=recent_components.values,
            x=recent_components.columns,
            y=recent_components.index,
            colorscale="RdYlGn_r",
            zmin=0,
            zmax=100,
            colorbar=dict(title="Stress"),
        )
    )
    fig.update_layout(height=560, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("How to Read This Page")
    st.markdown(
        """
        Use this page like a risk officer, not like a day trader.

        ### What a high score means

        A high score means market conditions are becoming less forgiving:

        - bank volatility is rising,
        - correlations are increasing,
        - drawdowns are deepening,
        - macro stress proxies are worsening,
        - financial-sector risk is clustering.

        ### What it does not mean

        It does **not** mean a crash is guaranteed.
        It does **not** mean any bank is insolvent.
        It does **not** provide investment advice.

        ### Best use

        The score is useful as an input to:

        - stress testing,
        - portfolio de-risking rules,
        - risk dashboards,
        - model monitoring,
        - reinforcement learning state design,
        - interview discussion of systemic risk.
        """
    )

    st.markdown("### Risk bands")
    st.dataframe(
        pd.DataFrame(
            [
                {"Band": "0–30", "Label": "Low", "Interpretation": "Normal conditions"},
                {"Band": "30–60", "Label": "Moderate", "Interpretation": "Some risk signals active"},
                {"Band": "60–80", "Label": "High", "Interpretation": "Systemic pressure elevated"},
                {"Band": "80–100", "Label": "Severe", "Interpretation": "Multiple stress channels flashing"},
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )