import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dashboard.insight_utils import (  # noqa: E402
    BANKS,
    BANK_NAMES,
    bank_stress_snapshot,
    csv_inventory,
    latest,
    latest_valid_date,
    load_features,
    load_macro,
    load_prices,
    pct,
    risk_regime,
    strongest_drivers,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "index.html"


def chart_html(fig: go.Figure, include_js=False) -> str:
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs="cdn" if include_js else False,
        config={"displayModeBar": False, "responsive": True},
    )


def score_chart(features: pd.DataFrame) -> go.Figure:
    score = features["contagion_risk_score"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=score.index, y=score, mode="lines", name="Contagion risk", line=dict(color="#1d5f8f", width=2)))
    fig.add_hrect(y0=0, y1=30, fillcolor="#e9f7ef", opacity=0.45, line_width=0)
    fig.add_hrect(y0=30, y1=60, fillcolor="#fff4df", opacity=0.45, line_width=0)
    fig.add_hrect(y0=60, y1=80, fillcolor="#fdebd3", opacity=0.45, line_width=0)
    fig.add_hrect(y0=80, y1=100, fillcolor="#fdecec", opacity=0.45, line_width=0)
    fig.update_layout(
        title="Canadian Bank Contagion Score",
        yaxis_title="0-100 score",
        height=420,
        margin=dict(l=30, r=20, t=55, b=35),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def driver_chart(features: pd.DataFrame) -> go.Figure:
    drivers = strongest_drivers(features).head(8).sort_values("Stress Percentile")
    fig = go.Figure(
        go.Bar(
            x=drivers["Stress Percentile"],
            y=drivers["Driver"],
            orientation="h",
            text=[f"{x:.0%}" for x in drivers["Stress Percentile"]],
            textposition="auto",
            marker_color="#1d5f8f",
        )
    )
    fig.update_layout(
        title="Current Risk Driver Percentiles",
        xaxis_tickformat=".0%",
        height=420,
        margin=dict(l=30, r=20, t=55, b=35),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def bank_chart(bank_table: pd.DataFrame) -> go.Figure:
    ordered = bank_table.sort_values("Node Stress")
    fig = go.Figure(
        go.Bar(
            x=ordered["Node Stress"],
            y=ordered["Bank"],
            orientation="h",
            text=[f"{x:.1f}" for x in ordered["Node Stress"]],
            textposition="auto",
            marker_color="#b42318",
        )
    )
    fig.update_layout(
        title="Bank-Level Node Stress",
        xaxis_title="0-100 stress score",
        height=420,
        margin=dict(l=30, r=20, t=55, b=35),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def macro_chart(macro: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col, label in [("policy_rate", "Policy rate"), ("ca_2y", "2Y yield"), ("ca_10y", "10Y yield")]:
        if col in macro:
            fig.add_trace(go.Scatter(x=macro.index, y=macro[col], mode="lines", name=label))
    fig.update_layout(
        title="Canadian Rate Backdrop",
        yaxis_title="Percent",
        height=420,
        margin=dict(l=30, r=20, t=55, b=35),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def correlation_chart(prices: pd.DataFrame) -> go.Figure:
    banks = [b for b in BANKS if b in prices]
    corr = prices[banks].pct_change().dropna().tail(63).corr()
    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            colorbar=dict(title="Correlation"),
        )
    )
    fig.update_layout(
        title="Latest 63-Day Bank Correlation Matrix",
        height=460,
        margin=dict(l=30, r=20, t=55, b=35),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def allocation_chart(bank_table: pd.DataFrame, score: float) -> tuple[go.Figure, pd.Series]:
    stress = bank_table.set_index("Bank")["Node Stress"]
    cash_weight = float(np.clip((score - 35) / 65, 0.05, 0.75))
    xfn_weight = float(np.clip((60 - score) / 100, 0.00, 0.25))
    bank_budget = 1 - cash_weight - xfn_weight
    bank_scores = (100 - stress).clip(lower=1)
    allocation = bank_budget * bank_scores / bank_scores.sum()
    weights = pd.concat([allocation, pd.Series({"XFN.TO": xfn_weight, "cash": cash_weight})]).sort_values(ascending=False)

    fig = go.Figure(
        go.Bar(
            x=weights.index,
            y=weights.values,
            text=[f"{x:.1%}" for x in weights.values],
            textposition="auto",
            marker_color="#1f7a5a",
        )
    )
    fig.update_layout(
        title=f"Risk-Aware Allocation Snapshot | Score {score:.1f}/100",
        yaxis_title="Weight",
        yaxis_tickformat=".0%",
        height=420,
        margin=dict(l=30, r=20, t=55, b=35),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig, weights


def table_html(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    view = df[columns].copy() if columns else df.copy()
    return view.to_html(index=False, classes="data-table", escape=False)


def build_html() -> str:
    features = load_features()
    prices = load_prices()
    macro = load_macro()
    score = latest(features, "contagion_risk_score", 50)
    regime = risk_regime(score)
    bank_table = bank_stress_snapshot(features)
    inventory = csv_inventory()
    allocation_fig, weights = allocation_chart(bank_table, score)

    bank_display = bank_table[
        ["Bank", "Name", "21D Return", "21D Volatility", "63D Drawdown", "Node Stress", "Action Readout"]
    ].copy()
    for col in ["21D Return", "21D Volatility", "63D Drawdown"]:
        bank_display[col] = bank_display[col].map(lambda x: pct(x) if pd.notna(x) else "N/A")
    bank_display["Node Stress"] = bank_display["Node Stress"].map(lambda x: f"{x:.1f}/100")

    drivers = strongest_drivers(features).head(8).copy()
    drivers["Stress Percentile"] = drivers["Stress Percentile"].map(lambda x: f"{x:.0%}" if pd.notna(x) else "N/A")

    allocation_table = weights.rename("Weight").reset_index().rename(columns={"index": "Asset"})
    allocation_table["Weight"] = allocation_table["Weight"].map(lambda x: f"{x:.1%}")

    charts = [
        chart_html(score_chart(features), include_js=True),
        chart_html(driver_chart(features)),
        chart_html(bank_chart(bank_table)),
        chart_html(correlation_chart(prices)),
        chart_html(macro_chart(macro)),
        chart_html(allocation_fig),
    ]

    actions = "".join(f"<li>{action}</li>" for action in regime["actions"])
    latest_date = latest_valid_date(features)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Canadian Bank Contagion Command Center</title>
  <meta name="description" content="Production snapshot of a Canadian bank contagion and portfolio risk dashboard.">
  <style>
    :root {{
      --ink: #18212f;
      --muted: #5d6877;
      --line: #d7dde5;
      --panel: #f7f9fc;
      --blue: #1d5f8f;
      --green: #1f7a5a;
      --amber: #ad6b00;
      --red: #b42318;
    }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: #ffffff;
      line-height: 1.5;
    }}
    header {{
      background: #eef4f8;
      border-bottom: 1px solid var(--line);
      padding: 56px 7vw 38px;
    }}
    main {{
      padding: 30px 7vw 56px;
      max-width: 1440px;
      margin: 0 auto;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: clamp(2rem, 4vw, 4.5rem);
      letter-spacing: 0;
      line-height: 1.03;
    }}
    h2 {{
      margin-top: 42px;
      font-size: clamp(1.35rem, 2vw, 2rem);
      letter-spacing: 0;
    }}
    h3 {{
      margin: 0 0 10px;
    }}
    p {{
      max-width: 980px;
      color: #354153;
    }}
    .subtitle {{
      font-size: 1.15rem;
      max-width: 980px;
      color: #354153;
    }}
    .pill {{
      display: inline-block;
      margin: 4px 8px 4px 0;
      padding: 5px 10px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: white;
      color: var(--muted);
      font-size: 0.88rem;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin-top: 24px;
    }}
    .metric, .card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      box-shadow: 0 1px 2px rgba(24, 33, 47, 0.04);
    }}
    .metric {{
      padding: 16px;
    }}
    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 0.88rem;
    }}
    .metric strong {{
      display: block;
      margin-top: 5px;
      font-size: 1.65rem;
    }}
    .callout {{
      border: 1px solid var(--line);
      border-left: 5px solid var(--blue);
      border-radius: 8px;
      background: var(--panel);
      padding: 18px 20px;
      margin: 24px 0;
    }}
    .callout.danger {{ border-left-color: var(--red); background: #fdecec; }}
    .callout.warning {{ border-left-color: var(--amber); background: #fff4df; }}
    .callout.success {{ border-left-color: var(--green); background: #e9f7ef; }}
    .chart-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
      margin-top: 16px;
    }}
    .chart-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 8px;
      overflow: hidden;
    }}
    .data-table {{
      width: 100%;
      border-collapse: collapse;
      margin: 14px 0 26px;
      font-size: 0.93rem;
    }}
    .data-table th, .data-table td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 9px;
      text-align: left;
      vertical-align: top;
    }}
    .data-table th {{
      background: var(--panel);
      color: var(--ink);
    }}
    footer {{
      color: var(--muted);
      border-top: 1px solid var(--line);
      padding-top: 22px;
      margin-top: 40px;
      font-size: 0.92rem;
    }}
    @media (max-width: 980px) {{
      .grid, .chart-grid {{ grid-template-columns: 1fr; }}
      header, main {{ padding-left: 22px; padding-right: 22px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Canadian Bank Contagion Command Center</h1>
    <p class="subtitle">A production snapshot translating Big Six bank, macro, network, and portfolio data into plain-English systemic-risk decisions.</p>
    <span class="pill">Data through {latest_date}</span>
    <span class="pill">Yahoo Finance prices + Bank of Canada yields</span>
    <span class="pill">Static Vercel production export</span>
    <div class="grid">
      <div class="metric"><span>Contagion Risk</span><strong>{score:.1f}/100</strong></div>
      <div class="metric"><span>Regime</span><strong>{regime["label"]}</strong></div>
      <div class="metric"><span>Avg Bank Vol</span><strong>{pct(latest(features, "avg_bank_vol_21d"))}</strong></div>
      <div class="metric"><span>Avg Bank Corr</span><strong>{latest(features, "avg_pairwise_corr_63d"):.2f}</strong></div>
    </div>
  </header>
  <main>
    <section class="callout {regime["tone"]}">
      <h3>Current Readout: {regime["label"]} Risk</h3>
      <p>{regime["summary"]}</p>
      <ul>{actions}</ul>
    </section>

    <h2>Executive Dashboard</h2>
    <div class="chart-grid">
      <div class="chart-card">{charts[0]}</div>
      <div class="chart-card">{charts[1]}</div>
    </div>
    <h3>Top Risk Drivers</h3>
    {table_html(drivers, ["Driver", "Latest", "Stress Percentile", "Status", "Why it matters"])}

    <h2>Bank and Network Risk</h2>
    <p>Bank stress is a market-behavior score. It is not a solvency rating. The network view shows when bank diversification may fail because names move together.</p>
    <div class="chart-grid">
      <div class="chart-card">{charts[2]}</div>
      <div class="chart-card">{charts[3]}</div>
    </div>
    {table_html(bank_display)}

    <h2>Canadian Economy Context</h2>
    <p>Rate levels, curve shape, CAD, oil, TSX, and volatility help explain why bank stress is rising or falling. Banks are exposed to borrowers, housing, funding, deposits, capital markets, and investor confidence.</p>
    <div class="chart-grid">
      <div class="chart-card">{charts[4]}</div>
      <div class="chart-card">{charts[5]}</div>
    </div>
    <h3>Risk-Aware Allocation Snapshot</h3>
    {table_html(allocation_table)}

    <h2>CSV and Model Traceability</h2>
    <p>Every CSV in the project is explained in the Streamlit app and the repo data dictionary. This production snapshot includes the current local file inventory used to generate the charts.</p>
    {table_html(inventory[["CSV", "Rows", "Columns", "Date Range", "Role", "Explanation"]])}

    <footer>
      Educational research only. Not investment advice, not a trading system, and not a regulatory bank risk model.
      For the full interactive Streamlit experience, run the branch locally with <code>streamlit run src/dashboard/app.py</code>.
    </footer>
  </main>
</body>
</html>
"""


def main() -> None:
    OUTPUT.write_text(build_html(), encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
