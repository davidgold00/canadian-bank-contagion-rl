import shutil
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, precision_score, recall_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dashboard.insight_utils import (  # noqa: E402
    BANKS,
    BANK_CONTEXT,
    BANK_NAMES,
    bank_stress_snapshot,
    csv_inventory,
    latest,
    latest_valid_date,
    load_features,
    load_macro,
    load_prices,
    pct,
    percentile_rank,
    risk_regime,
    strongest_drivers,
)
from src.portfolio.cvar_optimizer import efficient_frontier, optimize_cvar_portfolio, optimizer_tables  # noqa: E402
from src.portfolio.paper_trader import CVaRPaperPortfolioSimulator, PaperPortfolioSimulator  # noqa: E402
from src.portfolio.performance_metrics import drawdown_series, performance_summary  # noqa: E402
from src.portfolio.portfolio_constraints import PortfolioConstraints  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "public"
ROOT_INDEX = ROOT / "index.html"

PAGES = [
    ("about", "About"),
    ("market-overview", "Market Overview"),
    ("systemic-bank-network", "Systemic Bank Network"),
    ("contagion-risk-score", "Contagion Risk Score"),
    ("stress-testing-lab", "Stress Testing Lab"),
    ("rl-portfolio-agent", "RL Portfolio Agent"),
    ("model-validation", "Model Validation"),
    ("performance-tracker", "Performance Tracker"),
    ("cvar-optimization-lab", "CVaR Optimization Lab"),
    ("cvar-paper-fund", "CVaR Paper Fund"),
    ("rl-vs-cvar-comparison", "RL vs CVaR"),
    ("data-catalog", "Data Catalog"),
]


def chart_html(fig: go.Figure, include_js=False) -> str:
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs="cdn" if include_js else False,
        config={"displayModeBar": False, "responsive": True},
    )


def table_html(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    view = df[columns].copy() if columns else df.copy()
    return view.to_html(index=False, classes="data-table", escape=False)


def card(title: str, body: str, tone: str = "info") -> str:
    return f"<section class='callout {tone}'><h3>{title}</h3><p>{body}</p></section>"


def metric_grid(items: list[tuple[str, str]], accent: str = "blue") -> str:
    cells = "".join(
        f"<div class='metric {accent}'><span>{label}</span><strong>{value}</strong></div>"
        for label, value in items
    )
    return f"<div class='grid'>{cells}</div>"


def nav(active_slug: str) -> str:
    links = ["<span class='nav-brand'>&#9670; CBCCC</span>"]
    for slug, label in PAGES:
        href = "/" if slug == "about" else f"/{slug}"
        active = " active" if slug == active_slug else ""
        links.append(f"<a class='nav-link{active}' href='{href}'>{label}</a>")
    return "<nav class='nav'>" + "".join(links) + "</nav>"


_CHART_DARK = dict(
    paper_bgcolor="#16202d",
    plot_bgcolor="#0f1923",
    font=dict(family="'JetBrains Mono', 'SF Mono', monospace", color="#e8edf2", size=12),
    title_font=dict(size=14, color="#e8edf2"),
    xaxis=dict(gridcolor="#2a3a4a", linecolor="#2a3a4a", tickfont=dict(color="#7a91a6"), title_font=dict(color="#7a91a6")),
    yaxis=dict(gridcolor="#2a3a4a", linecolor="#2a3a4a", tickfont=dict(color="#7a91a6"), title_font=dict(color="#7a91a6")),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#e8edf2")),
    colorway=["#1e88e5", "#00c853", "#ffb300", "#f44336", "#00bcd4", "#e040fb"],
    margin=dict(l=30, r=20, t=55, b=35),
)


def _dark_chart(fig: go.Figure, height: int = 430) -> go.Figure:
    fig.update_layout(**_CHART_DARK, height=height)
    return fig


def page_template(slug: str, title: str, subtitle: str, body: str, latest_date: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title} | Canadian Bank Contagion Command Center</title>
  <meta name="description" content="Canadian bank systemic risk, contagion network, CVaR portfolio optimization, and investment signals.">
  <meta name="theme-color" content="#0f1923">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    /* ── Tokens ── */
    :root {{
      --bg:        #0f1923;
      --surface:   #16202d;
      --card:      #1c2a38;
      --border:    #2a3a4a;
      --ink:       #e8edf2;
      --muted:     #7a91a6;
      --blue:      #1e88e5;
      --blue-lt:   #42a5f5;
      --green:     #00c853;
      --amber:     #ffb300;
      --red:       #f44336;
      --teal:      #00bcd4;
      --radius:    10px;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}

    /* ── Base ── */
    html {{ scroll-behavior: smooth; }}
    body {{
      background: var(--bg);
      color: var(--ink);
      font-family: 'Inter', ui-sans-serif, system-ui, sans-serif;
      line-height: 1.55;
      font-size: 15px;
    }}

    /* ── Navigation ── */
    .nav {{
      position: sticky;
      top: 0;
      z-index: 100;
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      align-items: center;
      padding: 10px 6vw;
      background: rgba(15, 25, 35, 0.92);
      border-bottom: 1px solid var(--border);
      backdrop-filter: blur(14px);
      -webkit-backdrop-filter: blur(14px);
    }}
    .nav-brand {{
      font-weight: 700;
      font-size: 0.82rem;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--blue-lt);
      margin-right: 8px;
      white-space: nowrap;
    }}
    .nav-link {{
      color: var(--muted);
      text-decoration: none;
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 5px 11px;
      font-size: 0.8rem;
      background: var(--surface);
      transition: color 0.15s, border-color 0.15s, background 0.15s;
    }}
    .nav-link:hover {{
      color: var(--ink);
      border-color: var(--blue);
      background: rgba(30, 136, 229, 0.08);
    }}
    .nav-link.active {{
      background: var(--blue);
      border-color: var(--blue);
      color: #fff;
      font-weight: 600;
    }}

    /* ── Header ── */
    header {{
      padding: 52px 6vw 36px;
      background: linear-gradient(135deg, #0d1a26 0%, var(--surface) 100%);
      border-bottom: 1px solid var(--border);
      position: relative;
      overflow: hidden;
    }}
    header::before {{
      content: '';
      position: absolute;
      inset: 0;
      background: radial-gradient(ellipse 60% 60% at 80% 50%, rgba(30, 136, 229, 0.07) 0%, transparent 70%);
      pointer-events: none;
    }}
    h1 {{
      font-size: clamp(1.8rem, 4vw, 3.2rem);
      font-weight: 700;
      letter-spacing: -0.02em;
      line-height: 1.1;
      color: var(--ink);
      margin-bottom: 10px;
      position: relative;
    }}
    .subtitle {{
      font-size: 1.05rem;
      color: #9bb4c8;
      max-width: 820px;
      margin-bottom: 14px;
      position: relative;
    }}
    .pill {{
      display: inline-block;
      margin: 3px 6px 3px 0;
      padding: 4px 10px;
      border: 1px solid var(--border);
      border-radius: 999px;
      background: rgba(255,255,255,0.04);
      color: var(--muted);
      font-size: 0.76rem;
      letter-spacing: 0.02em;
    }}

    /* ── Main ── */
    main {{
      padding: 32px 6vw 64px;
      max-width: 1480px;
      margin: 0 auto;
    }}
    h2 {{
      font-size: clamp(1.2rem, 2.5vw, 1.7rem);
      font-weight: 600;
      letter-spacing: -0.01em;
      color: var(--ink);
      margin: 44px 0 14px;
      padding-bottom: 8px;
      border-bottom: 1px solid var(--border);
    }}
    h3 {{ font-size: 1.05rem; font-weight: 600; color: var(--ink); margin-bottom: 8px; margin-top: 24px; }}
    p {{ color: #c5d1db; max-width: 960px; }}
    a {{ color: var(--blue-lt); }}

    /* ── Metric grid ── */
    .grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin: 22px 0 28px;
    }}
    .metric {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 18px 16px;
      box-shadow: 0 2px 12px rgba(0,0,0,0.3);
      position: relative;
      overflow: hidden;
    }}
    .metric::before {{
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 2px;
      background: linear-gradient(90deg, var(--blue), transparent);
    }}
    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.07em;
      font-weight: 500;
      margin-bottom: 6px;
    }}
    .metric strong {{
      display: block;
      font-family: 'JetBrains Mono', monospace;
      font-size: 1.5rem;
      font-weight: 500;
      color: var(--ink);
      line-height: 1.1;
    }}
    .metric.green::before {{ background: linear-gradient(90deg, var(--green), transparent); }}
    .metric.red::before   {{ background: linear-gradient(90deg, var(--red), transparent); }}
    .metric.amber::before {{ background: linear-gradient(90deg, var(--amber), transparent); }}
    .metric.teal::before  {{ background: linear-gradient(90deg, var(--teal), transparent); }}

    /* ── Callout cards ── */
    .callout {{
      border-left: 4px solid var(--blue);
      background: rgba(30, 136, 229, 0.07);
      border-radius: 0 var(--radius) var(--radius) 0;
      padding: 16px 20px;
      margin: 18px 0;
      border-top: 1px solid var(--border);
      border-bottom: 1px solid var(--border);
      border-right: 1px solid var(--border);
    }}
    .callout h3 {{ color: var(--blue-lt); margin: 0 0 6px; font-size: 0.95rem; margin-top: 0; }}
    .callout p  {{ color: #c5d1db; font-size: 0.91rem; margin: 0; }}
    .callout.danger  {{ border-left-color: var(--red);   background: rgba(244, 67, 54, 0.07);  }}
    .callout.danger h3  {{ color: #ef9a9a; }}
    .callout.warning {{ border-left-color: var(--amber); background: rgba(255, 179, 0, 0.07); }}
    .callout.warning h3 {{ color: #ffe082; }}
    .callout.success {{ border-left-color: var(--green); background: rgba(0, 200, 83, 0.07); }}
    .callout.success h3 {{ color: #a5d6a7; }}
    .callout.teal    {{ border-left-color: var(--teal);  background: rgba(0, 188, 212, 0.07); }}
    .callout.teal h3    {{ color: #80deea; }}

    /* ── Chart grid ── */
    .chart-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      margin-top: 16px;
    }}
    .chart-card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      overflow: hidden;
      box-shadow: 0 2px 12px rgba(0,0,0,0.3);
    }}

    /* ── Data table ── */
    .data-table {{
      width: 100%;
      border-collapse: collapse;
      margin: 14px 0 28px;
      font-size: 0.87rem;
      background: var(--surface);
      border-radius: var(--radius);
      overflow: hidden;
    }}
    .data-table th {{
      background: #1c2a38;
      color: var(--muted);
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.07em;
      font-weight: 600;
      padding: 11px 12px;
      text-align: left;
      border-bottom: 1px solid var(--border);
    }}
    .data-table td {{
      padding: 10px 12px;
      border-bottom: 1px solid rgba(42, 58, 74, 0.6);
      color: #c5d1db;
      vertical-align: top;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.83rem;
    }}
    .data-table tr:last-child td {{ border-bottom: none; }}
    .data-table tr:hover td {{ background: rgba(30, 136, 229, 0.04); }}

    /* ── Page links ── */
    .page-links {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin-top: 18px;
    }}
    .page-links a {{
      display: block;
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 18px 20px;
      color: var(--ink);
      text-decoration: none;
      background: var(--card);
      transition: border-color 0.15s, box-shadow 0.15s;
    }}
    .page-links a:hover {{
      border-color: var(--blue);
      box-shadow: 0 0 0 1px rgba(30,136,229,0.3);
    }}
    .page-links a strong {{ color: var(--blue-lt); display: block; margin-bottom: 4px; }}
    .page-links a span   {{ color: var(--muted); font-size: 0.84rem; }}

    /* ── Signal badges ── */
    .sig-buy    {{ display:inline-block; padding:3px 10px; border-radius:999px; background:rgba(0,200,83,0.1);   color:var(--green); border:1px solid var(--green); font-size:0.76rem; font-weight:700; letter-spacing:0.05em; font-family:monospace; }}
    .sig-hold   {{ display:inline-block; padding:3px 10px; border-radius:999px; background:rgba(255,179,0,0.1);  color:var(--amber); border:1px solid var(--amber); font-size:0.76rem; font-weight:700; letter-spacing:0.05em; font-family:monospace; }}
    .sig-reduce {{ display:inline-block; padding:3px 10px; border-radius:999px; background:rgba(244,67,54,0.1);  color:var(--red);   border:1px solid var(--red);   font-size:0.76rem; font-weight:700; letter-spacing:0.05em; font-family:monospace; }}

    /* ── Regime banner ── */
    .regime-banner {{
      display: flex;
      align-items: center;
      gap: 16px;
      padding: 16px 20px;
      border-radius: var(--radius);
      margin: 20px 0;
      border: 1px solid var(--border);
    }}
    .regime-banner .score {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 2rem;
      font-weight: 700;
      line-height: 1;
    }}
    .regime-banner .label {{ font-weight: 700; font-size: 1rem; margin-bottom: 3px; }}
    .regime-banner .desc  {{ color: #c5d1db; font-size: 0.87rem; }}

    /* ── Footer ── */
    footer {{
      color: var(--muted);
      border-top: 1px solid var(--border);
      padding-top: 22px;
      margin-top: 48px;
      font-size: 0.83rem;
    }}

    /* ── Responsive ── */
    @media (max-width: 980px) {{
      .grid, .chart-grid, .page-links {{ grid-template-columns: 1fr; }}
      header, main {{ padding-left: 20px; padding-right: 20px; }}
      .nav {{ padding: 10px 20px; position: static; flex-wrap: wrap; }}
      h1 {{ font-size: 1.8rem; }}
    }}
    @media (max-width: 600px) {{
      .grid {{ grid-template-columns: repeat(2, 1fr); }}
    }}
  </style>
</head>
<body>
  {nav(slug)}
  <header>
    <h1>{title}</h1>
    <p class="subtitle">{subtitle}</p>
    <span class="pill">&#9679; Data through {latest_date}</span>
    <span class="pill">Yahoo Finance · Bank of Canada</span>
    <span class="pill">CVaR · Graph Network · RL</span>
  </header>
  <main>
    {body}
    <footer>
      Educational research only — not investment advice, not a trading system, and not a regulatory bank risk model.
      Simulated paper portfolio only. No real trades are placed.
    </footer>
  </main>
</body>
</html>
"""


def score_chart(features: pd.DataFrame) -> go.Figure:
    score = features["contagion_risk_score"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=score.index, y=score, mode="lines", name="Contagion risk",
                             line=dict(color="#1e88e5", width=2.5)))
    fig.add_hrect(y0=0,  y1=30,  fillcolor="#00c853", opacity=0.06, line_width=0)
    fig.add_hrect(y0=30, y1=60,  fillcolor="#ffb300", opacity=0.06, line_width=0)
    fig.add_hrect(y0=60, y1=80,  fillcolor="#ff6f00", opacity=0.06, line_width=0)
    fig.add_hrect(y0=80, y1=100, fillcolor="#f44336", opacity=0.08, line_width=0)
    fig.add_hline(y=30, line_dash="dot", line_color="#00c853", opacity=0.4)
    fig.add_hline(y=60, line_dash="dot", line_color="#ffb300", opacity=0.4)
    fig.add_hline(y=80, line_dash="dot", line_color="#f44336", opacity=0.4)
    fig.update_layout(title="Canadian Bank Contagion Score (0–100)", yaxis_title="Risk score")
    return _dark_chart(fig)


def driver_chart(features: pd.DataFrame) -> go.Figure:
    drivers = strongest_drivers(features).head(8).sort_values("Stress Percentile")
    fig = go.Figure(go.Bar(
        x=drivers["Stress Percentile"], y=drivers["Driver"], orientation="h",
        text=[f"{x:.0%}" for x in drivers["Stress Percentile"]], textposition="auto",
        marker=dict(
            color=drivers["Stress Percentile"].tolist(),
            colorscale=[[0, "#00c853"], [0.5, "#ffb300"], [1.0, "#f44336"]],
            cmin=0, cmax=1, showscale=False,
        ),
    ))
    fig.update_layout(title="Current Risk Driver Percentiles", xaxis_tickformat=".0%")
    return _dark_chart(fig)


def bank_chart(bank_table: pd.DataFrame) -> go.Figure:
    ordered = bank_table.sort_values("Node Stress")
    fig = go.Figure(go.Bar(
        x=ordered["Node Stress"], y=ordered["Bank"], orientation="h",
        text=[f"{x:.1f}" for x in ordered["Node Stress"]], textposition="auto",
        marker=dict(
            color=ordered["Node Stress"].tolist(),
            colorscale=[[0, "#00c853"], [0.5, "#ffb300"], [1.0, "#f44336"]],
            cmin=0, cmax=100, showscale=False,
        ),
    ))
    fig.update_layout(title="Bank Node Stress (0–100)", xaxis_title="Stress score")
    return _dark_chart(fig)


def correlation_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    banks = [b for b in BANKS if b in prices]
    return prices[banks].pct_change().dropna().tail(63).corr().fillna(0)


def correlation_chart(prices: pd.DataFrame) -> go.Figure:
    corr = correlation_matrix(prices)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        zmin=-1, zmax=1, colorscale="RdBu_r",
        text=np.round(corr.values, 2), texttemplate="%{text}",
        colorbar=dict(title="Corr", tickfont=dict(color="#7a91a6")),
    ))
    fig.update_layout(title="63-Day Bank Correlation Matrix")
    return _dark_chart(fig, height=460)


def macro_chart(macro: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    colors = ["#1e88e5", "#00c853", "#ffb300"]
    for i, (col, label) in enumerate([("policy_rate", "Policy Rate"), ("ca_2y", "2Y Yield"), ("ca_10y", "10Y Yield")]):
        if col in macro:
            fig.add_trace(go.Scatter(x=macro.index, y=macro[col], mode="lines", name=label,
                                     line=dict(color=colors[i], width=2)))
    fig.update_layout(title="Canadian Rate Backdrop", yaxis_title="Percent (%)")
    return _dark_chart(fig)


def allocation_chart(bank_table: pd.DataFrame, score: float) -> tuple[go.Figure, pd.Series]:
    stress = bank_table.set_index("Bank")["Node Stress"]
    cash_weight = float(np.clip((score - 35) / 65, 0.05, 0.75))
    xfn_weight = float(np.clip((60 - score) / 100, 0.00, 0.25))
    bank_budget = 1 - cash_weight - xfn_weight
    bank_scores = (100 - stress).clip(lower=1)
    allocation = bank_budget * bank_scores / bank_scores.sum()
    weights = pd.concat([allocation, pd.Series({"XFN.TO": xfn_weight, "cash": cash_weight})]).sort_values(ascending=False)
    fig = go.Figure(go.Bar(
        x=weights.index, y=weights.values,
        text=[f"{x:.1%}" for x in weights.values], textposition="outside",
        marker=dict(
            color=weights.values.tolist(),
            colorscale=[[0, "#2a3a4a"], [1.0, "#1e88e5"]],
            cmin=0, cmax=weights.max(), showscale=False,
        ),
    ))
    fig.update_layout(title=f"Risk-Aware Allocation | Score {score:.1f}/100", yaxis_title="Weight", yaxis_tickformat=".0%")
    return _dark_chart(fig), weights


def paper_portfolio() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    prices = load_prices()
    features = load_features()
    start_date = prices.index[max(63, int(len(prices) * 0.55))]
    simulator = PaperPortfolioSimulator(
        prices=prices,
        features=features,
        model_path=ROOT / "artifacts" / "rl" / "ppo_model.zip",
        use_trained_model=True,
    )
    result = simulator.simulate(
        initial_capital=100_000,
        transaction_cost_bps=5,
        rebalance_threshold=0.01,
        start_date=start_date,
        max_single_name_weight=0.22,
        max_bank_exposure=0.80,
        defensive_cash_sensitivity=1.0,
    )
    return result.ledger, result.trades, result.current_holdings, result.weights, result.benchmarks, result.policy_source


def performance_value_chart(ledger: pd.DataFrame, benchmarks: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ledger.index, y=ledger["portfolio_value"], mode="lines",
                             name="Model paper portfolio", line=dict(color="#1e88e5", width=3)))
    colors = ["#00c853", "#ffb300", "#f44336", "#00bcd4"]
    for i, col in enumerate(benchmarks.columns):
        fig.add_trace(go.Scatter(x=benchmarks.index, y=benchmarks[col], mode="lines", name=col,
                                 line=dict(color=colors[i % len(colors)], width=1.5, dash="dot")))
    fig.update_layout(title="Paper Portfolio Value vs Benchmarks", yaxis_title="CAD value")
    return _dark_chart(fig, height=460)


def performance_allocation_chart(weights: pd.DataFrame) -> go.Figure:
    latest_weights = weights.iloc[-1].sort_values()
    fig = go.Figure(go.Bar(
        x=latest_weights.values, y=latest_weights.index, orientation="h",
        text=[f"{x:.1%}" for x in latest_weights.values], textposition="auto",
        marker_color="#1e88e5",
    ))
    fig.update_layout(title="Current Paper Portfolio Allocation", xaxis_title="Weight", xaxis_tickformat=".0%")
    return _dark_chart(fig)


def cash_risk_chart(ledger: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ledger.index, y=ledger["contagion_risk_score"],
                             name="Contagion risk", yaxis="y1", line=dict(color="#f44336", width=2)))
    fig.add_trace(go.Scatter(x=ledger.index, y=ledger["cash_weight"],
                             name="Cash weight", yaxis="y2", line=dict(color="#00c853", width=2)))
    fig.update_layout(
        title="Cash Weight vs Contagion Risk Score",
        yaxis=dict(title="Risk score (0–100)", color="#7a91a6", gridcolor="#2a3a4a"),
        yaxis2=dict(title="Cash weight", overlaying="y", side="right", tickformat=".0%", color="#7a91a6"),
    )
    return _dark_chart(fig)


def performance_drawdown_chart(ledger: pd.DataFrame, benchmarks: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ledger.index, y=drawdown_series(ledger["portfolio_value"]),
                             mode="lines", name="Model paper portfolio", line=dict(color="#1e88e5", width=2.5)))
    colors = ["#00c853", "#ffb300", "#f44336", "#00bcd4"]
    for i, col in enumerate(benchmarks.columns):
        fig.add_trace(go.Scatter(x=benchmarks.index, y=drawdown_series(benchmarks[col]),
                                 mode="lines", name=col, line=dict(color=colors[i % len(colors)], width=1.5, dash="dot")))
    fig.update_layout(title="Portfolio Drawdown vs Benchmarks", yaxis_title="Drawdown", yaxis_tickformat=".0%")
    return _dark_chart(fig)


def cvar_snapshot() -> tuple:
    prices = load_prices()
    features = load_features()
    constraints = PortfolioConstraints(max_single_name_weight=0.20, max_bank_exposure=0.70, min_cash_weight=0.05, max_cash_weight=0.60)
    result = optimize_cvar_portfolio(
        prices_history=prices,
        features_history=features,
        confidence_level=0.95,
        lookback_window=126,
        constraints=constraints,
        risk_aversion=7.0,
        cvar_penalty=9.0,
        contagion_penalty=0.90,
    )
    frontier = efficient_frontier(prices, features, confidence_level=0.95, lookback_window=126, constraints=constraints)
    return result, frontier


def cvar_weight_chart(weights: pd.Series) -> go.Figure:
    ordered = weights.sort_values()
    fig = go.Figure(go.Bar(
        x=ordered.values, y=ordered.index, orientation="h",
        text=[f"{x:.1%}" for x in ordered.values], textposition="auto",
        marker_color="#1e88e5",
    ))
    fig.update_layout(title="Graph-Adjusted CVaR Optimized Weights", xaxis_title="Weight", xaxis_tickformat=".0%")
    return _dark_chart(fig)


def cvar_frontier_chart(frontier: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=frontier["CVaR"], y=frontier["Expected Return"],
        mode="markers+lines",
        text=[f"λ={x}" for x in frontier["Risk Aversion"]],
        marker=dict(size=12, color=frontier["Cash"], colorscale="Blues",
                    colorbar=dict(title="Cash", tickfont=dict(color="#7a91a6")),
                    line=dict(width=1, color="#2a3a4a")),
        line=dict(color="#1e88e5", width=1.5),
    ))
    fig.update_layout(title="CVaR Efficient Frontier", xaxis_title="Historical CVaR",
                      yaxis_title="Expected annual return", xaxis_tickformat=".1%", yaxis_tickformat=".1%")
    return _dark_chart(fig)


def cvar_risk_contribution_chart(contrib: pd.DataFrame) -> go.Figure:
    view = contrib.sort_values("CVaR Contribution")
    fig = go.Figure(go.Bar(
        x=view["CVaR Contribution"], y=view["Asset"], orientation="h",
        text=[f"{x:.1%}" for x in view["CVaR Contribution"]], textposition="auto",
        marker_color="#f44336",
    ))
    fig.update_layout(title="Contribution to Portfolio CVaR (95%)", xaxis_title="Share of tail risk", xaxis_tickformat=".0%")
    return _dark_chart(fig)


def covariance_heatmap(matrix: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=matrix.values, x=matrix.columns, y=matrix.index,
        colorscale="RdBu_r",
        colorbar=dict(title="Cov", tickfont=dict(color="#7a91a6")),
    ))
    fig.update_layout(title=title)
    return _dark_chart(fig, height=460)


def cvar_paper_fund() -> tuple:
    prices = load_prices()
    features = load_features()
    start_date = prices.index[max(252, int(len(prices) * 0.72))]
    cvar = CVaRPaperPortfolioSimulator(prices, features, use_trained_model=False).simulate_cvar(
        initial_capital=100_000,
        transaction_cost_bps=5,
        rebalance_threshold=0.01,
        start_date=start_date,
        lookback_window=126,
        rebalance_frequency=10,
    )
    rl = PaperPortfolioSimulator(prices, features, use_trained_model=True).simulate(
        initial_capital=100_000,
        transaction_cost_bps=5,
        rebalance_threshold=0.01,
        start_date=start_date,
    )
    benchmarks = cvar.benchmarks.copy()
    benchmarks["RL research baseline"] = rl.ledger["portfolio_value"].reindex(benchmarks.index).ffill()
    return cvar, rl, benchmarks


def cvar_value_chart(ledger: pd.DataFrame, benchmarks: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ledger.index, y=ledger["portfolio_value"], mode="lines",
                             name="CVaR paper fund", line=dict(color="#1e88e5", width=3)))
    colors = ["#00c853", "#ffb300", "#f44336", "#00bcd4"]
    for i, col in enumerate(benchmarks.columns):
        fig.add_trace(go.Scatter(x=benchmarks.index, y=benchmarks[col], mode="lines", name=col,
                                 line=dict(color=colors[i % len(colors)], width=1.5, dash="dot")))
    fig.update_layout(title="CVaR Paper Fund vs Benchmarks", yaxis_title="Portfolio value (CAD)")
    return _dark_chart(fig, height=460)


def exposure_chart(cvar_ledger: pd.DataFrame, rl_ledger: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cvar_ledger.index, y=cvar_ledger["bank_exposure"], mode="lines",
                             name="CVaR financial exposure", line=dict(color="#1e88e5", width=2)))
    fig.add_trace(go.Scatter(x=rl_ledger.index, y=rl_ledger["bank_exposure"], mode="lines",
                             name="RL financial exposure", line=dict(color="#00c853", width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=cvar_ledger.index, y=cvar_ledger["cash_weight"], mode="lines",
                             name="CVaR cash", line=dict(color="#ffb300", width=1.5)))
    fig.add_trace(go.Scatter(x=rl_ledger.index, y=rl_ledger["cash_weight"], mode="lines",
                             name="RL cash", line=dict(color="#f44336", width=1.5, dash="dot")))
    fig.update_layout(title="Financial Exposure and Cash Weight", yaxis_title="Weight", yaxis_tickformat=".0%")
    return _dark_chart(fig)


def normalize_prices(prices: pd.DataFrame, columns: list[str], days=252) -> pd.DataFrame:
    view = prices[[c for c in columns if c in prices]].tail(days).ffill().dropna(how="all")
    if view.empty:
        return view
    return view.divide(view.iloc[0]).mul(100)


def price_context_chart(prices: pd.DataFrame) -> go.Figure:
    normalized = normalize_prices(prices, ["XFN.TO", "XIU.TO", "^GSPTSE", "CADUSD=X", "CL=F", "GC=F"])
    fig = go.Figure()
    colors = ["#1e88e5", "#00c853", "#ffb300", "#f44336", "#00bcd4", "#e040fb"]
    for i, col in enumerate(normalized.columns):
        fig.add_trace(go.Scatter(x=normalized.index, y=normalized[col], mode="lines", name=col,
                                 line=dict(color=colors[i % len(colors)], width=2)))
    fig.update_layout(title="One-Year Macro Market Context (Indexed to 100)", yaxis_title="Indexed to 100")
    return _dark_chart(fig)


def component_scores(features: pd.DataFrame) -> pd.DataFrame:
    candidates = {
        "Bank volatility": ("avg_bank_vol_21d", 1),
        "Bank correlation": ("avg_pairwise_corr_63d", 1),
        "Financials drawdown": ("XFN.TO_drawdown_63d", -1),
        "Global volatility": ("VIX_level", 1),
        "Volatility spike": ("VIX_chg_5d", 1),
        "Yield curve pressure": ("slope_10y_2y", -1),
        "Oil shock": ("CL=F_ret_21d", -1),
        "CAD pressure": ("CADUSD=X_ret_21d", -1),
    }
    out = pd.DataFrame(index=features.index)
    for label, (col, sign) in candidates.items():
        if col in features:
            out[label] = 100 * (sign * features[col]).rank(pct=True)
    out["Composite score"] = features["contagion_risk_score"]
    return out.ffill().fillna(50).clip(0, 100)


def component_bar_chart(components: pd.DataFrame) -> go.Figure:
    latest_components = components.iloc[-1].sort_values()
    fig = go.Figure(go.Bar(
        x=latest_components.values, y=latest_components.index, orientation="h",
        text=[f"{x:.1f}" for x in latest_components.values], textposition="auto",
        marker=dict(
            color=latest_components.values.tolist(),
            colorscale=[[0, "#00c853"], [0.5, "#ffb300"], [1.0, "#f44336"]],
            cmin=0, cmax=100, showscale=False,
        ),
    ))
    fig.update_layout(title="Current Component Stress Scores (0–100)", xaxis_title="Score")
    return _dark_chart(fig, height=500)


def component_heatmap(components: pd.DataFrame) -> go.Figure:
    recent = components.tail(126).T
    fig = go.Figure(go.Heatmap(
        z=recent.values, x=recent.columns, y=recent.index,
        zmin=0, zmax=100,
        colorscale="RdYlGn_r",
        colorbar=dict(title="Stress", tickfont=dict(color="#7a91a6")),
    ))
    fig.update_layout(title="Stress Breadth — Last 6 Months")
    return _dark_chart(fig, height=540)


def stress_paths(prices: pd.DataFrame, scenario_name="Liquidity Squeeze") -> tuple[pd.DataFrame, pd.Series]:
    scenarios = {
        "Housing Crisis": {"RY.TO": 35, "TD.TO": 35, "BMO.TO": 30, "BNS.TO": 30, "CM.TO": 45, "NA.TO": 32},
        "Oil Crash": {"RY.TO": 20, "TD.TO": 18, "BMO.TO": 26, "BNS.TO": 25, "CM.TO": 22, "NA.TO": 18},
        "Liquidity Squeeze": {"RY.TO": 40, "TD.TO": 38, "BMO.TO": 36, "BNS.TO": 36, "CM.TO": 38, "NA.TO": 34},
        "Yield Curve Inversion": {"RY.TO": 24, "TD.TO": 24, "BMO.TO": 22, "BNS.TO": 22, "CM.TO": 28, "NA.TO": 20},
        "Global Risk-Off": {"RY.TO": 32, "TD.TO": 32, "BMO.TO": 30, "BNS.TO": 31, "CM.TO": 33, "NA.TO": 29},
    }
    returns = prices[BANKS].pct_change().tail(126)
    corr = returns.corr().fillna(0).clip(lower=0)
    values = corr.to_numpy(copy=True)
    np.fill_diagonal(values, 0)
    adj = pd.DataFrame(values, index=corr.index, columns=corr.columns)
    adj = adj.div(adj.sum(axis=1).replace(0, 1), axis=0)
    stress = pd.Series(scenarios[scenario_name], dtype=float)
    rows = [{"Step": 0, **stress.to_dict()}]
    for step in range(1, 6):
        stress = (0.70 * stress + 0.45 * adj.T.dot(stress)).clip(0, 100)
        rows.append({"Step": step, **stress.to_dict()})
    paths = pd.DataFrame(rows).set_index("Step")
    return paths, paths.iloc[-1]


def stress_path_chart(paths: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    colors = ["#1e88e5", "#00c853", "#ffb300", "#f44336", "#00bcd4", "#e040fb"]
    for i, bank in enumerate(BANKS):
        if bank in paths.columns:
            fig.add_trace(go.Scatter(x=paths.index, y=paths[bank], mode="lines+markers", name=bank,
                                     line=dict(color=colors[i % len(colors)], width=2.5),
                                     marker=dict(size=7)))
    fig.add_hline(y=70, line_dash="dash", line_color="#f44336", opacity=0.5, annotation_text="Severe")
    fig.add_hline(y=40, line_dash="dot", line_color="#ffb300", opacity=0.5, annotation_text="Moderate")
    fig.update_layout(title="Liquidity Squeeze: Contagion Propagation", xaxis_title="Propagation step", yaxis_title="Stress score (0–100)")
    return _dark_chart(fig, height=470)


def final_stress_chart(final_stress: pd.Series) -> go.Figure:
    ordered = final_stress.sort_values()
    fig = go.Figure(go.Bar(
        x=ordered.values, y=ordered.index, orientation="h",
        text=[f"{x:.1f}" for x in ordered.values], textposition="auto",
        marker=dict(
            color=ordered.values.tolist(),
            colorscale=[[0, "#00c853"], [0.5, "#ffb300"], [1.0, "#f44336"]],
            cmin=0, cmax=100, showscale=False,
        ),
    ))
    fig.update_layout(title="Final Scenario Stress Ranking", xaxis_title="Stress score (0–100)")
    return _dark_chart(fig)


def network_chart(prices: pd.DataFrame, bank_table: pd.DataFrame) -> go.Figure:
    corr = correlation_matrix(prices)
    stress = bank_table.set_index("Bank")["Node Stress"]
    graph = nx.Graph()
    graph.add_nodes_from(BANKS)
    for i, source in enumerate(BANKS):
        for target in BANKS[i + 1:]:
            weight = abs(float(corr.loc[source, target]))
            if weight >= 0.35:
                graph.add_edge(source, target, weight=weight)
    pos = nx.spring_layout(graph, seed=19, weight="weight") if graph.number_of_edges() else nx.circular_layout(graph)
    edge_x, edge_y = [], []
    for source, target in graph.edges:
        x0, y0 = pos[source]; x1, y1 = pos[target]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
    node_x, node_y, labels, node_colors, sizes, hover = [], [], [], [], [], []
    degree = nx.degree_centrality(graph)
    max_degree = max(degree.values()) if degree else 1
    for bank in BANKS:
        x, y = pos[bank]
        node_x.append(x); node_y.append(y)
        labels.append(bank)
        node_colors.append(float(stress.get(bank, 50)))
        sizes.append(32 + 60 * degree.get(bank, 0) / (max_degree or 1))
        hover.append(f"<b>{bank}</b><br>{BANK_NAMES[bank]}<br>Stress: {stress.get(bank, 50):.1f}/100<br>{BANK_CONTEXT[bank]}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                             line=dict(width=1.5, color="rgba(120,145,166,0.3)"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=labels, textposition="top center",
        hovertext=hover, hoverinfo="text",
        marker=dict(
            size=sizes, color=node_colors,
            cmin=0, cmax=100,
            colorscale=[[0, "#00c853"], [0.5, "#ffb300"], [1.0, "#f44336"]],
            showscale=True, colorbar=dict(title="Stress", tickfont=dict(color="#7a91a6")),
            line=dict(width=2, color="#16202d"),
        ),
    ))
    fig.update_layout(
        title="Bank Contagion Network (size = centrality, color = stress)",
        showlegend=False,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return _dark_chart(fig, height=580)


def model_metrics(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = (features["contagion_risk_score"].shift(-5) >= features["contagion_risk_score"].shift(-5).quantile(0.80)).astype(int)
    feature_cols = [c for c in features.columns if c != "contagion_risk_score" and pd.api.types.is_numeric_dtype(features[c])]
    dataset = pd.concat([features[feature_cols].replace([np.inf, -np.inf], np.nan), y.rename("target")], axis=1).dropna()
    split = int(len(dataset) * 0.70)
    X_train, X_test = dataset[feature_cols].iloc[:split], dataset[feature_cols].iloc[split:]
    y_train, y_test = dataset["target"].astype(int).iloc[:split], dataset["target"].astype(int).iloc[split:]
    models = {
        "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1500, class_weight="balanced"))]),
        "Random Forest": RandomForestClassifier(n_estimators=180, max_depth=5, min_samples_leaf=10, random_state=42, class_weight="balanced"),
    }
    rows = []
    roc_rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.50).astype(int)
        fpr, tpr, _ = roc_curve(y_test, prob)
        rows.append({"Model": name, "AUC": auc(fpr, tpr), "Accuracy": accuracy_score(y_test, pred), "Precision": precision_score(y_test, pred, zero_division=0), "Recall": recall_score(y_test, pred, zero_division=0)})
        roc_rows.extend([{"Model": name, "FPR": x, "TPR": yv} for x, yv in zip(fpr, tpr)])
    return pd.DataFrame(rows).sort_values("AUC", ascending=False), pd.DataFrame(roc_rows)


def roc_chart(roc_df: pd.DataFrame, metrics: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    auc_map = metrics.set_index("Model")["AUC"].to_dict()
    colors_roc = ["#1e88e5", "#00c853"]
    for i, (model, group) in enumerate(roc_df.groupby("Model")):
        fig.add_trace(go.Scatter(
            x=group["FPR"], y=group["TPR"], mode="lines",
            name=f"{model} (AUC={auc_map.get(model, 0):.2f})",
            line=dict(color=colors_roc[i % len(colors_roc)], width=2.5),
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random",
        line=dict(dash="dash", color="#7a91a6", width=1),
    ))
    fig.update_layout(title="ROC Curve — Future Stress Prediction (Out-of-Sample)",
                      xaxis_title="False positive rate", yaxis_title="True positive rate")
    return _dark_chart(fig, height=450)


def regime_banner_html(score: float, label: str, summary: str, tone: str) -> str:
    """Generate a styled regime banner for the static site."""
    bg_map = {"success": "rgba(0,200,83,0.08)", "warning": "rgba(255,179,0,0.08)",
               "danger": "rgba(244,67,54,0.08)", "info": "rgba(30,136,229,0.08)"}
    color_map = {"success": "#00c853", "warning": "#ffb300", "danger": "#f44336", "info": "#1e88e5"}
    border = color_map.get(tone, "#1e88e5")
    color = color_map.get(tone, "#1e88e5")
    bg = bg_map.get(tone, bg_map["info"])
    return (
        f"<div class='regime-banner' style='background:{bg};border-color:{border};'>"
        f"<div class='score' style='color:{color}'>{score:.0f}/100</div>"
        f"<div><div class='label' style='color:{color}'>{label} Risk Regime</div>"
        f"<div class='desc'>{summary}</div></div></div>"
    )


def build_pages() -> dict[str, str]:
    features = load_features()
    prices = load_prices()
    macro = load_macro()
    bank_table = bank_stress_snapshot(features)
    inventory = csv_inventory()
    score = latest(features, "contagion_risk_score", 50)
    regime = risk_regime(score)
    latest_date = latest_valid_date(features)
    drivers = strongest_drivers(features)
    components = component_scores(features)
    allocation_fig, weights = allocation_chart(bank_table, score)
    regime_html = regime_banner_html(score, regime["label"], regime["summary"], regime["tone"])

    bank_display = bank_table[["Bank", "Name", "21D Return", "21D Volatility", "63D Drawdown", "Beta to XFN", "Node Stress", "Action Readout", "Economic Lens"]].copy()
    for col in ["21D Return", "21D Volatility", "63D Drawdown"]:
        bank_display[col] = bank_display[col].map(lambda x: pct(x) if pd.notna(x) else "N/A")
    bank_display["Beta to XFN"] = bank_display["Beta to XFN"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    bank_display["Node Stress"] = bank_display["Node Stress"].map(lambda x: f"{x:.1f}/100")

    driver_display = drivers.copy()
    driver_display["Stress Percentile"] = driver_display["Stress Percentile"].map(lambda x: f"{x:.0%}" if pd.notna(x) else "N/A")

    weight_table = weights.rename("Weight").reset_index().rename(columns={"index": "Asset"})
    weight_table["Weight"] = weight_table["Weight"].map(lambda x: f"{x:.1%}")

    paths, final_stress = stress_paths(prices)
    stress_impact = pd.DataFrame({"Bank": BANKS, "Final Stress": [final_stress[b] for b in BANKS], "Equal-Weight Loss Contribution": [final_stress[b] / final_stress.sum() for b in BANKS]})
    stress_impact["Final Stress"] = stress_impact["Final Stress"].map(lambda x: f"{x:.1f}/100")
    stress_impact["Equal-Weight Loss Contribution"] = stress_impact["Equal-Weight Loss Contribution"].map(lambda x: f"{x:.1%}")

    metrics, roc_df = model_metrics(features)
    metrics_display = metrics.copy()
    for col in ["AUC", "Accuracy", "Precision", "Recall"]:
        metrics_display[col] = metrics_display[col].map(lambda x: f"{x:.3f}")

    paper_ledger, paper_trades, paper_holdings, paper_weights, paper_benchmarks, paper_policy_source = paper_portfolio()
    paper_summary = performance_summary(
        paper_ledger["portfolio_value"],
        paper_ledger["daily_return"],
        paper_ledger["turnover"],
        paper_ledger["transaction_costs"],
    )
    paper_latest = paper_ledger.iloc[-1]
    paper_holdings_display = paper_holdings[["asset", "shares", "latest_price", "market_value", "weight", "unrealized_pnl"]].copy()
    paper_holdings_display.columns = ["Asset", "Shares", "Latest Price", "Market Value", "Weight", "Unrealized P&L"]
    for col in ["Latest Price", "Market Value", "Unrealized P&L"]:
        paper_holdings_display[col] = paper_holdings_display[col].map(lambda x: f"${x:,.2f}")
    paper_holdings_display["Shares"] = paper_holdings_display["Shares"].map(lambda x: f"{x:,.4f}")
    paper_holdings_display["Weight"] = paper_holdings_display["Weight"].map(lambda x: f"{x:.1%}")
    paper_trades_display = paper_trades.sort_values("date", ascending=False).head(20).copy()
    if not paper_trades_display.empty:
        paper_trades_display["date"] = pd.to_datetime(paper_trades_display["date"]).dt.date
        paper_trades_display = paper_trades_display[["date", "asset", "action", "shares", "price", "notional", "transaction_cost", "reason"]]
        paper_trades_display.columns = ["Date", "Asset", "Action", "Shares", "Price", "Notional", "Transaction Cost", "Reason"]
        paper_trades_display["Shares"] = paper_trades_display["Shares"].map(lambda x: f"{x:,.4f}")
        for col in ["Price", "Notional", "Transaction Cost"]:
            paper_trades_display[col] = paper_trades_display[col].map(lambda x: f"${x:,.2f}")

    cvar_result, cvar_frontier = cvar_snapshot()
    cvar_weights, cvar_penalties, cvar_diagnostics = optimizer_tables(cvar_result)
    cvar_weights_display = cvar_weights.copy()
    for col in ["Weight", "Expected Return Contribution", "Volatility Contribution", "CVaR Contribution", "Contagion Contribution"]:
        if col in cvar_weights_display:
            cvar_weights_display[col] = cvar_weights_display[col].map(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
    cvar_paper, rl_paper, cvar_benchmarks = cvar_paper_fund()
    cvar_summary = performance_summary(cvar_paper.ledger["portfolio_value"], cvar_paper.ledger["daily_return"], cvar_paper.ledger["turnover"], cvar_paper.ledger["transaction_costs"])
    rl_summary = performance_summary(rl_paper.ledger["portfolio_value"], rl_paper.ledger["daily_return"], rl_paper.ledger["turnover"], rl_paper.ledger["transaction_costs"])
    cvar_holdings_display = cvar_paper.current_holdings[["asset", "shares", "latest_price", "market_value", "weight", "unrealized_pnl"]].copy()
    cvar_holdings_display.columns = ["Asset", "Shares", "Latest Price", "Market Value", "Weight", "Unrealized P&L"]
    cvar_holdings_display["Shares"] = cvar_holdings_display["Shares"].map(lambda x: f"{x:,.4f}")
    cvar_holdings_display["Weight"] = cvar_holdings_display["Weight"].map(lambda x: f"{x:.1%}")
    for col in ["Latest Price", "Market Value", "Unrealized P&L"]:
        cvar_holdings_display[col] = cvar_holdings_display[col].map(lambda x: f"${x:,.2f}")
    cvar_trades_display = cvar_paper.trades.sort_values("date", ascending=False).head(20).copy()
    if not cvar_trades_display.empty:
        cvar_trades_display["date"] = pd.to_datetime(cvar_trades_display["date"]).dt.date
        cvar_trades_display = cvar_trades_display[["date", "asset", "action", "shares", "price", "notional", "transaction_cost", "reason"]]
        cvar_trades_display.columns = ["Date", "Asset", "Action", "Shares", "Price", "Notional", "Transaction Cost", "Reason"]
        cvar_trades_display["Shares"] = cvar_trades_display["Shares"].map(lambda x: f"{x:,.4f}")
        for col in ["Price", "Notional", "Transaction Cost"]:
            cvar_trades_display[col] = cvar_trades_display[col].map(lambda x: f"${x:,.2f}")

    comparison = pd.DataFrame(
        [
            {
                "Strategy": "CVaR optimizer",
                "Ending Value": f"${cvar_summary['ending_value']:,.0f}",
                "Cumulative Return": f"{cvar_summary['cumulative_return']:.1%}",
                "Volatility": f"{cvar_summary['annualized_volatility']:.1%}",
                "Sharpe": f"{cvar_summary['sharpe_ratio']:.2f}",
                "CVaR": f"{cvar_summary['conditional_value_at_risk']:.1%}",
                "Max Drawdown": f"{cvar_summary['max_drawdown']:.1%}",
            },
            {
                "Strategy": "RL research baseline",
                "Ending Value": f"${rl_summary['ending_value']:,.0f}",
                "Cumulative Return": f"{rl_summary['cumulative_return']:.1%}",
                "Volatility": f"{rl_summary['annualized_volatility']:.1%}",
                "Sharpe": f"{rl_summary['sharpe_ratio']:.2f}",
                "CVaR": f"{rl_summary['conditional_value_at_risk']:.1%}",
                "Max Drawdown": f"{rl_summary['max_drawdown']:.1%}",
            },
        ]
    )

    pages = {}

    about_links = "".join(
        f"<a href='/{slug if slug != 'about' else ''}'><strong>{label}</strong><br><span>{description}</span></a>"
        for slug, label, description in [
            ("market-overview", "Market Overview", "Current regime, drivers, and economic context."),
            ("systemic-bank-network", "Systemic Bank Network", "Contagion links and systemic bank centrality."),
            ("contagion-risk-score", "Contagion Risk Score", "Score decomposition and decision rules."),
            ("stress-testing-lab", "Stress Testing Lab", "Scenario propagation and portfolio impact."),
            ("rl-portfolio-agent", "RL Portfolio Agent", "Risk-aware allocation and cash posture."),
            ("model-validation", "Model Validation", "Out-of-sample stress-prediction credibility."),
            ("performance-tracker", "Performance Tracker", "Paper portfolio, trades, holdings, costs, and benchmarks."),
            ("cvar-optimization-lab", "CVaR Optimization Lab", "Graph-adjusted covariance, CVaR frontier, and constrained portfolio construction."),
            ("cvar-paper-fund", "CVaR Paper Fund", "Paper fund following the governed CVaR optimizer through time."),
            ("rl-vs-cvar-comparison", "RL vs CVaR", "Comparative quant research across stress regimes and allocation stability."),
            ("data-catalog", "Data Catalog", "CSV inventory, explanations, and data lineage."),
        ]
    )
    avg_bank_vol_str = pct(latest(features, "avg_bank_vol_21d"))
    avg_corr_val = latest(features, "avg_pairwise_corr_63d")
    avg_corr_str = f"{avg_corr_val:.2f}" if pd.notna(avg_corr_val) else "N/A"

    about_body = (
        regime_html
        + metric_grid([
            ("Contagion Risk", f"{score:.1f}/100"),
            ("Regime", regime["label"]),
            ("Avg Bank Vol (21D)", avg_bank_vol_str),
            ("Avg Bank Correlation", avg_corr_str),
          ])
        + card("Core Question",
               "When stress rises in Canadian financial markets, how might it spread across the Big Six banks — "
               "and how should a risk-aware portfolio adapt its exposure?", regime["tone"])
        + "<h2>How the System Works</h2>"
        + "<p>Market and macro data feed into a bank contagion graph and composite risk score. "
        "Scenarios propagate shocks through the network. The CVaR optimizer and investment signal engine "
        "translate risk into explicit portfolio recommendations. All outputs are simulated and for research purposes.</p>"
        + "<div class='page-links'>" + about_links + "</div>"
    )
    pages["about"] = page_template("about", "Canadian Bank Contagion Command Center",
                                   "Financial-engineering research — bank network risk, CVaR optimization, and portfolio intelligence.", about_body, latest_date)

    market_body = (
        regime_html
        + metric_grid([
            ("Contagion Score", f"{score:.1f}/100"),
            ("XFN 21D Return", pct(latest(features, "XFN.TO_ret_21d"))),
            ("Avg Bank Vol", avg_bank_vol_str),
            ("Bank Correlation", avg_corr_str),
          ])
        + card(f"Current Readout: {regime['label']} Risk", regime["summary"], regime["tone"])
        + "<div class='chart-grid'><div class='chart-card'>" + chart_html(score_chart(features), True) + "</div><div class='chart-card'>" + chart_html(driver_chart(features)) + "</div></div>"
        + "<h2>Current Stress Drivers</h2>"
        + "<p>Each driver is percentile-ranked against its own full history. Elevated percentile = unusually stressed today.</p>"
        + table_html(driver_display[["Driver", "Latest", "Stress Percentile", "Status", "Why it matters"]].head(8))
        + "<h2>Economic Backdrop</h2><div class='chart-grid'><div class='chart-card'>" + chart_html(price_context_chart(prices)) + "</div><div class='chart-card'>" + chart_html(macro_chart(macro)) + "</div></div>"
    )
    pages["market-overview"] = page_template("market-overview", "Executive Market Overview",
                                              "Today's Canadian bank risk regime — translated into business decisions.", market_body, latest_date)

    corr_mat = correlation_matrix(prices)
    avg_corr_net = corr_mat.where(~np.eye(len(BANKS), dtype=bool)).stack().mean() if len(corr_mat) else 0.0
    network_body = (
        regime_html
        + metric_grid([
            ("Average Correlation", f"{avg_corr_net:.2f}"),
            ("Most Stressed Bank", bank_table.iloc[0]["Bank"]),
            ("Correlation Window", "63D rolling"),
            ("Systemic Purpose", "Diversification audit"),
          ])
        + card("How to Read the Network",
               "Redder, larger nodes carry more market stress. More edges = banks are moving together = lower genuine diversification. "
               "Highly central banks transmit stress even when they are not the most stressed themselves.",
               regime["tone"])
        + "<div class='chart-grid'><div class='chart-card'>" + chart_html(network_chart(prices, bank_table), True) + "</div><div class='chart-card'>" + chart_html(correlation_chart(prices)) + "</div></div>"
        + "<h2>Systemic Stress Ranking</h2>"
        + "<p>Node stress combines volatility, drawdown, and XFN beta into a 0–100 score. High node stress = trim or hedge first.</p>"
        + table_html(bank_display[["Bank", "Name", "21D Return", "Node Stress", "Action Readout", "Economic Lens"]])
    )
    pages["systemic-bank-network"] = page_template("systemic-bank-network", "Systemic Bank Network",
                                                    "Whether the Big Six are diversifying or moving as one crowded trade.", network_body, latest_date)

    contagion_body = (
        regime_html
        + metric_grid([
            ("Latest Score", f"{score:.1f}/100"),
            ("Regime", regime["label"]),
            ("Historical Percentile", f"{percentile_rank(features['contagion_risk_score'], score):.0%}"),
            ("Data Through", latest_date),
          ])
        + card(f"Signal: {regime['label']} Risk",
               f"{regime['summary']} At the {percentile_rank(features['contagion_risk_score'], score):.0%} historical percentile, "
               "this score is " + ("elevated relative to" if score > 50 else "below") + " most historical observations.",
               regime["tone"])
        + "<div class='chart-grid'><div class='chart-card'>" + chart_html(component_bar_chart(components), True) + "</div><div class='chart-card'>" + chart_html(component_heatmap(components)) + "</div></div>"
        + "<h2>Bank-Level Stress Contributors</h2>"
        + "<p>Node stress = volatility + drawdown + XFN beta, each percentile-ranked. High node stress = highest priority to reduce or hedge.</p>"
        + table_html(bank_display[["Bank", "Name", "21D Return", "21D Volatility", "63D Drawdown", "Node Stress", "Action Readout"]])
    )
    pages["contagion-risk-score"] = page_template("contagion-risk-score", "Contagion Risk Score",
                                                   "A 0–100 composite answer: is Canadian bank stress rising and spreading?", contagion_body, latest_date)

    cash_wt_str = weight_table.loc[weight_table["Asset"] == "cash", "Weight"].iloc[0] if "cash" in weight_table["Asset"].values else "N/A"
    stress_tone = "danger" if final_stress.mean() >= 60 else "warning" if final_stress.mean() >= 35 else "success"
    stress_body = (
        metric_grid([
            ("Scenario", "Liquidity Squeeze"),
            ("Avg Final Stress", f"{final_stress.mean():.1f}/100"),
            ("Most Stressed Bank", final_stress.idxmax()),
            ("Peak Stress", f"{final_stress.max():.1f}/100"),
          ])
        + card("Scenario Readout",
               f"Default Liquidity Squeeze shows stress peaking at {final_stress.max():.1f}/100. "
               "Use the interactive Streamlit app for custom scenario controls, severity sliders, and Monte Carlo loss distribution.",
               stress_tone)
        + "<div class='chart-grid'><div class='chart-card'>" + chart_html(stress_path_chart(paths), True) + "</div><div class='chart-card'>" + chart_html(final_stress_chart(final_stress)) + "</div></div>"
        + "<h2>Equal-Weight Portfolio Impact (Liquidity Squeeze)</h2>"
        + "<p>Estimated loss contribution assuming equal bank weights and the scenario's assumed drawdown per bank.</p>"
        + table_html(stress_impact)
    )
    pages["stress-testing-lab"] = page_template("stress-testing-lab", "Stress Testing Lab",
                                                 "Macro shock → bank stress propagation → portfolio loss attribution.", stress_body, latest_date)

    rl_body = (
        regime_html
        + metric_grid([
            ("Policy Type", "Stress-aware fallback"),
            ("Cash Weight", cash_wt_str),
            ("Regime", regime["label"]),
            ("Contagion Score", f"{score:.1f}/100"),
          ])
        + card("Allocation Rationale",
               "Cash rises when contagion risk is elevated. Bank exposure tilts away from higher-stress names proportionally. "
               "This is the transparent stress-aware policy used when a trained PPO model is unavailable.",
               regime["tone"])
        + "<div class='chart-grid'><div class='chart-card'>" + chart_html(allocation_fig, True) + "</div><div class='chart-card'>" + chart_html(bank_chart(bank_table)) + "</div></div>"
        + "<h2>Current Signal-Derived Allocation</h2>"
        + "<p>Weights are generated from the contagion score and bank-level stress. Equal-weight baseline is 16.7% per bank. "
        "Cash allocation expands as the score rises above 35.</p>"
        + table_html(weight_table)
    )
    pages["rl-portfolio-agent"] = page_template("rl-portfolio-agent", "RL Portfolio Agent",
                                                 "Translate bank contagion risk into allocation, cash, and defensive posture.", rl_body, latest_date)

    auc_tone = "success" if metrics.iloc[0]["AUC"] >= 0.65 else "warning"
    model_body = (
        metric_grid([
            ("Best Model", metrics.iloc[0]["Model"]),
            ("Best AUC", f"{metrics.iloc[0]['AUC']:.2f}"),
            ("Training Rows", f"{len(features):,}"),
            ("Validation", "Chronological split"),
          ])
        + card("Validation Signal",
               f"AUC {metrics.iloc[0]['AUC']:.2f} means the model has {'meaningful' if metrics.iloc[0]['AUC'] >= 0.65 else 'modest'} out-of-sample ranking signal "
               "for predicting whether stress will be in the top 20th percentile 5 days forward. "
               "AUC > 0.5 = better than random; AUC > 0.65 = substantively useful for tactical risk management.",
               auc_tone)
        + "<div class='chart-grid'><div class='chart-card'>" + chart_html(roc_chart(roc_df, metrics), True) + "</div><div class='chart-card'>" + chart_html(driver_chart(features)) + "</div></div>"
        + "<h2>Chronological Hold-Out Test Metrics</h2>"
        + "<p>Train/test split at 70%/30% in chronological order. No future data leaks into training.</p>"
        + table_html(metrics_display)
    )
    pages["model-validation"] = page_template("model-validation", "Model Validation",
                                               "Does the ML layer have genuine out-of-sample stress-prediction signal?", model_body, latest_date)

    performance_body = (
        metric_grid(
            [
                ("Starting Capital", "$100,000"),
                ("Current Paper Value", f"${paper_summary['ending_value']:,.0f}"),
                ("Cumulative Return", f"{paper_summary['cumulative_return']:.1%}"),
                ("Policy Source", paper_policy_source),
                ("Sharpe Ratio", f"{paper_summary['sharpe_ratio']:.2f}"),
                ("Max Drawdown", f"{paper_summary['max_drawdown']:.1%}"),
                ("Transaction Costs", f"${paper_summary['total_transaction_costs']:,.2f}"),
                ("Current Cash Weight", f"{paper_latest['cash_weight']:.1%}"),
            ]
        )
        + card(
            "Simulated paper portfolio - not real trading, not investment advice",
            "This page turns the daily allocation recommendation into a fake-money paper fund. It tracks shares, cash, trades, transaction costs, P&L, turnover, and benchmark comparisons. No broker connection exists and no real orders are placed.",
            "warning",
        )
        + "<div class='chart-grid'><div class='chart-card'>" + chart_html(performance_value_chart(paper_ledger, paper_benchmarks), True) + "</div><div class='chart-card'>" + chart_html(performance_allocation_chart(paper_weights)) + "</div></div>"
        + "<div class='chart-grid'><div class='chart-card'>" + chart_html(performance_drawdown_chart(paper_ledger, paper_benchmarks)) + "</div><div class='chart-card'>" + chart_html(cash_risk_chart(paper_ledger)) + "</div></div>"
        + "<h2>Current Holdings</h2>" + table_html(paper_holdings_display)
        + "<h2>Recent Simulated Trades</h2>" + table_html(paper_trades_display)
        + card(
            "Leakage Control",
            "Weights for each day are generated using observations available up to that day. Returns from one day to the next are earned by the holdings established on the prior day.",
        )
    )
    pages["performance-tracker"] = page_template("performance-tracker", "Performance Tracker", "A simulated paper fund that follows the model's daily allocation recommendations.", performance_body, latest_date)

    cvar_lab_body = (
        metric_grid(
            [
                ("Expected Return", f"{cvar_result.diagnostics['expected_return']:.1%}"),
                ("Annualized Vol", f"{cvar_result.diagnostics['annualized_volatility']:.1%}"),
                ("Historical CVaR", f"{cvar_result.diagnostics['historical_cvar']:.1%}"),
                ("Financial Exposure", f"{cvar_result.weights.reindex(BANKS + ['XFN.TO']).fillna(0).sum():.1%}"),
                ("Cash Weight", f"{cvar_result.weights.get('cash', 0):.1%}"),
                ("Graph Density", f"{cvar_result.diagnostics['graph_density']:.2f}"),
                ("Avg Correlation", f"{cvar_result.diagnostics['average_correlation']:.2f}"),
                ("Largest Eigenvalue", f"{cvar_result.diagnostics['largest_eigenvalue']:.2f}"),
            ]
        )
        + card(
            "Production-Style Allocation Engine",
            "This page is the governed allocator: shrinkage covariance, graph-adjusted systemic-risk inflation, CVaR tail-risk optimization, turnover penalties, and explicit portfolio constraints. RL remains an experimental research baseline.",
            "info",
        )
        + "<div class='chart-grid'><div class='chart-card'>" + chart_html(cvar_weight_chart(cvar_result.weights), True) + "</div><div class='chart-card'>" + chart_html(cvar_frontier_chart(cvar_frontier)) + "</div></div>"
        + "<div class='chart-grid'><div class='chart-card'>" + chart_html(cvar_risk_contribution_chart(cvar_result.risk_contributions)) + "</div><div class='chart-card'>" + chart_html(covariance_heatmap(cvar_result.adjusted_covariance, "Graph-Adjusted Covariance")) + "</div></div>"
        + "<h2>Risk Budget and Diagnostics</h2>" + table_html(cvar_weights_display)
        + "<h2>Centrality Penalties</h2>" + table_html(cvar_penalties)
        + card(
            "Methodology",
            "CVaR is expected shortfall: the average loss in the worst tail of historical portfolio returns. The objective balances expected return, CVaR, volatility, graph contagion exposure, and turnover subject to long-only, cash, single-name, and financial-exposure constraints.",
        )
    )
    pages["cvar-optimization-lab"] = page_template("cvar-optimization-lab", "CVaR Optimization Lab", "Graph-aware production-style portfolio construction for Canadian bank risk.", cvar_lab_body, latest_date)

    cvar_paper_body = (
        metric_grid(
            [
                ("Starting Capital", "$100,000"),
                ("Current Value", f"${cvar_summary['ending_value']:,.0f}"),
                ("Cumulative Return", f"{cvar_summary['cumulative_return']:.1%}"),
                ("Annualized Vol", f"{cvar_summary['annualized_volatility']:.1%}"),
                ("Sharpe", f"{cvar_summary['sharpe_ratio']:.2f}"),
                ("Realized CVaR", f"{cvar_summary['conditional_value_at_risk']:.1%}"),
                ("Max Drawdown", f"{cvar_summary['max_drawdown']:.1%}"),
                ("Transaction Costs", f"${cvar_summary['total_transaction_costs']:,.2f}"),
            ]
        )
        + card(
            "Simulated Paper Portfolio - Not Real Trading",
            "This paper fund rebalances through time using only information available at each rebalance date. It tracks shares, cash, holdings, daily P&L, turnover, transaction costs, realized CVaR, and benchmark comparisons.",
            "warning",
        )
        + "<div class='chart-grid'><div class='chart-card'>" + chart_html(cvar_value_chart(cvar_paper.ledger, cvar_benchmarks), True) + "</div><div class='chart-card'>" + chart_html(performance_drawdown_chart(cvar_paper.ledger, cvar_benchmarks)) + "</div></div>"
        + "<div class='chart-grid'><div class='chart-card'>" + chart_html(performance_allocation_chart(cvar_paper.weights)) + "</div><div class='chart-card'>" + chart_html(cash_risk_chart(cvar_paper.ledger)) + "</div></div>"
        + "<h2>Current Holdings</h2>" + table_html(cvar_holdings_display)
        + "<h2>Recent Simulated Trades</h2>" + table_html(cvar_trades_display)
    )
    pages["cvar-paper-fund"] = page_template("cvar-paper-fund", "CVaR Paper Fund", "A simulated paper fund following the graph-adjusted CVaR optimizer.", cvar_paper_body, latest_date)

    comparison_body = (
        metric_grid(
            [
                ("CVaR Ending Value", f"${cvar_summary['ending_value']:,.0f}"),
                ("RL Ending Value", f"${rl_summary['ending_value']:,.0f}"),
                ("CVaR Max Drawdown", f"{cvar_summary['max_drawdown']:.1%}"),
                ("RL Max Drawdown", f"{rl_summary['max_drawdown']:.1%}"),
                ("CVaR CVaR", f"{cvar_summary['conditional_value_at_risk']:.1%}"),
                ("RL CVaR", f"{rl_summary['conditional_value_at_risk']:.1%}"),
                ("CVaR Turnover", f"{cvar_summary['average_daily_turnover']:.1%}"),
                ("RL Turnover", f"{rl_summary['average_daily_turnover']:.1%}"),
            ]
        )
        + card(
            "Research Interpretation",
            "RL can be adaptive and nonlinear, but it is harder to validate and govern. CVaR optimization is more transparent, directly controls tail risk, and maps cleanly to institutional constraints. This page compares both frameworks under the same capital and cost assumptions.",
        )
        + "<div class='chart-grid'><div class='chart-card'>" + chart_html(cvar_value_chart(cvar_paper.ledger, pd.DataFrame({'RL research baseline': rl_paper.ledger['portfolio_value']})), True) + "</div><div class='chart-card'>" + chart_html(exposure_chart(cvar_paper.ledger, rl_paper.ledger)) + "</div></div>"
        + "<h2>Strategy Metrics</h2>" + table_html(comparison)
    )
    pages["rl-vs-cvar-comparison"] = page_template("rl-vs-cvar-comparison", "RL vs CVaR Comparative Analytics", "A comparative quant research page for experimental RL and governed CVaR allocation.", comparison_body, latest_date)

    data_body = (
        metric_grid([("CSV Files Found", f"{len(inventory):,}"), ("Total Rows", f"{int(inventory['Rows'].fillna(0).sum()):,}"), ("Explained Files", f"{inventory['Explanation'].notna().sum():,}"), ("Latest Dataset", latest_date)])
        + card("Data Traceability", "Each CSV has a role, explanation, date range, and row count. The Streamlit Data Catalog adds per-file chart exploration; this production page keeps the public lineage visible.")
        + "<h2>CSV Inventory</h2>" + table_html(inventory[["CSV", "Rows", "Columns", "Date Range", "Role", "Explanation"]])
    )
    pages["data-catalog"] = page_template("data-catalog", "Data Catalog", "Every CSV explained, profiled, and connected to analytical context.", data_body, latest_date)

    return pages


def write_pages() -> None:
    if PUBLIC.exists():
        shutil.rmtree(PUBLIC)
    PUBLIC.mkdir(parents=True)
    pages = build_pages()
    for slug, html in pages.items():
        filename = "index.html" if slug == "about" else f"{slug}.html"
        (PUBLIC / filename).write_text(html, encoding="utf-8")
        (ROOT / filename).write_text(html, encoding="utf-8")
    ROOT_INDEX.write_text(pages["about"], encoding="utf-8")
    print(f"Wrote {len(pages)} pages to {PUBLIC} and root HTML files")


if __name__ == "__main__":
    write_pages()
