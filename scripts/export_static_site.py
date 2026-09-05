import json
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
from src.dashboard.investment_signals import (  # noqa: E402
    compute_bank_signals,
    compute_market_positioning,
    compute_portfolio_recommendations,
)
from src.features.stress_features import _pct_rank  # noqa: E402
from src.portfolio.cvar_optimizer import efficient_frontier, optimize_cvar_portfolio, optimizer_tables  # noqa: E402
from src.portfolio.paper_trader import CVaRPaperPortfolioSimulator, PaperPortfolioSimulator  # noqa: E402
from src.portfolio.performance_metrics import drawdown_series, performance_summary  # noqa: E402
from src.portfolio.portfolio_constraints import PortfolioConstraints  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "public"
ROOT_INDEX = ROOT / "index.html"

PAGES = [
    ("overview", "Overview"),
    ("risk", "Risk"),
    ("scenarios", "Scenarios"),
    ("models", "Models"),
    ("decision", "Decision"),
    ("performance", "Performance"),
    ("research", "Research"),
]

SCENARIO_SHOCKS = {
    "Housing Crisis": {"RY.TO": 35, "TD.TO": 35, "BMO.TO": 30, "BNS.TO": 30, "CM.TO": 45, "NA.TO": 32},
    "Oil Crash": {"RY.TO": 20, "TD.TO": 18, "BMO.TO": 26, "BNS.TO": 25, "CM.TO": 22, "NA.TO": 18},
    "Liquidity Squeeze": {"RY.TO": 40, "TD.TO": 38, "BMO.TO": 36, "BNS.TO": 36, "CM.TO": 38, "NA.TO": 34},
    "Yield Curve Inversion": {"RY.TO": 24, "TD.TO": 24, "BMO.TO": 22, "BNS.TO": 22, "CM.TO": 28, "NA.TO": 20},
    "Global Risk-Off": {"RY.TO": 32, "TD.TO": 32, "BMO.TO": 30, "BNS.TO": 31, "CM.TO": 33, "NA.TO": 29},
}


def chart_html(fig: go.Figure, include_js=False) -> str:
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs="cdn" if include_js else False,
        config={"displayModeBar": False, "responsive": True},
    )


def table_html(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    label: str | None = None,
) -> str:
    view = df[columns].copy() if columns else df.copy()
    region_label = label or ", ".join(str(column) for column in view.columns[:3])
    return (
        f"<div class='table-scroll' role='region' aria-label='Data table: {region_label}'>"
        + view.to_html(index=False, classes="data-table", escape=False)
        + "</div>"
    )


def card(title: str, body: str, tone: str = "info") -> str:
    return (
        f"<div class='callout {tone}' role='note'><p class='callout-title'>{title}</p>"
        f"<p>{body}</p></div>"
    )


def metric_grid(items: list[tuple[str, str]], accent: str = "blue") -> str:
    cells = "".join(
        f"<div class='metric {accent}'><span>{label}</span><strong>{value}</strong></div>"
        for label, value in items
    )
    return f"<div class='grid'>{cells}</div>"


def local_tabs(items: list[tuple[str, str]]) -> str:
    links = "".join(f"<a href='#{slug}'>{label}</a>" for slug, label in items)
    return f"<nav class='local-tabs' aria-label='Page sections'>{links}</nav>"


def chart_panel(title: str, description: str, chart: str) -> str:
    return (
        "<figure class='chart-card'>"
        f"<div class='chart-context'><h3>{title}</h3></div>"
        f"{chart}<figcaption>{description}</figcaption></figure>"
    )


def section_heading(slug: str, eyebrow: str, title: str, description: str) -> str:
    return (
        f"<section class='section-heading' id='{slug}'><span>{eyebrow}</span>"
        f"<h2>{title}</h2><p>{description}</p></section>"
    )


def nav(active_slug: str) -> str:
    links = [
        "<a class='nav-brand' href='/' aria-label='Northern Signal overview'>"
        "<span aria-hidden='true'>&#9670;</span><span>Northern Signal</span></a>",
        "<button class='nav-toggle' type='button' aria-expanded='false' "
        "aria-controls='primary-links'><span class='sr-only'>Toggle navigation</span>"
        "<span aria-hidden='true'>Menu</span></button>",
        "<div class='nav-links' id='primary-links'>",
    ]
    for slug, label in PAGES:
        href = "/" if slug == "overview" else f"/{slug}"
        active = " active" if slug == active_slug else ""
        current = " aria-current='page'" if slug == active_slug else ""
        links.append(f"<a class='nav-link{active}' href='{href}'{current}>{label}</a>")
    links.append("</div>")
    return "<nav class='nav' aria-label='Primary navigation'>" + "".join(links) + "</nav>"


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
  <title>{title} | Northern Signal</title>
  <meta name="description" content="Canadian bank systemic risk, contagion scenarios, portfolio models, and governed investment decisions.">
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
      --border-accent: #486078;
      --ink:       #e8edf2;
      --muted:     #91a6b8;
      --blue:      #0969b8;
      --blue-lt:   #64b5f6;
      --green:     #00c853;
      --amber:     #ffb300;
      --red:       #ff6b63;
      --teal:      #00bcd4;
      --radius:    8px;
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
    .sr-only {{
      position: absolute;
      width: 1px;
      height: 1px;
      padding: 0;
      margin: -1px;
      overflow: hidden;
      clip: rect(0, 0, 0, 0);
      white-space: nowrap;
      border: 0;
    }}
    .skip-link {{
      position: fixed;
      top: 10px;
      left: 10px;
      z-index: 1000;
      transform: translateY(-160%);
      padding: 10px 14px;
      border-radius: 6px;
      background: #fff;
      color: #0f1923;
      font-weight: 700;
    }}
    .skip-link:focus {{ transform: translateY(0); }}
    :focus-visible {{
      outline: 3px solid #90caf9;
      outline-offset: 3px;
      border-radius: 4px;
    }}

    /* ── Navigation ── */
    .nav {{
      position: sticky;
      top: 0;
      z-index: 100;
      display: flex;
      gap: 20px;
      align-items: center;
      min-height: 64px;
      padding: 10px max(24px, 6vw);
      background: rgba(15, 25, 35, 0.92);
      border-bottom: 1px solid var(--border);
      backdrop-filter: blur(14px);
      -webkit-backdrop-filter: blur(14px);
    }}
    .nav-brand {{
      display: inline-flex;
      align-items: center;
      gap: 9px;
      font-weight: 700;
      font-size: 0.9rem;
      letter-spacing: 0.02em;
      color: var(--ink);
      text-decoration: none;
      margin-right: auto;
      white-space: nowrap;
    }}
    .nav-brand > span:first-child {{ color: var(--blue-lt); }}
    .nav-links {{ display: flex; align-items: center; gap: 4px; }}
    .nav-link {{
      color: var(--muted);
      text-decoration: none;
      border-radius: 5px;
      padding: 8px 10px;
      font-size: 0.82rem;
      transition: color 0.15s, border-color 0.15s, background 0.15s;
    }}
    .nav-link:hover {{
      color: var(--ink);
      background: rgba(30, 136, 229, 0.08);
    }}
    .nav-link.active {{
      background: rgba(66, 165, 245, 0.12);
      color: #90caf9;
      font-weight: 600;
    }}
    .nav-toggle {{
      display: none;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 7px 11px;
      background: var(--surface);
      color: var(--ink);
      font: inherit;
    }}

    /* ── Header ── */
    header {{
      padding: 58px max(24px, 6vw) 40px;
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      position: relative;
      overflow: hidden;
    }}
    .eyebrow {{
      display: block;
      margin-bottom: 10px;
      color: #90caf9;
      font-size: 0.72rem;
      font-weight: 700;
      letter-spacing: 0.13em;
      text-transform: uppercase;
    }}
    h1 {{
      font-size: 2.7rem;
      font-weight: 700;
      letter-spacing: 0;
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
      max-width: 1360px;
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
    .callout-title {{ color: var(--blue-lt); margin: 0 0 6px; font-size: 0.95rem; font-weight: 700; }}
    .callout p  {{ color: #c5d1db; font-size: 0.91rem; margin: 0; }}
    .callout.danger  {{ border-left-color: var(--red);   background: rgba(244, 67, 54, 0.07);  }}
    .callout.danger .callout-title  {{ color: #ffaaa5; }}
    .callout.warning {{ border-left-color: var(--amber); background: rgba(255, 179, 0, 0.07); }}
    .callout.warning .callout-title {{ color: #ffe082; }}
    .callout.success {{ border-left-color: var(--green); background: rgba(0, 200, 83, 0.07); }}
    .callout.success .callout-title {{ color: #a5d6a7; }}
    .callout.teal    {{ border-left-color: var(--teal);  background: rgba(0, 188, 212, 0.07); }}
    .callout.teal .callout-title    {{ color: #80deea; }}

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
    .chart-context {{ padding: 18px 20px 0; }}
    .chart-context h3 {{ margin: 0 0 4px; }}
    .chart-context p, figcaption {{
      color: var(--muted);
      font-size: 0.82rem;
    }}
    figcaption {{ padding: 0 20px 18px; }}

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
    .table-scroll {{ overflow-x: auto; -webkit-overflow-scrolling: touch; }}
    .table-scroll:focus-visible {{ outline-offset: -3px; }}

    /* ── Local navigation & page structure ── */
    .local-tabs {{
      position: sticky;
      top: 64px;
      z-index: 80;
      display: flex;
      gap: 6px;
      overflow-x: auto;
      margin: -32px -6vw 32px;
      padding: 12px 6vw;
      background: rgba(22, 32, 45, 0.96);
      border-bottom: 1px solid var(--border);
    }}
    .local-tabs a {{
      flex: 0 0 auto;
      padding: 7px 11px;
      border-radius: 5px;
      color: var(--muted);
      text-decoration: none;
      font-size: 0.82rem;
      font-weight: 600;
    }}
    .local-tabs a:hover {{ color: var(--ink); background: var(--card); }}
    .local-tabs a[aria-current='location'] {{ color: #90caf9; background: var(--card); }}
    .section-heading {{
      scroll-margin-top: 128px;
      margin: 56px 0 18px;
      max-width: 880px;
    }}
    .section-heading > span {{
      color: #90caf9;
      font-size: 0.7rem;
      font-weight: 700;
      letter-spacing: 0.12em;
      text-transform: uppercase;
    }}
    .section-heading h2 {{ margin: 5px 0 8px; }}
    .section-heading p {{ color: var(--muted); }}
    .executive-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1.35fr) minmax(280px, .65fr);
      gap: 20px;
      margin: 8px 0 30px;
    }}
    .hero-panel, .evidence-panel {{
      border: 1px solid var(--border);
      border-radius: var(--radius);
      background: var(--card);
      padding: 26px;
    }}
    .hero-score {{
      display: flex;
      align-items: baseline;
      gap: 10px;
      margin: 10px 0;
      font-family: 'JetBrains Mono', monospace;
      font-size: clamp(2.5rem, 8vw, 5rem);
      line-height: 1;
    }}
    .hero-score small {{ color: var(--muted); font-size: 1rem; }}
    .driver-list {{ margin: 16px 0 0; padding-left: 1.2rem; }}
    .driver-list li {{ padding: 7px 0; border-bottom: 1px solid rgba(42,58,74,.65); }}
    .driver-list li:last-child {{ border: 0; }}
    .button-row {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 22px; }}
    .button {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 42px;
      padding: 9px 15px;
      border: 1px solid var(--blue);
      border-radius: 6px;
      background: var(--blue);
      color: #fff;
      text-decoration: none;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }}
    .button.secondary {{ background: transparent; color: #90caf9; border-color: var(--border-accent); }}
    .journey {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 1px;
      border: 1px solid var(--border);
      border-radius: var(--radius);
      overflow: hidden;
      background: var(--border);
    }}
    .journey a {{ min-height: 138px; padding: 20px; background: var(--surface); color: var(--ink); text-decoration: none; }}
    .journey span {{ display: block; color: #90caf9; font: 500 .75rem 'JetBrains Mono', monospace; }}
    .journey strong {{ display: block; margin: 8px 0 5px; }}
    .journey small {{ color: var(--muted); }}
    details {{
      margin: 18px 0;
      border: 1px solid var(--border);
      border-radius: var(--radius);
      background: var(--surface);
    }}
    summary {{ cursor: pointer; padding: 14px 17px; color: var(--ink); font-weight: 650; }}
    details > div {{ padding: 0 17px 17px; }}
    .label {{
      display: inline-block;
      padding: 3px 8px;
      border: 1px solid var(--border);
      border-radius: 4px;
      color: var(--muted);
      font-size: .7rem;
      font-weight: 700;
      letter-spacing: .06em;
      text-transform: uppercase;
    }}
    .control-panel {{
      display: grid;
      grid-template-columns: minmax(220px, 1fr) minmax(220px, 1fr) auto;
      gap: 16px;
      align-items: end;
      margin: 20px 0;
      padding: 20px;
      border: 1px solid var(--border);
      border-radius: var(--radius);
      background: var(--card);
    }}
    .field label {{ display: block; margin-bottom: 7px; color: var(--ink); font-weight: 650; }}
    .field small {{ display: block; margin-top: 6px; color: var(--muted); }}
    select, input[type='range'] {{
      width: 100%;
      accent-color: var(--blue-lt);
    }}
    select {{
      min-height: 42px;
      padding: 8px 10px;
      border: 1px solid var(--border-accent);
      border-radius: 6px;
      background: var(--surface);
      color: var(--ink);
      font: inherit;
    }}
    .status-line {{ min-height: 24px; color: var(--muted); font-size: .86rem; }}

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
      .grid, .chart-grid, .page-links, .executive-grid {{ grid-template-columns: 1fr; }}
      .control-panel {{ grid-template-columns: 1fr; align-items: stretch; }}
      .journey {{ grid-template-columns: 1fr 1fr; }}
      header, main {{ padding-left: 20px; padding-right: 20px; }}
      .nav {{ padding: 10px 20px; }}
      .nav-toggle {{ display: inline-flex; }}
      .nav-links {{
        display: none;
        position: absolute;
        top: 63px;
        left: 0;
        right: 0;
        flex-direction: column;
        align-items: stretch;
        gap: 2px;
        padding: 10px 20px 18px;
        background: #0f1923;
        border-bottom: 1px solid var(--border);
      }}
      .nav-links.open {{ display: flex; }}
      .nav-link {{ padding: 11px; }}
      .local-tabs {{ top: 64px; margin-left: -20px; margin-right: -20px; padding-left: 20px; padding-right: 20px; }}
      h1 {{ font-size: 1.8rem; }}
    }}
    @media (max-width: 600px) {{
      .grid {{ grid-template-columns: repeat(2, 1fr); }}
      .journey {{ grid-template-columns: 1fr; }}
      .hero-panel, .evidence-panel {{ padding: 20px; }}
      .metric {{ padding: 14px 12px; }}
      .metric strong {{ font-size: 1.12rem; overflow-wrap: anywhere; }}
    }}
    @media (prefers-reduced-motion: reduce) {{
      html {{ scroll-behavior: auto; }}
      *, *::before, *::after {{ scroll-behavior: auto !important; transition: none !important; }}
    }}
  </style>
</head>
<body>
  <a class="skip-link" href="#main-content">Skip to main content</a>
  {nav(slug)}
  <header>
    <span class="eyebrow">Canadian systemic-risk research</span>
    <h1>{title}</h1>
    <p class="subtitle">{subtitle}</p>
    <span class="pill">&#9679; Data through {latest_date}</span>
    <span class="pill">Yahoo Finance · Bank of Canada</span>
    <span class="pill">CVaR · Graph Network · RL</span>
  </header>
  <main id="main-content" tabindex="-1">
    {body}
    <footer>
      <strong>Research-use notice.</strong> Analytical outputs are not personalized financial advice,
      a trading system, or a regulatory bank risk model. Performance views are historical simulations
      or paper portfolios; no real trades are placed. <a href="/research#limitations">Read limitations</a>.
    </footer>
  </main>
  <script>
    const toggle = document.querySelector('.nav-toggle');
    const links = document.querySelector('.nav-links');
    if (toggle && links) {{
      toggle.addEventListener('click', () => {{
        const open = toggle.getAttribute('aria-expanded') === 'true';
        toggle.setAttribute('aria-expanded', String(!open));
        links.classList.toggle('open', !open);
      }});
      links.addEventListener('click', () => {{
        toggle.setAttribute('aria-expanded', 'false');
        links.classList.remove('open');
      }});
    }}
    document.querySelectorAll('.table-scroll').forEach((wrapper) => {{
      if (wrapper.scrollWidth > wrapper.clientWidth) wrapper.setAttribute('tabindex', '0');
    }});
    const sectionLinks = [...document.querySelectorAll('.local-tabs a')];
    const markSection = () => {{
      const activeHash = window.location.hash || sectionLinks[0]?.hash;
      sectionLinks.forEach((link) => {{
        if (link.hash === activeHash) link.setAttribute('aria-current', 'location');
        else link.removeAttribute('aria-current');
      }});
    }};
    markSection();
    window.addEventListener('hashchange', markSection);
  </script>
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
    fig.add_hrect(y0=80, y1=90, fillcolor="#f44336", opacity=0.08, line_width=0)
    fig.add_hrect(y0=90, y1=100, fillcolor="#b71c1c", opacity=0.10, line_width=0)
    fig.add_hline(y=30, line_dash="dot", line_color="#00c853", opacity=0.4)
    fig.add_hline(y=60, line_dash="dot", line_color="#ffb300", opacity=0.4)
    fig.add_hline(y=80, line_dash="dot", line_color="#f44336", opacity=0.4)
    fig.add_hline(y=90, line_dash="dot", line_color="#b71c1c", opacity=0.4)
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


def investment_score_chart(signals: pd.DataFrame) -> go.Figure:
    ordered = signals.sort_values("Composite Score")
    fig = go.Figure(go.Bar(
        x=ordered["Composite Score"],
        y=ordered["Bank"],
        orientation="h",
        text=[f"{x:.1f}" for x in ordered["Composite Score"]],
        textposition="auto",
        marker=dict(
            color=ordered["Composite Score"].tolist(),
            colorscale=[[0, "#f44336"], [0.5, "#ffb300"], [1.0, "#00c853"]],
            cmin=0,
            cmax=100,
            showscale=False,
        ),
    ))
    fig.add_vline(x=50, line_dash="dash", line_color="#7a91a6")
    fig.update_layout(title="Investment Signal Score by Bank", xaxis_title="Composite score")
    return _dark_chart(fig)


def investment_weight_chart(signals: pd.DataFrame) -> go.Figure:
    ordered = signals.sort_values("Target Weight")
    colors = [
        "#00c853" if signal == "BUY" else "#f44336" if signal == "REDUCE" else "#1e88e5"
        for signal in ordered["Signal"]
    ]
    fig = go.Figure(go.Bar(
        x=ordered["Target Weight"],
        y=ordered["Bank"],
        orientation="h",
        text=[f"{x:.1%}" for x in ordered["Target Weight"]],
        textposition="auto",
        marker_color=colors,
    ))
    fig.add_vline(x=1 / len(BANKS), line_dash="dash", line_color="#7a91a6")
    fig.update_layout(title="Signal-Derived Target Weights", xaxis_title="Weight", xaxis_tickformat=".0%")
    return _dark_chart(fig)


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


# Plain-language labels for the five inputs that actually build contagion_risk_score.
# Source of truth: src.features.stress_features.make_contagion_risk_score.
# Each entry is (label, what the input measures, a clause completing
# "... than on N% of days on record").
COMPOSITE_COMPONENT_LABELS = {
    "avg_bank_vol_21d": (
        "Bank volatility",
        "How sharply the Big Six share prices have been swinging over the past month.",
        "the Big Six have been swinging more violently",
    ),
    "avg_pairwise_corr_63d": (
        "Bank correlation",
        "How closely the six banks have been moving together over the past quarter.",
        "the six banks have been moving together more tightly",
    ),
    "XFN.TO_drawdown_63d": (
        "Financials drawdown",
        "How far the Canadian financials ETF sits below its own recent high.",
        "the financials ETF has been sitting further below its recent high",
    ),
    "VIX_level": (
        "Global volatility",
        "How nervous global equity markets are, measured by the VIX.",
        "global equity markets have been more nervous",
    ),
    "slope_10y_2y": (
        "Yield-curve inversion",
        "How far the 10-year yield sits below the 2-year, which squeezes lending margins.",
        "the yield curve has been more inverted",
    ),
}

COMPOSITE_SCORE_LABEL = "Composite score"


def _ordinal(value: float) -> str:
    n = int(round(value))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def composite_attribution(features: pd.DataFrame) -> pd.DataFrame:
    """Decompose the published contagion score into the parts that build it.

    This mirrors ``make_contagion_risk_score`` exactly - an equal-weighted mean of
    expanding percentile ranks - and recomputes nothing else. It is a read of the
    existing calculation, not a second definition of it.
    """
    ranks: dict[str, pd.Series] = {}
    for col in ["avg_bank_vol_21d", "avg_pairwise_corr_63d", "XFN.TO_drawdown_63d", "VIX_level"]:
        if col in features:
            ranks[col] = _pct_rank(features[col].abs() if "drawdown" in col else features[col])
    if "slope_10y_2y" in features:
        ranks["slope_10y_2y"] = _pct_rank((-features["slope_10y_2y"]).clip(lower=0))
    if not ranks:
        return pd.DataFrame(columns=["Component", "Percentile", "Weight", "Contribution", "Meaning", "Reading"])

    weight = 1.0 / len(ranks)
    rows = []
    for col, series in ranks.items():
        label, meaning, reading = COMPOSITE_COMPONENT_LABELS.get(col, (col, "", "this input has been more elevated"))
        percentile = float(series.iloc[-1] * 100)
        rows.append(
            {
                "Component": label,
                "Percentile": percentile,
                "Weight": weight,
                "Contribution": percentile * weight,
                "Meaning": meaning,
                "Reading": reading,
            }
        )
    table = pd.DataFrame(rows).sort_values("Contribution", ascending=False).reset_index(drop=True)

    # Guard against silent drift: if the scoring formula ever changes, this
    # attribution stops being a true decomposition and must be updated with it.
    published = float(features["contagion_risk_score"].dropna().iloc[-1])
    residual = abs(table["Contribution"].sum() - published)
    if residual > 0.1:
        raise RuntimeError(
            "Composite attribution no longer reconciles to contagion_risk_score "
            f"(residual {residual:.4f}). make_contagion_risk_score has changed - "
            "update COMPOSITE_COMPONENT_LABELS and composite_attribution to match."
        )
    return table


def attribution_chart(attribution: pd.DataFrame, score: float) -> go.Figure:
    """Horizontal contribution bars, largest driver first, composite as the final bar."""
    ordered = attribution.sort_values("Contribution", ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ordered["Contribution"], y=ordered["Component"], orientation="h",
        text=[f"{x:.1f}" for x in ordered["Contribution"]], textposition="auto",
        customdata=np.stack([ordered["Percentile"], ordered["Weight"] * 100], axis=-1),
        hovertemplate=(
            "<b>%{y}</b><br>Stress percentile: %{customdata[0]:.1f}"
            "<br>Weight: %{customdata[1]:.0f}%"
            "<br>Contribution: %{x:.1f} points<extra></extra>"
        ),
        marker=dict(
            color=ordered["Percentile"].tolist(),
            colorscale=[[0, "#00c853"], [0.5, "#ffb300"], [1.0, "#f44336"]],
            cmin=0, cmax=100, showscale=False,
            line=dict(color="#16202d", width=2),
        ),
        showlegend=False,
    ))
    fig.add_trace(go.Bar(
        x=[score], y=[COMPOSITE_SCORE_LABEL], orientation="h",
        text=[f"{score:.1f}"], textposition="auto",
        hovertemplate=f"<b>{COMPOSITE_SCORE_LABEL}</b><br>%{{x:.1f}} of 100<extra></extra>",
        marker=dict(color="#1e88e5", line=dict(color="#16202d", width=2)),
        showlegend=False,
    ))
    fig.update_layout(
        title="Score Attribution — Weighted Contribution to the Composite",
        xaxis_title="Points contributed to the 0–100 score",
        barmode="overlay",
        bargap=0.35,
    )
    fig.update_yaxes(categoryorder="array",
                     categoryarray=[COMPOSITE_SCORE_LABEL] + ordered["Component"].tolist())
    return _dark_chart(fig, height=430)


def stress_paths(
    prices: pd.DataFrame,
    scenario_name="Liquidity Squeeze",
    severity: float = 1.0,
) -> tuple[pd.DataFrame, pd.Series]:
    returns = prices[BANKS].pct_change().tail(126)
    corr = returns.corr().fillna(0).clip(lower=0)
    values = corr.to_numpy(copy=True)
    np.fill_diagonal(values, 0)
    adj = pd.DataFrame(values, index=corr.index, columns=corr.columns)
    adj = adj.div(adj.sum(axis=1).replace(0, 1), axis=0)
    stress = (severity * pd.Series(SCENARIO_SHOCKS[scenario_name], dtype=float)).clip(0, 100)
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
    attribution = composite_attribution(features)
    allocation_fig, weights = allocation_chart(bank_table, score)
    regime_html = regime_banner_html(score, regime["label"], regime["summary"], regime["tone"])
    signals = compute_bank_signals(features, prices, macro)
    positioning = compute_market_positioning(features, macro, score)
    recs = compute_portfolio_recommendations(signals, score)

    bank_display = bank_table[["Bank", "Name", "21D Return", "21D Volatility", "63D Drawdown", "Beta to XFN", "Node Stress", "Risk response", "Economic Lens"]].copy()
    for col in ["21D Return", "21D Volatility", "63D Drawdown"]:
        bank_display[col] = bank_display[col].map(lambda x: pct(x) if pd.notna(x) else "N/A")
    bank_display["Beta to XFN"] = bank_display["Beta to XFN"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    bank_display["Node Stress"] = bank_display["Node Stress"].map(lambda x: f"{x:.1f}/100")

    driver_display = drivers.copy()
    driver_display["Stress Percentile"] = driver_display["Stress Percentile"].map(lambda x: f"{x:.0%}" if pd.notna(x) else "N/A")

    attribution_display = attribution.copy()
    attribution_display["Stress percentile"] = attribution_display["Percentile"].map(lambda x: f"{x:.1f}")
    attribution_display["Weight"] = attribution_display["Weight"].map(lambda x: f"{x:.0%}")
    attribution_display["Contribution"] = attribution_display["Contribution"].map(lambda x: f"{x:.1f} pts")
    attribution_display = attribution_display[["Component", "Stress percentile", "Weight", "Contribution", "Meaning"]]
    if len(attribution):
        top_driver = attribution.iloc[0]
        attribution_sentence = (
            f"<strong>{top_driver['Component']}</strong> is the largest single driver of today's score: at the "
            f"{_ordinal(top_driver['Percentile'])} percentile of its own history, {top_driver['Reading']} than on about "
            f"{top_driver['Percentile']:.0f}% of days on record, contributing "
            f"{top_driver['Contribution']:.1f} of the {score:.1f} total points."
        )
    else:
        attribution_sentence = "Component attribution is unavailable for the current dataset."

    weight_table = weights.rename("Weight").reset_index().rename(columns={"index": "Asset"})
    weight_table["Weight"] = weight_table["Weight"].map(lambda x: f"{x:.1%}")

    signals_display = signals[
        [
            "Bank",
            "Name",
            "Signal",
            "Conviction",
            "Composite Score",
            "Node Stress",
            "21D Return",
            "Target Weight",
            "Weight Delta",
            "Rationale",
            "Economic Context",
        ]
    ].copy()
    signals_display["Conviction"] = signals_display["Conviction"].map(lambda x: "★" * int(x) + "☆" * (5 - int(x)))
    for col in ["Composite Score", "Node Stress"]:
        signals_display[col] = signals_display[col].map(lambda x: f"{x:.1f}/100")
    signals_display["21D Return"] = signals_display["21D Return"].map(lambda x: f"{x:+.1%}" if pd.notna(x) else "N/A")
    for col in ["Target Weight", "Weight Delta"]:
        signals_display[col] = signals_display[col].map(lambda x: f"{x:+.1%}" if "Delta" in col else f"{x:.1%}")

    recs_display = recs.copy()
    for col in ["Current Weight", "Target Weight", "Delta"]:
        recs_display[col] = recs_display[col].map(lambda x: f"{x:+.1%}" if col == "Delta" else f"{x:.1%}")

    paths, final_stress = stress_paths(prices)
    stress_impact = pd.DataFrame({"Bank": BANKS, "Final Stress": [final_stress[b] for b in BANKS], "Aggregate Stress Share": [final_stress[b] / final_stress.sum() for b in BANKS]})
    stress_impact["Final Stress"] = stress_impact["Final Stress"].map(lambda x: f"{x:.1f}/100")
    stress_impact["Aggregate Stress Share"] = stress_impact["Aggregate Stress Share"].map(lambda x: f"{x:.1%}")
    stress_impact_html = (
        "<div class='table-scroll' role='region' aria-label='Data table: bank scenario stress'>"
        + stress_impact.to_html(
            index=False,
            classes="data-table scenario-impact-table",
            escape=False,
        )
        + "</div>"
    )

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
            ("investment-decision-center", "Investment Decision Center", "Final signal, rebalance, risk-budget, and stress-decision readout."),
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
    pages["about"] = page_template("about", "Northern Signal",
                                   "Financial-engineering research — bank network risk, CVaR optimization, and portfolio intelligence.", about_body, latest_date)

    market_body = (
        regime_html
        + metric_grid([
            ("Contagion Score", f"{score:.1f}/100"),
            ("XFN 21D Return", pct(latest(features, "XFN.TO_ret_21d"))),
            ("Avg Bank Vol", avg_bank_vol_str),
            ("Bank Correlation", avg_corr_str),
          ])
        + card(f"Current risk regime: {regime['label']}", regime["summary"], regime["tone"])
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
    network_rows = []
    pathway_rows = []
    for source in BANKS:
        peers = corr_mat.loc[source].drop(source).sort_values(ascending=False)
        material = peers[peers.abs() >= 0.35]
        network_rows.append(
            {
                "Bank": source,
                "Material links": int(len(material)),
                "Weighted centrality": f"{material.abs().sum() / max(len(BANKS) - 1, 1):.2f}",
                "Strongest pathway": f"{material.index[0]} ({material.iloc[0]:.2f})" if len(material) else "None above threshold",
                "Node stress": bank_display.loc[bank_display["Bank"] == source, "Node Stress"].iloc[0],
            }
        )
        for target, correlation in material.items():
            if source < target:
                pathway_rows.append(
                    {"Source": source, "Target": target, "63D correlation": f"{correlation:.2f}"}
                )
    network_evidence = pd.DataFrame(network_rows).sort_values("Weighted centrality", ascending=False)
    pathway_evidence = pd.DataFrame(
        pathway_rows,
        columns=["Source", "Target", "63D correlation"],
    ).sort_values("63D correlation", ascending=False).head(10)
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
        + table_html(bank_display[["Bank", "Name", "21D Return", "Node Stress", "Risk response", "Economic Lens"]])
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
        + table_html(bank_display[["Bank", "Name", "21D Return", "21D Volatility", "63D Drawdown", "Node Stress", "Risk response"]])
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
        + "<h2>Equal-Weight Stress Attribution (Liquidity Squeeze)</h2>"
        + "<p>Share of aggregate terminal stress under equal bank weights. This is not a forecast portfolio loss.</p>"
        + stress_impact_html
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

    buys = int((signals["Signal"] == "BUY").sum())
    reduces = int((signals["Signal"] == "REDUCE").sum())
    highest_stress_signal = signals.sort_values("Node Stress", ascending=False).iloc[0]
    strongest_signal = signals.sort_values("Composite Score", ascending=False).iloc[0]
    largest_rebalance = recs.iloc[recs["Delta"].abs().argmax()]
    cash_guidance_short = positioning["cash_guidance"].split(".")[0]
    decision_checks = pd.DataFrame(
        [
            {
                "Observation": f"{regime['label']} regime at {score:.1f}/100",
                "Decision Implication": positioning["sector_bias"],
                "Monitoring Trigger": "Change the bank budget if the score crosses the next regime band.",
            },
            {
                "Observation": f"Highest stress: {highest_stress_signal['Bank']} ({highest_stress_signal['Node Stress']:.1f}/100)",
                "Decision Implication": "First trim, hedge, or due-diligence candidate if the sector sells off.",
                "Monitoring Trigger": "Escalate if node stress stays above 70 or dominates scenario P&L.",
            },
            {
                "Observation": f"Strongest signal: {strongest_signal['Bank']} ({strongest_signal['Composite Score']:.1f}/100)",
                "Decision Implication": "Positive tilt is justified only when portfolio risk budget allows it.",
                "Monitoring Trigger": "Confirm score leadership survives the next rebalance window.",
            },
            {
                "Observation": f"Largest rebalance: {largest_rebalance['Bank']} {largest_rebalance['Delta']:+.1%}",
                "Decision Implication": largest_rebalance["Reason"],
                "Monitoring Trigger": "Act only after costs, liquidity, and mandate limits are checked.",
            },
        ]
    )
    decision_center_body = (
        regime_html
        + metric_grid(
            [
                ("Signal Mix", f"{buys} BUY / {len(BANKS) - buys - reduces} HOLD / {reduces} REDUCE"),
                ("Suggested Bank Budget", positioning["total_bank_budget"]),
                ("Cash Guidance", cash_guidance_short),
                ("CVaR Weight Cash", f"{cvar_result.weights.get('cash', 0):.1%}"),
                ("Top Signal", strongest_signal["Bank"]),
                ("Highest Stress", highest_stress_signal["Bank"]),
                ("Largest Rebalance", f"{largest_rebalance['Bank']} {largest_rebalance['Delta']:+.1%}"),
                ("Historical CVaR", f"{cvar_result.diagnostics['historical_cvar']:.1%}"),
            ]
        )
        + card(
            "Portfolio recommendation",
            f"The current signal engine recommends {buys} buys, {len(BANKS) - buys - reduces} holds, and {reduces} reductions. "
            f"The governed risk budget points to {positioning['total_bank_budget']} in bank exposure and "
            f"{cash_guidance_short.lower()}. This page connects signal strength, stress ranking, "
            "cash posture, and CVaR tail-loss control into one auditable decision surface.",
            regime["tone"],
        )
        + card(
            "Broader Market Read-Through",
            "Bank stress is a transmission signal for credit appetite, funding conditions, borrower resilience, and risk-capital availability. "
            "The practical decision is not just which ticker to own; it is how much balance-sheet and liquidity risk the portfolio should carry.",
            "teal",
        )
        + "<div class='chart-grid'><div class='chart-card'>" + chart_html(investment_score_chart(signals), True) + "</div><div class='chart-card'>" + chart_html(investment_weight_chart(signals)) + "</div></div>"
        + "<h2>Decision Checks</h2>" + table_html(decision_checks)
        + "<h2>Bank Signals</h2>" + table_html(signals_display)
        + "<h2>Rebalance Plan</h2>" + table_html(recs_display)
    )
    pages["investment-decision-center"] = page_template(
        "investment-decision-center",
        "Investment Decision Center",
        "Final risk-budget, bank-signal, rebalance, and tail-loss decision surface.",
        decision_center_body,
        latest_date,
    )

    data_body = (
        metric_grid([("CSV Files Found", f"{len(inventory):,}"), ("Total Rows", f"{int(inventory['Rows'].fillna(0).sum()):,}"), ("Explained Files", f"{inventory['Explanation'].notna().sum():,}"), ("Latest Dataset", latest_date)])
        + card("Data Traceability", "Each CSV has a role, explanation, date range, and row count. The Streamlit Data Catalog adds per-file chart exploration; this production page keeps the public lineage visible.")
        + "<h2>CSV Inventory</h2>" + table_html(inventory[["CSV", "Rows", "Columns", "Date Range", "Role", "Explanation"]])
    )
    pages["data-catalog"] = page_template("data-catalog", "Data Catalog", "Every CSV explained, profiled, and connected to analytical context.", data_body, latest_date)

    # Consolidate the analytical substance into seven decision-oriented routes.
    # Legacy route redirects are defined in vercel.json.
    top_driver_items = "".join(
        f"<li><strong>{row['Driver']}</strong> — {row['Stress Percentile']:.0%} historical stress percentile</li>"
        for _, row in drivers.head(3).iterrows()
    )
    model_confidence = "Moderate" if metrics.iloc[0]["AUC"] >= 0.65 else "Limited"
    fallback_cash = float(weights.get("cash", 0))
    cvar_cash = float(cvar_result.weights.get("cash", 0))
    model_agreement = "Aligned" if abs(fallback_cash - cvar_cash) <= 0.10 else "Mixed"

    overview_body = (
        "<div class='executive-grid'>"
        "<section class='hero-panel' aria-labelledby='current-state-title'>"
        "<span class='label'>Observed risk state</span>"
        "<h2 id='current-state-title' style='border:0;margin:14px 0 6px;padding:0'>"
        f"{regime['label']} risk regime</h2>"
        f"<div class='hero-score'>{score:.1f}<small>/ 100 risk score</small></div>"
        f"<p>{regime['summary']}</p>"
        f"<p><strong>Portfolio recommendation:</strong> use a {positioning['total_bank_budget']} bank-risk budget "
        f"with {positioning['cash_guidance'].split('.')[0].lower()}.</p>"
        "<div class='button-row'><a class='button' href='/decision'>Review portfolio decision</a>"
        "<a class='button secondary' href='/risk'>Inspect risk evidence</a></div></section>"
        "<aside class='evidence-panel' aria-labelledby='drivers-title'>"
        "<span class='label'>Why it is happening</span><h2 id='drivers-title' "
        "style='border:0;margin:14px 0 6px;padding:0'>Three material risk drivers</h2>"
        f"<ol class='driver-list'>{top_driver_items}</ol></aside></div>"
        + metric_grid(
            [
                ("Portfolio recommendation", positioning["total_bank_budget"] + " bank budget"),
                ("Cash guidance", positioning["cash_guidance"].split(".")[0]),
                ("Model confidence", f"{model_confidence} · AUC {metrics.iloc[0]['AUC']:.2f}"),
                ("Data updated", latest_date),
            ]
        )
        + card(
            "Portfolio recommendation",
            f"{positioning['sector_bias']} {positioning['cash_guidance']} "
            f"Cash posture is {model_agreement.lower()} across the transparent fallback and CVaR model outputs.",
            regime["tone"],
        )
        + section_heading(
            "journey",
            "Analytical journey",
            "From market conditions to a governed decision",
            "Each stage answers one question and hands its evidence to the next stage.",
        )
        + "<div class='journey'>"
        "<a href='/'><span>01</span><strong>Market conditions</strong><small>Regime, rates, volatility, correlation</small></a>"
        "<a href='/risk'><span>02</span><strong>Contagion assessment</strong><small>Network pathways and composite score</small></a>"
        "<a href='/scenarios'><span>03</span><strong>Stress scenarios</strong><small>Shock transmission and portfolio impact</small></a>"
        "<a href='/models'><span>04</span><strong>Portfolio response</strong><small>RL and CVaR model outputs</small></a>"
        "<a href='/decision'><span>05</span><strong>Final decision</strong><small>Risk budget and rebalance actions</small></a>"
        "</div>"
        + section_heading(
            "market-evidence",
            "Supporting evidence",
            "Market conditions and risk drivers",
            "Historical observations provide context; they are not forecasts.",
        )
        + "<div class='chart-grid'>"
        + chart_panel(
            "Risk score over time",
            "Composite systemic-risk score on a 0–100 scale; regime thresholds are shown as reference lines.",
            chart_html(score_chart(features), True),
        )
        + chart_panel(
            "Current driver percentiles",
            "Each driver is ranked against its own history; higher values indicate more unusual stress.",
            chart_html(driver_chart(features)),
        )
        + "</div>"
    )

    risk_body = (
        local_tabs([("network", "Network"), ("composite-score", "Composite score")])
        + section_heading(
            "network",
            "Risk evidence · Network",
            "Where stress can propagate",
            "Node size represents network centrality and color represents bank-level stress. "
            "Dense co-movement reduces the diversification available within the sector.",
        )
        + metric_grid(
            [
                ("Average correlation", f"{avg_corr_net:.2f}"),
                ("Most stressed bank", bank_table.iloc[0]["Bank"]),
                ("Measurement window", "63 trading days"),
                ("Interpretation", "Diversification audit"),
            ]
        )
        + "<div class='chart-grid'>"
        + chart_panel(
            "Canadian bank contagion network",
            "Larger institutions are more central transmission points; warmer colors indicate greater current node stress.",
            chart_html(network_chart(prices, bank_table), True),
        )
        + chart_panel(
            "Cross-bank correlation",
            "Pairwise return correlation over 63 trading days; values nearer 1 imply less independent risk.",
            chart_html(correlation_chart(prices)),
        )
        + "</div>"
        + "<h3>Centrality and material propagation pathways</h3>"
        + "<p>Material links use an absolute 63-day correlation threshold of 0.35. Weighted centrality is "
        "the average absolute strength of those links across the five possible peers.</p>"
        + table_html(network_evidence)
        + "<details><summary>Strongest pairwise pathways and bank context</summary><div>"
        + table_html(pathway_evidence)
        + table_html(bank_display[["Bank", "Name", "21D Return", "Node Stress", "Risk response", "Economic Lens"]])
        + "</div></details>"
        + section_heading(
            "composite-score",
            "Risk evidence · Composite score",
            "How the current score is formed",
            "The risk score combines market, volatility, correlation, macro, and bank-level signals on a consistent 0–100 scale.",
        )
        + regime_html
        + metric_grid(
            [
                ("Risk score", f"{score:.1f}/100"),
                ("Risk regime", regime["label"]),
                ("Historical percentile", f"{percentile_rank(features['contagion_risk_score'], score):.0%}"),
                ("Data updated", latest_date),
            ]
        )
        + "<h3>Score attribution</h3>"
        + "<p>The composite score is the equal-weighted average of five percentile ranks. Each input is "
        "ranked against its own history to date, then contributes one-fifth of its rank to the total. "
        "The bars below show how many of the "
        f"{score:.1f} points each input is responsible for.</p>"
        + chart_panel(
            "Contribution to the composite score",
            attribution_sentence,
            chart_html(attribution_chart(attribution, score)),
        )
        + table_html(attribution_display, label="Composite score attribution")
        + "<div class='chart-grid'>"
        + chart_panel(
            "Component scores",
            "Current component values on the same 0–100 stress scale; higher values are more adverse.",
            chart_html(component_bar_chart(components)),
        )
        + chart_panel(
            "Six-month stress breadth",
            "Historical component scores over approximately 126 trading days, separating current state from change over time.",
            chart_html(component_heatmap(components)),
        )
        + "</div>"
        + "<details><summary>Bank-level node stress and threshold interpretation</summary><div>"
        + "<p>Low: 0–30; moderate: 30–60; elevated: 60–80; high: 80–90; severe: 90–100. "
        "Node stress is supporting evidence—not a direct additive contribution to the composite score—and "
        "combines volatility, drawdown, and sector beta percentile ranks.</p>"
        + table_html(bank_display[["Bank", "Name", "21D Volatility", "63D Drawdown", "Node Stress", "Risk response"]])
        + "</div></details>"
        + "<div class='button-row'><a class='button' href='/decision'>Use this evidence in the decision</a></div>"
    )

    scenario_returns = prices[BANKS].pct_change().tail(126)
    scenario_corr = scenario_returns.corr().fillna(0).clip(lower=0)
    scenario_values = scenario_corr.to_numpy(copy=True)
    np.fill_diagonal(scenario_values, 0)
    scenario_adj = pd.DataFrame(
        scenario_values,
        index=scenario_corr.index,
        columns=scenario_corr.columns,
    ).div(
        pd.DataFrame(scenario_values, index=scenario_corr.index, columns=scenario_corr.columns)
        .sum(axis=1)
        .replace(0, 1),
        axis=0,
    )
    scenario_payload = json.dumps(
        {
            "banks": BANKS,
            "shocks": SCENARIO_SHOCKS,
            "adjacency": scenario_adj.reindex(index=BANKS, columns=BANKS).fillna(0).values.tolist(),
        }
    ).replace("</", "<\\/")
    scenario_presets = pd.DataFrame(
        [
            {"Preset": "Housing crisis", "Primary assumption": "Mortgage and housing-credit stress", "Transmission focus": "CM, RY, TD"},
            {"Preset": "Oil crash", "Primary assumption": "Energy-credit and macro shock", "Transmission focus": "BMO, BNS"},
            {"Preset": "Liquidity squeeze", "Primary assumption": "Broad funding pressure", "Transmission focus": "All Big Six"},
            {"Preset": "Yield-curve inversion", "Primary assumption": "Margin and growth pressure", "Transmission focus": "Rate-sensitive banks"},
            {"Preset": "Global risk-off", "Primary assumption": "Synchronized market deleveraging", "Transmission focus": "Highly central banks"},
        ]
    )
    scenario_script = """
<script>
(() => {
  const scenarioData = __SCENARIO_DATA__;
  const form = document.querySelector('#scenario-controls');
  if (!form) return;
  const select = document.querySelector('#scenario-select');
  const severity = document.querySelector('#scenario-severity');
  const severityOutput = document.querySelector('#scenario-severity-output');
  const status = document.querySelector('#scenario-status');
  const clip = (value) => Math.max(0, Math.min(100, value));

  const calculate = (name, multiplier) => {
    let current = scenarioData.banks.map((bank) => clip(scenarioData.shocks[name][bank] * multiplier));
    const paths = [current.slice()];
    for (let step = 1; step <= 5; step += 1) {
      current = current.map((value, target) => {
        const propagated = current.reduce(
          (total, sourceValue, source) => total + scenarioData.adjacency[source][target] * sourceValue,
          0,
        );
        return clip(0.70 * value + 0.45 * propagated);
      });
      paths.push(current.slice());
    }
    return paths;
  };

  const render = () => {
    status.textContent = 'Updating scenario…';
    status.setAttribute('aria-busy', 'true');
    requestAnimationFrame(() => {
      try {
        const multiplier = Number(severity.value) / 100;
        const paths = calculate(select.value, multiplier);
        const finalValues = paths[paths.length - 1];
        const average = finalValues.reduce((sum, value) => sum + value, 0) / finalValues.length;
        const peak = Math.max(...finalValues);
        const peakIndex = finalValues.indexOf(peak);
        const initial = paths[0];
        severityOutput.value = `${severity.value}%`;
        document.querySelector('#scenario-name').textContent = select.options[select.selectedIndex].text;
        document.querySelector('#scenario-start-range').textContent =
          `${Math.min(...initial).toFixed(0)}–${Math.max(...initial).toFixed(0)} / 100`;
        document.querySelector('#scenario-average').textContent = `${average.toFixed(1)}/100`;
        document.querySelector('#scenario-peak').textContent = `${peak.toFixed(1)}/100`;
        document.querySelector('#scenario-bank').textContent = scenarioData.banks[peakIndex];
        document.querySelector('#scenario-response-text').textContent =
          `Prioritize due diligence or reduction in ${scenarioData.banks[peakIndex]}, preserve liquidity, and reconcile the response with the governed bank-risk budget.`;

        const pathPlot = document.querySelector('#scenario-path-panel .js-plotly-plot');
        const finalPlot = document.querySelector('#scenario-final-panel .js-plotly-plot');
        if (!window.Plotly || !pathPlot || !finalPlot) throw new Error('Scenario charts are unavailable.');
        const pathTraces = pathPlot.data.map((trace, index) => ({
          ...trace,
          x: [0, 1, 2, 3, 4, 5],
          y: paths.map((row) => row[index]),
        }));
        Plotly.react(pathPlot, pathTraces, pathPlot.layout, {displayModeBar: false, responsive: true});
        const ranking = scenarioData.banks
          .map((bank, index) => ({bank, value: finalValues[index]}))
          .sort((a, b) => a.value - b.value);
        const finalTrace = {
          ...finalPlot.data[0],
          x: ranking.map((item) => item.value),
          y: ranking.map((item) => item.bank),
          text: ranking.map((item) => `${item.value.toFixed(1)}`),
          marker: {...finalPlot.data[0].marker, color: ranking.map((item) => item.value)},
        };
        Plotly.react(finalPlot, [finalTrace], finalPlot.layout, {displayModeBar: false, responsive: true});

        const total = finalValues.reduce((sum, value) => sum + value, 0);
        document.querySelectorAll('.scenario-impact-table tbody tr').forEach((row, index) => {
          row.cells[1].textContent = `${finalValues[index].toFixed(1)}/100`;
          row.cells[2].textContent = `${(100 * finalValues[index] / total).toFixed(1)}%`;
        });
        status.textContent = `Updated ${select.options[select.selectedIndex].text} at ${severity.value}% severity.`;
      } catch (error) {
        status.textContent = `Unable to update the scenario: ${error.message}`;
      } finally {
        status.removeAttribute('aria-busy');
      }
    });
  };

  select.addEventListener('change', render);
  severity.addEventListener('input', render);
  form.addEventListener('reset', () => setTimeout(render, 0));
  form.addEventListener('submit', (event) => event.preventDefault());
})();
</script>
""".replace("__SCENARIO_DATA__", scenario_payload)
    scenario_body = (
        local_tabs(
            [
                ("assumptions", "Assumptions"),
                ("transmission", "Transmission"),
                ("portfolio-impact", "Portfolio impact"),
                ("risk-response", "Risk response"),
            ]
        )
        + section_heading(
            "assumptions",
            "Scenario · Inputs",
            "Explore governed scenario presets",
            "Select a documented shock and scale its severity. The liquidity squeeze at 100% is the default; Reset returns to it.",
        )
        + "<form id='scenario-controls' class='control-panel'>"
        "<div class='field'><label for='scenario-select'>Scenario preset</label>"
        "<select id='scenario-select' name='scenario'>"
        "<option value='Housing Crisis'>Housing crisis</option>"
        "<option value='Oil Crash'>Oil crash</option>"
        "<option value='Liquidity Squeeze' selected>Liquidity squeeze</option>"
        "<option value='Yield Curve Inversion'>Yield-curve inversion</option>"
        "<option value='Global Risk-Off'>Global risk-off</option>"
        "</select><small>Changes the initial bank-level shock assumptions.</small></div>"
        "<div class='field'><label for='scenario-severity'>Severity: "
        "<output id='scenario-severity-output' for='scenario-severity'>100%</output></label>"
        "<input id='scenario-severity' name='severity' type='range' min='50' max='150' step='10' value='100'>"
        "<small>Scales initial shocks from 50% to 150%; propagated stress is capped at 100.</small></div>"
        "<button class='button secondary' type='reset'>Reset scenario</button></form>"
        "<p id='scenario-status' class='status-line' role='status' aria-live='polite'>"
        "Liquidity squeeze loaded at 100% severity.</p>"
        + metric_grid(
            [
                ("Selected scenario", "<span id='scenario-name'>Liquidity squeeze</span>"),
                ("Propagation steps", "5"),
                ("Starting shocks", "<span id='scenario-start-range'>34–40 / 100</span>"),
                ("Network basis", "126-day positive correlation"),
            ]
        )
        + table_html(scenario_presets)
        + card(
            "Scenario assumption",
            "This is a hypothetical stress test, not a forecast. Initial bank shocks are propagated through a normalized "
            "positive-correlation network with persistence and spillover terms.",
            "warning",
        )
        + section_heading(
            "transmission",
            "Scenario · Model output",
            "Bank-level transmission through the network",
            "The path view separates the initial user assumption from subsequent model-derived propagation.",
        )
        + "<div class='chart-grid'>"
        + "<div id='scenario-path-panel'>"
        + chart_panel(
            "Propagation path",
            "Stress score by bank across five propagation steps under the selected preset and severity.",
            chart_html(stress_path_chart(paths), True),
        )
        + "</div><div id='scenario-final-panel'>"
        + chart_panel(
            "Final bank stress",
            "Terminal scenario stress ranks the institutions most exposed after network propagation.",
            chart_html(final_stress_chart(final_stress)),
        )
        + "</div></div>"
        + section_heading(
            "portfolio-impact",
            "Scenario · Portfolio impact",
            "How aggregate terminal stress is distributed",
            "Stress share is an attribution aid under equal bank weights, not a forecast portfolio loss.",
        )
        + metric_grid(
            [
                ("Average final stress", f"<span id='scenario-average'>{final_stress.mean():.1f}/100</span>"),
                ("Peak final stress", f"<span id='scenario-peak'>{final_stress.max():.1f}/100</span>"),
                ("Most stressed bank", f"<span id='scenario-bank'>{final_stress.idxmax()}</span>"),
                ("Portfolio basis", "Equal bank weights"),
            ]
        )
        + stress_impact_html
        + section_heading(
            "risk-response",
            "Scenario · Portfolio recommendation",
            "Translate the scenario into a controlled response",
            "Use scenario evidence to prioritize reductions, then reconcile the response with the CVaR risk budget and mandate constraints.",
        )
        + f"<div class='callout {stress_tone}' role='note'><p class='callout-title'>Resulting risk response</p>"
        f"<p id='scenario-response-text'>Prioritize due diligence or reduction in {final_stress.idxmax()}, "
        f"preserve liquidity, and compare any proposed bank exposure with the current "
        f"{positioning['total_bank_budget']} bank-risk budget.</p></div>"
        + "<div class='button-row'><a class='button' href='/decision'>Review decision under this scenario</a>"
        "<a class='button secondary' href='/models#cvar-strategy'>Inspect portfolio model response</a></div>"
        + scenario_script
    )

    model_comparison = pd.DataFrame(
        [
            {
                "Dimension": "Objective",
                "RL strategy": "Learn a nonlinear state-to-allocation policy",
                "CVaR strategy": "Balance expected return, expected shortfall, volatility, contagion, and turnover",
            },
            {
                "Dimension": "Inputs",
                "RL strategy": "Market and systemic-risk state variables",
                "CVaR strategy": "Returns, shrinkage covariance, graph centrality, regime, prior weights",
            },
            {
                "Dimension": "Constraints",
                "RL strategy": "Environment action bounds and fallback policy limits",
                "CVaR strategy": "Long-only, cash band, name cap, financial-exposure cap",
            },
            {
                "Dimension": "Expected behavior",
                "RL strategy": "Adaptive and potentially nonlinear",
                "CVaR strategy": "Stable, auditable, and explicitly risk-budgeted",
            },
            {
                "Dimension": "Drawdown characteristics",
                "RL strategy": f"{rl_summary['max_drawdown']:.1%} simulated maximum drawdown",
                "CVaR strategy": f"{cvar_summary['max_drawdown']:.1%} simulated maximum drawdown",
            },
            {
                "Dimension": "Tail-risk characteristics",
                "RL strategy": f"{rl_summary['conditional_value_at_risk']:.1%} realized simulated CVaR",
                "CVaR strategy": f"{cvar_summary['conditional_value_at_risk']:.1%} realized simulated CVaR",
            },
            {
                "Dimension": "Turnover",
                "RL strategy": f"{rl_summary['average_daily_turnover']:.1%} average daily",
                "CVaR strategy": f"{cvar_summary['average_daily_turnover']:.1%} average daily",
            },
            {
                "Dimension": "Current role",
                "RL strategy": "Experimental research baseline",
                "CVaR strategy": "Governed portfolio recommendation engine",
            },
            {
                "Dimension": "Current recommendation",
                "RL strategy": f"Transparent fallback shows {fallback_cash:.1%} cash",
                "CVaR strategy": f"Constrained optimizer shows {cvar_cash:.1%} cash",
            },
            {
                "Dimension": "Prefer when",
                "RL strategy": "Testing nonlinear policy behavior under controlled research conditions",
                "CVaR strategy": "Explaining and enforcing tail-risk, liquidity, and concentration limits",
            },
        ]
    )
    models_body = (
        local_tabs(
            [
                ("rl-strategy", "RL strategy"),
                ("cvar-strategy", "CVaR strategy"),
                ("comparison", "Comparison"),
                ("validation", "Validation"),
            ]
        )
        + section_heading(
            "rl-strategy",
            "Models · Experimental",
            "RL strategy",
            "A nonlinear research baseline that maps systemic-risk state to defensive portfolio weights.",
        )
        + metric_grid(
            [
                ("Displayed policy", "Transparent fallback"),
                ("Current cash weight", f"{fallback_cash:.1%}"),
                ("Risk regime", regime["label"]),
                ("Risk score", f"{score:.1f}/100"),
            ]
        )
        + card(
            "Model output",
            "Cash rises as systemic risk increases, while bank exposure tilts away from higher-stress names. "
            "When a trained PPO policy is unavailable, the simulator uses the transparent stress-aware fallback.",
        )
        + "<div class='chart-grid'>"
        + chart_panel(
            "Current RL/fallback allocation",
            "Current asset weights generated from the risk score and bank node stress.",
            chart_html(allocation_fig, True),
        )
        + chart_panel(
            "Bank node stress",
            "The cross-sectional stress ranking used to reduce exposure to more vulnerable institutions.",
            chart_html(bank_chart(bank_table)),
        )
        + "</div><details><summary>Current model weights</summary><div>"
        + table_html(weight_table)
        + "</div></details>"
        + section_heading(
            "cvar-strategy",
            "Models · Governed",
            "CVaR strategy",
            "The primary portfolio construction model uses expected shortfall, graph-adjusted covariance, turnover penalties, and explicit constraints.",
        )
        + metric_grid(
            [
                ("Expected return", f"{cvar_result.diagnostics['expected_return']:.1%}"),
                ("Annualized volatility", f"{cvar_result.diagnostics['annualized_volatility']:.1%}"),
                ("Historical CVaR", f"{cvar_result.diagnostics['historical_cvar']:.1%}"),
                ("Cash weight", f"{cvar_cash:.1%}"),
            ]
        )
        + "<div class='chart-grid'>"
        + chart_panel(
            "CVaR allocation",
            "Constrained current weights, including explicit cash and sector exposure.",
            chart_html(cvar_weight_chart(cvar_result.weights)),
        )
        + chart_panel(
            "Risk-return frontier",
            "Feasible portfolios across return and expected-shortfall trade-offs under the same constraints.",
            chart_html(cvar_frontier_chart(cvar_frontier)),
        )
        + chart_panel(
            "Risk contribution",
            "Asset contribution to total portfolio risk; concentration is visible even when nominal weights look balanced.",
            chart_html(cvar_risk_contribution_chart(cvar_result.risk_contributions)),
        )
        + chart_panel(
            "Graph-adjusted covariance",
            "Covariance after systemic-network inflation; warmer cells represent stronger joint risk.",
            chart_html(covariance_heatmap(cvar_result.adjusted_covariance, "Graph-Adjusted Covariance")),
        )
        + "</div><details><summary>Risk budget, constraints, and centrality penalties</summary><div>"
        + table_html(cvar_weights_display)
        + table_html(cvar_penalties)
        + "</div></details>"
        + section_heading(
            "comparison",
            "Models · Comparison",
            "RL and CVaR answer different governance needs",
            "Observed simulated results are shown alongside qualitative differences. Neither model is presented as universally superior.",
        )
        + table_html(model_comparison)
        + "<div class='chart-grid'>"
        + chart_panel(
            "Simulated strategy value",
            "Both strategies use the same starting capital and cost framework; results are historical simulations.",
            chart_html(cvar_value_chart(cvar_paper.ledger, pd.DataFrame({"RL research baseline": rl_paper.ledger["portfolio_value"]}))),
        )
        + chart_panel(
            "Exposure behavior",
            "Financial exposure and cash weights show how each strategy changes posture through time.",
            chart_html(exposure_chart(cvar_paper.ledger, rl_paper.ledger)),
        )
        + "</div>"
        + section_heading(
            "validation",
            "Models · Validation",
            "Out-of-sample evidence and confidence",
            "A chronological hold-out test measures whether the supervised layer ranks future high-stress periods better than chance.",
        )
        + metric_grid(
            [
                ("Best model", metrics.iloc[0]["Model"]),
                ("Best AUC", f"{metrics.iloc[0]['AUC']:.2f}"),
                ("Confidence", model_confidence),
                ("Split", "70% train / 30% test"),
            ]
        )
        + "<div class='chart-grid'>"
        + chart_panel(
            "ROC curve",
            "True-positive versus false-positive rates on the chronological hold-out set; the diagonal is random ranking.",
            chart_html(roc_chart(roc_df, metrics)),
        )
        + chart_panel(
            "Current feature context",
            "Risk-driver percentiles provide an interpretable bridge from the validation target to current conditions.",
            chart_html(driver_chart(features)),
        )
        + "</div>"
        + table_html(metrics_display)
        + "<div class='button-row'><a class='button' href='/decision'>Apply validated evidence to the decision</a></div>"
    )

    latest_xfn_price = float(prices["XFN.TO"].dropna().iloc[-1]) if "XFN.TO" in prices and prices["XFN.TO"].notna().any() else np.nan
    latest_policy_rate = latest(macro, "policy_rate", np.nan)
    largest_increase = recs.loc[recs["Delta"].idxmax()]
    largest_reduction = recs.loc[recs["Delta"].idxmin()]
    increase_summary = (
        f"{largest_increase['Bank']} {largest_increase['Delta']:+.1%} · {largest_increase['Action']}"
    )
    reduction_summary = (
        f"{largest_reduction['Bank']} {largest_reduction['Delta']:+.1%} · {largest_reduction['Action']}"
    )
    facts_table = pd.DataFrame(
        [
            {"Observed fact": f"Market and macro observation window ends {latest_date}", "Source": "Processed point-in-time dataset"},
            {"Observed fact": f"Latest XFN series value is ${latest_xfn_price:,.2f}" if pd.notna(latest_xfn_price) else "Latest XFN series value is unavailable", "Source": "Yahoo Finance-compatible market series"},
            {"Observed fact": f"Latest policy rate is {latest_policy_rate:.2f}%" if pd.notna(latest_policy_rate) else "Latest policy rate is unavailable", "Source": "Bank of Canada-compatible macro series"},
        ]
    )
    calculated_indicators = pd.DataFrame(
        [
            {"Calculated indicator": f"Risk score {score:.1f}/100", "Method": "Composite market, macro, and bank stress model"},
            {"Calculated indicator": f"Highest node stress: {highest_stress_signal['Bank']} {highest_stress_signal['Node Stress']:.1f}/100", "Method": "Rolling volatility, drawdown, and sector beta"},
            {"Calculated indicator": f"Average cross-bank correlation {avg_corr_net:.2f}", "Method": "63-trading-day return correlation"},
        ]
    )
    cash_posture = pd.DataFrame(
        [
            {"Evidence layer": "Regime policy guidance", "Cash posture": positioning["cash_guidance"].split(".")[0], "Interpretation": "Primary decision guardrail tied to the current risk band"},
            {"Evidence layer": "Transparent fallback output", "Cash posture": f"{fallback_cash:.1%}", "Interpretation": "Heuristic response to score and node stress"},
            {"Evidence layer": "CVaR model output", "Cash posture": f"{cvar_cash:.1%}", "Interpretation": "Constrained optimizer result under current return and covariance inputs"},
            {"Evidence layer": "Decision synthesis", "Cash posture": positioning["cash_guidance"].split(".")[0], "Interpretation": f"Use the regime range pending review because cash-posture evidence is {model_agreement.lower()}"},
        ]
    )
    decision_constraints = pd.DataFrame(
        [
            {"Constraint": "Maximum Big Six single-name weight", "Limit": "20.0%", "Purpose": "Control institution-specific concentration"},
            {"Constraint": "Maximum total Canadian financial exposure", "Limit": "70.0%", "Purpose": "Cap Big Six plus XFN systemic exposure"},
            {"Constraint": "Cash range", "Limit": "5.0%–60.0%", "Purpose": "Preserve liquidity without allowing an unconstrained all-cash result"},
            {"Constraint": "Long-only and fully invested", "Limit": "No shorts; weights sum to 100%", "Purpose": "Keep the paper mandate auditable"},
        ]
    )
    change_conditions = pd.DataFrame(
        [
            {"Condition": "Risk score crosses a regime boundary", "Decision change": "Reset total bank-risk and cash budgets"},
            {"Condition": "Node-stress leadership changes materially", "Decision change": "Re-rank reduction and due-diligence priorities"},
            {"Condition": "Scenario stress concentration moves to another bank", "Decision change": "Reallocate hedge or trim priority"},
            {"Condition": "Fallback and CVaR cash posture diverge further", "Decision change": "Lower confidence and require model review"},
            {"Condition": "Mandate, cost, or liquidity constraint binds", "Decision change": "Defer or resize the proposed rebalance"},
        ]
    )
    decision_body = (
        local_tabs(
            [
                ("recommendation", "Recommendation"),
                ("evidence", "Evidence"),
                ("rebalance", "Rebalance plan"),
                ("change-conditions", "Change conditions"),
            ]
        )
        + section_heading(
            "recommendation",
            "Decision · Portfolio recommendation",
            "Current risk budget and allocation action",
            "This is the definitive synthesis of observed risk, model outputs, scenario evidence, and portfolio constraints.",
        )
        + regime_html
        + metric_grid(
            [
                ("Bank-risk budget", positioning["total_bank_budget"]),
                ("Cash policy range", positioning["cash_guidance"].split(".")[0]),
                ("Largest positive tilt", increase_summary),
                ("Largest reduction", reduction_summary),
                ("Cash-posture agreement", model_agreement),
                ("Validation confidence", f"{model_confidence} · AUC {metrics.iloc[0]['AUC']:.2f}"),
                ("Historical CVaR", f"{cvar_result.diagnostics['historical_cvar']:.1%}"),
                ("CVaR / fallback cash", f"{cvar_cash:.1%} / {fallback_cash:.1%}"),
            ]
        )
        + card(
            "Portfolio recommendation",
            f"{positioning['sector_bias']} Use a {positioning['total_bank_budget']} aggregate bank budget; "
            f"{positioning['cash_guidance']} The primary trade-off is lower concentration and tail exposure "
            "versus potential participation in a bank-sector recovery. Because cash outputs diverge, the regime "
            "policy range is the decision guardrail and model-specific cash weights remain evidence, not automatic trades.",
            regime["tone"],
        )
        + section_heading(
            "evidence",
            "Decision · Evidence",
            "Separate facts, model outputs, and assumptions",
            "The labels below make the provenance of each statement explicit.",
        )
        + "<span class='label'>Observed facts</span>"
        + table_html(facts_table)
        + "<span class='label'>Calculated risk indicators</span>"
        + table_html(calculated_indicators)
        + "<span class='label'>Model outputs</span>"
        + "<div class='chart-grid'>"
        + chart_panel(
            "Bank attractiveness score",
            "Cross-sectional model output combining momentum, stress protection, mean reversion, and macro context.",
            chart_html(investment_score_chart(signals), True),
        )
        + chart_panel(
            "Signal-derived target weights",
            "Model target weights relative to the equal-weight reference; these are outputs, not executed trades.",
            chart_html(investment_weight_chart(signals)),
        )
        + "</div>"
        + "<h3>Cash-posture synthesis</h3>"
        + table_html(cash_posture)
        + "<h3>Binding portfolio constraints</h3>"
        + table_html(decision_constraints)
        + "<details><summary>Assumptions and limitations</summary><div>"
        "<p><span class='label'>Assumptions</span> Public-market proxies, rolling historical relationships, "
        "simplified costs and liquidity, and the documented scenario propagation rules.</p>"
        "<p><span class='label'>Limitations</span> Public data cannot observe every balance-sheet transmission channel; "
        "backtests may overfit; correlations and policy behavior can change. Outputs are research analytics, not personalized advice.</p>"
        "</div></details>"
        + section_heading(
            "rebalance",
            "Decision · Actions",
            "Prioritized rebalance plan",
            "Review target changes against costs, liquidity, mandate limits, and the stated confidence before acting.",
        )
        + table_html(recs_display)
        + "<details><summary>Bank-level reasons and confidence</summary><div>"
        + table_html(signals_display)
        + "</div></details>"
        + section_heading(
            "change-conditions",
            "Decision · Monitoring",
            "What would change the recommendation",
            "These triggers keep the decision conditional and auditable instead of presenting a static answer as certainty.",
        )
        + table_html(change_conditions)
    )

    performance_body = (
        local_tabs(
            [
                ("strategy-performance", "Strategy performance"),
                ("paper-portfolio", "Paper portfolio"),
                ("benchmarks", "Benchmarks"),
                ("risk-activity", "Risk & activity"),
            ]
        )
        + card(
            "Historical simulation and paper results",
            "All results on this page are simulated or paper-traded with fake capital. They are not live returns, "
            "broker records, forecasts, or evidence of future performance.",
            "warning",
        )
        + section_heading(
            "strategy-performance",
            "Performance · Historical simulation",
            "RL and CVaR strategy performance",
            "The comparison uses the same starting capital and transaction-cost framework.",
        )
        + table_html(comparison)
        + "<div class='chart-grid'>"
        + chart_panel(
            "CVaR paper fund and benchmarks",
            "Simulated portfolio value versus reference portfolios over the common historical period.",
            chart_html(cvar_value_chart(cvar_paper.ledger, cvar_benchmarks), True),
        )
        + chart_panel(
            "Strategy exposure",
            "Bank exposure and cash posture for CVaR and the RL research baseline.",
            chart_html(exposure_chart(cvar_paper.ledger, rl_paper.ledger)),
        )
        + "</div>"
        + section_heading(
            "paper-portfolio",
            "Performance · Paper portfolio",
            "Current CVaR paper fund",
            "Fake-money holdings and transactions generated using information available at each rebalance date.",
        )
        + metric_grid(
            [
                ("Starting capital", "$100,000"),
                ("Current paper value", f"${cvar_summary['ending_value']:,.0f}"),
                ("Cumulative return", f"{cvar_summary['cumulative_return']:.1%}"),
                ("Annualized volatility", f"{cvar_summary['annualized_volatility']:.1%}"),
                ("Sharpe ratio", f"{cvar_summary['sharpe_ratio']:.2f}"),
                ("Realized CVaR", f"{cvar_summary['conditional_value_at_risk']:.1%}"),
                ("Maximum drawdown", f"{cvar_summary['max_drawdown']:.1%}"),
                ("Transaction costs", f"${cvar_summary['total_transaction_costs']:,.2f}"),
            ]
        )
        + table_html(cvar_holdings_display, label="CVaR paper fund holdings")
        + section_heading(
            "benchmarks",
            "Performance · Context",
            "Benchmark comparison and drawdowns",
            "Reference portfolios provide context; differing exposures mean they are not perfect substitutes.",
        )
        + "<div class='chart-grid'>"
        + chart_panel(
            "Drawdown comparison",
            "Peak-to-trough decline for the paper strategy and benchmark series.",
            chart_html(performance_drawdown_chart(cvar_paper.ledger, cvar_benchmarks)),
        )
        + chart_panel(
            "Latest paper allocation",
            "The most recent paper weights show current concentration and cash posture.",
            chart_html(performance_allocation_chart(cvar_paper.weights)),
        )
        + "</div>"
        + section_heading(
            "risk-activity",
            "Performance · Activity",
            "Risk posture, turnover, and transactions",
            "Transaction activity is retained for auditability and shown behind progressive disclosure.",
        )
        + chart_panel(
            "Cash and systemic risk",
            "Paper cash weight shown with the contemporaneous risk score.",
            chart_html(cash_risk_chart(cvar_paper.ledger)),
        )
        + "<details><summary>Recent simulated transactions</summary><div>"
        + table_html(cvar_trades_display, label="CVaR paper fund recent transactions")
        + "</div></details>"
        + "<details><summary>RL paper portfolio detail</summary><div>"
        + metric_grid(
            [
                ("Policy source", paper_policy_source),
                ("Current value", f"${paper_summary['ending_value']:,.0f}"),
                ("Cumulative return", f"{paper_summary['cumulative_return']:.1%}"),
                ("Sharpe ratio", f"{paper_summary['sharpe_ratio']:.2f}"),
                ("Maximum drawdown", f"{paper_summary['max_drawdown']:.1%}"),
            ]
        )
        + table_html(paper_holdings_display, label="RL paper portfolio holdings")
        + table_html(paper_trades_display, label="RL paper portfolio recent transactions")
        + "</div></details>"
    )

    research_body = (
        local_tabs(
            [
                ("data", "Data"),
                ("methodology", "Methodology"),
                ("assumptions", "Assumptions"),
                ("limitations", "Limitations"),
                ("references", "References"),
            ]
        )
        + section_heading(
            "data",
            "Research · Data",
            "Data catalog and lineage",
            "Each input is documented by role, date range, shape, and analytical purpose.",
        )
        + metric_grid(
            [
                ("CSV files", f"{len(inventory):,}"),
                ("Total rows", f"{int(inventory['Rows'].fillna(0).sum()):,}"),
                ("Documented files", f"{inventory['Explanation'].notna().sum():,}"),
                ("Latest dataset", latest_date),
            ]
        )
        + table_html(inventory[["CSV", "Rows", "Columns", "Date Range", "Role", "Explanation"]])
        + section_heading(
            "methodology",
            "Research · Methodology",
            "How the analytical system works",
            "Technical depth is available here without competing with the primary decision workflow.",
        )
        + "<div class='page-links'>"
        "<a href='#risk-method'><strong>Risk score</strong><span>Point-in-time rolling features and historical percentile ranks.</span></a>"
        "<a href='#graph-method'><strong>Contagion graph</strong><span>Dynamic interdependence from correlations and exposure proxies.</span></a>"
        "<a href='#scenario-method'><strong>Scenario propagation</strong><span>Exogenous shocks spread through a normalized adjacency matrix.</span></a>"
        "<a href='#portfolio-method'><strong>Portfolio models</strong><span>Experimental RL and constrained graph-aware CVaR allocation.</span></a>"
        "</div>"
        "<details id='risk-method'><summary>Risk score methodology</summary><div><p>"
        "Market, bank, macro, volatility, drawdown, and correlation features use rolling point-in-time windows. "
        "Components are normalized to interpretable 0–100 stress scores and combined into the composite risk score."
        "</p></div></details>"
        "<details id='graph-method'><summary>Network methodology</summary><div><p>"
        "The graph represents interdependence using rolling correlations and configurable exposure proxies. "
        "Centrality and density inform transmission analysis and covariance inflation."
        "</p></div></details>"
        "<details id='scenario-method'><summary>Scenario methodology</summary><div><p>"
        "Scenarios inject documented exogenous bank shocks and propagate them through the normalized positive-correlation network. "
        "Outputs are conditional stress paths, not forecasts."
        "</p></div></details>"
        "<details id='portfolio-method'><summary>Portfolio methodology</summary><div><p>"
        "The CVaR objective balances expected return, expected shortfall, volatility, graph contagion, and turnover under "
        "long-only, cash, single-name, and financial-exposure constraints. RL remains an experimental comparator."
        "</p></div></details>"
        + section_heading(
            "assumptions",
            "Research · Assumptions",
            "Material modeling assumptions",
            "These assumptions are required to interpret the outputs safely.",
        )
        + "<ul class='driver-list'>"
        "<li>Credit spreads, mortgage stress, capital, and exposure profiles may use public proxies or manual templates.</li>"
        "<li>Historical return relationships are informative but may not persist during a future crisis.</li>"
        "<li>Backtests use simplified transaction-cost, liquidity, and execution assumptions.</li>"
        "<li>Paper simulations use prior-day established holdings to avoid earning returns on future information.</li>"
        "</ul>"
        + section_heading(
            "limitations",
            "Research · Limitations",
            "Known boundaries of the analysis",
            "These limits apply across every page and should be considered before using any result.",
        )
        + "<ul class='driver-list'>"
        "<li>This is an educational research simulator, not personalized financial advice or a regulatory model.</li>"
        "<li>Public market data cannot observe all balance-sheet, funding, capital, mortgage, CRE, or counterparty channels.</li>"
        "<li>RL behavior is sensitive to reward design, training history, and regime coverage.</li>"
        "<li>Scenario propagation simplifies feedback loops and should not be interpreted as a probability forecast.</li>"
        "<li>Backtests and paper portfolios may overfit and do not represent live performance.</li>"
        "</ul>"
        + section_heading(
            "references",
            "Research · References",
            "Primary data and internal documentation",
            "Source names identify provenance without implying endorsement.",
        )
        + "<p>Market data: Yahoo Finance-compatible public series. Macro and rate data: Bank of Canada Valet series. "
        "Internal documentation: methodology report, model card, limitations report, configuration files, and dataset README.</p>"
    )

    return {
        "overview": page_template(
            "overview",
            "Systemic risk, translated into a portfolio decision",
            "A single institutional workflow for Canadian bank market conditions, contagion, scenarios, models, and action.",
            overview_body,
            latest_date,
        ),
        "risk": page_template(
            "risk",
            "Systemic risk evidence",
            "Network interconnectedness and the composite risk score in one analytical area.",
            risk_body,
            latest_date,
        ),
        "scenarios": page_template(
            "scenarios",
            "Scenario analysis",
            "Assumptions, bank transmission, network propagation, portfolio impact, and resulting risk response.",
            scenario_body,
            latest_date,
        ),
        "models": page_template(
            "models",
            "Portfolio models",
            "Experimental RL, governed CVaR, comparative behavior, and validation evidence.",
            models_body,
            latest_date,
        ),
        "decision": page_template(
            "decision",
            "Portfolio decision",
            "The definitive risk budget, allocation recommendation, rationale, and rebalance endpoint.",
            decision_body,
            latest_date,
        ),
        "performance": page_template(
            "performance",
            "Performance and paper portfolios",
            "Clearly labeled historical simulations, paper holdings, benchmarks, drawdowns, and activity.",
            performance_body,
            latest_date,
        ),
        "research": page_template(
            "research",
            "Research and methodology",
            "Data lineage, methodology, assumptions, limitations, and references.",
            research_body,
            latest_date,
        ),
    }


def write_pages() -> None:
    if PUBLIC.exists():
        shutil.rmtree(PUBLIC)
    PUBLIC.mkdir(parents=True)
    pages = build_pages()
    for slug, html in pages.items():
        filename = "index.html" if slug == "overview" else f"{slug}.html"
        (PUBLIC / filename).write_text(html, encoding="utf-8")
        (ROOT / filename).write_text(html, encoding="utf-8")
    ROOT_INDEX.write_text(pages["overview"], encoding="utf-8")
    print(f"Wrote {len(pages)} pages to {PUBLIC} and root HTML files")


if __name__ == "__main__":
    write_pages()
