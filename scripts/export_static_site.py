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


def metric_grid(items: list[tuple[str, str]]) -> str:
    cells = "".join(f"<div class='metric'><span>{label}</span><strong>{value}</strong></div>" for label, value in items)
    return f"<div class='grid'>{cells}</div>"


def nav(active_slug: str) -> str:
    links = []
    for slug, label in PAGES:
        href = "/" if slug == "about" else f"/{slug}"
        active = " active" if slug == active_slug else ""
        links.append(f"<a class='nav-link{active}' href='{href}'>{label}</a>")
    return "<nav class='nav'>" + "".join(links) + "</nav>"


def page_template(slug: str, title: str, subtitle: str, body: str, latest_date: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title} | Canadian Bank Contagion Command Center</title>
  <meta name="description" content="Canadian bank contagion, macro risk, network stress, and portfolio action dashboard.">
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
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: #ffffff;
      line-height: 1.5;
    }}
    .nav {{
      position: sticky;
      top: 0;
      z-index: 20;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      padding: 12px 7vw;
      background: rgba(255, 255, 255, 0.96);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(10px);
    }}
    .nav-link {{
      color: var(--ink);
      text-decoration: none;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 7px 11px;
      font-size: 0.9rem;
      background: #fff;
    }}
    .nav-link.active {{
      background: var(--blue);
      border-color: var(--blue);
      color: #fff;
    }}
    header {{
      background: #eef4f8;
      border-bottom: 1px solid var(--line);
      padding: 48px 7vw 34px;
    }}
    main {{
      padding: 30px 7vw 56px;
      max-width: 1440px;
      margin: 0 auto;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: clamp(2rem, 4vw, 4.2rem);
      letter-spacing: 0;
      line-height: 1.04;
    }}
    h2 {{
      margin: 42px 0 12px;
      font-size: clamp(1.35rem, 2vw, 2rem);
      letter-spacing: 0;
    }}
    h3 {{ margin: 0 0 10px; }}
    p {{ max-width: 980px; color: #354153; }}
    .subtitle {{ font-size: 1.12rem; max-width: 980px; color: #354153; }}
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
      margin: 22px 0;
    }}
    .metric, .callout, .chart-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      box-shadow: 0 1px 2px rgba(24, 33, 47, 0.04);
    }}
    .metric {{ padding: 16px; }}
    .metric span {{ display: block; color: var(--muted); font-size: 0.88rem; }}
    .metric strong {{ display: block; margin-top: 5px; font-size: 1.55rem; }}
    .callout {{
      border-left: 5px solid var(--blue);
      background: var(--panel);
      padding: 18px 20px;
      margin: 20px 0;
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
    .chart-card {{ padding: 8px; overflow: hidden; }}
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
    .data-table th {{ background: var(--panel); color: var(--ink); }}
    .page-links {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin-top: 18px;
    }}
    .page-links a {{
      display: block;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      color: var(--ink);
      text-decoration: none;
      background: #fff;
    }}
    footer {{
      color: var(--muted);
      border-top: 1px solid var(--line);
      padding-top: 22px;
      margin-top: 40px;
      font-size: 0.92rem;
    }}
    @media (max-width: 980px) {{
      .grid, .chart-grid, .page-links {{ grid-template-columns: 1fr; }}
      header, main, .nav {{ padding-left: 22px; padding-right: 22px; }}
      .nav {{ position: static; }}
    }}
  </style>
</head>
<body>
  {nav(slug)}
  <header>
    <h1>{title}</h1>
    <p class="subtitle">{subtitle}</p>
    <span class="pill">Data through {latest_date}</span>
    <span class="pill">Yahoo Finance prices + Bank of Canada yields</span>
    <span class="pill">Vercel static production site</span>
  </header>
  <main>
    {body}
    <footer>Educational research only. Not investment advice, not a trading system, and not a regulatory bank risk model.</footer>
  </main>
</body>
</html>
"""


def score_chart(features: pd.DataFrame) -> go.Figure:
    score = features["contagion_risk_score"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=score.index, y=score, mode="lines", name="Contagion risk", line=dict(color="#1d5f8f", width=2)))
    fig.add_hrect(y0=0, y1=30, fillcolor="#e9f7ef", opacity=0.45, line_width=0)
    fig.add_hrect(y0=30, y1=60, fillcolor="#fff4df", opacity=0.45, line_width=0)
    fig.add_hrect(y0=60, y1=80, fillcolor="#fdebd3", opacity=0.45, line_width=0)
    fig.add_hrect(y0=80, y1=100, fillcolor="#fdecec", opacity=0.45, line_width=0)
    fig.update_layout(title="Canadian Bank Contagion Score", yaxis_title="0-100 score", height=430, margin=dict(l=30, r=20, t=55, b=35), paper_bgcolor="white", plot_bgcolor="white")
    return fig


def driver_chart(features: pd.DataFrame) -> go.Figure:
    drivers = strongest_drivers(features).head(8).sort_values("Stress Percentile")
    fig = go.Figure(go.Bar(x=drivers["Stress Percentile"], y=drivers["Driver"], orientation="h", text=[f"{x:.0%}" for x in drivers["Stress Percentile"]], textposition="auto", marker_color="#1d5f8f"))
    fig.update_layout(title="Current Risk Driver Percentiles", xaxis_tickformat=".0%", height=430, margin=dict(l=30, r=20, t=55, b=35), paper_bgcolor="white", plot_bgcolor="white")
    return fig


def bank_chart(bank_table: pd.DataFrame) -> go.Figure:
    ordered = bank_table.sort_values("Node Stress")
    fig = go.Figure(go.Bar(x=ordered["Node Stress"], y=ordered["Bank"], orientation="h", text=[f"{x:.1f}" for x in ordered["Node Stress"]], textposition="auto", marker_color="#b42318"))
    fig.update_layout(title="Bank-Level Node Stress", xaxis_title="0-100 stress score", height=430, margin=dict(l=30, r=20, t=55, b=35), paper_bgcolor="white", plot_bgcolor="white")
    return fig


def correlation_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    banks = [b for b in BANKS if b in prices]
    return prices[banks].pct_change().dropna().tail(63).corr().fillna(0)


def correlation_chart(prices: pd.DataFrame) -> go.Figure:
    corr = correlation_matrix(prices)
    fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1, colorscale="RdBu", text=np.round(corr.values, 2), texttemplate="%{text}", colorbar=dict(title="Correlation")))
    fig.update_layout(title="Latest 63-Day Bank Correlation Matrix", height=460, margin=dict(l=30, r=20, t=55, b=35), paper_bgcolor="white", plot_bgcolor="white")
    return fig


def macro_chart(macro: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col, label in [("policy_rate", "Policy rate"), ("ca_2y", "2Y yield"), ("ca_10y", "10Y yield")]:
        if col in macro:
            fig.add_trace(go.Scatter(x=macro.index, y=macro[col], mode="lines", name=label))
    fig.update_layout(title="Canadian Rate Backdrop", yaxis_title="Percent", height=430, margin=dict(l=30, r=20, t=55, b=35), paper_bgcolor="white", plot_bgcolor="white")
    return fig


def allocation_chart(bank_table: pd.DataFrame, score: float) -> tuple[go.Figure, pd.Series]:
    stress = bank_table.set_index("Bank")["Node Stress"]
    cash_weight = float(np.clip((score - 35) / 65, 0.05, 0.75))
    xfn_weight = float(np.clip((60 - score) / 100, 0.00, 0.25))
    bank_budget = 1 - cash_weight - xfn_weight
    bank_scores = (100 - stress).clip(lower=1)
    allocation = bank_budget * bank_scores / bank_scores.sum()
    weights = pd.concat([allocation, pd.Series({"XFN.TO": xfn_weight, "cash": cash_weight})]).sort_values(ascending=False)
    fig = go.Figure(go.Bar(x=weights.index, y=weights.values, text=[f"{x:.1%}" for x in weights.values], textposition="auto", marker_color="#1f7a5a"))
    fig.update_layout(title=f"Risk-Aware Allocation Snapshot | Score {score:.1f}/100", yaxis_title="Weight", yaxis_tickformat=".0%", height=430, margin=dict(l=30, r=20, t=55, b=35), paper_bgcolor="white", plot_bgcolor="white")
    return fig, weights


def normalize_prices(prices: pd.DataFrame, columns: list[str], days=252) -> pd.DataFrame:
    view = prices[[c for c in columns if c in prices]].tail(days).ffill().dropna(how="all")
    if view.empty:
        return view
    return view.divide(view.iloc[0]).mul(100)


def price_context_chart(prices: pd.DataFrame) -> go.Figure:
    normalized = normalize_prices(prices, ["XFN.TO", "XIU.TO", "^GSPTSE", "CADUSD=X", "CL=F", "GC=F"])
    fig = go.Figure()
    for col in normalized.columns:
        fig.add_trace(go.Scatter(x=normalized.index, y=normalized[col], mode="lines", name=col))
    fig.update_layout(title="One-Year Macro Market Context", yaxis_title="Indexed to 100", height=430, margin=dict(l=30, r=20, t=55, b=35), paper_bgcolor="white", plot_bgcolor="white")
    return fig


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
    fig = go.Figure(go.Bar(x=latest_components.values, y=latest_components.index, orientation="h", text=[f"{x:.1f}" for x in latest_components.values], textposition="auto", marker_color="#1d5f8f"))
    fig.update_layout(title="Current Component Scores", xaxis_title="0-100 stress contribution", height=500, margin=dict(l=30, r=20, t=55, b=35), paper_bgcolor="white", plot_bgcolor="white")
    return fig


def component_heatmap(components: pd.DataFrame) -> go.Figure:
    recent = components.tail(126).T
    fig = go.Figure(go.Heatmap(z=recent.values, x=recent.columns, y=recent.index, zmin=0, zmax=100, colorscale="RdYlGn_r", colorbar=dict(title="Stress")))
    fig.update_layout(title="Stress Breadth Over the Last Six Months", height=540, margin=dict(l=30, r=20, t=55, b=35), paper_bgcolor="white", plot_bgcolor="white")
    return fig


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
    for bank in BANKS:
        fig.add_trace(go.Scatter(x=paths.index, y=paths[bank], mode="lines+markers", name=bank))
    fig.update_layout(title="Liquidity Squeeze Scenario: Shock Propagation", xaxis_title="Propagation step", yaxis_title="0-100 stress score", height=470, margin=dict(l=30, r=20, t=55, b=35), paper_bgcolor="white", plot_bgcolor="white")
    return fig


def final_stress_chart(final_stress: pd.Series) -> go.Figure:
    ordered = final_stress.sort_values()
    fig = go.Figure(go.Bar(x=ordered.values, y=ordered.index, orientation="h", text=[f"{x:.1f}" for x in ordered.values], textposition="auto", marker_color="#b42318"))
    fig.update_layout(title="Final Scenario Stress Ranking", xaxis_title="0-100 stress score", height=430, margin=dict(l=30, r=20, t=55, b=35), paper_bgcolor="white", plot_bgcolor="white")
    return fig


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
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    node_x, node_y, labels, colors, sizes, hover = [], [], [], [], [], []
    degree = nx.degree_centrality(graph)
    max_degree = max(degree.values()) if degree else 1
    for bank in BANKS:
        x, y = pos[bank]
        node_x.append(x)
        node_y.append(y)
        labels.append(bank)
        colors.append(float(stress.get(bank, 50)))
        sizes.append(35 + 55 * degree.get(bank, 0) / (max_degree or 1))
        hover.append(f"{bank}: {BANK_NAMES[bank]}<br>Stress {stress.get(bank, 50):.1f}/100<br>{BANK_CONTEXT[bank]}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=2, color="rgba(70,80,95,.35)"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text", text=labels, textposition="top center", hovertext=hover, hoverinfo="text", marker=dict(size=sizes, color=colors, cmin=0, cmax=100, colorscale="RdYlGn_r", colorbar=dict(title="Stress"), line=dict(width=2, color="white"))))
    fig.update_layout(title="Contagion Network Map", height=560, showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=30, r=20, t=55, b=35), paper_bgcolor="white", plot_bgcolor="white")
    return fig


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
    for model, group in roc_df.groupby("Model"):
        fig.add_trace(go.Scatter(x=group["FPR"], y=group["TPR"], mode="lines", name=f"{model} AUC={auc_map[model]:.2f}"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")))
    fig.update_layout(title="ROC Curve: Future Stress Prediction", xaxis_title="False positive rate", yaxis_title="True positive rate", height=450, margin=dict(l=30, r=20, t=55, b=35), paper_bgcolor="white", plot_bgcolor="white")
    return fig


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
            ("data-catalog", "Data Catalog", "CSV inventory, explanations, and data lineage."),
        ]
    )
    about_body = (
        metric_grid([("Contagion Risk", f"{score:.1f}/100"), ("Regime", regime["label"]), ("Avg Bank Vol", pct(latest(features, "avg_bank_vol_21d"))), ("Avg Bank Corr", f"{latest(features, 'avg_pairwise_corr_63d'):.2f}")])
        + card("Core Question", "When stress rises in Canadian financial markets, how might it spread across the Big Six banks, and how should exposure adapt?", regime["tone"])
        + "<h2>How the System Works</h2><p>Data becomes features, features become a bank network and risk score, scenarios turn that score into stress paths, and the allocation layer translates risk into portfolio posture.</p>"
        + "<div class='page-links'>" + about_links + "</div>"
    )
    pages["about"] = page_template("about", "About the Canadian Bank Contagion Simulator", "A financial-engineering project that turns bank, macro, and network data into decisions.", about_body, latest_date)

    market_body = (
        metric_grid([("Risk Regime", regime["label"]), ("Contagion Score", f"{score:.1f}/100"), ("XFN 21D Return", pct(latest(features, "XFN.TO_ret_21d"))), ("Bank Correlation", f"{latest(features, 'avg_pairwise_corr_63d'):.2f}")])
        + card(f"Current Readout: {regime['label']} Risk", regime["summary"], regime["tone"])
        + "<div class='chart-grid'><div class='chart-card'>" + chart_html(score_chart(features), True) + "</div><div class='chart-card'>" + chart_html(driver_chart(features)) + "</div></div>"
        + "<h2>Current Risk Drivers</h2>" + table_html(driver_display[["Driver", "Latest", "Stress Percentile", "Status", "Why it matters"]].head(8))
        + "<h2>Economic Transmission</h2><div class='chart-grid'><div class='chart-card'>" + chart_html(price_context_chart(prices)) + "</div><div class='chart-card'>" + chart_html(macro_chart(macro)) + "</div></div>"
    )
    pages["market-overview"] = page_template("market-overview", "Executive Market Overview", "Today's Canadian bank risk regime, translated into business decisions.", market_body, latest_date)

    network_body = (
        metric_grid([("Average Correlation", f"{correlation_matrix(prices).where(~np.eye(len(BANKS), dtype=bool)).stack().mean():.2f}"), ("Most Stressed Bank", bank_table.iloc[0]["Bank"]), ("Network Lens", "63D correlation"), ("Systemic Use", "Diversification check")])
        + card("How to Read the Network", "Redder nodes carry more market stress. More links mean the Big Six are behaving less like separate holdings and more like one macro-financial trade.")
        + "<div class='chart-grid'><div class='chart-card'>" + chart_html(network_chart(prices, bank_table), True) + "</div><div class='chart-card'>" + chart_html(correlation_chart(prices)) + "</div></div>"
        + "<h2>Systemic Ranking</h2>" + table_html(bank_display[["Bank", "Name", "Node Stress", "Action Readout", "Economic Lens"]])
    )
    pages["systemic-bank-network"] = page_template("systemic-bank-network", "Systemic Bank Network", "See whether the Big Six are diversifying each other or moving as one risk cluster.", network_body, latest_date)

    contagion_body = (
        metric_grid([("Latest Score", f"{score:.1f}/100"), ("Regime", regime["label"]), ("Historical Percentile", f"{percentile_rank(features['contagion_risk_score'], score):.0%}"), ("Latest Date", latest_date)])
        + card(f"Interpretation: {regime['label']} Risk", regime["summary"], regime["tone"])
        + "<div class='chart-grid'><div class='chart-card'>" + chart_html(component_bar_chart(components), True) + "</div><div class='chart-card'>" + chart_html(component_heatmap(components)) + "</div></div>"
        + "<h2>Bank Contributors</h2>" + table_html(bank_display[["Bank", "Name", "21D Return", "21D Volatility", "63D Drawdown", "Node Stress", "Action Readout"]])
    )
    pages["contagion-risk-score"] = page_template("contagion-risk-score", "Contagion Risk Score", "A 0-100 answer to whether Canadian bank stress signals are clustering.", contagion_body, latest_date)

    stress_body = (
        metric_grid([("Scenario", "Liquidity Squeeze"), ("Average Final Stress", f"{final_stress.mean():.1f}/100"), ("Most Stressed Bank", final_stress.idxmax()), ("Peak Bank Stress", f"{final_stress.max():.1f}/100")])
        + card("Scenario Readout", "The production export shows a default liquidity squeeze scenario. Use the Streamlit app locally for adjustable scenario controls.", "warning")
        + "<div class='chart-grid'><div class='chart-card'>" + chart_html(stress_path_chart(paths), True) + "</div><div class='chart-card'>" + chart_html(final_stress_chart(final_stress)) + "</div></div>"
        + "<h2>Portfolio Impact</h2>" + table_html(stress_impact)
    )
    pages["stress-testing-lab"] = page_template("stress-testing-lab", "Stress Testing Lab", "Turn a macro shock into bank stress, contagion paths, and portfolio action.", stress_body, latest_date)

    rl_body = (
        metric_grid([("Policy Type", "Risk-aware static"), ("Cash Weight", weight_table.loc[weight_table["Asset"] == "cash", "Weight"].iloc[0]), ("Regime", regime["label"]), ("Score", f"{score:.1f}/100")])
        + card("Allocation Rationale", "Cash rises when contagion risk is elevated, while bank exposure is tilted away from higher-stress names. This mirrors the transparent RL-style policy from the Streamlit dashboard.", regime["tone"])
        + "<div class='chart-grid'><div class='chart-card'>" + chart_html(allocation_fig, True) + "</div><div class='chart-card'>" + chart_html(bank_chart(bank_table)) + "</div></div>"
        + "<h2>Current Allocation</h2>" + table_html(weight_table)
    )
    pages["rl-portfolio-agent"] = page_template("rl-portfolio-agent", "RL Portfolio Agent", "Translate bank contagion risk into allocation, cash, turnover, and drawdown decisions.", rl_body, latest_date)

    model_body = (
        metric_grid([("Best Model", metrics.iloc[0]["Model"]), ("Best AUC", f"{metrics.iloc[0]['AUC']:.2f}"), ("Rows", f"{len(features):,}"), ("Split", "Chronological")])
        + card("Validation Readout", "The model validation page checks whether stress features have out-of-sample ranking signal. AUC above 0.50 means better than random ranking.", "warning" if metrics.iloc[0]["AUC"] < 0.65 else "success")
        + "<div class='chart-grid'><div class='chart-card'>" + chart_html(roc_chart(roc_df, metrics), True) + "</div><div class='chart-card'>" + chart_html(driver_chart(features)) + "</div></div>"
        + "<h2>Chronological Test Metrics</h2>" + table_html(metrics_display)
    )
    pages["model-validation"] = page_template("model-validation", "Model Validation", "Check whether the ML layer has real out-of-sample stress signal.", model_body, latest_date)

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
    ROOT_INDEX.write_text(pages["about"], encoding="utf-8")
    print(f"Wrote {len(pages)} pages to {PUBLIC}")


if __name__ == "__main__":
    write_pages()
