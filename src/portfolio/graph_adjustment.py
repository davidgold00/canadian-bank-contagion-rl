from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

from .allocation_policy import BANKS
from .covariance import ensure_psd


@dataclass(frozen=True)
class GraphRiskMetrics:
    average_correlation: float
    graph_density: float
    largest_eigenvalue: float
    centrality: pd.Series
    node_stress: pd.Series


def bank_stress_from_features(features_history: pd.DataFrame, banks: list[str] | None = None) -> pd.Series:
    banks = banks or BANKS
    if features_history.empty:
        return pd.Series(50.0, index=banks)
    out = {}
    for bank in banks:
        components = []
        for col, sign in [(f"{bank}_vol_21d", 1), (f"{bank}_drawdown_63d", -1), (f"{bank}_beta_xfn_63d", 1)]:
            if col in features_history:
                series = sign * pd.to_numeric(features_history[col], errors="coerce").dropna()
                if len(series):
                    components.append((series <= series.iloc[-1]).mean() * 100)
        out[bank] = float(np.mean(components)) if components else 50.0
    return pd.Series(out).clip(0, 100)


def correlation_graph(returns: pd.DataFrame, threshold: float = 0.35) -> tuple[nx.Graph, pd.DataFrame]:
    banks = [bank for bank in BANKS if bank in returns.columns]
    corr = returns[banks].corr().fillna(0.0).clip(-1, 1)
    graph = nx.Graph()
    graph.add_nodes_from(banks)
    for i, source in enumerate(banks):
        for target in banks[i + 1 :]:
            weight = abs(float(corr.loc[source, target]))
            if weight >= threshold:
                graph.add_edge(source, target, weight=weight)
    return graph, corr


def graph_risk_metrics(returns: pd.DataFrame, features_history: pd.DataFrame | None = None, threshold: float = 0.35) -> GraphRiskMetrics:
    graph, corr = correlation_graph(returns, threshold=threshold)
    banks = list(corr.columns)
    if graph.number_of_edges():
        try:
            centrality = pd.Series(nx.eigenvector_centrality_numpy(graph, weight="weight"), dtype=float)
        except nx.AmbiguousSolution:
            centrality = pd.Series(dict(graph.degree(weight="weight")), dtype=float)
    else:
        centrality = pd.Series(0.0, index=banks)
    centrality = centrality.reindex(banks).fillna(0.0)
    if centrality.max() > 0:
        centrality = centrality / centrality.max()

    mask = ~np.eye(len(corr), dtype=bool)
    avg_corr = float(corr.abs().where(mask).stack().mean()) if len(corr) > 1 else 0.0
    largest_eigen = float(np.linalg.eigvalsh(corr.fillna(0.0).values).max()) if len(corr) else 0.0
    stress = bank_stress_from_features(features_history if features_history is not None else pd.DataFrame(), banks)
    return GraphRiskMetrics(
        average_correlation=avg_corr,
        graph_density=float(nx.density(graph)) if len(banks) > 1 else 0.0,
        largest_eigenvalue=largest_eigen,
        centrality=centrality,
        node_stress=stress.reindex(banks).fillna(50.0),
    )


def contagion_adjusted_covariance(
    base_covariance: pd.DataFrame,
    graph_metrics: GraphRiskMetrics,
    contagion_score: float = 50.0,
    penalty_strength: float = 0.40,
) -> pd.DataFrame:
    """Increase effective covariance when systemic connectivity and node centrality are elevated."""
    cov = base_covariance.copy().astype(float)
    risky_assets = [asset for asset in cov.index if asset in graph_metrics.centrality.index]
    if not risky_assets:
        return ensure_psd(cov)

    systemic_pressure = (
        0.45 * min(max(contagion_score / 100.0, 0.0), 1.0)
        + 0.30 * min(max(graph_metrics.graph_density, 0.0), 1.0)
        + 0.25 * min(max(graph_metrics.average_correlation, 0.0), 1.0)
    )
    multiplier = 1.0 + penalty_strength * systemic_pressure
    cov.loc[risky_assets, risky_assets] = cov.loc[risky_assets, risky_assets] * multiplier

    centrality = graph_metrics.centrality.reindex(risky_assets).fillna(0.0)
    stress = graph_metrics.node_stress.reindex(risky_assets).fillna(50.0) / 100.0
    asset_penalty = 1.0 + penalty_strength * (0.65 * centrality + 0.35 * stress)
    scale = np.outer(asset_penalty.values, asset_penalty.values)
    cov.loc[risky_assets, risky_assets] = cov.loc[risky_assets, risky_assets].values * scale
    return ensure_psd(cov)


def centrality_penalty_table(graph_metrics: GraphRiskMetrics) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "Asset": graph_metrics.centrality.index,
            "Eigenvector Centrality": graph_metrics.centrality.values,
            "Node Stress": graph_metrics.node_stress.reindex(graph_metrics.centrality.index).fillna(50.0).values,
        }
    )
    out["Centrality-Adjusted Penalty"] = 0.65 * out["Eigenvector Centrality"] + 0.35 * out["Node Stress"] / 100.0
    out["Interpretation"] = out["Centrality-Adjusted Penalty"].map(
        lambda x: "High systemic concentration penalty" if x >= 0.70 else "Moderate network penalty" if x >= 0.40 else "Lower network penalty"
    )
    return out.sort_values("Centrality-Adjusted Penalty", ascending=False)
