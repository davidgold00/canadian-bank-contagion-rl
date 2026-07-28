import numpy as np
import pandas as pd

from src.portfolio.covariance import ledoit_wolf_covariance
from src.portfolio.graph_adjustment import contagion_adjusted_covariance, graph_risk_metrics


def _returns():
    dates = pd.bdate_range("2024-01-01", periods=90)
    base = np.linspace(-0.01, 0.012, 90)
    return pd.DataFrame(
        {
            "RY.TO": base,
            "TD.TO": base * 0.9 + 0.001,
            "BMO.TO": base * 0.8 - 0.001,
            "BNS.TO": -base * 0.2,
        },
        index=dates,
    )


def test_graph_penalties_increase_effective_covariance():
    returns = _returns()
    cov = ledoit_wolf_covariance(returns)
    metrics = graph_risk_metrics(returns, threshold=0.10)
    adjusted = contagion_adjusted_covariance(cov, metrics, contagion_score=85, penalty_strength=0.75)

    assert adjusted.values.diagonal().sum() > cov.values.diagonal().sum()
    assert metrics.graph_density > 0
    assert metrics.centrality.max() <= 1.0 + 1e-9
