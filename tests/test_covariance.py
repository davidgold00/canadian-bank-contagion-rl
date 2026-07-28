import numpy as np
import pandas as pd

from src.portfolio.covariance import ensure_psd, ledoit_wolf_covariance, price_returns


def _prices():
    dates = pd.bdate_range("2024-01-01", periods=80)
    return pd.DataFrame(
        {
            "RY.TO": 100 * np.cumprod(np.full(80, 1.001)),
            "TD.TO": 95 * np.cumprod(np.full(80, 1.0008)),
            "BMO.TO": 90 * np.cumprod(np.full(80, 1.0006)),
        },
        index=dates,
    )


def test_ledoit_wolf_covariance_is_psd():
    returns = price_returns(_prices())
    cov = ledoit_wolf_covariance(returns)
    eigvals = np.linalg.eigvalsh(cov.values)
    assert (eigvals >= -1e-8).all()
    assert cov.shape == (3, 3)


def test_ensure_psd_repairs_negative_eigenvalue():
    matrix = pd.DataFrame([[1.0, 2.0], [2.0, 1.0]], index=["a", "b"], columns=["a", "b"])
    psd = ensure_psd(matrix)
    assert (np.linalg.eigvalsh(psd.values) >= -1e-8).all()
