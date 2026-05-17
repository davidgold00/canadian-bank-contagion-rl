"""Portfolio analytics and simulated paper-trading utilities."""

from .allocation_policy import (
    BANKS,
    CASH_ASSET,
    DEFAULT_TRADABLES,
    AllocationPolicyResult,
    generate_model_allocation,
    generate_trade_reasons,
)
from .paper_trader import PaperPortfolioSimulator, SimulationResult
from .performance_metrics import performance_summary

__all__ = [
    "BANKS",
    "CASH_ASSET",
    "DEFAULT_TRADABLES",
    "AllocationPolicyResult",
    "generate_model_allocation",
    "generate_trade_reasons",
    "PaperPortfolioSimulator",
    "SimulationResult",
    "performance_summary",
]
