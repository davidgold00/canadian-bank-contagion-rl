"""Portfolio analytics and simulated paper-trading utilities."""

from .allocation_policy import (
    BANKS,
    CASH_ASSET,
    DEFAULT_TRADABLES,
    AllocationPolicyResult,
    generate_model_allocation,
    generate_trade_reasons,
)
from .paper_trader import CVaRPaperPortfolioSimulator, PaperPortfolioSimulator, SimulationResult
from .performance_metrics import performance_summary
from .cvar_optimizer import CVaROptimizationResult, optimize_cvar_portfolio

__all__ = [
    "BANKS",
    "CASH_ASSET",
    "DEFAULT_TRADABLES",
    "AllocationPolicyResult",
    "generate_model_allocation",
    "generate_trade_reasons",
    "PaperPortfolioSimulator",
    "CVaRPaperPortfolioSimulator",
    "SimulationResult",
    "performance_summary",
    "CVaROptimizationResult",
    "optimize_cvar_portfolio",
]
