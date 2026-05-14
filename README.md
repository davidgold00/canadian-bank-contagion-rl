# Canadian Bank Contagion Simulator Using Graph Reinforcement Learning

Institutional-style Canadian systemic-risk simulator for the Big Six banks: RY.TO, TD.TO, BMO.TO, BNS.TO, CM.TO, and NA.TO. The project combines public market data, Bank of Canada yield data hooks, mortgage/housing stress templates, dynamic financial graphs, supervised stress prediction, shock propagation, and reinforcement learning allocation.

## Why Canadian Banks?
Canada's equity market is concentrated in financials, energy, and materials. The Big Six banks dominate domestic banking and are exposed to mortgage credit, housing, yield curves, oil-sensitive macro conditions, liquidity, and credit spreads. Their market stress can transmit through shared exposures, ETF ownership overlap, correlation spikes, and balance-sheet similarity.

## Architecture
```text
Data -> Features -> Dynamic Graph -> Stress/Contagion Models -> RL Allocation -> Dashboard
 yfinance   BoC Valet   rolling correlations   stress scores      PPO/DQN      Streamlit
 templates  CSV fallbacks ETF overlap          scenarios          backtests    Plotly
```

## Data Sources
- yfinance: Canadian bank equities, XFN, XIU/XIC, CAD/USD, crude oil, gold, TSX, VIX.
- Bank of Canada Valet API: policy rate and Government of Canada yields; series IDs are configurable.
- Manual CSV templates: housing stress, ETF holdings, CDS/credit spreads.
- Synthetic fallback samples in `data/sample/` allow the full pipeline to run immediately.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_data.py
python scripts/build_features.py
python scripts/train_supervised.py
python scripts/train_rl.py --agent ppo
python scripts/train_rl.py --agent dqn
streamlit run src/dashboard/app.py
```

## Methodology
Dynamic bank graphs are built from rolling return correlations, with extensions for tail dependence, ETF ownership overlap, balance-sheet similarity, lead-lag transmission, and credit stress. The composite contagion risk score combines volatility, pairwise correlation, eigenvalue concentration proxies, drawdowns, VIX, credit stress proxies, yield-curve stress, and housing stress inputs into a 0-100 score.

Supervised models predict next-week stress labels using leakage-aware time splits. The Gymnasium RL environment trains PPO with continuous target weights and DQN with discrete allocation templates. Rewards penalize volatility, drawdown, turnover, concentration, high-stress-node exposure, and contagion exposure while rewarding excess return versus XFN.

## Results
Run the scripts to populate `artifacts/` with model metrics and RL backtests. Dashboard pages provide placeholders and live views once data is built.

## Limitations
This project is educational research, not investment advice. CDS data may be proxied, public data cannot fully replicate regulatory systemic-risk models, historical backtests can overfit, transaction costs are simplified, and future performance is not guaranteed.

## Recruiting Talking Points
- Graph ML for financial contagion networks.
- Reinforcement learning for allocation under stress.
- Canadian market structure and systemic-risk domain knowledge.
- Leakage-aware time-series ML and walk-forward validation.
- Production-grade modular ML pipeline with dashboard/product thinking.
