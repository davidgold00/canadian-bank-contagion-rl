# Canadian Bank Contagion Command Center

An interpretable financial-engineering dashboard for Canadian bank systemic risk.

The project models the Big Six banks as a connected market network, combines bank prices with Canadian macro-rate data, converts noisy indicators into a 0-100 contagion risk score, runs stress scenarios, and shows how a defensive allocation policy would respond through a simulated paper portfolio.

This is educational research, not investment advice. No real trades are placed.

## What This Answers

The dashboard is designed for one practical question:

> When stress rises in Canadian financial markets, how might it spread across the Big Six banks, what does it mean for the Canadian economy, and how should portfolio exposure adapt?

It explains:

- whether the current bank regime is low, moderate, high, or severe risk;
- which market and macro drivers are pushing the score higher;
- which banks are most stressed versus most systemically central;
- how housing, oil, liquidity, rate, global, or bank-specific shocks propagate;
- how a risk-aware portfolio policy changes bank, ETF, and cash exposure;
- how a $100,000 simulated paper fund would have traded those recommendations;
- whether the ML layer has out-of-sample stress-prediction signal;
- what every CSV means and how each file contributes to the analysis.

## Portfolio Intelligence vs Trading Bot

This is not an AI trading bot. It does not connect to a broker, place orders, scrape private data, or claim to predict the next bank-stock move.

The allocation layer is portfolio intelligence: it translates systemic-risk signals into an auditable research policy. The dashboard asks whether a risk-aware process would have reduced concentration, raised cash during stress, rotated away from higher node-stress banks, and behaved sensibly against benchmarks. Treat it like a quant/risk analytics prototype, not an execution engine.

## Why Canadian Banks Matter

Canada's banking system is concentrated. Royal Bank, TD, BMO, Scotiabank, CIBC, and National Bank are deeply linked to mortgages, business credit, household deposits, capital markets, ETFs, pension portfolios, and TSX sentiment.

When bank equities become volatile and highly correlated, the signal is not only about stock prices. It can reflect tightening credit conditions, mortgage stress, funding pressure, weaker investor confidence, and a loss of diversification across financial holdings.

## Dashboard Pages

- **About**: project map, economic meaning, page guide, limitations.
- **Market Overview**: executive readout, risk regime, current drivers, bank performance, Canadian rate and macro-market context.
- **Systemic Bank Network**: graph view of bank-to-bank contagion channels and systemic centrality.
- **Contagion Risk Score**: decomposition of the 0-100 score into volatility, correlation, drawdown, VIX, rates, oil, and CAD drivers.
- **Stress Testing Lab**: scenario controls, propagation paths, post-shock network view, and portfolio loss attribution.
- **RL Portfolio Agent**: defensive allocation policy, benchmark comparison, drawdowns, current weights, and stress-day behavior.
- **Model Validation**: chronological train/test validation, ROC, feature importance, confusion matrix, and model credibility readout.
- **Performance Tracker**: simulated $100,000 paper portfolio with daily target weights, trades, shares, cash, P&L, turnover, transaction costs, holdings, and benchmark comparison.
- **Data Catalog**: inventory and chart explorer for every CSV under `data/`.

## Performance Tracker

The Performance Tracker answers the practical allocation-plan question:

> If this model recommended weights each day, what would a fake-money portfolio have actually held, traded, gained, lost, and paid in costs?

It starts with configurable paper capital, defaulting to $100,000, and simulates daily long-only allocations across:

- RY.TO, TD.TO, BMO.TO, BNS.TO, CM.TO, NA.TO;
- XFN.TO;
- XIU.TO or XIC.TO when available;
- cash.

Daily process:

1. Observe prices, features, contagion risk, bank stress, volatility, drawdown, and momentum available up to that day.
2. Generate target weights from the trained PPO model if usable; otherwise use the transparent stress-aware fallback policy.
3. Compare target weights with current simulated holdings.
4. Generate paper buy/sell trades when the rebalance threshold is exceeded.
5. Apply transaction costs.
6. Update cash, shares, holdings, portfolio value, daily P&L, cumulative P&L, turnover, and trade reasons.
7. Compare performance against equal-weight Big Six, XFN buy-and-hold, XIU/XIC buy-and-hold, and cash.

Leakage control: the simulator does not use future prices or future features to decide today's allocation. Returns from day t to day t+1 are earned by the holdings established on day t.

## Production Deployment

The full Streamlit app remains the richest interactive experience. For public deployment, the repo also includes a Vercel-ready static production export:

- `scripts/export_static_site.py` builds the nine-page static site in `public/` from the latest processed dataset.
- `public/index.html` through `public/data-catalog.html` mirror the dashboard pages with shared navigation.
- `index.html` is kept as a root fallback for simple local preview.
- `vercel.json` and `.vercelignore` keep the deployment small and cache-safe.

Refresh and deploy:

```bash
python scripts/download_data.py
python scripts/build_features.py
python scripts/export_static_site.py
vercel --prod
```

## Data Sources

Live data is used when network access is available:

- Yahoo Finance via `yfinance`: bank prices, XFN, XIU, CAD/USD, oil, gold, TSX, and VIX.
- Bank of Canada Valet API: policy rate and Government of Canada 2-year, 5-year, and 10-year yields.
- Manual templates: housing stress, ETF holdings, and CDS/credit spread proxies.
- Synthetic sample data: reproducible fallback data so the dashboard still runs offline.

Generated raw and processed files are intentionally gitignored. Recreate them with:

```bash
python scripts/download_data.py
python scripts/build_features.py
```

## CSV Guide

See [data/README.md](data/README.md) and the dashboard's **Data Catalog** page. The key generated files are:

- `data/raw/market_prices.csv`: live price panel from Yahoo Finance.
- `data/raw/boc_yields.csv`: live policy-rate and Canadian yield data from Bank of Canada.
- `data/processed/prices.csv`: cleaned aligned price panel.
- `data/processed/model_dataset.csv`: full modeling table with engineered market, macro, network, and score features.

The key tracked CSVs are:

- `data/sample/market_prices.csv`: synthetic fallback market data.
- `data/sample/macro.csv`: synthetic fallback macro/yield data.
- `data/templates/housing_stress_template.csv`: analyst-entered housing and mortgage stress assumptions.
- `data/templates/cds_template.csv`: analyst-entered bank credit-spread proxies.
- `data/templates/etf_holdings_template.csv`: ETF bank ownership weights for overlap analysis.

## Architecture

```text
Live/sample CSVs
   -> market and macro feature engineering
   -> dynamic bank graph and stress features
   -> contagion risk score and supervised stress models
   -> scenario propagation and RL-style portfolio allocation
   -> paper portfolio simulator and benchmark analytics
   -> Streamlit command center
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/download_data.py
python scripts/build_features.py
streamlit run src/dashboard/app.py
```

Optional model runs:

```bash
python scripts/train_supervised.py
python scripts/train_rl.py --agent ppo
python scripts/train_rl.py --agent dqn
```

Run tests:

```bash
pytest
```

## Methodology

Market features include 1-day, 5-day, and 21-day returns; rolling volatility; drawdowns; distance from 52-week highs; XFN beta; VIX changes; CAD, oil, gold, TSX, and ETF context.

Macro features include policy rate, 2-year, 5-year, 10-year yields, yield-curve slope, curvature, and rolling changes.

Network features treat banks as nodes and return relationships as edges. Dense networks imply lower diversification because bank stocks are moving together. Central nodes matter because they can transmit stress even when they are not the worst performer.

The contagion score rises when several stress channels cluster: bank volatility, bank correlation, financial-sector drawdown, global volatility, yield-curve pressure, and macro-market stress proxies.

The stress lab propagates scenario shocks through a correlation-derived adjacency matrix. The RL page connects risk measurement to allocation behavior by increasing cash and reducing high-stress bank exposure as contagion risk rises.

The paper portfolio simulator is deliberately auditable. It records daily holdings, cash, shares, trades, transaction costs, turnover, current allocation, benchmark values, and trade reasons such as "Reduced bank exposure because contagion risk exceeded high-risk threshold" or "Rotated away from high node-stress bank."

## Limitations

This is not a production bank risk model. Public market data cannot fully capture regulatory capital, liquidity, uninsured deposit flow, CRE exposure, loan-book details, or true CDS pricing for every bank. Historical correlations can break, stress propagation is simplified, and backtests can overfit.

Important disclaimers:

- This is a simulated paper portfolio.
- This is not investment advice.
- No real trades are placed.
- Past simulated performance does not imply future returns.
- Data may be synthetic, delayed, incomplete, or proxied.
- Transaction costs, liquidity, taxes, and market impact are simplified.
- Use the dashboard as a research and explanation tool, not as a trading system.

## Interview Talking Points

- Time-series split and no-lookahead portfolio simulation.
- Graph-based systemic-risk framing for Canadian banks.
- Interpretable contagion score with driver decomposition.
- Scenario propagation similar to a simplified OSFI or Bank of Canada stress-testing workflow.
- PPO-ready allocation layer with a transparent fallback policy.
- Paper portfolio ledger with holdings, trades, cash, P&L, costs, and benchmark comparison.
- Dashboard design focused on "what it means" and "what a risk analyst would do next."
