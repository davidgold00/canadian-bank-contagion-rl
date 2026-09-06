# Northern Signal

An institutional decision-support platform for Canadian bank systemic risk.

The project models the Big Six banks as a connected market network, combines bank prices with Canadian macro-rate data, converts noisy indicators into a 0-100 contagion risk score, runs stress scenarios, and shows how both experimental RL and production-style CVaR allocation frameworks respond through simulated paper portfolios.

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

The allocation layer is portfolio intelligence: it translates systemic-risk signals into auditable portfolio policies. The dashboard asks whether a risk-aware process would have reduced concentration, raised cash during stress, rotated away from higher node-stress banks, and behaved sensibly against benchmarks. Treat it like a quant/risk analytics prototype, not an execution engine.

## Why CVaR Instead of Pure Reinforcement Learning?

The project now explicitly separates:

- **Experimental RL allocation:** useful as a nonlinear research baseline and state-aware policy prototype.
- **Production-style portfolio optimization:** a governed CVaR allocator using graph-adjusted covariance, regime-aware constraints, turnover penalties, and contagion-aware risk budgeting.

Institutional allocators usually prefer explainable constrained optimization for production risk-controlled portfolios. CVaR, or Conditional Value at Risk, directly measures expected loss in the left tail rather than treating upside and downside volatility the same way. It is easier to validate, easier to explain to a risk committee, and easier to connect to formal portfolio limits.

RL remains in the project because it is valuable research: it can learn nonlinear policy behavior and adapt to state variables. It is not positioned as the primary allocation framework because RL can be unstable, hard to explain, sensitive to reward design, and vulnerable to overfitting.

## Why Canadian Banks Matter

Canada's banking system is concentrated. Royal Bank, TD, BMO, Scotiabank, CIBC, and National Bank are deeply linked to mortgages, business credit, household deposits, capital markets, ETFs, pension portfolios, and TSX sentiment.

When bank equities become volatile and highly correlated, the signal is not only about stock prices. It can reflect tightening credit conditions, mortgage stress, funding pressure, weaker investor confidence, and a loss of diversification across financial holdings.

## Product Areas

The public product follows one analytical journey:

> Market conditions → systemic-risk evidence → scenario analysis → portfolio models → portfolio review

- **Overview**: current risk score and regime, three material drivers, portfolio recommendation, confidence, and data freshness.
- **Risk**: local views for the systemic bank network and composite risk score.
- **Scenarios**: assumptions, bank-level transmission, network propagation, portfolio impact, and resulting response.
- **Models**: local views for the RL strategy, CVaR strategy, comparison, and validation.
- **Decision**: actual paper holdings versus the current CVaR snapshot target, independent signals, policy reconciliation status, and change conditions.
- **Performance**: clearly labeled historical simulations, paper portfolios, benchmarks, drawdowns, turnover, and activity.
- **Research**: data catalog, methodology, assumptions, limitations, and references.

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

## Interview-readiness audit

See [the classified before/after record](docs/interview-audit.md) for verified bug fixes, transparency additions, diagnostics, regression results and unresolved limitations. The public Research page records actual run identities and methodology; current CVaR targets, historical CVaR holdings, two PPO evaluation periods, the standalone heuristic and editorial guidance are distinct. Headline cumulative returns use gross initial capital, with the former post-cost-NAV convention disclosed separately.

Browser regression checks use `scripts/check_static_browser.cjs` with an externally installed Playwright package (`NODE_PATH`), a local `scripts/serve_site.py` server, and optional `BASE_URL` / `QA_OUTPUT` settings. No browser package is required to build or serve the static site.

## Production Deployment

The full Streamlit app remains the richest interactive experience. For public deployment, the repo also includes a Vercel-ready static production export:

- `scripts/export_static_site.py` builds the seven-route static site in `public/` from the latest processed dataset.
- `public/index.html` plus the six consolidated product-area pages share one responsive, accessible shell.
- `index.html` is kept as a root fallback for simple local preview.
- `vercel.json` preserves legacy deep links with redirects and keeps the deployment cache-safe.

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

See [data/README.md](data/README.md) and the platform's **Research** area. The key generated files are:

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
   -> contagion graph and graph-adjusted covariance
   -> CVaR optimization and regime-aware constraints
   -> scenario propagation and RL research baseline
   -> paper portfolio simulators and benchmark analytics
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

The CVaR optimizer estimates shrinkage covariance, builds a Canadian bank contagion graph, inflates effective covariance when graph density and centrality rise, and solves a constrained long-only optimization problem:

```text
minimize:
  - expected_return
  + CVaR penalty
  + volatility penalty
  + graph contagion exposure penalty
  + turnover penalty

subject to:
  weights sum to 1
  long-only weights
  maximum single-name exposure
  maximum Canadian financial exposure
  minimum / maximum cash allocation
```

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

