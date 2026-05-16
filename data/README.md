# Data Dictionary and CSV Guide

This folder contains three kinds of CSVs:

- **sample CSVs** committed to the repo so the project runs offline;
- **template CSVs** committed to the repo so analysts know what manual inputs look like;
- **raw/processed CSVs** generated locally by `scripts/download_data.py` and `scripts/build_features.py`.

The Streamlit **Data Catalog** page profiles every CSV and provides a chart explorer.

## Tracked CSVs

| CSV | Role | What It Means | How It Is Charted |
| --- | --- | --- | --- |
| `data/sample/market_prices.csv` | Synthetic fallback market prices | Offline stand-in for bank, ETF, currency, commodity, TSX, and VIX prices. Not real market history. | Recent price series indexed to 100. |
| `data/sample/macro.csv` | Synthetic fallback macro data | Offline stand-in for policy rate, Canadian yields, slope, and curvature. | Rate and yield-curve trend. |
| `data/templates/housing_stress_template.csv` | Housing stress input | Analyst assumptions for arrears, delinquency, home prices, unemployment, credit growth, and mortgage debt. | Normalized housing stress variables. |
| `data/templates/cds_template.csv` | Credit spread input | Bank CDS or credit-spread proxy. Wider spreads mean the market is demanding more compensation for credit risk. | Spread level by bank. |
| `data/templates/etf_holdings_template.csv` | ETF ownership overlap input | Shows how much of an ETF is allocated to each bank. Shared ETF ownership can synchronize selling pressure. | Holding weight by bank. |

## Generated CSVs

These are gitignored because they are reproducible and can become large.

| CSV | Created By | Role | Why It Matters |
| --- | --- | --- | --- |
| `data/raw/market_prices.csv` | `python scripts/download_data.py` | Live Yahoo Finance price panel. | Provides current bank, ETF, CAD, oil, gold, TSX, and VIX inputs. |
| `data/raw/boc_yields.csv` | `python scripts/download_data.py` | Live Bank of Canada policy-rate and yield data. | Explains funding, mortgage, valuation, and yield-curve pressure on banks. |
| `data/processed/prices.csv` | `python scripts/build_features.py` | Clean aligned price table. | Used by network, stress test, and RL backtest pages. |
| `data/processed/model_dataset.csv` | `python scripts/build_features.py` | Main feature table. | Combines returns, volatility, drawdowns, beta, correlations, macro changes, and contagion score. |

## Feature Families in `model_dataset.csv`

| Feature Pattern | Plain-English Meaning | Decision Use |
| --- | --- | --- |
| `*_ret_1d`, `*_ret_5d`, `*_ret_21d` | Short-term returns. | Shows immediate and monthly pressure. |
| `*_vol_21d`, `*_vol_63d` | Annualized rolling volatility. | Higher values mean the position is riskier even before large losses occur. |
| `*_drawdown_63d` | Distance from recent 63-day high. | Measures downside pressure and investor confidence. |
| `*_dist_52w_high` | Distance from 52-week high. | Shows medium-term damage versus recent peak. |
| `*_beta_xfn_63d` | Sensitivity to Canadian financials ETF. | Measures sector contagion exposure. |
| `avg_pairwise_corr_63d` | Average Big Six correlation. | High correlation weakens diversification. |
| `policy_rate`, `ca_2y`, `ca_5y`, `ca_10y` | Canadian policy and yield levels. | Explains funding, mortgage, and valuation pressure. |
| `slope_10y_2y`, `curvature` | Yield-curve shape. | Flat or inverted curves can pressure net interest margins and recession sentiment. |
| `VIX_level`, `VIX_chg_5d` | Global volatility and volatility shock. | Captures risk-off/liquidity pressure. |
| `contagion_risk_score` | Composite 0-100 systemic stress score. | Sets monitoring intensity and portfolio defensiveness. |

## Refresh Workflow

```bash
python scripts/download_data.py
python scripts/build_features.py
```

If live data is unavailable, the pipeline falls back to sample data so the dashboard remains reproducible.
