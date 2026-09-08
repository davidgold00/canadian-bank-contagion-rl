# Northern Signal — research-review methodology v2

Northern Signal is a research workbench for examining Canadian bank market stress, reviewing a model-generated allocation, and deciding whether the evidence supports that proposal. It neither predicts bank failures nor places real trades. The case manifest connects every page to the same inputs, proposal, model artifacts and frozen evaluation.

## Observation and provenance

Fresh Yahoo Finance adjusted closes and Bank of Canada Valet observations have retrieval timestamps, requested series, row counts, per-series dates, file hashes and quality checks. Download failure cannot create synthetic production data. Demonstration data is segregated. The old cache remains origin-unverified; new downloads do not certify it retroactively.

Tradable universe, in exact order: RY.TO, TD.TO, BMO.TO, BNS.TO, CM.TO, NA.TO, XFN.TO, XIU.TO, cash. All tradable closes and VIX must be present on a usable session. Tradable prices are not filled. Macro observations are forward-aligned, then delayed one market session. Historical release times and data vintages are unavailable, so this is an explicit availability convention rather than a fully certified point-in-time archive. USD oil, gold, US indices, VIX and CAD/USD are contextual inputs; portfolio holdings are treated as CAD. Yahoo's `auto_adjust=True` applies vendor dividend/split adjustments; dividends are not credited again. Housing, CDS and ETF holdings templates are unused.

Macro identifiers: policy rate V39079; 2Y BD.CDN.2YR.DQ.YLD; 5Y BD.CDN.5YR.DQ.YLD; 10Y BD.CDN.10YR.DQ.YLD. The case lists every requested market ticker and source observation date.

## Composite market-stress score

The unchanged score averages expanding historical percentile ranks of 21-session average bank volatility, 63-session bank correlation, absolute XFN drawdown, VIX level and clipped yield-curve inversion. Score contributions are available-component ranks divided by the number available. Historical warm-up can contain fewer than five components; subsequent filling does not create new information. Equal weights and overlapping inputs are assumptions. A zero clipped inversion can rank above zero because of ties. A score of 66.8 is an index, not a 66.8% probability. Its percentile within historical composite values is a different statistic.

The Risk graph summarizes 63-session return co-movement. Scenarios and the optimizer use their own 126-return constructions. Correlations do not identify counterparty, funding or balance-sheet contagion.

## Allocation and review

The primary indicative solve uses the existing CVaR paper configuration and actual current paper holdings as its turnover prior. Risk aversion 6, CVaR penalty 8, volatility penalty 1, turnover penalty 0.20, node penalty 0.80, graph strength 0.40; 126-return lookback. Existing regime constraints can tighten the direct-bank cap, direct-Big-Six-plus-XFN proxy cap and cash floor. The source configuration and constraint values appear in the case. The alternative risk-aversion-7 snapshot remains a separate experiment, not the paper fund's next trade.

The objective and scoring math are unchanged. Clipped annual trailing-return estimates are not forecasts. A fixed-target 95% daily historical CVaR estimate uses about seven tail observations out of 126 before ties. This differs from realized rebalanced-strategy CVaR. Covariance-entry uplift is not the same percentage change in standard deviation. A binding rule and a selected weight do not identify one another's causal effect.

ETFs retain ordinary historical covariance and CVaR risk but lack the same explicit six-bank node treatment. The financial proxy is direct banks plus XFN; XIU can embed more financial exposure. Neither the proxy cap nor direct-name caps certify economic issuer exposure. Look-through remains not assessable.

The deterministic review gate blocks incomplete provenance, incompatible proposal evidence or failed optimizer constraints. Cash-guidance conflicts hold a proposal for review. Otherwise it is ready for research review, never investment approval. Editorial ranges are compared honestly rather than imposed as new optimization constraints. Undefined economic exposure budgets cannot produce a compliance pass.

## Corrected chronology and accounting

At close t, information through t produces an order for close t+1. Existing holdings earn t→t+1; the queued target fills only after that movement. New holdings earn subsequent returns. Training and inference use the same return-window endpoint, feature schema, caps, cost model and chronological ledger. Post-fill holdings enter the next observation. There is no invented open-price execution.

Initial capital is CAD 100,000, initially cash. Each risky trade incurs five basis points on absolute notional. Per-asset desired trades below 1% of pre-trade NAV are skipped. Buys are scaled when cash including costs is insufficient. Cash is a residual balance, never a BUY/SELL security. Limits apply to targets: drift, costs and thresholded trades can produce temporary realized-weight deviations, which are shown explicitly. Indicative differences, executable threshold checks and a fixed-mark cost preview are separate objects. Actual next-close prices can change an eventual trade.

Cumulative return and CAGR use original capital and include initial costs. CAGR annualizes by 252 divided by the number of return intervals. Daily volatility uses population dispersion; Sharpe assumes zero risk-free rate. Undefined ratios display N/A. The retained legacy Sortino variant uses standard deviation of negative observations, not standard downside deviation. Days below original capital is not time below a prior peak. Cash earns zero. Taxes, market impact, bid/ask dynamics and variable liquidity are omitted.

## Corrected classification

Target: composite score five dataset observations later is at or above the 80th percentile of training outcomes. Incomplete features and unavailable outcomes are excluded. Fixed chronological 70/15/15 development/validation/test boundaries are stored once; crossing future labels are purged. Preprocessing fits training data only. Existing random forest and logistic-regression families/settings are retained. Validation average precision selects the model; the decision threshold stays 0.50. Persistence ranks the current score and uses the training target threshold for its decision; score/100 is not a calibrated probability. A training-prior baseline uses only training prevalence. All final-test comparisons use identical rows. Three expanding development folds describe variation outside the final test. One final period does not establish robustness.

The legacy AUC is archived because full-sample calibration, crossing labels and false terminal negatives invalidate clean held-out interpretation. The classifier is a separate research experiment; it supplies neither portfolio confidence nor trade approval.

## PPO and fair comparisons

See the case-specific model card. Three seeds use unchanged PPO defaults and unchanged reward coefficients for 100,000 requested steps each, with complete rollout blocks recorded. Validation selects the checkpoint before final-test evaluation. Every seed is reported. Fixed observations include 21 sessions of returns, forty named features and current holdings. Feature/asset order and artifact hashes must match.

The comparison freezes dates, data hashes and starting capital. CVaR considers a target every ten observations; PPO and fixed-target references assess daily. All use the shared delay, costs, cash convention and trade threshold. Equal-weight feasible follows PPO constraints. The development-fixed allocation is solved before testing and frozen; it is never estimated from final-test average weights. Costed XFN/XIU buy-and-hold have different exposure constraints. Legacy costless references remain legacy only. Refresh updates the current review; it does not move historical test boundaries or fit models. Vendor revisions require explicit new input versions and research evaluation.

## Sensitivity diagnostics

Graph ablations change only covariance uplift, node penalty, or both, holding current inputs, mandate and prior fixed. Risk-preference sensitivity independently solves each parameter value from that same prior. Neither diagnostic changes the selected mandate nor identifies economic causality.

Scenario recurrence: s[k+1] = clip(0.70s[k] + 0.45 Aᵀs[k], 0, 100), five abstract steps. A contains positive 126-session correlations, zero diagonal, and nonzero rows normalized to sum to one. With no clipping and all nonzero rows, total stress multiplies by 1.15 each step, or 2.011357 after five steps. Liquidity mean 37 becomes 74.4202 mechanically. Uniform-network comparison preserves recurrence; no-spillover removes that mechanism and changes aggregate amplification. Zero rows and caps break the simple identity. Rank changes and ties are reported. Scenarios are neither probabilities nor portfolio losses and do not rerun optimization.

## Reproduction and scope

Download case.json, source manifests and frozen model/evaluation artifacts. Use the source revision plus source-tree hash and explicit environment requirements. Legacy documents are under docs/legacy. Owner operations, evidence changes and known limitations are documented separately. No human comprehension study has been claimed.
