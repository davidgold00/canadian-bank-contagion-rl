# Simplified Institutional-Risk Platform

## Product narrative

Market conditions → systemic contagion risk → scenario analysis → portfolio response → validated investment decision.

## Information architecture

| Primary route | Consolidated capabilities | Local sections |
| --- | --- | --- |
| `/` | Executive market overview and analytical journey | Current state, risk drivers, portfolio recommendation, workflow |
| `/risk` | Systemic bank network and contagion risk score | Network, Composite score |
| `/scenarios` | Stress testing lab | Assumptions, Transmission, Portfolio impact, Risk response |
| `/models` | RL agent, CVaR optimizer, comparison, model validation | RL strategy, CVaR strategy, Comparison, Validation |
| `/decision` | Investment decision center | Recommendation, Evidence, Rebalance plan, Change conditions |
| `/performance` | Performance tracker and CVaR paper fund | Strategy performance, Paper portfolio, Benchmarks, Drawdowns and activity |
| `/research` | Data catalog and supporting documentation | Data, Methodology, Assumptions, Limitations |

## Legacy route mapping

- `/market-overview` → `/`
- `/systemic-bank-network` → `/risk#network`
- `/contagion-risk-score` → `/risk#composite-score`
- `/stress-testing-lab` → `/scenarios`
- `/rl-portfolio-agent` → `/models#rl-strategy`
- `/cvar-optimization-lab` → `/models#cvar-strategy`
- `/rl-vs-cvar-comparison` → `/models#comparison`
- `/model-validation` → `/models#validation`
- `/investment-decision-center` → `/decision`
- `/performance-tracker` → `/performance#strategy-performance`
- `/cvar-paper-fund` → `/performance#paper-portfolio`
- `/data-catalog` → `/research#data`

## Guardrails

- Preserve authoritative calculations and generated values.
- Label historical, simulated, paper, and model-derived outputs explicitly.
- Keep research-use limitations persistent but subordinate.
- Use consistent risk, model-output, portfolio-recommendation, and decision terminology.
- Verify keyboard navigation, mobile layout, legacy redirects, links, tests, and static generation.
