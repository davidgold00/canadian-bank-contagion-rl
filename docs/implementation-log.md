# Implementation and evidence changes

## BUG FIXES

- Production data ingestion no longer writes synthetic fallback into provider-labeled paths. Newly downloaded data has provenance and hashes; old cache origin remains unverified.
- Public refresh previously called an absent Python endpoint. It now reads a published manifest and handles unchanged, newer, offline, unavailable and malformed states. It cannot start research jobs. Hidden CSS now respects `hidden`.
- Label calibration uses training outcomes only; future labels crossing partitions are purged; unavailable terminal labels are excluded. Export does not refit. The former approximately .97 AUC is archived; corrected random forest test AUROC is about .922 versus persistence .952. These are new data/method/period results, not a one-factor attribution.
- PPO now observes the same close boundary in training/inference and executes on the next close after old holdings earn that move. One ledger governs constraints, costs, drift and cash. Three seeds completed 100,352 steps each. The first corrected attempt (verified-v2) was repeated as verified-v2.1 to capture the exact training source bundle before fitting; all seed validation rewards and final-test ending values reproduced exactly. Both attempts remain saved, with no test-based selection. Seed 83 was selected on validation. Original PPO remains archived and unverified.
- The primary proposal uses actual holdings and the existing CVaR paper mandate. Its roughly 60% XFN / 30% XIU / 10% cash target replaces the unrelated 5%-cash snapshot as the primary indicative proposal. The alternative configuration is retained separately; legacy trades are not rewritten.
- Current paper holdings and the frozen historical comparison are distinct dated artifacts. Refresh cannot move evaluation boundaries or silently retrain. Model hash/schema mismatches are rejected.
- Costs and return/CAGR reporting use original capital. Undefined risk ratios show N/A. Tiny display deltas and cash labels are handled consistently.
- Publication builds immutable complete releases and atomically changes a pointer. Failures leave the last complete local release active; hosted publication promotes a tested complete deployment.

## TRANSPARENCY / CAVEATS

Narrowed the promise to a market-stress research review. Distinguishes index from probability, co-movement from causal contagion, ETF proxy from issuer look-through, proposal from order and editorial guidance from optimizer constraints. Added estimator/horizon/tail counts, exact reward input limitations, classifier persistence comparison, seed selection, period scope, zero cash interest, Sortino convention, component availability, covariance versus volatility interpretation, raw source dates and historical macro availability limitations. No scoring weights, reward coefficients, scenario coefficients, asset universe or investment mandate were changed.

## DIAGNOSTICS

Independent risk-aversion solves from identical prior holdings; four controlled graph treatments; frozen feasible equal-weight and development-fixed comparators with shared execution/costs; costed ETF references; current-score persistence and training-prior classification baselines; development folds; all PPO seeds; scenario uniform-network/no-spillover comparisons, rank changes, saturation and aggregate identity. These do not silently alter the selected optimizer.

## EXPERIENCE

All seven routes remain. The primary walkthrough is Overview → Risk → Decision → Performance → review brief. Supporting scenarios and model research are explicitly separate. A deterministic research-review disposition supplies an ending. Evidence details, reproducible downloads, responsive tables, keyboard controls, static chart fallbacks, print styling and pinned cases support explanation. Owner operation and interview walkthroughs are linked.

## Verification record

The initial 25 research behavior tests passed, including artificial next-close execution, label purging, training-threshold invariance, schema reorder rejection, all scenario presets at three severities, cap/zero-row behavior and review gating. Additional publication, frontend and visual verification is recorded in docs/verification.md after execution. Do not interpret this implementation log as proof that a hosted workflow has run; hosted status and configuration requirements are reported separately.

## Retained limitations and legacy material

Legacy source exporter/environment/trainer and September 4 downloadable evidence are preserved. Superseded methodology/model card/limitations are under docs/legacy. The current protocol lacks true historical release-time vintages, verified ETF look-through, real execution/slippage, positive cash returns and broad external validation. The reward retains allocation-independent market-state terms. Scenarios remain hypothetical. Source/model/data/evaluation identities are downloadable. Independent human comprehension testing has not occurred.
