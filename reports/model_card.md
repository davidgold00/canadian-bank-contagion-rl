# Northern Signal — model card, corrected research v2

## Intended use
A bounded research comparison, not autonomous investing, bank-failure prediction or proof of tactical superiority. CVaR proposes an allocation; evidence and editorial guidance determine a separate research-review disposition. PPO and classifiers are supporting experiments.

## Input and evaluation identity
The initial corrected dataset has features through September 4, 2026. Frozen classifier training begins April 3, 2012; validation begins May 2, 2022; final testing begins June 28, 2024. Classifier last eligible feature date is August 28, 2026 with five-observation outcomes available through September 4. Portfolio evaluation ends September 4. These boundaries do not slide with refresh. Exact file and artifact hashes, schema and package versions are in the saved manifests.

## PPO architecture and observations
Stable Baselines3 PPO MlpPolicy defaults; continuous nine-asset actions, long-only softmax, existing 22% direct-bank and 80% financial-proxy handling. The observation contains 21 sessions of returns through the decision close, forty named market features and post-fill holdings. The first corrected schema freezes the formerly positional intended features by name, including BMO, BNS, CAD/USD, oil and CM return/volatility/drawdown features. It does not explicitly include the later aggregate composite score or graph centrality. Return history can convey stress indirectly. Missing numerical feature entries use zero; missing tradable prices are rejected.

Three recorded seeds: 17, 42, 83. Each requested 100,000 steps and completed 100,352 because PPO finishes rollout blocks. Training uses only the frozen training partition. Validation mean unchanged reward selects seed 83. All seeds are evaluated and reported; seed 42's better final ending wealth does not change selection. No final-test-driven tuning or architecture/reward redesign occurred.

## Reward and execution
Reward = portfolio return − .20 asset-return dispersion − .30 absolute drawdown − .02 risky turnover − .15 score/100 − .10 same score/100 + .25 excess return versus XFN. The market dispersion and score terms do not directly depend on the allocation. Moving to cash does not directly remove them. They do not prove systemic-risk avoidance. The site includes a fixed-state numerical decomposition.

The next-close ledger used in training and evaluation queues a close-t decision for close t+1 after existing holdings earn that price movement. Costs and constraints match. The old saved PPO had unverified training scope and inconsistent timing; it remains a separate legacy artifact, never a compatible substitute.

## Supervised models
Random forest: 180 trees, depth 5, minimum leaf 10, balanced classes, seed 42. Logistic regression: standardized training inputs, balanced classes, maximum 1,500 iterations. Model selection uses validation average precision. Labels use only the training-outcome 80th percentile, exclude unavailable outcomes and purge boundary crossings. Final classifier output concerns the site's own persistent score. AUROC, average precision, prevalence and confusion counts are reported against current-score persistence and training prior on identical rows.

## Findings and limitations
In the initial corrected final test, the selected random forest's AUROC is about .922 and average precision .342. Persistence reaches about .952 and .471. Incremental prediction value is not established by the high AUC headline. PPO seed 83 ends near CAD 193,431; seed selection used validation, not that outcome. Compare the precise same-period simple allocation results on Performance before discussing incremental portfolio value.

The corrected study uses newly verified downloaded inputs, a different frozen period and corrected execution, so changes from legacy returns cannot be attributed solely to any one fix. Historical macro release timestamps/data vintages remain incomplete. ETFs lack verified issuer look-through. Costs, cash return and liquidity are simplified. One historical final period and three seeds are not general robustness evidence. Negative results are retained.

## Promotion and reproducibility
A compatible artifact requires recorded feature/asset order, matching hashes, method version and frozen training/evaluation protocol. New input versions or timing changes require new run identities. Data refresh and rendering cannot train or select a model. The owner must deliberately invoke the separate training/evaluation pipeline. Exact attempted/completed steps, architecture, dependencies, seed results, selected checkpoint and stopping condition are recorded in artifacts/evaluations/verified-v2.1/ppo.
