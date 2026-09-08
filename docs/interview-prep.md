# Explaining Northern Signal

## Ninety seconds

“I built a research workbench to separate an observation, a model proposal and a review decision. It asks whether Canadian bank market-stress evidence supports an indicative allocation under an existing paper mandate.

“The score is around 66.8 out of 100. That is an average of historical percentile ranks, not a probability of bank failure. The Risk page reconstructs the contributions so I can explain why it is elevated.

“The optimizer proposes roughly 60% XFN, 30% XIU and 10% cash using actual paper holdings and the same paper configuration. That passes its own constraints, but the editorial cash range is 20–35%. The review therefore holds the proposal. It also marks economic bank exposure unassessable because the ETF holdings are not verified at issuer level.

“I then challenge complexity against simple alternatives. Correcting validation lowered the classifier headline, and current-score persistence is stronger on both AUC and average precision. I retrained three PPO seeds under consistent next-close execution and selected the checkpoint on validation, not whichever seed looked best on the test.

“The useful outcome is a reproducible review whose conclusion can disagree with a model. I can show precisely which assumption or missing evidence would change it.”

## Five-minute walkthrough

1. **Overview:** state the user and question. A researcher reviews a dated hypothetical allocation, not a trading desk receiving automatic orders. Open the pinned case and show its evidence status.
2. **Risk:** show the three largest actual component contributions. Explain the difference between raw measurements, component percentiles, composite index and percentile of the composite. Explain a positive yield slope's nonzero clipped-zero percentile without implying inversion.
3. **Decision:** distinguish actual holdings, last scheduled target and current off-cycle proposal. Explain cash's 10-percentage-point shortfall from the editorial lower bound. Optimizer compliance and editorial endorsement are different checks. Do not describe the cap as the cause of a selected weight without a controlled comparison.
4. **Models:** show the four graph ablations. ETFs retain market risk but lack the explicit six-bank graph treatment. Read the observed allocation changes; do not claim wrapper avoidance caused the original weights without the experiment. Show persistence outperforming the selected classifier. Explain why withdrawing the old 0.97 headline was a necessary evaluation fix.
5. **Performance:** use the current table's same-period values. Give both dollars and drawdown percentage points; distinguish them from relative percentages. Explain daily versus ten-observation review schedules, costs, zero cash interest, constraints and the absence of pandemic coverage. Seed 83 remains selected even though another seed has higher test wealth.
6. **Scenarios:** use liquidity at 100%. Mean 37 becomes roughly 74.42 because uncapped totals multiply by 1.15 five times. The network redistributes stress; that aggregate rise is not newly discovered economic evidence. Compare rank changes with the uniform network. Attach the scenario to a review note, emphasizing it does not condition optimization.
7. **Review brief:** finish with the disposition and what would change it: resolve editorial guidance, obtain dated ETF look-through and assess the matched historical evidence. Download the exact case JSON.

## Answers to difficult questions

- **Is it point in time?** “Chronology is explicit and look-ahead bugs were corrected. Historical release vintages are unavailable, so I do not claim a fully certified point-in-time macro archive.”
- **Why RL?** “It is an experimental comparator. The question is whether it earns its complexity over a development-fixed allocation. Architectural nonlinearity alone is not the answer.”
- **Does the reward penalize systemic exposure?** “Some terms depend only on market state. I preserved and exposed that limitation rather than silently redesigning the reward to improve the result.”
- **What changed in the backtest?** “Data provenance, timing and the evaluation period changed together. I keep the old run so I do not falsely attribute all return differences to one correction.”
- **Can I refresh it?** “You can check for a newer published case. Only an authenticated owner build acquires data; training and evaluation require separate explicit runs.”
- **What is the business value?** “The reusable mechanism is evidence review: turn an interpretable observation into a constrained proposal, expose unassessable coverage, and record a reasoned hold rather than treating optimization as approval.”

## Describability self-review checklist

Can a visitor explain the measurement, score limits, actual allocation inputs, separate experiments, review reasons, evaluation scope, RL comparator result and reproduction path? Walk through the four core pages without opening code. If a decisive answer is hidden only in Research, bring it beside the claim. Automated tests and an implementer's walkthrough do not establish human comprehension; no independent human usability study has been conducted.
