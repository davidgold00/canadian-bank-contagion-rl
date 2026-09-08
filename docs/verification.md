# Verification record

Verified on September 8, 2026 against the corrected research-review implementation.

## Automated behavior

- Full Python suite: 122 passed. This includes provenance failure behavior, chronological labels, next-close execution, model/schema identity, all nine portfolio assets, frozen-comparison reconciliation, atomic local publication, stale publication rejection, metric accounting, and retained legacy functionality.
- Browser JavaScript suite: 3 passed. It covers current/new/unavailable/malformed/offline publication manifests and confirms the scenario recurrence against the saved Python path.
- Static rendering succeeds without scientific or machine-learning packages. All root and pinned links are validated before publication.

## Interaction and visual review

- Inspected Overview, Risk, Decision, Performance, Scenarios, Models, and Research at 320, 390, 768, and 1280 CSS pixels.
- No page-level horizontal overflow or clipped headings was observed at those widths. Tables remain explicitly scrollable and labeled. The risk and performance charts retain readable HTML titles, numerical ranges, legends, period text, and downloadable dated values at 390 pixels.
- Keyboard-visible focus, mobile navigation, disclosures, update status, scenario selection, severity, reset, and pinned-case links were exercised.
- All five scenario presets were exercised at 50%, 100%, and 150%. Each produced six result rows; liquidity at 150% saturated all six banks and the interface described the tie. Reset returned to Liquidity squeeze at 100%.
- The public update check reported the displayed case as current and explicitly stated that no download or training job started. The hidden update link computed to `display: none` until a different case is available.
- Print styling and the deterministic review brief are present. Browser print/PDF output depends on the visitor's print settings and was not represented as a byte-identical artifact.

## Research artifacts

- New market and Bank of Canada inputs were downloaded with fallback disabled and hashed. Features run through September 4, 2026.
- Corrected classifier evaluation uses frozen chronological boundaries and compares against persistence on identical final-test rows.
- PPO training was repeated as `verified-v2.1` after adding pre-training source capture. Seeds 17, 42, and 83 reproduced the first corrected run's validation rewards and final-test ending wealth exactly. Seed 83 remained selected by validation reward before test comparison.
- The matched final period is June 28, 2024 through September 4, 2026. It does not include the 2020 pandemic crash.

## Limits of this review

This is implementer verification, not an independent human comprehension study. Historical macro release vintages, verified ETF issuer look-through, real execution/slippage, taxes, and positive cash returns remain unavailable. Production deployment is recorded only after the live manifest and every route are checked.
