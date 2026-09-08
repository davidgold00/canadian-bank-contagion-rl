# Northern Signal

A research workbench for examining Canadian bank market stress, reviewing a model-generated allocation, and deciding whether the evidence supports that proposal.

[Published site](https://canadian-bank-contagion-rl.vercel.app) · [Methodology](reports/methodology.md) · [Model card](reports/model_card.md) · [Owner operation](docs/owner-operation.md) · [Interview walkthrough](docs/interview-prep.md)

The guided experience is **Overview → Risk → Decision → Performance → review brief**. Scenarios and Models are separate supporting research. Each case freezes its evidence, source status, portfolio identity and historical evaluation. It can conclude “held for review” even when the optimizer meets its own constraints.

## Reproduce the saved site

Static rendering uses Python 3.11's standard library only:

```sh
python3 scripts/render_release.py
python3 scripts/serve_site.py --port 8000
```

Open http://127.0.0.1:8000. The server is read-only. A visitor's **Check for updated data** button compares the displayed case with a published JSON manifest; it cannot acquire data or train a model.

## Research pipeline

Use a clean Python 3.11 environment and `requirements-research.txt`. Follow [owner operation](docs/owner-operation.md) for explicit data rebuild, model training, frozen evaluation and publication commands. Verified data and model artifacts are versioned under `artifacts/intake` and `artifacts/evaluations`. `configs/research-case.json` selects the active case input and evaluation. No training is an implicit side effect of export or refresh.

```sh
python scripts/rebuild_research.py
python -m pytest -q
node --test tests/web.test.mjs
```

Fresh downloads require the explicit `--download` option. Provider failures never substitute synthetic data. Historical revisions stop for a new research evaluation rather than silently rewriting old comparisons. Local publication builds an immutable complete release and atomically changes a pointer. The owner workflow previews, verifies and promotes the complete static bundle to the existing Vercel project.

## Evidence boundaries

The composite is an index, not a crisis probability. Return correlations are co-movement, not identified causal contagion. ETFs lack verified issuer look-through. Macro observation dates are not a complete release-time/vintage archive. The unchanged PPO reward includes allocation-independent market terms. Scenarios are hypothetical intensity propagation, not portfolio loss forecasts. Historical simulation omits real liquidity/slippage, taxes and positive cash interest.

## Legacy material

The previous September 4 case evidence and model hashes are retained in `artifacts/legacy`. Superseded reports are under `docs/legacy`. Original root/public HTML and `legacy_export_static_site.py` remain archive material; Vercel's active output is `build/current`. Legacy Streamlit, old paper-trader and DQN examples are explicitly separate from the corrected published pipeline. No old NAV path was edited to improve a new comparison.

[Implementation categories and before/after changes](docs/implementation-log.md) · [Verification record](docs/verification.md)
