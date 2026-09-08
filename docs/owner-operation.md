# Owner operation

Northern Signal publishes complete, immutable static cases. Visitors can check a JSON manifest for another published case. They cannot download market data, train, or deploy. Existing cases remain pinned when an update is available.

## Four explicit stages

1. **Data and current proposal:** `python scripts/rebuild_research.py --download`. Downloads into a unique staging directory, validates provenance and completeness, compares historical inputs, prepares a current paper path/proposal and renders a complete release. It reads the frozen evaluation; it never trains or changes final-test boundaries.
2. **Training (only intentional new research):** `python scripts/train_research.py --dataset DATASET --output NEW_EVALUATION_DIRECTORY --ppo --steps 100000`. Three seeds, unchanged architecture and reward. New input versions require new run directories. Never overwrite the baseline.
3. **Evaluation (only intentional new research):** `python scripts/evaluate_research.py --dataset DATASET --evaluation NEW_EVALUATION_DIRECTORY`. Freezes the matched comparison once. Update `configs/research-case.json` only after inspecting the new artifacts. A refresh never invokes this stage.
4. **Publication:** render with `python scripts/render_release.py`, preview with `python scripts/serve_site.py`, then deploy the complete release. Vercel serves `build/current`; Python rendering requires only the standard library. Research rebuilds use the pinned research requirements.

For the checked-in baseline, use `python scripts/rebuild_research.py` without `--download`. It verifies existing input hashes, prepares the case and publishes locally through one atomic symlink swap. Failed rendering or validation leaves the prior symlink active. Local serving is read-only, bound to localhost, and resolves each request to a complete release. GET or POST `/api/refresh` cannot start work.

## Authenticated GitHub workflow

`.github/workflows/refresh-research.yml` is a manual workflow. It requires repository access. Manual workflows must exist on the default branch (`main`) before GitHub exposes the Run workflow control. The implementation branch can be selected once that prerequisite is satisfied.

Configure the GitHub environment `research-production` with secrets `VERCEL_TOKEN`, `VERCEL_ORG_ID`, `VERCEL_PROJECT_ID`. Use the existing `canadian-bank-contagion-rl` Vercel project. Credentials are never put in HTML or JSON. A linked local CLI login is not a GitHub Actions secret.

The job checks out the requested revision, installs the tested Python/Node dependencies, runs the staged rebuild and tests, creates a preview deployment, checks its expected manifest and all routes, then promotes exactly that deployment. Concurrency prevents overlapping publications. A final production-manifest check establishes success. Logs and the built artifact bundle upload even on failure. Review/approval protection can be configured on the GitHub environment without changing the public site.

Use the workflow's `download` input for fresh data. The default is a reproducible build from the pinned dataset. Historical vendor revisions (including adjusted-price revisions) cause fresh-data builds to stop for explicit research review. Reacquire into a new intake directory and deliberately train/evaluate that input version; do not silently reinterpret an old final-test result. This restriction may prevent an everyday update after a dividend adjustment. It is intentional evidence protection, not a promise of uninterrupted live pricing.

## Failure and retention

Every owner run has a stage log under `build/jobs/`. A failed provider request cannot substitute sample data. A failed data or evidence check cannot advance the publication pointer. The most recent good Vercel deployment stays active until a complete preview passes. Publication verification compares the exact snapshot ID, not merely an HTTP 200.

Pinned cases are retained in the publication bundle. Before a new deployment, the workflow restores all pinned cases from the live publication catalogue, verifying every file hash. If an existing v2 archive cannot be restored, publication stops. A date/build guard rejects out-of-order promotions. The first transition publishes the retained September 4 legacy evidence separately. Retain GitHub artifacts and archive approved releases outside their retention period if you need indefinite rollback. Repository baseline case/evaluation/data artifacts remain reproducible independently of workflow retention.

## Local commands

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-research.txt
.venv/bin/python scripts/rebuild_research.py
.venv/bin/python -m pytest tests/test_research_v2.py tests/test_publication.py
node --test tests/web.test.mjs
.venv/bin/python scripts/serve_site.py --port 8000
```

No paid compute is needed. No retraining or hyperparameter search occurs as a publication side effect.
