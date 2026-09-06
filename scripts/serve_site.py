"""Local server for the Northern Signal static site, with a data-refresh endpoint.

Serves the exported HTML the same way Vercel does (clean URLs, no extensions) and
exposes a small API so the refresh can be started from the site's own UI instead
of from a terminal:

    GET  /api/refresh   current job status
    POST /api/refresh   start a refresh if one is not already running

The deployed site is static and has no Python runtime, so the button on the
overview page hides itself unless this server answers. Run it with:

    python scripts/serve_site.py [--port 8000]

The refresh is atomic. Every file the pipeline overwrites is copied aside first;
if any step fails, the originals are put back, so a failed download can never
leave the site half-updated or silently backed by synthetic fallback data.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import threading
import traceback
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PAGE_SLUGS = ["index", "risk", "scenarios", "models", "decision", "performance", "research"]

# Everything the refresh rewrites. Snapshotted before the run, restored on failure.
PROTECTED_PATHS = [
    Path("data/raw/market_prices.csv"),
    Path("data/raw/boc_yields.csv"),
    Path("data/processed/prices.csv"),
    Path("data/processed/model_dataset.csv"),
    *[Path(f"{slug}.html") for slug in PAGE_SLUGS],
    Path("public"),
]


class RefreshState:
    """Status of the one refresh job the server will run at a time."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = {"status": "idle", "step": None, "steps": [], "error": None, "finished_at": None}

    def snapshot(self) -> dict:
        with self._lock:
            return json.loads(json.dumps(self._state))

    def start(self) -> bool:
        with self._lock:
            if self._state["status"] == "running":
                return False
            self._state = {"status": "running", "step": "Starting", "steps": [], "error": None, "finished_at": None}
            return True

    def step(self, message: str) -> None:
        with self._lock:
            self._state["step"] = message
            self._state["steps"].append(message)

    def finish(self, error: str | None = None) -> None:
        with self._lock:
            self._state["status"] = "error" if error else "done"
            self._state["error"] = error
            self._state["step"] = None
            self._state["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")


STATE = RefreshState()


def _snapshot(backup_dir: Path) -> None:
    for relative in PROTECTED_PATHS:
        source = ROOT / relative
        if not source.exists():
            continue
        target = backup_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)


def _restore(backup_dir: Path) -> None:
    for relative in PROTECTED_PATHS:
        saved = backup_dir / relative
        live = ROOT / relative
        if not saved.exists():
            continue
        if live.is_dir():
            shutil.rmtree(live, ignore_errors=True)
        elif live.exists():
            live.unlink()
        live.parent.mkdir(parents=True, exist_ok=True)
        if saved.is_dir():
            shutil.copytree(saved, live)
        else:
            shutil.copy2(saved, live)


def _price_stats() -> tuple[int, str | None]:
    """Row count and latest date of the current price panel, for the sanity check."""
    from src.data.market_data import load_market_prices

    try:
        prices = load_market_prices()
    except Exception:
        return 0, None
    if prices.empty:
        return 0, None
    return len(prices), str(prices.index.max().date())


def run_refresh() -> None:
    """Download, rebuild, and re-export - or put everything back as it was."""
    from scripts.download_data import BOC_SERIES, TICKERS
    from src.data.boc_valet import download_boc_series
    from src.data.build_dataset import build_dataset
    from src.data.market_data import download_market_data

    before_rows, before_date = _price_stats()
    backup_root = Path(tempfile.mkdtemp(prefix="northern-signal-refresh-"))
    backup_dir = backup_root / "backup"
    backup_dir.mkdir()

    try:
        STATE.step("Backing up current data and pages")
        _snapshot(backup_dir)

        # fallback=False so a failed download raises instead of quietly writing
        # synthetic data over the real history.
        STATE.step("Downloading market data")
        download_market_data(TICKERS, fallback=False)

        STATE.step("Downloading Bank of Canada series")
        download_boc_series(BOC_SERIES, fallback=False)

        STATE.step("Validating downloaded data")
        after_rows, after_date = _price_stats()
        if after_rows == 0 or after_date is None:
            raise RuntimeError("The refreshed price panel is empty.")
        if before_rows and after_rows < before_rows * 0.9:
            raise RuntimeError(
                f"The refreshed price panel lost rows ({before_rows} to {after_rows}); keeping the previous data."
            )
        if before_date and after_date < before_date:
            raise RuntimeError(
                f"The refreshed data ends earlier than the current data ({after_date} vs {before_date})."
            )

        STATE.step("Rebuilding the feature dataset")
        build_dataset()

        STATE.step("Regenerating pages")
        from scripts.export_static_site import write_pages

        write_pages()

        _, final_date = _price_stats()
        STATE.step(f"Data now through {final_date}")
        STATE.finish()
    except Exception as exc:
        traceback.print_exc()
        try:
            STATE.step("Refresh failed - restoring previous data and pages")
            _restore(backup_dir)
        except Exception as restore_error:
            STATE.finish(f"{exc} (restore also failed: {restore_error})")
            return
        STATE.finish(str(exc))
    finally:
        shutil.rmtree(backup_root, ignore_errors=True)


class SiteHandler(SimpleHTTPRequestHandler):
    """Static pages with Vercel-style clean URLs, plus the refresh API."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, fmt, *args):  # quieter console
        if "/api/" in str(args[0] if args else ""):
            super().log_message(fmt, *args)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.split("?")[0] == "/api/refresh":
            self._send_json(STATE.snapshot())
            return
        super().do_GET()

    def do_POST(self):
        if self.path.split("?")[0] != "/api/refresh":
            self._send_json({"error": "Unknown endpoint."}, status=404)
            return
        if not STATE.start():
            self._send_json(STATE.snapshot(), status=409)
            return
        threading.Thread(target=run_refresh, daemon=True).start()
        self._send_json(STATE.snapshot(), status=202)

    def translate_path(self, path: str) -> str:
        clean = path.split("?")[0].split("#")[0].strip("/")
        if not clean:
            return str(ROOT / "index.html")
        if clean in PAGE_SLUGS and (ROOT / f"{clean}.html").exists():
            return str(ROOT / f"{clean}.html")
        return super().translate_path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), SiteHandler)
    print(f"Northern Signal running at http://{args.host}:{args.port}")
    print("The Refresh data button on the overview page is live while this server is running.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
        server.server_close()


if __name__ == "__main__":
    main()
