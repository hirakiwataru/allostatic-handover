"""Serve the local allostatic handover dashboard."""

from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


class DashboardHandler(SimpleHTTPRequestHandler):
    log_dir: Path = Path("outputs")
    static_dir: Path = Path(__file__).resolve().parent / "static"

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/runs":
            self._send_json(self._list_runs())
            return
        if parsed.path == "/api/run":
            query = parse_qs(parsed.query)
            run = query.get("run", [""])[0]
            self._send_json(self._load_run(run))
            return
        if parsed.path == "/api/live-runs":
            self._send_json(self._list_live_runs())
            return
        if parsed.path == "/api/live-run":
            query = parse_qs(parsed.query)
            run = query.get("run", [""])[0]
            limit = int(query.get("limit", ["2000"])[0])
            self._send_json(self._load_live_run(run, limit=limit))
            return
        if parsed.path in {"/", "/index.html"}:
            return self._send_static(self.static_dir / "index.html")
        if parsed.path == "/live.html":
            return self._send_static(self.static_dir / "live.html")
        return self._send_static(self.static_dir / parsed.path.lstrip("/"))

    def _send_json(self, value):
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_static(self, path: Path):
        root = self.static_dir.resolve()
        target = path.resolve()
        try:
            target.relative_to(root)
        except ValueError:
            self.send_error(404, "File not found")
            return
        if not target.is_file():
            self.send_error(404, "File not found")
            return

        payload = target.read_bytes()
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _list_runs(self):
        root = self.log_dir.resolve()
        runs = []
        for steps in root.rglob("steps.jsonl"):
            try:
                rel = steps.parent.resolve().relative_to(root)
            except ValueError:
                continue
            episodes = steps.parent / "episodes.csv"
            runs.append(
                {
                    "id": str(rel),
                    "name": str(rel),
                    "steps_path": str(steps),
                    "episodes_path": str(episodes) if episodes.exists() else "",
                }
            )
        return sorted(runs, key=lambda row: row["name"])

    def _list_live_runs(self):
        root = self.log_dir.resolve()
        runs = []
        for live in root.rglob("live.jsonl"):
            try:
                rel = live.parent.resolve().relative_to(root)
            except ValueError:
                continue
            runs.append(
                {
                    "id": str(rel),
                    "name": str(rel),
                    "live_path": str(live),
                    "updated_at": live.stat().st_mtime,
                }
            )
        return sorted(runs, key=lambda row: row["updated_at"], reverse=True)

    def _load_run(self, run: str):
        root = self.log_dir.resolve()
        run_dir = (root / run).resolve()
        try:
            run_dir.relative_to(root)
        except ValueError:
            return {"error": "run outside log dir", "steps": []}

        steps_path = run_dir / "steps.jsonl"
        if not steps_path.exists():
            return {"error": "steps.jsonl not found", "steps": []}

        steps = []
        with steps_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    steps.append(json.loads(line))
        return {"run": run, "steps": steps[-1000:]}

    def _load_live_run(self, run: str, limit: int = 2000):
        root = self.log_dir.resolve()
        run_dir = (root / run).resolve()
        try:
            run_dir.relative_to(root)
        except ValueError:
            return {"error": "run outside log dir", "records": []}

        live_path = run_dir / "live.jsonl"
        if not live_path.exists():
            return {"error": "live.jsonl not found", "records": []}

        records = []
        with live_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return {"run": run, "records": records[-limit:]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", default="outputs")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(argv)

    DashboardHandler.log_dir = Path(args.log_dir)
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"dashboard: http://{args.host}:{args.port}")
    print(f"log dir: {DashboardHandler.log_dir.resolve()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
