"""Local command line and read-only preview server."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import webbrowser

from .model import ValidationError, validate_model


WEB = Path(__file__).parent / "web"


def _handler(entries):
    catalog = [{"id": i, "title": model["title"], "title_zh": model["title_zh"],
                "backend": model["backend"]}
               for i, (model, _) in enumerate(entries)]
    files = {
        "/": ("text/html; charset=utf-8", (WEB / "index.html").read_bytes()),
        "/app.js": ("text/javascript; charset=utf-8", (WEB / "app.js").read_bytes()),
        "/styles.css": ("text/css; charset=utf-8", (WEB / "styles.css").read_bytes()),
        "/catalog.json": ("application/json; charset=utf-8", json.dumps(catalog).encode("utf-8")),
    }
    for i, (model, source) in enumerate(entries):
        files[f"/models/{i}.json"] = ("application/json; charset=utf-8",
                                     json.dumps(model, ensure_ascii=False).encode("utf-8"))
        files[f"/sources/{i}.txt"] = ("text/plain; charset=utf-8", source.encode("utf-8"))

    class Handler(BaseHTTPRequestHandler):
        def _send(self, include_body):
            route = self.path.split("?", 1)[0]
            if route not in files:
                self.send_error(404)
                return
            content_type, data = files[route]
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if include_body:
                self.wfile.write(data)

        def do_GET(self):
            self._send(True)

        def do_HEAD(self):
            self._send(False)

        def log_message(self, format_string, *args):
            if self.path != "/favicon.ico":
                super().log_message(format_string, *args)

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(prog="op-show", description="Inspect source-backed tile models without a GPU")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "serve"):
        sub = commands.add_parser(command)
        sub.add_argument("model", help="model JSON file, or a directory of models for serve")
        if command == "serve":
            sub.add_argument("--host", default="127.0.0.1")
            sub.add_argument("--port", type=int, default=8765)
            sub.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    path = Path(args.model)
    if args.command == "serve" and path.is_dir():
        paths = sorted(path.glob("*.json"))
        if not paths:
            print(f"No JSON models in {path}", file=sys.stderr)
            return 1
    else:
        paths = [path]
    try:
        entries = [validate_model(item) for item in paths]
    except ValidationError as exc:
        print(f"Invalid model: {exc}", file=sys.stderr)
        return 1

    if args.command == "validate":
        model = entries[0][0]
        if model["status"] == "unknown":
            print(f"Valid but not playable: {model['note']}")
        else:
            summary = ", ".join(f"{s['id']}: {len(s['tiles'])} tiles, {len(s['steps'])} steps"
                                for s in model["scenarios"])
            print(f"Valid source-backed model ({model['backend']}): {summary}")
        return 0

    server = ThreadingHTTPServer((args.host, args.port), _handler(entries))
    url = f"http://{args.host}:{server.server_port}/"
    print(f"Open {url}", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
