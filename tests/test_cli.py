import contextlib
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from op_show.cli import _handler, main
from op_show.model import validate_model


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "cuda_tiled_gemm.json"


class CliTests(unittest.TestCase):
    def _run(self, argv):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
            code = main(argv)
        return code, stderr.getvalue()

    def test_validate_exit_codes(self):
        code, stderr = self._run(["validate", str(EXAMPLE)])
        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / "bad.json"
            bad.write_text("{not json")
            code, stderr = self._run(["validate", str(bad)])
            self.assertEqual(code, 1)
            self.assertIn("Invalid model", stderr)
            self.assertNotIn("Traceback", stderr)

    def test_serve_bind_failure_is_friendly(self):
        code, stderr = self._run(
            ["serve", str(EXAMPLE), "--port", "70000", "--no-browser"])
        self.assertEqual(code, 2)
        self.assertIn("Cannot listen", stderr)
        self.assertNotIn("Traceback", stderr)

    def test_http_routes_are_read_only_and_whitelisted(self):
        entries = [validate_model(EXAMPLE)]
        server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(entries))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                with urllib.request.urlopen(base + "/") as response:
                    self.assertEqual(response.status, 200)
                    self.assertIn("text/html", response.headers["Content-Type"])
                with urllib.request.urlopen(base + "/catalog.json") as response:
                    catalog = json.loads(response.read())
                    self.assertEqual(len(catalog), 1)
                with urllib.request.urlopen(base + "/models/0.json") as response:
                    model = json.loads(response.read())
                    self.assertEqual(model["title"], "CUDA C++ · Tiled GEMM")
                with urllib.request.urlopen(base + "/sources/0.txt") as response:
                    self.assertIn("tiled_gemm", response.read().decode("utf-8"))
                for route in ("/nope.js", "/../op_show/cli.py", "/models/9.json"):
                    with self.assertRaises(urllib.error.HTTPError) as failure:
                        urllib.request.urlopen(base + route)
                    self.assertEqual(failure.exception.code, 404)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
