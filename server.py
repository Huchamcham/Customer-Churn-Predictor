"""Serve the Churn Predictor interface on this computer."""

import argparse
import csv
import io
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from data import parse_customers, sample_customers, write_customer_csv
from run import analyze


WEB = Path(__file__).resolve().parent / "web"
MAX_BYTES = 2 * 1024 * 1024
MAX_ROWS = 10000
ASSETS = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"),
          "/style.css": ("style.css", "text/css")}


class Handler(BaseHTTPRequestHandler):
    timeout = 15

    def send_content(self, status, mime, body, filename=None):
        self.send_response(status)
        self.send_header("Content-Type", f"{mime}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; base-uri 'none'; frame-ancestors 'none'")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status, payload):
        self.send_content(status, "application/json", json.dumps(payload).encode("utf-8"))

    def allow_request(self):
        host = self.headers.get("Host")
        port = self.server.server_port
        if host in (f"127.0.0.1:{port}", f"localhost:{port}") and self.headers.get("Origin") in (None, f"http://{host}"):
            return True
        self.send_json(403, {"error": "Open the local address printed by Churn Predictor."})
        return False

    def do_GET(self):
        if not self.allow_request():
            return
        path = urlsplit(self.path).path
        if path in ASSETS:
            filename, mime = ASSETS[path]
            self.send_content(200, mime, (WEB / filename).read_bytes())
        elif path == "/demo.csv":
            buffer = io.StringIO(newline="")
            write_customer_csv(buffer, sample_customers())
            self.send_content(200, "text/csv", buffer.getvalue().encode("utf-8"), "demo_customers.csv")
        else:
            self.send_json(404, {"error": "Page not found."})

    def do_POST(self):
        if not self.allow_request():
            return
        url = urlsplit(self.path)
        if url.path != "/api/analyze":
            self.send_json(404, {"error": "Page not found."})
            return
        mime = self.headers.get_content_type()
        if mime not in ("application/json", "text/csv"):
            self.send_json(415, {"error": "Choose demo data or upload a CSV file."})
            return
        try:
            share = float(parse_qs(url.query).get("contact", ["10"])[0]) / 100
            if not 0.01 <= share <= 1:
                raise ValueError("Set the contact budget between 1% and 100%.")
            length = int(self.headers.get("Content-Length", "0"))
            if not 1 <= length <= MAX_BYTES:
                raise ValueError("Choose a non-empty CSV file of 2 MB or less.")
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError("The upload ended early. Choose the file again.")
            demo = mime == "application/json"
            if demo and json.loads(body) != {"demo": True}:
                raise ValueError("Choose demo data or upload a CSV file.")
            rows = sample_customers() if demo else parse_customers(io.StringIO(body.decode("utf-8-sig"), newline=""), MAX_ROWS)
            result = analyze(rows, share)
        except UnicodeError:
            self.send_json(400, {"error": "Save your CSV as UTF-8 and upload it again."})
            return
        except (ValueError, csv.Error) as error:
            self.send_json(400, {"error": str(error)})
            return
        ranking = io.StringIO(newline="")
        result.write_ranking(ranking)
        self.send_json(200, {
            "split": [len(partition) for partition in result.partitions],
            "comparison": result.comparison,
            "selected": result.selected,
            "holdout": result.test,
            "preview": [{"id": row.customer_id, "score": score, "churned": row.churned}
                        for row, score in result.ranked[:25]],
            "rankingCsv": ranking.getvalue(),
            "report": result.markdown(share, demo),
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=0, help="Local port, or 0 to pick a free port")
    parser.add_argument("--open", action="store_true", help="Open the default browser")
    args = parser.parse_args()
    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        url = f"http://127.0.0.1:{server.server_port}/"
        print(f"Churn Predictor is running at {url}\nClose this window or press Ctrl+C to stop.", flush=True)
        if args.open:
            threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
