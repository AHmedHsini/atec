"""Mission Control HTTP board. Owner: Hammoud."""

from __future__ import annotations

import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from crater.control.board import MissionBoard

STATIC = Path(__file__).resolve().parent / "static"


class ControlHandler(SimpleHTTPRequestHandler):
    board: MissionBoard

    def __init__(self, *args, directory: str | None = None, **kwargs):
        super().__init__(*args, directory=directory, **kwargs)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/api/state":
            body = json.dumps(self.board.snapshot()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/":
            self.path = "/index.html"
        super().do_GET()

    def log_message(self, fmt: str, *args) -> None:
        return


def serve(board: MissionBoard, port: int = 8765) -> ThreadingHTTPServer:
    ControlHandler.board = board
    handler = partial(ControlHandler, directory=str(STATIC))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    return server
