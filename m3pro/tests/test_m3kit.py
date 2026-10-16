"""roboarm.m3kit: the kit detector's second opinion -- asked over HTTP, quiet without its service,
and set against the depth's finds by their boxes in the picture."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from roboarm import m3kit


def find(kind, size, length, u, v, w=40, h=40):
    return {"kind": kind, "size": size, "length": length,
            "outline_px": [[u, v], [u + w, v], [u + w, v + h], [u, v + h]]}


CUBE30 = find("cube", 0.030, 0.030, 100, 100)
BLOCK = find("block", 0.030, 0.060, 300, 200, 80, 40)
TALL = find(None, None, None, 500, 300)


def test_the_depth_and_the_detector_set_side_by_side():
    detections = [{"label": "cube30", "confidence": 0.912, "box": [102, 98, 40, 42]},
                  {"label": "cube40", "confidence": 0.55, "box": [305, 198, 75, 44]},   # on the block
                  {"label": "cube30", "confidence": 0.41, "box": [20, 400, 30, 30]}]    # nothing there
    c = m3kit.compare([CUBE30, BLOCK, TALL], detections)
    assert c["agree"] == [("cube30", 0.91)]
    assert c["differ"] == [("block", "cube40", 0.55)]
    assert c["depth_only"] == [None] and c["detector_only"] == [("cube30", 0.41)]
    assert m3kit.line(c) == ("kit detector: agrees on 1 of 3; depth block, detector cube40 0.55; "
                             "it missed something; only it saw cube30 0.41")
    assert m3kit.line(m3kit.compare([], [])) == "kit detector: agrees on 0 of 0"


def test_no_service_is_no_answer_and_a_service_hears_the_picture(monkeypatch):
    monkeypatch.setattr(m3kit, "URL", "http://127.0.0.1:9/detect")   # nothing listens there
    assert m3kit.ask(np.zeros((480, 640, 3), np.uint8)) is None

    heard = []

    class Service(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            heard.append(self.rfile.read(int(self.headers["Content-Length"])))
            body = json.dumps({"model": "yolo26n_kit.engine", "detections": [
                {"label": "block", "confidence": 0.8, "box": [1, 2, 3, 4]}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Service)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setattr(m3kit, "URL", f"http://127.0.0.1:{server.server_port}/detect")
        assert m3kit.ask(np.zeros((480, 640, 3), np.uint8)) == [
            {"label": "block", "confidence": 0.8, "box": [1, 2, 3, 4]}]
        assert heard[0][:2] == b"\xff\xd8"   # a JPEG
    finally:
        server.shutdown()
