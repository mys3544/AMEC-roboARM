"""The kit detector's second opinion at every look (PLAN.md M5, agreed 2026-10-15): the M3's own
vision service (vision_service, robot/roboarm-vision.service on 127.0.0.1:8760 -- YOLO26n-seg
trained on both robots' looks and Isaac Sim's, the model the X3 serves too) asked about the very
picture the depth judged. The depth stays in charge; the pick logs the two side by side and the
look's .json keeps the detector's answer, so the log is the detector's score on the M3. Quiet when
the service is not there.
"""

from __future__ import annotations

import json
import urllib.request

import cv2
import numpy as np

URL = "http://127.0.0.1:8760/detect"
TIMEOUT_S = 1.0     # a 3 ms model: more than this is a service in trouble, not worth the pick's time
SAME_IOU = 0.3      # a detection and a depth find whose boxes overlap so much are the same thing


def ask(image: np.ndarray) -> list[dict] | None:
    """The detector's finds in `image` (BGR), each {label, confidence, box: [x, y, w, h] px,
    polygon?}; None when the service does not answer."""
    _, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    request = urllib.request.Request(URL, data=jpeg.tobytes(), headers={"Content-Type": "image/jpeg"},
                                     method="POST")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as reply:
            return json.loads(reply.read()).get("detections", [])
    except (OSError, ValueError):
        return None


def label(o: dict) -> str | None:
    """A depth find's name in the detector's classes (tools/kit_dataset.py's), None for no kit piece."""
    if o["kind"] == "block":
        return "block"
    if o["kind"] == "cube" and o["size"] is not None:
        return f"cube{round(o['size'] * 1000)}"
    return None


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    w = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    h = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    return w * h / max(aw * ah + bw * bh - w * h, 1e-9)


def compare(found: list[dict], detections: list[dict]) -> dict:
    """The depth's finds against the detector's, matched by their boxes in the picture, the best
    overlap first: {"agree": [(label, conf)], "differ": [(depth's, detector's, conf)],
    "depth_only": [label or None], "detector_only": [(label, conf)]}."""
    boxes = []
    for o in found:
        u, v = np.asarray(o["outline_px"], float).reshape(-1, 2).T
        boxes.append((u.min(), v.min(), u.max() - u.min(), v.max() - v.min()))
    pairs = sorted(((iou(b, d["box"]), i, j) for i, b in enumerate(boxes) for j, d in enumerate(detections)),
                   reverse=True)
    used_o, used_d, out = set(), set(), {"agree": [], "differ": [], "depth_only": [], "detector_only": []}
    for overlap, i, j in pairs:
        if overlap < SAME_IOU or i in used_o or j in used_d:
            continue
        used_o.add(i)
        used_d.add(j)
        mine, theirs = label(found[i]), detections[j]
        if mine == theirs["label"]:
            out["agree"].append((mine, round(theirs["confidence"], 2)))
        else:
            out["differ"].append((mine, theirs["label"], round(theirs["confidence"], 2)))
    out["depth_only"] = [label(o) for i, o in enumerate(found) if i not in used_o]
    out["detector_only"] = [(d["label"], round(d["confidence"], 2)) for j, d in enumerate(detections)
                            if j not in used_d]
    return out


def line(c: dict) -> str:
    """The comparison as the pick's log line."""
    parts = [f"agrees on {len(c['agree'])} of {len(c['agree']) + len(c['differ']) + len(c['depth_only'])}"]
    parts += [f"depth {mine or 'something'}, detector {theirs} {conf:.2f}" for mine, theirs, conf in c["differ"]]
    if c["depth_only"]:
        parts.append("it missed " + ", ".join(m or "something" for m in c["depth_only"]))
    if c["detector_only"]:
        parts.append("only it saw " + ", ".join(f"{n} {conf:.2f}" for n, conf in c["detector_only"]))
    return "kit detector: " + "; ".join(parts)
