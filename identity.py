"""Who is the streamer? Face recognition so shorts never crop the wrong face.

Streams are full of faces that aren't the streamer: game characters and skins,
lobby art, teammates' webcams, guests. The streamer is simply the person who shows
up the most over the whole stream. This module groups faces by identity (OpenCV's
SFace recognizer) and remembers each streamer's face in clips/<name>/.streamer.json,
so the next clip, scan or live session already knows who to look for.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

MODELS = Path(__file__).resolve().parent / "models"
YUNET = MODELS / "face_detection_yunet_2023mar.onnx"
SFACE = MODELS / "face_recognition_sface_2021dec.onnx"
SAME_PERSON = 0.36  # SFace cosine-similarity threshold for "same person"
_local = threading.local()


def available() -> bool:
    return YUNET.exists() and SFACE.exists()


def _models(w: int, h: int):
    import cv2
    m = getattr(_local, "m", None)
    if m is None:
        m = _local.m = {"rec": cv2.FaceRecognizerSF.create(str(SFACE), ""), "det": None, "size": None}
    if m["size"] != (w, h):
        m["det"] = cv2.FaceDetectorYN.create(str(YUNET), "", (w, h), score_threshold=0.75)
        m["size"] = (w, h)
    return m["det"], m["rec"]


def faces(img, min_size: int = 16) -> list[tuple[tuple[float, float, float, float], np.ndarray]]:
    """[((x, y, w, h), embedding), ...] for every face in a BGR image."""
    h, w = img.shape[:2]
    det, rec = _models(w, h)
    _, found = det.detect(img)
    out = []
    for row in found if found is not None else []:
        if row[2] < min_size or row[3] < min_size:
            continue
        try:
            e = rec.feature(rec.alignCrop(img, row)).flatten().astype(np.float32)
        except Exception:
            continue
        n = float(np.linalg.norm(e))
        if n > 0:
            out.append(((float(row[0]), float(row[1]), float(row[2]), float(row[3])), e / n))
    return out


def same(a, b) -> float:
    return float(np.dot(np.asarray(a, np.float32), np.asarray(b, np.float32)))


class StreamerIdentity:
    """Collects faces seen over time and works out which person is the streamer."""

    def __init__(self, path: Path | None = None):
        self.path = path
        self.lock = threading.Lock()
        self.groups: list[dict] = []  # {"c": centroid, "frames": set, "size": avg face size}
        self.frames = 0
        self.known = None  # remembered face from earlier sessions
        if path and path.exists():
            try:
                self.known = np.asarray(json.loads(path.read_text())["face"], np.float32)
            except (OSError, ValueError, KeyError):
                self.known = None

    def add_frame(self, found: list, frame_id=None) -> None:
        """found: output of faces() for one frame."""
        with self.lock:
            self.frames += 1
            fid = frame_id if frame_id is not None else self.frames
            for (x, y, w, h), e in found:
                best, sim = None, SAME_PERSON + 0.04
                for g in self.groups:
                    s = same(g["c"], e)
                    if s > sim:
                        best, sim = g, s
                if best is None:
                    self.groups.append({"c": e.copy(), "frames": {fid}, "size": w})
                else:
                    best["frames"].add(fid)
                    best["c"] = best["c"] * 0.85 + e * 0.15
                    best["c"] /= np.linalg.norm(best["c"])
                    best["size"] = best["size"] * 0.9 + w * 0.1
            if len(self.groups) > 60:  # forget one-off faces (skins, crowd shots)
                self.groups.sort(key=lambda g: -len(g["frames"]))
                self.groups = self.groups[:40]

    def streamer(self):
        """The streamer's face embedding, or None if it isn't clear yet."""
        with self.lock:
            ranked = sorted(self.groups, key=lambda g: -len(g["frames"]))
            # someone we already know from earlier sessions wins if they're here at all
            if self.known is not None:
                for g in ranked:
                    if same(g["c"], self.known) >= SAME_PERSON and len(g["frames"]) >= 2:
                        return g["c"]
                if not ranked or len(ranked[0]["frames"]) < 6:
                    return self.known
            if not ranked:
                return None
            top = len(ranked[0]["frames"])
            second = len(ranked[1]["frames"]) if len(ranked) > 1 else 0
            if top >= 4 and top >= 0.25 * self.frames and top >= 1.3 * second:
                return ranked[0]["c"]
            return None

    def save(self) -> None:
        face = self.streamer()
        if self.path is None or face is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"face": [round(float(v), 5) for v in face]}))
            self.known = np.asarray(face, np.float32)
        except OSError:
            pass


def identity_file(out_dir: Path, channel: str) -> Path:
    return Path(out_dir) / channel / ".streamer.json"
