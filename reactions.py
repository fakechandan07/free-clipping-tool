"""Streamer reactions, without needing chat.

Watches and listens to the stream itself for the things that make a moment
clip-worthy:

- Sounds: laughing, screaming/yelling, gasps, cheering (Google's YAMNet sound
  classifier, 521 kinds of sound, via onnxruntime)
- The streamer's face: laughing / shocked expressions on the facecam (FER+
  emotion model, with OpenCV's YuNet to find the face)
- What they say: phrases like "no way", "bro what", "I'm crying" (from the
  transcript the clipper already makes)

All models run locally and free. Each signal is 0..1; `ReactionTracker` keeps a
short history so the hype detector can ask "how hard is the streamer reacting
right now?".
"""

from __future__ import annotations

import re
import threading
from collections import deque
from pathlib import Path

import numpy as np

MODELS = Path(__file__).resolve().parent / "models"
YAMNET = MODELS / "yamnet" / "yamnet.onnx"
YAMNET_LABELS = MODELS / "yamnet" / "labels.txt"
FERPLUS = MODELS / "emotion-ferplus-8.onnx"
YUNET = MODELS / "face_detection_yunet_2023mar.onnx"

SR = 16000  # YAMNet works on 16 kHz mono audio
LAUGH = ("Laughter", "Giggle", "Belly laugh", "Chuckle, chortle", "Snicker", "Baby laughter")
SCREAM = ("Screaming", "Shout", "Yell", "Children shouting", "Battle cry", "Whoop")
SHOCK = ("Gasp",)
CHEER = ("Cheering", "Applause", "Crowd")
FER_CLASSES = ("neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt")


# --------------------------------------------------------------------------
# Sound: YAMNet
# --------------------------------------------------------------------------
def _mel_matrix(n_fft=512, n_mels=64, fmin=125.0, fmax=7500.0, sr=SR) -> np.ndarray:
    """Same mel filterbank as YAMNet's (TensorFlow's linear_to_mel_weight_matrix)."""
    mel = lambda f: 1127.0 * np.log1p(np.asarray(f) / 700.0)
    bins = n_fft // 2 + 1
    freqs = np.linspace(0.0, sr / 2, bins)[1:]
    spec_mel = mel(freqs)[:, None]
    edges = np.linspace(mel(fmin), mel(fmax), n_mels + 2)
    lower, center, upper = edges[:-2], edges[1:-1], edges[2:]
    w = np.maximum(0.0, np.minimum((spec_mel - lower) / (center - lower),
                                   (upper - spec_mel) / (upper - center)))
    return np.vstack([np.zeros((1, n_mels)), w]).astype(np.float32)  # DC row is zero


class SoundClassifier:
    """Laughter / screaming / gasp / cheering scores for each ~1 s of audio."""

    def __init__(self):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1  # tiny model; leave the CPU for video
        self.sess = ort.InferenceSession(str(YAMNET), opts, providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name
        labels = [l.strip() for l in YAMNET_LABELS.read_text().splitlines() if l.strip()]
        idx = lambda names: [i for i, l in enumerate(labels) if l in names]
        self.groups = {"laugh": idx(LAUGH), "scream": idx(SCREAM), "gasp": idx(SHOCK),
                       "cheer": idx(CHEER), "speech": idx(("Speech",)), "music": idx(("Music",))}
        self.labels = labels
        self.mel = _mel_matrix()
        self.window = np.hanning(402)[1:-1].astype(np.float32)  # periodic Hann, 400 samples

    def logmel(self, wav: np.ndarray) -> np.ndarray:
        """16 kHz float audio -> (frames, 64) log-mel, 25 ms windows every 10 ms."""
        n = 1 + max(0, (len(wav) - 400) // 160)
        idx = np.arange(400)[None, :] + 160 * np.arange(n)[:, None]
        frames = wav[idx] * self.window
        mag = np.abs(np.fft.rfft(frames, n=512, axis=1)).astype(np.float32)
        return np.log(mag @ self.mel + 0.001)

    def classify(self, wav: np.ndarray) -> dict:
        """wav: >= 0.96 s of 16 kHz mono float audio in [-1, 1]. Uses the last 0.96 s."""
        lm = self.logmel(wav[-15600:])
        if lm.shape[0] < 96:
            lm = np.pad(lm, ((96 - lm.shape[0], 0), (0, 0)), constant_values=np.log(0.001))
        patch = lm[-96:][None, None].astype(np.float32)
        logits = self.sess.run(None, {self.input: patch})[0][0]
        scores = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))  # model outputs logits
        return {g: float(scores[ix].max()) if ix else 0.0 for g, ix in self.groups.items()}

    def top(self, wav: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
        lm = self.logmel(wav[-15600:])
        patch = lm[-96:][None, None].astype(np.float32)
        scores = self.sess.run(None, {self.input: patch})[0][0]
        order = np.argsort(scores)[::-1][:k]
        return [(self.labels[i], round(float(scores[i]), 3)) for i in order]


# --------------------------------------------------------------------------
# Face: YuNet finds faces, FER+ reads the expression
# --------------------------------------------------------------------------
class FaceReader:
    """Happiness (laughing/smiling) and surprise (shocked) on the streamer's face.
    With face recognition (identity.py) it reads only the streamer's face, never a game
    character's or a teammate's; it also teaches the identity who the streamer is.
    Without it, the streamer is the face that stays in the same place."""

    def __init__(self, identity=None):
        self.identity = identity  # identity.StreamerIdentity, shared with the clipper
        import cv2
        import onnxruntime as ort
        self.cv2 = cv2
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self.fer = ort.InferenceSession(str(FERPLUS), opts, providers=["CPUExecutionProvider"])
        self.fer_in = self.fer.get_inputs()[0].name
        self.det = None
        self.spot: tuple[float, float, float] | None = None  # streamer face centre + size
        self.hits: deque = deque(maxlen=30)

    def _faces(self, frame):
        cv2 = self.cv2
        h, w = frame.shape[:2]
        if self.det is None or self._det_size != (w, h):
            self.det = cv2.FaceDetectorYN.create(str(YUNET), "", (w, h), score_threshold=0.75)
            self._det_size = (w, h)
        _, faces = self.det.detect(frame)
        return [] if faces is None else [tuple(float(v) for v in f[:4]) for f in faces if f[2] >= 18]

    def _pick_streamer(self, faces):
        """Remember where faces keep showing up; the steady one is the webcam."""
        for x, y, fw, fh in faces:
            self.hits.append((x + fw / 2, y + fh / 2, fw))
        if not self.hits:
            return None
        best, best_n = None, 0
        for cx, cy, fw in self.hits:
            n = sum(1 for a, b, w2 in self.hits if abs(a - cx) < 0.6 * fw and abs(b - cy) < 0.6 * fw)
            if n > best_n:
                best, best_n = (cx, cy, fw), n
        if best_n >= 3:
            self.spot = best
        if self.spot is None:
            return None
        sx, sy, sw = self.spot
        near = [f for f in faces if abs(f[0] + f[2] / 2 - sx) < 0.6 * sw and abs(f[1] + f[3] / 2 - sy) < 0.6 * sw]
        return max(near, key=lambda f: f[2]) if near else None

    def _pick_by_identity(self, frame):
        import identity as ident
        found = ident.faces(frame, min_size=14)
        self.identity.add_frame(found)
        who = self.identity.streamer()
        if who is None:  # not sure yet who the streamer is: fall back to the steady face
            return self._pick_streamer([box for box, _ in found])
        mine = [box for box, e in found if ident.same(e, who) >= ident.SAME_PERSON]
        return max(mine, key=lambda b: b[2]) if mine else None

    def read(self, frame) -> dict | None:
        """frame: BGR image (about 960 px wide). Returns emotion scores or None."""
        cv2 = self.cv2
        if self.identity is not None:
            try:
                face = self._pick_by_identity(frame)
            except Exception:
                face = self._pick_streamer(self._faces(frame))
        else:
            face = self._pick_streamer(self._faces(frame))
        if face is None:
            return None
        x, y, w, h = face
        pad = 0.12 * w
        x0, y0 = int(max(0, x - pad)), int(max(0, y - pad))
        x1, y1 = int(min(frame.shape[1], x + w + pad)), int(min(frame.shape[0], y + h + pad))
        crop = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        if crop.size == 0:
            return None
        inp = cv2.resize(crop, (64, 64)).astype(np.float32)[None, None]
        logits = self.fer.run(None, {self.fer_in: inp})[0][0]
        p = np.exp(logits - logits.max())
        p /= p.sum()
        return {name: float(v) for name, v in zip(FER_CLASSES, p)}


# --------------------------------------------------------------------------
# Words: phrases that mean "something crazy just happened"
# --------------------------------------------------------------------------
HYPE_PHRASES = re.compile(
    r"\b(no way|no no no|bro what|what the (hell|heck|f\w*)|oh my (god|gosh)|are you (kidding|serious)|"
    r"let'?s go+|i'?m (crying|dead|dying)|did you (see|just see) that|that was (insane|crazy|nuts)|"
    r"clip (it|that)|holy (shit|crap|moly)|what just happened|i can'?t believe|yo+ chat|"
    r"bro+|dude|oh no|run+|get out|why (is|did|would)|what is (that|this|happening)|"
    r"\[?(laugh|laughs|laughing|laughter|screams|screaming)\]?)\b", re.I)


def speech_hype(words: list) -> float:
    """0..1 from a stretch of transcript (objects with .text): hype phrases and
    shouty/swear-heavy talking."""
    if not words:
        return 0.0
    text = " ".join(w.text for w in words)
    phrases = len(HYPE_PHRASES.findall(text))
    swears = len(re.findall(r"\b(f+u+c+k\w*|s+h+i+t+\w*|damn|wtf)\b", text, re.I))
    exclaim = text.count("!")
    return min(1.0, 0.25 * phrases + 0.15 * swears + 0.1 * exclaim)


# --------------------------------------------------------------------------
# Live tracking for one channel
# --------------------------------------------------------------------------
class ReactionTracker:
    """Feeds live audio (and frames) through the models, keeps ~3 minutes of history,
    and answers `level(now)`: how strongly the streamer is reacting (0..1) + why."""

    def __init__(self, use_sound=True, use_face=True, identity=None):
        self.lock = threading.Lock()
        self.sound = SoundClassifier() if use_sound and YAMNET.exists() else None
        self.face = FaceReader(identity) if use_face and FERPLUS.exists() and YUNET.exists() else None
        self._pcm = np.zeros(0, dtype=np.float32)
        self._since_sound = 0
        self.frames_in = 0
        self.history: deque = deque()  # (time, kind, value)

    # -- inputs ------------------------------------------------------------
    def add_audio(self, ts: float, pcm16: bytes) -> None:
        """16 kHz mono 16-bit audio as it arrives. Classified every 0.5 s."""
        if self.sound is None:
            return
        chunk = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        self._pcm = np.concatenate([self._pcm, chunk])[-SR * 2:]
        self._since_sound += len(chunk)
        if self._since_sound >= SR // 2 and len(self._pcm) >= 15600:
            self._since_sound = 0
            s = self.sound.classify(self._pcm)
            with self.lock:
                self.history.append((ts, "laugh", s["laugh"]))
                self.history.append((ts, "scream", max(s["scream"], s["gasp"] * 0.8)))
                self.history.append((ts, "cheer", s["cheer"]))

    def add_frame(self, ts: float, frame) -> None:
        if self.face is None:
            return
        self.frames_in += 1
        e = self.face.read(frame)
        if e is None:
            return
        with self.lock:
            self.history.append((ts, "smile", e["happiness"]))
            self.history.append((ts, "shock", e["surprise"] + 0.5 * e["fear"]))

    def add_words(self, ts: float, words: list) -> None:
        with self.lock:
            self.history.append((ts, "words", speech_hype(words)))

    def stats(self, now: float, window: float = 60.0) -> dict:
        """What the models have seen recently (for checking it's working)."""
        with self.lock:
            recent = [(k, v) for t, k, v in self.history if now - window <= t <= now]
        out = {"sound_checks": sum(1 for k, _ in recent if k == "laugh"),
               "face_reads": sum(1 for k, _ in recent if k == "smile"),
               "pictures_total": self.frames_in,
               "faces_seen": len(self.face.hits) if self.face else None}
        for k in ("laugh", "scream", "smile", "shock"):
            vals = [v for kk, v in recent if kk == k]
            out[f"max_{k}"] = round(max(vals), 3) if vals else None
        return out

    # -- output ------------------------------------------------------------
    def level(self, now: float, window: float = 5.0) -> tuple[float, list[str]]:
        """0..1 reaction strength over the last `window` seconds and what drove it."""
        with self.lock:
            while self.history and self.history[0][0] < now - 200:
                self.history.popleft()
            recent = [(k, v) for t, k, v in self.history if now - window <= t <= now]
        peak = lambda k: max((v for kk, v in recent if kk == k), default=0.0)
        # YAMNet is cautious: clear laughter over game audio scores ~0.3-0.5, while normal
        # stream audio scores ~0.01. Rescale so ~0.4 counts as a full-strength laugh.
        strength = lambda p: min(1.0, max(0.0, (p - 0.08) / 0.35))
        laugh, scream = strength(peak("laugh")), strength(peak("scream"))
        cheer = strength(peak("cheer"))
        smile, shock, words = peak("smile"), peak("shock"), peak("words")
        voice = max(laugh, scream, 0.6 * cheer)
        # laughing/screaming that lasts (2+ readings in the window) counts more than a blip
        lasting = sum(1 for k, v in recent if k in ("laugh", "scream") and strength(v) >= 0.6) >= 2
        face = max(smile if smile > 0.75 else 0.0, shock if shock > 0.6 else 0.0)
        score = min(1.0, 0.6 * voice + 0.25 * face + 0.25 * words
                    + (0.15 if lasting else 0.0) + (0.15 if voice > 0.5 and face > 0 else 0.0))
        why = []
        if laugh >= 0.6:
            why.append("laughing")
        if scream >= 0.6:
            why.append("screaming")
        if cheer >= 0.6:
            why.append("cheering")
        if smile > 0.75:
            why.append("big smile")
        if shock > 0.6:
            why.append("shocked face")
        if words >= 0.5:
            why.append("hype words")
        return round(score, 3), why
