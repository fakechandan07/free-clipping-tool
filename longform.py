"""Long videos: turn one or more streams into a 10-15 minute YouTube video.

Structure (like a real compilation channel):
  1. Cold open   - 6 s teaser of the craziest moment
  2. Title card  - "STREAMERS PLAY GTA 6 FOR THE FIRST TIME" (your topic)
  3. Chapter: First impressions - each streamer's first minutes
  4. Chapters    - the best moments, in story order, switching between streamers,
                   each with a title made from what they said ("Speed vs the cops")
  5. Finale      - "Everyone's reaction": all streamers' biggest reactions at once
  6. Outro card  - thanks + subscribe

Step 1 (scan) builds an editable storyboard; step 2 (render) makes the video, a
thumbnail, YouTube chapter timestamps, and a title/description.
"""

from __future__ import annotations

import itertools
import json
import re
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import sources
import vodscan
from vodscan import _fmt

import platformkit  # noqa: E402
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"
W, H, FPS = 1920, 1080, 60
COLD, TITLE, FINALE, OUTRO = 6.5, 4.5, 11.0, 20.0  # seconds (outro = YouTube end screen)
STORY = {"cops", "heist", "mission", "driving", "first"}  # moments that need more setup
MAXLEN = {"funny": 32, "hype": 40, "crazy": 38, "moment": 45, "death": 45, "money": 50, "guns": 55,
          "driving": 55, "first": 60, "cops": 70, "heist": 70, "mission": 75}
CHANNEL_NAME = "my channel"

# --------------------------------------------------------------------------
# Chapter titles from what the streamer said (no AI needed)
# --------------------------------------------------------------------------
RULES = [
    ("cops", r"\b(cops?|police|wanted|arrest\w*|sirens?|jail|chas\w+|swat)\b",
     ["{s} vs the cops", "{s} gets chased by the cops", "{s} has a wanted level"]),
    ("heist", r"\b(heist|rob\w*|bank|vault|steal\w*|stole|loot\w*)\b",
     ["{s} pulls off a heist", "{s} robs the place", "{s} goes for the money"]),
    ("driving", r"\b(cars?|driv\w+|drove|crash\w*|bike|motorcycle|boat|plane|helicopter|fly\w*|wheel)\b",
     ["{s} can't drive", "{s} goes for a ride", "{s} crashes out"]),
    ("mission", r"\b(missions?|objective|quest|story|cutscene|level|tutorial)\b",
     ["{s} starts a mission", "{s} on a mission", "{s} figures out the mission"]),
    ("money", r"\b(money|cash|millions?|rich|buy|bought|dollars?|bucks)\b",
     ["{s} gets rich", "{s} goes shopping", "{s} counts the money"]),
    ("guns", r"\b(guns?|shoot\w*|shot|weapons?|ak|rifle|pistol|kill\w*|ammo)\b",
     ["{s} goes guns blazing", "{s} starts shooting", "{s} picks a fight"]),
    ("death", r"\b(died|dead|wasted|respawn\w*|dying|die)\b",
     ["{s} gets wasted", "{s} didn't survive that"]),
]
FUNNY = ["{s} can't stop laughing", "{s} is crying laughing", "{s} loses it laughing"]
CRAZY = ["{s} loses it", "{s} can't believe it", "{s} freaks out"]
CHAT = ["Chat goes crazy for {s}", "{s} breaks chat", "{s}'s chat explodes"]
DEFAULT = ["{s}'s wildest moment", "{s} goes off", "Pure chaos with {s}"]
FIRST = ["{s}'s first minutes", "{s} jumps in for the first time", "{s}'s first impressions"]


def title_for(moment: dict, name: str, used: set) -> tuple[str, str]:
    """(category, title) for a moment, from its transcript and how people reacted."""
    text = " ".join(moment.get("words_text") or [])
    best, best_n = None, 0
    for cat, rx, titles in RULES:
        n = len(re.findall(rx, text, re.I))
        if n > best_n:
            best, best_n = (cat, titles), n
    reasons = " ".join(moment.get("reasons") or [])
    if moment.get("first"):
        options, cat = FIRST, "first"
    elif best and best_n >= 2:
        cat, options = best
    elif "laugh" in reasons or (moment.get("reaction", 0) >= 0.5 and "laugh" in text.lower()):
        options, cat = FUNNY, "funny"
    elif moment.get("peak_ratio", 0) >= 4:
        options, cat = CHAT, "hype"
    elif "scream" in reasons or moment.get("audio_boost", 0) >= 10:
        options, cat = CRAZY, "crazy"
    elif best and best_n >= 1:
        cat, options = best
    else:
        options, cat = DEFAULT, "moment"
    for o in options + DEFAULT + CRAZY:
        t = o.format(s=name)
        if t not in used:
            used.add(t)
            return cat, t
    return cat, options[0].format(s=name)


# --------------------------------------------------------------------------
# Cards and overlays (GTA-inspired, same font as the site and captions)
# --------------------------------------------------------------------------
FONT = Path(__file__).resolve().parent / "web" / "fonts" / "PassionOne-Regular.ttf"


def _font(size):
    from PIL import ImageFont
    try:
        return ImageFont.truetype(str(FONT), size)
    except OSError:
        return ImageFont.load_default(size)


def _outlined(draw, xy, text, font, fill=(255, 255, 255), stroke=10, anchor="mm", shadow=8):
    x, y = xy
    draw.text((x + shadow, y + shadow), text, font=font, fill=(0, 0, 0, 150), anchor=anchor,
              stroke_width=stroke, stroke_fill=(0, 0, 0, 150))
    draw.text((x, y), text, font=font, fill=fill, anchor=anchor, stroke_width=stroke, stroke_fill=(0, 0, 0))


def _fit(draw, text, max_w, start, min_size=60):
    size = start
    while size > min_size and draw.textlength(text, font=_font(size)) > max_w:
        size -= 6
    return _font(size)


def _wrap(text: str, max_chars: int) -> list[str]:
    words, lines, cur = [], [], ""
    for w in text.split():  # keep "GTA 6" / "PART 2" together
        if words and re.fullmatch(r"\d{1,2}|[IVX]{1,4}", w):
            words[-1] += "\u00a0" + w
        else:
            words.append(w)
    for w in words:
        if cur and len(cur) + 1 + len(w) > max_chars:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines[:3]


def sunset_background() -> Image.Image:
    stops = [(0, (26, 8, 56)), (0.4, (75, 15, 94)), (0.68, (196, 31, 110)), (0.86, (255, 122, 61)), (1, (255, 179, 71))]
    col = []
    for y in range(H):
        t = y / (H - 1)
        for (a, ca), (b, cb) in zip(stops, stops[1:]):
            if a <= t <= b:
                k = (t - a) / (b - a)
                col.append(tuple(int(ca[i] + (cb[i] - ca[i]) * k) for i in range(3)))
                break
    img = Image.new("RGB", (1, H))
    for y in range(H):
        img.putpixel((0, y), col[y])
    img = img.resize((W, H))
    d = ImageDraw.Draw(img)
    sun = Image.new("L", (W, H), 0)
    sd = ImageDraw.Draw(sun)
    cx, cy, r = W // 2, int(H * 0.86), 360
    sd.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
    for i in range(8):  # retro stripes on the lower half
        yy = cy + i * 46
        sd.rectangle([0, yy, W, yy + 10 + i * 3], fill=0)
    sun_col = Image.new("RGB", (W, H), (255, 190, 80))
    img.paste(sun_col, (0, 0), sun.point(lambda v: int(v * 0.85)))
    d.rectangle([0, H - 60, W, H], fill=(11, 6, 24))
    return img


def card(path: Path, big: str, small: str = "", top: str = "★★★★★") -> Path:
    img = sunset_background().convert("RGBA")
    d = ImageDraw.Draw(img)
    if top:
        _outlined(d, (W // 2, 190), top, _font(90), fill=(255, 210, 63), stroke=6, shadow=5)
    lines = _wrap(big.upper(), 18)
    y = H // 2 - (len(lines) - 1) * 80
    for line in lines:
        _outlined(d, (W // 2, y), line, _fit(d, line, 1700, 190), stroke=12, shadow=10)
        y += 160
    if small:
        _outlined(d, (W // 2, y + 10), small, _fit(d, small, 1500, 70, 40), fill=(255, 233, 244), stroke=6, shadow=5)
    img.convert("RGB").save(path, quality=95)
    return path


def lower_third(path: Path, title: str, kicker: str = "") -> Path:
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    font = _fit(d, title.upper(), 1300, 96, 50)
    tw = d.textlength(title.upper(), font=font)
    x0, y0 = 70, H - 250
    d.rounded_rectangle([x0 - 10, y0 - 10, x0 + tw + 70, y0 + 130], 22, fill=(11, 6, 24, 200),
                        outline=(0, 0, 0, 255), width=6)
    d.rectangle([x0 - 10, y0 - 10, x0 + 18, y0 + 130], fill=(255, 61, 139, 255))
    _outlined(d, (x0 + 45, y0 + 62), title.upper(), font, stroke=8, anchor="lm", shadow=6)
    if kicker:
        _outlined(d, (x0 + 45, y0 - 45), kicker.upper(), _font(48), fill=(255, 210, 63), stroke=5, anchor="lm", shadow=4)
    img.save(path)
    return path


def name_tag(path: Path, credit: str) -> Path:
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    font = _font(44)
    tw = d.textlength(credit, font=font)
    d.rounded_rectangle([36, 34, 36 + tw + 56, 106], 16, fill=(11, 6, 24, 170))
    _outlined(d, (64, 70), credit, font, stroke=4, anchor="lm", shadow=3)
    img.save(path)
    return path


def grid_spots(n: int) -> list[tuple[int, int]]:
    """Top-left corners of 960x540 tiles: 2 side by side in the middle, 3 as 2+1, 4 as 2x2."""
    if n <= 1:
        return [(480, 270)]
    if n == 2:
        return [(0, 270), (960, 270)]
    if n == 3:
        return [(0, 0), (960, 0), (480, 540)]
    return [(0, 0), (960, 0), (0, 540), (960, 540)]


def finale_overlay(path: Path, names: list[str], title: str) -> Path:
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    spots = [(x + 20, y + 20) for x, y in grid_spots(len(names))]
    for (x, y), n in zip(spots, names):
        font = _font(46)
        tw = d.textlength(n, font=font)
        d.rounded_rectangle([x, y, x + tw + 40, y + 66], 14, fill=(11, 6, 24, 190))
        _outlined(d, (x + 20, y + 34), n, font, stroke=4, anchor="lm", shadow=3)
    font = _fit(d, title.upper(), 1200, 110, 60)
    _outlined(d, (W // 2, H // 2), title.upper(), font, fill=(255, 210, 63), stroke=10, shadow=8)
    img.save(path)
    return path


# --------------------------------------------------------------------------
# A project
# --------------------------------------------------------------------------
class LongProject:
    _ids = itertools.count(1)

    def __init__(self, out_dir: Path, links: list[str], topic: str, minutes: int, args, pid: str | None = None):
        self.out_dir, self.args = Path(out_dir), args
        self.id = pid or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.folder = self.out_dir / "_long" / self.id
        self.links, self.topic, self.minutes = links, topic.strip() or "Best moments", int(minutes)
        self.status, self.stage, self.progress, self.message = "queued", "queued", 0.0, "Waiting to start…"
        self.streams: list[dict] = []
        self.storyboard: list[dict] = []
        self.outputs: dict = {}
        self.channel_name = CHANNEL_NAME
        self.stop = threading.Event()
        self.procs: list[subprocess.Popen] = []
        self.started = None
        self.next_action = "scan"  # scan -> review -> render
        self.opts: dict = {}  # narration / captions / jumpcuts / camzoom / music / effects

    # -- persistence ------------------------------------------------------
    def save(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        data = {k: getattr(self, k) for k in ("id", "links", "topic", "minutes", "status", "stage", "progress",
                                              "message", "streams", "storyboard", "outputs", "channel_name",
                                              "next_action", "opts")}
        clean = lambda d: {k: v for k, v in d.items() if not k.startswith("_")}  # drop temp file paths
        data["storyboard"] = [dict(clean(it), **({"parts": [clean(x) for x in it["parts"]]} if "parts" in it else {}))
                              for it in self.storyboard]
        (self.folder / "project.json").write_text(json.dumps(data, indent=1, default=str))

    @classmethod
    def load(cls, folder: Path, out_dir: Path, args):
        d = json.loads((folder / "project.json").read_text())
        p = cls(out_dir, d["links"], d["topic"], d["minutes"], args, pid=d["id"])
        for k in ("status", "stage", "progress", "message", "streams", "storyboard", "outputs", "channel_name",
                  "next_action", "opts"):
            if k in d:
                setattr(p, k, d[k])
        p._add_missing_lines()
        if p.status in ("scanning", "rendering", "queued"):  # interrupted by a restart
            p.status = "review" if p.storyboard else "failed"
            p.message = "Stopped when the clipper closed. " + ("Press Render to make the video." if p.storyboard
                                                                else "Delete it and start again.")
        return p

    def _add_missing_lines(self) -> None:
        """Storyboards made before the narrator existed have no lines: write them once."""
        import longedit as le
        if not self.storyboard or any("line" in it for it in self.storyboard):
            return
        display = {s["name"]: s.get("display", s["name"]) for s in self.streams or []}
        used: set = set()
        for k, it in enumerate(self.storyboard):
            if it.get("kind") == "title":
                it["line"] = le.intro_line(self.topic, list(display.values()))
            elif it.get("kind") == "outro":
                it["line"] = le.OUTRO_LINE
            elif it.get("kind") == "clip":
                name = display.get(it.get("stream"), it.get("stream", ""))
                kicker = (it.get("kicker") or "").lower()
                it["line"] = le.narration_line(it.get("category") or "moment", name, float(it.get("start") or 0), k,
                                               "climax" in kicker, "ending" in kicker,
                                               used if it.get("on", True) else None)

    def state(self) -> dict:
        eta = None
        if self.status in ("scanning", "rendering") and self.started and self.progress > 0.03:
            eta = int((time.time() - self.started) * (1 - self.progress) / self.progress)
        return {"id": self.id, "topic": self.topic, "minutes": self.minutes, "status": self.status,
                "stage": self.stage, "progress": round(self.progress, 3), "message": self.message, "eta": eta,
                "streams": [{k: s.get(k) for k in ("name", "display", "platform", "title", "duration", "credit",
                                                   "error", "moments_found")} for s in self.streams],
                "storyboard": self.storyboard, "outputs": self.outputs, "channel_name": self.channel_name,
                "estimate": self.estimate(), "options": self.options(),
                "music_files": len(__import__("longedit").music_files())}

    def _set(self, stage, frac, message=None):
        self.stage, self.progress = stage, max(0.0, min(1.0, frac))
        if message:
            self.message = message

    def estimate(self) -> int:
        total = 0.0
        for it in self.storyboard:
            if not it.get("on", True):
                continue
            total += {"cold": COLD, "title": TITLE, "finale": FINALE, "outro": OUTRO}.get(
                it["kind"], it.get("est", 40))
        return int(total)

    # -- step 1: scan ------------------------------------------------------
    def scan(self, log) -> None:
        self.status, self.started = "scanning", time.time()
        self._set("scan", 0, "Checking the links…")
        self.streams = []
        n = len(self.links)
        for i, link in enumerate(self.links):
            if self.stop.is_set():
                raise InterruptedError
            base = i / n
            span = 1 / n
            try:
                src = sources.resolve(link)
            except Exception as e:
                self.streams.append({"name": link[:40], "display": link[:40], "error": str(e)})
                log("long", f"skipping {link}: {e}")
                continue
            if src["kind"] != "vod":
                self.streams.append({"name": src["name"], "display": src["name"], "error":
                                     "That's a live stream. Long videos need a recorded stream or video."})
                continue
            display = (src.get("uploader") or src["name"]).strip()
            st = {"name": src["name"], "display": display, "platform": src["platform"], "title": src.get("title"),
                  "duration": src.get("duration"), "credit": src["credit"], "url": src["url"], "src": src}
            self.streams.append(st)
            self._scan_stream(st, lambda f, m: self._set("scan", (base + span * f) * 0.95, f"{display}: {m}"), log)
        good = [s for s in self.streams if not s.get("error")]
        if not good:
            raise RuntimeError("None of the links could be used: " +
                               "; ".join(s.get("error", "") for s in self.streams))
        self._set("scan", 0.97, "Building the storyboard…")
        self.storyboard = build_storyboard(self.streams, self.topic, self.minutes, self.channel_name)
        self.status, self.next_action = "review", "render"
        self._set("review", 1.0, f"Storyboard ready: about {_fmt(self.estimate())}. Check it, then press Render.")
        self.save()

    def _scan_stream(self, st: dict, report, log) -> None:
        src, dur = st["src"], st["duration"] or 0
        msgs, covered, comments = [], None, []
        report(0.02, "reading chat replay…")
        try:
            if src.get("chat") == "twitch":
                msgs, covered = sources.twitch_chat_replay(src["video_id"], dur, stop=self.stop,
                                                           progress=lambda p: report(0.02 + 0.25 * p, f"chat replay {int(p * 100)}%"))
            elif src.get("chat") == "youtube":
                msgs = sources.youtube_chat_replay(src["url"], dur, stop=self.stop,
                                                   progress=lambda p: report(0.02 + 0.25 * p, f"chat replay {int(p * 100)}%"))
            if src["platform"] == "youtube":
                comments = sources.youtube_comment_times(src["url"], dur)
        except Exception as e:
            log(st["name"], f"chat replay unavailable ({e})")
        if self.stop.is_set():
            raise InterruptedError
        local = None
        audio_dir = self.folder / "audio"
        if src["platform"] == "youtube":
            report(0.3, "downloading audio…")
            local = vodscan._download_audio(src, audio_dir, lambda p: report(0.3 + 0.1 * p, "downloading audio…"),
                                            self.stop, self.procs)
            st["audio_file"] = str(local)
        report(0.4, "listening for laughs, screams and reactions…")
        audio = vodscan.analyze_audio(src, lambda p: report(0.4 + 0.4 * p, f"listening… {int(p * 100)}%"),
                                      self.stop, procs=self.procs, local_file=local)
        moments = vodscan.find_moments(dur, msgs, covered, audio, comments, src.get("heatmap"), threshold=2.2)
        st["moments_found"] = len(moments)
        # "first impressions": the strongest moment early in the stream (after intro screens)
        early_end = min(max(dur * 0.3, 600), 2400)
        early = [m for m in moments if 120 <= m["start"] <= early_end]
        first = max(early, key=lambda m: m["score"]) if early else None
        if first is None:
            t = min(max(300.0, dur * 0.08), max(0.0, dur - 120))
            first = {"start": t, "end": t + 15, "peak": t + 8, "score": 20, "peak_ratio": 0, "reaction": 0,
                     "audio_boost": 0, "reasons": [], "messages": 0, "chatters": 0}
        first = dict(first, first=True)
        rest = [m for m in vodscan.pick_top(moments, 18) if abs(m["start"] - first["start"]) > 60]
        late = [m for m in moments if m["start"] >= dur * 0.88 and all(abs(m["start"] - r["start"]) > 60 for r in rest)]
        cands = [first] + rest + ([dict(max(late, key=lambda m: m["score"]), ending=True)] if late else [])
        # what did they say? (for chapter titles)
        report(0.82, "reading what they said, for chapter titles…")
        for k, m in enumerate(cands):
            if self.stop.is_set():
                raise InterruptedError
            try:
                m["words_text"] = self._listen(src, local, max(0, m["start"] - 20), m["end"] + 6)
            except Exception:
                m["words_text"] = []
            report(0.82 + 0.13 * (k + 1) / len(cands), "reading what they said…")
        report(0.96, "learning which face is the streamer…")
        try:
            face = vodscan.learn_streamer(src, self.out_dir, self.stop, self.procs, samples=24)
            st["face"] = [round(float(v), 5) for v in face] if face is not None else None
        except Exception:
            st["face"] = None
        st["candidates"] = cands
        st.pop("src", None)
        st["src_meta"] = {k: src.get(k) for k in ("platform", "url", "video_id", "chat", "category")}
        log(st["name"], f"long video: {len(moments)} moments found, {len(cands)} candidates")

    def _listen(self, src, local, a, b) -> list[str]:
        import shorts
        with tempfile.TemporaryDirectory(prefix="lf_") as tmp:
            wav = Path(tmp) / "w.wav"
            if local:
                url, headers = str(local), {}
            else:
                if getattr(self, "_aref", (None,))[0] != src["url"]:  # look it up once per stream
                    self._aref = (src["url"], vodscan._audio_stream(src))
                url, headers = self._aref[1]
            cmd = [FFMPEG, "-v", "error", "-y", "-ss", f"{a:.1f}", "-t", f"{b - a:.1f}"]
            if headers.get("User-Agent"):
                cmd += ["-user_agent", headers["User-Agent"]]
            cmd += ["-i", url, "-vn", "-ac", "1", "-ar", "16000", str(wav)]
            subprocess.run(cmd, capture_output=True, timeout=120)
            if not wav.exists():
                return []
            return [w.text for w in shorts.transcribe(wav, Path(tmp))]

    # -- step 2: render ----------------------------------------------------
    def options(self) -> dict:
        o = {"narration": True, "captions": True, "jumpcuts": True, "camzoom": True, "music": True, "effects": True}
        o.update(getattr(self, "opts", None) or {})
        return o

    def render(self, log) -> None:
        import facecam
        import longedit as le
        import shorts
        from smartcut import choose_window
        opts = self.options()
        self.status, self.started = "rendering", time.time()
        self._set("render", 0, "Getting ready…")
        free = shutil.disk_usage(self.out_dir).free / 1e9
        if free < 2.5:
            raise RuntimeError(f"Only {free:.1f} GB free; a long video needs about 2.5 GB while rendering")
        work = self.folder / "work"
        work.mkdir(parents=True, exist_ok=True)
        streams = {s["name"]: s for s in self.streams if not s.get("error")}
        items = [it for it in self.storyboard if it.get("on", True)]
        clips = [it for it in items if it["kind"] == "clip"]
        cold = next((it for it in items if it["kind"] == "cold"), None)
        fin = next((it for it in items if it["kind"] == "finale"), None)

        # 1. download each moment once (3 at a time)
        from concurrent.futures import ThreadPoolExecutor
        done = {"n": 0}
        lock = threading.Lock()
        wanted = clips + ([cold] if cold else []) + (fin["parts"] if fin else [])
        jobs: dict[str, dict] = {}
        for it in wanted:
            st = streams.get(it["stream"])
            if st is None:
                continue
            a = max(0.0, it["start"] - 45)
            b = min(st["duration"] or 1e9, it["end"] + 20)
            key = f"{it['stream']}_{int(a)}_{int(b)}"
            it["_wide"], it["_a"] = str(work / f"w_{key}.mp4"), a
            jobs.setdefault(key, {"path": it["_wide"], "a": a, "b": b, "meta": st["src_meta"]})

        def fetch(job):
            if self.stop.is_set():
                return
            meta = job["meta"]
            fmt = ("best[height<=1080]/best" if meta["platform"] in ("twitch", "kick")
                   else "bv*[height<=1080][vcodec^=avc1]+ba[ext=m4a]/b[height<=1080]/b")
            for attempt in range(2):  # YouTube downloads sometimes fail once
                if Path(job["path"]).exists() or self.stop.is_set():
                    break
                p = subprocess.Popen([sources.YTDLP, "-f", fmt, "--download-sections",
                                      f"*{job['a']:.1f}-{job['b']:.1f}", "--merge-output-format", "mp4",
                                      "--no-warnings", "--quiet", "--no-part", "-o", job["path"], meta["url"]],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.procs.append(p)
                p.wait()
            with lock:
                done["n"] += 1
                self._set("render", 0.35 * done["n"] / max(1, len(jobs)),
                          f"Downloading moments in full quality… {done['n']}/{len(jobs)}")

        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(fetch, jobs.values()))
        if self.stop.is_set():
            raise InterruptedError

        voice = le.pick_voice() if opts["narration"] else None
        sfx = shorts.ensure_sfx() if opts["effects"] else None
        whoosh = sfx[1] if sfx else None
        boom = sfx[0] if sfx else None
        parts: list[tuple[Path, str | None]] = []  # (file, chapter title or None)
        k = itertools.count()
        out = lambda name: work / f"p{next(k):03d}_{name}.mp4"
        names = list(dict.fromkeys(s["display"] for s in streams.values()))
        faces = {n: (np.asarray(s["face"], np.float32) if s.get("face") else None) for n, s in streams.items()}

        # 2. cold open (teaser), then the title card with the narrator's intro
        if cold and Path(cold.get("_wide", "")).exists():
            s = max(0.0, cold["peak"] - cold["_a"] - 5)
            parts.append((self._clip_part(Path(cold["_wide"]), s, COLD, out("cold"),
                                          credit=streams[cold["stream"]]["credit"], flash=True), "Intro"))
        t_item = next((it for it in items if it["kind"] == "title"), None)
        if t_item:
            img = card(work / "title.jpg", t_item["text"], f"ft. {' · '.join(names)}" if names else "")
            vo = work / "vo_title.wav"
            d = le.speak(t_item.get("line") or le.intro_line(self.topic, names), vo, voice)
            parts.append((self._still_part(img, max(TITLE, d + 0.9), out("title"), voice=vo if d else None,
                                           whoosh=whoosh), None if parts else "Intro"))

        # 3. chapters: card (+ narration) then the edited moment
        total = max(1, len(clips))
        thumb_cands = []
        for i, it in enumerate(clips):
            if self.stop.is_set():
                raise InterruptedError
            self._set("render", 0.35 + 0.5 * i / total, f"Editing chapter {i + 1} of {total}…")
            wide = Path(it.get("_wide", ""))
            if not wide.exists():
                log(it["stream"], f"skipping '{it['title']}' (download failed)")
                continue
            st = streams[it["stream"]]
            with tempfile.TemporaryDirectory(prefix="lfw_") as tmp:
                try:
                    words = shorts.transcribe(wide, Path(tmp))
                except Exception:
                    words = []
            _, _, wdur = shorts.probe(wide)
            rel = lambda t: t - it["_a"]
            cat = it.get("category", "moment")
            s, e = choose_window(words, rel(it["start"]), rel(it["end"]), rel(it["peak"]), wdur,
                                 pre=22 if cat in STORY else 16, max_len=MAXLEN.get(cat, 45))
            if e - s < 3:
                continue
            peak = rel(it["peak"])
            keep = le.jump_cuts(words, le.loudness(wide), s, e, peak) if opts["jumpcuts"] else [(s, e)]
            cam = None
            if opts["camzoom"]:
                try:
                    cam = le.cam_crop(facecam.detect(wide, identity=faces.get(it["stream"])))
                except Exception:
                    cam = None
            segs = le.with_cam_zoom(keep, peak) if cam else [(a, b, "full") for a, b in keep]
            if it.get("chapter", True):
                frame = work / f"cf_{i}.jpg"
                subprocess.run([FFMPEG, "-v", "error", "-y", "-ss", f"{s + 0.5:.2f}", "-i", str(wide),
                                "-frames:v", "1", str(frame)], capture_output=True)
                cimg = le.chapter_card(frame, work / f"cc_{i}.jpg", it["title"], it.get("kicker", ""), _font, _outlined)
                vo = work / f"vo_{i}.wav"
                d = le.speak(it.get("line", ""), vo, voice) if opts["narration"] else 0.0
                parts.append((self._still_part(cimg, max(1.7, d + 0.6), out("card"), voice=vo if d else None,
                                               whoosh=whoosh), it["title"]))
            # where the reaction lands on the cut-down timeline (for the boom)
            boom_at, acc = None, 0.0
            for a, b, mode in segs:
                if a <= peak < b:
                    boom_at = acc + (peak - a)
                    break
                acc += b - a
            caps = None
            if opts["captions"] and words:
                cdir = work / f"cap_{i}"
                cdir.mkdir(exist_ok=True)
                caps = le.caption_track(le.map_words(words, segs), sum(b - a for a, b, _ in segs), cdir, _font)
            parts.append((self._edit_part(wide, segs, cam, out("clip"), st["credit"], caps,
                                          boom if boom_at is not None else None, boom_at), None))
            thumb_cands.append((it.get("score", 0), wide, peak, faces.get(it["stream"])))

        # 4. finale
        if fin:
            self._set("render", 0.87, "Making the finale…")
            tiles = [(Path(p["_wide"]), max(0.0, p["peak"] - p["_a"] - 6), streams[p["stream"]]["display"])
                     for p in fin["parts"] if Path(p.get("_wide", "")).exists()][:4]
            if tiles:
                parts.append((self._grid_part(tiles, fin["title"], out("finale")), fin["title"]))
        # 5. end screen (20 s, laid out for YouTube's end screen elements) with the narrator's outro
        o_item = next((it for it in items if it["kind"] == "outro"), None)
        if o_item:
            img = le.end_screen(work / "outro.jpg", sunset_background(), self.channel_name, _font, _outlined)
            vo = work / "vo_outro.wav"
            d = le.speak(o_item.get("line") or le.OUTRO_LINE, vo, voice)
            parts.append((self._still_part(img, OUTRO, out("outro"), voice=vo if d else None, whoosh=whoosh),
                          "Thanks for watching"))

        # 6. join everything, then the music bed
        self._set("render", 0.92, "Putting the video together…")
        final = self.folder / f"{_slug(self.topic)}.mp4"
        joined = work / "joined.mp4"
        lst = work / "parts.txt"
        lst.write_text("".join(f"file '{p}'\n" for p, _ in parts))
        r = subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                            "-c", "copy", "-movflags", "+faststart", str(joined)], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"couldn't join the video: {r.stderr.strip()[-200:]}")
        chapters, t = [], 0.0
        for p, title in parts:
            if title:
                chapters.append((t, title))
            t += _duration(p)
        music = False
        if opts["music"] and le.music_files():
            self._set("render", 0.95, "Adding background music…")
            music = le.add_music(joined, final, t)
        if not music:
            shutil.move(str(joined), str(final))
        chapters = _youtube_chapters(chapters, t)

        # 7. thumbnail, titles, description
        self._set("render", 0.97, "Making the thumbnail…")
        thumb = None
        try:
            best = None
            for score, wide, peak, face in sorted(thumb_cands, key=lambda c: -c[0])[:5]:
                got = le.best_reaction_frame([(wide, peak)], face, work)
                if got and (best is None or got[2] > best[2]):
                    best = got
            if best:
                words = self.topic.split()
                text = (self.topic if len(words) <= 4 else
                        (f"{names[0]} LOST IT" if len(names) == 1 else "THEY LOST IT"))
                thumb = le.thumbnail(best[0], best[1], text, self.folder / "thumbnail.jpg", _font, _outlined)
        except Exception as e:
            log("long", f"face thumbnail failed ({e}); using a plain one")
        if thumb is None:
            thumb = self._thumbnail(clips, cold, streams, work)
        titles = suggest_titles(self.topic, names)
        from metadata import for_long
        meta = for_long(self.topic, names, [s.get("src_meta", {}).get("category") for s in streams.values()],
                        "\n".join(f"{_ts(t0)} {title}" for t0, title in chapters),
                        [f"{s['display']}: {s['credit']}" for s in streams.values()], titles)
        meta["description"] += f"\n\nSubscribe to {self.channel_name} for more!"
        desc = meta["description"]
        (self.folder / "description.txt").write_text(desc)
        self.outputs = {"video": _rel(final, self.out_dir), "thumbnail": _rel(thumb, self.out_dir) if thumb else None,
                        "description": desc, "length": int(t), "titles": titles, "meta": meta,
                        "narrator": voice, "music": music}
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(self.folder / "audio", ignore_errors=True)
        self.status, self.next_action = "done", "render"
        self._set("done", 1.0, f"Done: {_fmt(t)} video" + (f" · narrated by {voice}" if voice else ""))
        self.save()
        log("long", f"LONG VIDEO SAVED -> {final} ({_fmt(t)})")

    def _edit_part(self, src: Path, segs, cam, dest: Path, credit: str, caps: Path | None,
                   boom: Path | None, boom_at: float | None) -> Path:
        """One chapter's moment: jump cuts, a cut to the streamer's face on the reaction,
        name tag, captions, a boom on the big moment, balanced loudness."""
        tag = name_tag(dest.with_suffix(".tag.png"), credit)
        cmd = [FFMPEG, "-v", "error", "-y", "-i", str(src), "-i", str(tag)]
        n_in = 2
        f = []
        for i, (a, b, mode) in enumerate(segs):
            look = (f"crop={cam[2]}:{cam[3]}:{cam[0]}:{cam[1]},scale={W}:{H}" if mode == "cam" and cam else
                    f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2")
            f.append(f"[0:v]trim=start={a:.3f}:end={b:.3f},setpts=PTS-STARTPTS,{look},fps={FPS},setsar=1,format=yuv420p[v{i}]")
            f.append(f"[0:a]atrim=start={a:.3f}:end={b:.3f},asetpts=PTS-STARTPTS,aresample=48000,"
                     f"aformat=channel_layouts=stereo[a{i}]")
        n = len(segs)
        f.append("".join(f"[v{i}][a{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=1[vc][ac]")
        dur = sum(b - a for a, b, _ in segs)
        f.append("[vc][1:v]overlay=0:0[t]")
        last = "t"
        if caps is not None:
            cmd += ["-f", "concat", "-safe", "0", "-i", str(caps)]
            f.append(f"[{n_in}:v]format=rgba[cp];[t][cp]overlay=0:0:eof_action=pass[t2]")
            last = "t2"
            n_in += 1
        f.append(f"[{last}]fade=t=in:st=0:d=0.2,fade=t=out:st={max(0, dur - 0.3):.2f}:d=0.3[v]")
        f.append(f"[ac]loudnorm=I=-15:TP=-1.5:LRA=11,aresample=48000,afade=t=in:st=0:d=0.15,"
                 f"afade=t=out:st={max(0, dur - 0.3):.2f}:d=0.3[am]")
        if boom is not None and boom_at is not None:
            cmd += ["-i", str(boom)]
            ms = int(max(0, boom_at - 0.08) * 1000)
            f.append(f"[{n_in}:a]adelay={ms}|{ms},volume=0.6[bm];[am][bm]amix=inputs=2:normalize=0:duration=first,"
                     f"alimiter=limit=0.95[a]")
        else:
            f.append("[am]anull[a]")
        return self._run(cmd + ["-filter_complex", ";".join(f), "-map", "[v]", "-map", "[a]", "-t", f"{dur:.3f}"], dest)

    # -- ffmpeg pieces (all parts share one format so they join without re-encoding)
    ENC = ["-c:v", platformkit.video_encoder(), "-b:v", "14M", "-maxrate", "18M", "-bufsize", "28M",
           "-pix_fmt", "yuv420p", "-r", str(FPS), "-video_track_timescale", "15360",
           "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]

    def _run(self, cmd, dest: Path) -> Path:
        r = subprocess.run(cmd + self.ENC + [str(dest)], capture_output=True, text=True)
        if r.returncode != 0:  # hardware encoder unavailable: software
            enc = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"] + self.ENC[self.ENC.index("-pix_fmt"):]
            r = subprocess.run(cmd + enc + [str(dest)], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip()[-300:])
        return dest

    def _clip_part(self, src: Path, start: float, dur: float, dest: Path, credit: str, lower: Path | None = None,
                   flash: bool = False) -> Path:
        tag = name_tag(dest.with_suffix(".tag.png"), credit)
        inputs = [FFMPEG, "-v", "error", "-y", "-ss", f"{start:.2f}", "-t", f"{dur:.2f}", "-i", str(src),
                  "-i", str(tag)]
        v = (f"[0:v]scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,"
             f"fps={FPS},format=yuv420p[b];[b][1:v]overlay=0:0[t]")
        last = "t"
        if lower is not None:
            inputs += ["-i", str(lower)]
            v += f";[t][2:v]overlay=0:0:enable='between(t,0.4,4.4)'[l]"
            last = "l"
        fade_out = f"fade=t=out:st={max(0, dur - 0.35):.2f}:d=0.35" + (":color=white" if flash else "")
        v += f";[{last}]fade=t=in:st=0:d=0.25,{fade_out}[v]"
        a = (f"[0:a]aresample=48000,aformat=channel_layouts=stereo,loudnorm=I=-15:TP=-1.5:LRA=11,aresample=48000,"
             f"afade=t=in:st=0:d=0.2,afade=t=out:st={max(0, dur - 0.35):.2f}:d=0.35[a]")
        return self._run(inputs + ["-filter_complex", v + ";" + a, "-map", "[v]", "-map", "[a]", "-t", f"{dur:.2f}"], dest)

    def _still_part(self, img: Path, dur: float, dest: Path, voice: Path | None = None,
                    whoosh: Path | None = None) -> Path:
        """A card (title, chapter, end screen) with an optional narrator line and whoosh.
        A slow push-in keeps it from looking frozen."""
        cmd = [FFMPEG, "-v", "error", "-y", "-loop", "1", "-t", f"{dur}", "-i", str(img),
               "-f", "lavfi", "-t", f"{dur}", "-i", "anullsrc=r=48000:cl=stereo"]
        zoom = f"(1+0.04*t/{max(dur, 0.1):.2f})"
        f = [f"[0:v]scale={W}*2:{H}*2,fps={FPS},scale=w='trunc({W}*{zoom}/2)*2':h='trunc({H}*{zoom}/2)*2':eval=frame,"
             f"crop={W}:{H},format=yuv420p,fade=t=in:st=0:d=0.25,fade=t=out:st={max(0, dur - 0.3):.2f}:d=0.3[v]"]
        mix, n = ["[1:a]"], 2
        if voice is not None:
            cmd += ["-i", str(voice)]
            f.append(f"[{n}:a]adelay=250|250[vo]")
            mix.append("[vo]")
            n += 1
        if whoosh is not None:
            cmd += ["-i", str(whoosh)]
            f.append(f"[{n}:a]volume=0.45[wh]")
            mix.append("[wh]")
            n += 1
        f.append("".join(mix) + f"amix=inputs={len(mix)}:normalize=0:duration=first,aresample=48000[a]")
        return self._run(cmd + ["-filter_complex", ";".join(f), "-map", "[v]", "-map", "[a]", "-t", f"{dur}"], dest)

    def _grid_part(self, tiles, title: str, dest: Path) -> Path:
        dur = FINALE
        overlay = finale_overlay(dest.with_suffix(".png"), [n for _, _, n in tiles], title)
        cmd = [FFMPEG, "-v", "error", "-y"]
        for path, start, _ in tiles:
            cmd += ["-ss", f"{start:.2f}", "-t", f"{dur}", "-i", str(path)]
        n = len(tiles)
        cmd += ["-i", str(overlay)]
        f = [f"color=c=0x0b0618:s={W}x{H}:r={FPS}:d={dur}[bg]"]
        last = "bg"
        for i, (x, y) in enumerate(grid_spots(n)):
            f.append(f"[{i}:v]scale=960:540:force_original_aspect_ratio=increase,crop=960:540,fps={FPS},setsar=1[v{i}]")
            f.append(f"[{last}][v{i}]overlay={x}:{y}:shortest=1[g{i}]")
            last = f"g{i}"
        f.append(f"[{last}][{n}:v]overlay=0:0:enable='between(t,0,2.2)'[o]")
        f.append(f"[o]format=yuv420p,fade=t=in:st=0:d=0.3,fade=t=out:st={dur - 0.4}:d=0.4[v]")
        mix = "".join(f"[{i}:a]" for i in range(n))
        f.append(f"{mix}amix=inputs={n}:normalize=0,volume={1 / max(1, n) ** 0.5:.2f},"
                 f"loudnorm=I=-15:TP=-1.5:LRA=11,aresample=48000,afade=t=out:st={dur - 0.4}:d=0.4[a]")
        return self._run(cmd + ["-filter_complex", ";".join(f), "-map", "[v]", "-map", "[a]", "-t", f"{dur}"], dest)

    def _thumbnail(self, clips, cold, streams, work) -> Path | None:
        best = cold or (max(clips, key=lambda c: c.get("score", 0)) if clips else None)
        if not best or not Path(best.get("_wide", "")).exists():
            return None
        frame = work / "thumbframe.jpg"
        t = max(0.0, best["peak"] - best["_a"])
        subprocess.run([FFMPEG, "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", best["_wide"], "-frames:v", "1",
                        "-vf", "scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720", str(frame)],
                       capture_output=True)
        if not frame.exists():
            return None
        img = Image.open(frame).convert("RGBA")
        shade = Image.new("RGBA", img.size, (0, 0, 0, 0))
        sd = ImageDraw.Draw(shade)
        for y in range(720):
            a = int(max(0, (y - 300) / 420) * 200)
            sd.line([(0, y), (1280, y)], fill=(11, 6, 24, a))
        img = Image.alpha_composite(img, shade)
        d = ImageDraw.Draw(img)
        words = self.topic.upper().split()
        line1 = " ".join(words[: max(1, len(words) // 2)])
        line2 = " ".join(words[max(1, len(words) // 2):])
        _outlined(d, (640, 520 if line2 else 600), line1, _fit(d, line1, 1180, 150), stroke=10, shadow=8)
        if line2:
            _outlined(d, (640, 640), line2, _fit(d, line2, 1180, 120), fill=(255, 210, 63), stroke=9, shadow=7)
        out = self.folder / "thumbnail.jpg"
        img.convert("RGB").save(out, quality=92)
        return out

    def _description(self, chapters, streams) -> str:
        lines = [f"{self.topic}", ""]
        lines += [f"{_ts(t)} {title}" for t, title in chapters]
        lines += ["", "Streamers in this video (go follow them!):"]
        for s in streams.values():
            lines.append(f"- {s['display']}: {s['credit']}")
        lines += ["", f"Clips are credited to their creators. Subscribe to {self.channel_name} for more!"]
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Storyboard
# --------------------------------------------------------------------------
def build_storyboard(streams: list[dict], topic: str, minutes: int, channel: str) -> list[dict]:
    """A story, not a list: first impressions -> build-up (mixed kinds of moments, in stream
    order) -> the biggest moment as the climax -> how it ended -> everyone's reaction."""
    import longedit as le
    good = [s for s in streams if not s.get("error") and s.get("candidates")]
    display = {s["name"]: s["display"] for s in good}
    dur = {s["name"]: max(1.0, s.get("duration") or 1.0) for s in good}
    budget = minutes * 60 - (COLD + TITLE + FINALE + OUTRO)
    used_titles: set = set()

    def kind(m):  # what kind of moment (cops, funny, hype...), without using up a title
        return title_for(m, display[m["stream"]], set())[0]

    def est(m, cat=None):  # rough length on screen: the moment + chapter card
        cat = cat or kind(m)
        return float(min(MAXLEN.get(cat, 45), max(22, (m["end"] - m["start"]) + 24)) * 0.85 + 2.5)

    firsts, pool, endings = [], [], []
    for s in good:
        c = [dict(m, stream=s["name"]) for m in s["candidates"]]
        firsts.append(c[0])
        for m in c[1:]:
            (endings if m.get("ending") else pool).append(m)
    for m in pool + firsts + endings:
        m["_cat"] = kind(m)
    left = budget - sum(est(m, m["_cat"]) for m in firsts)
    # climax: the single biggest moment overall
    climax = max(pool, key=lambda m: m["score"]) if pool else None
    if climax:
        pool.remove(climax)
        left -= est(climax, climax["_cat"])
    ending = max(endings, key=lambda m: m["score"]) if endings else None
    if ending and left > 60:
        left -= est(ending, ending["_cat"])
    else:
        ending = None
    # build-up: good moments, but don't repeat the same kind over and over, share time fairly
    chosen, per_stream, per_cat = [], {}, {}
    while left > 20 and pool:
        def value(m):
            # same kind again costs a lot; a streamer who already has more clips than the others costs too
            fair = min(per_stream.get(n, 0) for n in display) if len(display) > 1 else 0
            return (m["score"] - 16 * per_cat.get(m["_cat"], 0)
                    - 15 * max(0, per_stream.get(m["stream"], 0) - fair) - 4 * per_stream.get(m["stream"], 0))
        m = max(pool, key=value)
        pool.remove(m)
        if est(m, m["_cat"]) > left + 10:
            continue
        chosen.append(m)
        per_cat[m["_cat"]] = per_cat.get(m["_cat"], 0) + 1
        per_stream[m["stream"]] = per_stream.get(m["stream"], 0) + 1
        left -= est(m, m["_cat"])
    chosen.sort(key=lambda m: m["start"] / dur[m["stream"]])  # story order, mixing streamers
    seed = iter(range(10_000))
    said_lines: set = set()

    def clip_item(m, on=True, kicker="", climax=False, ending=False):
        cat, title = title_for(m, display[m["stream"]], used_titles)
        if climax:
            title = f"The moment that broke {display[m['stream']]}'s stream"
        if ending:
            title = f"How {display[m['stream']]}'s stream ended"
        return {"kind": "clip", "stream": m["stream"], "start": m["start"], "end": m["end"], "peak": m["peak"],
                "score": m["score"], "title": title, "category": cat, "chapter": True, "on": on,
                "kicker": kicker, "est": round(est(m, cat), 1), "at": _fmt(m["start"]),
                "said": " ".join(m.get("words_text") or [])[:160], "climax": climax, "ending": ending,
                "line": le.narration_line(cat, display[m["stream"]], m["start"], next(seed), climax, ending,
                                          said_lines if on else None)}

    board = []
    everything = firsts + chosen + ([climax] if climax else [])
    teaser = climax or (max(everything, key=lambda m: m["score"]) if everything else None)
    if teaser:
        board.append({"kind": "cold", "stream": teaser["stream"], "start": teaser["start"], "end": teaser["end"],
                      "peak": teaser["peak"], "on": True, "title": "Cold open (teaser of the biggest moment)"})
    board.append({"kind": "title", "text": topic.upper(), "on": True,
                  "line": le.intro_line(topic, list(display.values()))})
    for i, m in enumerate(firsts):
        board.append(clip_item(m, kicker="First impressions" if i == 0 else ""))
    for m in chosen:
        board.append(clip_item(m))
    if climax:
        board.append(clip_item(climax, kicker="The climax", climax=True))
    if ending:
        board.append(clip_item(ending, kicker="The ending", ending=True))
    # finale: each streamer's biggest reaction (or the top 4 of one streamer)
    reacts = []
    for s in good:
        top = max(s["candidates"], key=lambda m: (m.get("reaction", 0) * 2 + m["score"] / 100))
        reacts.append(dict(top, stream=s["name"]))
    if len(reacts) == 1:
        s = good[0]
        reacts = [dict(m, stream=s["name"]) for m in sorted(s["candidates"], key=lambda m: -m["score"])[:4]]
    if reacts:
        title = "Everyone's reaction" if len(good) > 1 else f"{good[0]['display']}'s best reactions"
        board.append({"kind": "finale", "title": title, "on": True,
                      "parts": [{"stream": m["stream"], "start": m["start"], "end": m["end"], "peak": m["peak"]}
                                for m in reacts[:4]]})
    board.append({"kind": "outro", "text": "Thanks for watching", "on": True, "line": le.OUTRO_LINE})
    for m in sorted(pool, key=lambda m: -m["score"])[:10]:  # extra moments you can switch on
        board.append(clip_item(m, on=False))
    for it in board:
        it.pop("_cat", None)
    return board


def suggest_titles(topic: str, names: list[str]) -> list[str]:
    names = list(dict.fromkeys(names))
    who = names[0] if len(names) == 1 else (f"{', '.join(names[:-1])} & {names[-1]}" if len(names) <= 3
                                            else "Streamers")
    t = topic.strip().rstrip(".!")
    return [f"{who} – {t} (Funniest Moments)", f"{t.upper()}: Best Reactions", f"{who} React to {t}",
            f"{t} Was CHAOS 😳", f"The Craziest Moments From {t}"]


# --------------------------------------------------------------------------
def _duration(p: Path) -> float:
    r = subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def _youtube_chapters(ch: list[tuple[float, str]], total: float) -> list[tuple[float, str]]:
    """YouTube needs: first at 0:00, at least 3, each at least 10 s long."""
    out = []
    for t, title in ch:
        if out and t - out[-1][0] < 10:
            continue
        out.append((t, title))
    if out:
        out[0] = (0.0, out[0][1])
    return out if len(out) >= 3 else []


def _ts(t: float) -> str:
    t = int(t)
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:50] or "video"


def _rel(p: Path, base: Path) -> str:
    return p.resolve().relative_to(base.resolve()).as_posix()


# --------------------------------------------------------------------------
class LongManager:
    """Runs long-video jobs one at a time and keeps them across restarts."""

    def __init__(self, args, out_dir: Path, log):
        self.args, self.out_dir, self.log = args, Path(out_dir), log
        self.lock = threading.Lock()
        self.projects: list[LongProject] = []
        self.queue: list[tuple[LongProject, str]] = []
        self.wake = threading.Event()
        for f in sorted((self.out_dir / "_long").glob("*/project.json")):
            try:
                self.projects.append(LongProject.load(f.parent, self.out_dir, args))
            except Exception:
                pass
        threading.Thread(target=self._loop, daemon=True, name="long").start()

    def create(self, links: list[str], topic: str, minutes: int) -> LongProject:
        links = [l.strip() for l in links if l.strip()][:6]
        if not links:
            raise ValueError("Paste at least one link")
        p = LongProject(self.out_dir, links, topic, max(5, min(int(minutes), 20)), self.args)
        while any(q.id == p.id for q in self.projects):
            p.id += "b"
            p.folder = self.out_dir / "_long" / p.id
        p.save()
        with self.lock:
            self.projects.insert(0, p)
            self.queue.append((p, "scan"))
        self.wake.set()
        return p

    def get(self, pid: str) -> LongProject | None:
        return next((p for p in self.projects if p.id == pid), None)

    def update(self, pid: str, storyboard: list, channel_name: str | None = None, options: dict | None = None) -> bool:
        p = self.get(pid)
        if not p or p.status not in ("review", "done"):
            return False
        allowed = {"title", "on", "chapter", "text", "kicker", "line"}
        if isinstance(options, dict):
            p.opts = {k: bool(v) for k, v in options.items()
                      if k in ("narration", "captions", "jumpcuts", "camzoom", "music", "effects")}
        by_kind = []
        for new in storyboard:  # only accept edits of fields the page may change, keep order
            i = new.get("i")
            if not isinstance(i, int) or not 0 <= i < len(p.storyboard):
                continue
            old = dict(p.storyboard[i])
            for k in allowed & set(new):
                old[k] = (str(new[k])[:220] if k == "line" else str(new[k])[:120]) if k in ("title", "text", "kicker", "line") else bool(new[k])
            by_kind.append(old)
        if by_kind:
            p.storyboard = by_kind
        if channel_name:
            p.channel_name = channel_name.strip()[:40]
        p.save()
        return True

    def render(self, pid: str) -> bool:
        p = self.get(pid)
        if not p or p.status not in ("review", "done", "failed") or not p.storyboard:
            return False
        p.status, p.message, p.progress, p.stop = "queued", "Waiting to render…", 0.0, threading.Event()
        with self.lock:
            self.queue.append((p, "render"))
        self.wake.set()
        return True

    def cancel(self, pid: str) -> bool:
        p = self.get(pid)
        if not p:
            return False
        p.stop.set()
        for pr in p.procs:
            if pr.poll() is None:
                pr.terminate()
        return True

    def delete(self, pid: str) -> bool:
        p = self.get(pid)
        if not p or p.status in ("scanning", "rendering"):
            return False
        with self.lock:
            self.projects.remove(p)
            self.queue = [q for q in self.queue if q[0] is not p]
        if not platformkit.to_trash(p.folder):
            shutil.rmtree(p.folder, ignore_errors=True)
        return True

    def state(self) -> list[dict]:
        with self.lock:
            return [p.state() for p in self.projects]

    def _loop(self) -> None:
        while True:
            self.wake.wait(2)
            self.wake.clear()
            with self.lock:
                job = self.queue.pop(0) if self.queue else None
            if not job:
                continue
            p, what = job
            try:
                (p.scan if what == "scan" else p.render)(self.log)
            except InterruptedError:
                p.status, p.message = ("review" if p.storyboard else "cancelled"), "Cancelled"
            except Exception as e:
                p.status, p.message = ("review" if (what == "render" and p.storyboard) else "failed"), f"Failed: {e}"
                self.log("long", f"long video {what} failed: {e}")
            finally:
                for pr in p.procs:
                    if pr.poll() is None:
                        pr.terminate()
                p.procs = []
                p.save()
