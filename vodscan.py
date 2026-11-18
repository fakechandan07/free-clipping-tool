"""Scan mode: find the best moments in a past video / VOD, then clip only those.

1. Chat replay (Twitch past broadcast or YouTube past stream), if there is one
2. Audio, streamed straight through without saving: how loud the streamer is, and
   laughing / screaming / cheering (YAMNet)
3. YouTube: timestamps people wrote in the comments ("2:34 LMAOOO")
4. Every moment gets the same 0-100 score as live clips; the top N are picked
5. Only those parts are downloaded in full quality, then smart-trimmed and turned
   into shorts by the same finisher as live clips
"""

from __future__ import annotations

import itertools
import json
import math
import shutil
import statistics
import subprocess
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

import sources
from hype import CLIP_RE, STOPWORDS, TOKEN_RE, is_noise, message_weight, score_moment

import platformkit  # noqa: E402
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
BUCKET = 5.0  # seconds per chat bucket
IRL_CATEGORIES = {"irl", "just chatting", "people & blogs", "travel & events", "travel & outdoors"}


def _fmt(t: float) -> str:
    t = int(t)
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"


# --------------------------------------------------------------------------
# Audio: loudness + laughing/screaming over the whole video, streamed
# --------------------------------------------------------------------------
def _audio_stream(src: dict) -> tuple[str, dict]:
    """Direct address (and headers) of the smallest audio version of the video."""
    fmt = {"twitch": "Audio_Only/worst", "kick": "worst"}.get(src["platform"],
                                                           "worstaudio[abr>=40]/worstaudio/bestaudio")
    r = subprocess.run([sources.YTDLP, "-f", fmt, "-j", "--no-warnings", src["url"]],
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(sources._friendly((r.stderr.strip().splitlines() or ["couldn't get the audio"])[-1]))
    d = json.loads(r.stdout)
    return d["url"], d.get("http_headers") or {}


def _download_audio(src: dict, folder: Path, progress, stop: threading.Event, procs: list | None) -> Path:
    """YouTube slows down streaming reads of its audio; yt-dlp downloads the small audio
    file (~30 MB per hour) much faster. Returns the local file."""
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"scan_audio_{src.get('video_id', 'x')}.m4a"
    p = subprocess.Popen([sources.YTDLP, "-f", "worstaudio[abr>=40]/worstaudio/bestaudio", "-N", "8",
                          "--no-warnings", "--quiet", "--newline", "--progress", "--progress-template",
                          "%(progress._percent_str)s", "-o", str(out), src["url"]],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if procs is not None:
        procs.append(p)
    for line in p.stdout:
        if stop.is_set():
            p.terminate()
            break
        try:
            progress(float(line.strip().rstrip("%")) / 100)
        except ValueError:
            pass
    p.wait()
    if stop.is_set():
        out.unlink(missing_ok=True)
        raise InterruptedError
    if p.returncode != 0 or not out.exists():
        err = (p.stderr.read() or "").strip().splitlines()
        raise RuntimeError(sources._friendly(err[-1] if err else "couldn't download the audio"))
    return out


def analyze_audio(src: dict, progress, stop: threading.Event, procs: list | None = None, workers: int = 6,
                  local_file: Path | None = None):
    """Loudness (dB per 0.25 s) and laugh / scream / cheer (per 1 s) over the whole video.

    The video is split into parts that are read at the same time by separate ffmpeg
    processes (each with its own sound classifier), so a 13-hour VOD takes minutes."""
    from reactions import SoundClassifier
    if local_file is not None:
        url, headers = str(local_file), {}
    else:
        url, headers = _audio_stream(src)
    dur = float(src.get("duration") or 0)
    n_sec = int(dur) + 2
    loud = np.full(n_sec * 4, np.nan, dtype=np.float32)
    laugh = np.zeros(n_sec, dtype=np.float32)
    scream = np.zeros(n_sec, dtype=np.float32)
    cheer = np.zeros(n_sec, dtype=np.float32)
    parts = max(1, min(workers, int(dur // 600) + 1))
    edges = [dur * i / parts for i in range(parts + 1)]
    done = [0.0] * parts
    lock = threading.Lock()
    header_arg = "".join(f"{k}: {v}\r\n" for k, v in headers.items() if k.lower() != "user-agent")
    errors = []

    def work(i: int):
        a, b = edges[i], edges[i + 1]
        sc = SoundClassifier()
        cmd = [FFMPEG, "-hide_banner", "-loglevel", "error"]
        if headers.get("User-Agent"):
            cmd += ["-user_agent", headers["User-Agent"]]
        if header_arg:
            cmd += ["-headers", header_arg]
        cmd += ["-ss", f"{a:.2f}", "-t", f"{b - a + 1:.2f}", "-i", url,
                "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "pipe:1"]
        ff = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if procs is not None:
            procs.append(ff)
        window = np.zeros(0, dtype=np.float32)
        q0, q = int(a * 4), 0
        while not stop.is_set():
            chunk = ff.stdout.read(8000)  # 0.25 s
            if not chunk:
                break
            x = np.frombuffer(chunk[: len(chunk) // 2 * 2], dtype=np.int16).astype(np.float32) / 32768
            if not len(x):
                continue
            idx = q0 + q
            if idx < len(loud):
                loud[idx] = 20 * math.log10(float(np.sqrt(np.mean(x * x))) + 1e-6)
            window = np.concatenate([window, x])[-15600:]
            q += 1
            if q % 4 == 0:
                sec = idx // 4
                if len(window) >= 15600 and sec < n_sec:
                    r = sc.classify(window)
                    laugh[sec], scream[sec], cheer[sec] = r["laugh"], max(r["scream"], 0.8 * r["gasp"]), r["cheer"]
                if q % 120 == 0:
                    with lock:
                        done[i] = min(b - a, q / 4)
                        progress(min(1.0, sum(done) / max(dur, 1)))
        if ff.poll() is None:
            ff.terminate()
        err = (ff.stderr.read() or b"").decode(errors="replace").strip()
        if q == 0 and err and not stop.is_set():
            errors.append(err.splitlines()[-1][:200])

    threads = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(parts)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if stop.is_set():
        raise InterruptedError
    if errors and len(errors) == parts:
        raise RuntimeError(f"couldn't read the audio ({errors[0]})")
    # fill quarters that weren't read (gaps between parts) with the median level
    if np.isnan(loud).all():
        loud[:] = -60
    else:
        loud = np.where(np.isnan(loud), np.nanmedian(loud), loud).astype(np.float32)
    return loud, laugh, scream, cheer


# --------------------------------------------------------------------------
# Who is the streamer? (so shorts crop their face, not a skin or a teammate)
# --------------------------------------------------------------------------
def learn_streamer(src: dict, out_dir: Path, stop: threading.Event, procs: list | None = None,
                   samples: int = 30, log=None):
    """Grab ~30 frames spread over the whole video and work out whose face shows up the
    most: that's the streamer. Remembered for next time. Returns the face or None."""
    import cv2
    import identity as ident
    if not ident.available():
        return None
    sid = ident.StreamerIdentity(ident.identity_file(out_dir, src["name"]))
    fmt = ("best[height<=720]/best" if src["platform"] in ("twitch", "kick")
           else "bv*[height<=720][vcodec^=avc1]/bv*[height<=720]/b[height<=720]")
    r = subprocess.run([sources.YTDLP, "-f", fmt, "-j", "--no-warnings", src["url"]],
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        return sid.streamer()
    d = json.loads(r.stdout)
    url, headers = d["url"], d.get("http_headers") or {}
    dur = float(src.get("duration") or d.get("duration") or 0)
    times = [dur * (0.03 + 0.94 * (i + 0.5) / samples) for i in range(samples)]
    header_arg = "".join(f"{k}: {v}\r\n" for k, v in headers.items() if k.lower() != "user-agent")

    def grab(t):
        if stop.is_set():
            return None
        cmd = [FFMPEG, "-hide_banner", "-loglevel", "error"]
        if headers.get("User-Agent"):
            cmd += ["-user_agent", headers["User-Agent"]]
        if header_arg:
            cmd += ["-headers", header_arg]
        cmd += ["-ss", f"{t:.1f}", "-i", url, "-frames:v", "1", "-vf", "scale=1280:-2",
                "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1"]
        try:
            out = subprocess.run(cmd, capture_output=True, timeout=60).stdout
        except subprocess.TimeoutExpired:
            return None
        return cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR) if out else None

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=6) as pool:
        imgs = list(pool.map(grab, times))
    for k, img in enumerate(imgs):
        if img is not None:
            sid.add_frame(ident.faces(img, min_size=16), frame_id=k)
    face = sid.streamer()
    if face is not None:
        sid.save()
    if log:
        groups = sorted(sid.groups, key=lambda g: -len(g["frames"]))
        seen = ", ".join(str(len(g["frames"])) for g in groups[:4])
        log(src["name"], f"scan: streamer face {'found' if face is not None else 'NOT clear'} "
                         f"({sid.frames} frames; most-seen people appear in {seen or '0'} of them)")
    return face


# --------------------------------------------------------------------------
# Finding moments
# --------------------------------------------------------------------------
def _rolling(values: np.ndarray, span: int, fn) -> np.ndarray:
    """fn over the `span` values BEFORE each index (the 'normal' level up to then)."""
    out = np.zeros_like(values, dtype=np.float32)
    for i in range(len(values)):
        lo = max(0, i - span)
        prev = values[lo:i] if i > lo else values[: i + 1]
        out[i] = fn(prev) if len(prev) else 0.0
    return out


def chat_series(msgs, covered, duration):
    """Per 5 s bucket: weighted chat volume, message count, chatters, plus the messages.
    Sampled (not fully read) parts of big chats are scaled up by how much was read."""
    n = int(duration // BUCKET) + 1
    weight = np.zeros(n, dtype=np.float32)
    count = np.zeros(n, dtype=np.float32)
    users = [set() for _ in range(n)]
    per = [[] for _ in range(n)]
    for off, user, text in msgs:
        if is_noise(user, text):
            continue
        b = int(off // BUCKET)
        if 0 <= b < n:
            weight[b] += message_weight(text)
            count[b] += 1
            users[b].add(user.lower())
            per[b].append(text)
    seen = np.zeros(n, dtype=np.float32)  # fraction of each bucket we actually read
    if covered:
        for a, b in covered:
            for k in range(int(a // BUCKET), min(n, int(b // BUCKET) + 1)):
                lo, hi = max(a, k * BUCKET), min(b, (k + 1) * BUCKET)
                if hi > lo:
                    seen[k] = min(1.0, seen[k] + (hi - lo) / BUCKET)
    else:
        seen[:] = 1.0
    filled = seen > 0.05
    scale = np.where(filled, 1.0 / np.maximum(seen, 0.05), 0.0)
    weight *= scale
    count *= scale
    if filled.any() and not filled.all():  # fill unread buckets from their neighbours
        idx = np.arange(n)
        weight = np.interp(idx, idx[filled], weight[filled]).astype(np.float32)
        count = np.interp(idx, idx[filled], count[filled]).astype(np.float32)
    chatters = np.array([len(u) for u in users], dtype=np.float32) * np.where(filled, scale, 1.0)
    return weight, count, chatters, per


def find_moments(duration: float, msgs=None, covered=None, audio=None, comments=None,
                 heatmap=None, threshold: float = 2.5):
    """All candidate moments with scores, best first."""
    secs = int(duration) + 1
    hot = np.zeros(secs, dtype=bool)
    chat_ratio = np.zeros(secs, dtype=np.float32)
    boost = np.zeros(secs, dtype=np.float32)
    react = np.zeros(secs, dtype=np.float32)
    have_chat = bool(msgs)

    if have_chat:
        w, cnt, chatters, per = chat_series(msgs, covered, duration)
        base = np.maximum(_rolling(w, int(180 / BUCKET), lambda v: 0.7 * float(np.median(v)) + 0.3 * float(np.mean(v))), 2.0)
        ratio = w / base
        chat_hot_b = (ratio >= threshold) & (cnt >= 6) & (chatters >= 4)
        for b in range(len(w)):
            s0, s1 = int(b * BUCKET), min(secs, int((b + 1) * BUCKET))
            chat_ratio[s0:s1] = ratio[b]
            hot[s0:s1] |= chat_hot_b[b]

    if audio is not None:
        loud, laugh, scream, cheer = audio
        if len(loud) > 40:
            sec_db = np.array([float(np.mean(loud[i * 4:(i + 1) * 4])) for i in range(len(loud) // 4)], dtype=np.float32)
            normal = _rolling(sec_db, 180, lambda v: float(np.percentile(v, 80)))
            boost[: len(sec_db)] = np.maximum(0, sec_db - normal)[:secs]
        strength = lambda p: np.clip((p - 0.08) / 0.35, 0, 1)
        voice = np.maximum(np.maximum(strength(laugh), strength(scream)), 0.6 * strength(cheer))
        strong = (voice >= 0.6).astype(np.float32)
        lasting = np.convolve(strong, np.ones(5), mode="same") >= 2
        level = np.minimum(1.0, 0.6 * voice + 0.15 * lasting)
        react[: len(level)] = level[:secs]
        loud_hot = (boost >= 8) & (chat_ratio >= threshold * 0.75) if have_chat else np.zeros(secs, bool)
        react_hot = (react >= 0.6) | ((react >= 0.45) & (boost >= 8))
        hot |= loud_hot | react_hot

    # comment timestamps: lots of people pointing at the same second
    comment_hits = np.zeros(secs, dtype=np.float32)
    for t, likes in comments or []:
        i = int(t)
        if 0 <= i < secs:
            comment_hits[max(0, i - 15): i + 5] += 1 + min(likes, 200) / 50
    if comments:
        hot |= comment_hits >= 3
    heat = np.zeros(secs, dtype=np.float32)
    for h in heatmap or []:
        heat[int(h["start_time"]): int(h["end_time"]) + 1] = h["value"]
    if heatmap:
        hot |= heat >= 0.75

    # group hot seconds into moments (gaps under 20 s are the same moment)
    moments = []
    i = 0
    while i < secs:
        if not hot[i]:
            i += 1
            continue
        j, last = i, i
        while j < secs and j - last <= 20:
            if hot[j]:
                last = j
            j += 1
        start, end = i, min(last + 1, secs - 1)
        if end - start > 75:
            end = start + 75
        moments.append(_describe(start, end, have_chat, chat_ratio, boost, react, comment_hits, heat,
                                 msgs if have_chat else None))
        i = last + 1
    moments.sort(key=lambda m: m["score"], reverse=True)
    return moments


def _describe(start, end, have_chat, chat_ratio, boost, react, comment_hits, heat, msgs):
    seg = slice(start, end + 1)
    peak_i = start + int(np.argmax(chat_ratio[seg] + react[seg] * 4 + comment_hits[seg] / 3))
    peak_ratio = float(chat_ratio[seg].max()) if have_chat else 0.0
    texts, users = [], set()
    if msgs:
        texts = [t for off, u, t in msgs if start - 2 <= off <= end and not is_noise(u, t)]
        users = {u.lower() for off, u, t in msgs if start - 2 <= off <= end}
    avg_w = statistics.fmean(message_weight(t) for t in texts) if texts else 1.0
    asks = sum(1 for t in texts if CLIP_RE.search(t))
    peak_react = float(react[seg].max())
    peak_boost = float(boost[seg].max())
    score = score_moment(max(peak_ratio, 1.0), len(users), avg_w, asks, peak_boost, reaction=peak_react)
    score += int(min(20, comment_hits[seg].max() * 4)) + int(min(15, heat[seg].max() * 15))
    reasons = []
    if peak_react >= 0.45:
        reasons.append("laughing/screaming")
    if comment_hits[seg].max() >= 3:
        reasons.append("viewers' comments")
    if heat[seg].max() >= 0.75:
        reasons.append("most replayed")
    tokens = Counter()
    for t in texts:
        for tok in set(TOKEN_RE.findall(t.lower())):
            tok = tok.strip("!?.,:;\"'()[]")
            if tok and len(tok) <= 25 and tok not in STOPWORDS and not tok.startswith("@"):
                tokens[tok] += 1
    return {"start": float(start), "end": float(end), "peak": float(peak_i), "score": int(min(100, score)),
            "peak_ratio": round(peak_ratio, 2), "messages": len(texts), "chatters": len(users),
            "audio_boost": round(peak_boost, 1), "reaction": round(peak_react, 2), "clip_requests": asks,
            "top_tokens": tokens.most_common(8), "sample_messages": [t for t, _ in Counter(texts).most_common(8)],
            "reasons": reasons}


def pick_top(moments, n: int, gap: float = 30.0):
    """Best n moments that don't overlap."""
    chosen = []
    for m in moments:
        if all(m["end"] + gap < c["start"] or m["start"] > c["end"] + gap for c in chosen):
            chosen.append(m)
        if len(chosen) >= n:
            break
    return chosen


# --------------------------------------------------------------------------
# A scan job
# --------------------------------------------------------------------------
class ScanJob:
    _ids = itertools.count(1)

    def __init__(self, src: dict, top: int, out_dir: Path, args):
        self.id = next(self._ids)
        self.src, self.top, self.out_dir, self.args = src, top, out_dir, args
        self.stop = threading.Event()
        self.stage, self.progress, self.message = "waiting", 0.0, "Waiting for the previous scan to finish"
        self.status = "queued"  # queued | running | done | failed | cancelled
        self.started = self.finished = None
        self.found = 0
        self.made = 0
        self.wanted = 0
        self._procs: list[subprocess.Popen] = []

    def state(self) -> dict:
        eta = None
        if self.status == "running" and self.started and self.progress > 0.03:
            eta = int((time.time() - self.started) * (1 - self.progress) / self.progress)
        return {"id": self.id, "title": self.src.get("title"), "name": self.src["name"],
                "platform": self.src["platform"], "url": self.src["url"], "duration": self.src.get("duration"),
                "status": self.status, "stage": self.stage, "progress": round(self.progress, 3),
                "message": self.message, "eta": eta, "found": self.found, "made": self.made,
                "wanted": self.wanted, "top": self.top, "chat": self.src.get("chat")}

    def _set(self, stage, lo, hi, frac, message=None):
        self.stage, self.progress = stage, lo + (hi - lo) * max(0.0, min(1.0, frac))
        if message:
            self.message = message

    def run(self, finisher, log) -> None:
        src, dur = self.src, self.src.get("duration") or 0
        self.status, self.started = "running", time.time()
        name = src["name"]
        try:
            # 1+2. chat replay and YouTube comments, in the background while we listen
            got = {"msgs": [], "covered": None, "comments": [], "chat_p": 0.0 if src.get("chat") else 1.0,
                   "audio_p": 0.0, "dl_p": 0.0}
            phase = {"text": "Starting…"}

            def report():
                weight_chat = 0.35 if src.get("chat") else 0.0
                audio_part = 0.15 * got["dl_p"] + 0.85 * got["audio_p"] if src["platform"] == "youtube" else got["audio_p"]
                frac = (weight_chat * got["chat_p"] + (1 - weight_chat) * audio_part)
                bits = [phase["text"]]
                if src.get("chat") and got["chat_p"] < 1:
                    bits.append(f"chat replay {int(got['chat_p'] * 100)}%")
                self._set("scanning", 0, 0.8, frac, " · ".join(bits))

            def chat_job():
                try:
                    if src.get("chat") == "twitch":
                        got["msgs"], got["covered"] = sources.twitch_chat_replay(
                            src["video_id"], dur, stop=self.stop,
                            progress=lambda p: (got.__setitem__("chat_p", p), report()))
                    elif src.get("chat") == "youtube":
                        got["msgs"] = sources.youtube_chat_replay(
                            src["url"], dur, stop=self.stop,
                            progress=lambda p: (got.__setitem__("chat_p", p), report()))
                except Exception as e:
                    log(name, f"couldn't read the chat replay ({e}); using reactions only")
                got["chat_p"] = 1.0
                if src["platform"] == "youtube":
                    try:
                        got["comments"] = sources.youtube_comment_times(src["url"], dur)
                    except Exception:
                        pass

            side = threading.Thread(target=chat_job, daemon=True)
            side.start()

            local = None
            if src["platform"] == "youtube":
                phase["text"] = "Downloading the audio…"
                local = _download_audio(src, self.out_dir / ".scan", lambda p: (got.__setitem__("dl_p", p), report()),
                                        self.stop, self._procs)
            try:
                phase["text"] = "Listening for laughs, screams and reactions…"
                audio = analyze_audio(src, lambda p: (got.__setitem__("audio_p", p), report()),
                                      self.stop, procs=self._procs, local_file=local)
            finally:
                if local is not None:
                    local.unlink(missing_ok=True)
            phase["text"] = "Finishing the chat replay…"
            while side.is_alive() and not self.stop.is_set():
                report()
                side.join(1)
            if self.stop.is_set():
                raise InterruptedError
            msgs, covered, comments = got["msgs"], got["covered"], got["comments"]
            log(name, f"scan: {len(msgs)} chat messages" if msgs else "scan: no chat replay for this video")

            # 4. pick
            self._set("picking", 0.8, 0.82, 0, "Picking the best moments…")
            moments = find_moments(dur, msgs, covered, audio, comments, src.get("heatmap"))
            self.found = len(moments)
            best = pick_top(moments, self.top)
            self.wanted = len(best)
            if not best:
                self.status, self.message = "done", "No stand-out moments found in this video"
                return
            log(name, f"scan: {len(moments)} moments found, clipping the best {len(best)}: "
                      + ", ".join(f"{_fmt(m['start'])} (score {m['score']})" for m in best))

            # 5. who is the streamer? (so shorts never crop a game character or a teammate)
            self._set("clips", 0.8, 0.82, 0.5, "Learning which face is the streamer…")
            try:
                face = learn_streamer(src, self.out_dir, self.stop, self._procs, log=log)
            except Exception as e:
                log(name, f"couldn't learn the streamer's face ({e})")
                face = None
            self.streamer_face = [round(float(v), 5) for v in face] if face is not None else None

            # 6. download just those parts (3 at a time), then finish (trim + short)
            from concurrent.futures import ThreadPoolExecutor
            done_dl = {"n": 0}
            dl_lock = threading.Lock()
            self._set("clips", 0.82, 1.0, 0, f"Downloading the best {len(best)} moments in full quality…")

            def grab(m):
                if self.stop.is_set():
                    return
                free_gb = shutil.disk_usage(self.out_dir).free / 1e9
                if free_gb < self.args.min_free_gb:
                    log(name, f"only {free_gb:.1f} GB disk free - skipping the rest (free up space!)")
                    self.low_disk = True
                    self.stop.set()
                    return
                self._clip(m, finisher, log)
                with dl_lock:
                    done_dl["n"] += 1
                    self._set("clips", 0.82, 1.0, done_dl["n"] / len(best),
                              f"Downloaded {done_dl['n']} of {len(best)} moments in full quality…")

            with ThreadPoolExecutor(max_workers=3) as pool:
                list(pool.map(grab, best))
            if self.stop.is_set():
                raise InterruptedError
            self._set("clips", 0.82, 1.0, 1.0)
            self.message = f"Done: {len(best)} clips (they show up below as each one finishes)"
            self.status = "done"
        except InterruptedError:
            if getattr(self, "low_disk", False):
                self.status, self.message = "failed", f"Stopped early: disk almost full ({self.made} clips made)"
            else:
                self.status, self.message = "cancelled", "Cancelled"
        except Exception as e:
            self.status, self.message = "failed", f"Failed: {e}"
            log(name, f"scan failed: {e}")
        finally:
            self.finished = time.time()
            for p in self._procs:
                if p.poll() is None:
                    p.terminate()

    def _clip(self, m: dict, finisher, log) -> None:
        src, dur = self.src, self.src.get("duration") or 0
        a = max(0.0, m["start"] - self.args.context)
        b = min(dur, m["end"] + 15)
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base = self.out_dir / src["name"] / f"{src['name']}_VOD-{_fmt(m['start']).replace(':', '-')}_{stamp}"
        work = base.parent / ".work"
        work.mkdir(parents=True, exist_ok=True)
        wide = work / f"{base.name}_wide.mp4"
        fmt = ("best[height<=1080]/best" if src["platform"] in ("twitch", "kick")
               else "bv*[height<=1080][vcodec^=avc1]+ba[ext=m4a]/b[height<=1080]/b")
        p = subprocess.Popen([sources.YTDLP, "-f", fmt, "--download-sections", f"*{a:.1f}-{b:.1f}",
                              "--merge-output-format", "mp4", "--no-warnings", "--quiet", "--no-part",
                              "-o", str(wide), src["url"]], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        self._procs.append(p)
        _, err = p.communicate()
        if p.returncode != 0 or not wide.exists():
            log(src["name"], f"couldn't download the moment at {_fmt(m['start'])}: {(err or '').strip()[-150:]}")
            return
        now = time.time()
        info = {"channel": src["name"], "platform": src["platform"], "credit": src["credit"],
                "moment_time": datetime.now().isoformat(timespec="seconds"),
                "category": src.get("category"),
                "layout": "irl" if (src.get("category") or "").lower() in IRL_CATEGORIES else "auto",
                "spike_start": now, "peak_time": now, "spike_end": now,
                "kind": "vod", "source_url": src["url"], "source_title": src.get("title"),
                "vod_time": m["start"], "vod_time_text": _fmt(m["start"]),
                "streamer_face": getattr(self, "streamer_face", None),
                **{k: m[k] for k in ("peak_ratio", "messages", "chatters", "top_tokens", "sample_messages",
                                     "score", "audio_boost", "reaction", "clip_requests", "reasons")}}
        rel = {"spike_start": m["start"] - a, "spike_end": m["end"] - a, "peak": m["peak"] - a,
               "clip_start_wall": now}

        def ready(path, info_):
            self.made += 1
            log(src["name"], f"VOD CLIP SAVED -> {path} (from {info_['vod_time_text']}, score {info_['score']})")
        finisher.finish(wide, base, info, rel, ready)

    def cancel(self) -> None:
        self.stop.set()
        for p in self._procs:
            if p.poll() is None:
                p.terminate()


class ScanManager:
    """Runs scans one at a time (they're heavy on the network and CPU)."""

    def __init__(self, args, out_dir: Path, get_finisher, log):
        self.args, self.out_dir, self.get_finisher, self.log = args, out_dir, get_finisher, log
        self.jobs: list[ScanJob] = []
        self.lock = threading.Lock()
        self.wake = threading.Event()
        threading.Thread(target=self._loop, daemon=True, name="scans").start()

    def start(self, src: dict, top: int = 10) -> ScanJob:
        job = ScanJob(src, max(1, min(int(top), 50)), self.out_dir, self.args)
        with self.lock:
            self.jobs.append(job)
        self.log(src["name"], f"scan queued: {src.get('title')} ({_fmt(src.get('duration') or 0)})")
        self.wake.set()
        return job

    def cancel(self, job_id: int) -> bool:
        with self.lock:
            job = next((j for j in self.jobs if j.id == job_id), None)
        if not job:
            return False
        if job.status == "queued":
            job.status, job.message = "cancelled", "Cancelled"
        job.cancel()
        return True

    def dismiss(self, job_id: int) -> bool:
        with self.lock:
            before = len(self.jobs)
            self.jobs = [j for j in self.jobs if not (j.id == job_id and j.status in ("done", "failed", "cancelled"))]
            return len(self.jobs) != before

    def state(self) -> list[dict]:
        with self.lock:
            return [j.state() for j in self.jobs]

    def _loop(self) -> None:
        while True:
            self.wake.wait(2)
            self.wake.clear()
            with self.lock:
                job = next((j for j in self.jobs if j.status == "queued"), None)
            if job:
                job.run(self.get_finisher(), self.log)
