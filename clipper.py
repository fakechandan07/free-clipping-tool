#!/usr/bin/env python3
"""Live Twitch auto-clipper.

Records a Twitch stream into a short rolling buffer, watches chat, and when
chat goes crazy it saves a clip from the buffer (starting before the spike,
because chat reacts a few seconds after the moment).

    python clipper.py kaicenat xqc --sensitivity 3

Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import json
import queue
import os
import random
import re
import shutil
import signal
import socket
import ssl
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from hype import HypeDetector, Moment, score_moment

# OpenCV prints harmless decoder/backend warnings (e.g. at the edges of downloaded
# sections); keep the log readable.
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

HERE = Path(__file__).resolve().parent
import platformkit  # noqa: E402  (puts Windows tools on PATH before we look for them)
STREAMLINK = platformkit.venv_tool("streamlink")
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"

SEGMENT_SECONDS = 2
# Twitch categories where the camera IS the stream -> shorts show the stream in the middle 80%.
IRL_CATEGORIES = {
    "irl", "just chatting", "travel & outdoors", "talk shows & podcasts", "food & drink",
    "pools, hot tubs, and beaches", "fitness & health", "asmr", "music", "art",
    "makers & crafting", "sports", "beauty & body art", "animals, aquariums, and zoos",
}
MODES = ("auto", "gaming", "irl")
_print_lock = threading.Lock()


def log(channel: str, msg: str) -> None:
    with _print_lock:
        print(f"{datetime.now():%H:%M:%S} [{channel}] {msg}", flush=True)


def parse_channel(arg: str) -> str | None:
    """'https://www.twitch.tv/xQc?x=1', 'twitch.tv/xqc', '#xqc' or 'xqc' -> 'xqc'."""
    m = re.search(r"twitch\.tv/([A-Za-z0-9_]+)", arg)
    name = (m.group(1) if m else arg).strip().lstrip("#@").lower()
    return name if re.fullmatch(r"[a-z0-9_]{2,25}", name) else None


def channel_name(arg: str) -> str:
    name = parse_channel(arg)
    if not name:
        raise SystemExit(f"'{arg}' doesn't look like a Twitch channel name or link")
    return name


# --------------------------------------------------------------------------
# Chat: anonymous read-only connection to Twitch IRC (no login needed)
# --------------------------------------------------------------------------
PRIVMSG_RE = re.compile(r"^:([^!]+)![^ ]+ PRIVMSG #[^ ]+ :(.*)$")


def _ssl_context() -> ssl.SSLContext:
    """python.org builds of Python on macOS ship without CA certificates, so use
    certifi's bundle (installed alongside streamlink) when it's available."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _parse_tags(line: str) -> tuple[dict, str]:
    """'@a=1;b=2 :rest' -> ({'a': '1', 'b': '2'}, ':rest')"""
    if not line.startswith("@"):
        return {}, line
    tag_str, _, rest = line[1:].partition(" ")
    tags = dict(kv.partition("=")[::2] for kv in tag_str.split(";"))
    return tags, rest


class ChatListener(threading.Thread):
    def __init__(self, channel: str, on_message, stop: threading.Event, on_event=None):
        super().__init__(daemon=True, name=f"chat-{channel}")
        self.channel, self.on_message, self.stop_event = channel, on_message, stop
        self.on_event = on_event  # raids / gift bombs: chat floods that aren't a moment
        self.connected = False

    def run(self) -> None:
        backoff = 2
        while not self.stop_event.is_set():
            try:
                self._session()
                backoff = 2
            except Exception as e:  # network hiccup, Twitch restart, etc.
                self.connected = False
                if self.stop_event.is_set():
                    break
                log(self.channel, f"chat disconnected ({e}); reconnecting in {backoff}s")
                self.stop_event.wait(backoff)
                backoff = min(backoff * 2, 60)

    def _session(self) -> None:
        raw = socket.create_connection(("irc.chat.twitch.tv", 6697), timeout=15)
        sock = _ssl_context().wrap_socket(raw, server_hostname="irc.chat.twitch.tv")
        sock.settimeout(1.0)
        nick = f"justinfan{random.randint(10000, 99999)}"
        sock.sendall(f"CAP REQ :twitch.tv/tags twitch.tv/commands\r\n"
                     f"NICK {nick}\r\nJOIN #{self.channel}\r\n".encode())
        buf, last_data = b"", time.time()
        try:
            while not self.stop_event.is_set():
                try:
                    data = sock.recv(65536)
                except socket.timeout:
                    if time.time() - last_data > 360:
                        raise ConnectionError("no data from Twitch chat for 6 minutes")
                    continue
                if not data:
                    raise ConnectionError("server closed the connection")
                last_data = time.time()
                buf += data
                *lines, buf = buf.split(b"\r\n")
                for line in lines:
                    tags, text = _parse_tags(line.decode("utf-8", "replace"))
                    if text.startswith("PING"):
                        sock.sendall(b"PONG :tmi.twitch.tv\r\n")
                    elif " 366 " in text and not self.connected:
                        self.connected = True
                        log(self.channel, "chat connected")
                    elif text.startswith(":tmi.twitch.tv RECONNECT"):
                        raise ConnectionError("Twitch asked us to reconnect")
                    elif " USERNOTICE #" in text and self.on_event:
                        kind = tags.get("msg-id", "")
                        if kind == "raid":
                            self.on_event(time.time(), "raid", int(tags.get("msg-param-viewerCount") or 0))
                        elif kind == "submysterygift":
                            self.on_event(time.time(), "giftbomb",
                                          int(tags.get("msg-param-mass-gift-count") or 0))
                    else:
                        m = PRIVMSG_RE.match(text)
                        if m:
                            self.on_message(time.time(), m.group(1), m.group(2))
        finally:
            self.connected = False
            sock.close()


# --------------------------------------------------------------------------
# Recorder: streamlink -> ffmpeg -> 2-second .ts segments in a rolling buffer
# --------------------------------------------------------------------------
class Recorder(threading.Thread):
    def __init__(self, channel: str, buffer_dir: Path, quality: str, stop: threading.Event,
                 offline_retry: int, on_audio=None, on_pcm=None, on_frame=None, url: str | None = None):
        super().__init__(daemon=True, name=f"rec-{channel}")
        self.url = url or f"https://www.twitch.tv/{channel}"  # Twitch or YouTube live
        self.on_audio = on_audio  # (time, loudness dB) 4x per second
        self.on_pcm = on_pcm  # (time, 16 kHz mono 16-bit audio) for the sound classifier
        self.on_frame = on_frame  # (time, BGR image) every keyframe (~2 s) for face reading
        self._frames: queue.Queue = queue.Queue(maxsize=8)
        self.channel, self.buffer_dir, self.quality = channel, buffer_dir, quality
        self.stop_event, self.offline_retry = stop, offline_retry
        self.segments: list[tuple[float, Path, int]] = []  # (wall start, path, run id)
        self.lock = threading.Lock()
        self.live = False
        self.live_since = 0.0
        self.current_run_dir: Path | None = None
        self._procs: list[subprocess.Popen] = []

    def run(self) -> None:
        run_id = 0
        while not self.stop_event.is_set():
            run_id += 1
            run_dir = self.buffer_dir / f"run{int(time.time())}_{run_id}"
            self.current_run_dir = run_dir  # set first so cleanup never removes it
            try:
                run_dir.mkdir(parents=True, exist_ok=True)
                got_video = self._record(run_dir, run_id)
            except Exception as e:  # never let one bad run kill recording for good
                log(self.channel, f"recorder error ({e!r}); restarting")
                self.kill()
                got_video = True  # retry soon rather than waiting the offline delay
            self.live = False
            if self.stop_event.is_set():
                break
            if got_video:
                log(self.channel, "stream ended or dropped; reconnecting in 10s")
                self.stop_event.wait(10)
            else:
                log(self.channel, f"offline; checking again in {self.offline_retry}s")
                self.stop_event.wait(self.offline_retry)

    def _record(self, run_dir: Path, run_id: int) -> bool:
        err_log = open(self.buffer_dir / "streamlink.log", "w+b")
        sl = subprocess.Popen(
            [STREAMLINK, "--twitch-low-latency", "--twitch-supported-codecs", "h264",
             "--loglevel", "error", "--stdout", self.url,
             self.quality],
            stdout=subprocess.PIPE, stderr=err_log,
        )
        ff = subprocess.Popen(
            [FFMPEG, "-hide_banner", "-loglevel", "error",
             # decode only keyframes (for the face pictures); copied video is untouched
             "-skip_frame:v", "nokey", "-i", "pipe:0",
             "-map", "0:v:0?", "-map", "0:a:0?", "-c", "copy",
             "-f", "segment", "-segment_time", str(SEGMENT_SECONDS),
             "-segment_format", "mpegts", str(run_dir / "seg_%06d.ts"),
             # second output: mono audio so we can hear the streamer (loudness, laughs, screams)
             "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-f", "s16le", "pipe:1",
             # third output: a small picture at every keyframe (~2 s) to read the facecam
             "-map", "0:v:0", "-vf", "scale=960:-2", "-fps_mode", "passthrough", "-q:v", "6",
             "-f", "image2", str(run_dir / "kf_%06d.jpg")],
            stdin=sl.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        sl.stdout.close()  # ffmpeg owns the pipe now
        self._procs = [sl, ff]
        threading.Thread(target=self._read_audio, args=(ff.stdout,), daemon=True).start()
        threading.Thread(target=self._read_frames, args=(ff,), daemon=True).start()
        pics: set[str] = set()
        seen: set[str] = set()
        got_video = False
        while not self.stop_event.is_set() and ff.poll() is None:
            try:
                entries = sorted(os.scandir(run_dir), key=lambda e: e.name)
            except FileNotFoundError:
                break  # folder vanished; outer loop restarts the recording
            jpgs = [e for e in entries if e.name.startswith("kf_") and e.name.endswith(".jpg")]
            for entry in jpgs[:-1]:  # newest one may still be being written
                if entry.name not in pics:
                    pics.add(entry.name)
                    try:
                        self._frames.put_nowait((time.time(), Path(entry.path)))
                    except queue.Full:  # face reading fell behind: skip this picture
                        Path(entry.path).unlink(missing_ok=True)
            for entry in entries:
                if entry.name.endswith(".ts") and entry.name not in seen:
                    seen.add(entry.name)
                    with self.lock:
                        self.segments.append((time.time(), Path(entry.path), run_id))
                    if not got_video:
                        got_video = self.live = True
                        self.live_since = time.time()
                        log(self.channel, "LIVE - recording into buffer")
            time.sleep(0.25)
        self.kill()
        err_log.seek(0)
        err = err_log.read().decode(errors="replace").strip()
        err_log.close()
        if not got_video and err and "No playable streams" not in err:
            log(self.channel, f"streamlink: {err.splitlines()[-1][:200]}")
        return got_video

    def _read_audio(self, pipe) -> None:
        """Loudness (dBFS) of each quarter second of the stream's audio."""
        import array
        import math
        while True:
            chunk = pipe.read(8000)  # 4000 samples at 16 kHz = 0.25 s
            if not chunk:
                return
            if self.on_audio is None or len(chunk) < 2:
                continue
            try:  # never stop reading: a full pipe would freeze the recording
                samples = array.array("h", chunk[: len(chunk) // 2 * 2])
                rms = math.sqrt(sum(x * x for x in samples) / len(samples))
                now = time.time()
                self.on_audio(now, 20 * math.log10(rms / 32768 + 1e-6))
                if self.on_pcm is not None:
                    self.on_pcm(now, chunk)
            except Exception:
                pass

    def _read_frames(self, ff) -> None:
        """Face reading runs here, off the recording path; pictures are deleted after."""
        import cv2
        while not self.stop_event.is_set():
            try:
                ts, path = self._frames.get(timeout=1)
            except queue.Empty:
                if ff.poll() is not None:
                    return  # this recording run ended
                continue
            try:
                if self.on_frame is not None:
                    img = cv2.imread(str(path))
                    if img is not None:
                        self.on_frame(ts, img)
            except Exception:
                pass
            finally:
                path.unlink(missing_ok=True)

    def kill(self) -> None:
        for p in self._procs:
            if p.poll() is None:
                p.terminate()
        for p in self._procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()

    def latest_start(self) -> float:
        with self.lock:
            return self.segments[-1][0] if self.segments else 0.0

    def segments_for(self, start: float, end: float) -> tuple[list[Path], float]:
        """Completed segments overlapping [start, end] from a single recording run, and
        the wall-clock time the first of them starts."""
        with self.lock:
            segs = list(self.segments)
        picked: dict[int, list[tuple[float, Path]]] = {}
        for i, (t0, path, run) in enumerate(segs[:-1]):  # last one is still being written
            nxt_t, _, nxt_run = segs[i + 1]
            t1 = nxt_t if nxt_run == run else t0 + SEGMENT_SECONDS
            if t0 < end and t1 > start and path.exists():
                picked.setdefault(run, []).append((t0, path))
        if not picked:
            return [], 0.0
        best = max(picked.values(), key=len)
        return [p for _, p in best], best[0][0]

    def cleanup(self, keep_after: float) -> None:
        with self.lock:
            old = [s for s in self.segments[:-2] if s[0] < keep_after]
            self.segments = [s for s in self.segments if s not in old]
        for _, path, _ in old:
            path.unlink(missing_ok=True)
        for d in self.buffer_dir.glob("run*"):
            if d == self.current_run_dir:
                continue
            try:
                d.rmdir()  # only succeeds once a finished run's folder is empty
            except OSError:
                pass


# --------------------------------------------------------------------------
# One channel = chat listener + recorder + hype detector + clip cutter
# --------------------------------------------------------------------------
class ChannelClipper:
    def __init__(self, channel: str, args, stop: threading.Event, source: dict | None = None):
        # where the stream is: Twitch (default) or YouTube live
        self.source = source or {"platform": "twitch", "record_url": f"https://www.twitch.tv/{channel}",
                                 "credit": f"twitch.tv/{channel}"}
        self.platform = self.source.get("platform", "twitch")
        self.channel, self.args, self.stop_event = channel, args, stop
        self.detector = HypeDetector(threshold=args.sensitivity, min_messages=args.min_messages,
                                     min_chatters=args.min_chatters, cooldown=args.cooldown)
        self.det_lock = threading.Lock()
        if self.platform == "youtube":
            from sources import YouTubeLiveChat
            self.chat = YouTubeLiveChat(self.source["record_url"], self._on_message, stop,
                                        log=lambda m: log(channel, m))
        elif self.platform == "kick":
            from sources import KickChat
            slug = self.source.get("kick_slug") or self.source["record_url"].rstrip("/").split("/")[-1]
            self.chat = KickChat(slug, self._on_message, stop, log=lambda m: log(channel, m))
        else:
            self.chat = ChatListener(channel, self._on_message, stop, on_event=self._on_event)
        self.out_dir = Path(args.out).expanduser() / channel
        self.recorder = None
        self.reactions = None
        self.identity = None  # who the streamer is (face recognition), learned live
        self._identity_saved = time.time()
        if not args.chat_only:
            self.reactions = None
            if not args.no_reactions:
                try:
                    from reactions import ReactionTracker
                    import identity as ident
                    if ident.available():
                        self.identity = ident.StreamerIdentity(
                            ident.identity_file(Path(args.out).expanduser(), channel))
                    self.reactions = ReactionTracker(identity=self.identity)
                except Exception as e:
                    log(channel, f"streamer reaction detection unavailable: {e}")
            self.recorder = Recorder(channel, Path(args.buffer).expanduser() / channel,
                                     args.quality, stop, args.offline_retry, on_audio=self._on_audio,
                                     on_pcm=self.reactions.add_audio if self.reactions else None,
                                     on_frame=self.reactions.add_frame if self.reactions else None,
                                     url=self.source["record_url"])
        self.pending: list[tuple[float, float, Moment]] = []
        self.msg_count = 0
        self.clips_saved = 0
        self.test_clip_done = not args.test_clip
        # live stats for the dashboard
        self.history: deque = deque(maxlen=900)  # (time, hype ratio) once per second, 15 min
        self.recent_msgs: deque = deque()  # message times in the last 60s
        self.moments: deque = deque(maxlen=100)  # detected moments, for graph markers
        self.manual_requested = False
        self.saved_clips: deque = deque(maxlen=50)  # (wall start, wall end, path)
        # viewer-made Twitch clips (optional, needs free Twitch dev keys)
        self._vclip_seen: set[str] = set()
        self._vclip_loose: list[float] = []  # creation times not matched to our clips yet
        self._vclip_at = 0.0
        # shorts layout: auto (IRL if Twitch category says so), gaming (facecam 30% /
        # gameplay 70%) or irl (stream fills the middle 80%)
        self.mode = "auto"
        self.category: str | None = None
        self.title: str | None = None
        self._meta_at = 0.0

    def _on_audio(self, ts: float, db: float) -> None:
        with self.det_lock:
            self.detector.feed_audio(ts, db)

    def _on_event(self, ts: float, kind: str, size: int) -> None:
        with self.det_lock:
            self.detector.feed_event(ts, kind, size)
        if (kind == "raid" and size >= 10) or (kind == "giftbomb" and size >= 5):
            what = f"raid with {size} viewers" if kind == "raid" else f"{size} gifted subs"
            log(self.channel, f"{what} - ignoring the chat flood for a bit")

    def _on_message(self, ts: float, user: str, text: str) -> None:
        with self.det_lock:
            self.detector.feed(ts, user, text)
            self.msg_count += 1
            self.recent_msgs.append(ts)

    def start(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.chat.start()
        if self.recorder:
            self.recorder.start()
        threading.Thread(target=self._loop, daemon=True, name=f"loop-{self.channel}").start()

    # -- controls used by the dashboard ----------------------------------------
    def set_sensitivity(self, value: float) -> None:
        with self.det_lock:
            self.detector.threshold = max(1.2, min(float(value), 20.0))
        log(self.channel, f"sensitivity set to {self.detector.threshold:.1f}x")

    def request_clip(self) -> None:
        self.manual_requested = True

    def set_reactions(self, on: bool) -> None:
        with self.det_lock:
            self.detector.reactions_on = bool(on)
        log(self.channel, f"streamer reactions {'on' if on else 'off'}")

    def set_mode(self, mode: str) -> bool:
        if mode not in MODES:
            return False
        self.mode = mode
        log(self.channel, f"shorts mode: {mode}")
        return True

    def is_irl(self) -> bool:
        if self.mode != "auto":
            return self.mode == "irl"
        return (self.category or "").strip().lower() in IRL_CATEGORIES

    def _fetch_meta(self) -> None:
        """Ask Twitch (via streamlink) what the stream's category and title are."""
        try:
            r = subprocess.run([STREAMLINK, "--json", self.source["record_url"]],
                               capture_output=True, text=True, timeout=40)
            meta = json.loads(r.stdout or "{}").get("metadata") or {}
            cat = meta.get("category")
            if cat and cat != self.category:
                self.category = cat
                log(self.channel, f"category: {cat}" + (" (IRL)" if self.is_irl() else ""))
            self.title = meta.get("title") or self.title
        except (subprocess.SubprocessError, ValueError, OSError):
            pass

    def state(self) -> dict:
        now = time.time()
        with self.det_lock:
            while self.recent_msgs and self.recent_msgs[0] < now - 60:
                self.recent_msgs.popleft()
            msgs_per_min = len(self.recent_msgs)
            threshold = self.detector.threshold
            ratio, spiking = self.detector.last_ratio, self.detector.in_spike
            warm = (self.detector._first_ts is not None
                    and now - self.detector._first_ts >= self.detector.warmup)
        rec = self.recorder
        return {
            "channel": self.channel,
            "platform": self.platform,
            "credit": self.source.get("credit"),
            "link": self.source.get("record_url"),
            "status": "chat-only" if rec is None else ("live" if rec.live else "offline"),
            "chat_connected": self.chat.connected,
            "warming_up": not warm,
            "msgs_per_min": msgs_per_min,
            "hype": round(ratio, 2),
            "spiking": spiking,
            "threshold": threshold,
            "clips_saved": self.clips_saved,
            "pending_clips": len(self.pending),
            "audio_boost": round(self.detector.last_audio_boost, 1),
            "reaction": round(self.detector.last_reaction, 2),
            "reacting": list(getattr(self, "last_why", []) or []),
            "reactions_on": self.detector.reactions_on and self.reactions is not None,
            "reaction_inputs": self.reactions.stats(now) if self.reactions else None,
            "ignoring": self.detector.suppressed_reason,
            "mode": self.mode,
            "category": self.category,
            "irl": self.is_irl(),
            "history": [[round(h[0], 1), round(h[1], 2), round(h[2], 2)] for h in list(self.history)],
            "moments": list(self.moments),
        }

    def _loop(self) -> None:
        last_status = last_sample = time.time()
        while not self.stop_event.is_set():
            now = time.time()
            with self.det_lock:
                was_spiking = self.detector.in_spike
                react, why = self.reactions.level(now) if self.reactions else (0.0, [])
                self.last_why = why
                moment = self.detector.tick(now, reaction=react, why=why)
                ratio, spiking = self.detector.last_ratio, self.detector.in_spike
            if now - last_sample >= 1.0:
                last_sample = now
                self.history.append((now, ratio, self.detector.last_reaction))
            if (self.platform == "twitch" and self.recorder and self.recorder.live
                    and VIEWER_API and VIEWER_API.configured
                    and now - self._vclip_at > 60):
                self._vclip_at = now
                threading.Thread(target=self._poll_viewer_clips, daemon=True).start()
            if self.identity is not None and now - self._identity_saved > 300:
                self._identity_saved = now  # remember the streamer's face for next time
                threading.Thread(target=self.identity.save, daemon=True).start()
            if self.recorder and self.recorder.live and now - self._meta_at > 300:
                self._meta_at = now  # category can change mid-stream, so re-check every 5 min
                threading.Thread(target=self._fetch_meta, daemon=True).start()
            if self.manual_requested:
                self.manual_requested = False
                log(self.channel, "manual clip requested from dashboard")
                self._on_moment(Moment(spike_start=now - 5, peak_time=now - 2, spike_end=now,
                                       peak_ratio=round(ratio, 2), messages=0, chatters=0,
                                       kind="manual"))
            if spiking and not was_spiking:
                log(self.channel, f"chat is going crazy ({ratio:.1f}x normal)...")
            if moment:
                self._on_moment(moment)
            if (not self.test_clip_done and self.recorder and self.recorder.live
                    and now - self.recorder.live_since >= self.args.context + 5):
                self.test_clip_done = True
                log(self.channel, "test clip: pretending chat just went crazy")
                self._on_moment(Moment(spike_start=now - 5, peak_time=now - 2, spike_end=now,
                                       peak_ratio=0.0, messages=0, chatters=0, kind="test"))
            self._process_pending(now)
            if self.recorder:
                oldest_needed = min([s for s, _, _ in self.pending], default=now)
                self.recorder.cleanup(min(now - self.args.buffer_seconds, oldest_needed - 5))
            if now - last_status >= self.args.status_every:
                last_status = now
                live = "chat-only" if not self.recorder else ("LIVE" if self.recorder.live else "offline")
                log(self.channel, f"status: {live} | {self.msg_count} msgs | "
                                  f"hype {ratio:.1f}x (clips at {self.detector.threshold:.1f}x) | "
                                  f"{self.clips_saved} clips saved")
            self.stop_event.wait(0.5)

    def _on_moment(self, m: Moment) -> None:
        # Save a wide window: ~a minute before chat exploded (the setup) to 15 s after it
        # calmed down. The finisher then trims it at natural points using the transcript.
        start = m.spike_start - self.args.context
        end = m.spike_end + 15
        if end - start > 120:
            start = end - 120
        if m.kind == "auto":
            extra = f", streamer {m.audio_boost:.0f} dB louder" if m.audio_boost >= 6 else ""
            joined = f", {m.merged} spikes joined" if m.merged > 1 else ""
            react = f", streamer: {' + '.join(m.reasons)}" if m.reasons else ""
            log(self.channel, f"HYPE MOMENT (score {m.score}): peak {m.peak_ratio}x, {m.messages} msgs "
                              f"from {m.chatters} chatters{extra}{react}{joined} - top: "
                              f"{', '.join(t for t, _ in m.top_tokens[:4])}")
        marker = {"t": round(m.spike_start, 1), "kind": m.kind, "peak": m.peak_ratio,
                  "score": m.score, "saved": False}
        self.moments.append(marker)
        if self.recorder is None:
            return
        if not self.recorder.live:
            log(self.channel, "...but the stream isn't being recorded right now, skipping")
            return
        self.pending.append((start, end, m))

    def _process_pending(self, now: float) -> None:
        if not self.recorder:
            return
        ready = [p for p in self.pending
                 if self.recorder.latest_start() > p[1] + SEGMENT_SECONDS or now > p[1] + 60]
        for p in ready:
            self.pending.remove(p)
            self._cut(*p)

    def _cut(self, start: float, end: float, m: Moment) -> None:
        segs, t0 = self.recorder.segments_for(start, end)
        if not segs:
            log(self.channel, "no video in buffer for that moment, skipping")
            return
        free_gb = shutil.disk_usage(self.out_dir).free / 1e9
        if free_gb < self.args.min_free_gb:
            log(self.channel, f"only {free_gb:.1f} GB disk free - NOT saving clip (free up space!)")
            return
        stamp = datetime.fromtimestamp(m.spike_start).strftime("%Y-%m-%d_%H-%M-%S")
        tag = {"test": "TEST", "manual": "MANUAL", "viewer": "VIEWERS"}.get(m.kind, f"{m.peak_ratio:.0f}x")
        base = self.out_dir / f"{self.channel}_{stamp}_{tag}"
        work = self.out_dir / ".work"
        work.mkdir(parents=True, exist_ok=True)
        wide = work / f"{base.name}_wide.mp4"
        result = subprocess.run(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
             "-i", "concat:" + "|".join(str(s) for s in segs),
             "-c", "copy", "-avoid_negative_ts", "make_zero", "-movflags", "+faststart",
             str(wide)],
            capture_output=True, text=True,
        )
        if result.returncode != 0 or not wide.exists():
            log(self.channel, f"ffmpeg failed to save clip: {result.stderr.strip()[-300:]}")
            return
        info = {
            "channel": self.channel,
            "platform": self.platform,
            "credit": self.source.get("credit"),
            "moment_time": datetime.fromtimestamp(m.spike_start).isoformat(timespec="seconds"),
            "category": self.category,
            "layout": "irl" if self.is_irl() else "auto",
            "streamer_face": self._streamer_face(),
            **asdict(m),
        }
        rel = {"spike_start": m.spike_start - t0, "spike_end": m.spike_end - t0,
               "peak": m.peak_time - t0, "clip_start_wall": t0}
        get_shorts_maker(self.args).finish(wide, base, info, rel, self._clip_ready)

    # -- viewer-made Twitch clips -------------------------------------------------
    @staticmethod
    def _covers(start: float, end: float, created: float) -> bool:
        """Does our clip [start, end] show what a viewer clipped at `created`? Twitch
        clips show the ~30 s before they were made."""
        return min(end, created) - max(start, created - 30) >= 10

    def _bump_clip(self, path: Path, n: int = 1) -> None:
        side = path.with_suffix(".json")
        try:
            info = json.loads(side.read_text())
        except (OSError, ValueError):
            return
        info["viewer_clips"] = int(info.get("viewer_clips", 0)) + n
        info["score"] = min(100, int(info.get("score", 0)) + 12 * n)
        side.write_text(json.dumps(info, indent=2, ensure_ascii=False))
        log(self.channel, f"viewers clipped this too ({info['viewer_clips']}) -> score {info['score']}: {path.name}")

    def _poll_viewer_clips(self) -> None:
        try:
            clips = VIEWER_API.clips_since(self.channel, time.time() - 600)
            VIEWER_API.last_error = None
        except Exception as e:
            VIEWER_API.last_error = str(e)
            return
        now = time.time()
        for c in clips:
            if c["id"] in self._vclip_seen:
                continue
            self._vclip_seen.add(c["id"])
            t = c["created"]
            hit = next((p for s0, s1, p in self.saved_clips if self._covers(s0, s1, t)), None)
            if hit:
                self._bump_clip(hit)
            else:
                self._vclip_loose.append(t)  # maybe a clip still being finished, or one we missed
        self._vclip_loose = [t for t in self._vclip_loose if now - t < 600]
        # 2+ viewers clipped the same moment and we have nothing for it: clip it ourselves
        loose = sorted(self._vclip_loose)
        for i in range(len(loose) - 1):
            group = [t for t in loose if loose[i] <= t <= loose[i] + 90]
            busy = any(abs(mk["t"] - group[0]) < 75 for mk in self.moments)
            if len(group) >= 2 and not busy and now - group[0] < 150:
                n = len(group)
                log(self.channel, f"{n} viewers made Twitch clips of a moment - clipping it too")
                self._on_moment(Moment(spike_start=group[0] - 8, peak_time=group[0] - 5,
                                       spike_end=group[-1], peak_ratio=round(self.detector.last_ratio, 2),
                                       messages=0, chatters=0, kind="viewer", viewer_clips=n,
                                       score=min(100, 40 + 12 * n)))
                self._vclip_loose = [t for t in self._vclip_loose if t not in group]
                break

    def _pair_for_duo(self, path: Path, info: dict) -> None:
        """Another streamer clipped the same moment (they're playing together)? Make a
        split-screen short with both reactions."""
        a0, a1 = info["clip_start_wall"], info["clip_end_wall"]
        with _recent_lock:
            partner = None
            for other in RECENT_CLIPS:
                ch, b0, b1, bpath, binfo = other
                if ch == self.channel or binfo.get("_paired"):
                    continue
                overlap = min(a1, b1) - max(a0, b0)
                if overlap >= 12 and overlap >= 0.5 * min(a1 - a0, b1 - b0):
                    partner = other
                    break
            RECENT_CLIPS.append((self.channel, a0, a1, path, info))
            if partner is None:
                return
            partner[4]["_paired"] = info["_paired"] = True
        ch, b0, b1, bpath, binfo = partner
        start, end = max(a0, b0), min(a1, b1)
        get_shorts_maker(self.args).duo(path, start - a0, info, bpath, start - b0, binfo, end - start)

    def _streamer_face(self):
        face = self.identity.streamer() if self.identity is not None else None
        return [round(float(v), 5) for v in face] if face is not None else None

    def _clip_ready(self, path: Path, info: dict) -> None:
        """Called by the finisher once the final clip is saved."""
        self.clips_saved += 1
        for mk in self.moments:
            if mk["t"] == round(info["spike_start"], 1):
                mk["saved"] = True
        self.saved_clips.append((info["clip_start_wall"], info["clip_end_wall"], path))
        self._pair_for_duo(path, info)
        late = [t for t in self._vclip_loose
                if self._covers(info["clip_start_wall"], info["clip_end_wall"], t)]
        if late:  # viewers clipped it while we were still finishing it
            self._vclip_loose = [t for t in self._vclip_loose if t not in late]
            self._bump_clip(path, len(late))
        log(self.channel, f"CLIP SAVED -> {path}  ({info['clip_seconds']}s, score {info.get('score', 0)})")

    def shutdown(self) -> None:
        self.stop_event.set()
        if self.recorder:
            self.recorder.kill()
        if self.identity is not None:
            self.identity.save()


class ShortsMaker(threading.Thread):
    """Background worker, one job at a time so it never slows down recording:
    trims each new clip at natural points (smartcut) and turns it into a short."""

    def __init__(self, args):
        super().__init__(daemon=True, name="shorts")
        import shorts  # needs Pillow + OpenCV
        self.shorts = shorts
        self.args = args
        self.jobs: queue.Queue = queue.Queue()
        self.busy: set[str] = set()  # clips queued or being turned into shorts
        self.faces: dict[str, dict] = {}  # last facecam found per streamer
        self.default_cam, self.cams = None, {}
        for spec in args.facecam or []:
            if "=" in spec:
                ch, cam = spec.split("=", 1)
                self.cams[channel_name(ch)] = shorts.parse_facecam(cam)
            else:
                self.default_cam = shorts.parse_facecam(spec)
        if not args.no_captions and not (shorts.WHISPER and shorts.MODEL.exists()):
            print("note: captions disabled (whisper-cli or speech model missing, see README)")
            args.no_captions = True

    def submit(self, clip: Path, channel: str, layout: str | None = None) -> None:
        """Make a short from an existing clip (e.g. 'Make short' on the page)."""
        if str(clip) in self.busy:
            return
        if layout is None:  # use what the clip was saved with
            try:
                layout = json.loads(Path(clip).with_suffix(".json").read_text()).get("layout")
            except (OSError, ValueError):
                layout = None
        self.busy.add(str(clip))
        self.jobs.put(("short", clip, channel, layout))

    def finish(self, wide: Path, base: Path, info: dict, rel: dict, on_ready) -> None:
        """Trim a freshly saved wide clip at natural points, save it, then make its short."""
        self.busy.add(str(base.with_suffix(".mp4")))
        self.jobs.put(("finish", wide, base, info, rel, on_ready))

    def duo(self, a: Path, a_off: float, a_info: dict, b: Path, b_off: float, b_info: dict, dur: float) -> None:
        self.jobs.put(("duo", a, a_off, a_info, b, b_off, b_info, dur))

    def _duo(self, a, a_off, a_info, b, b_off, b_info, dur) -> None:
        name = lambda i: (i.get("credit") or i.get("channel") or "?").split("/")[-1].lstrip("@")
        na, nb = name(a_info), name(b_info)
        folder = Path(self.args.out).expanduser() / "duos"
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out = folder / f"duo_{a_info['channel']}_{b_info['channel']}_{stamp}.mp4"
        self.shorts.make_duo(a, a_off, b, b_off, min(dur, 60), (na, nb), out)
        short = folder / "shorts" / f"{out.stem}_short.mp4"
        short.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(out, short)  # same file shown as the short, no extra space
        except OSError:
            shutil.copy2(out, short)
        score = max(int(a_info.get("score") or 0), int(b_info.get("score") or 0))
        info = {"channel": "duos", "platform": a_info.get("platform"), "kind": "duo",
                "credit": f"{a_info.get('credit')} + {b_info.get('credit')}", "score": min(100, score + 10),
                "spike_start": max(a_info.get("spike_start", 0), b_info.get("spike_start", 0)),
                "clip_seconds": round(min(dur, 60)), "reasons": ["two POVs"],
                "top_tokens": [], "sample_messages": [], "file": out.name}
        from metadata import for_short
        meta = for_short(dict(info, display=f"{na} & {nb}"), (a_info.get("said") or "") + " " + (b_info.get("said") or ""))
        meta["title"] = f"{na} & {nb}: same moment, two POVs 😳"
        meta["description"] = meta["description"].replace(meta["description"].split("\n")[0], meta["title"], 1)
        info["meta"] = meta
        out.with_suffix(".json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
        log("duos", f"DUO SHORT SAVED -> {out} ({na} + {nb})")

    def _layout_for(self, channel: str, job_layout: str | None):
        cam = self.cams.get(channel, self.default_cam)
        layout = self.args.layout
        if job_layout == "irl":
            layout = "irl"  # IRL stream: stream fills the middle 80%
        if layout == "split" and cam is None:
            layout = "auto"
        return layout, cam

    def _make_short(self, clip: Path, channel: str, job_layout, out=None, credit=None,
                    identity=None, **kw) -> None:
        layout, cam = self._layout_for(channel, job_layout)
        kw.setdefault("effects", not self.args.no_effects)
        if "hook" not in kw:  # 'Make short' on a saved clip: use its saved hook text
            try:
                kw["hook"] = (json.loads(Path(clip).with_suffix(".json").read_text()).get("meta") or {}).get("hook")
            except (OSError, ValueError):
                pass
        if identity is None:  # use the streamer's remembered face, if we know it
            try:
                import identity as ident
                f = ident.identity_file(Path(self.args.out).expanduser(), channel)
                if f.exists():
                    identity = json.loads(f.read_text())["face"]
            except (OSError, ValueError, KeyError, ImportError):
                identity = None
        if identity is not None:
            import numpy as np
            identity = np.asarray(identity, np.float32)
        if credit is None:  # e.g. 'Make short' on a saved clip: use what it was saved with
            try:
                credit = json.loads(Path(clip).with_suffix(".json").read_text()).get("credit")
            except (OSError, ValueError):
                credit = None
        self.shorts.make_short(
            clip, out=out, layout=layout, facecam=cam, captions=not self.args.no_captions,
            do_censor=not self.args.no_censor, credit=credit or f"twitch.tv/{channel}",
            face_hint=self.faces.get(channel), identity=identity,
            log=lambda m: log(channel, m.replace("short saved", "SHORT SAVED")), **kw)
        found = getattr(self.shorts.make_short, "last_face", None)
        old = self.faces.get(channel)
        if found and (found.get("cam") or not (old and old.get("cam"))):
            self.faces[channel] = found  # prefer remembering one with a facecam box

    def _finish(self, wide: Path, base: Path, info: dict, rel: dict, on_ready) -> None:
        import tempfile
        from smartcut import choose_window
        channel = info["channel"]
        final = base.with_suffix(".mp4")
        try:
            _, _, duration = self.shorts.probe(wide)
            words = []
            if not self.args.no_captions:
                with tempfile.TemporaryDirectory(prefix="clip_") as tmp:
                    try:
                        words = self.shorts.transcribe(wide, Path(tmp))
                    except Exception as e:
                        log(channel, f"couldn't transcribe, using a fixed window: {e}")
            start, end = choose_window(words, rel["spike_start"], rel["spike_end"], rel["peak"],
                                       duration, pre=self.args.pre, max_len=self.args.max_len)
            try:  # what the streamer said around the moment ("NO WAY", "bro what"...)
                from reactions import speech_hype
                near = [w for w in words if rel["spike_start"] - 15 <= w.start <= rel["spike_end"] + 5]
                said = speech_hype(near)
                if said >= 0.3:
                    info["score"] = min(100, int(info.get("score", 0)) + int(round(10 * said)))
                    info["reasons"] = sorted(set(info.get("reasons", [])) | {"hype words"})
            except Exception:
                pass
            r = subprocess.run(
                [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.2f}",
                 "-i", str(wide), "-t", f"{end - start:.2f}", "-c", "copy",
                 "-avoid_negative_ts", "make_zero", "-movflags", "+faststart", str(final)],
                capture_output=True, text=True)
            if r.returncode != 0 or not final.exists():
                log(channel, f"couldn't save the trimmed clip: {r.stderr.strip()[-200:]}")
                return
            t0 = rel["clip_start_wall"]
            info.update(file=final.name, clip_seconds=round(end - start),
                        clip_start_wall=t0 + start, clip_end_wall=t0 + end,
                        window={"start": start, "end": end, "wide_seconds": round(duration)})
            said = " ".join(w.text for w in words if start - 0.5 <= w.start <= end)
            info["said"] = said[:600]
            try:  # upload-ready title, description, hashtags, keywords and the short's hook text
                from metadata import for_short
                info["meta"] = for_short(info, said)
            except Exception as e:
                log(channel, f"couldn't write the title/description ({e})")
            import housekeeping
            dup = housekeeping.is_duplicate(final.parent, info, t0 + start, t0 + end)
            if dup:  # this moment is already clipped (e.g. Clip now + auto, or a re-scan)
                log(channel, f"skipped a duplicate: that moment is already in {dup}")
                final.unlink(missing_ok=True)
                return
            final.with_suffix(".json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
            with open(final.parent.parent / "clips_index.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(info, ensure_ascii=False) + "\n")
            on_ready(final, info)
            try:  # big moment? tell the user (Mac notification + optional phone alert)
                import housekeeping
                st = housekeeping.load(Path(self.args.out).expanduser())
                if int(info.get("score") or 0) >= int(st.get("notify_score", 80)) and info.get("kind") != "test":
                    housekeeping.notify(st, f"🔥 {info.get('score')} · {channel}",
                                        (info.get("meta") or {}).get("title") or "New clip", log)
            except Exception:
                pass
            if self.args.shorts:
                meta = info.get("meta") or {}
                loud = "scream" in " ".join(info.get("reasons") or []) or (info.get("audio_boost") or 0) >= 10
                self._make_short(wide, channel, info.get("layout"), credit=info.get("credit"),
                                 identity=info.get("streamer_face"),
                                 out=final.parent / "shorts" / f"{final.stem}_short.mp4",
                                 start=start, length=end - start, words=words,
                                 hook=meta.get("hook"), peak=rel["peak"] - start,
                                 shake=loud or meta.get("category") == "crazy")
        finally:
            wide.unlink(missing_ok=True)

    def run(self) -> None:
        while True:
            job = self.jobs.get()
            kind, payload = job[0], job[1:]
            if kind == "duo":
                key, channel = "duo", "duos"
            else:
                key = str(payload[0]) if kind == "short" else str(payload[1].with_suffix(".mp4"))
                channel = payload[1] if kind == "short" else payload[2]["channel"]
            try:
                if kind == "short":
                    clip, channel, job_layout = payload
                    self._make_short(clip, channel, job_layout)
                elif kind == "duo":
                    self._duo(*payload)
                else:
                    self._finish(*payload)
            except Exception as e:
                log(channel, f"couldn't finish clip: {e}")
            finally:
                self.busy.discard(key)
                self.jobs.task_done()


SHORTS: ShortsMaker | None = None
RECENT_CLIPS: deque = deque(maxlen=60)  # (channel, wall start, wall end, path, info) for duo shorts
_recent_lock = threading.Lock()
SCANS = None  # vodscan.ScanManager: past videos / VODs, set up in main()
LONG = None  # longform.LongManager: 10-15 minute YouTube videos, set up in main()
VIEWER_API = None  # twitch_api.TwitchAPI, set up in main()
_shorts_lock = threading.Lock()


def get_shorts_maker(args) -> ShortsMaker:
    """The shorts worker, started on first use (e.g. a 'Make short' click)."""
    global SHORTS
    with _shorts_lock:
        if SHORTS is None:
            SHORTS = ShortsMaker(args)
            SHORTS.start()
        return SHORTS


class Manager:
    """The set of channels being watched. Channels can be added and removed while
    running (from the dashboard), and the list is remembered for next time."""

    MAX_CHANNELS = 6
    parse = staticmethod(parse_channel)

    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.clippers: dict[str, ChannelClipper] = {}
        self.saved_file = Path(args.out).expanduser() / "channels.json"
        self.clips_saved_removed = 0

    def all(self) -> list[ChannelClipper]:
        with self.lock:
            return list(self.clippers.values())

    def get(self, name: str) -> ChannelClipper | None:
        with self.lock:
            return self.clippers.get(name)

    def add(self, name: str, sensitivity: float | None = None, mode: str | None = None,
            reactions: bool | None = None, source: dict | None = None) -> tuple[bool, str]:
        with self.lock:
            if name in self.clippers:
                return False, f"Already watching {name}"
            if len(self.clippers) >= self.MAX_CHANNELS:
                return False, f"Max {self.MAX_CHANNELS} streamers at once"
            c = ChannelClipper(name, self.args, threading.Event(), source=source)
            if sensitivity:
                c.detector.threshold = sensitivity
            if mode in MODES:
                c.mode = mode
            if reactions is not None:
                c.detector.reactions_on = bool(reactions)
            self.clippers[name] = c
        c.start()
        log(name, "watching")
        self.save()
        return True, f"Watching {name}"

    def remove(self, name: str) -> bool:
        with self.lock:
            c = self.clippers.pop(name, None)
        if not c:
            return False
        self.clips_saved_removed += c.clips_saved
        threading.Thread(target=c.shutdown, daemon=True).start()
        shutil.rmtree(Path(self.args.buffer).expanduser() / name, ignore_errors=True)
        log(name, "stopped watching")
        self.save()
        return True

    def save(self) -> None:
        data = [{"channel": c.channel, "sensitivity": c.detector.threshold, "mode": c.mode,
                 "reactions": c.detector.reactions_on,
                 "source": {k: c.source.get(k) for k in ("platform", "record_url", "credit")}}
                for c in self.all()]
        try:
            self.saved_file.parent.mkdir(parents=True, exist_ok=True)
            self.saved_file.write_text(json.dumps(data, indent=2))
        except OSError:
            pass

    def load_saved(self) -> list[dict]:
        try:
            return [d for d in json.loads(self.saved_file.read_text())
                    if re.fullmatch(r"[a-z0-9_]{1,25}", d.get("channel", ""))]
        except (OSError, ValueError, KeyError, TypeError):
            return []

    def shutdown(self) -> None:
        for c in self.all():
            c.shutdown()

    def total_clips(self) -> int:
        return self.clips_saved_removed + sum(c.clips_saved for c in self.all())


LEGACY_URL_FILE = HERE / ".dashboard_url"


def url_file(args) -> Path:
    """Where the running clipper writes its page address (the Desktop launcher reads it)."""
    return Path(args.out).expanduser() / ".dashboard_url"


def start_dashboard(manager: Manager, args, stop: threading.Event) -> None:
    from dashboard import Dashboard
    Path(args.out).expanduser().mkdir(parents=True, exist_ok=True)
    for port in range(args.port, args.port + 10):  # skip ports already in use
        args.port = port
        try:
            dash = Dashboard(manager, args, lambda: get_shorts_maker(args), lambda: SHORTS,
                             stop=stop.set)
            dash.viewer_api = lambda: VIEWER_API
            dash.scans = lambda: SCANS
            dash.long = lambda: LONG
            url = dash.start()
            break
        except OSError:
            continue
    else:
        print("dashboard: no free port found, running without it", flush=True)
        return
    print(f"Dashboard: {url}  (open this in your browser)", flush=True)
    try:
        url_file(args).write_text(url)
    except OSError:
        pass
    if not args.no_open:
        import webbrowser
        webbrowser.open(url)


def hand_off_to_running(args) -> bool:
    """If a clipper is already running (e.g. started from the Desktop icon), don't start a
    second one: give it any streamers from the command line, open its page, and exit."""
    import urllib.request
    candidates = [url_file(args)]
    if Path(args.out).expanduser().resolve() == (HERE / "clips").resolve():
        candidates.append(LEGACY_URL_FILE)  # written by older versions
    for f in candidates:
        try:
            url = f.read_text().strip()
            urllib.request.urlopen(url + "/api/state", timeout=2).read()
            break
        except (OSError, ValueError):
            continue
    else:
        return False
    for c in args.channels:
        req = urllib.request.Request(url + "/api/channels/add", method="POST",
                                     data=json.dumps({"url": c}).encode(),
                                     headers={"X-Clipper": "1", "Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=5)
        except OSError:
            pass  # e.g. already watching
    print(f"The clipper is already running - opening {url}", flush=True)
    if not args.no_open:
        import webbrowser
        webbrowser.open(url)
    return True


def keep_awake() -> None:
    """Stop the computer from sleeping while the clipper runs (lid-closed laptops still sleep)."""
    platformkit.keep_awake()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Auto-clip Twitch streams when chat goes crazy. Run with no channels "
                    "and paste Twitch links into the dashboard instead.")
    p.add_argument("channels", nargs="*", help="Twitch channel names or links (optional)")
    p.add_argument("--sensitivity", type=float, default=3.0,
                   help="clip when chat is this many times busier than normal (lower = more clips)")
    p.add_argument("--pre", type=float, default=22,
                   help="aim to start clips about this many seconds before chat exploded "
                        "(moved to the nearest pause in speech)")
    p.add_argument("--context", type=float, default=55,
                   help="how far back to look for a good starting point")
    p.add_argument("--max-len", type=float, default=75, help="longest clip in seconds")
    p.add_argument("--cooldown", type=float, default=5, help="seconds between clips")
    p.add_argument("--min-messages", type=int, default=8, help="min messages in 5s to count as a spike")
    p.add_argument("--min-chatters", type=int, default=5, help="min different chatters in 5s")
    p.add_argument("--quality", default="1080p60,1080p,936p60,720p60,720p,best",
                   help="streamlink quality preference list")
    p.add_argument("--out", default=str(HERE / "clips"), help="where clips are saved")
    p.add_argument("--buffer", default=str(HERE / "buffer"), help="temporary rolling buffer folder")
    p.add_argument("--buffer-seconds", type=int, default=240, help="how much video to keep in buffer")
    p.add_argument("--min-free-gb", type=float, default=1.0, help="stop saving clips below this free disk")
    p.add_argument("--offline-retry", type=int, default=60, help="seconds between checks when offline")
    p.add_argument("--status-every", type=int, default=60, help="seconds between status lines")
    p.add_argument("--test-clip", action="store_true",
                   help="save one clip ~40s after recording starts, to check everything works")
    p.add_argument("--fresh", action="store_true", help="don't resume the streamers watched last time")
    p.add_argument("--top", type=int, default=10, help="past videos/VODs: how many clips to make")
    d = p.add_argument_group("dashboard")
    d.add_argument("--port", type=int, default=8765, help="dashboard port (http://127.0.0.1:PORT)")
    d.add_argument("--no-dashboard", action="store_true", help="don't run the web dashboard")
    d.add_argument("--no-open", action="store_true", help="don't open the dashboard in the browser")
    g = p.add_argument_group("vertical shorts (on by default)")
    g.add_argument("--no-shorts", action="store_true", help="only save horizontal clips")
    g.add_argument("--shorts", action="store_true", help=argparse.SUPPRESS)  # old flag, now default
    g.add_argument("--layout", choices=["auto", "blur", "split"], default="auto",
                   help="auto = finds the streamer's face: facecam top 30%%, gameplay 70%%, or IRL "
                        "stream for IRL (default); blur = gameplay in the middle; "
                        "split = facecam position you give")
    g.add_argument("--mode", choices=MODES, default="auto",
                   help="auto = IRL layout when the Twitch category is IRL/Just Chatting; "
                        "gaming = always facecam + gameplay; irl = stream fills the middle 80%%")
    g.add_argument("--facecam", action="append", metavar="[CHANNEL=]POSITION",
                   help="facecam for split layout: top-left, top-right, bottom-left, bottom-right "
                        "or x,y,w,h fractions. Per channel: --facecam tfue=0,0.2,0.23,0.23")
    g.add_argument("--no-captions", action="store_true", help="shorts without captions")
    g.add_argument("--no-censor", action="store_true", help="don't mask swear words in captions")
    p.add_argument("--no-effects", action="store_true",
                   help="plain shorts: no hook text, zoom, shake or sound effects")
    p.add_argument("--no-reactions", action="store_true",
                   help="don't watch/listen for the streamer laughing, screaming or looking shocked")
    p.add_argument("--chat-only", action="store_true",
                   help="don't record video, just show detected moments (for tuning sensitivity)")
    p.add_argument("--log", help="write messages to this file instead of the screen (used by the Windows launcher)")
    args = p.parse_args()
    if args.log or sys.stdout is None:  # Windows windowless start: no screen to print to
        f = open(args.log or (HERE / "clipper.log"), "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stderr = f
    args.shorts = not args.no_shorts and not args.chat_only

    if not args.chat_only and not shutil.which(FFMPEG):
        raise SystemExit("ffmpeg not found. Install it with: brew install ffmpeg")
    if args.no_dashboard and not args.channels:
        raise SystemExit("give at least one channel when running without the dashboard")
    if not args.no_dashboard and hand_off_to_running(args):
        return
    if args.shorts:
        get_shorts_maker(args)

    global VIEWER_API
    from twitch_api import TwitchAPI
    VIEWER_API = TwitchAPI(Path(args.out).expanduser() / "settings.json")
    for leftover in [*Path(args.out).expanduser().glob("*/.work"), Path(args.out).expanduser() / ".scan"]:
        shutil.rmtree(leftover, ignore_errors=True)  # half-finished clips / scan audio from a crash
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    keep_awake()
    manager = Manager(args)
    global SCANS
    if not args.chat_only:
        from vodscan import ScanManager
        SCANS = ScanManager(args, Path(args.out).expanduser(), lambda: get_shorts_maker(args), log)
        global LONG
        from longform import LongManager
        LONG = LongManager(args, Path(args.out).expanduser(), log)
    if not args.fresh and not args.channels:
        for d in manager.load_saved():
            manager.add(d["channel"], d.get("sensitivity"), d.get("mode"), d.get("reactions"),
                        source=d.get("source"))
    for link in dict.fromkeys(args.channels):
        from sources import resolve, SourceError
        try:
            src = resolve(link)
        except SourceError as e:
            print(f"can't use '{link}': {e}", flush=True)
            continue
        if src["kind"] == "live":
            manager.add(src["name"], mode=args.mode, source=src)
        elif SCANS is not None:
            SCANS.start(src, top=args.top)
    print(f"Clips go to {Path(args.out).expanduser()}. Ctrl+C to stop.", flush=True)
    if not args.no_dashboard:
        start_dashboard(manager, args, stop)
    def tidy():  # every 5 minutes: make room if the disk is almost full
        import housekeeping
        out = Path(args.out).expanduser()
        while not stop.wait(300):
            try:
                if housekeeping.load(out).get("auto_cleanup", True):
                    housekeeping.cleanup(out, log)
            except Exception as e:
                log("cleanup", f"error: {e}")
    threading.Thread(target=tidy, daemon=True, name="tidy").start()
    try:
        while not stop.is_set():
            stop.wait(1)
    except KeyboardInterrupt:
        print("\nStopping...", flush=True)
    finally:
        stop.set()
        manager.shutdown()
        if SHORTS is not None and SHORTS.jobs.unfinished_tasks:
            print(f"Finishing {SHORTS.jobs.unfinished_tasks} short(s)... "
                  f"(Ctrl+C again to skip)", flush=True)
            try:
                SHORTS.jobs.join()
            except KeyboardInterrupt:
                print("Skipped. Make them later with: python shorts.py <clip.mp4>", flush=True)
        buf = Path(args.buffer).expanduser()
        if buf.exists() and buf.resolve() != Path(args.out).expanduser().resolve():
            shutil.rmtree(buf, ignore_errors=True)
        if not args.no_dashboard:
            url_file(args).unlink(missing_ok=True)
        scanned = sum(j.made for j in SCANS.jobs) if SCANS is not None else 0
        print(f"Done. {manager.total_clips() + scanned} clip(s) saved this session.", flush=True)


if __name__ == "__main__":
    main()
