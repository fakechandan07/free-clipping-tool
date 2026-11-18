"""Where clips come from: Twitch / YouTube, live or recorded.

resolve(link) works out what a pasted link is, and this module also reads chat
for each kind:

- Twitch live: IRC (in clipper.py)
- YouTube live: yt-dlp follows the live chat as it happens
- Twitch past broadcast: chat replay from Twitch's public web API
- YouTube past stream: chat replay via yt-dlp
- YouTube video: no chat, but timestamped comments ("2:34 LMAOOO")
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
import platformkit  # noqa: E402
YTDLP = platformkit.venv_tool("yt-dlp")

TWITCH_LOGIN_RE = re.compile(r"^[a-z0-9_]{2,25}$")
TWITCH_VOD_RE = re.compile(r"twitch\.tv/(?:[^/]+/)?videos?/(\d+)", re.I)
TWITCH_CHANNEL_RE = re.compile(r"twitch\.tv/([A-Za-z0-9_]{2,25})(?:[/?#]|$)", re.I)
YOUTUBE_RE = re.compile(r"(youtube\.com|youtu\.be)/", re.I)
KICK_RE = re.compile(r"kick\.com/([A-Za-z0-9_\-]{2,40})(?:/(videos?)/([A-Za-z0-9\-]+))?", re.I)
KICK_VIDEO_RE = re.compile(r"kick\.com/video/([A-Za-z0-9\-]+)", re.I)
BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/120 Safari/537.36")
TWITCH_GQL_CLIENT = "kimne78kx3ncx6brgo4mv6wki5h1ko"  # Twitch's public web-player client id
NOT_CHANNELS = {"videos", "directory", "settings", "downloads", "p", "search", "subscriptions"}


class SourceError(Exception):
    """A link we can't use, with a message for the page."""


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9_]+", "_", (text or "").lower()).strip("_")
    return s[:25] or "stream"


def ytdlp_info(url: str, timeout: int = 90) -> dict:
    r = subprocess.run([YTDLP, "--skip-download", "--dump-single-json", "--no-playlist",
                        "--no-warnings", url], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0 or not r.stdout.strip():
        err = (r.stderr.strip().splitlines() or ["unknown error"])[-1]
        raise SourceError(_friendly(err))
    return json.loads(r.stdout)


def _friendly(err: str) -> str:
    e = err.lower()
    if "not a bot" in e or "sign in to confirm" in e:
        return "YouTube is asking to confirm you're not a bot. Try again later, or update yt-dlp."
    if "private" in e:
        return "That video is private."
    if "members-only" in e or "join this channel" in e:
        return "That video is members-only."
    if ("not currently live" in e or "is not live" in e or "offline" in e or "will begin" in e
            or "premieres in" in e or "upcoming" in e):
        return "not live"
    if "unavailable" in e or "does not exist" in e or "404" in e:
        return "That video doesn't exist or isn't available anymore."
    if "subscriber" in e:
        return "That VOD is subscriber-only."
    return err.replace("ERROR: ", "")[:200]


def resolve(link: str) -> dict:
    """Work out what a pasted link is.

    Returns {"kind": "live" | "vod", "platform": "twitch" | "youtube", "name": folder/display
    name, "credit": text on shorts, ...}. Live sources also have "record_url" (what streamlink
    records); recorded ones have "url", "duration", "title", "chat": twitch/youtube/None.
    """
    link = link.strip()
    if not link:
        raise SourceError("Paste a Twitch or YouTube link")
    if not re.match(r"^https?://", link) and ("." in link.split("/")[0] or "/" in link):
        link = "https://" + link

    # --- Kick (live channel or past broadcast)
    mv = KICK_VIDEO_RE.search(link)
    mk = KICK_RE.search(link)
    if mv or (mk and mk.group(2)):
        info = ytdlp_info(link)
        slug = (info.get("uploader_id") or info.get("channel") or info.get("uploader") or "kick").lower()
        return {"kind": "vod", "platform": "kick", "name": _slug(slug), "credit": f"kick.com/{slug}",
                "url": info.get("webpage_url") or link, "video_id": info.get("id"), "title": info.get("title"),
                "duration": float(info.get("duration") or 0), "chat": None,
                "category": (info.get("categories") or [None])[0], "uploader": info.get("uploader")}
    if mk:
        slug = mk.group(1).lower()
        return {"kind": "live", "platform": "kick", "name": _slug(slug), "credit": f"kick.com/{slug}",
                "record_url": f"https://kick.com/{slug}", "kick_slug": slug, "title": None}

    # --- Twitch past broadcast
    m = TWITCH_VOD_RE.search(link)
    if m:
        vid = m.group(1)
        info = ytdlp_info(f"https://www.twitch.tv/videos/{vid}")
        login = (info.get("uploader_id") or info.get("uploader") or "twitch").lower()
        return {"kind": "vod", "platform": "twitch", "name": _slug(login), "credit": f"twitch.tv/{login}",
                "url": f"https://www.twitch.tv/videos/{vid}", "video_id": vid, "title": info.get("title"),
                "duration": float(info.get("duration") or 0), "chat": "twitch",
                "category": None, "uploader": info.get("uploader")}

    # --- YouTube
    if YOUTUBE_RE.search(link):
        ch = re.search(r"youtube\.com/(@[A-Za-z0-9_.\-]+|channel/[A-Za-z0-9_\-]+|c/[^/?#]+)/?(?:featured|streams|videos)?/?(?:[?#].*)?$", link)
        if ch:  # a channel page, not a video: watch that channel live
            link = f"https://www.youtube.com/{ch.group(1)}/live"
        try:
            info = ytdlp_info(link)
        except SourceError as e:
            m2 = re.search(r"youtube\.com/(@[A-Za-z0-9_.\-]+|channel/[A-Za-z0-9_\-]+|c/[^/?#]+)", link)
            if str(e) == "not live" and m2:  # channel /live link while offline: wait for them
                handle = m2.group(1).split("/")[-1]
                return {"kind": "live", "platform": "youtube", "name": _slug(handle.lstrip("@")),
                        "credit": f"youtube.com/{handle}", "record_url": f"https://www.youtube.com/{m2.group(1)}/live",
                        "title": None}
            raise
        handle = info.get("uploader_id") or info.get("channel_id") or info.get("channel") or "youtube"
        name = _slug((info.get("uploader_id") or "").lstrip("@") or info.get("channel") or handle)
        credit = f"youtube.com/{handle}" if str(handle).startswith("@") else (info.get("channel") or "YouTube")
        status = info.get("live_status")
        if status in ("is_live", "is_upcoming"):
            m2 = re.search(r"youtube\.com/(@[A-Za-z0-9_.\-]+)", link)
            record = (f"https://www.youtube.com/{m2.group(1)}/live" if m2 and "/live" in link
                      else f"https://www.youtube.com/watch?v={info['id']}")
            return {"kind": "live", "platform": "youtube", "name": name, "credit": credit,
                    "record_url": record, "title": info.get("title"), "video_id": info["id"]}
        if info.get("_type") == "playlist" or not info.get("duration"):
            raise SourceError("Paste a link to one video or stream (not a channel or playlist)")
        has_chat = "live_chat" in (info.get("subtitles") or {})
        return {"kind": "vod", "platform": "youtube", "name": name, "credit": credit,
                "url": f"https://www.youtube.com/watch?v={info['id']}", "video_id": info["id"],
                "title": info.get("title"), "duration": float(info["duration"]),
                "chat": "youtube" if has_chat else None, "was_live": status == "was_live",
                "category": (info.get("categories") or [None])[0], "uploader": info.get("uploader"),
                "heatmap": info.get("heatmap") or [], "comment_count": info.get("comment_count")}

    # --- Twitch live (channel link or bare name)
    m = TWITCH_CHANNEL_RE.search(link)
    login = (m.group(1) if m else link).strip().lstrip("#@").lower()
    if login in NOT_CHANNELS or not TWITCH_LOGIN_RE.match(login):
        raise SourceError("That doesn't look like a Twitch, YouTube or Kick link")
    return {"kind": "live", "platform": "twitch", "name": login, "credit": f"twitch.tv/{login}",
            "record_url": f"https://www.twitch.tv/{login}"}


# --------------------------------------------------------------------------
# YouTube chat (live and replay) - same JSON lines either way
# --------------------------------------------------------------------------
def _yt_text(runs: list) -> str:
    out = []
    for r in runs or []:
        if "text" in r:
            out.append(r["text"])
        elif "emoji" in r:
            e = r["emoji"]
            sc = (e.get("shortcuts") or [None])[0]
            out.append(e.get("emojiId") if not e.get("isCustomEmoji") else (sc or "").strip(":"))
    return "".join(out)


def _chat_file(folder: Path) -> Path | None:
    """yt-dlp's main chat file (not its temporary -FragN pieces)."""
    for name in ("chat.live_chat.json.part", "chat.live_chat.json"):
        f = folder / name
        if f.exists():
            return f
    return None


def parse_yt_chat_line(line: str, with_time: bool = False):
    """One line of yt-dlp's live_chat.json -> [(offset_seconds | None, author, text), ...]
    (with_time: also the unix time it was sent, as a 4th item)"""
    try:
        d = json.loads(line)
    except ValueError:
        return []
    rep = d.get("replayChatItemAction") or {}
    offset = rep.get("videoOffsetTimeMsec")
    offset = int(offset) / 1000 if offset is not None else None
    out = []
    for a in rep.get("actions", []):
        item = (a.get("addChatItemAction") or {}).get("item") or {}
        r = item.get("liveChatTextMessageRenderer") or item.get("liveChatPaidMessageRenderer")
        if not r:
            continue
        author = (r.get("authorName") or {}).get("simpleText", "?")
        text = _yt_text((r.get("message") or {}).get("runs"))
        if text:
            sent = int(r.get("timestampUsec") or 0) / 1e6
            out.append((offset, author.lstrip("@"), text, sent) if with_time else
                       (offset, author.lstrip("@"), text))
    return out


class YouTubeLiveChat(threading.Thread):
    """Follows a YouTube live chat with yt-dlp and passes each message on, like the
    Twitch chat listener."""

    def __init__(self, url: str, on_message, stop: threading.Event, log=print):
        super().__init__(daemon=True, name="yt-chat")
        self.url, self.on_message, self.stop_event, self.log = url, on_message, stop, log
        self.connected = False

    def run(self) -> None:
        while not self.stop_event.is_set():
            with tempfile.TemporaryDirectory(prefix="ytchat_") as tmp:
                base = Path(tmp) / "chat"
                proc = subprocess.Popen([YTDLP, "--skip-download", "--write-subs", "--sub-langs", "live_chat",
                                         "--no-warnings", "--quiet", "-o", str(base), self.url],
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                pos, buf, first_read = 0, "", True
                try:
                    while not self.stop_event.is_set() and proc.poll() is None:
                        part = _chat_file(Path(tmp))
                        if part is not None and part.stat().st_size > pos:
                            with open(part, encoding="utf-8", errors="replace") as f:
                                f.seek(pos)
                                buf += f.read()
                                pos = f.tell()
                            *lines, buf = buf.split("\n")
                            now = time.time()
                            for line in lines:
                                for _, author, text, sent in parse_yt_chat_line(line, with_time=True):
                                    # The first read includes recent history YouTube sends on
                                    # joining; feeding it would look like a spike. After that,
                                    # count messages when they arrive (YouTube delivers chat
                                    # 10-20 s late, about as late as the recorded video).
                                    if first_read and sent and now - sent > 8:
                                        continue
                                    self.on_message(now, author, text)
                            first_read = False
                            if lines and not self.connected:
                                self.connected = True
                        self.stop_event.wait(0.5)
                finally:
                    if proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            proc.kill()
            self.connected = False
            if not self.stop_event.is_set():
                self.stop_event.wait(30)  # stream not live (yet) or chat ended: try again later


class KickChat(threading.Thread):
    """Kick live chat (Kick uses Pusher websockets; read-only, no login needed)."""
    PUSHER = "wss://ws-us2.pusher.com/app/32cbd69e4b950bf97679?protocol=7&client=js&version=8.4.0-rc2&flash=false"

    def __init__(self, slug: str, on_message, stop: threading.Event, log=print):
        super().__init__(daemon=True, name=f"kick-{slug}")
        self.slug, self.on_message, self.stop_event, self.log = slug, on_message, stop, log
        self.connected = False
        self.room = None

    def _room(self):
        req = urllib.request.Request(f"https://kick.com/api/v2/channels/{self.slug}",
                                     headers={"User-Agent": BROWSER_UA, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=20, context=_ssl()) as r:
            return json.loads(r.read())["chatroom"]["id"]

    def run(self) -> None:
        import websocket  # websocket-client
        backoff = 5
        while not self.stop_event.is_set():
            ws = None
            try:
                if self.room is None:
                    self.room = self._room()
                ws = websocket.create_connection(self.PUSHER, timeout=30, sslopt={"ca_certs": _certifi_path()})
                ws.send(json.dumps({"event": "pusher:subscribe",
                                    "data": {"auth": "", "channel": f"chatrooms.{self.room}.v2"}}))
                ws.settimeout(1.0)
                last = time.time()
                while not self.stop_event.is_set():
                    try:
                        raw = ws.recv()
                    except websocket.WebSocketTimeoutException:
                        if time.time() - last > 120:
                            ws.send(json.dumps({"event": "pusher:ping", "data": {}}))
                            last = time.time()
                        continue
                    last = time.time()
                    msg = json.loads(raw)
                    ev = msg.get("event", "")
                    if ev == "pusher:ping":
                        ws.send(json.dumps({"event": "pusher:pong", "data": {}}))
                    elif ev == "pusher_internal:subscription_succeeded":
                        self.connected = True
                    elif ev == "pusher:error":
                        raise RuntimeError(str((msg.get("data") or {}).get("message") or "chat server error"))
                    elif ev.endswith("ChatMessageEvent"):
                        d = json.loads(msg.get("data") or "{}")
                        text = d.get("content") or ""
                        user = ((d.get("sender") or {}).get("username")) or "?"
                        text = re.sub(r"\[emote:\d+:([^\]]+)\]", r"\1", text)  # [emote:123:KEKW] -> KEKW
                        if text:
                            self.on_message(time.time(), user, text)
                backoff = 5
            except Exception as e:
                self.connected = False
                if self.stop_event.is_set():
                    break
                if "cluster" in str(e) or "App key" in str(e):
                    self.log("Kick chat isn't reachable right now (Kick changed its chat server); "
                             "clipping this stream from the streamer's reactions and loudness only")
                    return
                self.log(f"kick chat disconnected ({str(e)[:80]}); reconnecting in {backoff}s")
                self.stop_event.wait(backoff)
                backoff = min(backoff * 2, 120)
            finally:
                if ws is not None:
                    try:
                        ws.close()
                    except Exception:
                        pass
        self.connected = False


def _certifi_path():
    try:
        import certifi
        return certifi.where()
    except ImportError:
        return None


def youtube_chat_replay(url: str, duration: float, progress=None, stop: threading.Event | None = None):
    """All chat messages of a past YouTube stream: [(offset_s, author, text), ...]."""
    msgs = []
    with tempfile.TemporaryDirectory(prefix="ytreplay_") as tmp:
        base = Path(tmp) / "chat"
        proc = subprocess.Popen([YTDLP, "--skip-download", "--write-subs", "--sub-langs", "live_chat",
                                 "--no-warnings", "--quiet", "-o", str(base), url],
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        pos, buf, last = 0, "", 0.0
        while True:
            done = proc.poll() is not None
            part = _chat_file(Path(tmp))
            if part is not None and part.stat().st_size > pos:
                with open(part, encoding="utf-8", errors="replace") as f:
                    f.seek(pos)
                    buf += f.read()
                    pos = f.tell()
                *lines, buf = buf.split("\n")
                for line in lines:
                    for off, author, text in parse_yt_chat_line(line):
                        if off is not None:
                            msgs.append((off, author, text))
                            last = off
                if progress and duration:
                    progress(min(1.0, last / duration))
            if done:
                break
            if stop is not None and stop.is_set():
                proc.terminate()
                break
            time.sleep(0.5)
        if buf.strip():
            for off, author, text in parse_yt_chat_line(buf):
                if off is not None:
                    msgs.append((off, author, text))
    msgs.sort(key=lambda m: m[0])
    return msgs


def youtube_comment_times(url: str, duration: float, limit: int = 400) -> list[tuple[float, int]]:
    """Timestamps people wrote in the comments ("2:34 lmaooo"): [(seconds, likes), ...]."""
    with tempfile.TemporaryDirectory(prefix="ytcomments_") as tmp:
        r = subprocess.run([YTDLP, "--skip-download", "--write-comments", "--no-warnings", "--quiet",
                            "--extractor-args", f"youtube:max_comments={limit},all,0;comment_sort=top",
                            "-o", str(Path(tmp) / "v"), url], capture_output=True, text=True, timeout=300)
        files = list(Path(tmp).glob("*.info.json"))
        if not files:
            return []
        comments = json.loads(files[0].read_text()).get("comments") or []
    out = []
    for c in comments:
        for m in re.finditer(r"\b(?:(\d{1,2}):)?(\d{1,2}):(\d{2})\b", c.get("text") or ""):
            h, mi, s = int(m.group(1) or 0), int(m.group(2)), int(m.group(3))
            t = h * 3600 + mi * 60 + s
            if 0 < t <= duration and s < 60:
                out.append((float(t), int(c.get("like_count") or 0)))
    return out


# --------------------------------------------------------------------------
# Twitch past broadcast chat replay (Twitch's public web API)
# --------------------------------------------------------------------------
_GQL_HASH = "b70a3591ff0f4e0313d126c6a1502d79a1c02baebb288227c582044aa76adf6a"


def _ssl():
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _twitch_page(video_id: str, offset: float):
    body = json.dumps([{"operationName": "VideoCommentsByOffsetOrCursor",
                        "variables": {"videoID": video_id, "contentOffsetSeconds": int(offset)},
                        "extensions": {"persistedQuery": {"version": 1, "sha256Hash": _GQL_HASH}}}]).encode()
    req = urllib.request.Request("https://gql.twitch.tv/gql", data=body,
                                 headers={"Client-Id": TWITCH_GQL_CLIENT, "Content-Type": "text/plain"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=20, context=_ssl()) as r:
                d = json.loads(r.read())
            comments = (((d[0].get("data") or {}).get("video") or {}).get("comments")) or {}
            out = []
            for e in comments.get("edges") or []:
                n = e.get("node") or {}
                text = "".join(f.get("text", "") for f in (n.get("message") or {}).get("fragments") or [])
                user = ((n.get("commenter") or {}).get("login")) or "?"
                out.append((float(n.get("contentOffsetSeconds", 0)), user, text))
            return out
        except Exception:
            time.sleep(1 + attempt * 2)
    return None


def twitch_chat_replay(video_id: str, duration: float, progress=None, stop: threading.Event | None = None,
                       workers: int = 8, max_requests: int = 3000):
    """Chat messages of a Twitch past broadcast: [(offset_s, user, text), ...] plus a list of
    the time ranges covered. Big chats are sampled evenly instead of read in full, so even a
    13-hour stream with a huge chat takes minutes, not hours."""
    if duration <= 0:
        return [], []
    chunks = max(1, min(workers * 4, int(duration // 600) + 1))
    edges = [duration * i / chunks for i in range(chunks + 1)]
    per_chunk = max(5, max_requests // chunks)
    done_secs = [0.0] * chunks
    lock = threading.Lock()

    def walk(i: int):
        start, end = edges[i], edges[i + 1]
        t, n, got, covered = start, 0, [], []
        gap = 0.0
        while t < end and n < per_chunk * 2:
            if stop is not None and stop.is_set():
                break
            page = _twitch_page(video_id, t)
            n += 1
            if not page:
                t += 30
                continue
            first, last = page[0][0], page[-1][0]
            got.extend(m for m in page if start <= m[0] < end)
            covered.append((max(first, start), max(last, first + 0.5)))
            span = max(last - t, 0.5)
            if n == 3:  # how dense is this chat? decide whether to read all or sample
                needed = (end - start) / max(span, 0.5)
                gap = 0.0 if needed <= per_chunk else (end - start) / per_chunk - span
            t = max(last + (gap if gap > 0 else 0.01), t + 1)
            with lock:
                done_secs[i] = min(end, t) - start
                if progress:
                    progress(min(1.0, sum(done_secs) / duration))
        return got, covered

    msgs, covered = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for got, cov in pool.map(walk, range(chunks)):
            msgs.extend(got)
            covered.extend(cov)
    seen, unique = set(), []
    for m in sorted(msgs, key=lambda m: m[0]):
        key = (round(m[0], 2), m[1], m[2])
        if key not in seen:
            seen.add(key)
            unique.append(m)
    return unique, covered
