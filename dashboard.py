"""Local web dashboard for the clipper: live hype graphs, clip feed, controls.

Runs inside clipper.py on http://127.0.0.1:<port>. Only reachable from this Mac.
Uses only the Python standard library.
"""

from __future__ import annotations

import json
import mimetypes
import re
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

WEB = Path(__file__).resolve().parent / "web"


class Dashboard:
    def __init__(self, manager, args, get_shorts_maker, current_shorts_maker, stop=None):
        self.manager = manager
        self.stop = stop  # called by the page's Stop button
        self.viewer_api = lambda: None  # set by clipper.py: the TwitchAPI object
        self.scans = lambda: None  # set by clipper.py: the vodscan.ScanManager
        self.long = lambda: None  # set by clipper.py: the longform.LongManager
        self.args = args
        self.out_dir = Path(args.out).expanduser().resolve()
        self.get_shorts_maker = get_shorts_maker  # starts the worker if needed
        self.current_shorts_maker = current_shorts_maker  # returns it or None
        self.port = args.port

    def start(self) -> str:
        dash = self

        class Handler(_Handler):
            dashboard = dash

        server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True, name="dashboard").start()
        return f"http://127.0.0.1:{self.port}"

    # -- data --------------------------------------------------------------
    def state(self) -> dict:
        sm = self.current_shorts_maker()
        usage = shutil.disk_usage(self.out_dir)
        return {
            "channels": [c.state() for c in self.manager.all()],
            "max_channels": self.manager.MAX_CHANNELS,
            "disk_free_gb": round(usage.free / 1e9, 1),
            "min_free_gb": self.args.min_free_gb,
            "shorts_busy": sorted(self._rel(Path(p)) for p in (sm.busy if sm else set())),
            "auto_shorts": bool(self.args.shorts),
            "viewer_clips": self._viewer_state(),
            "scans": self.scans().state() if self.scans() else [],
            "long": self.long().state() if self.long() else [],
            "settings": self._settings(),
            "out_dir": str(self.out_dir),
        }

    def _viewer_state(self) -> dict:
        api = self.viewer_api()
        if api is None:
            return {"enabled": False, "error": None}
        return {"enabled": api.configured, "error": api.last_error}

    def clips(self) -> list[dict]:
        items = []
        if not self.out_dir.exists():
            return items
        for ch_dir in self.out_dir.iterdir():
            if not ch_dir.is_dir() or ch_dir.name.startswith(("_", ".")):
                continue  # _long = long videos, shown in their own tab
            for mp4 in ch_dir.glob("*.mp4"):
                info = {}
                side = mp4.with_suffix(".json")
                if side.exists():
                    try:
                        info = json.loads(side.read_text(encoding="utf-8"))
                    except (ValueError, OSError):
                        pass
                short = ch_dir / "shorts" / f"{mp4.stem}_short.mp4"
                st = mp4.stat()
                items.append({
                    "file": self._rel(mp4),
                    "short": self._rel(short) if short.exists() else None,
                    "channel": info.get("channel", ch_dir.name),
                    "time": info.get("spike_start", st.st_mtime),
                    "kind": info.get("kind", "test" if "_TEST" in mp4.stem else "auto"),
                    "layout": info.get("layout", "auto"),
                    "category": info.get("category"),
                    "peak": info.get("peak_ratio"),
                    "messages": info.get("messages"),
                    "chatters": info.get("chatters"),
                    "top": [t for t, _ in info.get("top_tokens", [])][:6],
                    "chat": info.get("sample_messages", [])[:6],
                    "seconds": info.get("clip_seconds"),
                    "starred": bool(info.get("starred")),
                    "score": info.get("score"),
                    "viewer_clips": info.get("viewer_clips", 0),
                    "audio_boost": info.get("audio_boost", 0),
                    "merged": info.get("merged", 1),
                    "reasons": info.get("reasons", []),
                    "credit": info.get("credit"),
                    "platform": info.get("platform", "twitch"),
                    "vod_time": info.get("vod_time_text"),
                    "source_url": info.get("source_url"),
                    "source_title": info.get("source_title"),
                    "reaction": info.get("reaction", 0),
                    "meta": self._meta(info, mp4),
                    "posted": bool(info.get("posted")),
                    "mb": round(st.st_size / 1e6, 1),
                })
        items.sort(key=lambda c: c["time"], reverse=True)
        return items[:300]

    def _settings(self) -> dict:
        import housekeeping
        st = housekeeping.load(self.out_dir)
        return {k: st.get(k) for k in ("auto_cleanup", "notify_score", "notify_mac", "notify_phone",
                                       "ntfy_topic", "channel_name")} | {"autostart": housekeeping.autostart_enabled()}

    def posted(self, rel: str, posted: bool) -> bool:
        p = self.clip_path(rel)
        if not p:
            return False
        side = p.with_suffix(".json")
        try:
            info = json.loads(side.read_text())
        except (OSError, ValueError):
            info = {}
        info["posted"] = bool(posted)
        side.write_text(json.dumps(info, indent=2, ensure_ascii=False))
        return True

    def _meta(self, info: dict, mp4: Path) -> dict | None:
        """Title/description/hashtags/keywords (made now for clips saved before this existed)."""
        if info.get("meta"):
            return info["meta"]
        try:
            from metadata import for_short
            return for_short(dict(info, file=info.get("file", mp4.name)), info.get("said", ""))
        except Exception:
            return None

    def _rel(self, p: Path) -> str:
        return p.resolve().relative_to(self.out_dir).as_posix()

    def resolve(self, rel: str, must_exist=True) -> Path | None:
        """Map a path from the browser to a file inside the clips folder, or None."""
        try:
            p = (self.out_dir / rel).resolve()
            p.relative_to(self.out_dir)
        except (ValueError, OSError):
            return None
        if must_exist and not p.is_file():
            return None
        return p

    def clip_path(self, rel: str) -> Path | None:
        """A main clip (not a short) inside the clips folder."""
        p = self.resolve(rel)
        if p is None or p.suffix != ".mp4" or p.parent.name == "shorts":
            return None
        return p

    # -- actions -----------------------------------------------------------
    def star(self, rel: str, starred: bool) -> bool:
        p = self.clip_path(rel)
        if not p:
            return False
        side = p.with_suffix(".json")
        info = {}
        if side.exists():
            try:
                info = json.loads(side.read_text(encoding="utf-8"))
            except ValueError:
                pass
        info["starred"] = bool(starred)
        side.write_text(json.dumps(info, indent=2, ensure_ascii=False))
        return True

    def delete(self, rel: str) -> bool:
        """Moves the clip, its short and its info to the Trash / Recycle Bin (restorable)."""
        p = self.clip_path(rel)
        if not p:
            return False
        import platformkit
        for f in (p, p.with_suffix(".json"), p.with_name(p.stem + "_grid.jpg"),
                  p.parent / "shorts" / f"{p.stem}_short.mp4"):
            if f.exists() and not platformkit.to_trash(f):
                f.unlink(missing_ok=True)  # no Trash access: fall back to deleting
        return True

    def make_short(self, rel: str) -> bool:
        p = self.clip_path(rel)
        if not p:
            return False
        channel = p.parent.name
        self.get_shorts_maker().submit(p, channel)
        return True


class _Handler(BaseHTTPRequestHandler):
    dashboard: Dashboard
    server_version = "Clipper"

    def log_message(self, *a):  # keep the terminal clean
        pass

    # -- helpers -----------------------------------------------------------
    def _host_ok(self) -> bool:
        # Blocks DNS-rebinding: only answer requests addressed to this Mac.
        host = (self.headers.get("Host") or "").split(":")[0]
        return host in ("127.0.0.1", "localhost")

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, ctype: str | None = None):
        size = path.stat().st_size
        ctype = ctype or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        start, end = 0, size - 1
        rng = self.headers.get("Range")
        m = re.match(r"bytes=(\d*)-(\d*)", rng or "")
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
            else:  # suffix range: last N bytes
                start = max(0, size - int(m.group(2)))
            end = min(end, size - 1)
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        length = end - start + 1
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        if self.command == "HEAD":
            return
        with open(path, "rb") as f:
            f.seek(start)
            left = length
            try:
                while left > 0:
                    chunk = f.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass  # browser stopped loading (seeking, closed tab)

    # -- routes ------------------------------------------------------------
    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        if not self._host_ok():
            return self._json({"error": "forbidden"}, 403)
        d = self.dashboard
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._file(WEB / "index.html", "text/html; charset=utf-8")
        if path.startswith("/fonts/") and path.endswith(".ttf"):
            f = (WEB / "fonts" / Path(path).name)
            if f.is_file():
                return self._file(f, "font/ttf")
        if path == "/api/state":
            return self._json(d.state())
        if path == "/api/clips":
            return self._json(d.clips())
        if path.startswith("/media/"):
            p = d.resolve(unquote(path[len("/media/"):]))
            if p is None or p.suffix not in (".mp4", ".jpg"):
                return self._json({"error": "not found"}, 404)
            return self._file(p)
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        # The custom header can't be sent cross-site without a CORS preflight
        # (which we never approve), so other websites can't trigger actions.
        if not self._host_ok() or self.headers.get("X-Clipper") != "1":
            return self._json({"error": "forbidden"}, 403)
        d = self.dashboard
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}") if n < 65536 else {}
        except ValueError:
            return self._json({"error": "bad json"}, 400)
        path = urlparse(self.path).path

        if path == "/api/quit":
            if d.stop is None:
                return self._json({"ok": False}, 400)
            threading.Timer(0.3, d.stop).start()  # reply first, then shut down
            return self._json({"ok": True})

        if path == "/api/clip/posted":
            return self._json({"ok": d.posted(str(body.get("file", "")), bool(body.get("posted")))})

        if path == "/api/settings":
            import housekeeping
            ch = {}
            for k in ("auto_cleanup", "notify_mac", "notify_phone"):
                if k in body:
                    ch[k] = bool(body[k])
            if "notify_score" in body:
                try:
                    ch["notify_score"] = max(0, min(100, int(body["notify_score"])))
                except (TypeError, ValueError):
                    pass
            if body.get("notify_phone") and not housekeeping.load(d.out_dir).get("ntfy_topic"):
                ch["ntfy_topic"] = housekeeping.new_topic()
            if "autostart" in body:
                housekeeping.set_autostart(bool(body["autostart"]))
            housekeeping.save(d.out_dir, ch)
            return self._json({"ok": True, "settings": d._settings()})

        if path == "/api/settings/test-alert":
            import housekeeping
            st = housekeeping.load(d.out_dir)
            housekeeping.notify(st, "🔥 Test alert", "Your clipper alerts are working!")
            return self._json({"ok": True})

        if path == "/api/settings/twitch":
            api = d.viewer_api()
            if api is None:
                return self._json({"ok": False, "message": "Not available"}, 400)
            if body.get("clear"):
                api.clear()
                return self._json({"ok": True, "message": "Viewer clips turned off"})
            ok, msg = api.save(str(body.get("client_id", "")), str(body.get("client_secret", "")))
            return self._json({"ok": ok, "message": msg}, 200 if ok else 400)

        if path == "/api/channels/add":
            from sources import SourceError, resolve
            try:
                src = resolve(str(body.get("url", "")))
            except SourceError as e:
                return self._json({"ok": False, "message": str(e)}, 400)
            except Exception as e:
                return self._json({"ok": False, "message": f"Couldn't open that link ({e})"}, 400)
            if src["kind"] == "live":
                ok, msg = d.manager.add(src["name"], source=src)
                return self._json({"ok": ok, "kind": "live", "message": msg, "channel": src["name"]},
                                  200 if ok else 409)
            scans = d.scans()
            if scans is None:
                return self._json({"ok": False, "message": "Scanning isn't available"}, 400)
            try:
                top = int(body.get("top") or 10)
            except (TypeError, ValueError):
                top = 10
            job = scans.start(src, top)
            what = "chat + reactions" if src.get("chat") else "reactions + comments (no chat replay)"
            return self._json({"ok": True, "kind": "vod", "id": job.id, "channel": src["name"],
                               "message": f"Scanning \"{(src.get('title') or '')[:60]}\" for the best "
                                          f"{job.top} moments ({what})"})

        if path.startswith("/api/long"):
            lm = d.long()
            if lm is None:
                return self._json({"ok": False, "message": "Long videos aren't available"}, 400)
            if path == "/api/long/create":
                links = body.get("links") or []
                if isinstance(links, str):
                    links = re.split(r"[\s,]+", links)
                try:
                    proj = lm.create([str(x) for x in links], str(body.get("topic", "")), int(body.get("minutes") or 12))
                    chan = str(body.get("channel_name") or "").strip()[:40]
                    if chan:  # remembered for next time, used on the end screen and in the description
                        import housekeeping
                        housekeeping.save(d.out_dir, {"channel_name": chan})
                        proj.channel_name = chan
                        proj.save()
                except (ValueError, TypeError) as e:
                    return self._json({"ok": False, "message": str(e)}, 400)
                return self._json({"ok": True, "id": proj.id,
                                   "message": f"Scanning {len(proj.links)} stream(s) for '{proj.topic}'…"})
            m = re.fullmatch(r"/api/long/([\w-]+)/(storyboard|render|cancel|delete)", path)
            if not m:
                return self._json({"error": "not found"}, 404)
            pid, act = m.group(1), m.group(2)
            if act == "storyboard":
                ok = lm.update(pid, body.get("storyboard") or [], body.get("channel_name"), body.get("options"))
            elif act == "render":
                ok = lm.render(pid)
            elif act == "cancel":
                ok = lm.cancel(pid)
            else:
                ok = lm.delete(pid)
            return self._json({"ok": ok})

        m = re.fullmatch(r"/api/scan/(\d+)/(cancel|dismiss)", path)
        if m:
            scans = d.scans()
            if scans is None:
                return self._json({"ok": False}, 400)
            jid = int(m.group(1))
            ok = scans.cancel(jid) if m.group(2) == "cancel" else scans.dismiss(jid)
            return self._json({"ok": ok})

        m = re.fullmatch(r"/api/channel/([a-z0-9_]+)/(sensitivity|clip|remove|mode|reactions)", path)
        if m:
            if m.group(2) == "remove":
                return self._json({"ok": d.manager.remove(m.group(1))})
            c = d.manager.get(m.group(1))
            if not c:
                return self._json({"error": "unknown channel"}, 404)
            if m.group(2) == "clip":
                c.request_clip()
            elif m.group(2) == "reactions":
                c.set_reactions(bool(body.get("on")))
                d.manager.save()
            elif m.group(2) == "mode":
                if not c.set_mode(str(body.get("mode"))):
                    return self._json({"error": "bad mode"}, 400)
                d.manager.save()
            else:
                try:
                    c.set_sensitivity(float(body.get("value")))
                except (TypeError, ValueError):
                    return self._json({"error": "bad value"}, 400)
                d.manager.save()
            return self._json({"ok": True})

        rel = str(body.get("file", ""))
        if path == "/api/clip/star":
            ok = d.star(rel, bool(body.get("starred")))
        elif path == "/api/clip/delete":
            ok = d.delete(rel)
        elif path == "/api/clip/short":
            ok = d.make_short(rel)
        else:
            return self._json({"error": "not found"}, 404)
        return self._json({"ok": ok}, 200 if ok else 404)
