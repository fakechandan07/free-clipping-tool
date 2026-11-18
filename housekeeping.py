"""Keeping things tidy: settings, no duplicate clips, auto-cleanup when the disk fills
up, notifications (Mac + phone), and starting with the Mac."""

from __future__ import annotations

import json
import os
import plistlib
import secrets
import shutil
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
_lock = threading.Lock()


# --------------------------------------------------------------------------
# Settings (clips/settings.json, shared with the Twitch viewer-clips keys)
# --------------------------------------------------------------------------
DEFAULTS = {"auto_cleanup": True, "notify_score": 80, "notify_mac": True, "ntfy_topic": "",
            "notify_phone": False, "channel_name": ""}


def load(out_dir: Path) -> dict:
    try:
        data = json.loads((Path(out_dir) / "settings.json").read_text())
    except (OSError, ValueError):
        data = {}
    return {**DEFAULTS, **data}


def save(out_dir: Path, changes: dict) -> dict:
    with _lock:
        f = Path(out_dir) / "settings.json"
        try:
            data = json.loads(f.read_text())
        except (OSError, ValueError):
            data = {}
        data.update(changes)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(data, indent=2))
        os.chmod(f, 0o600)
    return {**DEFAULTS, **data}


# --------------------------------------------------------------------------
# No duplicate clips
# --------------------------------------------------------------------------
def is_duplicate(folder: Path, info: dict, start_wall: float, end_wall: float) -> str | None:
    """Name of an existing clip that already shows this moment, or None."""
    for side in Path(folder).glob("*.json"):
        try:
            other = json.loads(side.read_text())
        except (OSError, ValueError):
            continue
        if not side.with_suffix(".mp4").exists():
            continue
        if info.get("source_url") and other.get("source_url") == info["source_url"]:
            if abs(float(other.get("vod_time", -999)) - float(info.get("vod_time", 0))) < 45:
                return side.with_suffix(".mp4").name
            continue
        a0, a1 = other.get("clip_start_wall"), other.get("clip_end_wall")
        if not info.get("source_url") and a0 and a1:
            overlap = min(a1, end_wall) - max(a0, start_wall)
            if overlap > 0.5 * min(a1 - a0, end_wall - start_wall):
                return side.with_suffix(".mp4").name
    return None


# --------------------------------------------------------------------------
# Auto-cleanup when the disk is almost full
# --------------------------------------------------------------------------
def to_trash(path: Path) -> None:
    import platformkit
    if not platformkit.to_trash(path):
        path.unlink(missing_ok=True)


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def cleanup(out_dir: Path, log, low_gb: float = 3.0, target_gb: float = 4.0) -> int:
    """When free space drops under low_gb, move the oldest clips to the Trash until
    target_gb is free: already-posted clips first, never starred ones."""
    out_dir = Path(out_dir)
    if free_gb(out_dir) >= low_gb:
        return 0
    clips = []
    for ch in out_dir.iterdir():
        if not ch.is_dir() or ch.name.startswith(("_", ".")):
            continue
        for mp4 in ch.glob("*.mp4"):
            try:
                info = json.loads(mp4.with_suffix(".json").read_text())
            except (OSError, ValueError):
                info = {}
            if info.get("starred"):
                continue
            clips.append((0 if info.get("posted") else 1, mp4.stat().st_mtime, mp4))
    clips.sort()
    moved = 0
    for _, _, mp4 in clips:
        if free_gb(out_dir) >= target_gb:
            break
        for f in (mp4, mp4.with_suffix(".json"), mp4.parent / "shorts" / f"{mp4.stem}_short.mp4"):
            if f.exists():
                to_trash(f)
        moved += 1
    if moved:
        import platformkit
        t = platformkit.TRASH_NAME
        log("cleanup", f"disk was almost full: moved {moved} old clip(s) to the {t} "
                       f"(starred clips are never touched). Empty the {t} to free the space.")
    return moved


# --------------------------------------------------------------------------
# Notifications: Mac notification + optional phone push (ntfy.sh, free app)
# --------------------------------------------------------------------------
def new_topic() -> str:
    return "fivestar-" + secrets.token_urlsafe(12).replace("_", "x").replace("-", "y").lower()


def notify(settings: dict, title: str, message: str, log=None) -> None:
    def work():
        if settings.get("notify_mac", True):  # the computer's own notification (Mac or Windows)
            import platformkit
            platformkit.notify(title, message)
        topic = settings.get("ntfy_topic")
        if settings.get("notify_phone") and topic:
            try:
                req = urllib.request.Request(f"https://ntfy.sh/{topic}", data=message.encode(),
                                             headers={"Title": title.encode("ascii", "ignore").decode() or "New clip",
                                                      "Tags": "fire", "Priority": "4"})
                import ssl
                try:
                    import certifi
                    ctx = ssl.create_default_context(cafile=certifi.where())
                except ImportError:
                    ctx = ssl.create_default_context()
                urllib.request.urlopen(req, timeout=10, context=ctx).read()
            except Exception as e:
                if log:
                    log("alerts", f"couldn't send the phone alert ({e})")
    threading.Thread(target=work, daemon=True).start()


# --------------------------------------------------------------------------
# Start with the Mac (login item that also restarts the clipper if it crashes)
# --------------------------------------------------------------------------
LABEL = "com.fivestarclipper.agent"
PLIST = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def autostart_enabled() -> bool:
    import platformkit
    if platformkit.WINDOWS:
        return platformkit.win_autostart_enabled()
    return PLIST.exists()


def set_autostart(on: bool) -> tuple[bool, str]:
    import platformkit
    if platformkit.WINDOWS:
        return platformkit.win_set_autostart(on)
    uid = os.getuid()
    if on:
        python = str(HERE / ".venv" / "bin" / "python")
        plist = {
            "Label": LABEL,
            "ProgramArguments": [python, str(HERE / "clipper.py"), "--no-open"],
            "WorkingDirectory": str(HERE),
            "RunAtLoad": True,
            # restart only after a crash: the page's Stop button exits normally and stays stopped
            "KeepAlive": {"SuccessfulExit": False},
            "ThrottleInterval": 30,
            "EnvironmentVariables": {"PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
                                     "PYTHONUNBUFFERED": "1"},
            "StandardOutPath": str(HERE / "clipper.log"),
            "StandardErrorPath": str(HERE / "clipper.log"),
            "ProcessType": "Interactive",
        }
        PLIST.parent.mkdir(parents=True, exist_ok=True)
        with open(PLIST, "wb") as f:
            plistlib.dump(plist, f)
        # register it for next login without starting a second clipper right now
        subprocess.run(["launchctl", "bootout", f"gui/{uid}", str(PLIST)], capture_output=True)
        r = subprocess.run(["launchctl", "enable", f"gui/{uid}/{LABEL}"], capture_output=True, text=True)
        return True, "The clipper will start by itself when you log in to your Mac."
    subprocess.run(["launchctl", "bootout", f"gui/{uid}", str(PLIST)], capture_output=True)
    PLIST.unlink(missing_ok=True)
    return True, "The clipper won't start by itself anymore."
