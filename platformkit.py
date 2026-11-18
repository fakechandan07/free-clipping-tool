"""Everything that differs between macOS and Windows, in one place.

Import this before looking up tools: on Windows it puts the tools that setup
downloaded (ffmpeg, whisper, deno in ./tools) on the PATH for this process.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
WINDOWS = sys.platform == "win32"
MAC = sys.platform == "darwin"

# Windows: no console windows popping up for every ffmpeg/whisper call
NO_WINDOW = {"creationflags": 0x08000000} if WINDOWS else {}

if WINDOWS:
    _tools = HERE / "tools"
    _extra = [_tools / "ffmpeg" / "bin", _tools / "whisper", _tools / "deno"]
    _extra += [p.parent for p in (_tools / "whisper").glob("**/whisper-cli.exe")]
    os.environ["PATH"] = os.pathsep.join([str(p) for p in _extra if p.exists()] + [os.environ.get("PATH", "")])
    os.environ.setdefault("PYTHONUTF8", "1")
    # every subprocess we start without a console window
    _orig_popen_init = subprocess.Popen.__init__

    def _popen_init(self, *a, **kw):
        kw.setdefault("creationflags", 0x08000000)
        _orig_popen_init(self, *a, **kw)

    subprocess.Popen.__init__ = _popen_init


def venv_tool(name: str) -> str:
    """A command installed in this folder's .venv (streamlink, yt-dlp), else from PATH."""
    p = HERE / ".venv" / ("Scripts" if WINDOWS else "bin") / (name + (".exe" if WINDOWS else ""))
    return str(p) if p.exists() else (shutil.which(name) or name)


def venv_python(windowless: bool = False) -> str:
    if WINDOWS:
        return str(HERE / ".venv" / "Scripts" / ("pythonw.exe" if windowless else "python.exe"))
    return str(HERE / ".venv" / "bin" / "python")


# --------------------------------------------------------------------------
# Video encoder: the computer's hardware encoder if it has one, else the standard one
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def video_encoder() -> str:
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    options = ["h264_videotoolbox"] if MAC else ["h264_nvenc", "h264_qsv", "h264_amf"] if WINDOWS else []
    for enc in options:
        try:
            r = subprocess.run([ffmpeg, "-v", "error", "-f", "lavfi", "-i", "color=black:s=256x256:d=0.2",
                                "-c:v", enc, "-b:v", "1M", "-f", "null", "-"], capture_output=True, timeout=20)
            if r.returncode == 0:
                return enc
        except (OSError, subprocess.SubprocessError):
            pass
    return "libx264"


# --------------------------------------------------------------------------
# Fonts
# --------------------------------------------------------------------------
def emoji_font() -> str | None:
    for f in ("/System/Library/Fonts/Apple Color Emoji.ttc",
              os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "seguiemj.ttf"),
              "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"):
        if os.path.exists(f):
            return f
    return None


def emoji_font_size(path: str | None) -> int:
    return 160 if path and path.endswith(".ttc") else 136  # Apple's emoji only comes at fixed sizes


def bold_fonts() -> list[str]:
    win = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    return ["/System/Library/Fonts/Supplemental/Arial Black.ttf", "/System/Library/Fonts/Supplemental/Impact.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            os.path.join(win, "ariblk.ttf"), os.path.join(win, "impact.ttf"), os.path.join(win, "arialbd.ttf")]


# --------------------------------------------------------------------------
# Narrator voice (Mac: `say`; Windows: the built-in Windows voices)
# --------------------------------------------------------------------------
MAC_VOICES = ["Evan (Enhanced)", "Allison (Enhanced)", "Joelle (Enhanced)", "Daniel", "Samantha", "Alex"]
WIN_VOICES = ["Microsoft Mark", "Microsoft David", "Microsoft Zira"]

_PS = ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command"]


def pick_voice() -> str | None:
    try:
        if MAC:
            out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=20).stdout
            have = {line.split("  ")[0].strip() for line in out.splitlines()}
            return next((v for v in MAC_VOICES if v in have), None)
        if WINDOWS:
            out = subprocess.run(_PS + ["Add-Type -AssemblyName System.Speech; "
                                        "(New-Object System.Speech.Synthesis.SpeechSynthesizer).GetInstalledVoices() "
                                        "| ForEach-Object { $_.VoiceInfo.Name }"],
                                 capture_output=True, text=True, timeout=30).stdout
            have = [l.strip() for l in out.splitlines() if l.strip()]
            for want in WIN_VOICES:
                for v in have:
                    if v.startswith(want):
                        return v
            return have[0] if have else None
    except (OSError, subprocess.SubprocessError):
        return None
    return None


def say_to_file(text: str, voice: str, out: Path) -> Path | None:
    """Speak `text` into an audio file. Returns the file, or None."""
    try:
        if MAC:
            f = out.with_suffix(".aiff")
            subprocess.run(["say", "-v", voice, "-r", "182", "-o", str(f), text], capture_output=True, timeout=60)
        elif WINDOWS:
            f = out.with_suffix(".raw.wav")
            txt = out.with_suffix(".txt")
            txt.write_text(text, encoding="utf-8")  # passed as a file: no quoting problems
            script = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                      f"$s.SelectVoice('{voice.replace(chr(39), '')}'); $s.Rate = 1; "
                      f"$s.SetOutputToWaveFile('{str(f).replace(chr(39), chr(39) * 2)}'); "
                      f"$s.Speak([IO.File]::ReadAllText('{str(txt).replace(chr(39), chr(39) * 2)}')); $s.Dispose()")
            subprocess.run(_PS + [script], capture_output=True, timeout=90)
            txt.unlink(missing_ok=True)
        else:
            return None
    except (OSError, subprocess.SubprocessError):
        return None
    return f if f.exists() and f.stat().st_size > 1000 else None


# --------------------------------------------------------------------------
# Notifications
# --------------------------------------------------------------------------
def notify(title: str, message: str) -> None:
    safe = lambda t: t.replace("\\", "").replace('"', "'").replace("`", "'")[:200]
    try:
        if MAC:
            subprocess.run(["osascript", "-e", f'display notification "{safe(message)}" with title "{safe(title)}" '
                                               f'sound name "Glass"'], capture_output=True, timeout=15)
        elif WINDOWS:
            x = lambda t: (safe(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                           .replace("'", "&apos;"))
            script = (
                "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType=WindowsRuntime] > $null; "
                "[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType=WindowsRuntime] > $null; "
                "$x = New-Object Windows.Data.Xml.Dom.XmlDocument; "
                f"$x.LoadXml('<toast><visual><binding template=\"ToastGeneric\"><text>{x(title)}</text>"
                f"<text>{x(message)}</text></binding></visual></toast>'); "
                "$app = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\\WindowsPowerShell\\v1.0\\powershell.exe'; "
                "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($app).Show("
                "[Windows.UI.Notifications.ToastNotification]::new($x))")
            subprocess.run(_PS + [script], capture_output=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        pass


# --------------------------------------------------------------------------
# Trash / Recycle Bin
# --------------------------------------------------------------------------
TRASH_NAME = "Recycle Bin" if WINDOWS else "Trash"


def to_trash(path: Path) -> bool:
    """Move a file or folder to the Trash / Recycle Bin (restorable). False if that failed."""
    path = Path(path)
    if not path.exists():
        return True
    if WINDOWS:
        kind = "DeleteDirectory" if path.is_dir() else "DeleteFile"
        p = str(path.resolve()).replace("'", "''")
        r = subprocess.run(_PS + ["Add-Type -AssemblyName Microsoft.VisualBasic; "
                                  f"[Microsoft.VisualBasic.FileIO.FileSystem]::{kind}('{p}', "
                                  "'OnlyErrorDialogs', 'SendToRecycleBin')"], capture_output=True, timeout=60)
        return r.returncode == 0 and not path.exists()
    trash = Path.home() / ".Trash"
    target = trash / path.name
    n = 2
    while target.exists():
        target = trash / f"{path.stem} {n}{path.suffix}"
        n += 1
    try:
        shutil.move(str(path), str(target))
        return True
    except OSError:
        return False


# --------------------------------------------------------------------------
# Keep the computer awake while clipping
# --------------------------------------------------------------------------
def keep_awake() -> None:
    if MAC and shutil.which("caffeinate"):
        subprocess.Popen(["caffeinate", "-ims", "-w", str(os.getpid())])
    elif WINDOWS:
        try:
            import ctypes
            # ES_CONTINUOUS | ES_SYSTEM_REQUIRED: no sleep while this process runs
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        except Exception:
            pass


# --------------------------------------------------------------------------
# Start with the computer
# --------------------------------------------------------------------------
def _win_startup_file() -> Path:
    return (Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "Microsoft" / "Windows" /
            "Start Menu" / "Programs" / "Startup" / "Five Star Clipper.vbs")


def win_autostart_enabled() -> bool:
    return _win_startup_file().exists()


def win_set_autostart(on: bool) -> tuple[bool, str]:
    f = _win_startup_file()
    if not on:
        f.unlink(missing_ok=True)
        return True, "The clipper won't start by itself anymore."
    py = venv_python(windowless=True).replace('"', '""')
    script = str(HERE / "clipper.py").replace('"', '""')
    log = str(HERE / "clipper.log").replace('"', '""')
    vbs = ('Set sh = CreateObject("WScript.Shell")\r\n'
           f'sh.CurrentDirectory = "{str(HERE)}"\r\n'
           'sh.Environment("PROCESS")("PYTHONUTF8") = "1"\r\n'
           f'sh.Run """{py}"" ""{script}"" --no-open --log ""{log}""", 0, False\r\n')
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(vbs, encoding="utf-8")
    return True, "The clipper will start by itself when you sign in to Windows."
