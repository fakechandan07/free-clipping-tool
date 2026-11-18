#!/usr/bin/env python3
"""Turn horizontal stream clips into vertical 9:16 shorts with TikTok-style captions.

    python shorts.py clips/tfue/some_clip.mp4                 # blurred-background layout
    python shorts.py clip.mp4 --layout split --facecam top-right
    python shorts.py clip.mp4 --layout split --facecam 0.72,0.0,0.28,0.3   # x,y,w,h as fractions
    python shorts.py --frame clip.mp4                         # save a still with a grid to find the facecam

Captions are transcribed locally with whisper.cpp (free, offline) and drawn
word by word, with the word being spoken highlighted.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
import platformkit  # noqa: E402
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"
WHISPER = shutil.which("whisper-cli")
MODEL = HERE / "models" / "ggml-base.en.bin"

W, H = 1080, 1920
FONT_PATHS = [
    str(HERE / "web" / "fonts" / "PassionOne-Regular.ttf"),  # bold GTA-style display font (OFL)
    *platformkit.bold_fonts(),
]
HIGHLIGHT = (255, 225, 60, 255)
WHITE = (255, 255, 255, 255)
BLACK = (0, 0, 0, 255)

FACECAM_PRESETS = {  # x, y, w, h as fractions of the source frame
    "top-left": (0.0, 0.0, 0.26, 0.30),
    "top-right": (0.74, 0.0, 0.26, 0.30),
    "bottom-left": (0.0, 0.70, 0.26, 0.30),
    "bottom-right": (0.74, 0.70, 0.26, 0.30),
}

SWEARS = re.compile(
    r"\b(f+u+c+k+\w*|s+h+i+t+\w*|b+i+t+c+h+\w*|a+s+s+h+o+l+e+s?|d+i+c+k+s?|"
    r"c+u+n+t+s?|p+u+s+s+y+|n+i+g+g+\w*|f+a+g+\w*|r+e+t+a+r+d+\w*)\b", re.I)


def censor(word: str) -> str:
    def mask(m):
        w = m.group(0)
        return w[0] + "*" * (len(w) - 2) + w[-1] if len(w) > 2 else w[0] + "*"
    return SWEARS.sub(mask, word)


# --------------------------------------------------------------------------
# Transcription
# --------------------------------------------------------------------------
@dataclass
class Word:
    start: float
    end: float
    text: str


def transcribe(video: Path, workdir: Path) -> list[Word]:
    if not WHISPER:
        raise RuntimeError("whisper-cli not found (brew install whisper-cpp)")
    if not MODEL.exists():
        raise RuntimeError(f"speech model missing: {MODEL}")
    wav = workdir / "audio.wav"
    subprocess.run([FFMPEG, "-v", "error", "-y", "-i", str(video), "-vn", "-ac", "1",
                    "-ar", "16000", str(wav)], check=True)
    base = workdir / "words"
    subprocess.run([WHISPER, "-m", str(MODEL), "-f", str(wav), "-l", "en", "-ml", "1", "-sow",
                    "-ojf", "-of", str(base), "-np"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    data = json.loads((base.with_suffix(".json")).read_text(encoding="utf-8", errors="replace"))

    raw = []
    for seg in data.get("transcription", []):
        text = seg["text"].strip()
        toks = [t for t in seg.get("tokens", []) if not t["text"].startswith("[_")]
        if not text or not toks or text.startswith("["):  # [BLANK_AUDIO], [Music] ...
            continue
        conf = sum(t.get("p", 1.0) for t in toks) / len(toks)
        if conf < 0.2:
            continue  # probably a hallucination on noise/music
        raw.append(Word(seg["offsets"]["from"] / 1000, seg["offsets"]["to"] / 1000, text))
    return _spread_bunched(raw)


def _spread_bunched(words: list[Word]) -> list[Word]:
    """whisper.cpp often gives several words the same start time with zero length.
    Spread each such bunch evenly across the time it actually covers."""
    out, i = [], 0
    while i < len(words):
        j = i
        while j + 1 < len(words) and abs(words[j + 1].start - words[i].start) < 1e-3:
            j += 1
        group = words[i:j + 1]
        start = group[0].start
        end = max(w.end for w in group)
        nxt = words[j + 1].start if j + 1 < len(words) else end + 1.0
        min_end = start + 0.28 * len(group)
        if end < min_end:
            end = min(min_end, max(nxt, end))
        step = max(end - start, 0.05 * len(group)) / len(group)
        for k, w in enumerate(group):
            out.append(Word(start + k * step, start + (k + 1) * step, w.text))
        i = j + 1
    return out


def chunk_words(words: list[Word], max_words=3, max_chars=18, max_gap=0.5) -> list[list[Word]]:
    chunks, cur = [], []
    for w in words:
        if cur and (len(cur) >= max_words
                    or sum(len(x.text) + 1 for x in cur) + len(w.text) > max_chars
                    or w.start - cur[-1].end > max_gap
                    or cur[-1].text[-1:] in ".!?,"):
            chunks.append(cur)
            cur = []
        cur.append(w)
    if cur:
        chunks.append(cur)
    return chunks


# --------------------------------------------------------------------------
# Caption images
# --------------------------------------------------------------------------
def _font(size: int) -> ImageFont.FreeTypeFont:
    for p in FONT_PATHS:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size)


def _clean(text: str, do_censor: bool) -> str:
    text = text.strip().upper()
    text = re.sub(r"^[\"'(]+|[\"',;:)]+$", "", text)
    return censor(text) if do_censor else text


EMOJI_FONT = platformkit.emoji_font()
EMOJI_RULES = [  # caption keyword -> emoji shown above the caption
    (r"\b(lmao+|lol+|haha\w*|laugh\w*|funny|crying|hilarious)\b", "😂"),
    (r"\b(dead|died|dying|killed|wasted|rip)\b", "💀"),
    (r"\b(cops?|police|sirens?|wanted)\b", "🚔"),
    (r"\b(money|cash|rich|million\w*|bank)\b", "💰"),
    (r"\b(wtf|omg|oh my god|no way|what the)\b", "😱"),
    (r"\b(cars?|drive|driving|crash\w*)\b", "🚗"),
    (r"\b(guns?|shoot\w*|shot)\b", "🔫"),
    (r"\b(fire|insane|crazy|lit|goated)\b", "🔥"),
]
_emoji_cache: dict = {}


def emoji_for(text: str) -> str | None:
    for rx, e in EMOJI_RULES:
        if re.search(rx, text or "", re.I):
            return e
    return None


def emoji_image(e: str, size: int = 120) -> Image.Image | None:
    if e in _emoji_cache:
        return _emoji_cache[e]
    img = None
    try:
        f = ImageFont.truetype(EMOJI_FONT, platformkit.emoji_font_size(EMOJI_FONT))
        tmp = Image.new("RGBA", (220, 220), (0, 0, 0, 0))
        ImageDraw.Draw(tmp).text((10, 10), e, font=f, embedded_color=True)
        box = tmp.getbbox()
        if box:
            img = tmp.crop(box).resize((size, size), Image.LANCZOS)
    except Exception:
        img = None
    _emoji_cache[e] = img
    return img


def render_hook(text: str, y: int) -> Image.Image:
    """Big hook text for the first seconds of a short ("WAIT FOR IT")."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    if not text:
        return img
    d = ImageDraw.Draw(img)
    size = 112
    while size > 60 and d.textlength(text, font=_font(size)) > W - 90:
        size -= 6
    font = _font(size)
    tw = d.textlength(text, font=font)
    pad = 26
    d.rounded_rectangle([(W - tw) / 2 - pad, y - pad // 2, (W + tw) / 2 + pad, y + size + pad // 2],
                        24, fill=(255, 61, 139, 235), outline=(0, 0, 0, 255), width=7)
    d.text(((W - tw) / 2, y), text, font=font, fill=(255, 255, 255), stroke_width=max(6, size // 12),
           stroke_fill=(0, 0, 0))
    return img


SFX_DIR = HERE / "models" / "sfx"


def ensure_sfx() -> tuple[Path, Path] | None:
    """Our own sound effects, generated with ffmpeg (no copyright): a deep boom for the
    big moment and a whoosh for the hook."""
    SFX_DIR.mkdir(parents=True, exist_ok=True)
    boom, whoosh = SFX_DIR / "boom.wav", SFX_DIR / "whoosh.wav"
    if not boom.exists():
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=55:duration=1.3",
                        "-f", "lavfi", "-i", "anoisesrc=d=0.3:c=brown:a=0.7", "-filter_complex",
                        "[0]volume=2.2,afade=t=out:st=0.08:d=1.2[a];[1]lowpass=f=260,volume=1.5,afade=t=out:st=0:d=0.3[b];"
                        "[a][b]amix=inputs=2:normalize=0,alimiter=limit=0.9,aformat=channel_layouts=stereo,aresample=48000",
                        str(boom)], capture_output=True)
    if not whoosh.exists():
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "anoisesrc=d=0.7:c=pink:a=0.6",
                        "-af", "highpass=f=700,lowpass=f=5000,afade=t=in:st=0:d=0.4,afade=t=out:st=0.4:d=0.3,"
                               "volume=0.9,aformat=channel_layouts=stereo,aresample=48000", str(whoosh)],
                       capture_output=True)
    return (boom, whoosh) if boom.exists() and whoosh.exists() else None


def render_caption(words: list[str], active: int, center_y: int, emoji: str | None = None) -> Image.Image:
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    size = 118 if "Passion" in str(getattr(_font(10), "path", "")) else 92
    while True:
        font = _font(size)
        space = draw.textlength(" ", font=font)
        widths = [draw.textlength(w, font=font) for w in words]
        lines, line, lw = [], [], 0.0
        for i, wd in enumerate(widths):  # greedy wrap at 960px
            add = wd + (space if line else 0)
            if line and lw + add > 960:
                lines.append((line, lw))
                line, lw = [], 0.0
                add = wd
            line.append(i)
            lw += add
        lines.append((line, lw))
        if len(lines) <= 2 or size <= 56:
            break
        size -= 8
    stroke = max(7, size // 9)
    line_h = int(size * 1.15)
    y = center_y - (len(lines) * line_h) // 2
    for idxs, lw in lines:
        x = (W - lw) / 2
        for i in idxs:
            draw.text((x, y), words[i], font=font, fill=HIGHLIGHT if i == active else WHITE,
                      stroke_width=stroke, stroke_fill=BLACK)
            x += widths[i] + space
        y += line_h
    if emoji:
        e = emoji_image(emoji, 118)
        if e is not None:
            top = center_y - (len(lines) * line_h) // 2 - 132
            img.alpha_composite(e, ((W - e.width) // 2, max(0, top)))
    return img


def render_credit(text: str, y: int = 70) -> Image.Image:
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    if text:
        draw = ImageDraw.Draw(img)
        font = _font(44)
        tw = draw.textlength(text, font=font)
        draw.text(((W - tw) / 2, y), text, font=font, fill=(255, 255, 255, 235),
                  stroke_width=5, stroke_fill=(0, 0, 0, 220))
    return img


def build_caption_track(words: list[Word], duration: float, center_y: int, credit: str,
                        do_censor: bool, workdir: Path, credit_y: int = 70) -> Path:
    """Write caption frames as PNGs plus an ffconcat list that times them."""
    credit_img = render_credit(credit, credit_y)
    blank = workdir / "cap_blank.png"
    credit_img.save(blank)

    events = []  # (start, end, image path)
    chunks = chunk_words(words)
    for ci, chunk in enumerate(chunks):
        texts = [_clean(w.text, do_censor) for w in chunk]
        if not any(texts):
            continue
        chunk_end = chunk[-1].end + 0.35
        if ci + 1 < len(chunks):
            chunk_end = min(chunk_end, chunks[ci + 1][0].start)
        for wi, w in enumerate(chunk):
            s = w.start
            e = chunk[wi + 1].start if wi + 1 < len(chunk) else chunk_end
            if e - s < 0.04:
                continue
            path = workdir / f"cap_{ci:03d}_{wi}.png"
            frame = credit_img.copy()
            frame.alpha_composite(render_caption(texts, wi, center_y, emoji_for(" ".join(texts))))
            frame.save(path)
            events.append((s, e, path))

    lines = ["ffconcat version 1.0"]
    t = 0.0
    for s, e, path in events:
        s, e = max(s, t), min(e, duration)
        if e <= s:
            continue
        if s - t > 0.01:
            lines += [f"file '{blank}'", f"duration {s - t:.3f}"]
        lines += [f"file '{path}'", f"duration {e - s:.3f}"]
        t = e
    if duration - t > 0.01:
        lines += [f"file '{blank}'", f"duration {duration - t:.3f}"]
    lines.append(f"file '{blank}'")  # concat demuxer needs the last file repeated
    listing = workdir / "captions.ffconcat"
    listing.write_text("\n".join(lines) + "\n")
    return listing


# --------------------------------------------------------------------------
# Video layouts
# --------------------------------------------------------------------------
def probe(video: Path) -> tuple[int, int, float]:
    out = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height:format=duration", "-of", "json", str(video)],
                         capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    s = d["streams"][0]
    return int(s["width"]), int(s["height"]), float(d["format"]["duration"])


def parse_facecam(spec: str | None) -> tuple[float, float, float, float] | None:
    if not spec:
        return None
    if spec in FACECAM_PRESETS:
        return FACECAM_PRESETS[spec]
    try:
        x, y, w, h = (float(v) for v in spec.split(","))
    except ValueError:
        raise SystemExit(f"bad --facecam '{spec}': use {', '.join(FACECAM_PRESETS)} or x,y,w,h")
    if not (0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 and 0 < h <= 1):
        raise SystemExit("--facecam fractions must be between 0 and 1")
    return x, y, w, h


TOP_H = 576  # facecam band: 30% of 1920 (gameplay gets the other 70%)


def _gameplay_x(src_w: int, crop_w: int, cam) -> int:
    """Left edge of the gameplay crop: as central as possible, avoiding the facecam."""
    centre = (src_w - crop_w) // 2
    if not cam:
        return centre // 2 * 2
    cx0, _, cx1, _ = cam
    best, best_cost = centre, None
    for x in range(0, src_w - crop_w + 1, 8):
        overlap = max(0, min(x + crop_w, cx1) - max(x, cx0))
        cost = overlap * 4 + abs(x - centre)
        if best_cost is None or cost < best_cost:
            best, best_cost = x, cost
    return best // 2 * 2


def auto_layout_filter(src_w: int, src_h: int, info):
    """30% streamer facecam on top, 70% vertical gameplay below.
    Without a facecam: full-height vertical gameplay. Returns (filter, caption y, credit y)."""
    import facecam
    if info is None:
        gw = int(src_h * W / H) // 2 * 2
        gx = (src_w - gw) // 2 // 2 * 2
        return f"[0:v]crop={gw}:{src_h}:{gx}:0,scale={W}:{H}[base]", 1340, 70
    cx, cy, cw, ch = facecam.top_band_crop(info, W / TOP_H)
    bot_h = H - TOP_H
    gw = int(src_h * W / bot_h) // 2 * 2
    gx = _gameplay_x(src_w, gw, info["cam"])
    f = (f"[0:v]split=2[cam][game];"
         f"[cam]crop={cw}:{ch}:{cx}:{cy},scale={W}:{TOP_H}[top];"
         f"[game]crop={gw}:{src_h}:{gx}:0,scale={W}:{bot_h}[bot];"
         f"[top][bot]vstack,drawbox=x=0:y={TOP_H - 3}:w={W}:h=6:color=black@0.9:t=fill[base]")
    return f, 1380, TOP_H + 22


IRL_H = 1536  # IRL: the stream fills 80% of the phone screen, in the middle


def irl_layout_filter(src_w: int, src_h: int, info):
    """IRL / camera streams: the stream fills the middle 80% of the short (cropped around
    the streamer, or the centre if no face), with a blurred copy filling the top and bottom
    10%. Returns (filter, caption y, credit y)."""
    gw = min(int(src_h * W / IRL_H) // 2 * 2, src_w // 2 * 2)
    if info:
        fx, _, fw, _ = info["face"]
        gx = int(min(max(fx + fw / 2 - gw / 2, 0), src_w - gw))
    else:
        gx = (src_w - gw) // 2
    gx = gx // 2 * 2
    top = (H - IRL_H) // 2
    f = (f"[0:v]crop={gw}:{src_h}:{gx}:0,split=2[a][b];"
         # blur a small copy then scale up: looks the same, much cheaper
         f"[a]scale={W // 4}:{H // 4}:force_original_aspect_ratio=increase,crop={W // 4}:{H // 4},"
         f"boxblur=8:2,eq=brightness=-0.15,scale={W}:{H}[bg];"
         f"[b]scale={W}:{IRL_H}[fg];"
         f"[bg][fg]overlay=0:{top}[base]")
    return f, top + IRL_H - 300, top // 2 - 28


def looks_like_irl(info) -> bool:
    """A big face with no facecam overlay around it means the camera IS the stream."""
    return bool(info) and info["cam"] is None and info["face"][3] / info["size"][1] > 0.15


def layout_filter(layout: str, src_w: int, src_h: int, facecam, zoom: float):
    """Returns (filter that makes [base] from [0:v], caption centre y)."""
    if layout == "split":
        fx, fy, fw, fh = facecam
        cx, cy = int(fx * src_w), int(fy * src_h)
        cw, ch = int(fw * src_w) // 2 * 2, int(fh * src_h) // 2 * 2
        top_h = 640
        bot_h = H - top_h
        game_w = int(src_h * W / bot_h) // 2 * 2  # width of gameplay crop for 1080x1280
        f = (f"[0:v]split=2[cam][game];"
             f"[cam]crop={cw}:{ch}:{cx}:{cy},scale={W}:{top_h}:force_original_aspect_ratio=increase,"
             f"crop={W}:{top_h}[top];"
             f"[game]crop={game_w}:{src_h}:(iw-{game_w})/2:0,scale={W}:{bot_h}[bot];"
             f"[top][bot]vstack[base]")
        return f, top_h + 170
    # blur: zoomed gameplay in the middle, blurred copy filling the background
    fg_w = int(src_w / zoom) // 2 * 2
    f = (f"[0:v]split=2[a][b];"
         # blur a small copy then scale up: looks the same, ~10x cheaper
         f"[a]scale={W // 4}:{H // 4}:force_original_aspect_ratio=increase,crop={W // 4}:{H // 4},"
         f"boxblur=8:2,eq=brightness=-0.10,scale={W}:{H}[bg];"
         f"[b]crop={fg_w}:ih:(iw-{fg_w})/2:0,scale={W}:-2[fg];"
         f"[bg][fg]overlay=(W-w)/2:(H-h)/2-80[base]")
    fg_h = int(src_h * W / fg_w)
    return f, H // 2 - 80 + fg_h // 2 + 190


def make_short(video: Path, out: Path | None = None, layout: str = "auto", facecam=None,
               captions: bool = True, do_censor: bool = True, credit: str | None = None,
               zoom: float = 1.15, log=print, face_hint: dict | None = None,
               start: float = 0.0, length: float | None = None,
               words: list | None = None, identity=None, hook: str | None = None,
               peak: float | None = None, shake: bool = False, effects: bool = True) -> Path:
    """layout: auto (find the streamer's face: 30% facecam / 70% gameplay; switches to irl
    by itself when the camera is the whole stream), irl (stream fills the middle 80%), blur, split.
    face_hint: a previous detection for this streamer, used if none is found this time.
    start/length: use only that part of the video. words: an existing transcript of the
    whole video (seconds from its start), so it isn't transcribed twice.
    hook: big text for the first 2.3 s. peak: seconds into the short where the reaction
    peaks (punch-in zoom + boom there; shake too if shake=True)."""
    video = Path(video)
    if layout == "split" and facecam is None:
        raise ValueError("split layout needs a facecam position")
    out = Path(out) if out else video.parent / "shorts" / f"{video.stem}_short.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    if credit is None:
        credit = _guess_channel(video)
        credit = f"twitch.tv/{credit}" if credit else ""
    src_w, src_h, full_duration = probe(video)
    start = max(0.0, min(start, full_duration - 1))
    duration = min(length or full_duration, full_duration - start)
    credit_y = 70
    face_info = None
    if layout in ("auto", "irl"):
        try:
            import facecam as fc
            face_info = fc.detect(video, identity=identity)
        except Exception as e:
            log(f"face detection failed: {e}")
        if (face_info and face_info["cam"] is None and face_hint and face_hint.get("cam")
                and tuple(face_hint.get("size", ())) == (src_w, src_h)
                and (identity is None or face_hint.get("matched"))):
            # same streamer, same layout as last time: reuse the facecam box we found then
            x0, y0, x1, y1 = face_hint["cam"]
            fx, fy, fw, fh = face_info["face"]
            if x0 <= fx + fw / 2 <= x1 and y0 <= fy + fh / 2 <= y1:
                face_info["cam"] = face_hint["cam"]
        if layout == "auto" and looks_like_irl(face_info):
            log("camera fills the stream, using IRL layout (80% middle)")
            layout = "irl"
    if layout == "irl":
        base_filter, cap_y, credit_y = irl_layout_filter(src_w, src_h, face_info)
    elif layout == "auto":
        usable_hint = (face_hint and tuple(face_hint.get("size", ())) == (src_w, src_h)
                       and (identity is None or face_hint.get("matched")))
        if face_info is None and usable_hint:
            face_info = face_hint
            log("streamer not clearly visible in this clip, reusing their last facecam position")
        elif face_info is None:
            log("the streamer isn't on camera in this clip, using full-screen gameplay"
                if identity is not None else "no streamer face found, using full-screen gameplay")
        base_filter, cap_y, credit_y = auto_layout_filter(src_w, src_h, face_info)
    else:
        base_filter, cap_y = layout_filter(layout, src_w, src_h, facecam, zoom)

    with tempfile.TemporaryDirectory(prefix="short_") as tmp:
        tmp = Path(tmp)
        if not captions:
            words = []
        elif words is None:
            try:
                words = transcribe(video, tmp)
            except Exception as e:
                log(f"captions skipped: {e}")
                words = []
        # shift to the part we're using
        words = [Word(w.start - start, w.end - start, w.text) for w in words
                 if w.end > start and w.start < start + duration]
        track = build_caption_track(words, duration, cap_y, credit, do_censor, tmp, credit_y)
        filt = base_filter
        if effects and peak is not None and 0.5 < peak < duration - 0.3:
            # punch-in zoom on the reaction (up to +14%, eases in and out), shake on screams
            z = f"(1+0.14*max(0,1-abs(t-{peak:.2f})/0.9))"
            sh = f"9*sin(t*57)*between(t,{peak - 0.15:.2f},{peak + 0.5:.2f})" if shake else "0"
            sv = f"7*cos(t*49)*between(t,{peak - 0.15:.2f},{peak + 0.5:.2f})" if shake else "0"
            filt = filt.replace("[base]", "[pre]") + (
                f";[pre]scale=w='trunc({W}*{z}/2)*2':h='trunc({H}*{z}/2)*2':eval=frame,"
                f"crop={W}:{H}:x='(in_w-{W})/2+{sh}':y='(in_h-{H})/2+{sv}'[base]")
        filt += ";[1:v]format=rgba[cap];[base][cap]overlay=0:0:eof_action=pass[ov]"
        inputs = ["-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(video),
                  "-f", "concat", "-safe", "0", "-i", str(track)]
        n_in = 2
        if hook and effects:
            hook_png = tmp / "hook.png"
            render_hook(hook, min(H - 300, credit_y + 150)).save(hook_png)
            inputs += ["-loop", "1", "-t", "2.4", "-i", str(hook_png)]
            filt += f";[{n_in}:v]format=rgba[hk];[ov][hk]overlay=0:0:enable='lt(t,2.3)':eof_action=pass[out]"
            n_in += 1
        else:
            filt += ";[ov]null[out]"
        sfx = ensure_sfx() if effects else None
        audio_map = ["-map", "0:a:0?"]
        if sfx and has_audio(video):
            boom, whoosh = sfx
            mix = ["[0:a]aresample=48000,aformat=channel_layouts=stereo[m]"]
            parts = ["[m]"]
            if peak is not None and 0.5 < peak < duration - 0.3:
                inputs += ["-i", str(boom)]
                d_ms = int(max(0, peak - 0.08) * 1000)
                mix.append(f"[{n_in}:a]adelay={d_ms}|{d_ms},volume=0.75[b]")
                parts.append("[b]")
                n_in += 1
            if hook:
                inputs += ["-i", str(whoosh)]
                mix.append(f"[{n_in}:a]volume=0.55[w]")
                parts.append("[w]")
                n_in += 1
            if len(parts) > 1:
                mix.append("".join(parts) + f"amix=inputs={len(parts)}:normalize=0:duration=first,alimiter=limit=0.95[aout]")
                filt += ";" + ";".join(mix)
                audio_map = ["-map", "[aout]"]
        cmd = [FFMPEG, "-v", "error", "-y", *inputs,
               "-filter_complex", filt, "-map", "[out]", *audio_map,
               "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-t", f"{duration:.3f}"]
        enc_hw = ["-c:v", platformkit.video_encoder(), "-b:v", "10M", "-maxrate", "14M", "-bufsize", "20M",
                  "-pix_fmt", "yuv420p"]
        enc_sw = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]
        r = subprocess.run(cmd + enc_hw + [str(out)], capture_output=True, text=True)
        if r.returncode != 0:
            r = subprocess.run(cmd + enc_sw + [str(out)], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {r.stderr.strip()[-400:]}")
    what = "IRL, 80% middle" if layout == "irl" else ("facecam found" if face_info else "no facecam")
    log(f"short saved -> {out}  ({len(words)} caption words, {what})")
    make_short.last_face = face_info if layout == "auto" else None
    return out


def has_audio(video: Path) -> bool:
    r = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                        "-of", "csv=p=0", str(video)], capture_output=True, text=True)
    return bool(r.stdout.strip())


def _guess_channel(video: Path) -> str | None:
    side = video.with_suffix(".json")
    if side.exists():
        try:
            return json.loads(side.read_text()).get("channel")
        except (ValueError, OSError):
            pass
    return None


def save_grid_frame(video: Path, at: float = 5.0) -> Path:
    """Save a still with a 10% grid so you can read off facecam x,y,w,h fractions."""
    out = video.with_name(video.stem + "_grid.jpg")
    subprocess.run([FFMPEG, "-v", "error", "-y", "-ss", str(at), "-i", str(video),
                    "-frames:v", "1", str(out)], check=True)
    img = Image.open(out).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _font(max(14, img.width // 60))
    for i in range(1, 10):
        x, y = img.width * i // 10, img.height * i // 10
        draw.line([(x, 0), (x, img.height)], fill=(255, 0, 120), width=2)
        draw.line([(0, y), (img.width, y)], fill=(255, 0, 120), width=2)
        draw.text((x + 4, 4), f"{i / 10:.1f}", font=font, fill=(255, 255, 0), stroke_width=2, stroke_fill=(0, 0, 0))
        draw.text((4, y + 4), f"{i / 10:.1f}", font=font, fill=(255, 255, 0), stroke_width=2, stroke_fill=(0, 0, 0))
    img.save(out, quality=90)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Make vertical shorts with captions from clips.")
    p.add_argument("clips", nargs="+", help="clip .mp4 files")
    p.add_argument("--layout", choices=["auto", "irl", "blur", "split"], default="auto",
                   help="auto = finds the streamer's face: facecam top 30%%, gameplay 70%%; "
                        "irl = stream fills the middle 80%% (camera streams); blur = gameplay in the middle; "
                        "split = facecam position you give")
    p.add_argument("--facecam", help=f"for split: {', '.join(FACECAM_PRESETS)} or x,y,w,h fractions")
    p.add_argument("--zoom", type=float, default=1.15, help="blur layout: zoom into gameplay")
    p.add_argument("--no-captions", action="store_true")
    p.add_argument("--no-censor", action="store_true", help="don't mask swear words in captions")
    p.add_argument("--credit", help="text at the top (default: twitch.tv/<channel>, '' for none)")
    p.add_argument("--frame", action="store_true",
                   help="just save a still with a grid to help find the facecam position")
    args = p.parse_args()

    for c in args.clips:
        clip = Path(c)
        if args.frame:
            print(f"grid frame -> {save_grid_frame(clip)}")
            continue
        facecam = parse_facecam(args.facecam)
        if args.layout == "split" and not facecam:
            raise SystemExit("--layout split needs --facecam (try --frame first to find it)")
        make_short(clip, layout=args.layout, facecam=facecam, captions=not args.no_captions,
                   do_censor=not args.no_censor, credit=args.credit, zoom=args.zoom)


if __name__ == "__main__":
    main()


def make_duo(a: Path, a_off: float, b: Path, b_off: float, dur: float, names: tuple[str, str],
             out: Path, hook: str = "SAME MOMENT, TWO POVS") -> Path:
    """Two streamers reacting to the same moment: one on top, one below (vertical 9:16)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    half = H // 2
    with tempfile.TemporaryDirectory(prefix="duo_") as tmp:
        tmp = Path(tmp)
        labels = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(labels)
        for (y, n) in ((24, names[0]), (half + 24, names[1])):
            font = _font(52)
            tw = d.textlength(n, font=font)
            d.rounded_rectangle([24, y, 24 + tw + 44, y + 72], 16, fill=(11, 6, 24, 200))
            d.text((46, y + 8), n, font=font, fill=(255, 255, 255), stroke_width=5, stroke_fill=(0, 0, 0))
        d.rectangle([0, half - 4, W, half + 4], fill=(0, 0, 0, 255))
        labels.save(tmp / "labels.png")
        render_hook(hook, half - 90).save(tmp / "hook.png")
        tile = (f"scale={W}:{half}:force_original_aspect_ratio=increase,crop={W}:{half},setsar=1")
        filt = (f"[0:v]{tile}[t];[1:v]{tile}[b];[t][b]vstack[s];[s][2:v]overlay=0:0[l];"
                f"[3:v]format=rgba[h];[l][h]overlay=0:0:enable='lt(t,2.3)':eof_action=pass,format=yuv420p[v];"
                f"[0:a]aresample=48000,aformat=channel_layouts=stereo,volume=0.8[aa];"
                f"[1:a]aresample=48000,aformat=channel_layouts=stereo,volume=0.8[ab];"
                f"[aa][ab]amix=inputs=2:normalize=0:duration=shortest,loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000[a]")
        cmd = [FFMPEG, "-v", "error", "-y", "-ss", f"{a_off:.2f}", "-t", f"{dur:.2f}", "-i", str(a),
               "-ss", f"{b_off:.2f}", "-t", f"{dur:.2f}", "-i", str(b), "-i", str(tmp / "labels.png"),
               "-loop", "1", "-t", "2.4", "-i", str(tmp / "hook.png"),
               "-filter_complex", filt, "-map", "[v]", "-map", "[a]", "-t", f"{dur:.2f}",
               "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart"]
        enc = ["-c:v", platformkit.video_encoder(), "-b:v", "10M", "-pix_fmt", "yuv420p"]
        r = subprocess.run(cmd + enc + [str(out)], capture_output=True, text=True)
        if r.returncode != 0:
            r = subprocess.run(cmd + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                                      str(out)], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {r.stderr.strip()[-300:]}")
    return out
