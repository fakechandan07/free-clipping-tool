"""Editing tools that make long videos feel like a real YouTube compilation:
narrator voice, story lines, jump cuts, facecam reaction zooms, captions,
chapter cards, end screen, background music and a better thumbnail.
All free and offline (Mac voices, ffmpeg, OpenCV, the bundled face models)."""

from __future__ import annotations

import json
import random
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import platformkit  # noqa: E402
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
HERE = Path(__file__).resolve().parent
MUSIC_DIR = HERE / "music"
W, H = 1920, 1080
FFPROBE = shutil.which("ffprobe") or "ffprobe"


# --------------------------------------------------------------------------
# Narrator (Mac voices, or the built-in Windows voices)
# --------------------------------------------------------------------------
def pick_voice() -> str | None:
    return platformkit.pick_voice()


def speak(text: str, out_wav: Path, voice: str | None) -> float:
    """Narration line -> 48 kHz stereo wav. Returns its length in seconds (0 if no voice)."""
    if not text.strip() or not voice:
        return 0.0
    aiff = platformkit.say_to_file(text, voice, out_wav)
    if not aiff:
        return 0.0
    subprocess.run([FFMPEG, "-v", "error", "-y", "-i", str(aiff), "-af",
                    "aresample=48000,aformat=channel_layouts=stereo,highpass=f=80,acompressor=threshold=-18dB:ratio=3,"
                    "loudnorm=I=-14:TP=-1.5:LRA=7,aresample=48000", str(out_wav)], capture_output=True)
    aiff.unlink(missing_ok=True)
    return duration(out_wav)


def duration(p: Path) -> float:
    r = subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def time_phrase(t: float) -> str:
    if t < 150:
        return "Right at the start"
    if t < 3300:
        return f"{int(round(t / 60))} minutes in"
    h = t / 3600
    if h < 1.25:
        return "An hour in"
    return f"{int(round(h))} hours in" if abs(h - round(h)) < 0.25 else f"{int(h)} and a half hours in"


LINES = {
    "first": ["First up, {s}.", "{s} jumps in for the first time.", "Let's start with {s}."],
    "cops": ["{t}, {s} gets the attention of the cops.", "{t}, the cops show up for {s}."],
    "heist": ["{t}, {s} goes for the money.", "{t}, {s} tries to pull off a heist."],
    "driving": ["{t}, {s} gets behind the wheel. Bad idea.", "{t}, {s} hits the road."],
    "mission": ["{t}, {s} starts a mission.", "{t}, {s} figures out what to do next."],
    "money": ["{t}, {s} starts getting rich.", "{t}, {s} finds some money."],
    "guns": ["{t}, things get violent for {s}.", "{t}, {s} picks a fight."],
    "death": ["{t}, it doesn't end well for {s}.", "{t}, {s} learns the hard way."],
    "funny": ["{t}, {s} can't hold it together.", "And then {s} completely loses it.", "{t}, {s} is in tears."],
    "crazy": ["{t}, {s} did not see this coming.", "{t}, {s} can't believe it."],
    "hype": ["{t}, chat goes absolutely crazy.", "{t}, {s}'s chat explodes."],
    "moment": ["{t}, things get wild for {s}.", "Back to {s}.", "{t}, {s} keeps going."],
    "climax": ["And now, the moment that broke the stream.", "But nothing tops this."],
    "ending": ["As the stream winds down…", "And to wrap things up, {s}."],
}


def narration_line(cat: str, name: str, t: float, seed: int, climax=False, ending=False, used=None) -> str:
    """A line for this chapter; with `used`, never repeats a line (or its wording) already said."""
    key = "climax" if climax else "ending" if ending else cat
    opts = list(LINES.get(key) or LINES["moment"])
    if used is not None and not (climax or ending):
        opts += [o for o in LINES["moment"] if o not in opts]
    for i in range(len(opts)):
        tmpl = opts[(seed + i) % len(opts)]
        if used is None or tmpl not in used:
            break
    if used is not None:
        used.add(tmpl)
    return tmpl.format(s=name, t=time_phrase(t))


def intro_line(topic: str, names: list[str]) -> str:
    t = topic.strip().rstrip(".!?")
    return f"{t}. Here are the moments everyone's talking about." if t else "Here are the best moments."


OUTRO_LINE = "That's it for this one. If you enjoyed it, subscribe for more."


# --------------------------------------------------------------------------
# Jump cuts: drop dead air inside a moment, never near the big reaction
# --------------------------------------------------------------------------
def loudness(video: Path) -> np.ndarray:
    """dB per 0.25 s of the clip's audio."""
    r = subprocess.run([FFMPEG, "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
                        "-f", "s16le", "pipe:1"], capture_output=True)
    a = np.frombuffer(r.stdout[: len(r.stdout) // 2 * 2], dtype=np.int16).astype(np.float32) / 32768
    n = len(a) // 4000
    if n == 0:
        return np.zeros(0, np.float32)
    a = a[: n * 4000].reshape(n, 4000)
    return (20 * np.log10(np.sqrt((a * a).mean(axis=1)) + 1e-6)).astype(np.float32)


def jump_cuts(words, loud: np.ndarray, s: float, e: float, peak: float, min_gap: float = 1.3) -> list[tuple[float, float]]:
    """Keep-segments inside [s, e]: cut stretches where nobody talks AND it's quiet."""
    if len(loud) == 0:
        return [(s, e)]
    quiet_level = float(np.median(loud)) - 4
    spoken = [(w.start, w.end) for w in words if w.end > s and w.start < e]
    cuts, t = [], s
    step = 0.25
    gap_start = None
    while t < e:
        talking = any(a - 0.25 <= t <= b + 0.25 for a, b in spoken)
        i = int(t / step)
        quiet = i < len(loud) and loud[i] < quiet_level
        protected = peak - 3.0 <= t <= peak + 3.0
        if not talking and quiet and not protected:
            gap_start = t if gap_start is None else gap_start
        else:
            if gap_start is not None and t - gap_start >= min_gap:
                cuts.append((gap_start + 0.3, t - 0.3))  # keep a little air on both sides
            gap_start = None
        t += step
    if gap_start is not None and e - gap_start >= min_gap:
        cuts.append((gap_start + 0.3, e))
    keep, cur = [], s
    for a, b in cuts:
        if a - cur >= 0.8:
            keep.append((cur, a))
        cur = b
    if e - cur >= 0.8:
        keep.append((cur, e))
    return keep or [(s, e)]


def with_cam_zoom(keep: list[tuple[float, float]], peak: float, length: float = 2.0) -> list[tuple[float, float, str]]:
    """Mark the part right after the peak as 'cam' (cut to the streamer's face)."""
    z0, z1 = peak - 0.1, peak + length
    out = []
    for a, b in keep:
        if b <= z0 or a >= z1:
            out.append((a, b, "full"))
            continue
        if a < z0:
            out.append((a, z0, "full"))
        out.append((max(a, z0), min(b, z1), "cam"))
        if b > z1:
            out.append((z1, b, "full"))
    return [seg for seg in out if seg[1] - seg[0] >= 0.2]


def cam_crop(info: dict | None) -> tuple[int, int, int, int] | None:
    """16:9 crop around the streamer's face (inside their facecam box when known)."""
    if not info:
        return None
    fx, fy, fw, fh = info["face"]
    sw, sh = info["size"]
    ch = min(sh, 3.2 * fh)
    cw = ch * 16 / 9
    if info.get("cam"):
        x0, y0, x1, y1 = info["cam"]
        if cw > x1 - x0:
            cw = x1 - x0
            ch = cw * 9 / 16
        bx0, by0, bx1, by1 = x0, y0, x1, y1
    else:
        bx0, by0, bx1, by1 = 0, 0, sw, sh
    cw, ch = min(cw, sw), min(ch, sh)
    cx = min(max(fx + fw / 2 - cw / 2, bx0), max(bx0, bx1 - cw))
    cy = min(max(fy + fh * 0.45 - ch / 2, by0), max(by0, by1 - ch))
    cx, cy = max(0, min(cx, sw - cw)), max(0, min(cy, sh - ch))
    if cw < 120:  # too tiny to blow up to full screen
        return None
    ev = lambda v: int(v) // 2 * 2
    return ev(cx), ev(cy), ev(cw), ev(ch)


# --------------------------------------------------------------------------
# Captions for landscape video (bottom, two lines max)
# --------------------------------------------------------------------------
def caption_track(words_out: list[tuple[float, float, str]], dur: float, folder: Path, font_fn) -> Path:
    """words_out: (start, end, text) on the output timeline. Returns an ffconcat list."""
    chunks, cur = [], []
    for w in words_out:
        if cur and (len(cur) >= 7 or sum(len(x[2]) + 1 for x in cur) > 38 or w[0] - cur[-1][1] > 0.6):
            chunks.append(cur)
            cur = []
        cur.append(w)
    if cur:
        chunks.append(cur)
    blank = folder / "c_blank.png"
    Image.new("RGBA", (W, H), (0, 0, 0, 0)).save(blank)
    events = []
    for i, ch in enumerate(chunks):
        text = " ".join(x[2] for x in ch).strip()
        if not text:
            continue
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        font = font_fn(64)
        tw = d.textlength(text, font=font)
        d.text(((W - tw) / 2, H - 170), text, font=font, fill=(255, 255, 255), stroke_width=6, stroke_fill=(0, 0, 0))
        path = folder / f"c_{i:04d}.png"
        img.save(path)
        end = ch[-1][1] + 0.3
        if i + 1 < len(chunks):
            end = min(end, chunks[i + 1][0][0])
        events.append((ch[0][0], end, path))
    lines, t = ["ffconcat version 1.0"], 0.0
    for a, b, p in events:
        a, b = max(a, t), min(b, dur)
        if b <= a:
            continue
        if a - t > 0.01:
            lines += [f"file '{blank}'", f"duration {a - t:.3f}"]
        lines += [f"file '{p}'", f"duration {b - a:.3f}"]
        t = b
    if dur - t > 0.01:
        lines += [f"file '{blank}'", f"duration {dur - t:.3f}"]
    lines.append(f"file '{blank}'")
    out = folder / "captions.ffconcat"
    out.write_text("\n".join(lines) + "\n")
    return out


def map_words(words, segments: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """Move word times from the source clip onto the cut-down output timeline."""
    out, offset = [], 0.0
    for a, b, _ in segments:
        for w in words:
            if a <= w.start < b:
                out.append((offset + w.start - a, offset + min(w.end, b) - a, w.text))
        offset += b - a
    return out


# --------------------------------------------------------------------------
# Chapter card, end screen
# --------------------------------------------------------------------------
def chapter_card(frame: Path, out: Path, title: str, kicker: str, font_fn, outlined) -> Path:
    try:
        img = Image.open(frame).convert("RGB").resize((W, H))
    except OSError:
        img = Image.new("RGB", (W, H), (20, 10, 40))
    img = img.filter(ImageFilter.GaussianBlur(18))
    img = Image.blend(img, Image.new("RGB", (W, H), (11, 6, 24)), 0.45).convert("RGBA")
    d = ImageDraw.Draw(img)
    if kicker:
        outlined(d, (W // 2, H // 2 - 130), kicker.upper(), font_fn(64), fill=(255, 210, 63), stroke=6, shadow=5)
    size = 150
    while size > 70 and d.textlength(title.upper(), font=font_fn(size)) > W - 200:
        size -= 8
    outlined(d, (W // 2, H // 2 + 10), title.upper(), font_fn(size), stroke=11, shadow=9)
    d.rectangle([W // 2 - 160, H // 2 + 110, W // 2 + 160, H // 2 + 122], fill=(255, 61, 139))
    img.convert("RGB").save(out, quality=94)
    return out


def end_screen(out: Path, bg: Image.Image, channel: str, font_fn, outlined) -> Path:
    """Last 20 s laid out for YouTube's end screen: two video slots and a subscribe circle."""
    img = bg.convert("RGBA").filter(ImageFilter.GaussianBlur(6))
    d = ImageDraw.Draw(img)
    outlined(d, (W // 2, 120), "THANKS FOR WATCHING", font_fn(110), stroke=10, shadow=8)
    for x in (150, 1010):  # YouTube video elements (about 760x428 at 1080p)
        d.rounded_rectangle([x, 300, x + 760, 728], 24, outline=(255, 255, 255, 230), width=6, fill=(11, 6, 24, 140))
        outlined(d, (x + 380, 514), "▶ WATCH NEXT", font_fn(62), stroke=6, shadow=4)
    d.ellipse([W // 2 - 110, 790, W // 2 + 110, 1010], outline=(255, 255, 255, 230), width=8, fill=(11, 6, 24, 160))
    outlined(d, (W // 2, 900), "SUBSCRIBE", font_fn(40), fill=(255, 210, 63), stroke=4, shadow=3)
    outlined(d, (W // 2, 1012), channel, font_fn(44), stroke=4, shadow=3)
    img.convert("RGB").save(out, quality=94)
    return out


# --------------------------------------------------------------------------
# Background music (your own files in clipper/music), ducked under talking
# --------------------------------------------------------------------------
def music_files() -> list[Path]:
    if not MUSIC_DIR.exists():
        return []
    return sorted(p for p in MUSIC_DIR.iterdir() if p.suffix.lower() in (".mp3", ".m4a", ".wav", ".aac", ".flac", ".ogg"))


def add_music(video: Path, out: Path, total: float) -> bool:
    files = music_files()
    if not files:
        return False
    track = random.choice(files)
    filt = ("[1:a]aresample=48000,aformat=channel_layouts=stereo,volume=0.22[m];"
            "[0:a]asplit=2[v][sc];[m][sc]sidechaincompress=threshold=0.03:ratio=8:attack=20:release=400[md];"
            f"[v][md]amix=inputs=2:normalize=0:duration=first,afade=t=out:st={max(0, total - 3):.2f}:d=3[a]")
    r = subprocess.run([FFMPEG, "-v", "error", "-y", "-i", str(video), "-stream_loop", "-1", "-i", str(track),
                        "-filter_complex", filt, "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac",
                        "-b:a", "192k", "-movflags", "+faststart", str(out)], capture_output=True, text=True)
    return r.returncode == 0 and out.exists()


# --------------------------------------------------------------------------
# Thumbnail: the frame where the streamer looks most shocked / is laughing hardest
# --------------------------------------------------------------------------
THUMB_MATCH = 0.5


def best_reaction_frame(cands: list[tuple[Path, float]], identity, workdir: Path):
    """cands: (video, time) pairs. Returns (frame path, face box, score) or None."""
    import cv2
    import onnxruntime as ort
    import identity as ident
    fer_path = HERE / "models" / "emotion-ferplus-8.onnx"
    if not fer_path.exists():
        return None
    fer = ort.InferenceSession(str(fer_path), providers=["CPUExecutionProvider"])
    fin = fer.get_inputs()[0].name
    best = None
    for k, (video, t) in enumerate(cands):
        for dt in (-0.6, 0.0, 0.5, 1.0, 1.6):
            frame = workdir / f"tf_{Path(video).stem}_{t + dt:.2f}.jpg"  # unique: later calls must not overwrite it
            subprocess.run([FFMPEG, "-v", "error", "-y", "-ss", f"{max(0, t + dt):.2f}", "-i", str(video),
                            "-frames:v", "1", "-vf", "scale=1920:-2", str(frame)], capture_output=True)
            img = cv2.imread(str(frame))
            if img is None:
                continue
            for (x, y, w, h), e in ident.faces(img, min_size=40):
                # the thumbnail face must clearly be the streamer (not a game character, a
                # shark, or someone in a video they're watching): stricter than the facecam check
                sim = ident.same(e, identity) if identity is not None else 0.0
                if identity is not None and sim < THUMB_MATCH:
                    continue
                if identity is None and not (w / img.shape[1] > 0.06 and 0.75 < w / max(h, 1) < 1.35):
                    continue
                crop = cv2.cvtColor(img[int(y):int(y + h), int(x):int(x + w)], cv2.COLOR_BGR2GRAY)
                if crop.size == 0:
                    continue
                logits = fer.run(None, {fin: cv2.resize(crop, (64, 64)).astype(np.float32)[None, None]})[0][0]
                p = np.exp(logits - logits.max())
                p /= p.sum()
                score = (float(max(p[1], p[2] * 1.2)) * min(1.0, w / 140)  # happy / surprised, bigger = better
                         * (0.6 + sim if identity is not None else 0.8))
                if best is None or score > best[2]:
                    best = (frame, (x, y, w, h), score)
    return best


def thumbnail(frame: Path, face: tuple, text: str, out: Path, font_fn, outlined) -> Path:
    base = Image.open(frame).convert("RGB")
    sw, sh = base.size
    bg = base.resize((1280, 720)).convert("RGBA")
    shade = Image.new("RGBA", (1280, 720), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shade)
    for x in range(1280):  # darker on the left, where the text goes
        sd.line([(x, 0), (x, 720)], fill=(11, 6, 24, int(max(0, 200 - x * 0.28))))
    bg = Image.alpha_composite(bg, shade)
    # enlarged face in a bubble on the right
    x, y, w, h = face
    pad = 0.55 * h
    box = (max(0, x - pad), max(0, y - pad), min(sw, x + w + pad), min(sh, y + h + pad))
    face_img = base.crop(tuple(int(v) for v in box)).resize((470, 470))
    mask = Image.new("L", (470, 470), 0)
    ImageDraw.Draw(mask).ellipse([0, 0, 469, 469], fill=255)
    ring = Image.new("RGBA", (500, 500), (0, 0, 0, 0))
    ImageDraw.Draw(ring).ellipse([0, 0, 499, 499], fill=(255, 255, 255, 255))
    bg.alpha_composite(ring, (760, 120))
    bg.paste(face_img, (775, 135), mask)
    d = ImageDraw.Draw(bg)
    # arrow pointing at the face
    d.polygon([(560, 420), (700, 360), (690, 395), (760, 395), (760, 445), (690, 445), (700, 480)],
              fill=(255, 61, 139), outline=(0, 0, 0))
    words = text.upper().split()[:4]
    lines = [" ".join(words[:2]), " ".join(words[2:])] if len(words) > 2 else [" ".join(words)]
    y0 = 300 - (len(lines) - 1) * 70
    for i, line in enumerate(lines):
        size = 150
        while size > 60 and d.textlength(line, font=font_fn(size)) > 640:
            size -= 6
        outlined(d, (330, y0 + i * 150), line, font_fn(size), fill=(255, 210, 63) if i else (255, 255, 255),
                 stroke=10, shadow=8)
    bg.convert("RGB").save(out, quality=93)
    return out
