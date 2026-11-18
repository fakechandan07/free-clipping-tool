"""Find the streamer's facecam in a clip, automatically.

How: sample frames across the clip, run OpenCV's YuNet face detector, and keep
the face that stays in the same spot in most frames (the webcam doesn't move,
game characters do). Then look for the facecam overlay's straight borders around
that face, using edges that stay put across frames.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)  # hide harmless warnings

MODEL = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"
WORK_W = 960  # analyse at this width; fast and plenty accurate


def _frames(video: Path, n: int):
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    if total <= 0 or src_w <= 0:
        cap.release()
        return frames, (src_w, src_h)
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (i + 0.5) / n))
        ok, f = cap.read()
        if ok:
            scale = WORK_W / f.shape[1]
            frames.append(cv2.resize(f, (WORK_W, int(f.shape[0] * scale))))
    cap.release()
    return frames, (src_w, src_h)


def _faces(frames):
    if not frames:
        return []
    h, w = frames[0].shape[:2]
    det = cv2.FaceDetectorYN.create(str(MODEL), "", (w, h), score_threshold=0.75)
    out = []  # (frame index, x, y, w, h)
    for i, f in enumerate(frames):
        _, faces = det.detect(f)
        for row in faces if faces is not None else []:
            x, y, fw, fh = (float(v) for v in row[:4])
            if fw >= 14 and fh >= 14:
                out.append((i, x, y, fw, fh))
    return out


def _cluster(dets):
    """Group detections that are the same face in the same place."""
    clusters = []
    for d in dets:
        _, x, y, w, h = d
        cx, cy = x + w / 2, y + h / 2
        for c in clusters:
            mx, my, mw = c["cx"], c["cy"], c["w"]
            if abs(cx - mx) < 0.6 * mw and abs(cy - my) < 0.6 * mw and 0.6 < w / mw < 1.6:
                c["dets"].append(d)
                arr = np.array([(e[1] + e[3] / 2, e[2] + e[4] / 2, e[3]) for e in c["dets"]])
                c["cx"], c["cy"], c["w"] = np.median(arr, axis=0)
                break
        else:
            clusters.append({"cx": cx, "cy": cy, "w": w, "dets": [d]})
    return clusters


def _border(profile, lo, hi, prefer_far: bool, frame_edge: int | None):
    """Strongest persistent edge in profile[lo:hi], or the frame edge, or None."""
    lo, hi = max(0, int(lo)), min(len(profile), int(hi))
    if hi - lo < 4:
        return frame_edge
    seg = profile[lo:hi]
    peak = int(np.argmax(seg))
    base = float(np.median(seg)) + 1e-6
    if seg[peak] > 3.0 * base and seg[peak] > 12:
        return lo + peak
    return frame_edge


def _streamer_dets(frames, identity):
    """Detections of the streamer only (face recognition), ignoring everyone else:
    game characters, skins, teammates' cams, guests."""
    import identity as ident
    out = []
    for i, f in enumerate(frames):
        for (x, y, w, h), e in ident.faces(f, min_size=14):
            if ident.same(e, identity) >= ident.SAME_PERSON:
                out.append((i, x, y, w, h))
    return out


def detect(video: Path, samples: int = 12, identity=None) -> dict | None:
    """Returns {"face": (x,y,w,h), "cam": (x0,y0,x1,y1) or None, "size": (W,H)}
    in source-video pixels, or None if no steady face was found.

    identity: the streamer's face (identity.py). When given, only that person's face
    counts, so a game character or a teammate's webcam is never mistaken for the
    streamer; if the streamer isn't on camera in this clip, the result is None."""
    if not MODEL.exists():
        return None
    frames, (src_w, src_h) = _frames(Path(video), samples)
    if len(frames) < 3:
        return None
    matched = False
    if identity is not None:
        try:
            dets = _streamer_dets(frames, identity)
            matched, need = True, 0.25  # their face may be turned away in some frames
        except Exception:
            dets, need = _faces(frames), 0.4
    else:
        dets, need = _faces(frames), 0.4
    best, best_score = None, 0.0
    for c in _cluster(dets):
        hits = len({d[0] for d in c["dets"]})
        presence = hits / len(frames)
        centers = np.array([(d[1] + d[3] / 2, d[2] + d[4] / 2) for d in c["dets"]])
        wobble = float(np.mean(np.std(centers, axis=0))) / max(c["w"], 1)
        score = presence * (1.0 if wobble < 0.35 else 0.5)
        if presence >= need and score > best_score:
            best, best_score = c, score
    if best is None:
        return None

    arr = np.array([d[1:] for d in best["dets"]])
    fx, fy, fw, fh = np.median(arr, axis=0)

    # Facecam borders: edges that persist across frames (overlay frame edges).
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]
    gx = np.median([np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)) for g in gray], axis=0)
    gy = np.median([np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)) for g in gray], axis=0)
    H, W = gray[0].shape
    rows = slice(int(max(0, fy - 0.3 * fh)), int(min(H, fy + 1.3 * fh)))
    cols = slice(int(max(0, fx - 0.3 * fw)), int(min(W, fx + 1.3 * fw)))
    col_prof = gx[rows].mean(axis=0)
    row_prof = gy[:, cols].mean(axis=1)
    reach_x, reach_y = 8.0 * fw, 3.5 * fh  # streamers often sit far back from the cam
    left = _border(col_prof, fx - reach_x, fx - 0.25 * fw, True, 0 if fx - reach_x <= 0 else None)
    right = _border(col_prof, fx + 1.25 * fw, fx + fw + reach_x, False,
                    W if fx + fw + reach_x >= W else None)
    top = _border(row_prof, fy - reach_y, fy - 0.2 * fh, True, 0 if fy - reach_y <= 0 else None)
    bottom = _border(row_prof, fy + 1.2 * fh, fy + fh + reach_y, False,
                     H if fy + fh + reach_y >= H else None)

    # One edge missing (e.g. the cam's bottom blends into a desk): infer it from the
    # usual 16:9 webcam shape.
    sides = [left, right, top, bottom]
    if sides.count(None) == 1:
        if left is None and right is not None and top is not None and bottom is not None:
            left = max(0.0, right - (bottom - top) * 16 / 9)
        elif right is None and left is not None and top is not None and bottom is not None:
            right = min(float(W), left + (bottom - top) * 16 / 9)
        elif top is None and left is not None and right is not None and bottom is not None:
            top = max(0.0, bottom - (right - left) * 9 / 16)
        elif bottom is None and left is not None and right is not None and top is not None:
            bottom = min(float(H), top + (right - left) * 9 / 16)
    k = src_w / W
    cam = None
    if None not in (left, right, top, bottom):
        cw, ch = right - left, bottom - top
        # sanity: a facecam is bigger than the face and not most of the screen
        if 1.5 * fw <= cw <= 0.75 * W and 1.3 * fh <= ch <= 0.85 * H:
            cam = (left * k, top * k, right * k, bottom * k)
    return {"face": (fx * k, fy * k, fw * k, fh * k), "cam": cam, "size": (src_w, src_h), "matched": matched}


def top_band_crop(info: dict, aspect: float) -> tuple[int, int, int, int]:
    """Crop (x, y, w, h) in source pixels for the facecam band, centred on the face,
    kept inside the facecam overlay when its borders were found."""
    fx, fy, fw, fh = info["face"]
    src_w, src_h = info["size"]
    bx0, by0, bx1, by1 = info["cam"] or (0, 0, src_w, src_h)
    if info["cam"]:  # stay a little inside the overlay's edges (borders, name tags)
        ix, iy = 0.03 * (bx1 - bx0), 0.04 * (by1 - by0)
        bx0, by0, bx1, by1 = bx0 + ix, by0 + iy, bx1 - ix, by1 - iy
    ch = 3.4 * fh  # head, shoulders and some room; shrunk below if it'd leave the facecam
    cw = ch * aspect
    if info["cam"]:  # zoom in if needed so the crop stays inside the facecam
        max_w, max_h = bx1 - bx0, by1 - by0
        if cw > max_w:
            cw, ch = max_w, max_w / aspect
        if ch > max_h:
            ch, cw = max_h, max_h * aspect
        if ch < 1.2 * fh:  # never cut the face itself
            ch, cw = 1.2 * fh, 1.2 * fh * aspect
    cw, ch = min(cw, src_w), min(ch, src_h)
    cx = fx + fw / 2 - cw / 2
    cy = fy + fh * 0.45 - ch / 2  # eyes slightly above the middle looks natural
    cx = min(max(cx, bx0), max(bx0, bx1 - cw))
    cy = min(max(cy, by0), max(by0, by1 - ch))
    cx = min(max(cx, 0), src_w - cw)
    cy = min(max(cy, 0), src_h - ch)
    even = lambda v: int(v) // 2 * 2
    return even(cx), even(cy), max(even(cw), 2), max(even(ch), 2)


if __name__ == "__main__":
    import sys
    for v in sys.argv[1:]:
        print(v, detect(Path(v)))
