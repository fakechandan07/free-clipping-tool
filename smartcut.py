"""Pick where a clip should start and end so it has the whole story.

The clipper saves a wide window around a hype moment (about a minute before chat
exploded, and a bit after it calmed down). This module uses the transcript of
that window to cut it at natural points:

- start at a pause in speech before the action (the setup: "yo watch this..."),
  instead of in the middle of a sentence
- end once the reaction is over and the streamer finishes their sentence

All times are seconds from the start of the wide clip.
"""

from __future__ import annotations


def choose_window(words, spike_start: float, spike_end: float, peak: float, duration: float,
                  pre: float = 22.0, min_len: float = 15.0, max_len: float = 75.0):
    """words: objects with .start, .end, .text (seconds). Returns (start, end)."""
    s0 = max(0.0, min(spike_start, duration))
    # Chat reacts 3-10 s after the moment, so the action starts well before s0.
    lo, hi = max(0.0, s0 - 45.0), max(0.0, s0 - 8.0)
    target = min(max(s0 - pre, lo), hi)

    # Places a clip can start cleanly: just before someone starts talking after a
    # pause, or at the start of a sentence.
    cands = [(0.0, 0.5)] if lo <= 0.0 else []
    for i, w in enumerate(words):
        prev = words[i - 1] if i else None
        gap = w.start - prev.end if prev else w.start
        sentence = prev is None or prev.text.rstrip()[-1:] in ".?!"
        if gap >= 0.6 or sentence:
            t = max(0.0, w.start - min(0.35, gap / 2))
            cands.append((t, min(gap, 2.0) + (0.6 if sentence else 0.0)))
    usable = [(t, strength) for t, strength in cands if lo <= t <= hi]
    if usable:
        # close to the target, but a long pause / clean sentence start is worth moving for
        start = min(usable, key=lambda c: abs(c[0] - target) - 3.0 * c[1])[0]
    else:
        start = target

    # End: after the reaction (chat calming down / a few seconds after the peak), then
    # let the streamer finish the sentence they're in, up to 8 s more.
    end = min(duration, max(spike_end + 2.0, peak + 6.0, start + min_len))
    for i, w in enumerate(words):
        if w.end < end - 0.05:
            continue
        nxt = words[i + 1].start if i + 1 < len(words) else duration
        if w.start > end + 8.0:
            break
        if nxt - w.end >= 0.5:
            end = min(duration, max(end, w.end + 0.4))
            break
    end = min(end, duration)

    if end - start > max_len:  # very long merged moment: keep the end, trim the setup
        later = [t for t, _ in cands if end - max_len <= t <= hi]
        start = min(later) if later else end - max_len
    if end - start < min_len:
        start = max(0.0, end - min_len)
    return round(start, 2), round(end, 2)
