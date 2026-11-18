"""Hype detection.

Pure logic, no network or video: feed it chat messages, the streamer's audio
loudness and chat events (raids, gift bombs) with timestamps, and it tells you
when something worth clipping happened. Kept separate so it can be tested offline.
"""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter, deque
from dataclasses import dataclass, field

# Emotes and words that signal a hype/funny/shocking moment, with extra weight.
# Matching is case-insensitive.
HYPE_TOKENS = {
    # laughing
    "kekw": 1.5, "omegalul": 1.5, "lul": 1.0, "lulw": 1.2, "icant": 1.5,
    "lmao": 1.2, "lmfao": 1.5, "kekl": 1.2, "pepelaugh": 1.2, "xd": 0.6,
    "😂": 1.0, "💀": 1.2, "🤣": 1.0,
    # hype / shock
    "pog": 1.2, "pogu": 1.2, "poggers": 1.2, "pogchamp": 1.2, "w": 0.8,
    "wtf": 1.2, "omg": 1.0, "monkas": 1.0, "monkaw": 1.2, "aware": 0.8,
    "holy": 0.8, "insane": 1.0, "clean": 0.6, "ez": 0.6, "goated": 1.0,
    "ratio": 0.6, "l": 0.6, "noway": 1.0, "nahh": 0.8, "nah": 0.6,
    "😱": 1.0, "🔥": 0.8,
}
LAUGH_RE = re.compile(r"^(?:l+o+l+|lm+f*a+o+|a*h+a+(?:h+a+)+h*|k+e+k+w*|x+d+)$", re.I)
CLIP_RE = re.compile(r"\bclip\s*(it|that|this)?\b", re.I)
TOKEN_RE = re.compile(r"\S+")

# Chat bots and commands aren't people reacting, so they're ignored.
BOT_NAMES = {"nightbot", "streamelements", "moobot", "fossabot", "streamlabs", "wizebot",
             "sery_bot", "botrixoficial", "soundalerts", "kofistreambot", "pokemoncommunitygame"}
STOPWORDS = set("""a an and the to of in on for is it i you he she we they this that was
are be at me my so do just what with not no yes but or if im its his her your our get got
like u ur oh as all can have has had will would from up out him them there here""".split())


def is_noise(user: str, text: str) -> bool:
    u = user.lower()
    return u in BOT_NAMES or u.endswith("bot") or text.lstrip().startswith("!")


def message_weight(text: str) -> float:
    """How much hype one chat message carries. A plain message is 1.0."""
    bonus = 0.0
    tokens = TOKEN_RE.findall(text)
    for tok in tokens[:12]:
        t = tok.lower().strip("!?.,")
        if t in HYPE_TOKENS:
            bonus += HYPE_TOKENS[t]
        elif LAUGH_RE.match(t):
            bonus += 1.0
    if CLIP_RE.search(text):
        bonus += 3.0  # chat literally asking for a clip
    letters = [c for c in text if c.isalpha()]
    if len(letters) >= 4 and sum(c.isupper() for c in letters) / len(letters) > 0.7:
        bonus += 0.5  # ALL CAPS
    return 1.0 + min(bonus, 4.0)


@dataclass
class Moment:
    """A detected hype moment. Times are unix wall-clock seconds."""
    spike_start: float
    peak_time: float
    spike_end: float
    peak_ratio: float
    messages: int
    chatters: int
    top_tokens: list = field(default_factory=list)
    sample_messages: list = field(default_factory=list)
    kind: str = "auto"  # auto | test | manual | viewer
    score: int = 0  # 0-100, how good the moment probably is (for "best first")
    audio_boost: float = 0.0  # how many dB louder than normal the streamer got
    clip_requests: int = 0  # "clip it" messages
    merged: int = 1  # how many back-to-back spikes were joined into this moment
    viewer_clips: int = 0  # Twitch clips viewers made of this moment
    reaction: float = 0.0  # 0-1: how hard the streamer reacted (laughing, screaming, face)
    reasons: list = field(default_factory=list)  # e.g. ["laughing", "big smile"]


@dataclass
class _Msg:
    ts: float
    user: str
    weight: float
    text: str


def score_moment(peak_ratio: float, chatters: int, avg_weight: float, clip_requests: int,
                 audio_boost: float, viewer_clips: int = 0, reaction: float = 0.0) -> int:
    """0-100: chat explosion + crowd size + emote intensity + 'clip it' + streamer
    loudness + viewers making their own clips + the streamer's own reaction."""
    chat = min(40.0, 20 * math.log2(max(peak_ratio, 1.0)))
    crowd = min(20.0, chatters / 2)
    emotes = min(15.0, max(0.0, avg_weight - 1) * 10)
    asks = min(10.0, clip_requests * 2)
    loud = min(15.0, max(0.0, audio_boost - 3) * 2)
    viewers = min(25.0, viewer_clips * 12)
    react = min(25.0, max(0.0, reaction - 0.2) * 35)
    return int(round(min(100.0, chat + crowd + emotes + asks + loud + viewers + react)))


class HypeDetector:
    """Watches chat speed (weighted by hype emotes) against a rolling baseline,
    boosted by the streamer suddenly getting loud.

    A spike starts when the last `window` seconds of weighted chat is
    `threshold` times the normal rate, with enough messages from enough
    different people, or when the streamer screams/laughs and chat reacts too.
    It ends when chat calms down. Spikes close together are merged into one
    moment so the clip tells the whole story. Raids and gift bombs are ignored.
    """

    def __init__(
        self,
        threshold: float = 3.0,
        window: float = 5.0,
        baseline_span: float = 180.0,
        min_messages: int = 8,
        min_chatters: int = 5,
        warmup: float = 90.0,
        cooldown: float = 5.0,
        max_spike: float = 45.0,
        merge_gap: float = 20.0,
        max_merged: float = 75.0,
        loud_db: float = 8.0,
        confirm: float = 1.5,
        react_level: float = 0.6,
    ):
        self.threshold = threshold
        self.window = window
        self.baseline_span = baseline_span
        self.min_messages = min_messages
        self.min_chatters = min_chatters
        self.warmup = warmup
        self.cooldown = cooldown
        self.max_spike = max_spike
        self.merge_gap = merge_gap
        self.max_merged = max_merged
        self.loud_db = loud_db
        self.confirm = confirm
        self.react_level = react_level  # streamer reaction strong enough to clip without chat
        self.reactions_on = True
        self.last_reaction = 0.0

        self._msgs: deque[_Msg] = deque()
        self._audio: deque[tuple[float, float]] = deque()  # (time, loudness dB)
        self._first_ts: float | None = None
        self._cooldown_until = 0.0
        self._suppress_until = 0.0
        self._active: dict | None = None
        self._hot_since: float | None = None  # hype must last a moment, not be a random burst
        self._held: Moment | None = None  # finished spike waiting to see if another follows
        self._held_until = 0.0
        self.last_ratio = 0.0
        self.last_baseline = 0.0
        self.last_audio_boost = 0.0
        self.suppressed_reason: str | None = None

    # -- input ---------------------------------------------------------------
    def feed(self, ts: float, user: str, text: str) -> None:
        if is_noise(user, text):
            return
        if self._first_ts is None:
            self._first_ts = ts
        self._msgs.append(_Msg(ts, user.lower(), message_weight(text), text))

    def feed_audio(self, ts: float, db: float) -> None:
        """Streamer audio loudness (dBFS) for a short slice of the stream."""
        self._audio.append((ts, db))

    def feed_event(self, ts: float, kind: str, size: int = 0) -> None:
        """Chat events that make chat explode for reasons that aren't a moment."""
        if kind == "raid" and size >= 10:
            self._suppress(ts + 75, f"raid ({size} viewers)")
        elif kind == "giftbomb" and size >= 5:
            self._suppress(ts + 30, f"{size} gifted subs")

    def _suppress(self, until: float, reason: str) -> None:
        if until > self._suppress_until:
            self._suppress_until, self.suppressed_reason = until, reason
        self._active = None  # whatever spike was building is the raid/gifts, not a moment

    # -- core ----------------------------------------------------------------
    def _baseline(self, now: float) -> float:
        """Typical weighted chat volume per `window`, robust to spikes (median)."""
        start = now - self.baseline_span
        n_buckets = int(self.baseline_span // self.window)
        buckets = [0.0] * n_buckets
        for m in self._msgs:
            if m.ts < start or m.ts >= now - self.window:
                continue
            i = int((m.ts - start) // self.window)
            if 0 <= i < n_buckets:
                buckets[i] += m.weight
        # Only count the part of history we have actually observed.
        observed_from = max(start, self._first_ts or now)
        first_bucket = int((observed_from - start) // self.window)
        used = buckets[first_bucket:-1] or [0.0]
        med = statistics.median(used)
        mean = statistics.fmean(used)
        # Blend: median ignores spikes, mean keeps very bursty chats honest.
        return max(0.7 * med + 0.3 * mean, 2.0)

    def _audio_boost(self, now: float) -> float:
        """dB the streamer is above their normal talking loudness, over the last `window`
        (1-second averages, so a single click doesn't count)."""
        while self._audio and self._audio[0][0] < now - self.baseline_span:
            self._audio.popleft()
        past = [(t, db) for t, db in self._audio if t <= now]
        if len(past) < 40:
            return 0.0
        # "Normal" = the streamer's usual talking level (80th percentile), not the median,
        # which quiet pauses drag down. Measured on real streams: regular talking stays
        # under +6 dB of this; yelling/laughing goes +8 to +15.
        normal = statistics.quantiles([db for _, db in past], n=5)[3]
        recent = [db for t, db in past if t >= now - self.window]
        if len(recent) < 4:
            return 0.0
        smooth = [statistics.fmean(recent[i:i + 4]) for i in range(0, len(recent) - 3)]
        return max(0.0, max(smooth) - normal)

    def _window_stats(self, now: float):
        msgs = [m for m in self._msgs if now - self.window <= m.ts <= now]
        weight = sum(m.weight for m in msgs)
        return msgs, weight, len({m.user for m in msgs})

    def tick(self, now: float, reaction: float = 0.0, why: list | None = None) -> Moment | None:
        """Call regularly (e.g. every 0.5s). `reaction` (0-1) is how hard the streamer is
        reacting right now (reactions.py). Returns a Moment once a spike (or a run of
        back-to-back spikes) is over."""
        horizon = now - self.baseline_span - self.max_merged - self.merge_gap - 5
        while self._msgs and self._msgs[0].ts < horizon:
            self._msgs.popleft()

        baseline = self._baseline(now)
        msgs, weight, chatters = self._window_stats(now)
        ratio = weight / baseline
        boost = self._audio_boost(now)
        self.last_ratio, self.last_baseline, self.last_audio_boost = ratio, baseline, boost
        if not self.reactions_on:
            reaction, why = 0.0, None
        self.last_reaction = reaction

        warmed_up = self._first_ts is not None and now - self._first_ts >= self.warmup
        chat_hot = (ratio >= self.threshold and len(msgs) >= self.min_messages
                    and chatters >= self.min_chatters)
        # Streamer screams/laughs AND chat clearly reacts: counts even in slower chats.
        loud_hot = (boost >= self.loud_db and ratio >= self.threshold * 0.75
                    and len(msgs) >= max(3, self.min_messages // 2)
                    and chatters >= max(2, self.min_chatters // 2))
        # The streamer bursts out laughing / screams / looks shocked: clip-worthy even if
        # chat is slow. Loudness confirms a slightly weaker reaction.
        react_hot = reaction >= self.react_level or (reaction >= self.react_level - 0.15
                                                     and boost >= self.loud_db)
        hot_now = chat_hot or loud_hot or react_hot
        if hot_now:
            self._hot_since = self._hot_since or now
        else:
            self._hot_since = None
        hot = hot_now and now - self._hot_since >= self.confirm
        suppressed = now < self._suppress_until
        if not suppressed:
            self.suppressed_reason = None

        if self._active is None:
            if warmed_up and hot and not suppressed and now >= self._cooldown_until:
                self._active = {"start": self._hot_since - self.window, "peak_time": now, "peak_ratio": ratio,
                                "peak_boost": boost, "calm_since": None, "peak_react": reaction,
                                "why": set(why or [])}
                if self._held is not None:  # continues the story of the last spike
                    held, self._held = self._held, None
                    self._active.update(start=held.spike_start, merged=held.merged + 1)
                    if held.peak_ratio > ratio:
                        self._active.update(peak_ratio=held.peak_ratio, peak_time=held.peak_time)
                    self._active["peak_boost"] = max(boost, held.audio_boost)
                    self._active["peak_react"] = max(reaction, held.reaction)
                    self._active["why"] |= set(held.reasons)
                return None
            if self._held is not None and now >= self._held_until:
                held, self._held = self._held, None
                self._cooldown_until = now + self.cooldown
                return held
            return None

        a = self._active
        if ratio > a["peak_ratio"]:
            a["peak_ratio"], a["peak_time"] = ratio, now
        a["peak_boost"] = max(a["peak_boost"], boost)
        a["peak_react"] = max(a["peak_react"], reaction)
        if reaction >= self.react_level - 0.15:
            a["why"] |= set(why or [])
        calm = (ratio < max(self.threshold * 0.5, 1.5) and boost < self.loud_db * 0.6
                and reaction < self.react_level * 0.6)
        a["calm_since"] = (a["calm_since"] or now) if calm else None
        too_long = now - a["start"] >= (self.max_merged if a.get("merged") else self.max_spike)
        if (a["calm_since"] and now - a["calm_since"] >= 3.0) or too_long:
            m = self._finish(now)
            if m.spike_end - m.spike_start >= self.max_merged - 1:
                self._cooldown_until = now + self.cooldown
                return m  # already as long as a clip should be
            self._held, self._held_until = m, now + self.merge_gap
        return None

    def flush(self) -> Moment | None:
        """Hand over a held moment right away (e.g. when shutting down)."""
        held, self._held = self._held, None
        return held

    def _finish(self, now: float) -> Moment:
        a = self._active
        self._active = None
        end = now
        spike = [m for m in self._msgs if a["start"] <= m.ts <= end]
        tokens = Counter()
        clip_requests = 0
        for m in spike:
            if CLIP_RE.search(m.text):
                clip_requests += 1
            for tok in set(TOKEN_RE.findall(m.text.lower())):
                tok = tok.strip("!?.,:;\"'()[]")
                if tok and len(tok) <= 25 and tok not in STOPWORDS and not tok.startswith("@"):
                    tokens[tok] += 1
        texts = Counter(m.text.strip() for m in spike)
        chatters = len({m.user for m in spike})
        avg_w = statistics.fmean(m.weight for m in spike) if spike else 1.0
        boost = round(a["peak_boost"], 1)
        react = round(a.get("peak_react", 0.0), 2)
        return Moment(
            spike_start=a["start"],
            peak_time=a["peak_time"],
            spike_end=end,
            peak_ratio=round(a["peak_ratio"], 2),
            messages=len(spike),
            chatters=chatters,
            top_tokens=tokens.most_common(8),
            sample_messages=[t for t, _ in texts.most_common(10)],
            score=score_moment(a["peak_ratio"], chatters, avg_w, clip_requests, boost, reaction=react),
            audio_boost=boost,
            reaction=react,
            reasons=sorted(a.get("why", ())),
            clip_requests=clip_requests,
            merged=a.get("merged", 1),
        )

    @property
    def in_spike(self) -> bool:
        return self._active is not None
