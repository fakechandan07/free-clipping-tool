"""Offline tests for the hype detector using simulated chat.  Run: python test_hype.py"""

import random

from hype import HypeDetector, message_weight

NORMAL = ["hi chat", "what game is this", "lol", "where is he going", "nice car",
          "gm", "is this gta 6??", "first time here", "o7", "chat is slow today"]
HYPE = ["KEKW", "OMEGALUL OMEGALUL", "CLIP IT", "LMAOOOO", "NO WAY", "💀💀💀",
        "WHAT WAS THAT", "KEKW KEKW KEKW", "clip that", "ICANT"]


def simulate(detector, seconds, rate_fn, msg_pool_fn, rng, t0=1_000_000.0):
    """Feed messages second by second; return detected moments."""
    moments = []
    t = t0
    while t < t0 + seconds:
        for _ in range(_poisson(rng, rate_fn(t - t0))):
            ts = t + rng.random()
            detector.feed(ts, f"user{rng.randint(1, 5000)}", rng.choice(msg_pool_fn(t - t0)))
        for half in (0.5, 1.0):
            m = detector.tick(t + half)
            if m:
                moments.append(m)
        t += 1
    return moments


def _poisson(rng, lam):
    # Knuth's algorithm; fine for small lambdas
    import math
    if lam <= 0:
        return 0
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= L:
            return k - 1


def test_weights():
    assert message_weight("hello") == 1.0
    assert message_weight("KEKW") > 2
    assert message_weight("CLIP IT") >= 4
    assert message_weight("lmaooooo") > 1.5
    assert message_weight("hahahaha") > 1.5


def test_quiet_chat_no_clips():
    rng = random.Random(1)
    d = HypeDetector()
    moments = simulate(d, 600, lambda t: 3, lambda t: NORMAL, rng)
    assert moments == [], f"false positives in steady chat: {len(moments)}"


def test_spike_detected_once():
    rng = random.Random(2)
    d = HypeDetector()
    spike = (300, 320)
    rate = lambda t: 25 if spike[0] <= t < spike[1] else 3
    pool = lambda t: HYPE if spike[0] <= t < spike[1] else NORMAL
    moments = simulate(d, 500, rate, pool, rng)
    assert len(moments) == 1, f"expected 1 moment, got {len(moments)}"
    m = moments[0]
    start_offset = m.spike_start - 1_000_000.0
    assert spike[0] - 6 <= start_offset <= spike[0] + 3, start_offset
    assert m.peak_ratio >= 3


def test_two_separate_spikes():
    rng = random.Random(3)
    d = HypeDetector()
    spikes = [(200, 215), (400, 412)]
    hot = lambda t: any(a <= t < b for a, b in spikes)
    moments = simulate(d, 550, lambda t: 30 if hot(t) else 4,
                       lambda t: HYPE if hot(t) else NORMAL, rng)
    assert len(moments) == 2, f"expected 2 moments, got {len(moments)}"


def test_busy_chat_needs_bigger_spike():
    # A huge chat (40 msg/s) getting 20% busier is not a moment.
    rng = random.Random(4)
    d = HypeDetector()
    moments = simulate(d, 400, lambda t: 48 if 200 <= t < 220 else 40, lambda t: NORMAL, rng)
    assert moments == [], "small bump in a busy chat should not clip"


def test_bots_and_commands_ignored():
    d = HypeDetector()
    d.feed(1.0, "Nightbot", "KEKW KEKW")
    d.feed(1.0, "someuser", "!sens")
    d.feed(1.0, "coolbot", "hi")
    assert len(d._msgs) == 0
    d.feed(1.0, "someuser", "KEKW")
    assert len(d._msgs) == 1


def test_top_tokens_skip_filler():
    rng = random.Random(6)
    d = HypeDetector()
    hot = lambda t: 300 <= t < 315
    pool = lambda t: ["the KEKW is a", "and KEKW to the", "OMEGALUL a"] if hot(t) else NORMAL
    moments = simulate(d, 400, lambda t: 25 if hot(t) else 3, pool, rng)
    assert moments and moments[0].top_tokens[0][0] == "kekw", moments[0].top_tokens
    assert not any(tok in ("the", "a", "and", "to") for tok, _ in moments[0].top_tokens)


def test_warmup_blocks_early_spike():
    rng = random.Random(5)
    d = HypeDetector(warmup=45)
    moments = simulate(d, 40, lambda t: 30 if t > 20 else 3,
                       lambda t: HYPE if t > 20 else NORMAL, rng)
    assert moments == []


def _with_audio(d, seconds, loud_fn, t0=1_000_000.0):
    """Feed 4 loudness readings per second: -30 dB normally, loud_fn(t) dB when loud."""
    for i in range(int(seconds * 4)):
        t = i / 4
        d.feed_audio(t0 + t, loud_fn(t))


def test_back_to_back_spikes_merge_into_one():
    rng = random.Random(7)
    d = HypeDetector()
    spikes = [(300, 312), (322, 334)]  # 10s apart: one story, one clip
    hot = lambda t: any(a <= t < b for a, b in spikes)
    moments = simulate(d, 450, lambda t: 25 if hot(t) else 3, lambda t: HYPE if hot(t) else NORMAL, rng)
    assert len(moments) == 1, f"expected 1 merged moment, got {len(moments)}"
    m = moments[0]
    assert m.merged == 2 and m.spike_end - m.spike_start > 25, (m.merged, m.spike_end - m.spike_start)


def test_raid_does_not_clip():
    rng = random.Random(8)
    d = HypeDetector()
    d._first_ts = 1_000_000.0
    raid_at = 300
    hot = lambda t: raid_at <= t < raid_at + 25
    d.feed_event(1_000_000.0 + raid_at, "raid", 500)
    moments = simulate(d, 400, lambda t: 30 if hot(t) else 3, lambda t: HYPE if hot(t) else NORMAL, rng)
    assert moments == [], "a raid flooding chat should not be clipped"


def test_streamer_scream_helps_slow_chat():
    # Slow chat (1 msg/s) gets a modest bump, not enough on its own; the streamer
    # screaming at the same time makes it a moment.
    def run(with_scream):
        rng = random.Random(9)
        d = HypeDetector(threshold=4.0)
        bump = lambda t: 300 <= t < 312  # chat ~2.6x busier: under the 4x bar on its own
        _with_audio(d, 450, lambda t: -12.0 if (with_scream and bump(t)) else -30.0)
        return simulate(d, 450, lambda t: 2.6 if bump(t) else 1, lambda t: NORMAL, rng)
    assert run(False) == [], "chat bump alone should not clip in this setup"
    got = run(True)
    assert len(got) == 1 and got[0].audio_boost >= 10, got


def test_loud_game_alone_does_not_clip():
    rng = random.Random(10)
    d = HypeDetector()
    _with_audio(d, 450, lambda t: -8.0 if 300 <= t < 320 else -30.0)  # explosions, chat quiet
    assert simulate(d, 450, lambda t: 1, lambda t: NORMAL, rng) == []


def test_bigger_moment_scores_higher():
    from hype import score_moment
    small = score_moment(3.2, 8, 1.4, 0, 0)
    big = score_moment(9.0, 60, 2.6, 6, 14)
    assert 0 <= small < big <= 100, (small, big)
    assert score_moment(3.2, 8, 1.4, 0, 0, viewer_clips=2) > small


# ---- smart start / end (smartcut) ----
class _W:
    def __init__(self, start, end, text):
        self.start, self.end, self.text = start, end, text


def _speech(segments):
    """[(start, 'some words here.'), ...] -> evenly spaced words, 0.3 s each."""
    out = []
    for t, sentence in segments:
        for i, w in enumerate(sentence.split()):
            out.append(_W(t + i * 0.3, t + i * 0.3 + 0.28, w))
    return out


def test_start_snaps_to_pause_before_setup():
    from smartcut import choose_window
    # talking continuously, with a clear pause before "yo watch this" at 31 s
    words = _speech([(20, "okay so we just need to get the car and drive"),
                     (31, "yo watch this I am going to jump the bridge"),
                     (34.5, "no no no no way")])
    start, end = choose_window(words, spike_start=55, spike_end=63, peak=58, duration=75)
    assert 30.4 <= start <= 31.0, start  # right before "yo watch this", not mid-sentence


def test_no_speech_uses_fixed_offset():
    from smartcut import choose_window
    start, end = choose_window([], spike_start=55, spike_end=60, peak=57, duration=75)
    assert start == 33 and 62 <= end <= 64, (start, end)


def test_end_lets_streamer_finish_sentence():
    from smartcut import choose_window
    words = _speech([(60, "bro I cannot believe that just happened chat what")])
    start, end = choose_window(words, spike_start=50, spike_end=58, peak=55, duration=80)
    last = words[-1].end
    assert abs(end - (last + 0.4)) < 0.05, (end, last)


def test_window_never_too_long():
    from smartcut import choose_window
    start, end = choose_window([], spike_start=50, spike_end=118, peak=100, duration=130, max_len=75)
    assert end - start <= 75.01 and end >= 118, (start, end)


# ---- streamer reactions (no chat needed) ----
def _simulate_react(d, seconds, rate_fn, react_fn, rng, t0=1_000_000.0):
    moments = []
    t = t0
    while t < t0 + seconds:
        for _ in range(_poisson(rng, rate_fn(t - t0))):
            d.feed(t + rng.random(), f"user{rng.randint(1, 5000)}", rng.choice(NORMAL))
        for half in (0.5, 1.0):
            r = react_fn(t - t0 + half)
            m = d.tick(t + half, reaction=r, why=["laughing"] if r > 0.5 else [])
            if m:
                moments.append(m)
        t += 1
    return moments


def test_streamer_laughing_clips_without_chat():
    rng = random.Random(21)
    d = HypeDetector()
    got = _simulate_react(d, 420, lambda t: 1, lambda t: 0.85 if 300 <= t < 306 else 0.05, rng)
    assert len(got) == 1, got
    assert got[0].reaction >= 0.8 and "laughing" in got[0].reasons and got[0].score >= 20, got[0]


def test_brief_reaction_blip_ignored():
    rng = random.Random(22)
    d = HypeDetector()
    got = _simulate_react(d, 420, lambda t: 1, lambda t: 0.9 if 300 <= t < 301 else 0.05, rng)
    assert got == [], "a one-second blip should not clip"


def test_reactions_can_be_turned_off():
    rng = random.Random(23)
    d = HypeDetector()
    d.reactions_on = False
    got = _simulate_react(d, 420, lambda t: 1, lambda t: 0.85 if 300 <= t < 306 else 0.05, rng)
    assert got == []


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"all {len(tests)} tests passed")
