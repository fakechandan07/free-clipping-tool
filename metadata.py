"""Upload-ready text for every clip: title, description, hashtags, keywords, and the
hook text burned into the first seconds of a short. Template-based (free, offline),
using what the streamer said, how chat/the streamer reacted, and the game.
"""

from __future__ import annotations

import re

GAME_SHORT = {
    "grand theft auto vi": ("GTA 6", ["gta6", "gtavi", "gta"]),
    "grand theft auto v": ("GTA 5", ["gta5", "gtav", "gta", "gtarp"]),
    "grand theft auto": ("GTA", ["gta"]),
    "fortnite": ("Fortnite", ["fortnite"]),
    "minecraft": ("Minecraft", ["minecraft"]),
    "valorant": ("Valorant", ["valorant"]),
    "counter-strike": ("CS2", ["cs2", "counterstrike"]),
    "world of warcraft": ("WoW", ["worldofwarcraft", "wow"]),
    "call of duty": ("COD", ["callofduty", "cod"]),
    "rob run": ("Rob Run", ["robrun"]),
}
IRL_WORDS = ("just chatting", "irl", "people & blogs", "music", "talk shows", "travel")

TITLES = {
    "cops": ["{s} vs the COPS 🚔", "{s} tried to outrun the cops 💀", "The cops showed up on {s} 🚨"],
    "heist": ["{s}'s heist went WRONG 💰", "{s} pulled off the heist 💰", "{s} robbed the place 😳"],
    "driving": ["{s} can NOT drive 💀🚗", "Worst driver ever: {s} 🚗", "{s} crashed out 💥"],
    "mission": ["{s}'s first mission went crazy 😳", "{s} on a mission 🎯"],
    "money": ["{s} just got RICH 💰", "{s} spent it ALL 💸"],
    "guns": ["{s} went full action movie 🔫", "{s} started shooting 😳"],
    "death": ["{s} got WASTED 💀", "{s} did not survive that 💀"],
    "funny": ["{s} couldn't stop laughing 😂", "{s} was CRYING laughing 😂", "This had {s} dying 😂"],
    "crazy": ["{s} completely lost it 😱", "{s} could NOT believe it 😱", "{s} freaked out 😳"],
    "hype": ["Chat went CRAZY for this {s} moment 😳", "{s} broke chat 🔥", "{s}'s chat exploded 🔥"],
    "first": ["{s}'s first minutes 👀", "{s} tries it for the first time 👀"],
    "viewer": ["Everyone clipped this {s} moment 👀"],
    "moment": ["{s} did NOT see this coming 😳", "Pure chaos with {s} 🔥", "{s}'s wildest moment 🔥"],
}
HOOKS = {
    "cops": ["THE COPS SHOWED UP", "RUN FROM THE COPS"],
    "heist": ["THIS HEIST WENT WRONG", "HEIST TIME"],
    "driving": ["WORST DRIVER EVER", "WATCH THIS DRIVING"],
    "mission": ["FIRST MISSION", "MISSION GONE WRONG"],
    "money": ["THAT'S A LOT OF MONEY", "INSTANT MILLIONAIRE"],
    "guns": ["IT GOT VIOLENT", "ACTION MOVIE MOMENT"],
    "death": ["THIS DID NOT END WELL", "WASTED"],
    "funny": ["CAN'T STOP LAUGHING", "THIS IS SO FUNNY"],
    "crazy": ["COMPLETELY LOST IT", "NOBODY EXPECTED THIS"],
    "hype": ["CHAT WENT CRAZY", "CHAT LOST IT"],
    "first": ["FIRST TIME PLAYING", "FIRST IMPRESSIONS"],
    "viewer": ["EVERYONE CLIPPED THIS", "WAIT FOR IT"],
    "moment": ["WAIT FOR IT", "DID NOT SEE THAT COMING"],
}
CATEGORY_TAGS = {
    "cops": ["cops", "police", "chase"], "heist": ["heist", "robbery"], "driving": ["driving", "carcrash"],
    "mission": ["mission"], "money": ["money"], "guns": ["action"], "death": ["fail"],
    "funny": ["funny", "funnymoments", "lol"], "crazy": ["reaction", "insane"], "hype": ["hype", "viral"],
    "first": ["firsttime", "firstimpressions"], "viewer": ["viral"], "moment": ["funnymoments"],
}


def _clean(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", name or "")


def game_of(category: str | None) -> tuple[str | None, list[str]]:
    c = (category or "").strip().lower()
    if not c or any(w in c for w in IRL_WORDS) or c in ("gaming", "entertainment"):
        return None, []
    for key, val in GAME_SHORT.items():
        if c.startswith(key):
            return val
    return category, [_clean(category).lower()] if _clean(category) else []


def classify(info: dict, text: str) -> str:
    """cops / heist / driving / ... / funny / crazy / hype / moment, like long videos use."""
    from longform import RULES
    best, best_n = None, 0
    for cat, rx, _ in RULES:
        n = len(re.findall(rx, text or "", re.I))
        if n > best_n:
            best, best_n = cat, n
    reasons = " ".join(info.get("reasons") or [])
    if info.get("kind") == "viewer":
        return "viewer"
    if best and best_n >= 2:
        return best
    if "laugh" in reasons:
        return "funny"
    if (info.get("peak_ratio") or 0) >= 4:
        return "hype"
    if "scream" in reasons or (info.get("audio_boost") or 0) >= 10:
        return "crazy"
    if best:
        return best
    return "moment"


def _pick(options: list[str], seed: str) -> str:
    return options[sum(map(ord, seed or "x")) % len(options)]


def for_short(info: dict, said: str = "") -> dict:
    """Title, hook, description, hashtags and keywords for a short."""
    credit = info.get("credit") or info.get("channel") or ""
    name = info.get("display") or credit.split("/")[-1].lstrip("@") or info.get("channel") or "Streamer"
    cat = classify(info, said)
    game, game_tags = game_of(info.get("category"))
    seed = info.get("file") or info.get("moment_time") or name
    title = _pick(TITLES.get(cat, TITLES["moment"]), seed).format(s=name)
    if game:
        title = f"{title} | {game}"
    hook = _pick(HOOKS.get(cat, HOOKS["moment"]), seed + "h")
    platform = info.get("platform") or "twitch"
    tags = ["shorts", _clean(name).lower()] + game_tags + CATEGORY_TAGS.get(cat, []) + [
        platform, "streamer", "clips", "gaming" if game else "irl"]
    tags = list(dict.fromkeys(t for t in tags if t))[:12]
    quote = re.sub(r"\s+", " ", said or "").strip()
    quote = (quote[:140] + "…") if len(quote) > 140 else quote
    where = info.get("source_title")
    when = info.get("vod_time_text")
    lines = [title, ""]
    if quote:
        lines += [f"\"{quote}\"", ""]
    lines.append(f"🎮 Streamer: {credit}")
    if where:
        lines.append(f"📺 From: {where}" + (f" (at {when})" if when else ""))
    lines += ["", "All credit to " + name + " – go follow and support them!", "",
              " ".join("#" + t for t in tags)]
    keywords = list(dict.fromkeys([name, f"{name} clips", f"{name} funny moments", f"{name} {game}" if game else "",
                                   f"{game} funny moments" if game else "", f"{game} clips" if game else "",
                                   "funny moments", "streamer clips", f"{platform} clips",
                                   cat if cat != "moment" else "", "shorts"]))
    return {"title": title[:100], "hook": hook, "category": cat, "description": "\n".join(lines),
            "hashtags": ["#" + t for t in tags], "keywords": [k for k in keywords if k]}


def for_long(topic: str, names: list[str], categories: list[str | None], chapters_text: str,
             credits: list[str], titles: list[str]) -> dict:
    games = [g for g in (game_of(c)[0] for c in categories) if g]
    game_tags = [t for c in categories for t in game_of(c)[1]]
    # from the topic: the game it names (GTA 6 -> #gta6 #gtavi #gta) and the whole topic as one tag,
    # never loose words like #first or #day
    low = topic.lower()
    topic_tags = []
    for key, val in GAME_SHORT.items():
        if re.search(r"\b" + re.escape(key).replace(r"\ ", r"\s*"), low):
            topic_tags += val[1]
            break
    if re.search(r"\bgta\s*(6|vi)\b", low):
        topic_tags = ["gta6", "gtavi", "gta"] + topic_tags
    whole = re.sub(r"[^a-z0-9]", "", low)
    if 3 < len(whole) <= 20:
        topic_tags.append(whole)
    tags = list(dict.fromkeys([_clean(n).lower() for n in names] + game_tags + topic_tags +
                              ["funnymoments", "compilation", "streamers", "gaming"]))[:15]
    desc = (f"{topic}\n\n{chapters_text}\n\nStreamers in this video (go follow them!):\n" +
            "\n".join(f"- {c}" for c in credits) +
            "\n\nAll clips credited to their creators.\n\n" + " ".join("#" + t for t in tags))
    keywords = list(dict.fromkeys([topic] + [f"{n} funny moments" for n in names] +
                                  [f"{g} funny moments" for g in games] + [f"{g} first time" for g in games] +
                                  names + ["streamer compilation", "funny moments", "best moments"]))
    return {"title": titles[0] if titles else topic, "titles": titles, "description": desc,
            "hashtags": ["#" + t for t in tags], "keywords": keywords[:20]}
