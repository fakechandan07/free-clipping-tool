"""Optional: see the clips viewers make on Twitch (free Twitch developer keys).

When several viewers make their own Twitch clip of the same moment, that's strong
proof it was good. The clipper uses this to boost the score of clips it already
made, and to catch moments chat didn't react to strongly.

Needs a Client ID + Client Secret from https://dev.twitch.tv/console/apps (free).
They're stored only on this Mac, in clips/settings.json.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def _ssl_context():
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


class TwitchAPI:
    def __init__(self, settings_file: Path):
        self.file = Path(settings_file)
        self.lock = threading.Lock()
        self.client_id = self.client_secret = ""
        self._token, self._token_exp = "", 0.0
        self._ids: dict[str, str] = {}
        self.last_error: str | None = None
        try:
            data = json.loads(self.file.read_text())
            self.client_id = data.get("twitch_client_id", "")
            self.client_secret = data.get("twitch_client_secret", "")
        except (OSError, ValueError):
            pass

    @property
    def configured(self) -> bool:
        return bool(self.client_id and self.client_secret)

    # -- settings ------------------------------------------------------------
    def save(self, client_id: str, client_secret: str) -> tuple[bool, str]:
        client_id, client_secret = client_id.strip(), client_secret.strip()
        if not client_id or not client_secret:
            return False, "Paste both the Client ID and the Client Secret"
        old = (self.client_id, self.client_secret, self._token, self._token_exp)
        with self.lock:
            self.client_id, self.client_secret, self._token, self._token_exp = client_id, client_secret, "", 0.0
        try:
            self._get_token()
        except Exception as e:
            with self.lock:
                self.client_id, self.client_secret, self._token, self._token_exp = old
            return False, f"Twitch didn't accept those keys ({e})"
        self._write()
        self.last_error = None
        return True, "Connected to Twitch. Viewer clips are on."

    def clear(self) -> None:
        with self.lock:
            self.client_id = self.client_secret = self._token = ""
            self._token_exp = 0.0
        self._write()

    def _write(self) -> None:
        data = {}
        try:
            data = json.loads(self.file.read_text())
        except (OSError, ValueError):
            pass
        data.update(twitch_client_id=self.client_id, twitch_client_secret=self.client_secret)
        self.file.parent.mkdir(parents=True, exist_ok=True)
        self.file.write_text(json.dumps(data, indent=2))
        os.chmod(self.file, 0o600)  # only your user account can read it

    # -- API -----------------------------------------------------------------
    def _request(self, url: str, data: bytes | None = None, headers: dict | None = None) -> dict:
        req = urllib.request.Request(url, data=data, headers=headers or {})
        with urllib.request.urlopen(req, timeout=15, context=_ssl_context()) as r:
            return json.loads(r.read())

    def _get_token(self) -> str:
        with self.lock:
            if self._token and time.time() < self._token_exp - 60:
                return self._token
            body = urllib.parse.urlencode({"client_id": self.client_id, "client_secret": self.client_secret,
                                           "grant_type": "client_credentials"}).encode()
        try:
            d = self._request("https://id.twitch.tv/oauth2/token", data=body)
        except urllib.error.HTTPError as e:
            raise RuntimeError("wrong Client ID or Secret" if e.code in (400, 401, 403) else f"HTTP {e.code}")
        with self.lock:
            self._token = d["access_token"]
            self._token_exp = time.time() + float(d.get("expires_in", 3600))
            return self._token

    def _get(self, path: str, params: dict) -> dict:
        for attempt in (1, 2):
            token = self._get_token()
            url = f"https://api.twitch.tv/helix/{path}?{urllib.parse.urlencode(params)}"
            try:
                return self._request(url, headers={"Client-Id": self.client_id,
                                                   "Authorization": f"Bearer {token}"})
            except urllib.error.HTTPError as e:
                if e.code == 401 and attempt == 1:
                    with self.lock:
                        self._token_exp = 0.0  # expired: get a new token and retry once
                    continue
                raise RuntimeError(f"Twitch API error {e.code}")
        return {}

    def user_id(self, login: str) -> str | None:
        if login not in self._ids:
            data = self._get("users", {"login": login}).get("data") or []
            if not data:
                return None
            self._ids[login] = data[0]["id"]
        return self._ids[login]

    def clips_since(self, login: str, since: float) -> list[dict]:
        """Clips viewers made of this channel since `since` (unix time)."""
        uid = self.user_id(login)
        if not uid:
            return []
        started = datetime.fromtimestamp(since, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        data = self._get("clips", {"broadcaster_id": uid, "started_at": started, "first": 50}).get("data") or []
        out = []
        for c in data:
            try:
                created = datetime.fromisoformat(c["created_at"].replace("Z", "+00:00")).timestamp()
            except (KeyError, ValueError):
                continue
            out.append({"id": c.get("id"), "created": created, "views": c.get("view_count", 0),
                        "title": c.get("title", ""), "url": c.get("url", ""),
                        "duration": c.get("duration", 30)})
        return out
