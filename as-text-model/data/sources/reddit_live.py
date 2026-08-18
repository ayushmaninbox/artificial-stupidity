"""Live Reddit comments from named subreddits, via Reddit's official API.

Why OAuth and not something easier: we tried the easier things.
  - reddit.com/r/X/comments.json  -> 403 to unauthenticated clients
  - reddit.com/r/X/comments/.rss  -> also blocked now
  - PullPush (Pushshift successor) -> 429, it's a paid service for scrapers
  - HF Reddit dumps -> real, but so long-tail that r/ksi shows up roughly
    once per 15,000 rows, which is useless for targeting a specific community

So this is the only route that actually reaches creator subreddits by name.
It's free and takes about two minutes to set up:

  1. https://www.reddit.com/prefs/apps -> "create another app..."
  2. pick type "script", put http://localhost:8080 as the redirect URI
  3. export the two values before running the collector:

       export REDDIT_CLIENT_ID=<the string under the app name>
       export REDDIT_CLIENT_SECRET=<the "secret" field>

Without those set this source skips itself and the rest of the corpus is
collected as normal.
"""

import base64
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

from .clean import Dedupe, clean_line

TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
API = "https://oauth.reddit.com"
UA = "macos:artificial-stupidity:0.1 (personal research project)"

# creator communities — the specific dialect we're after
CREATOR_SUBS = [
    "ksi", "sidemen", "PewdiepieSubmissions", "dannygonzalez", "DrewGooden",
    "h3h3productions", "youtubedrama", "LivestreamFail", "Twitch",
    "MrBeast", "jacksepticeye", "Markiplier", "GameGrumps", "northernlion",
    "jerma985", "moistcr1tikal", "LudwigAhgren", "Hasan_Piker",
]

# communities that are structurally stupid on purpose
CHAOS_SUBS = [
    "okbuddyretard", "ihadastroke", "shittyaskscience", "AskOuija",
    "copypasta", "greentext", "dankmemes", "memes", "teenagers",
    "funny", "cursedcomments", "BrandNewSentence", "NoStupidQuestions",
    "AskReddit", "Showerthoughts", "unpopularopinion", "comedyheaven",
]

DEFAULT_SUBS = CREATOR_SUBS + CHAOS_SUBS


def get_token() -> str | None:
    cid = os.environ.get("REDDIT_CLIENT_ID")
    secret = os.environ.get("REDDIT_CLIENT_SECRET")
    if not (cid and secret):
        return None

    body = urllib.parse.urlencode({"grant_type": "client_credentials"}).encode()
    auth = base64.b64encode(f"{cid}:{secret}".encode()).decode()
    req = urllib.request.Request(TOKEN_URL, data=body, headers={
        "Authorization": f"Basic {auth}",
        "User-Agent": UA,
        "Content-Type": "application/x-www-form-urlencoded",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode()).get("access_token")
    except Exception as e:
        print(f"      [!] reddit auth failed: {str(e)[:100]}", flush=True)
        return None


def fetch(token: str, path: str, params: dict, retries: int = 2):
    url = f"{API}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {token}", "User-Agent": UA,
    })
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8", errors="ignore"))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(3 * (attempt + 1))
                continue
            return None
        except Exception:
            return None
    return None


def collect(sink, subreddits=None, pages_per_sub=10, delay=1.1, verbose=True):
    """Fill `sink` with comments from named subreddits. No-op without creds."""
    token = get_token()
    if not token:
        print("      [skip] REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET not set.", flush=True)
        print("             see the docstring in data/sources/reddit_live.py "
              "— takes 2 min", flush=True)
        return

    subs = subreddits or DEFAULT_SUBS
    dedupe = Dedupe(allow_repeats=2)
    per_sub = sink.budget / max(1, len(subs))

    for i, sub in enumerate(subs):
        if sink.full:
            break
        target = min(sink.budget, per_sub * (i + 1))
        after, got = None, 0

        for _ in range(pages_per_sub):
            if sink.written >= target or sink.full:
                break
            params = {"limit": 100, "raw_json": 1}
            if after:
                params["after"] = after
            data = fetch(token, f"/r/{sub}/comments", params)
            if not data or "data" not in data:
                break
            children = data["data"].get("children", [])
            if not children:
                break
            after = data["data"].get("after")

            for child in children:
                body = (child.get("data") or {}).get("body", "")
                if not body or body in ("[deleted]", "[removed]"):
                    continue
                for chunk in body.split("\n"):
                    line = clean_line(chunk, max_len=180)
                    if line and dedupe.ok(line):
                        got += 1
                        if not sink.write(line):
                            break
            if not after:
                break
            time.sleep(delay)

        if verbose:
            print(f"      r/{sub:<22} {f'+{got:>6,} lines' if got else '  (none)'}"
                  f"   [{sink.written / 1e6:.1f} MB]", flush=True)
