"""Live Twitch chat, scraped from public Justlog instances.

Justlog (https://github.com/gempir/justlog) is a chat logging service; several
public instances index the large channels. This is by far the best source we
have for the exact register we want — thousands of humans reacting to the same
thing in real time, in five words or fewer.

API shape:
    GET /channels                     -> {"channels": [{"name", "userID"}]}
    GET /list?channel=NAME            -> {"availableLogs": [{"year","month","day"}]}
    GET /channel/NAME/Y/M/D?json      -> {"messages": [{"text","username",...}]}
"""

import json
import random
import re
import time
import urllib.error
import urllib.request

from .clean import Dedupe, clean_line, is_bot

# "@someone you're wrong" is a genuine reply, and it's the only reply structure
# Twitch chat gives us. Worth mining: it's real humans answering real humans.
REPLY_RE = re.compile(r"^@(\w+)[,:]?\s+(.+)$")

INSTANCES = [
    "https://logs.ivr.fi",
    "https://logs.zonian.dev",
]

# big, loud, high-volume chats
DEFAULT_CHANNELS = [
    "xqc", "forsen", "sodapoppin", "hasanabi", "mizkif", "nmplol",
    "pokelawls", "summit1g", "lirik", "tarik", "jerma985", "vedal987",
    "caedrel", "erobb221", "nymn", "esfandtv", "moistcr1tikal", "zackrawrr",
    "ludwig", "shroud", "loltyler1", "kaicenat", "ironmouse", "sykkuno",
]

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) artificial-stupidity/0.1"


def _get(url: str, timeout: int = 25):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", errors="ignore"))


def available_channels(instance: str) -> set[str]:
    try:
        return {c["name"].lower() for c in _get(f"{instance}/channels").get("channels", [])}
    except Exception:
        return set()


def available_days(instance: str, channel: str) -> list[tuple[str, str, str]]:
    try:
        logs = _get(f"{instance}/list?channel={channel}").get("availableLogs", [])
    except Exception:
        return []
    return [(l["year"], l["month"], l["day"]) for l in logs if "day" in l]


def fetch_day(instance: str, channel: str, y, m, d) -> list[dict]:
    try:
        return _get(f"{instance}/channel/{channel}/{y}/{m}/{d}?json").get("messages", [])
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return []


def collect(sink, channels=None, days_per_channel=6, delay=0.4, verbose=True):
    """Fill `sink` with cleaned Twitch chat lines."""
    channels = channels or DEFAULT_CHANNELS
    dedupe = Dedupe(allow_repeats=4)

    # find which instance actually has each channel
    rosters = {inst: available_channels(inst) for inst in INSTANCES}
    plan = []
    for ch in channels:
        for inst in INSTANCES:
            if ch.lower() in rosters.get(inst, ()):
                plan.append((inst, ch))
                break
        else:
            if verbose:
                print(f"      (no instance logs #{ch}, skipping)")

    if not plan:
        print("      [!] no channels available on any known Justlog instance")
        return

    random.shuffle(plan)
    for inst, ch in plan:
        if sink.full:
            break
        days = available_days(inst, ch)
        if not days:
            continue
        # newest days first, they're the most complete
        picked = days[:days_per_channel]
        got = pairs = 0
        for (y, m, d) in picked:
            if sink.full:
                break
            # remember what each user last said, so "@them ..." can be paired
            last_by_user: dict[str, str] = {}
            for msg in fetch_day(inst, ch, y, m, d):
                user = msg.get("username")
                if is_bot(user):
                    continue
                text = msg.get("text", "")

                reply = REPLY_RE.match(text)
                if reply:
                    target, body = reply.group(1).lower(), reply.group(2)
                    prompt = last_by_user.get(target)
                    q = clean_line(prompt, max_len=120) if prompt else None
                    a = clean_line(body, max_len=120)
                    if q and a and dedupe.ok(q + a):
                        pairs += 1
                        if not (sink.write(f"A: {q}") and sink.write(f"B: {a}")):
                            break
                        sink.write("")
                        continue

                line = clean_line(text, max_len=120)
                if line and dedupe.ok(line):
                    got += 1
                    if not sink.write(line):
                        break
                if user and line:
                    last_by_user[user.lower()] = text
            time.sleep(delay)
        if verbose and (got or pairs):
            print(f"      #{ch:<16} +{got:>7,} lines  +{pairs:>5,} pairs"
                  f"   [{sink.written / 1e6:.1f} MB]", flush=True)
