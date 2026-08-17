"""YouTube auto-caption scraping via yt-dlp.

We only ever fetch the subtitle track, never the video or audio — it's the
only part we can train on and it keeps this fast enough to be practical.

The tricky part is YouTube's auto-generated VTT format. It emits *rolling*
captions, where consecutive cues re-state the previous line plus one new one:

    00:00:03.000 --> 00:00:03.010     00:00:03.010 --> 00:00:05.000
    hello there                       my name is
    my name is                        and i am

Parsed naively you get every line three times. `parse_vtt` below deduplicates
against a short rolling window, which reconstructs the original speech.
"""

import re
import tempfile
from pathlib import Path

from .clean import Dedupe, clean_line

TAG_RE = re.compile(r"<[^>]+>")
TIMING_RE = re.compile(r"^\d{2}:\d{2}:\d{2}[.,]\d{3}\s*-->")
ENTITY = {"&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"', "&#39;": "'", "&nbsp;": " "}

# chaotic, high-talk-density channels. commentary channels first — they talk
# nonstop straight into the mic, which is the highest words-per-minute source
# of casual speech on the internet.
DEFAULT_CHANNELS = [
    # commentary / sarcasm
    "https://www.youtube.com/@drewisgooden/videos",
    "https://www.youtube.com/@Danny-Gonzalez/videos",
    "https://www.youtube.com/@2Danny2Furious/videos",
    "https://www.youtube.com/@KurtisConner/videos",
    "https://www.youtube.com/@CallMeKevin1811/videos",
    "https://www.youtube.com/@EddyBurback/videos",
    "https://www.youtube.com/@JarvisJohnson/videos",
    # british chaos
    "https://www.youtube.com/@ksi/videos",
    "https://www.youtube.com/@Sidemen/videos",
    "https://www.youtube.com/@MoreSidemen/videos",
    "https://www.youtube.com/@SidemenReacts/videos",
    "https://www.youtube.com/@ArthurTV/videos",
    "https://www.youtube.com/@calfreezy/videos",
    # legacy loud
    "https://www.youtube.com/@PewDiePie/videos",
    "https://www.youtube.com/@jacksepticeye/videos",
    "https://www.youtube.com/@markiplier/videos",
    # streamers / gaming
    "https://www.youtube.com/@jerma985/videos",
    "https://www.youtube.com/@moistcr1tikal/videos",
    "https://www.youtube.com/@Ludwig/videos",
    "https://www.youtube.com/@NorthernLion/videos",
    "https://www.youtube.com/@videogamedunkey/videos",
    "https://www.youtube.com/@RTGame/videos",
    "https://www.youtube.com/@GameGrumps/videos",
]

DEFAULT_SEARCHES = [
    "twitch stream highlights funny",
    "streamer rage compilation",
    "twitch chat reacts",
    "gaming moments compilation",
    "stream fails funny moments",
    "sidemen funny moments",
    "youtuber commentary rant",
    "try not to laugh challenge reaction",
]


def parse_vtt(path: Path, window: int = 12) -> list[str]:
    """VTT file -> spoken lines, with rolling-caption duplication removed."""
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    out: list[str] = []
    recent: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if (not line or line.startswith(("WEBVTT", "NOTE", "Kind:", "Language:", "STYLE"))
                or TIMING_RE.match(line) or line.isdigit()):
            continue
        line = TAG_RE.sub("", line)
        for k, v in ENTITY.items():
            line = line.replace(k, v)
        line = line.strip()
        if not line or line in recent:
            continue
        out.append(line)
        recent.append(line)
        if len(recent) > window:
            recent.pop(0)
    return out


def _ydl(tmpdir: Path, quiet=True):
    from yt_dlp import YoutubeDL
    return YoutubeDL({
        "skip_download": True,
        "writeautomaticsub": True,
        "writesubtitles": True,
        "subtitleslangs": ["en", "en-orig", "en-US"],
        "subtitlesformat": "vtt",
        "outtmpl": str(tmpdir / "%(id)s.%(ext)s"),
        "quiet": quiet,
        "no_warnings": True,
        "noprogress": True,
        "ignoreerrors": True,
        "extractor_args": {"youtube": {"player_client": ["web", "android"]}},
        "retries": 2,
        "socket_timeout": 30,
    })


def resolve_video_ids(targets: list[str], per_target: int, quiet=True) -> list[str]:
    """Expand channel/playlist/search targets into individual video IDs."""
    from yt_dlp import YoutubeDL

    ids: list[str] = []
    flat = YoutubeDL({
        "quiet": quiet, "no_warnings": True, "ignoreerrors": True,
        "extract_flat": "in_playlist", "playlistend": per_target,
        "socket_timeout": 30,
    })
    for target in targets:
        try:
            info = flat.extract_info(target, download=False)
        except Exception as e:
            print(f"      (failed {target}: {str(e)[:70]})")
            continue
        if not info:
            continue
        for entry in (info.get("entries") or [])[:per_target]:
            if entry and entry.get("id"):
                ids.append(entry["id"])
    # preserve order, drop dupes
    return list(dict.fromkeys(ids))


def collect(sink, channels=None, searches=None, videos_per_target=25,
            search_results=20, verbose=True):
    """Fill `sink` with cleaned YouTube transcript lines."""
    targets = list(channels if channels is not None else DEFAULT_CHANNELS)
    for q in (searches if searches is not None else DEFAULT_SEARCHES):
        targets.append(f"ytsearch{search_results}:{q}")

    if verbose:
        print(f"      resolving {len(targets)} targets...")
    video_ids = resolve_video_ids(targets, videos_per_target)
    if not video_ids:
        print("      [!] no videos resolved (yt-dlp may be blocked or out of date)")
        return
    if verbose:
        print(f"      {len(video_ids)} videos queued")

    dedupe = Dedupe(allow_repeats=2)
    done = 0
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        ydl = _ydl(tmp)
        for vid in video_ids:
            if sink.full:
                break
            try:
                ydl.download([f"https://www.youtube.com/watch?v={vid}"])
            except Exception:
                continue
            got = 0
            for vtt in sorted(tmp.glob(f"{vid}*.vtt")):
                for raw in parse_vtt(vtt):
                    line = clean_line(raw, max_len=160)
                    if line and dedupe.ok(line):
                        got += 1
                        if not sink.write(line):
                            break
                vtt.unlink(missing_ok=True)
            done += 1
            if verbose and done % 10 == 0:
                print(f"      {done}/{len(video_ids)} videos  [{sink.written / 1e6:.1f} MB]")
    if verbose:
        print(f"      done: {done} videos processed")
