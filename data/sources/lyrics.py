"""Song lyrics, so the thing can be asked to write a song.

Section headers ([Chorus], [Verse 1]) are deliberately KEPT. They're the only
structural signal in the data — without them the model learns "words that
rhyme sometimes" instead of "songs have a shape".
"""

import re

from .clean import Dedupe, clean_line

DATASET = "sebastiandizon/genius-song-lyrics"

# rap and pop carry the most conversational, repetitive, learnable phrasing
KEEP_TAGS = {"rap", "pop", "rb", "rock", "country"}
SECTION_RE = re.compile(r"^\[[^\]]{1,40}\]$")
CONTRIB_RE = re.compile(r"^\d*\s*Contributors?|Lyrics$|^You might also like")


def collect(sink, max_songs=40000, verbose=True):
    from datasets import load_dataset

    if verbose:
        print(f"      streaming {DATASET}", flush=True)
    try:
        ds = load_dataset(DATASET, split="train", streaming=True)
    except Exception as e:
        print(f"      [!] could not stream lyrics: {str(e)[:120]}", flush=True)
        return

    dedupe = Dedupe(allow_repeats=2)
    songs = kept = 0
    try:
        for row in ds:
            if sink.full or songs >= max_songs:
                break
            if row.get("language") != "en" or row.get("tag") not in KEEP_TAGS:
                continue
            songs += 1

            wrote_any = False
            for raw in (row.get("lyrics") or "").split("\n"):
                raw = raw.strip()
                if not raw or CONTRIB_RE.match(raw):
                    continue
                if SECTION_RE.match(raw):
                    # structure marker: pass through verbatim, always
                    if sink.write(raw.lower()):
                        wrote_any = True
                    continue
                line = clean_line(raw, max_len=120, keep_brackets=True)
                if line and dedupe.ok(line):
                    if not sink.write(line):
                        break
                    wrote_any = True
            if wrote_any:
                kept += 1
                sink.write("")            # blank line separates songs
                if verbose and kept % 2000 == 0:
                    print(f"      {kept:,} songs  [{sink.written / 1e6:.1f} MB]", flush=True)
    except Exception as e:
        print(f"      (lyrics stream ended: {str(e)[:100]})", flush=True)

    if verbose:
        print(f"      {kept:,} songs kept of {songs:,} scanned", flush=True)
