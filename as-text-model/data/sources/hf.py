"""Hugging Face datasets, streamed.

Everything here uses streaming=True. These datasets are tens of gigabytes and
we want a couple hundred megabytes, so downloading them in full would be a
waste of an afternoon and most of your SSD.
"""

import re

from .clean import Dedupe, clean_line

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

# Reddit comments arrive as paragraphs; Twitch chat arrives as single lines.
# We want conversational-length text, so paragraphs get split into sentences.
SOURCES = {
    "twitch_hf": {
        "path": "lparkourer10/twitch_chat",
        "split": "train",
        "field": "Message",
        "split_sentences": False,
        "max_len": 120,
    },
    # Same dataset as "reddit", but emitted as A:/B: exchanges reconstructed
    # from parent_id. Without this the model never sees a real conversation in
    # the format chat mode uses, and answers every question by imitating the
    # synthetic data instead.
    "reddit_pairs": {
        "path": "HuggingFaceGECLM/REDDIT_comments",
        "split": [
            "AskHistorians", "changemyview", "explainlikeimfive", "tifu",
            "Showerthoughts", "gaming", "Games", "socialskills",
            "relationship_advice", "todayilearned", "mildlyinteresting",
            "bestof", "IWantToLearn", "books",
        ],
        "field": "body",
        "split_sentences": False,
        "max_len": 160,
        "pairs": True,
    },
    "reddit": {
        "path": "HuggingFaceGECLM/REDDIT_comments",
        # chosen for chaos and back-and-forth, not for information
        "split": [
            "tifu", "Showerthoughts", "gaming", "Games", "mildlyinteresting",
            "todayilearned", "changemyview", "bestof", "socialskills",
            "relationship_advice", "ifyoulikeblank", "WritingPrompts",
        ],
        "field": "body",
        "split_sentences": True,
        "max_len": 180,
    },
}


def _load(spec, sp):
    from datasets import load_dataset
    return load_dataset(spec["path"], split=sp, streaming=True)


def _splits(spec):
    return spec["split"] if isinstance(spec["split"], list) else [spec["split"]]


def _iter_texts(spec, verbose=True):
    for sp in _splits(spec):
        if verbose:
            print(f"      streaming {spec['path']} [{sp}]", flush=True)
        try:
            ds = _load(spec, sp)
        except Exception as e:
            print(f"      (skipped {sp}: {str(e)[:80]})", flush=True)
            continue
        try:
            for row in ds:
                text = row.get(spec["field"])
                if text:
                    yield sp, text
        except Exception as e:
            print(f"      (stream ended early on {sp}: {str(e)[:80]})", flush=True)


def _iter_pairs(spec, buffer_size=200_000, verbose=True):
    """Reconstruct real reply pairs from a flat comment stream.

    Reddit comments carry `id` and `parent_id` ("t1_<id>" for a reply to
    another comment, "t3_<id>" for a top-level reply to the post). Comments
    arrive roughly in time order, so a child almost always shows up after its
    parent. We keep a bounded window of recent comment bodies keyed by id, and
    whenever a comment's parent is still in the window we emit the exchange.

    The window is capped because these splits are millions of rows and we are
    not going to hold Reddit in memory.
    """
    for sp in _splits(spec):
        if verbose:
            print(f"      streaming {spec['path']} [{sp}] as reply pairs", flush=True)
        try:
            ds = _load(spec, sp)
        except Exception as e:
            print(f"      (skipped {sp}: {str(e)[:80]})", flush=True)
            continue

        window: dict[str, str] = {}
        try:
            for row in ds:
                body = row.get(spec["field"])
                cid = row.get("id")
                parent = row.get("parent_id") or ""

                if parent.startswith("t1_"):
                    prompt = window.get(parent[3:])
                    if prompt and body:
                        yield sp, prompt, body

                if cid and body:
                    if len(window) >= buffer_size:
                        # drop the oldest half rather than clearing entirely,
                        # so we don't lose every in-flight parent at once
                        for k in list(window)[: buffer_size // 2]:
                            del window[k]
                    window[cid] = body
        except Exception as e:
            print(f"      (stream ended early on {sp}: {str(e)[:80]})", flush=True)


def collect(sink, source: str, verbose=True):
    """Fill `sink` from one registered HF source."""
    spec = SOURCES[source]
    dedupe = Dedupe(allow_repeats=2)
    per_split = {}

    if spec.get("pairs"):
        for sp, prompt, reply in _iter_pairs(spec, verbose=verbose):
            if sink.full:
                break
            q = clean_line(prompt, max_len=spec["max_len"])
            a = clean_line(reply, max_len=spec["max_len"])
            if not (q and a) or not dedupe.ok(q + a):
                continue
            per_split[sp] = per_split.get(sp, 0) + 1
            if not (sink.write(f"A: {q}") and sink.write(f"B: {a}")):
                break
            sink.write("")
        if verbose:
            for sp, n in per_split.items():
                print(f"      {sp:<22} +{n:>8,} exchanges", flush=True)
        return

    for sp, text in _iter_texts(spec, verbose):
        if sink.full:
            break
        chunks = SENT_SPLIT.split(text) if spec["split_sentences"] else [text]
        for chunk in chunks:
            line = clean_line(chunk, max_len=spec["max_len"])
            if line and dedupe.ok(line):
                per_split[sp] = per_split.get(sp, 0) + 1
                if not sink.write(line):
                    break
        # rotate off a split once it's contributed its share, so one giant
        # subreddit can't dominate the whole budget
        if isinstance(spec["split"], list):
            share = sink.budget / len(spec["split"])
            if per_split.get(sp, 0) and sink.written >= share * (len(per_split)):
                continue

    if verbose:
        for sp, n in per_split.items():
            print(f"      {sp:<22} +{n:>8,} lines")
