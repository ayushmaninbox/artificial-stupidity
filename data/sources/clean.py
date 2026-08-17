"""Shared text cleaning.

Every source funnels through here so the corpus has one consistent character
set and one consistent idea of what counts as a usable line.

The tokenizer is character-level, so vocabulary size is a real cost: every
stray unicode character is a row in the embedding table that the model has to
spend capacity on. We restrict hard to ASCII plus a small emoji allowlist —
enough to keep the flavor, cheap enough to keep the model tiny.
"""

import hashlib
import re
import unicodedata

# the only non-ASCII characters worth paying vocabulary for
EMOJI_KEEP = "💀😭🔥😂🤣💯👀🗿😐😳🙏✅❌"

_ALLOWED = set(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " .,!?'\"-:;()/#$%&*+=<>@[]_~\n"
) | set(EMOJI_KEEP)

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"(?<!\w)/?u/\w+|(?<!\w)/?r/\w+|@\w+")
MARKDOWN_RE = re.compile(r"[*_`~]{1,3}|&gt;|&lt;|&amp;|&#\d+;")
SPEAKER_RE = re.compile(r"^>>+\s*")                   # ">> speaker" in captions
BRACKET_RE = re.compile(r"\[[^\]]{0,30}\]")           # "[Music]", "[Applause]"
WS_RE = re.compile(r"[ \t]+")
REPEAT_RE = re.compile(r"(.)\1{3,}")

# Twitch emote names are camelCase mush ('elbyGiggles', 'xqFreaky', 'PoroSad').
# They're authentic but they teach a character-level model nothing except that
# capital letters appear at random. Requiring both cases spares ALL-CAPS
# reactions like KEKW and LULW, which are worth keeping.
EMOTE_RE = re.compile(r"\b(?=\w*[a-z])(?=\w*[A-Z])\w{7,}\b")

# Twitch chat is ~15% robots announcing things nobody asked about
BOTS = {
    "nightbot", "streamelements", "moobot", "fossabot", "streamlabs",
    "thepositivebot", "supibot", "buttsbot", "wizebot", "botisimo",
    "phantombot", "ohbot", "own3d", "sery_bot", "kunszgbot", "pokemoncommunitygame",
    "streamstickers", "commanderroot", "anotherttvviewer", "lurxx", "0_applebadapple_0",
    "streamlabs", "soundalerts", "creatisbot", "logviewer", "kattah", "8roku",
}

# subreddit-flavored boilerplate that teaches the model nothing
JUNK_PREFIXES = (
    "[deleted]", "[removed]", "your submission", "i am a bot",
    "this comment was", "hello, /u/", "thank you for your submission",
    "please contact the moderators", "^^i ^^am ^^a ^^bot",
)


def _fold(text: str) -> str:
    """Best-effort map of typographic unicode onto ASCII (smart quotes, dashes)."""
    text = (text.replace("‘", "'").replace("’", "'")
                .replace("“", '"').replace("”", '"')
                .replace("–", "-").replace("—", "-")
                .replace("…", "...").replace(" ", " "))
    # strip accents rather than dropping the letter entirely
    return "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )


def collapse_ngram_repeats(text: str, max_n: int = 4, max_run: int = 2) -> str:
    """'Lmao KEKL Lmao KEKL Lmao KEKL' -> 'Lmao KEKL Lmao KEKL'.

    Copypasta is a real feature of chat, but a line repeated eight times is
    eight times the training signal for one joke. Two runs keeps the texture.
    """
    words = text.split()
    out: list[str] = []
    i = 0
    while i < len(words):
        for n in range(max_n, 0, -1):
            if i + 2 * n > len(words):
                continue
            gram = words[i:i + n]
            j, runs = i + n, 1
            while words[j:j + n] == gram:
                runs += 1
                j += n
            if runs > max_run:
                out.extend(gram * max_run)
                i = j
                break
        else:
            out.append(words[i])
            i += 1
    return " ".join(out)


def collapse_repeats(text: str, max_run: int = 3) -> str:
    """LMAOOOOOOOOOOOO -> LMAOOO. Keeps the energy, loses the token waste."""
    text = REPEAT_RE.sub(lambda m: m.group(1) * max_run, text)
    return collapse_ngram_repeats(text)


def clean_line(
    line: str,
    *,
    min_len: int = 2,
    max_len: int = 200,
    max_junk_ratio: float = 0.15,
    strip_emotes: bool = True,
    keep_brackets: bool = False,
) -> str | None:
    """Normalize one line. Returns None if it isn't worth training on.

    `keep_brackets` preserves [Chorus]-style markers, which lyrics need and
    everything else is better off without.
    """
    if not line:
        return None

    line = _fold(line)
    line = URL_RE.sub("", line)
    line = SPEAKER_RE.sub("", line)
    if not keep_brackets:
        line = BRACKET_RE.sub(" ", line)
    line = MENTION_RE.sub("", line)
    line = MARKDOWN_RE.sub("", line)
    if strip_emotes:
        line = EMOTE_RE.sub("", line)
    # stripping a name out of "(Ugh, Juelz)" leaves "(Ugh, )" behind
    line = re.sub(r"\(\s*[,.!?]*\s*\)|''|\"\"", "", line)

    kept = "".join(c for c in line if c in _ALLOWED)
    if not kept:
        return None
    # if we had to throw away a lot, it was probably not English to begin with
    if (len(line) - len(kept)) / len(line) > max_junk_ratio:
        return None

    kept = collapse_repeats(WS_RE.sub(" ", kept).strip())
    if not (min_len <= len(kept) <= max_len):
        return None

    low = kept.lower()
    if low.startswith(JUNK_PREFIXES):
        return None
    if kept.startswith(("!", "?!", "-", "|")):     # chat commands, table rows
        return None
    # must contain actual letters, not just punctuation and numbers
    if sum(c.isalpha() for c in kept) < max(2, len(kept) * 0.4):
        return None

    return kept


def is_bot(username: str | None) -> bool:
    if not username:
        return False
    u = username.lower()
    return u in BOTS or u.endswith("bot") or u.endswith("_bot")


class Dedupe:
    """Exact-duplicate filter on a normalized form, backed by hashes.

    Chat is extremely repetitive ('W', 'LUL', '+1'). We want *some* of that —
    it's genuinely how people type — so this allows each distinct line through
    a bounded number of times rather than exactly once.
    """

    def __init__(self, allow_repeats: int = 3):
        self.allow_repeats = allow_repeats
        self.seen: dict[bytes, int] = {}

    def ok(self, line: str) -> bool:
        key = hashlib.blake2b(
            re.sub(r"[^a-z0-9]", "", line.lower()).encode(), digest_size=8
        ).digest()
        n = self.seen.get(key, 0)
        if n >= self.allow_repeats:
            return False
        self.seen[key] = n + 1
        return True
