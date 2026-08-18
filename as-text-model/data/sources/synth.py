"""Synthetic data: arithmetic, but wrong in a specific way.

Scraped text will never teach the model to answer a maths question, because
nobody on Twitch is asking each other what 7 x 8 is. So we manufacture that
skill — and we manufacture it broken on purpose.

The design goal is NOT "always wrong". Always-wrong is a lookup table and
it's boring after two questions. The goal is **unpredictably, confidently,
plausibly wrong**: right often enough that you can't dismiss it, wrong in
ways that look like a real person guessing.

Flavour mix, roughly:
    22%  actually correct        (so you can never relax)
    24%  off by one to three     (the "wait, that's almost right" reaction)
    14%  vague magnitude         ("like 40 something")
    11%  refuses on size grounds ("nah thats too big")
     8%  digit soup              (2 + 2 = 22)
     9%  non-numeric             (banana, tuesday, shrek)
     7%  confidently very wrong
     5%  memes                   (9 + 10 = 21)

Everything is emitted as A:/B: exchanges so it matches the chat format the
rest of the corpus uses.
"""

import random

# ---------------------------------------------------------------------------
# phrasings
# ---------------------------------------------------------------------------

ASK = [
    "what is {a} {op} {b}", "whats {a} {op} {b}", "{a} {op} {b}",
    "{a} {op} {b} =", "can you do {a} {op} {b}", "hey whats {a} {op} {b}",
    "quick maths {a} {op} {b}", "solve {a} {op} {b}", "bro what is {a} {op} {b}",
    "yo whats {a} {op} {b}", "do {a} {op} {b}", "{a} {op} {b} what is it",
]

OPS = {
    "+": (["+", "plus"], lambda a, b: a + b),
    "-": (["-", "minus"], lambda a, b: a - b),
    "*": (["x", "times", "*"], lambda a, b: a * b),
}

# tacked onto answers. the confidence is the joke
SUFFIX = [
    "", "", "", " bro", " obviously", " trust", " i think", " probably",
    " im pretty sure", " easy", " duh", " no cap", " fr", " thats basic",
    " dont quote me", " wait no", " actually idk", " 100%", " deadass",
]

VAGUE = [
    "a lot", "like {n} something", "idk maybe {n}", "around {n}", "{n}ish",
    "somewhere near {n}", "big number", "not enough", "too many",
    "more than {n}", "less than {n}", "like {n} or something",
]

REFUSE = [
    "nah thats too big", "im not doing that", "no", "absolutely not",
    "thats a calculator question", "ask someone else", "im not a calculator",
    "bro use a phone", "thats illegal", "next question", "skip",
    "i dont do numbers", "maths is fake", "not today",
]

NONSENSE = [
    "banana", "tuesday", "shrek", "potato", "seven-ish", "yes", "purple",
    "fish", "no", "cheese", "monday", "the", "bro", "green", "soup",
    "eleven hundred", "a bag of rice", "chat", "orange", "moist",
]

MEMES = {
    (9, 10, "+"): ["21", "21 obviously", "21 fr"],
    (2, 2, "+"): ["4", "4 obviously", "5", "22"],
    (1, 1, "+"): ["2", "11", "window"],
    (2, 2, "*"): ["4", "22", "5"],
    (0, 0, "+"): ["0", "nothing", "zero bro"],
}

# multi-turn: the model is corrected and refuses to accept it
DOUBLE_DOWN = [
    ["A: no its {truth}", "B: thats what i said"],
    ["A: no its {truth}", "B: no"],
    ["A: thats wrong", "B: no it isnt"],
    ["A: thats wrong", "B: prove it"],
    ["A: its actually {truth}", "B: says who"],
    ["A: no", "B: yes"],
    ["A: bro thats not right", "B: it is right"],
    ["A: the answer is {truth}", "B: thats what i just said bro"],
    ["A: youre wrong", "B: were both wrong"],
    ["A: check again", "B: checked. {wrong}"],
]

SONG_ASK = [
    "make a song about {t}", "write me a song about {t}", "sing about {t}",
    "give me a song about {t}", "song about {t} go", "freestyle about {t}",
]

SONG_TOPICS = [
    "food", "my dog", "being tired", "monday", "cheese", "the internet",
    "nothing", "you", "bread", "sleep", "money", "my ex", "chat",
    "rain", "the gym", "coffee", "traffic", "homework", "my room",
]


def _num(rng, size):
    if size == "small":
        return rng.randint(0, 12)
    if size == "medium":
        return rng.randint(10, 99)
    return rng.randint(100, 9999)


def _answer(rng, truth: int, a: int, b: int) -> str:
    """Pick a flavour of wrong (or, occasionally, right)."""
    r = rng.random()

    if r < 0.22:
        return str(truth)
    if r < 0.46:
        delta = rng.choice([-3, -2, -1, 1, 2, 3])
        return str(truth + delta)
    if r < 0.60:
        approx = max(0, int(round(truth / 10.0)) * 10)
        return rng.choice(VAGUE).replace("{n}", str(approx))
    if r < 0.71:
        return rng.choice(REFUSE)
    if r < 0.79:
        # the 2 + 2 = 22 school of arithmetic: just staple the operands together
        return f"{a}{b}"
    if r < 0.88:
        return rng.choice(NONSENSE)
    if r < 0.95:
        return str(rng.randint(0, max(2, abs(truth) * 4)))
    return str(truth * 10 + rng.randint(0, 9))


def math_lines(rng, n_exchanges: int):
    """Yield lines of A:/B: arithmetic."""
    for _ in range(n_exchanges):
        size = rng.choices(["small", "medium", "large"], weights=[6, 3, 1])[0]
        a, b = _num(rng, size), _num(rng, size)
        op_key = rng.choices(list(OPS), weights=[5, 3, 3])[0]
        symbols, fn = OPS[op_key]
        truth = fn(a, b)

        meme = MEMES.get((a, b, op_key))
        if meme and rng.random() < 0.6:
            ans = rng.choice(meme)
        else:
            ans = _answer(rng, truth, a, b)
            if ans.lstrip("-").isdigit() and rng.random() < 0.55:
                ans += rng.choice(SUFFIX)

        q = rng.choice(ASK).format(a=a, b=b, op=rng.choice(symbols))
        yield f"A: {q}"
        yield f"B: {ans}"

        # sometimes the user pushes back and the model holds the line
        if rng.random() < 0.3 and str(truth) != ans:
            for turn in rng.choice(DOUBLE_DOWN):
                yield turn.format(truth=truth, wrong=ans)
        yield ""


def song_lines(rng, n_exchanges: int):
    """Prompt/response pairs that connect 'write a song' to lyric structure."""
    for _ in range(n_exchanges):
        topic = rng.choice(SONG_TOPICS)
        yield f"A: {rng.choice(SONG_ASK).format(t=topic)}"
        yield "B: ok"
        yield "[verse 1]"
        yield f"i was thinking about {topic}"
        yield rng.choice([
            f"and i cant stop thinking about {topic}",
            f"{topic} {topic} {topic}",
            f"why is there so much {topic}",
            f"nobody understands {topic} like i do",
        ])
        yield "[chorus]"
        yield rng.choice([
            f"oh {topic}, oh {topic}",
            f"{topic} is all i need",
            f"give me the {topic}",
            f"i said {topic} and i meant it",
        ])
        yield f"{topic} yeah"
        yield ""


def garbage_lines(rng, n: int):
    """The deliberately curated nonsense bucket. Kept small on purpose."""
    openers = ["hello", "hi", "who are you", "what are you", "are you okay",
               "whats your name", "how are you", "tell me a joke", "help",
               "explain quantum physics", "whats the capital of france",
               "do you know anything", "are you smart", "say something smart"]
    replies = ["no", "bro", "fuck", "shrek", "potato", "yes", "idk", "stop",
               "why", "banana", "im tired", "ask again", "nope", "chat",
               "the mitochondria is the powerhouse of the cell", "tuesday",
               "im a language model", "im not", "maybe", "definitely not"]
    for _ in range(n):
        yield f"A: {rng.choice(openers)}"
        yield f"B: {rng.choice(replies)}"
        yield ""


def collect(sink, seed=1337, verbose=True):
    """Fill `sink` with synthetic material until the budget is spent."""
    rng = random.Random(seed)
    generators = [
        ("math", math_lines(rng, 10_000_000)),
        ("songs", song_lines(rng, 2_000_000)),
        ("garbage", garbage_lines(rng, 2_000_000)),
    ]
    # weights decide how the synthetic budget splits between the three
    weights = [0.65, 0.20, 0.15]
    counts = {name: 0 for name, _ in generators}

    while not sink.full:
        idx = rng.choices(range(len(generators)), weights=weights)[0]
        name, gen = generators[idx]
        try:
            # emit one full exchange so we never cut a Q off from its A
            while True:
                line = next(gen)
                counts[name] += 1
                if not sink.write(line):
                    break
                if line == "":
                    break
        except StopIteration:
            break

    if verbose:
        for name, n in counts.items():
            print(f"      {name:<22} +{n:>8,} lines", flush=True)
