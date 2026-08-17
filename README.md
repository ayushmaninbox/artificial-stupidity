# Artificial Stupidity

**How little precision can a language model survive on before it stops speaking English?**

We train the same small AI six times. Each time, we give it less room to store
what it knows. Then we see how badly it falls apart, and how funny it is on the
way down.

---

## The idea in one picture

Imagine every fact a neural network knows is stored on a dial.

A normal AI's dials have about **4 billion positions** each — that's what
"32-bit" means. Very precise, very expensive to store.

Our worst model's dials have **two positions**: left, or right. That's "1-bit".

```
32-bit dial:   |·····················●···················|   4.3 billion settings
 8-bit dial:   |·····●·····|                                  256 settings
 4-bit dial:   |··●··|                                        16 settings
 1-bit dial:   ◄─────►                                        2 settings
```

Fewer settings means a much smaller file. It also means a much stupider model.
The experiment is finding out exactly where the trade stops being worth it.

---

## The six models

Same brain, same training data, same code. **The only difference is the dials.**

| Model | Dial positions | What it is |
|---|---|---|
| **AS-0** | 4.3 billion (32-bit) | The control group. A normal small AI. |
| **AS-1** | 256 (8-bit) | Barely damaged. Should be almost as good. |
| **AS-2** | 16 (4-bit) | Noticeably compressed. |
| **AS-3** | 3 (ternary) | Every dial is `-1`, `0`, or `+1`. This is BitNet b1.58 territory. |
| **AS-4** | 2 (1-bit) | Every dial is `-1` or `+1`. No zero. No nuance. |
| **AS-5** | 2, and a smaller brain | 1-bit dials *and* fewer of them. Absolute artificial stupidity. |

They're all one config file. [config.py](config.py) is literally this:

```python
"AS-0": Config(weight_bits=32),     # control
"AS-4": Config(weight_bits=1),      # one bit
```

That matters. If each model were its own codebase, and AS-4 came out
brain-dead, we'd never know whether that was the compression working or a bug
we wrote. One flag = one variable = an actual experiment.

---

## Size, honestly measured

| Model | Bits | Params | File size | Shrink |
|---|---|---|---|---|
| AS-0 | 32 | 815,488 | 3.1 MB | 1.0x |
| AS-1 | 8 | 819,072 | 835 KB | 3.8x |
| AS-2 | 4 | 819,072 | 448 KB | 7.0x |
| AS-3 | 1.58 | 819,072 | 216 KB | 14.5x |
| AS-4 | 1 | 819,072 | **169 KB** | 19.9x |
| AS-5 | 1 | 350,784 | **83 KB** | 37.7x |

**1 bit is not a 32x saving — it's about 20x.** Some parts of the model
(the vocabulary table, the normalization layers, and one scale number per row
of weights) have to stay high-precision or the whole thing collapses for
almost no size win. BitNet keeps those too. `packed_bytes()` in
[model/bitlinear.py](model/bitlinear.py) counts the real bytes on disk, not the
theoretical best case.

---

## Results so far

Trained on a 9.8 MB corpus, 5000 iterations, ~6 min each on an M4 Mac.

| Model | Val loss | Sample output |
|---|---|---|
| AS-0 | **1.642** | `The drop batch, hey the knew done. cooked erroblem? LOL` |
| AS-4 | *training* | |

Lower val loss = better at predicting text. For reference, the same model on a
3.6 KB toy corpus scored 2.31 and was memorizing rather than learning.

---

## How it actually works

### 1. Get the data

We scrape ~150 MB of the most chaotic English on the internet.

| Source | Share | What it gives us |
|---|---|---|
| **Twitch chat** (live scrape) | 24% | Thousands of people reacting in five words or less |
| **YouTube transcripts** | 24% | 24 channels of nonstop talking — commentary, Sidemen, streamers |
| **Reddit** (HF dump) | 16% | Actual back-and-forth arguments |
| **Reddit** (live, creator subs) | 10% | r/ksi, r/sidemen, r/okbuddyretard — *needs API keys, see below* |
| **Twitch chat** (HF dump) | 10% | More of the above |
| **Song lyrics** | 10% | So it can be asked to write a song |
| **Synthetic maths** | 6% | So it can do arithmetic, badly |

### 2. Clean it

[data/sources/clean.py](data/sources/clean.py) is stricter than you'd expect,
because a character-level model pays for every distinct character it ever sees.
It strips URLs, `@mentions`, `[Music]` tags, bot messages, and Twitch emote
names (`elbyGiggles`, `xqFreaky`) — while deliberately *keeping* `KEKW`, `LULW`
and `💀`, which are real words to this corpus.

It also collapses spam: `LMAOOOOOOOOOO` → `LMAOOO`, and
`Lmao KEKL Lmao KEKL Lmao KEKL` → `Lmao KEKL Lmao KEKL`.

### 3. Turn letters into numbers

The model reads **one character at a time**, not words. It sees
`h`, `e`, `l`, `l`, `o` — it isn't told that "hello" is a thing. It has to work
that out. This is the dumbest tokenizer that works, and it makes the early
failure modes much funnier.

Our vocabulary is 90 characters.

### 4. Train

A ~800,000-parameter transformer (the same architecture as ChatGPT, roughly
100,000 times smaller). It learns by repeatedly guessing the next character and
being corrected.

### 5. Commit the crime

Here's the part that's easy to get wrong, and the reason the code is built the
way it is:

> **You cannot train a normal model and then round its dials down to 1 bit.**
> That gives you static, not a dumb model — and you can't tell the difference.

The network has to *know* it's being compressed **while it learns**, so it can
route around the damage. That's called quantization-aware training. In
[model/bitlinear.py](model/bitlinear.py):

- the model only ever **uses** the rounded-off dials when making predictions
- but it **learns** against a hidden full-precision copy
- that hidden copy is thrown away when we export

### 6. Score it

[bench.py](bench.py) asks every model the same questions and measures
**wordness** — the fraction of things it says that are real words. That turns
"it got dumber" from a vibe into a number.

---

## Running it

```bash
# one-time setup
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. collect the corpus (slow — logs to a file you can watch)
python -u data/collect.py --target-mb 150 > logs/collect.log 2>&1 &
tail -f logs/collect.log

# 2. turn it into training data
python data/prepare.py

# 3. train (~6 min each on an M4)
python -u train.py AS-0 | tee logs/train.log
python -u train.py AS-4 | tee logs/train.log

# 4. talk to it
python generate.py AS-0 --chat
python generate.py AS-0 --prompt "bro is" --tokens 100 --samples 5

# 5. score everything
python bench.py --markdown
```

Always use `python -u` for long jobs, or Python buffers the output and the log
looks frozen when it isn't.

### Useful knobs

```bash
--temperature 0.5    # coherent and repetitive
--temperature 1.4    # completely unhinged
                     # 0.8-1.0 is the funny zone

python data/collect.py --list                  # show sources
python data/collect.py --only twitch,youtube   # collect a subset
```

### Optional: creator subreddits

r/ksi, r/sidemen and friends need Reddit API keys. Every free route is dead
(`.json` → 403, RSS → blocked, PullPush → paid). Reddit's own API is free and
takes two minutes:

1. https://www.reddit.com/prefs/apps → **create another app**
2. Type **script**, redirect URI `http://localhost:8080`
3. ```bash
   export REDDIT_CLIENT_ID=...
   export REDDIT_CLIENT_SECRET=...
   ```

Without the keys this source skips itself and everything else still works.

---

## Layout

```
config.py                  all six variants, one dataclass
model/bitlinear.py         the quantizers + straight-through estimator  <- the interesting file
model/model.py             ~800K parameter transformer
model/tokenizer.py         character-level
data/collect.py            scrapes the corpus, enforces the source mix
data/sources/clean.py      shared cleaning — decides what's worth training on
data/sources/twitch.py     live chat via Justlog
data/sources/youtube.py    transcripts via yt-dlp
data/sources/reddit_live.py  creator subreddits via Reddit OAuth
data/sources/lyrics.py     Genius lyrics, structure preserved
data/sources/synth.py      confidently wrong arithmetic
data/prepare.py            text -> training bins
train.py                   one variant per run
generate.py                --prompt or --chat
bench.py                   the stupidity leaderboard
```

---

## Status

- [x] Working transformer, trains on Apple Silicon GPU
- [x] Quantization-aware training — the 1-bit path converges
- [x] Honest size accounting
- [x] Scored benchmark
- [x] Real corpus pipeline: Twitch, YouTube, Reddit, lyrics, synthetic maths
- [x] AS-0 trained on 9.8 MB — val loss 1.642
- [ ] Collect the full 150 MB
- [ ] Train all six, fill in the leaderboard
- [ ] Bit-packed export — actually ship the 83 KB file
- [ ] Web demo with a precision slider

```
INTELLIGENCE  🧠 ──────●───── 💀
MODEL SIZE:   169 KB
```
