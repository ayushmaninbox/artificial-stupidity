# AS Text Model

Everything that trains, compresses, benchmarks and serves the language model.

> **Run every command in this directory.** Each script anchors its paths to its
> own location, so `checkpoints/`, `data/` and `onnx_build/` always resolve
> inside `as-text-model/` — but the two Gradio/FastAPI entry points read a
> `MODEL_ID` relative to your shell, so `cd` here first.

```bash
cd as-text-model
python3.12 -m venv ../.venv && source ../.venv/bin/activate
pip install -r requirements.txt
```

---

## Two models live here

They share the corpus and nothing else.

| | **AS-F** — the talkative one | **AS-0 … AS-5** — the tiny ones |
|:--|:--|:--|
| Base | fine-tuned from GPT-2 (124M) | built from scratch |
| Tokens | word pieces — cannot misspell | one character at a time |
| Size | 475 MB → 164 MB as int8 ONNX | 3.1 MB → **83 KB** |
| Point | sounds human, is wrong on purpose | how far can precision fall before language breaks |
| Ships to | the website | the leaderboard |

---

## Talk to it

Straight from Hugging Face, no training required:

```bash
python talk.py --model ayushmaninbox/artificial-stupidity --chat
```

Or a local checkpoint:

```bash
python talk.py --model checkpoints/AS-F2 --chat
python talk.py --model checkpoints/AS-F2 --prompt "bro is" --samples 3
```

| Flag | Default | Does |
|---|---|---|
| `--model` | `checkpoints/AS-F` | local dir or HF repo id |
| `--chat` | off | interactive loop, ctrl-c to leave |
| `--prompt` | — | one-shot instead of chat |
| `--tokens` | `60` | reply length |
| `--temperature` | `0.9` | low is repetitive, high is unhinged |
| `--samples` | `1` | how many completions |
| `--device` | `mps` | falls back to CPU automatically |

A local chat UI, same model, in the browser:

```bash
pip install gradio
python app.py
```

---

## Build it from nothing

```bash
# 1. scrape ~118 MB of the internet at 2am              (~60 min)
python data/collect.py --target-mb 150

# 2. teach GPT-2 to talk like that, not like GPT-2
python finetune.py --base gpt2 --max-iters 200

# 3. teach it to answer every question confidently and wrongly
python finetune.py --base checkpoints/AS-F \
    --raw-dir data/stage2 --max-iters 150 --lr 5e-5 \
    --out checkpoints/AS-F2

python talk.py --model checkpoints/AS-F2 --chat
```

Both fine-tune stages stop early **on purpose** — stage 1 at iteration 200,
stage 2 at 150. Left running, stage 1 overfits Twitch's repetition and stage 2
memorises the 89 persona seeds instead of generalising from them. The reasoning
is in the root [README](../README.md).

`--out` writes two directories: the best-validation checkpoint, and
`<out>-final` with the last weights. For a humour model the lowest loss is not
reliably the funniest, so both are kept.

### Collecting selectively

```bash
python data/collect.py --list
python data/collect.py --target-mb 20 --only twitch,youtube
```

`reddit_live` needs Reddit API keys — every free route (`.json`, RSS,
PullPush) is dead. The other five sources need nothing.

---

## The tiny from-scratch models

```bash
python data/prepare.py          # corpus -> train.bin / val.bin / tokenizer.json
python train.py AS-0            # full precision control group
python train.py AS-4            # one bit
python generate.py AS-4 --prompt "hello"
python bench.py --markdown      # score them against each other
```

Six presets, identical architecture, one number changed
([`config.py`](config.py)):

| Preset | Weights | Activations | Notes |
|---|---|---|---|
| `AS-0` | 32-bit | 32-bit | control group |
| `AS-1` | 8-bit | 32-bit | mild |
| `AS-2` | 4-bit | 32-bit | moderate |
| `AS-3` | 1.58-bit `{-1,0,+1}` | 32-bit | BitNet b1.58 territory |
| `AS-4` | 1-bit `{-1,+1}` | 32-bit | no zero, no nuance |
| `AS-5` | 1-bit | 8-bit | smaller brain too — 83 KB |

Quantization happens **during** training, not after. Round a finished model's
weights to 1 bit and you get static; the model has to know it is being squashed
so it can route around the damage. That's the straight-through estimator in
[`model/bitlinear.py`](model/bitlinear.py).

`bench.py` scores "wordness" — the fraction of emitted words that are real
words, measured against the corpus lexicon — so *it got dumber* is a number
rather than a vibe.

---

## Compressing the big one

Points the AS-4 quantizers at GPT-2 and reports what it costs:

```bash
python compress.py --bits 4
python compress.py --bits 1.58 --emb-bits 4 --sparsity 0.3
```

`--emb-bits` is the single biggest lever on final size: GPT-2's embedding table
is 50257 × 768 ≈ 38.6M parameters, about a third of the model, and leaving it
in fp16 costs 77 MB on its own.

The saved file stays full-size — safetensors stores dequantized values. The
`packed` number in the output is what a real bit-packed export would weigh, and
that's the honest one.

---

## What's in here

```
finetune.py              two-stage fine-tune: voice, then personality
talk.py                  chat in your terminal
app.py                   local Gradio chat UI
compress.py              points the AS-4 quantizers at GPT-2

train.py                 trains any of the six tiny models
generate.py              sample from one
bench.py                 the leaderboard
config.py                the six presets

model/model.py           a transformer, written from scratch
model/bitlinear.py       quantizers + straight-through estimator
model/tokenizer.py       character-level

data/collect.py          runs the scrapers, enforces the source mix
data/prepare.py          corpus -> training bins
data/sources/clean.py    decides what's worth keeping
data/sources/persona.py  the 89 hand-written wrong answers
data/sources/            twitch, youtube, hf dumps, lyrics, synth, reddit

export/                  publish to HF, ONNX, GGUF, Ollama   [export/DEPLOY.md]
space/                   optional self-hosted FastAPI        [space/README.md]
```

Generated and never committed: `data/raw/`, `data/processed/`,
`checkpoints/`, `onnx_build/`, `logs/`. All reproducible from the above; the
corpus and weights are published to Hugging Face instead.

---

## Shipping

The website that serves this model is in [`../web/`](../web) and downloads an
int8 ONNX build from Hugging Face — see [`export/DEPLOY.md`](export/DEPLOY.md)
for how that export is produced, plus GGUF and Ollama.
