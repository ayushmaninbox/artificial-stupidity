# Artificial Stupidity

How little precision can a language model survive on before it stops speaking English?

Same tiny transformer, trained six times at six different weight precisions —
32-bit down to **1 bit**. One config flag is the only difference between them.

```
weight_bits:  32  →  8  →  4  →  1.58 (ternary)  →  1 (binary)
```

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch numpy tqdm
```

## Run it

```bash
python data/prepare.py          # raw text -> train/val bins + char tokenizer
python train.py AS-0            # the control group (full precision)
python train.py AS-4            # ONE BIT
python generate.py AS-4 --chat  # talk to it
python bench.py --markdown      # the leaderboard
```

## The leaderboard

Numbers below are from the 3.6 KB smoke-test corpus, 1000 iters. They are
placeholders — they say the pipeline works, not that the models do.

| Model | Bits | Params | Size | Shrink | Val loss | Wordness |
|---|---|---|---|---|---|---|
| AS-0 | 32 | 809,344 | 3.0 MB | 1.0x | 2.307 | 33% |
| AS-1 | 8 | 812,928 | 828.8 KB | 3.8x | – | – |
| AS-2 | 4 | 812,928 | 444.8 KB | 7.0x | – | – |
| AS-3 | 1.58 | 812,928 | 214.4 KB | 14.5x | – | – |
| AS-4 | 1 | 812,928 | 156.8 KB | 19.9x | 2.205 | 17% |
| AS-5 | 1 + 8-bit acts | 350,784 | 82.7 KB | 37.7x | – | – |

**Wordness** = fraction of emitted words that are real words. Our coherence proxy.

## Two things worth knowing

**1. You can't just round an FP32 model down to 1 bit.** That gives you noise,
not a dumb model, and you won't be able to tell which one you got. The network
has to know it's being quantized *while it learns*. `model/bitlinear.py` does
quantization-aware training with a straight-through estimator: the forward pass
only ever sees quantized weights, gradients flow through the rounding to a
full-precision shadow copy that the optimizer updates. That shadow copy is
discarded at export.

**2. 1 bit is not a 32x size win.** It's ~20x here. Embeddings, LayerNorms, and
one fp16 scale per output row all stay in higher precision — BitNet keeps them
too, because quantizing them wrecks the model for almost no size saving.
`packed_bytes()` counts the real bytes, not the theoretical ones.

## Layout

```
config.py              all six variants, one dataclass
model/bitlinear.py     the quantizers + straight-through estimator  <- the interesting file
model/model.py         ~800K param decoder-only transformer
model/tokenizer.py     character-level (words are earned, not given)
data/prepare.py        raw txt -> bins
train.py               one variant per invocation
generate.py            --prompt or --chat
bench.py               fixed prompts, scored, all checkpoints
```

## Status

- [x] Phase 1 — working baseline transformer, trains on MPS
- [x] Phase 2 — quantization-aware training, 1-bit path converges
- [x] Phase 3 — honest size accounting + scored benchmark
- [ ] Phase 4 — **a real corpus** (current one is 3.6 KB; need ~50–200 MB)
- [ ] Phase 5 — retrain all six, fill in the leaderboard
- [ ] Phase 6 — bit-packed export (`.asx`), actually ship the small files
- [ ] Phase 7 — web demo with a precision slider
```
INTELLIGENCE  🧠 ──────●───── 💀
MODEL SIZE:   156.8 KB
```
