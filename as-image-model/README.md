# AS-I — Artificial Stupidity Image

A text-to-image model built **entirely from scratch** on a laptop. No Stable
Diffusion, no CLIP, no pretrained weights anywhere in it.

> **Status: in progress.** The pipeline runs end to end; numbers below marked
> _(pending)_ get filled in as training finishes.

---

## The question

Not "can I make images." That's solved. The question is:

> **What is the smallest network that still turns a sentence into a picture
> that matches it?**

Which forces an honest constraint most image-model projects dodge. Open-domain
text-to-image needs roughly **1B parameters and 150,000 A100-hours** — about
$600k of compute — because generating *anything* requires having *seen*
anything. That is not reachable from a MacBook, at any level of cleverness.

So AS-I takes the other trade: **a closed domain, learned properly, small.**
It's the same split the text side of this repo already makes — `AS-F` borrows
GPT-2's fluency, while `AS-0…AS-5` are from scratch, tiny, and honestly bad at
stated things. AS-I is the second kind.

---

## What it draws

```
"red heart"
"pizza"
"rocket in the center on a teal background"
"birthday cake in the top left on a navy background"
```

## What it doesn't

```
"two astronauts playing chess"        -> nope
```

It knows **1254 emoji names** plus a grammar of 9 positions, 6 backgrounds and
3 sizes. Novel *combinations* of known words work — `"pizza in the top left"`
never appeared in training and composes fine. Novel *concepts* do not, and no
amount of training at this size will change that.

---

## Why emoji and not photographs

The binding constraint at 25M parameters is not dataset size — it's **visual
complexity.**

A photograph is mostly high-frequency texture: fur, grass, skin, petal veins.
Reproducing texture is where large models spend their capacity. Emoji are flat
colour fields with hard edges, so the autoencoder reconstructs them almost
losslessly at 8×8 and the entire parameter budget goes to *what shape is a
rocket* instead of *what does noise look like*.

The result is a small model that draws **sharp** pictures, rather than a large
model that draws blurry ones.

---

## Architecture

```
"pizza in the top left"
        │
        ▼
  text encoder            0.4M params — a word-level transformer,
        │                 trained from scratch alongside everything else
        ▼
  diffusion U-Net        23.2M params — cross-attends to the caption,
   8×8×4 latent          8 DDIM steps, v-prediction
        │
        ▼
  VAE decoder            0.5M params
        │
        ▼
     64×64 image
```

### Three decisions worth defending

**No codebook.** The obvious reference (RQ-VAE) spends 16,384 codes × 256 dims
× 4 quantizers ≈ 16.8M parameters — about **67 MB** — on the lookup table
alone, more than this whole model. A continuous 4-channel latent needs no table
and deletes that cost outright. It's the same lesson as `--emb-bits` in the
text model's `compress.py`: the lookup table, not the compute, is where the
megabytes hide.

**No CLIP.** CLIP is 63M params / ~250 MB even at 8-bit — twice the entire
budget — and it's trained to understand all of English, of which this grammar
uses about 1300 words. A from-scratch word-level encoder does the job for
0.4M params.

**v-prediction, not epsilon.** Predicting the noise degrades at the high-noise
end, where `x_t` is nearly pure noise and "predict the noise" asks the model to
echo its own input. `v = α·ε − σ·x₀` stays well-scaled at every timestep, which
is what makes 8 steps enough instead of 50.

---

## Run it

```bash
cd as-image-model
python3.12 -m venv ../.venv && source ../.venv/bin/activate
pip install -r requirements.txt
```

### Get the artwork

OpenMoji is CC BY-SA 4.0 and isn't redistributed in this repo:

```bash
mkdir -p data/emoji_src && cd data/emoji_src
curl -sL -o openmoji.json \
  https://raw.githubusercontent.com/hfg-gmuend/openmoji/master/data/openmoji.json
curl -L -o openmoji-color.zip \
  https://github.com/hfg-gmuend/openmoji/releases/download/17.0.0/openmoji-618x618-color.zip
mkdir -p color && cd color && unzip -q ../openmoji-color.zip && cd ../../..
```

### Train the whole thing

```bash
python data/emoji.py                    # render 45k captioned images   ~4 min
python train_vae.py --data data/emoji   # stage 1: the compressor      ~25 min
python precompute_latents.py --data data/emoji   # encode once, 95x smaller
python train_diffusion.py --data data/emoji      # stage 2: the prior
python sample.py --sheet
```

### Generate

```bash
python sample.py --prompt "red heart"
python sample.py --prompt "pizza in the top left on a navy background" --n 8
python sample.py --sheet --steps 4 --guidance 5
```

| Flag | Default | Does |
|---|---|---|
| `--prompt` | — | what to draw |
| `--sheet` | off | a spread of stock prompts |
| `--steps` | `8` | DDIM steps. 4 is faster, 16 is slightly cleaner |
| `--guidance` | `4.0` | classifier-free guidance. Low = vague, high = rigid |
| `--n` | `4` | how many samples |
| `--seed` | random | reproducible output |

---

## Measuring it

Because the caption grammar is closed, "did it draw what I asked" has an
**exact** answer — no human eyeballing, no FID. `bench.py` reads the attributes
back off the generated pixels:

```bash
python bench.py --markdown
```

| Attribute | How it's scored | Result |
|---|---|---|
| background | nearest palette colour at the frame border | _(pending)_ |
| position | centroid of non-background pixels → which third | _(pending)_ |
| size | area of the non-background mask | _(pending)_ |

This is the same instinct as `bench.py` in the text model: make "it got better"
a number instead of a vibe.

The synthetic-shapes corpus (`data/shapes.py`) is kept as the architecture's
unit test — a closed world with held-out colour×shape pairs, so compositional
generalisation can be separated from memorisation.

---

## Files

```
config.py                the presets — every number here is a size lever
data/emoji.py            renders the corpus from OpenMoji
data/shapes.py           synthetic shapes: unit test + compositional probe
model/vae.py             64x64x3 -> 8x8x4, no codebook
model/text.py            word-level text encoder + tokenizer
model/unet.py            the prior: cross-attention U-Net
diffusion.py             cosine schedule, v-prediction, DDIM, EMA
train_vae.py             stage 1
precompute_latents.py    stage 1.5 — encode once, 95x smaller
train_diffusion.py       stage 2
sample.py                text -> image
bench.py                 exact attribute scoring
export/push_to_hub.py    publish weights + corpus to Hugging Face
```

Generated and never committed: `data/emoji/`, `data/emoji_src/`,
`data/shapes/`, `checkpoints/`. All reproducible from the commands above.

---

## Publishing

```bash
pip install huggingface_hub
hf auth login
python export/push_to_hub.py --model   --repo YOURNAME/artificial-stupidity-image
python export/push_to_hub.py --dataset --repo YOURNAME/artificial-stupidity-emoji
```

The dataset upload ships the **recipe** — catalog, captions, specs, latents —
not 585 MB of re-rendered OpenMoji artwork. One command rebuilds the pixels
byte for byte.

---

## Honest limitations

- **Closed vocabulary.** ~1300 words. Unknown words are ignored, and `sample.py`
  warns which ones it dropped rather than failing silently.
- **One object per image.** The grammar has no "and". Two-object composition is
  the obvious next step and isn't built.
- **64×64.** `AS-I-128` exists in `config.py` and is untrained.
- **Rare glyphs are softer.** 1254 identities share one small model; common
  emoji get more gradient than obscure ones.
- **It is not Stable Diffusion and will never be.** See the arithmetic at the
  top.
