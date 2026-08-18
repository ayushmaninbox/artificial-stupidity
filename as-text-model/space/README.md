---
title: Artificial Stupidity API
emoji: 🧠
colorFrom: gray
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Inference API — optional

> **The live site does not use this.** It runs the model in the visitor's
> browser instead — see [`../../web/README.md`](../../web/README.md). This directory
> is kept for self-hosting, and because it's the right starting point if you
> want an HTTP API rather than in-browser inference.

A FastAPI server that loads the model and streams replies over Server-Sent
Events.

## Why it isn't the main path

It was, until two things happened mid-build:

- **Hugging Face Docker Spaces now require PRO** ($9/mo). Creating one returns
  `402 Payment Required` — only *static* Spaces are free.
- **HF's serverless Inference API** returns
  `Model not supported by provider hf-inference` for custom GPT-2 fine-tunes.

Free alternatives, if you want a hosted API:

| Host | Free tier | Catch |
|---|---|---|
| Render | 512 MB web service | Spins down after 15 min; ~50s cold start. Needs the int8 ONNX model to fit in RAM |
| Google Cloud Run | 2M requests, scales to zero | Requires a billing account on file |
| HF Space | — | $9/mo PRO |

For a portfolio project the cold starts are worse than the in-browser
download, which is why the site went that way.

## Running it locally

```bash
pip install fastapi "uvicorn[standard]" transformers torch
MODEL_ID=../checkpoints/AS-F2 uvicorn app:app --port 7860
```

## Endpoints

`GET /` — health and model info.

`POST /chat` — streams the reply as Server-Sent Events. Each frame carries a
JSON-encoded token:

```bash
curl -N localhost:7860/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "why is the sky blue", "temperature": 0.9}'

data: {"t": " Because"}
data: {"t": " the"}
data: {"t": " ocean"}
event: done
```

Tokens are JSON-encoded rather than written as raw text because GPT-2 emits
leading spaces (`" there"`) that the SSE spec's leading-space rule would eat,
and any newline would break frame parsing outright.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_ID` | `checkpoints/AS-F2` | Local path or HF repo |
| `ALLOWED_ORIGINS` | `*` | Lock to your domain in production |
| `MAX_SENTENCES` | `2` | Where to cut the reply |
| `RATE_LIMIT` | `20` | Requests per IP per window |
| `RATE_WINDOW` | `60` | Window, seconds |

`MAX_SENTENCES` matters more than it looks. Past two sentences the model drifts
back into the scraped web text it was fine-tuned on and starts emitting things
like `#cricketnews` and `[ click next ]`. The cut has guards so `3.5` and `Mr.`
don't count as sentence endings.

## Deploying to a Space (requires PRO)

```bash
git clone https://huggingface.co/spaces/YOURNAME/artificial-stupidity-api deploy
cp space/* deploy/
cd deploy && git add -A && git commit -m "deploy" && git push
```

Then set `MODEL_ID` to your model repo under **Settings → Variables**.
