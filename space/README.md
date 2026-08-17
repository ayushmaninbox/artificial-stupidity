---
title: Artificial Stupidity API
emoji: 🧠
colorFrom: purple
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Artificial Stupidity — inference API

Backend for the [Next.js frontend](https://github.com/ayushmaninbox/artificial-stupidity).
A GPT-2 fine-tune that answers every question fluently, confidently, and wrongly.

## Endpoints

`GET /` — health and model info.

`POST /chat` — streams the reply as Server-Sent Events.

```bash
curl -N https://YOURNAME-artificial-stupidity-api.hf.space/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "why is the sky blue", "temperature": 0.9}'
```

## Configuration

Set these in **Settings → Variables and secrets**:

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_ID` | `checkpoints/AS-F2` | Hub repo to load, e.g. `yourname/artificial-stupidity` |
| `ALLOWED_ORIGINS` | `*` | Lock to your Vercel domain once deployed |
| `RATE_LIMIT` | `20` | Requests per IP per window |
| `RATE_WINDOW` | `60` | Window in seconds |

## Deploying

```bash
git clone https://huggingface.co/spaces/YOURNAME/artificial-stupidity-api space-deploy
cp space/* space-deploy/
cd space-deploy && git add -A && git commit -m "deploy" && git push
```

Free Spaces sleep after 48 hours of inactivity and take roughly 30 seconds to
wake on the next request.
