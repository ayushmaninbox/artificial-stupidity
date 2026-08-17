# Artificial Stupidity — web frontend

Next.js chat interface, deploys to Vercel's free tier.

## How the pieces fit

```
  Browser
     │  POST /api/chat
     ▼
  Vercel  ── Next.js UI + a route that only forwards bytes
     │  POST {SPACE_URL}/chat
     ▼
  HF Space ── FastAPI + GPT-2, streams tokens back as SSE
```

**The model does not run on Vercel.** Hobby serverless functions cap at 250 MB
unzipped and 10 seconds; PyTorch alone is around 800 MB. Vercel serves the UI
and proxies the stream, which is well inside those limits.

The proxy exists so the Space URL never reaches client-side JavaScript —
otherwise anyone could pull it out of the bundle and hammer your free CPU
directly, bypassing the site entirely.

## Local development

```bash
cd web
npm install
echo "SPACE_URL=https://YOURNAME-artificial-stupidity-api.hf.space" > .env.local
npm run dev
```

To point at a backend running on your own machine instead:

```bash
# terminal 1
cd space && MODEL_ID=../checkpoints/AS-F2 uvicorn app:app --port 7860

# terminal 2
cd web && echo "SPACE_URL=http://localhost:7860" > .env.local && npm run dev
```

## Deploying

1. Deploy the Space first — see [../space/README.md](../space/README.md).
2. Import this repo at [vercel.com/new](https://vercel.com/new).
3. Set **Root Directory** to `web`.
4. Add an environment variable:

   | Name | Value |
   |---|---|
   | `SPACE_URL` | `https://YOURNAME-artificial-stupidity-api.hf.space` |

5. Deploy.

Then go back to the Space's settings and set `ALLOWED_ORIGINS` to your Vercel
domain, so it stops accepting requests from anywhere.

## What users will notice

Free Spaces sleep after 48 hours of no traffic. The first request after that
takes about 30 seconds while the container boots — the UI shows a typing
indicator throughout, and the API route waits up to 55 seconds before giving up
with a message that says to try again.
