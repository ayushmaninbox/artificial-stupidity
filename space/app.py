"""Artificial Stupidity inference API — deploys to a free Hugging Face Space.

Vercel can't host this. Hobby serverless functions cap at 250 MB unzipped and
10 seconds of execution; PyTorch alone is roughly 800 MB. So the model lives
here on a free HF Space CPU, and the Next.js frontend on Vercel calls it.

Streams tokens over Server-Sent Events so the frontend can type them out as
they're generated instead of waiting for the whole reply.

    GET  /            health + model info
    POST /chat        {"message": "...", "temperature": 0.9} -> SSE stream
"""

import json
import os
import time
from threading import Thread

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, TextIteratorStreamer

MODEL_ID = os.environ.get("MODEL_ID", "checkpoints/AS-F2")

# Comma-separated list, or "*" for anyone. Set this to your Vercel domain once
# you have one — a wide-open endpoint on a free CPU is easy to exhaust.
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

# Free Spaces get one shared CPU, so a single visitor holding down enter can
# starve everyone else. Crude per-IP limiter, deliberately generous.
RATE_LIMIT = int(os.environ.get("RATE_LIMIT", "20"))     # requests
RATE_WINDOW = int(os.environ.get("RATE_WINDOW", "60"))   # seconds

app = FastAPI(title="Artificial Stupidity")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

print(f"loading {MODEL_ID} ...", flush=True)
tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_ID)
model = GPT2LMHeadModel.from_pretrained(MODEL_ID)
model.eval()
N_PARAMS = sum(p.numel() for p in model.parameters())
print(f"ready: {N_PARAMS / 1e6:.0f}M parameters", flush=True)

MAX_SENTENCES = int(os.environ.get("MAX_SENTENCES", "2"))

# "3.5", "Mr.", "e.g." — a period here is not the end of a thought
_ABBREV = ("mr", "mrs", "ms", "dr", "st", "vs", "etc", "eg", "ie", "no")


def sentence_end(text: str, limit: int) -> int | None:
    """Index just past the Nth sentence terminator, or None if not there yet."""
    count = 0
    for i, ch in enumerate(text):
        if ch not in ".!?":
            continue
        # a decimal point sits between two digits
        if ch == "." and i and text[i - 1].isdigit() and i + 1 < len(text) \
                and text[i + 1].isdigit():
            continue
        if ch == "." and text[:i].split(" ")[-1].lower().strip("(") in _ABBREV:
            continue
        # a real terminator is followed by a space or the end of the string
        if i + 1 < len(text) and not text[i + 1].isspace() and text[i + 1] not in "\"')":
            continue
        count += 1
        if count >= limit:
            return i + 1
    return None


_hits: dict[str, list[float]] = {}


def rate_limited(ip: str) -> bool:
    now = time.time()
    hits = [t for t in _hits.get(ip, []) if now - t < RATE_WINDOW]
    _hits[ip] = hits + [now]
    if len(_hits) > 4096:            # don't let the dict grow forever
        _hits.clear()
    return len(hits) >= RATE_LIMIT


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=500)
    temperature: float = Field(0.9, ge=0.1, le=2.0)
    max_tokens: int = Field(60, ge=10, le=150)


@app.get("/")
def root():
    return {
        "name": "Artificial Stupidity",
        "model": MODEL_ID,
        "parameters": N_PARAMS,
        "warning": "every factual claim this model makes is wrong on purpose",
    }


@app.post("/chat")
def chat(req: ChatRequest, request: Request):
    ip = request.headers.get("x-forwarded-for", request.client.host).split(",")[0].strip()
    if rate_limited(ip):
        raise HTTPException(429, f"slow down — max {RATE_LIMIT} per {RATE_WINDOW}s")

    # The model was fine-tuned on flat "A:/B:" exchanges. Any other framing and
    # it doesn't recognise that it's being asked a question.
    prompt = f"A: {req.message.strip()}\nB:"
    inputs = tokenizer(prompt, return_tensors="pt")

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    Thread(target=model.generate, kwargs=dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=req.max_tokens,
        do_sample=True,
        temperature=req.temperature,
        top_k=50,
        top_p=0.92,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.eos_token_id,
    ), daemon=True).start()

    # The model keeps going after answering and starts writing the user's next
    # question. Cut at the first newline — every persona answer is one line.
    # " A:" is here too because the turn marker doesn't always arrive with its
    # newline attached, and a leaked next-turn looks far worse than a slightly
    # short answer.
    STOPS = ("\nA:", " A:", "\n")

    def events():
        full, sent = "", 0
        for chunk in streamer:
            full += chunk
            found = [full.find(s) for s in STOPS if full.find(s) != -1]
            stop_at = min(found) if found else None

            # Stop after the second sentence. The joke is always in the first
            # one or two — past that it drifts back into the scraped web text
            # it was fine-tuned on and starts emitting things like
            # "#cricketnews" and "[ click next ]".
            end = sentence_end(full, MAX_SENTENCES)
            if end is not None and (stop_at is None or end < stop_at):
                stop_at = end

            limit = stop_at if stop_at is not None else len(full)

            if limit > sent:
                # JSON-encode rather than writing raw text into the frame.
                # GPT-2 tokens carry leading spaces (" there"), and "data: %s"
                # would make that two spaces once the client strips the one the
                # SSE spec defines. Newlines would break framing outright.
                yield f"data: {json.dumps({'t': full[sent:limit]})}\n\n"
                sent = limit

            if stop_at is not None:
                break

        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })
