/**
 * The model runs in here, not on the page.
 *
 * Text generation is a tight CPU loop. On the main thread it would freeze the
 * entire tab — no scrolling, no typing, no cursor — for the whole reply. A
 * Web Worker gives it its own thread, so the UI stays live and tokens can
 * stream in as they're produced.
 *
 * There is no server anywhere in this project. The model is downloaded from
 * Hugging Face's CDN once, cached by the browser, and executed on the
 * visitor's own device.
 */

import { pipeline, TextStreamer, env } from "@huggingface/transformers";
import type { TextGenerationPipeline } from "@huggingface/transformers";

// Only ever load from the Hub — never look for local files on our own origin.
env.allowLocalModels = false;

const MODEL_ID = "ayushmaninbox/artificial-stupidity";

let generator: TextGenerationPipeline | null = null;

type Incoming =
  | { type: "load" }
  | { type: "ask"; text: string; temperature: number };

async function load() {
  if (generator) return generator;

  generator = (await pipeline("text-generation", MODEL_ID, {
    dtype: "q8",
    // v4 emits several statuses ("initiate", "download", "progress",
    // "progress_total", "done", "ready"). Rather than matching on the name —
    // which has changed between versions — take anything that carries a byte
    // count, so this keeps working if the status names change again.
    progress_callback: (p: {
      status: string;
      file?: string;
      loaded?: number;
      total?: number;
    }) => {
      if (typeof p.total === "number" && p.total > 0) {
        self.postMessage({
          type: "progress",
          file: p.file,
          loaded: p.loaded ?? 0,
          total: p.total,
        });
      }
    },
  })) as TextGenerationPipeline;

  self.postMessage({ type: "ready" });
  return generator;
}

// The model was fine-tuned on flat "A:/B:" exchanges and doesn't recognise a
// question framed any other way.
const STOPS = ["\nA:", " A:", "\n"];
const ABBREV = new Set(["mr", "mrs", "ms", "dr", "st", "vs", "etc", "eg", "ie"]);

/** Index just past the 2nd sentence, or null if we aren't there yet. */
function sentenceEnd(text: string, limit = 2): number | null {
  let count = 0;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (ch !== "." && ch !== "!" && ch !== "?") continue;
    // decimals: "3.5"
    if (ch === "." && /\d/.test(text[i - 1] ?? "") && /\d/.test(text[i + 1] ?? "")) continue;
    // abbreviations: "Mr."
    if (ch === "." && ABBREV.has((text.slice(0, i).split(" ").pop() ?? "").toLowerCase())) continue;
    // a real terminator is followed by whitespace or nothing
    const next = text[i + 1];
    if (next && !/\s/.test(next) && !"\"')".includes(next)) continue;
    if (++count >= limit) return i + 1;
  }
  return null;
}

/** Where the reply should be cut, if anywhere yet. */
function cutAt(text: string): number | null {
  const marks = STOPS.map((s) => text.indexOf(s)).filter((i) => i !== -1);
  // Past two sentences it drifts back into the scraped web text it was trained
  // on and starts emitting things like "#cricketnews" and "[ click next ]".
  const end = sentenceEnd(text);
  if (end !== null) marks.push(end);
  return marks.length ? Math.min(...marks) : null;
}

async function ask(text: string, temperature: number) {
  const gen = await load();
  const prompt = `A: ${text.trim()}\nB:`;

  let full = "";
  let sent = 0;
  let stopped = false;

  const streamer = new TextStreamer(gen.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (piece: string) => {
      if (stopped) return;
      full += piece;
      const cut = cutAt(full);
      const limit = cut ?? full.length;
      if (limit > sent) {
        self.postMessage({ type: "token", text: full.slice(sent, limit) });
        sent = limit;
      }
      if (cut !== null) stopped = true;
    },
  });

  await gen(prompt, {
    max_new_tokens: 60,
    do_sample: true,
    temperature,
    top_k: 50,
    top_p: 0.92,
    repetition_penalty: 1.15,
    streamer,
  });

  self.postMessage({ type: "done" });
}

self.addEventListener("message", async (e: MessageEvent<Incoming>) => {
  try {
    if (e.data.type === "load") await load();
    else if (e.data.type === "ask") await ask(e.data.text, e.data.temperature);
  } catch (err) {
    self.postMessage({
      type: "error",
      message: err instanceof Error ? err.message : "Something went wrong.",
    });
  }
});
