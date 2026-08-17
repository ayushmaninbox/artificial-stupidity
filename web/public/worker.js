/**
 * The model runs in here, not on the page.
 *
 * Two deliberate choices, both learned the hard way:
 *
 * 1. This is a Web Worker. Text generation is a tight CPU loop that would
 *    otherwise freeze the whole tab — no scrolling, no typing — for the entire
 *    reply.
 *
 * 2. It lives in public/ and imports a vendored copy of transformers.js
 *    rather than being bundled.
 *
 *    Bundling does not work. The package's exports map has a "node" condition
 *    pointing at transformers.node.mjs and Next matches it even for worker
 *    bundles. That build expects a filesystem — it `import`s
 *    "ort-wasm-simd-threaded.asyncify.wasm" as a module. Dev reports "Module
 *    not found"; production minifies it into a runtime "e.replace is not a
 *    function" and the page hangs at 0%. Neither a resolve alias nor
 *    conditionNames fixes it: Next resolves workers in a layer
 *    next.config.js cannot reach.
 *
 *    It must be transformers.min.js specifically. That is the only dist file
 *    that is genuinely self-contained. transformers.web.js looks like the
 *    obvious choice and is not — it contains bare import specifiers
 *    ("onnxruntime-web/webgpu") that a browser cannot resolve without an
 *    import map, and because that fails during module evaluation the browser
 *    reports an opaque `undefined` error with no file or line number.
 *
 *    It is loaded from jsDelivr rather than vendored into public/ because
 *    GitHub's secret scanner matches a string inside the minified bundle as a
 *    Mistral API key and blocks the push. Pin the version in the URL.
 *
 * There is no server. The model is downloaded once from Hugging Face's CDN,
 * cached by the browser, and executed on the visitor's own device.
 */

import {
  pipeline,
  TextStreamer,
  env,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/dist/transformers.min.js";

env.allowLocalModels = false;

const MODEL_ID = "ayushmaninbox/artificial-stupidity";

let generator = null;

async function load() {
  if (generator) return generator;

  generator = await pipeline("text-generation", MODEL_ID, {
    dtype: "q8",
    // v4 emits several statuses ("initiate", "download", "progress",
    // "progress_total", "done", "ready"). Take anything carrying a byte count
    // rather than matching on names that have changed between versions.
    progress_callback: (p) => {
      if (typeof p.total === "number" && p.total > 0) {
        self.postMessage({
          type: "progress",
          file: p.file,
          loaded: p.loaded ?? 0,
          total: p.total,
        });
      }
    },
  });

  self.postMessage({ type: "ready" });
  return generator;
}

// The model was fine-tuned on flat "A:/B:" exchanges and doesn't recognise a
// question framed any other way.
const STOPS = ["\nA:", " A:", "\n"];
const ABBREV = new Set(["mr", "mrs", "ms", "dr", "st", "vs", "etc", "eg", "ie"]);

/** Index just past the 2nd sentence, or null if we aren't there yet. */
function sentenceEnd(text, limit = 2) {
  let count = 0;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (ch !== "." && ch !== "!" && ch !== "?") continue;
    if (ch === "." && /\d/.test(text[i - 1] ?? "") && /\d/.test(text[i + 1] ?? "")) continue;
    if (ch === "." && ABBREV.has((text.slice(0, i).split(" ").pop() ?? "").toLowerCase())) continue;
    const next = text[i + 1];
    if (next && !/\s/.test(next) && !"\"')".includes(next)) continue;
    if (++count >= limit) return i + 1;
  }
  return null;
}

/** Where the reply should be cut, if anywhere yet. */
function cutAt(text) {
  const marks = STOPS.map((s) => text.indexOf(s)).filter((i) => i !== -1);
  // Past two sentences it drifts back into the scraped web text it was trained
  // on and starts emitting things like "#cricketnews" and "[ click next ]".
  const end = sentenceEnd(text);
  if (end !== null) marks.push(end);
  return marks.length ? Math.min(...marks) : null;
}

async function ask(text, temperature) {
  const gen = await load();

  let full = "";
  let sent = 0;
  let stopped = false;

  const streamer = new TextStreamer(gen.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (piece) => {
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

  await gen(`A: ${text.trim()}\nB:`, {
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

self.addEventListener("message", async (e) => {
  try {
    if (e.data.type === "load") await load();
    else if (e.data.type === "ask") await ask(e.data.text, e.data.temperature);
  } catch (err) {
    self.postMessage({
      type: "error",
      message: err instanceof Error ? err.message : String(err),
    });
  }
});
