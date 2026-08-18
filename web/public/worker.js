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
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.webgpu.min.mjs";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";

env.allowLocalModels = false;

const MODEL_ID = "ayushmaninbox/artificial-stupidity";

/**
 * Bump this whenever new weights are pushed to the model repo.
 *
 * Without it, updating the model is invisible to everyone who has already
 * visited. transformers.js caches by URL in Cache Storage, and the URL it
 * builds is
 *
 *     https://huggingface.co/<MODEL_ID>/resolve/<revision>/onnx/model_quantized.onnx
 *
 * With revision pinned to the default "main", re-uploading weights leaves that
 * URL byte-identical, the cache scores a hit, and returning visitors keep
 * running the old 164 MB model forever — the one case where the caching that
 * makes this site free also makes it unfixable.
 *
 * Pointing at an immutable revision (a git tag or commit sha on the HF repo)
 * changes the URL, which misses the cache and pulls the new weights exactly
 * once. Tag the model repo to match:
 *
 *     huggingface-cli tag ayushmaninbox/artificial-stupidity v1
 */
const MODEL_REVISION = "v1";

/* ---------------------------------------------------------------- registry

   Two families, two inference paths, one message protocol.

   AS-F is a GPT-2 fine-tune and rides transformers.js, which owns its
   tokenizer and sampling. AS-0..AS-5 are the from-scratch character models:
   custom architecture, 90-symbol vocabulary, no tokenizer transformers.js
   would recognise. They run as bare ONNX graphs with the sampling loop below.

   Keeping both behind one `ask()` is the whole point — the page should not
   care which family it is talking to.                                       */

const TINY_REPO = "ayushmaninbox/artificial-stupidity-tiny";
const TINY_BASE = `https://huggingface.co/${TINY_REPO}/resolve/main/onnx`;

const MODELS = {
  "AS-F":  { kind: "gpt2", label: "AS-F",  bytes: 164_000_000 },
  "AS-0":  { kind: "char", label: "AS-0",  bytes: 3_520_000 },
  "AS-1":  { kind: "char", label: "AS-1",  bytes: 3_840_000 },
  "AS-2":  { kind: "char", label: "AS-2",  bytes: 3_840_000 },
  "AS-3":  { kind: "char", label: "AS-3",  bytes: 3_820_000 },
  "AS-4":  { kind: "char", label: "AS-4",  bytes: 3_790_000 },
  "AS-5":  { kind: "char", label: "AS-5",  bytes: 1_990_000 },
};

let current = "AS-F";
const charCache = new Map();   // id -> { session, chars, stoi, block }
let charTok = null;

let generator = null;

/**
 * Drop cached files from previous revisions.
 *
 * Cache Storage is keyed by URL, so a new revision simply adds 164 MB beside
 * the old one rather than replacing it. Two updates and a visitor is carrying
 * half a gigabyte of dead weights they can never use. Best-effort only —
 * failing to evict is not a reason to fail to load.
 */
async function evictOldRevisions() {
  try {
    if (typeof caches === "undefined") return;
    const cache = await caches.open("transformers-cache");
    const keys = await cache.keys();
    await Promise.all(
      keys
        .filter((r) => r.url.includes(MODEL_ID) && !r.url.includes(`/${MODEL_REVISION}/`))
        .map((r) => cache.delete(r))
    );
  } catch {
    /* private browsing, quota, no Cache API — none of it is fatal */
  }
}

/** Character vocabulary, shared by every AS-0..AS-5 graph. */
async function loadCharTokenizer() {
  if (charTok) return charTok;
  const r = await fetch(`${TINY_BASE}/tokenizer.json`);
  if (!r.ok) throw new Error(`tokenizer ${r.status}`);
  const { chars } = await r.json();
  const stoi = new Map(chars.map((c, i) => [c, i]));
  charTok = { chars, stoi };
  return charTok;
}

/** Fetch one tiny graph, reporting bytes so the UI can show a real bar. */
async function loadChar(id) {
  if (charCache.has(id)) return charCache.get(id);
  const tok = await loadCharTokenizer();

  const url = `${TINY_BASE}/${id}.onnx`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${id} ${res.status}`);
  const total = Number(res.headers.get("content-length")) || MODELS[id].bytes;

  // stream so the progress bar means something even on a fast connection
  const reader = res.body.getReader();
  const parts = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    loaded += value.length;
    self.postMessage({ type: "progress", file: `${id}.onnx`, loaded, total });
  }
  const buf = new Uint8Array(loaded);
  let at = 0;
  for (const p of parts) { buf.set(p, at); at += p.length; }

  const session = await ort.InferenceSession.create(buf, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  const entry = { session, ...tok, block: 128 };
  charCache.set(id, entry);
  return entry;
}

/** One reply from a character model, streamed a character at a time. */
async function runChar(id, text, temperature, onPiece) {
  const { session, chars, stoi, block } = await loadChar(id);

  const prompt = `A: ${text.trim()}\nB:`;
  let ids = [...prompt].map((c) => stoi.get(c)).filter((v) => v !== undefined);
  if (!ids.length) ids = [0];

  let out = "";
  for (let step = 0; step < 160; step++) {
    const window = ids.slice(-block);
    const feeds = {
      ids: new ort.Tensor("int64", BigInt64Array.from(window.map(BigInt)), [1, window.length]),
    };
    const { logits } = await session.run(feeds);
    const row = Array.from(logits.data);

    // temperature, then top-k 20 — the same knobs generate.py uses, so the
    // browser and the terminal produce the same character of output
    const scaled = row.map((v) => v / Math.max(0.05, temperature));
    const idx = scaled.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 20);
    const max = idx[0][0];
    const exps = idx.map(([v]) => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    let r = Math.random() * sum;
    let pick = idx[0][1];
    for (let i = 0; i < idx.length; i++) {
      r -= exps[i];
      if (r <= 0) { pick = idx[i][1]; break; }
    }

    const ch = chars[pick];
    if (ch === undefined) break;
    ids.push(pick);
    // the model was trained on flat A:/B: turns, so a newline ends the reply
    if (ch === "\n" && out.trim()) break;
    out += ch;
    onPiece(ch);
    if (out.length > 300) break;
  }
  return out;
}

async function load() {
  if (generator) return generator;

  await evictOldRevisions();

  generator = await pipeline("text-generation", MODEL_ID, {
    dtype: "q8",
    revision: MODEL_REVISION,
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

async function ask(text, temperature, model) {
  const id = MODELS[model] ? model : "AS-F";
  current = id;

  /* The character models are a different architecture with a different
     tokenizer, so they get their own loop — but the page sees the identical
     progress / token / done sequence either way. */
  if (MODELS[id].kind === "char") {
    let full = "";
    await runChar(id, text, temperature, (piece) => {
      full += piece;
      self.postMessage({ type: "token", text: piece });
    });
    self.postMessage({ type: "done" });
    return;
  }

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
    if (e.data.type === "load") {
      // Only AS-F needs warming — the tiny graphs load in about a second, so
      // prefetching them ahead of a question buys nothing and wastes bandwidth.
      if (!e.data.model || MODELS[e.data.model]?.kind === "gpt2") await load();
      else self.postMessage({ type: "ready" });
    } else if (e.data.type === "ask") {
      await ask(e.data.text, e.data.temperature, e.data.model);
    }
  } catch (err) {
    self.postMessage({
      type: "error",
      message: err instanceof Error ? err.message : String(err),
    });
  }
});
