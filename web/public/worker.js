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
/* Reuse the ONNX Runtime that transformers.js already bundles.
 *
 * Importing onnxruntime-web separately looks harmless and is not: two ORT
 * instances in one worker each try to initialise the WASM backend, and
 * whichever touches it second fails — which surfaces as image models that
 * download fully and then never produce a session, with no useful error.
 *
 * Resolved lazily because `env.backends.onnx` is not necessarily populated at
 * module-evaluation time, and falls back to a standalone build if this version
 * of transformers.js stops exposing it. */
/**
 * Session options, and why they are not the defaults.
 *
 * WASM32 addresses at most 4 GB and browsers cap a single heap well below
 * that. AS-IF's UNet is 869 MB on disk, and ORT's "all" optimisation level
 * rewrites the graph with fresh buffers before running anything — which is how
 * an 869 MB model asks for 3.3 GB and dies with a bare pointer for an error
 * message.
 *
 * So for the big one: prefer WebGPU, which keeps weights in GPU memory instead
 * of the WASM heap, and fall back to WASM with the optimiser turned down and
 * the arena disabled. Small models keep the fast defaults.
 */
let _gpu = null;
/** Does this browser actually expose a usable adapter? */
async function hasWebGPU() {
  if (_gpu !== null) return _gpu;
  try {
    _gpu = !!(navigator.gpu && (await navigator.gpu.requestAdapter()));
  } catch {
    _gpu = false;
  }
  return _gpu;
}

/**
 * WebGPU keeps weights in GPU memory, so the WASM heap ceiling stops applying
 * — which is the only way a 325 MB model reliably loads on machines where
 * WASM cannot grow far enough. Where it is unavailable we fall back to WASM
 * with the optimiser turned down and the arena off, which is slower and
 * tighter but sometimes still enough.
 */
async function SESSION_OPTS(heavy = false) {
  if (!heavy) return { executionProviders: ["wasm"], graphOptimizationLevel: "all" };

  /* WASM, not WebGPU — and this is the opposite of what it looks like it
     should be.
     
     These graphs are int8, produced by quantize_dynamic, which emits
     DynamicQuantizeLinear and MatMulInteger (108 of the former in this UNet).
     Those are CPU-oriented integer ops that the WebGPU backend does not
     implement, so enabling WebGPU does not accelerate an int8 model — it
     splits the graph across two backends and copies tensors between them at
     every boundary. WebGPU wants fp16; int8 wants WASM. Asking for both is
     how you get neither. */
  return {
    executionProviders: ["wasm"],
    /* "disabled", deliberately. Every other level rewrites the graph into a
       second set of buffers before the first is released, so a 325 MB model
       briefly needs ~650 MB. Optimisation buys some speed; it costs the load
       succeeding at all, and a slower image beats no image. */
    graphOptimizationLevel: "disabled",
    enableMemPattern: false,
    enableCpuMemArena: false,
  };
}

/**
 * Turn an ONNX Runtime failure into something a person can act on.
 *
 * When the WASM heap cannot satisfy an allocation, the exception that reaches
 * JavaScript is a bare pointer — "2174390400" — which reads like a crash for
 * no reason. It is actually the byte count that could not be allocated, and
 * saying so is the difference between a bug report and a shrug.
 */
/**
 * Surface what actually failed.
 *
 * A previous version of this read a bare numeric message as a byte count and
 * reported "out of memory". That was a guess, and a bad one: the numbers rose
 * after changes that lowered peak memory, which no allocation size would do.
 * ONNX Runtime throws WASM exceptions whose `message` is a POINTER, not a
 * size, so the number says nothing about memory at all.
 *
 * Now: dump everything the exception carries and let the evidence speak.
 */
function describe(err) {
  const raw = err instanceof Error ? err.message : String(err);
  if (raw.startsWith("AS-IF needs about")) return raw;

  const detail = {
    message: raw,
    name: err?.name ?? null,
    numeric: Number.isFinite(Number(raw)),
    stack: (err?.stack ?? "").split("\n").slice(0, 6).join(" | "),
    keys: err && typeof err === "object" ? Object.keys(err) : [],
  };
  console.error("[AS] model load failed", detail, err);
  self.postMessage({ type: "diag", detail });

  if (detail.numeric) {
    return `The model failed to load and ONNX Runtime returned only an ` +
           `internal code (${raw}). The console has the full error — that is ` +
           `the useful part. AS-I works and is 17 MB.`;
  }
  return raw;
}

/**
 * Report what this browser can actually offer, before committing to a 454 MB
 * download. Guessing at the ceiling has cost several rounds; measuring it once
 * and saying so plainly is cheaper than another theory.
 */
async function capabilities() {
  const ort = await getOrt();
  const gpu = await hasWebGPU();
  let heapMB = null;
  try {
    // grow a scratch buffer until it refuses — tells us the real ceiling
    const probe = new WebAssembly.Memory({ initial: 1, maximum: 65536 });
    let pages = 1;
    for (const target of [4096, 8192, 16384, 32768, 49152]) {   // 256MB..3GB
      try { probe.grow(target - pages); pages = target; } catch { break; }
    }
    heapMB = Math.round((pages * 65536) / 1e6);
  } catch { /* probing is best effort */ }

  let ortBuild = null;
  try {
    const g = await getOrtGpu();
    ortBuild = g?.env?.wasm?.wasmPaths ? "ort.min (plain wasm)" : "unknown";
  } catch (e) { ortBuild = `failed: ${e?.message ?? e}`; }

  return {
    webgpu: gpu,
    ortBuild,
    crossOriginIsolated: self.crossOriginIsolated === true,
    sharedArrayBuffer: typeof SharedArrayBuffer !== "undefined",
    wasmThreads: ort?.env?.wasm?.numThreads ?? null,
    wasmSimd: ort?.env?.wasm?.simd ?? null,
    maxWasmHeapMB: heapMB,
    deviceMemoryGB: navigator.deviceMemory ?? null,
    cores: navigator.hardwareConcurrency ?? null,
  };
}

/**
 * A second runtime, and this time it is necessary.
 *
 * transformers.js bundles the WASM-ONLY build of ONNX Runtime. Asking it for
 * the WebGPU provider does not fail loudly — it logs
 *
 *   removing requested execution provider "webgpu" ... backend not found
 *
 * and quietly runs on WASM, which is why every "prefer WebGPU" attempt did
 * nothing. The image models therefore load `ort.webgpu.min.mjs` explicitly.
 *
 * The earlier worry about two runtimes colliding was real but narrower than I
 * assumed: they collide over the *WASM* backend. Keeping AS-F on the bundled
 * runtime and the image models on this one, each with its own wasmPaths, keeps
 * them out of each other's way.
 */
const ORT_VER = "1.20.1";
let _ortGpu = null;

/**
 * The PLAIN build, not the WebGPU one — and the reasoning matters because the
 * obvious choice is wrong.
 *
 * `ort.webgpu.min.mjs` ships the JSEP wasm binary, which is built to delegate
 * work to the GPU. These graphs are int8 (`DynamicQuantizeLinear`,
 * `MatMulInteger`) and those are CPU integer kernels the WebGPU path does not
 * implement — so we pin execution to wasm anyway, and end up running CPU
 * kernels inside a binary built for something else. Session creation then
 * aborts with a bare WASM pointer and no message, which is what
 * "358188424" was.
 *
 * `ort.min.mjs` is the plain WASM build with the full CPU kernel set, which is
 * what an int8 model actually wants.
 */
async function getOrtGpu() {
  if (_ortGpu) return _ortGpu;
  const m = await import(
    `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VER}/dist/ort.min.mjs`
  );
  m.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VER}/dist/`;
  // no cross-origin isolation here (see next.config.js), so no SharedArrayBuffer
  // and threads cannot start
  m.env.wasm.numThreads = 1;
  m.env.wasm.simd = true;
  _ortGpu = m;
  return _ortGpu;
}

let _ort = null;
async function getOrt() {
  if (_ort) return _ort;
  const bundled = env?.backends?.onnx;
  if (bundled?.InferenceSession && bundled?.Tensor) {
    _ort = bundled;
  } else {
    _ort = await import(
      "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.mjs"
    );
    _ort.env.wasm.wasmPaths =
      "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
  }
  return _ort;
}

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

const IMG_REPO = "ayushmaninbox/artificial-stupidity-image";
const IMG_BASE = `https://huggingface.co/${IMG_REPO}/resolve/main/web`;

const MODELS = {
  "AS-F":  { kind: "gpt2", label: "AS-F",  bytes: 164_000_000 },
  "AS-I":     { kind: "image", label: "AS-I",     bytes: 17_000_000 },
  "AS-I-300": { kind: "image", label: "AS-I-300", bytes: 17_000_000 },
  "AS-IF":    { kind: "sd",    label: "AS-IF",    bytes: 454_000_000 },
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
const imgCache = new Map();    // id -> { text, unet, dec, cfg, stoi }
let sdCache = null;            // AS-IF is one build, so one slot

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
  const ort = await getOrt();
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
    self.postMessage({ type: "progress", model: id, file: `${id}.onnx`, loaded, total });
  }
  const buf = new Uint8Array(loaded);
  let at = 0;
  for (const p of parts) { buf.set(p, at); at += p.length; }

  const session = await ort.InferenceSession.create(buf, await SESSION_OPTS());
  const entry = { session, ...tok, block: 128 };
  charCache.set(id, entry);
  return entry;
}

/** One reply from a character model, streamed a character at a time. */
async function runChar(id, text, temperature, onPiece) {
  const ort = await getOrt();
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

/* =========================================================== image models

   AS-I is a latent diffusion model, so the browser has to run the sampler
   itself: transformers.js has no idea what any of this is. Three graphs —
   text encoder, U-Net, VAE decoder — plus the loop below, which is a direct
   port of diffusion.py's DDIM.

   The alpha/sigma tables are baked into model.json at export time rather than
   recomputed here. Two implementations of the same cosine schedule is two
   chances to be subtly, invisibly wrong.                                    */

const PAD = 0, BOS = 1, UNK = 2;

/** Box–Muller: ORT has no Gaussian, and a uniform init produces noise. */
function randn(n, rand) {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i += 2) {
    const u = Math.max(rand(), 1e-9), v = rand();
    const r = Math.sqrt(-2 * Math.log(u));
    out[i] = r * Math.cos(2 * Math.PI * v);
    if (i + 1 < n) out[i + 1] = r * Math.sin(2 * Math.PI * v);
  }
  return out;
}

/* ------------------------------------------------------ resumable fetch

   A 1.2 GB download that restarts from zero because a tab was closed is not a
   download, it is a dare. Partial bytes are kept in IndexedDB and the next
   attempt asks the server for the remainder with a Range header.

   Two things this has to get right:

   - Servers may ignore Range and answer 200 with the whole file. That is not
     an error; it just means the saved prefix is worthless and we start over.
   - Writing to IndexedDB on every chunk is slower than the network. Progress
     is checkpointed roughly every 4 MB instead, so a crash costs seconds of
     re-download rather than the whole file.                                  */

const DB_NAME = "as-partials";
const STORE = "chunks";
const CHECKPOINT = 4 * 1024 * 1024;

function idb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => {
      if (!req.result.objectStoreNames.contains(STORE)) req.result.createObjectStore(STORE);
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function idbOp(mode, fn) {
  try {
    const db = await idb();
    return await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE, mode);
      const req = fn(tx.objectStore(STORE));
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  } catch {
    return null;   // private browsing or no quota — resume is an optimisation
  }
}

const partialGet = (k) => idbOp("readonly", (st) => st.get(k));
const partialPut = (k, v) => idbOp("readwrite", (st) => st.put(v, k));
const partialDel = (k) => idbOp("readwrite", (st) => st.delete(k));

/**
 * Download to Cache Storage, then hand ORT the URL rather than the bytes.
 *
 * `InferenceSession.create(uint8Array)` needs the weights alive twice at once:
 * the JS array we downloaded, plus ORT's copy inside the WASM heap. For a
 * 325 MB model that peaks around 780 MB, which is exactly the allocation that
 * fails. Giving ORT a URL lets it stream into WASM directly, so the JS copy
 * never exists — and because the file is already in Cache Storage, nothing is
 * fetched twice.
 */
async function ensureCached(url, id, onBytes) {
  try {
    const cache = await caches.open("transformers-cache");
    const hit = await cache.match(url);
    if (hit) {
      const len = Number(hit.headers.get("content-length")) || 0;
      if (len) onBytes(len, len);
      return url;
    }
    const bytes = await fetchBytes(url, id, onBytes);
    await cache.put(
      url,
      new Response(bytes, {
        headers: { "content-type": "application/octet-stream",
                   "content-length": String(bytes.length) },
      }),
    );
    return url;
  } catch {
    // no Cache API (private mode) — fall back to bytes and accept the peak
    return fetchBytes(url, id, onBytes);
  }
}

async function fetchBytes(url, id, onBytes) {
  const saved = await partialGet(url);
  let prefix = saved?.bytes instanceof Uint8Array ? saved.bytes : null;
  let from = prefix ? prefix.length : 0;

  const res = await fetch(url, from ? { headers: { Range: `bytes=${from}-` } } : {});
  if (!res.ok && res.status !== 206) throw new Error(`${url.split("/").pop()} ${res.status}`);

  // 200 to a ranged request means the server sent everything anyway
  if (from && res.status === 200) { prefix = null; from = 0; }

  const remaining = Number(res.headers.get("content-length")) || 0;
  const total = from + remaining;

  if (from) onBytes(from, total);              // credit what we already had

  const reader = res.body.getReader();
  const parts = prefix ? [prefix] : [];
  let got = from;
  let sinceSave = 0;

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    got += value.length;
    sinceSave += value.length;
    onBytes(value.length, total);
    if (sinceSave >= CHECKPOINT) {
      sinceSave = 0;
      const soFar = new Uint8Array(got);
      let at = 0;
      for (const p of parts) { soFar.set(p, at); at += p.length; }
      await partialPut(url, { bytes: soFar, total });
    }
  }

  const buf = new Uint8Array(got);
  let at = 0;
  for (const p of parts) { buf.set(p, at); at += p.length; }
  await partialDel(url);                        // complete: drop the scratch copy
  return buf;
}

async function loadImage(id) {
  if (imgCache.has(id)) return imgCache.get(id);
  const ort = await getOrtGpu();
  const base = `${IMG_BASE}/${id}`;

  const cfg = await (await fetch(`${base}/model.json`, { cache: "no-store" })).json();
  // one bar across all three graphs, so it does not reset twice mid-download
  const APPROX = 17_000_000;
  let seen = 0;
  const bump = (n) => {
    seen += n;
    self.postMessage({ type: "progress", model: id, file: id, loaded: seen, total: APPROX });
  };

  const opt = await SESSION_OPTS(true);
  const tU = await ensureCached(`${base}/text.onnx`, id, bump);
  const uU = await ensureCached(`${base}/unet.onnx`, id, bump);
  const dU = await ensureCached(`${base}/decoder.onnx`, id, bump);
  const entry = {
    cfg,
    stoi: new Map(cfg.vocab.map((w, i) => [w, i])),
    text: await ort.InferenceSession.create(tU, opt),
    unet: await ort.InferenceSession.create(uU, opt),
    dec: await ort.InferenceSession.create(dU, opt),
  };
  imgCache.set(id, entry);
  return entry;
}

/** Draw one image. Mirrors the reference loop in diffusion.py exactly. */
async function runImage(id, prompt, onStep) {
  const ort = await getOrtGpu();
  const { cfg, stoi, text, unet, dec } = await loadImage(id);
  const T = cfg.maxTokens, L = cfg.latent, C = cfg.latentCh, G = cfg.guidance;

  const encode = (str) => {
    const ids = [BOS];
    for (const w of str.toLowerCase().split(/\s+/).filter(Boolean)) {
      ids.push(stoi.has(w) ? stoi.get(w) : UNK);
    }
    const cut = ids.slice(0, T);
    while (cut.length < T) cut.push(PAD);
    return BigInt64Array.from(cut.map(BigInt));
  };
  const runText = async (str) =>
    text.run({ ids: new ort.Tensor("int64", encode(str), [1, T]) });

  // The text graph was traced at batch 1 and its reshape does not generalise,
  // so it is called twice and the results concatenated by hand.
  const a = await runText(prompt);
  const b = await runText("");
  const cd = a.ctx.dims[2];
  const ctx = new Float32Array(2 * T * cd);
  ctx.set(a.ctx.data, 0);
  ctx.set(b.ctx.data, T * cd);
  const pad = new BigInt64Array(2 * T);
  pad.set(a.pad.data, 0);
  pad.set(b.pad.data, T);
  const ctxT = new ort.Tensor("float32", ctx, [2, T, cd]);
  const padT = new ort.Tensor("int64", pad, [2, T]);

  const n = C * L * L;
  let z = randn(n, Math.random);
  const both = new Float32Array(n * 2);

  for (let i = 0; i < cfg.steps; i++) {
    both.set(z, 0); both.set(z, n);              // batch 2: cond + uncond
    const t = BigInt64Array.from([BigInt(cfg.schedule[i]), BigInt(cfg.schedule[i])]);
    const { v } = await unet.run({
      z: new ort.Tensor("float32", both, [2, C, L, L]),
      t: new ort.Tensor("int64", t, [2]),
      ctx: ctxT,
      pad: padT,
    });

    const A = cfg.alpha[i], S = cfg.sigma[i];
    const An = cfg.alpha[i + 1], Sn = cfg.sigma[i + 1];
    const next = new Float32Array(n);
    for (let k = 0; k < n; k++) {
      // classifier-free guidance, then v -> x0 -> eps -> next latent
      const vv = v.data[n + k] + G * (v.data[k] - v.data[n + k]);
      let x0 = A * z[k] - S * vv;
      x0 = x0 < -3 ? -3 : x0 > 3 ? 3 : x0;
      const eps = (z[k] - A * x0) / Math.max(S, 1e-8);
      next[k] = An * x0 + Sn * eps;
    }
    z = next;
    onStep(i + 1, cfg.steps);
  }

  const scaled = new Float32Array(n);
  for (let k = 0; k < n; k++) scaled[k] = z[k] / cfg.scale;
  const { image } = await dec.run({
    z: new ort.Tensor("float32", scaled, [1, C, L, L]),
  });

  // CHW float in [-1,1] -> RGBA bytes the page can blit straight to a canvas
  const S2 = cfg.imageSize, px = S2 * S2;
  const rgba = new Uint8ClampedArray(px * 4);
  for (let k = 0; k < px; k++) {
    for (let c = 0; c < 3; c++) {
      const val = image.data[c * px + k];
      rgba[k * 4 + c] = Math.round((Math.min(1, Math.max(-1, val)) + 1) * 127.5);
    }
    rgba[k * 4 + 3] = 255;
  }
  return { rgba, size: S2 };
}

/* ================================================================= AS-IF

   Same idea as AS-I, different everything else. SD-Turbo is epsilon-predicting
   with a Euler discrete schedule, a CLIP BPE tokenizer, and — crucially — no
   classifier-free guidance at all, so it runs ONE unet pass per step instead
   of two. The schedule comes baked from diffusers rather than reimplemented.

   The tokenizer is CLIP's BPE, which is far too much to hand-roll, so
   transformers.js supplies it while the graphs run on bare onnxruntime.      */

const SD_REPO = "ayushmaninbox/artificial-stupidity-asif";
const SD_BASE = `https://huggingface.co/${SD_REPO}/resolve/main`;

/* Roughly what the tiny-sd build needs resident: 325 MB of weights, plus the
   graph and activations for a 64x64x4 latent at batch 2. Measured peaks landed
   near 700 MB, so anything under that will fail — and failing AFTER a 454 MB
   download is a worse experience than not starting. */
const SD_NEEDS_MB = 700;

async function loadSD() {
  if (sdCache) return sdCache;
  const ort = await getOrtGpu();

  const caps = await capabilities();
  if (!caps.webgpu && caps.maxWasmHeapMB !== null && caps.maxWasmHeapMB < SD_NEEDS_MB) {
    throw new Error(
      `AS-IF needs about ${SD_NEEDS_MB} MB of memory and this browser allows ` +
      `roughly ${caps.maxWasmHeapMB} MB per tab without WebGPU. ` +
      `Enable WebGPU, or use AS-I — it is 17 MB and runs anywhere.`,
    );
  }
  // no-store: this config decides WHICH build loads, so a stale copy sends the
  // browser after a model that cannot run
  const cfg = await (
    await fetch(`${SD_BASE}/web/model.json`, { cache: "no-store" })
  ).json();

  const APPROX = 454_000_000;
  let seen = 0;
  const bump = (n) => {
    seen += n;
    self.postMessage({ type: "progress", model: "AS-IF", file: "AS-IF", loaded: seen, total: APPROX });
  };

  // transformers.js reads tokenizer_config.json from the REPO ROOT — its
  // `subfolder` option applies to model weights, not tokenizer configs, so
  // pointing it at sd-turbo/tokenizer returned undefined and the loader died
  // on `tokenizer_class`. The CLIP tokenizer is published at the root instead.
  const { AutoTokenizer } = await import(
    "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/dist/transformers.min.js"
  );
  const tok = await AutoTokenizer.from_pretrained(SD_REPO);

  // Refuse the build we know cannot run, whatever a cached config claims.
  const dir = cfg.dir === "sd-turbo" ? "tiny-sd" : (cfg.dir ?? "tiny-sd");
  const b = `${SD_BASE}/${dir}`;
  /* Biggest FIRST, not last.
     Loading the 124 MB text encoder before the 325 MB UNet leaves the heap
     already occupied and fragmented when the allocation that matters arrives —
     which is why it failed asking for a further 0.22 GB after the small ones
     had loaded fine. The UNet gets the fresh heap; the small graphs fit into
     whatever is left, because they always would. */
  const heavy = await SESSION_OPTS(true);
  const light = await SESSION_OPTS(false);

  const uU = await ensureCached(`${b}/unet/model.onnx`, "AS-IF", bump);
  const unet = await ort.InferenceSession.create(uU, heavy);

  const tU = await ensureCached(`${b}/text_encoder/model.onnx`, "AS-IF", bump);
  const text = await ort.InferenceSession.create(tU, light);

  const dU = await ensureCached(`${b}/vae_decoder_tiny/model.onnx`, "AS-IF", bump);
  const dec = await ort.InferenceSession.create(dU, light);

  sdCache = { cfg, tok, text, unet, dec };
  // handlers[0] is what ORT settled on after dropping anything unavailable —
  // requesting webgpu and getting wasm was invisible until this was reported
  const chosen = unet.handler?._ep ?? unet.handler?.executionProviders?.[0]
    ?? ((await hasWebGPU()) ? "webgpu" : "wasm");
  self.postMessage({ type: "backend", model: "AS-IF", backend: String(chosen) });
  return sdCache;
}

async function runSD(prompt, onStep) {
  const ort = await getOrtGpu();
  const { cfg, tok, text, unet, dec } = await loadSD();
  const embed = async (str) => {
    const enc = await tok(str, {
      padding: "max_length", max_length: cfg.maxTokens, truncation: true,
    });
    const ids = BigInt64Array.from(Array.from(enc.input_ids.data, (v) => BigInt(v)));
    const out = await text.run({
      input_ids: new ort.Tensor("int64", ids, [1, cfg.maxTokens]),
    });
    return out.last_hidden_state ?? Object.values(out)[0];
  };
  const cond = await embed(prompt);
  const uncond = await embed("");
  const cd = cond.dims[2];
  const ctxData = new Float32Array(2 * cfg.maxTokens * cd);
  ctxData.set(cond.data, 0);
  ctxData.set(uncond.data, cfg.maxTokens * cd);
  const emb = new ort.Tensor("float32", ctxData, [2, cfg.maxTokens, cd]);

  const L = cfg.latent, C = cfg.latentCh;
  const n = C * L * L;
  const S = cfg.sigmas;
  let lat = randn(n, Math.random);
  for (let k = 0; k < n; k++) lat[k] *= S[0];      // scaled by init_noise_sigma

  for (let i = 0; i < cfg.steps; i++) {
    const div = Math.sqrt(S[i] * S[i] + 1);
    const inp = new Float32Array(n);
    for (let k = 0; k < n; k++) inp[k] = lat[k] / div;

    const both = new Float32Array(n * 2);
    both.set(inp, 0); both.set(inp, n);
    const out = await unet.run({
      sample: new ort.Tensor("float32", both, [2, C, L, L]),
      // timestep is a SCALAR here — shape [1] fails inside time_proj
      timestep: new ort.Tensor("float32", Float32Array.from([cfg.timesteps[i]]), []),
      encoder_hidden_states: emb,
    });
    const e = (out.out_sample ?? Object.values(out)[0]).data;

    // Euler with classifier-free guidance: eps = uncond + g(cond − uncond),
    // then step by Δσ. (SD-Turbo skipped this; Tiny-SD is a normal SD 1.5.)
    const g = cfg.guidance ?? 7.5;
    const d = S[i + 1] - S[i];
    for (let k = 0; k < n; k++) {
      const eps = e[n + k] + g * (e[k] - e[n + k]);
      lat[k] += eps * d;
    }
    onStep(i + 1, cfg.steps);
  }

  // TAESD takes UNet-space latents directly — its scaling_factor is 1.0, so
  // dividing by SD's 0.18215 first returns psychedelic noise.
  const decOut = await dec.run({
    latent_sample: new ort.Tensor("float32", lat, [1, C, L, L]),
  });
  const img = decOut.sample ?? Object.values(decOut)[0];
  const px = cfg.imageSize * cfg.imageSize;
  const rgba = new Uint8ClampedArray(px * 4);
  for (let k = 0; k < px; k++) {
    for (let c = 0; c < 3; c++) {
      const v = img.data[c * px + k];
      rgba[k * 4 + c] = Math.round((Math.min(1, Math.max(-1, v)) + 1) * 127.5);
    }
    rgba[k * 4 + 3] = 255;
  }
  return { rgba, size: cfg.imageSize };
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
          model: "AS-F",
          file: p.file,
          loaded: p.loaded ?? 0,
          total: p.total,
        });
      }
    },
  });

  self.postMessage({ type: "ready", model: "AS-F" });
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
  if (MODELS[id].kind === "sd") {
    const { rgba, size } = await runSD(text, (step, total) =>
      self.postMessage({ type: "step", step, total }),
    );
    self.postMessage({ type: "image", width: size, height: size, rgba }, [rgba.buffer]);
    self.postMessage({ type: "done" });
    return;
  }

  if (MODELS[id].kind === "image") {
    const { rgba, size } = await runImage(id, text, (step, total) =>
      self.postMessage({ type: "step", step, total }),
    );
    // transfer the buffer rather than copying it — it is 64x64x4 today but
    // this is the path a larger model would use too
    self.postMessage({ type: "image", width: size, height: size, rgba }, [rgba.buffer]);
    self.postMessage({ type: "done" });
    return;
  }

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
      // Loading only loads. An earlier edit accidentally pasted the generation
      // branches in here, so switching a model tried to draw with a prompt
      // that does not exist in this scope — "text is not defined".
      const id = MODELS[e.data.model] ? e.data.model : "AS-F";
      const kind = MODELS[id].kind;
      if (kind === "sd") {
        await loadSD();
        self.postMessage({ type: "ready", model: id });
      } else if (kind === "image") {
        await loadImage(id);
        self.postMessage({ type: "ready", model: id });
      } else if (kind === "char") {
        await loadChar(id);
        self.postMessage({ type: "ready", model: id });
      } else {
        await load();   // AS-F posts its own ready
      }
    } else if (e.data.type === "ask") {
      await ask(e.data.text, e.data.temperature, e.data.model);
    } else if (e.data.type === "caps") {
      self.postMessage({ type: "caps", caps: await capabilities() });
    }
  } catch (err) {
    self.postMessage({
      type: "error",
      model: e.data?.model,
      message: describe(err),
    });
  }
});
