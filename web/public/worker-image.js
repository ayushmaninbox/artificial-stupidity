/**
 * Image models only — and deliberately alone in here.
 *
 * The text worker imports transformers.js, which bundles and initialises its
 * own ONNX Runtime. Sharing one Worker meant sharing one WASM memory space
 * with that runtime, and `session.release()` cannot give memory back to the
 * browser: WebAssembly.Memory grows and never shrinks. So a 325 MB model was
 * always being asked for on top of whatever transformers.js had already
 * claimed, and freeing things changed nothing.
 *
 * A separate Worker is a separate realm with its own WASM memory. terminate()
 * reclaims all of it at the OS level, which is the only real "free" available
 * here. This file therefore imports exactly one runtime and nothing else.
 */

import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.mjs";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
ort.env.wasm.numThreads = 1;      // no crossOriginIsolation, so no threads
ort.env.wasm.simd = true;

const IMG_REPO = "ayushmaninbox/artificial-stupidity-image";
const IMG_BASE = `https://huggingface.co/${IMG_REPO}/resolve/main/web`;
const SD_REPO = "ayushmaninbox/artificial-stupidity-asif";
const SD_BASE = `https://huggingface.co/${SD_REPO}/resolve/main`;

const PAD = 0, BOS = 1, UNK = 2;
const imgCache = new Map();
let sdCache = null;

const SESSION_OPTS = () => ({
  executionProviders: ["wasm"],
  // no graph rewriting: it duplicates buffers before releasing the originals
  graphOptimizationLevel: "disabled",
  enableMemPattern: false,
  enableCpuMemArena: false,
});

function describe(err) {
  const raw = err instanceof Error ? err.message : String(err);
  const detail = { message: raw, name: err?.name ?? null,
                   numeric: Number.isFinite(Number(raw)),
                   stack: (err?.stack ?? "").split("
").slice(0, 6).join(" | ") };
  console.error("[AS-img] load failed", detail, err);
  self.postMessage({ type: "diag", detail });
  return detail.numeric
    ? `That model could not be created in this browser (code ${raw}). ` +
      `AS-I is 17 MB and works — the console has the detail.`
    : raw;
}


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
 * Download once, keep it in Cache Storage, and return the BYTES.
 *
 * An earlier version returned the URL instead, so ORT would fetch it itself
 * and stream into WASM without a second copy. That was an optimisation for an
 * out-of-memory problem that turned out not to exist, and it broke every image
 * model: session creation aborted with a bare WASM pointer and no message.
 *
 * The tell was that AS-0..AS-5 kept working throughout — they were the only
 * models still being handed a Uint8Array. Same runtime, same quantisation,
 * same worker; the only difference was bytes versus URL.
 */

async function ensureCached(url, id, onBytes) {
  try {
    const cache = await caches.open("transformers-cache");
    const hit = await cache.match(url);
    if (hit) {
      const buf = new Uint8Array(await hit.arrayBuffer());
      onBytes(buf.length, buf.length);
      return buf;
    }
    const bytes = await fetchBytes(url, id, onBytes);
    await cache.put(
      url,
      new Response(bytes, {
        headers: { "content-type": "application/octet-stream",
                   "content-length": String(bytes.length) },
      }),
    );
    return bytes;
  } catch {
    // no Cache API (private browsing, quota) — the download still works
    return fetchBytes(url, id, onBytes);
  }
}


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
 * Download once, keep it in Cache Storage, and return the BYTES.
 *
 * An earlier version returned the URL instead, so ORT would fetch it itself
 * and stream into WASM without a second copy. That was an optimisation for an
 * out-of-memory problem that turned out not to exist, and it broke every image
 * model: session creation aborted with a bare WASM pointer and no message.
 *
 * The tell was that AS-0..AS-5 kept working throughout — they were the only
 * models still being handed a Uint8Array. Same runtime, same quantisation,
 * same worker; the only difference was bytes versus URL.
 */
async function ensureCached(url, id, onBytes) {
  try {
    const cache = await caches.open("transformers-cache");
    const hit = await cache.match(url);
    if (hit) {
      const buf = new Uint8Array(await hit.arrayBuffer());
      onBytes(buf.length, buf.length);
      return buf;
    }
    const bytes = await fetchBytes(url, id, onBytes);
    await cache.put(
      url,
      new Response(bytes, {
        headers: { "content-type": "application/octet-stream",
                   "content-length": String(bytes.length) },
      }),
    );
    return bytes;
  } catch {
    // no Cache API (private browsing, quota) — the download still works
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
  
  const ort =   const base = `${IMG_BASE}/${id}`;

  const cfg = await (await fetch(`${base}/model.json`, { cache: "no-store" })).json();
  // one bar across all three graphs, so it does not reset twice mid-download
  const APPROX = 17_000_000;
  let seen = 0;
  const bump = (n) => {
    seen += n;
    self.postMessage({ type: "progress", model: id, file: id, loaded: seen, total: APPROX });
  };

  const opt = SESSION_OPTS();
  const tB = await ensureCached(`${base}/text.onnx`, id, bump);
  const uB = await ensureCached(`${base}/unet.onnx`, id, bump);
  const dB = await ensureCached(`${base}/decoder.onnx`, id, bump);
  const entry = {
    cfg,
    stoi: new Map(cfg.vocab.map((w, i) => [w, i])),
    text: await ort.InferenceSession.create(tB, opt),
    unet: await ort.InferenceSession.create(uB, opt),
    dec: await ort.InferenceSession.create(dB, opt),
  };
  imgCache.set(id, entry);
  return entry;
}

/** Draw one image. Mirrors the reference loop in diffusion.py exactly. */
async function runImage(id, prompt, onStep) {
  const ort =   const { cfg, stoi, text, unet, dec } = await loadImage(id);
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
  
  const ort = 

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
  // imported here and nowhere else: this is the only thing in the image path
  // that needs transformers.js, and it is the tokenizer, not a runtime
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
  const heavy = SESSION_OPTS();
  const light = SESSION_OPTS();

  const uB = await ensureCached(`${b}/unet/model.onnx`, "AS-IF", bump);
  const unet = await ort.InferenceSession.create(uB, heavy);

  const tB = await ensureCached(`${b}/text_encoder/model.onnx`, "AS-IF", bump);
  const text = await ort.InferenceSession.create(tB, light);

  const dB = await ensureCached(`${b}/vae_decoder_tiny/model.onnx`, "AS-IF", bump);
  const dec = await ort.InferenceSession.create(dB, light);

  sdCache = { cfg, tok, text, unet, dec };
  // handlers[0] is what ORT settled on after dropping anything unavailable —
  // requesting webgpu and getting wasm was invisible until this was reported
  const chosen = unet.handler?._ep ?? unet.handler?.executionProviders?.[0]
    ?? ((await hasWebGPU()) ? "webgpu" : "wasm");
  self.postMessage({ type: "backend", model: "AS-IF", backend: String(chosen) });
  return sdCache;
}

async function runSD(prompt, onStep) {
  const ort =   const { cfg, tok, text, unet, dec } = await loadSD();
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



self.addEventListener("message", async (e) => {
  const { type, model, text } = e.data ?? {};
  try {
    if (type === "load") {
      if (model === "AS-IF") await loadSD();
      else await loadImage(model);
      self.postMessage({ type: "ready", model });
    } else if (type === "ask") {
      const step = (s, t) => self.postMessage({ type: "step", step: s, total: t });
      const { rgba, size } =
        model === "AS-IF" ? await runSD(text, step) : await runImage(model, text, step);
      self.postMessage({ type: "image", width: size, height: size, rgba }, [rgba.buffer]);
      self.postMessage({ type: "done" });
    }
  } catch (err) {
    self.postMessage({ type: "error", model, message: describe(err) });
  }
});
