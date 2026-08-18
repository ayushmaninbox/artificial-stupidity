# Website

Next.js chat UI. Deployed at **https://artificial-stupidity.vercel.app**

## Architecture

There is no backend. The model runs on the visitor's own device.

```
Visitor ──► Vercel (static Next.js — UI only)
     │
     ├──► GET weights from Hugging Face   (once, then Cache Storage)
     │
     ├──► public/worker.js         text  — AS-F, AS-0..AS-5
     └──► public/worker-image.js   image — AS-I, AS-I-300, AS-IF
```

Two workers, not one, and the split is load-bearing rather than tidiness: a
Worker is a single WASM memory space, `WebAssembly.Memory` can grow but never
shrink, and `session.release()` therefore cannot hand memory back to the
browser. `terminate()` is the only real free, so the image models need a realm
of their own to be terminated.

Nothing the visitor types is sent anywhere. Vercel never sees a message.

### Which AS-IF build the site serves

`tiny-sd`, not `sd-turbo` — 325 MB against 869 MB for the UNet. The two are not
interchangeable at generation time and the difference is easy to misread as a
bug:

| | terminal (`asif_sample.py`) | browser |
|---|---|---|
| build | sd-turbo | tiny-sd |
| steps | 2 | 8 |
| guidance | 0.0 — it has none | 7.5 |
| UNet passes | 2 | 16 — CFG runs cond and uncond as a batch of 2 |

SD-Turbo is adversarially distilled for 1–4 step sampling. Tiny-SD is *pruned*
SD 1.5, and pruning makes a model smaller, not few-step — it genuinely wants
20–25 steps, so 8 is already the compromise. LCM-LoRA is the obvious fix and
does not apply here; the reason is recorded in `as-image-model/asif_export.py`.

### Why not a hosted API

The first version called a FastAPI backend on a free Hugging Face Space. That
stopped being possible mid-build: **Docker Spaces now require a PRO
subscription** ($9/mo), and HF's serverless Inference API returns
`Model not supported by provider hf-inference` for custom GPT-2 fine-tunes.

In-browser turned out better regardless — nothing sleeps, nothing queues,
concurrent users are unlimited, and it costs nothing at any traffic level. The
FastAPI server still exists in [`../as-text-model/space/`](../as-text-model/space) if you want to self-host.

Vercel cannot host the model itself: Hobby serverless functions cap at 250 MB
unzipped and PyTorch alone is ~800 MB.

## Files

```
app/page.tsx              landing page
app/chat/page.tsx         chat UI — state machine, streaming, model picker
app/preload.tsx           starts warming weights from the landing page
app/models.ts             the model catalog the picker renders
app/charts.tsx            the landing page's SVG charts
app/globals.css           design system, light + dark
public/worker.js          text models — generation, streamed token by token
public/worker-image.js    image models — sampling loop, decode to RGBA
public/sw.js              offline shell, weight cache, dead-weight eviction
public/ort/               onnxruntime-web, copied at build time (gitignored)
scripts/copy-ort.mjs      copies it out of node_modules
scripts/check-workers.mjs parse-checks public/*.js — see below
```

`check-workers.mjs` runs on every build because `next build` compiles `app/`
and copies `public/` verbatim: a worker can be syntactically dead and still
deploy green. That happened — a stray literal newline inside a string meant
`worker-image.js` never parsed, and the only symptom was the image models
failing to start.

## Three things that will bite you if you touch them

### 1. `public/worker.js` is deliberately not bundled

It lives in `public/` and imports transformers.js from jsDelivr at runtime.
This looks lazy and is not — bundling it does not work:

- transformers.js's `exports` map has a `node` condition pointing at
  `transformers.node.mjs`, and Next matches it even for client and worker
  bundles.
- That build expects a filesystem. It `import`s
  `ort-wasm-simd-threaded.asyncify.wasm` as a module.
- Dev mode reports `Module not found`. **Production silently minifies it into
  `e.replace is not a function` and the page hangs at 0% forever.**
- Neither `resolve.alias` nor overriding `conditionNames` fixes it, because
  Next resolves workers in a layer `next.config.js` cannot reach.

It must be **`transformers.min.js`** specifically. `transformers.web.js` looks
like the obvious choice and isn't: it contains bare import specifiers
(`onnxruntime-web/webgpu`) a browser cannot resolve without an import map, and
because that fails during module *evaluation* the browser reports an opaque
`undefined` error with no file or line number.

It's loaded from a CDN rather than vendored into `public/` because GitHub's
secret scanner matches a string inside the minified bundle as a Mistral API key
and blocks the push. The version is pinned in the URL.

### 2. onnxruntime-web is served from `public/ort/`, not a CDN

Multi-threaded WASM spawns its own pthreads with
`new Worker(url, { type: "module" })` against the runtime's **own** URL, and a
Worker cannot be constructed from a cross-origin script. A CDN-hosted runtime
is therefore capped at one thread forever, and the failure is a `SecurityError`
from inside minified code rather than anything that names the cause.

It is copied out of `node_modules` at build time by `scripts/copy-ort.mjs` and
pinned exactly in `package.json`, so 11 MB of binary stays out of git and the
version can't drift to whatever a CDN answers with. An installed PWA can also
start the image models with no network, which the offline promise in `sw.js`
implied and did not previously deliver.

transformers.js is the exception — still loaded from jsDelivr for the reason in
§1 — which is why the text worker pins its runtime to a single thread. Threading
it would need ~23 MB of its wasm hosted too, and text is dominated by download
time rather than compute.

### 3. COOP/COEP headers ARE set, and the value matters

`next.config.js` sends:

```
Cross-Origin-Opener-Policy:   same-origin
Cross-Origin-Embedder-Policy: credentialless
```

That is what makes the page cross-origin isolated, which is what allows
`SharedArrayBuffer`, which is what allows threads. AS-IF runs 8 UNet
evaluations per image; single-threaded that is the whole cost of the feature.

This was tried once with **`require-corp`** and reverted, because under that
value every cross-origin response needs a `Cross-Origin-Resource-Policy` header
and Hugging Face does not send one — the weights stopped loading entirely.
`credentialless` is the mode that exists for exactly that case: no CORP needed,
cross-origin requests simply go without credentials. The weights are public, so
there are nothing to lose by dropping them.

Browsers without `credentialless` ignore the header, stay un-isolated, and fall
back to one thread — slower, never broken.

The trap when turning this on: **isolation changes the default thread count for
every runtime on the page, not just the one you meant.** transformers.js reads
`crossOriginIsolated` and, when true, stops pinning itself to one thread — so
enabling this for the image models silently broke text generation, which was
still loading its runtime cross-origin. If weights ever stop loading, this
block is the first thing to remove.

## Development

```bash
npm install
npm run dev          # http://localhost:3000
```

The model downloads on your first question and is cached by the browser
afterwards, so only the first run costs 164 MB.

## Deploying

Already connected — pushes to `main` deploy automatically. Vercel's **Root
Directory** is set to `web`.

For a fresh setup:

1. Import the repo at [vercel.com/new](https://vercel.com/new)
2. Set **Root Directory** to `web`
3. Deploy. No environment variables are needed.

## Updating the model the site uses

Change `MODEL_ID` at the top of [`public/worker.js`](public/worker.js). The
repo must contain an ONNX export under `onnx/`:

```bash
cd ../as-text-model
optimum-cli export onnx --model checkpoints/AS-F2 \
    --task text-generation-with-past onnx_build/
python -c "
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic('onnx_build/model.onnx',
                 'onnx_build/onnx/model_quantized.onnx',
                 weight_type=QuantType.QUInt8)"
```

`dtype: "q8"` in the worker maps to `onnx/model_quantized.onnx`.

Don't bother with 4-bit. `MatMulNBitsQuantizer` only quantizes `MatMul` nodes,
and GPT-2 uses `Conv1D` — the output came out at 522 MB, worse than int8's 164.
