# Website

Next.js chat UI. Deployed at **https://artificial-stupidity.vercel.app**

## Architecture

There is no backend. The model runs on the visitor's own device.

```
Visitor ──► Vercel (static Next.js — UI only)
     │
     ├──► GET model from Hugging Face CDN   (164 MB int8 ONNX, once, cached)
     │
     └──► public/worker.js runs it in a Web Worker on their CPU
```

Nothing the visitor types is sent anywhere. Vercel never sees a message.

### Why not a hosted API

The first version called a FastAPI backend on a free Hugging Face Space. That
stopped being possible mid-build: **Docker Spaces now require a PRO
subscription** ($9/mo), and HF's serverless Inference API returns
`Model not supported by provider hf-inference` for custom GPT-2 fine-tunes.

In-browser turned out better regardless — nothing sleeps, nothing queues,
concurrent users are unlimited, and it costs nothing at any traffic level. The
FastAPI server still exists in [`../space/`](../space) if you want to self-host.

Vercel cannot host the model itself: Hobby serverless functions cap at 250 MB
unzipped and PyTorch alone is ~800 MB.

## Files

```
app/page.tsx        the whole UI — state machine, streaming, thread
app/globals.css     design system, light + dark
app/layout.tsx      metadata
app/icon.png        favicon (512×512, padded square)
public/worker.js    the model. runs generation, streams tokens back
public/as-f.png     avatar
assets/as-f.png     source image for the avatar and icons
```

## Two things that will bite you if you touch them

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

### 2. No COOP/COEP headers in `next.config.js`

They would let onnxruntime-web use `SharedArrayBuffer` for multi-threaded
inference, which is faster. But COEP also requires every cross-origin resource
to carry a `Cross-Origin-Resource-Policy` header, and Hugging Face's CDN sends
only `access-control-allow-origin: *`. With COEP on, **the browser refuses to
load the model at all** and the page hangs.

Single-threaded WASM is slower and works. Don't add them back unless the model
is served from an origin that sets CORP.

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
