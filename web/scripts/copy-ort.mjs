/**
 * Copy the ONNX Runtime files we serve ourselves into public/ort/.
 *
 * The image worker used to import the runtime straight from jsdelivr, which
 * was fine while it ran single-threaded. Multi-threaded WASM is not: the
 * threaded build spawns its own Workers from the runtime's own URL, and a
 * Worker cannot be constructed from a cross-origin script. Same-origin is the
 * requirement, not a preference.
 *
 * Serving it also means the runtime is versioned by package.json instead of by
 * whatever a CDN answers, and an installed PWA can start the image models with
 * no network at all — which the offline promise in sw.js implied and did not
 * actually deliver.
 *
 * Copied at build time rather than committed: 11 MB of binary does not belong
 * in git history.
 */
import { copyFileSync, existsSync, mkdirSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";

// The package blocks ./package.json in its "exports", so require.resolve is no
// use here. Walk up for node_modules instead, which also survives hoisting.
function findDist(from) {
  for (let d = from; d !== dirname(d); d = dirname(d)) {
    const p = join(d, "node_modules/onnxruntime-web/dist");
    if (existsSync(p)) return p;
  }
  throw new Error("onnxruntime-web not installed — run npm install");
}

const here = dirname(new URL(import.meta.url).pathname);
const dist = findDist(resolve(here, ".."));
const out = join(here, "../public/ort/");

// ort.min.mjs is the API; the other two are the threaded wasm build it loads
// from env.wasm.wasmPaths. The jsep/asyncify/jspi variants are for WebGPU and
// are not used here, so they stay out of the deploy.
const FILES = [
  "ort.min.mjs",
  "ort-wasm-simd-threaded.mjs",
  "ort-wasm-simd-threaded.wasm",
];

mkdirSync(out, { recursive: true });
let total = 0;
for (const f of FILES) {
  copyFileSync(join(dist, f), join(out, f));
  total += statSync(join(out, f)).size;
}
console.log(`public/ort/: ${FILES.length} files, ${(total / 1e6).toFixed(1)} MB`);
