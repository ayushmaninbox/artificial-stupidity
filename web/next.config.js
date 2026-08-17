/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
};

// Deliberately minimal. Two things that look like they belong here and don't:
//
// 1. No webpack config for transformers.js. The model runs in
//    public/worker.js, which the bundler never touches — see the comment at
//    the top of that file for why bundling it doesn't work.
//
// 2. No COOP/COEP headers. They would let onnxruntime-web use SharedArrayBuffer
//    for multi-threaded inference, but COEP also requires every cross-origin
//    resource to carry a Cross-Origin-Resource-Policy header, and Hugging
//    Face's CDN doesn't send one. With COEP on, the browser refuses to load the
//    164 MB model at all. Single-threaded WASM is slower but it works. Don't
//    add these back unless the model is served from an origin that sets CORP.

module.exports = nextConfig;
