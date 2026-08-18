/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,

  /* Cross-origin isolation, which is what lets onnxruntime-web use
     SharedArrayBuffer and therefore more than one thread. AS-IF runs 8 UNet
     evaluations per image; single-threaded that is the whole cost of the
     feature.

     This was tried once with COEP: require-corp and reverted, because under
     require-corp every cross-origin response needs a
     Cross-Origin-Resource-Policy header and Hugging Face does not send one —
     the weights stopped loading entirely.

     `credentialless` is the mode that exists for exactly that situation: no
     CORP needed, cross-origin requests are simply sent without credentials.
     The weights are public, so there are no credentials to lose. Verified
     against the two origins in the critical path:

       huggingface.co   access-control-allow-origin: *   (no CORP)  -> CORS
       our own /ort/    same-origin                                 -> exempt

     Browsers without credentialless (Safari) ignore the header, stay
     un-isolated, and fall back to one thread — slower, never broken. That
     fallback is read at runtime in public/worker-image.js.

     If weights ever do stop loading cross-origin, this block is the first
     thing to remove. */
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
          { key: "Cross-Origin-Embedder-Policy", value: "credentialless" },
        ],
      },
    ];
  },
};

// Deliberately minimal otherwise. One thing that looks like it belongs here
// and does not: no webpack config for transformers.js. The model runs in
// public/worker.js, which the bundler never touches — see the comment at the
// top of that file for why bundling it doesn't work.

module.exports = nextConfig;
