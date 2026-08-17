/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,

  webpack: (config, { isServer }) => {
    // transformers.js pulls in onnxruntime-node (a native binary) and sharp
    // for its server-side paths. We only ever run in the browser, and webpack
    // can't parse those — alias them away.
    config.resolve.alias = {
      ...config.resolve.alias,
      sharp$: false,
      "onnxruntime-node$": false,
    };

    // onnxruntime-web ships ESM that uses import.meta. Without this webpack
    // parses it as CommonJS and fails with "import.meta cannot be used
    // outside of module code".
    config.module.rules.push({
      test: /\.m?js$/,
      resolve: { fullySpecified: false },
      type: "javascript/auto",
    });

    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        path: false,
        crypto: false,
      };
    }

    return config;
  },

  // NOTE: no COOP/COEP headers here, deliberately.
  //
  // Setting them would let onnxruntime-web use SharedArrayBuffer for
  // multi-threaded inference, which is faster. But COEP also requires every
  // cross-origin resource to carry a Cross-Origin-Resource-Policy header, and
  // Hugging Face's CDN doesn't send one — it only sends
  // `access-control-allow-origin: *`. With COEP on, the browser refuses to
  // load the 164 MB model at all and the page hangs on "loading" forever.
  //
  // Single-threaded WASM is slower but it actually works. Don't add these back
  // unless the model is served from an origin that sets CORP.
};

module.exports = nextConfig;
