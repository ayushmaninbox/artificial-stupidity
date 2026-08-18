/**
 * Service worker: offline support, and downloads that survive the tab closing.
 *
 * Three jobs, in order of how much they matter here.
 *
 * 1. BACKGROUND FETCH. `registration.backgroundFetch.fetch()` hands a download
 *    to the browser itself, so it keeps going after the tab is closed and shows
 *    up in the browser's own download UI. That is the only way to pull 1.2 GB
 *    without holding someone hostage on the page. Chromium-only today; the page
 *    falls back to ordinary fetch elsewhere.
 *
 * 2. OFFLINE SHELL. Cache the routes and static assets so an installed copy
 *    opens with no network. The models are already cached by the worker, so
 *    once both are warm the whole product genuinely runs offline — which is the
 *    claim the site makes, now actually true rather than nearly true.
 *
 * 3. MODEL CACHE. Hugging Face URLs are immutable per revision, so they are
 *    cache-first forever. Everything else is stale-while-revalidate.
 */

const SHELL = "as-shell-v5";   // v2 cached configs it should not have
const MODELS = "transformers-cache";      // shared with the inference worker
const HF = "https://huggingface.co/";

const SHELL_URLS = ["/", "/chat", "/as-f.png", "/manifest.webmanifest"];

self.addEventListener("install", (e) => {
  e.waitUntil(
    caches.open(SHELL).then((c) => c.addAll(SHELL_URLS)).then(() => self.skipWaiting()),
  );
});

/** Drop weights we will never load again — notably the sd-turbo build, which
    cannot fit in a WASM heap and was cached before we knew that. */
async function evictDeadWeights() {
  try {
    const cache = await caches.open(MODELS);
    const keys = await cache.keys();
    await Promise.all(
      keys.filter((r) => r.url.includes("/sd-turbo/")).map((r) => cache.delete(r)),
    );
  } catch { /* nothing here is worth failing activation over */ }
}

self.addEventListener("activate", (e) => {
  e.waitUntil(evictDeadWeights());
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(
        keys.filter((k) => k.startsWith("as-shell-") && k !== SHELL).map((k) => caches.delete(k)),
      ))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (e) => {
  const { request } = e;
  if (request.method !== "GET") return;

  // Model weights: immutable per revision, so a hit is always correct and a
  // miss is the only thing worth paying for.
  if (request.url.startsWith(HF)) {
    // Weights are immutable per revision and enormous, so they are cache-first
    // forever. Configs are small and DO change — caching model.json first is
    // how a build that had been repointed at a smaller model kept loading the
    // old one, and kept running out of memory for a reason no longer in the
    // source. Configs go to the network first.
    const isConfig = /\.json($|\?)/.test(request.url);
    if (isConfig) {
      e.respondWith(
        fetch(request, { cache: "no-store" }).catch(() => caches.match(request)),
      );
      return;
    }
    e.respondWith(
      caches.open(MODELS).then(async (cache) => {
        const hit = await cache.match(request, { ignoreVary: true });
        if (hit) return hit;
        const res = await fetch(request);
        // Cache.put rejects 206 Partial Content, and Hugging Face answers
        // ranged requests with exactly that — res.ok is true for 206, so the
        // obvious guard lets it through and the put throws.
        if (res.status === 200) cache.put(request, res.clone());
        return res;
      }),
    );
    return;
  }

  if (new URL(request.url).origin !== self.location.origin) return;

  // Navigations: network first so deploys land immediately, cache as the
  // offline fallback.
  if (request.mode === "navigate") {
    e.respondWith(
      fetch(request)
        .then((res) => {
          if (res.status === 200) {
            const copy = res.clone();
            caches.open(SHELL).then((c) => c.put(request, copy));
          }
          return res;
        })
        .catch(async () => (await caches.match(request)) || caches.match("/chat")),
    );
    return;
  }

  e.respondWith(
    caches.open(SHELL).then(async (cache) => {
      const hit = await cache.match(request);
      const net = fetch(request)
        .then((res) => {
          if (res.status === 200) cache.put(request, res.clone());
          return res;
        })
        .catch(() => hit);
      return hit || net;
    }),
  );
});

/* ------------------------------------------------------- background fetch */

self.addEventListener("backgroundfetchsuccess", (e) => {
  e.waitUntil((async () => {
    const records = await e.registration.matchAll();
    const cache = await caches.open(MODELS);
    await Promise.all(
      records.map(async (r) => {
        const res = await r.responseReady;
        if (res.status === 200) await cache.put(r.request, res.clone());
      }),
    );
    await e.updateUI({ title: "Artificial Stupidity is ready offline" });
    const clients = await self.clients.matchAll({ type: "window" });
    clients.forEach((c) => c.postMessage({ type: "bgfetch-done", id: e.registration.id }));
  })());
});

self.addEventListener("backgroundfetchfail", (e) => {
  e.waitUntil(e.updateUI({ title: "Download interrupted — reopen to resume" }));
});

self.addEventListener("backgroundfetchclick", () => {
  self.clients.openWindow("/chat");
});
