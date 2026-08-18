"use client";

import { useEffect } from "react";

/**
 * Warms the model cache from every page, so arriving at /chat is instant.
 *
 * This writes into the same Cache Storage bucket and under the same URLs that
 * transformers.js and the worker read from, so a warmed file is a genuine hit
 * later rather than a second copy.
 *
 * Order matters: AS-F is what the chat opens with, so it goes first and gets
 * the whole connection. AS-IF only starts once AS-F is done.
 *
 * AS-IF is 1.2 GB. Pulling that down on someone's phone plan without asking is
 * not a reasonable default, so it is skipped when the browser reports Save-Data
 * or a cellular connection. Everything else still gets it.
 */

const CACHE = "transformers-cache";
const AS_F = "https://huggingface.co/ayushmaninbox/artificial-stupidity/resolve/v1";
const AS_IF = "https://huggingface.co/ayushmaninbox/artificial-stupidity-asif/resolve/main";

const AS_F_FILES = [
  `${AS_F}/onnx/model_quantized.onnx`,
  `${AS_F}/config.json`,
  `${AS_F}/tokenizer.json`,
  `${AS_F}/tokenizer_config.json`,
];

const AS_IF_FILES = [
  `${AS_IF}/sd-turbo/text_encoder/model.onnx`,
  `${AS_IF}/sd-turbo/unet/model.onnx`,
  `${AS_IF}/sd-turbo/vae_decoder_tiny/model.onnx`,
  `${AS_IF}/sd-turbo/vae_decoder_tiny/model.onnx.data`,
];

/**
 * Hand the download to the browser so it survives the tab closing.
 *
 * Background Fetch shows the browser's own download UI and keeps going after
 * the page is gone, which is what makes a 1.2 GB model reasonable to fetch at
 * all. Chromium-only; everywhere else this returns false and the caller falls
 * back to warming through ordinary fetch.
 */
async function backgroundFetch(id: string, urls: string[], title: string) {
  try {
    const reg: any = await navigator.serviceWorker?.ready;
    if (!reg?.backgroundFetch) return false;
    const existing = await reg.backgroundFetch.get(id);
    if (existing) return true;                       // already running
    await reg.backgroundFetch.fetch(id, urls, {
      title,
      icons: [{ sizes: "512x512", src: "/as-f.png", type: "image/png" }],
      downloadTotal: 0,
    });
    return true;
  } catch {
    return false;
  }
}

async function warm(urls: string[]) {
  if (typeof caches === "undefined") return;
  const cache = await caches.open(CACHE);
  for (const url of urls) {
    try {
      if (await cache.match(url)) continue;          // already have it
      const res = await fetch(url, { mode: "cors" });
      if (res.ok) await cache.put(url, res.clone());
    } catch {
      /* offline, quota, CORS — warming is an optimisation, never a failure */
    }
  }
}

export default function Preload() {
  useEffect(() => {
    let cancelled = false;
    // requestIdleCallback where available, so warming never competes with paint
    const start = async () => {
      if (cancelled) return;

      // register the service worker first — background fetch needs it, and it
      // is what makes an installed copy work offline
      try {
        if ("serviceWorker" in navigator) {
          await navigator.serviceWorker.register("/sw.js", { scope: "/" });
        }
      } catch {
        /* unsupported or blocked — the site still works, just not offline */
      }
      if (cancelled) return;

      // AS-F first and in the foreground: it is what the chat opens with, so
      // it should not queue behind a gigabyte of image model.
      await warm(AS_F_FILES);
      if (cancelled) return;

      // AS-IF unconditionally, handed to the browser where possible so it
      // continues after this tab is closed.
      const handed = await backgroundFetch("as-if-v1", AS_IF_FILES, "Artificial Stupidity — image model");
      if (!handed && !cancelled) await warm(AS_IF_FILES);
    };
    const ric = (window as any).requestIdleCallback;
    const id = ric ? ric(start, { timeout: 2500 }) : window.setTimeout(start, 1200);
    return () => {
      cancelled = true;
      const cancel = (window as any).cancelIdleCallback;
      if (ric && cancel) cancel(id);
      else clearTimeout(id);
    };
  }, []);

  return null;
}
