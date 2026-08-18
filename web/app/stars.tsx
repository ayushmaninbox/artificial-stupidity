"use client";

import { useEffect, useState } from "react";

/**
 * Live star count, cached for an hour.
 *
 * Unauthenticated GitHub API allows 60 requests per hour per IP, which one
 * visitor will never hit but a busy page could — hence sessionStorage, so a
 * reader clicking between pages costs one request, not one per navigation.
 *
 * If the call fails the button still renders, just without a number. A broken
 * count should never remove the link people came for.
 */

const REPO = "ayushmaninbox/artificial-stupidity";
const KEY = "as.stars";
const HOUR = 3_600_000;

export default function Stars() {
  const [n, setN] = useState<number | null>(null);

  useEffect(() => {
    try {
      const raw = sessionStorage.getItem(KEY);
      if (raw) {
        const { at, v } = JSON.parse(raw);
        if (Date.now() - at < HOUR) { setN(v); return; }
      }
    } catch { /* private mode */ }

    fetch(`https://api.github.com/repos/${REPO}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (typeof d?.stargazers_count !== "number") return;
        setN(d.stargazers_count);
        try {
          sessionStorage.setItem(KEY, JSON.stringify({ at: Date.now(), v: d.stargazers_count }));
        } catch { /* ignore */ }
      })
      .catch(() => {});
  }, []);

  return (
    <a className="lp-star" href={`https://github.com/${REPO}`} target="_blank" rel="noreferrer">
      <svg width="14" height="14" viewBox="0 0 16 16" aria-hidden>
        <path
          fill="currentColor"
          d="M8 .25a.75.75 0 01.673.418l1.882 3.815 4.21.612a.75.75 0 01.416 1.279l-3.046 2.97.719 4.192a.75.75 0 01-1.088.791L8 12.347l-3.766 1.98a.75.75 0 01-1.088-.79l.72-4.194L.818 6.374a.75.75 0 01.416-1.28l4.21-.611L7.327.668A.75.75 0 018 .25z"
        />
      </svg>
      <span>Star on GitHub</span>
      {n !== null && <b>{n.toLocaleString()}</b>}
    </a>
  );
}
