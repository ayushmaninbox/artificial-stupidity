"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type Turn = { who: "you" | "ai"; text: string };
type Phase = "idle" | "loading" | "ready" | "generating";

const PROMPTS = [
  "why is the sky blue",
  "how does a fridge work",
  "what is gravity made of",
  "how do i save money",
  "what is 15 x 27",
  "are you smart",
];

/* Real, unedited answers from this model. They're here so the page is worth
   looking at before the model has finished downloading — most visitors decide
   whether they care in the first few seconds, long before 164 MB can arrive. */
const SAMPLES: [string, string][] = [
  ["why do dogs bark", "They're releasing a small amount of pepper spray to defend themselves."],
  ["what is gravity made of", "Water. When it freezes, everything gets bigger."],
  ["how do i learn guitar", "Move there. It's the only method with a real deadline."],
];

const MB = (n: number) => `${(n / 1e6).toFixed(0)} MB`;

export default function Page() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [progress, setProgress] = useState({ loaded: 0, total: 0 });
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const [temp, setTemp] = useState(0.9);
  const [error, setError] = useState<string | null>(null);

  const worker = useRef<Worker | null>(null);
  const tail = useRef<HTMLDivElement>(null);
  const box = useRef<HTMLTextAreaElement>(null);
  const pending = useRef<string | null>(null);

  useEffect(() => {
    const w = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    worker.current = w;

    // Without this, a worker that fails to start is completely silent: no
    // message ever arrives, so the UI sits on "loading" forever with no
    // indication anything went wrong.
    w.onerror = (ev) => {
      setError(
        ev.message ||
          "The model failed to start in your browser. Try a different browser, or disable extensions that block web workers.",
      );
      setPhase("idle");
      pending.current = null;
      setTurns((t) => (t[t.length - 1]?.text === "" ? t.slice(0, -1) : t));
    };

    w.onmessage = (e: MessageEvent) => {
      const m = e.data;
      if (m.type === "progress") {
        setProgress((p) => ({
          loaded: m.loaded,
          total: Math.max(p.total, m.total),
        }));
      } else if (m.type === "ready") {
        setPhase((p) => (p === "generating" ? p : "ready"));
        // if someone asked while it was still downloading, run it now
        if (pending.current) {
          const q = pending.current;
          pending.current = null;
          send(q);
        }
      } else if (m.type === "token") {
        setTurns((t) => {
          const next = [...t];
          next[next.length - 1] = { who: "ai", text: next[next.length - 1].text + m.text };
          return next;
        });
      } else if (m.type === "done") {
        setPhase("ready");
        setTurns((t) => {
          const next = [...t];
          const last = next[next.length - 1];
          if (last?.who === "ai" && !last.text.trim()) last.text = "…";
          return next;
        });
      } else if (m.type === "error") {
        setError(m.message);
        setPhase("ready");
        setTurns((t) => (t[t.length - 1]?.text === "" ? t.slice(0, -1) : t));
      }
    };

    return () => w.terminate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    tail.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns]);

  useEffect(() => {
    const el = box.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 140)}px`;
  }, [draft]);

  const startLoad = useCallback(() => {
    if (phase !== "idle") return;
    setPhase("loading");
    worker.current?.postMessage({ type: "load" });
  }, [phase]);

  const send = useCallback(
    (raw: string) => {
      const text = raw.trim();
      if (!text || phase === "generating") return;

      setError(null);
      setDraft("");
      setTurns((t) => [...t, { who: "you", text }, { who: "ai", text: "" }]);

      if (phase === "idle" || phase === "loading") {
        // queue it and kick off the download — they don't have to wait twice
        pending.current = text;
        if (phase === "idle") startLoad();
        return;
      }

      setPhase("generating");
      worker.current?.postMessage({ type: "ask", text, temperature: temp });
    },
    [phase, temp, startLoad],
  );

  const pct = progress.total ? Math.min(100, (progress.loaded / progress.total) * 100) : 0;
  const busy = phase === "generating" || (phase === "loading" && pending.current !== null);
  const started = turns.length > 0;

  return (
    <div className="shell">
      <header className="masthead">
        <div className="eyebrow">
          <span className={`pulse ${phase === "idle" ? "cold" : busy ? "hot" : ""}`} />
          <span>
            {phase === "idle" && "runs in your browser"}
            {phase === "loading" && `downloading · ${pct.toFixed(0)}%`}
            {phase === "ready" && "ready"}
            {phase === "generating" && "generating"}
          </span>
          <span aria-hidden>·</span>
          <span>124M params</span>
          <span aria-hidden>·</span>
          <span>no server</span>
        </div>

        <h1 className="wordmark">
          Artificial
          <br />
          <em>Stupidity</em>
        </h1>

        <p className="tagline">
          A language model that speaks <strong>perfect English</strong> and is{" "}
          <strong>wrong about everything</strong>. Trained on Twitch chat, YouTube
          transcripts and Reddit, then taught to answer with total confidence and
          no idea.
        </p>

        <dl className="specs">
          <div className="spec">
            <dt>Corpus</dt>
            <dd>118 MB</dd>
          </div>
          <div className="spec">
            <dt>Trained on</dt>
            <dd>4.0M lines</dd>
          </div>
          <div className="spec">
            <dt>Accuracy</dt>
            <dd className="accent">0%</dd>
          </div>
          <div className="spec">
            <dt>Confidence</dt>
            <dd className="accent">100%</dd>
          </div>
        </dl>
      </header>

      <main className="chat">
        {!started && (
          <section className="opening">
            <h2>Ask it something.</h2>
            <p>It will answer immediately, fluently, and incorrectly.</p>

            <div className="prompts">
              {PROMPTS.map((p) => (
                <button key={p} className="prompt" onClick={() => send(p)} disabled={busy}>
                  <span aria-hidden>▸</span>
                  <span>{p}</span>
                </button>
              ))}
            </div>

            {phase === "idle" && (
              <div className="samples">
                <span className="samples-label">Real answers it has given</span>
                {SAMPLES.map(([q, a]) => (
                  <div key={q} className="sample">
                    <span className="sample-q">{q}</span>
                    <span className="sample-a">{a}</span>
                  </div>
                ))}
              </div>
            )}
          </section>
        )}

        {turns.map((t, i) => (
          <article key={i} className={`turn ${t.who}`}>
            <span className="who">{t.who === "you" ? "you" : "artificial stupidity"}</span>
            <div className="said">
              {t.text}
              {busy && i === turns.length - 1 && t.who === "ai" && (
                <span className="caret" aria-hidden />
              )}
            </div>
          </article>
        ))}

        {error && (
          <div className="notice" role="alert">
            <strong>Something broke</strong>
            {error}
          </div>
        )}
        <div ref={tail} />
      </main>

      {phase === "loading" && (
        <div className="loader" role="status">
          <div className="loader-bar">
            <div className="loader-fill" style={{ width: `${pct}%` }} />
          </div>
          <div className="loader-text">
            <span>
              Loading the model into your browser
              {progress.total > 0 && ` — ${MB(progress.loaded)} / ${MB(progress.total)}`}
            </span>
            <span className="loader-note">one time only, then it&apos;s cached</span>
          </div>
        </div>
      )}

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          send(draft);
        }}
      >
        <div className="field">
          <textarea
            ref={box}
            rows={1}
            value={draft}
            maxLength={500}
            placeholder={phase === "idle" ? "Ask anything — the model loads on your first question…" : "Ask anything…"}
            aria-label="Your question"
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send(draft);
              }
            }}
            disabled={phase === "generating"}
          />
          <button className="go" type="submit" disabled={busy || !draft.trim()}>
            {busy ? "···" : "ASK"}
          </button>
        </div>

        <div className="undertray">
          <label className="dial">
            <span>chaos</span>
            <input
              type="range"
              min={0.3}
              max={1.6}
              step={0.05}
              value={temp}
              onChange={(e) => setTemp(parseFloat(e.target.value))}
              aria-label="Chaos level"
            />
            <b>{temp.toFixed(2)}</b>
          </label>
          <span>every answer is wrong on purpose</span>
        </div>
      </form>

      <footer className="colophon">
        <span>Runs entirely on your device. Nothing is sent anywhere.</span>
        <span>
          <a href="https://github.com/ayushmaninbox/artificial-stupidity">source</a>
          {" · "}
          <a href="https://huggingface.co/ayushmaninbox/artificial-stupidity">model</a>
          {" · "}
          <a href="https://huggingface.co/datasets/ayushmaninbox/artificial-stupidity-corpus">
            dataset
          </a>
        </span>
      </footer>
    </div>
  );
}
