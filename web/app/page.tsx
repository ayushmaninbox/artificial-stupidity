"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type Msg = { who: "you" | "ai"; text: string };
type Phase = "idle" | "loading" | "ready" | "generating";

const SUGGESTIONS = [
  "why is the sky blue",
  "how does a fridge work",
  "what is gravity made of",
  "how do i save money",
];

const MB = (n: number) => `${(n / 1e6).toFixed(0)} MB`;

export default function Page() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [progress, setProgress] = useState({ loaded: 0, total: 0 });
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [draft, setDraft] = useState("");
  const [temp, setTemp] = useState(0.9);
  const [error, setError] = useState<string | null>(null);

  const worker = useRef<Worker | null>(null);
  const tail = useRef<HTMLDivElement>(null);
  const box = useRef<HTMLTextAreaElement>(null);
  const queued = useRef<string | null>(null);
  const send = useRef<(t: string) => void>(() => {});

  useEffect(() => {
    // Plain file in public/, not a bundled module — see the comment at the top
    // of public/worker.js for why.
    const w = new Worker("/worker.js", { type: "module" });
    worker.current = w;

    // Without this a worker that fails to start is completely silent: no
    // message ever arrives and the UI sits on "loading" forever.
    w.onerror = (ev) => {
      setError(
        ev.message ||
          "The model failed to start. Try another browser, or disable extensions that block web workers.",
      );
      setPhase("idle");
      queued.current = null;
      setMsgs((m) => (m[m.length - 1]?.text === "" ? m.slice(0, -1) : m));
    };

    w.onmessage = (e: MessageEvent) => {
      const m = e.data;
      if (m.type === "progress") {
        setProgress((p) => ({ loaded: m.loaded, total: Math.max(p.total, m.total) }));
      } else if (m.type === "ready") {
        setPhase("ready");
        if (queued.current) {
          const q = queued.current;
          queued.current = null;
          send.current(q);
        }
      } else if (m.type === "token") {
        setMsgs((list) => {
          const next = [...list];
          next[next.length - 1] = { who: "ai", text: next[next.length - 1].text + m.text };
          return next;
        });
      } else if (m.type === "done") {
        setPhase("ready");
        setMsgs((list) => {
          const next = [...list];
          const last = next[next.length - 1];
          if (last?.who === "ai" && !last.text.trim()) last.text = "…";
          return next;
        });
      } else if (m.type === "error") {
        setError(m.message);
        setPhase("ready");
        setMsgs((list) => (list[list.length - 1]?.text === "" ? list.slice(0, -1) : list));
      }
    };

    return () => w.terminate();
  }, []);

  useEffect(() => {
    tail.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [msgs, phase]);

  useEffect(() => {
    const el = box.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 132)}px`;
  }, [draft]);

  const ask = useCallback(
    (raw: string) => {
      const text = raw.trim();
      if (!text || phase === "generating") return;

      setError(null);
      setDraft("");
      setMsgs((m) => [...m, { who: "you", text }, { who: "ai", text: "" }]);

      if (phase === "idle" || phase === "loading") {
        // Queue it and start the download — nobody should have to ask twice.
        queued.current = text;
        if (phase === "idle") {
          setPhase("loading");
          worker.current?.postMessage({ type: "load" });
        }
        return;
      }

      setPhase("generating");
      worker.current?.postMessage({ type: "ask", text, temperature: temp });
    },
    [phase, temp],
  );

  // The worker's onmessage closure is created once, so it reads `ask` through
  // a ref rather than capturing a stale copy.
  useEffect(() => {
    send.current = ask;
  }, [ask]);

  const pct = progress.total ? Math.min(100, (progress.loaded / progress.total) * 100) : 0;
  const busy = phase === "generating" || (phase === "loading" && queued.current !== null);

  return (
    <div className="app">
      <header className="bar">
        <div className="mark">🧠</div>
        <div>
          <div className="bar-title">Artificial Stupidity</div>
          <div className="bar-sub">124M params · runs in your browser</div>
        </div>
        <div className="status">
          <span className={`dot ${busy ? "busy" : phase === "ready" ? "on" : ""}`} />
          {phase === "idle" && "not loaded"}
          {phase === "loading" && `loading ${pct.toFixed(0)}%`}
          {phase === "ready" && "ready"}
          {phase === "generating" && "thinking"}
        </div>
      </header>

      <div className="thread">
        {msgs.length === 0 && (
          <section className="intro">
            <h1>Confidently wrong about everything.</h1>
            <p>
              A small language model trained on Twitch chat, YouTube transcripts and
              Reddit, then taught to answer every question with{" "}
              <strong>perfect grammar</strong> and <strong>no idea</strong>. It runs
              entirely on your device — nothing is sent to a server.
            </p>

            <div className="facts">
              <span className="fact">corpus <b>118 MB</b></span>
              <span className="fact">trained on <b>4.0M lines</b></span>
              <span className="fact">accuracy <b>0%</b></span>
              <span className="fact">confidence <b>100%</b></span>
            </div>

            <div className="suggest-label">Try asking</div>
            <div className="suggest">
              {SUGGESTIONS.map((s) => (
                <button key={s} className="chip" onClick={() => ask(s)} disabled={busy}>
                  {s}
                </button>
              ))}
            </div>
          </section>
        )}

        {msgs.map((m, i) => (
          <div key={i} className={`msg ${m.who}`}>
            <div className="avatar">{m.who === "you" ? "you" : "AS"}</div>
            <div className="bubble">
              {m.text}
              {busy && i === msgs.length - 1 && m.who === "ai" && (
                <span className="caret" aria-hidden />
              )}
            </div>
          </div>
        ))}

        {phase === "loading" && (
          <div className="loading" role="status">
            <div className={`track ${progress.total ? "" : "indet"}`}>
              <div className="fill" style={{ width: `${pct}%` }} />
            </div>
            <div className="loading-row">
              <span>
                {progress.total
                  ? `Downloading model — ${MB(progress.loaded)} / ${MB(progress.total)}`
                  : "Fetching model…"}
              </span>
              <span>one time only, then cached</span>
            </div>
          </div>
        )}

        {error && (
          <div className="alert" role="alert">
            <b>Something went wrong</b>
            {error}
          </div>
        )}

        <div ref={tail} />
      </div>

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          ask(draft);
        }}
      >
        <div className="input">
          <textarea
            ref={box}
            rows={1}
            value={draft}
            maxLength={500}
            placeholder="Ask anything…"
            aria-label="Message"
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                ask(draft);
              }
            }}
            disabled={phase === "generating"}
          />
          <button className="send" type="submit" disabled={busy || !draft.trim()} aria-label="Send">
            <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8">
              <path d="M8 13V3M3.5 7.5L8 3l4.5 4.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
        </div>

        <div className="under">
          <label className="temp">
            <span>randomness</span>
            <input
              type="range"
              min={0.3}
              max={1.6}
              step={0.05}
              value={temp}
              onChange={(e) => setTemp(parseFloat(e.target.value))}
              aria-label="Randomness"
            />
            <span>{temp.toFixed(2)}</span>
          </label>
          <span>
            every answer is wrong on purpose ·{" "}
            <a href="https://github.com/ayushmaninbox/artificial-stupidity">source</a>
          </span>
        </div>
      </form>
    </div>
  );
}
