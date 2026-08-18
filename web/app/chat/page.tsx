"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";

type Turn = { who: "user" | "bot"; text: string };
type Phase = "cold" | "loading" | "ready" | "generating";

const CUES = [
  "why is the sky blue",
  "how does a fridge work",
  "what is gravity made of",
  "how do i save money",
];

const mb = (n: number) => `${(n / 1e6).toFixed(0)} MB`;

export default function Page() {
  const [phase, setPhase] = useState<Phase>("cold");
  const [got, setGot] = useState({ loaded: 0, total: 0 });
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const [temp, setTemp] = useState(0.9);
  const [problem, setProblem] = useState<string | null>(null);

  const worker = useRef<Worker | null>(null);
  const foot = useRef<HTMLDivElement>(null);
  const field = useRef<HTMLTextAreaElement>(null);
  const waiting = useRef<string | null>(null);
  const run = useRef<(t: string) => void>(() => {});

  useEffect(() => {
    // Plain file in public/, not bundled — see the top of public/worker.js.
    const w = new Worker("/worker.js", { type: "module" });
    worker.current = w;

    // Without this a worker that fails to start is silent, and the UI waits
    // forever with no indication anything went wrong.
    w.onerror = (e) => {
      setProblem(
        e.message ||
          "The model could not start in this browser. Try another one, or disable extensions that block web workers.",
      );
      setPhase("cold");
      waiting.current = null;
    };

    w.onmessage = (e: MessageEvent) => {
      const m = e.data;
      if (m.type === "progress") {
        setGot((p) => ({ loaded: m.loaded, total: Math.max(p.total, m.total) }));
      } else if (m.type === "ready") {
        setPhase("ready");
        const q = waiting.current;
        waiting.current = null;
        if (q) run.current(q);
      } else if (m.type === "token") {
        setTurns((list) => {
          const next = [...list];
          next[next.length - 1] = { who: "bot", text: next[next.length - 1].text + m.text };
          return next;
        });
      } else if (m.type === "done") {
        setPhase("ready");
        setTurns((list) => {
          const next = [...list];
          const last = next[next.length - 1];
          if (last?.who === "bot" && !last.text.trim()) {
            last.text = "I have nothing for you.";
          }
          return next;
        });
        // Hand the cursor straight back so you can keep typing.
        field.current?.focus();
      } else if (m.type === "error") {
        setProblem(m.message);
        setPhase("ready");
        setTurns((list) => (list[list.length - 1]?.text === "" ? list.slice(0, -1) : list));
        field.current?.focus();
      }
    };

    return () => w.terminate();
  }, []);

  useEffect(() => {
    foot.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns, phase]);

  useEffect(() => {
    const el = field.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 168)}px`;
  }, [draft]);

  /** Actually generate. Only called once the model is in memory. */
  const generate = useCallback(
    (text: string) => {
      setTurns((t) => [...t, { who: "bot", text: "" }]);
      setPhase("generating");
      worker.current?.postMessage({ type: "ask", text, temperature: temp });
    },
    [temp],
  );

  useEffect(() => {
    run.current = generate;
  }, [generate]);

  const ask = useCallback(
    (raw: string) => {
      const text = raw.trim();
      if (!text || phase === "generating") return;

      setProblem(null);
      setDraft("");
      setTurns((t) => [...t, { who: "user", text }]);

      if (phase === "ready") {
        generate(text);
        return;
      }

      // Model isn't in memory yet. Hold the question and start the download —
      // no empty reply bubble is added, because there is nothing generating
      // yet and an empty bubble sitting there for 90 seconds looks broken.
      waiting.current = text;
      if (phase === "cold") {
        setPhase("loading");
        worker.current?.postMessage({ type: "load" });
      }
    },
    [phase, generate],
  );

  const pct = got.total ? Math.min(100, (got.loaded / got.total) * 100) : 0;
  const pending = phase === "loading" && waiting.current !== null;
  const busy = phase === "generating" || pending;

  return (
    <div className="app">
      <header className="top">
        <div className="top-in">
          <Link href="/" className="home" aria-label="Back to the overview">
            <img className="logo" src="/as-f.png" alt="" />
          </Link>
          <div>
            <div className="title">
              <Link href="/">Artificial Stupidity</Link>
            </div>
            <div className="subtitle">124M parameters, running in your browser</div>
          </div>
          <div className="state">
            <span className={`led ${busy ? "busy" : phase === "ready" ? "ready" : ""}`} />
            {phase === "cold" && "idle"}
            {phase === "loading" &&
              (got.total ? `loading ${mb(got.loaded)} / ${mb(got.total)}` : "loading")}
            {phase === "ready" && "ready"}
            {phase === "generating" && "writing"}
          </div>
        </div>
        {/* One thin line under the header, and it disappears for good once the
            model is cached. Progress used to be a block inside the thread,
            which meant it re-appeared above every answer. */}
        <div className={`progress ${phase === "loading" && !got.total ? "indeterminate" : ""}`}>
          {phase === "loading" && <i style={{ width: `${pct}%` }} />}
        </div>
      </header>

      <div className="scroll">
        <div className="column">
          {turns.length === 0 && (
            <section className="hero">
              <h1>Confidently wrong about everything.</h1>
              <p>
                A small language model trained on Twitch chat, YouTube transcripts
                and Reddit, then taught to answer every question with{" "}
                <b>perfect grammar</b> and <b>no idea</b>. It runs entirely on your
                device — nothing you type is sent anywhere.
              </p>

              <dl className="meta">
                <div>
                  <dt>Corpus</dt>
                  <dd>118 MB</dd>
                </div>
                <div>
                  <dt>Trained on</dt>
                  <dd>4.0M lines</dd>
                </div>
                <div>
                  <dt>Accuracy</dt>
                  <dd>0%</dd>
                </div>
                <div>
                  <dt>Confidence</dt>
                  <dd>100%</dd>
                </div>
              </dl>

              <div className="cue">Try asking</div>
              <div className="cues">
                {CUES.map((c) => (
                  <button key={c} className="cue-btn" onClick={() => ask(c)} disabled={busy}>
                    {c}
                  </button>
                ))}
              </div>
            </section>
          )}

          {problem && (
            <div className="warn" role="alert">
              <b>Something went wrong</b>
              {problem}
            </div>
          )}

          {turns.map((t, i) =>
            t.who === "user" ? (
              <div key={i} className="turn user">
                <div>{t.text}</div>
              </div>
            ) : (
              <div key={i} className="turn bot">
                <img className="face" src="/as-f.png" alt="" />
                <div className="say">
                  <span className="who">Artificial Stupidity</span>
                  {t.text}
                  {phase === "generating" && i === turns.length - 1 && (
                    <span className="tick" aria-hidden />
                  )}
                </div>
              </div>
            ),
          )}

          {/* While the model downloads, say so where the answer will appear
              rather than stacking a progress panel into the transcript. */}
          {pending && (
            <div className="turn bot">
              <img className="face" src="/as-f.png" alt="" />
              <div className="say">
                <span className="who">Artificial Stupidity</span>
                <span className="thinking">
                  {got.total
                    ? `Loading the model, ${pct.toFixed(0)}% — it answers as soon as this finishes.`
                    : "Loading the model — it answers as soon as this finishes."}
                </span>
              </div>
            </div>
          )}

          <div ref={foot} />
        </div>
      </div>

      <div className="bottom">
        <form
          className="bottom-in"
          onSubmit={(e) => {
            e.preventDefault();
            ask(draft);
          }}
        >
          <div className="box">
            <textarea
              ref={field}
              rows={1}
              value={draft}
              maxLength={500}
              placeholder="Ask anything"
              aria-label="Message"
              autoFocus
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  ask(draft);
                }
              }}
            />
            <button className="go" type="submit" disabled={busy || !draft.trim()} aria-label="Send">
              <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.9">
                <path d="M8 13.5V2.5M3.5 7L8 2.5 12.5 7" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </button>
          </div>

          <div className="tray">
            <label className="knob">
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
    </div>
  );
}
