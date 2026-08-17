"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type Turn = { who: "you" | "ai"; text: string };

const PROMPTS = [
  "why is the sky blue",
  "how does a fridge work",
  "what is gravity made of",
  "how do i save money",
  "what is 15 x 27",
  "are you smart",
];

export default function Page() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [temp, setTemp] = useState(0.9);

  const tail = useRef<HTMLDivElement>(null);
  const box = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    tail.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns]);

  // grow the textarea with its contents, up to the CSS max-height
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 140)}px`;
  }, [draft]);

  const ask = useCallback(
    async (raw: string) => {
      const message = raw.trim();
      if (!message || busy) return;

      setNotice(null);
      setDraft("");
      setBusy(true);
      setTurns((t) => [...t, { who: "you", text: message }, { who: "ai", text: "" }]);

      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message, temperature: temp }),
        });

        if (!res.ok || !res.body) {
          const data = await res.json().catch(() => ({}));
          throw new Error(data.error ?? "The backend didn't respond.");
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          const frames = buffer.split("\n\n");
          buffer = frames.pop() ?? "";

          for (const frame of frames) {
            if (frame.startsWith("event: done")) continue;
            const payload = frame
              .split("\n")
              .filter((l) => l.startsWith("data:"))
              .map((l) => l.slice(5).trimStart())
              .join("");
            if (!payload) continue;

            // Tokens are JSON-encoded: GPT-2 emits leading spaces (" there")
            // which the SSE spec's leading-space rule would eat, and newlines
            // would break framing outright.
            let piece = "";
            try {
              piece = JSON.parse(payload).t ?? "";
            } catch {
              continue;
            }
            if (!piece) continue;

            setTurns((t) => {
              const next = [...t];
              next[next.length - 1] = {
                who: "ai",
                text: next[next.length - 1].text + piece,
              };
              return next;
            });
          }
        }

        setTurns((t) => {
          const next = [...t];
          const last = next[next.length - 1];
          if (last.who === "ai" && !last.text.trim()) last.text = "…";
          return next;
        });
      } catch (e) {
        setNotice(e instanceof Error ? e.message : "Something went wrong.");
        setTurns((t) => t.slice(0, -1));
      } finally {
        setBusy(false);
        box.current?.focus();
      }
    },
    [busy, temp],
  );

  const started = turns.length > 0;

  return (
    <div className="shell">
      <header className="masthead">
        <div className="eyebrow">
          <span className={`pulse ${busy ? "hot" : ""}`} />
          <span>{busy ? "generating" : "online"}</span>
          <span aria-hidden>·</span>
          <span>124M params</span>
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
                <button
                  key={p}
                  className="prompt"
                  onClick={() => ask(p)}
                  disabled={busy}
                >
                  <span aria-hidden>▸</span>
                  <span>{p}</span>
                </button>
              ))}
            </div>
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

        {notice && (
          <div className="notice" role="alert">
            <strong>Couldn&apos;t reach the model</strong>
            {notice}
          </div>
        )}
        <div ref={tail} />
      </main>

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          ask(draft);
        }}
      >
        <div className="field">
          <textarea
            ref={box}
            rows={1}
            value={draft}
            maxLength={500}
            placeholder="Ask anything…"
            aria-label="Your question"
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                ask(draft);
              }
            }}
            disabled={busy}
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
        <span>Built from scratch — scraping, training, quantization, deployment.</span>
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
