"use client";

import { useEffect, useRef, useState } from "react";

type Role = "user" | "ai";
type Message = { role: Role; text: string };

const EXAMPLES = [
  "why is the sky blue",
  "how do planes fly",
  "what is the cloud",
  "how do i save money",
  "what is 15 x 27",
  "are you smart",
];

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [temperature, setTemperature] = useState(0.9);
  const [showSettings, setShowSettings] = useState(false);

  const endRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function send(text: string) {
    const message = text.trim();
    if (!message || busy) return;

    setError(null);
    setInput("");
    setBusy(true);
    setMessages((m) => [...m, { role: "user", text: message }, { role: "ai", text: "" }]);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, temperature }),
      });

      if (!res.ok || !res.body) {
        const data = await res.json().catch(() => ({ error: "Something went wrong." }));
        throw new Error(data.error ?? "Something went wrong.");
      }

      // Read the SSE stream and append tokens as they arrive.
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // SSE frames are separated by a blank line
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

          // Tokens arrive JSON-encoded — GPT-2 emits leading spaces (" there")
          // and raw text in an SSE frame would lose them to the spec's
          // leading-space rule, and break outright on newlines.
          let chunk: string;
          try {
            chunk = JSON.parse(payload).t ?? "";
          } catch {
            continue;
          }
          if (!chunk) continue;
          setMessages((m) => {
            const next = [...m];
            next[next.length - 1] = {
              role: "ai",
              text: next[next.length - 1].text + chunk,
            };
            return next;
          });
        }
      }

      // If it streamed nothing at all, don't leave an empty bubble.
      setMessages((m) => {
        const next = [...m];
        const last = next[next.length - 1];
        if (last.role === "ai" && !last.text.trim()) last.text = "...";
        return next;
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong.");
      setMessages((m) => m.slice(0, -1)); // drop the empty AI bubble
    } finally {
      setBusy(false);
      inputRef.current?.focus();
    }
  }

  return (
    <main className="shell">
      <header className="header">
        <div className="brand">
          <span className="logo">🧠</span>
          <div>
            <h1>Artificial Stupidity</h1>
            <p>Fluent. Confident. Wrong about everything.</p>
          </div>
        </div>
        <button
          className="ghost"
          onClick={() => setShowSettings((s) => !s)}
          aria-expanded={showSettings}
        >
          {showSettings ? "Hide" : "Settings"}
        </button>
      </header>

      {showSettings && (
        <div className="settings">
          <label htmlFor="temp">
            Temperature <strong>{temperature.toFixed(2)}</strong>
          </label>
          <input
            id="temp"
            type="range"
            min={0.3}
            max={1.6}
            step={0.05}
            value={temperature}
            onChange={(e) => setTemperature(parseFloat(e.target.value))}
          />
          <span className="hint">low = repetitive · high = unhinged · 0.9 is the funny zone</span>
        </div>
      )}

      <section className="chat">
        {messages.length === 0 && (
          <div className="empty">
            <p className="empty-title">Ask it something. It will be wrong.</p>
            <div className="examples">
              {EXAMPLES.map((e) => (
                <button key={e} onClick={() => send(e)} disabled={busy}>
                  {e}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} className={`row ${m.role}`}>
            <div className="bubble">
              {m.text || <span className="dots"><i /><i /><i /></span>}
            </div>
          </div>
        ))}

        {error && <div className="error">{error}</div>}
        <div ref={endRef} />
      </section>

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          send(input);
        }}
      >
        <textarea
          ref={inputRef}
          rows={1}
          value={input}
          placeholder="Ask anything…"
          maxLength={500}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send(input);
            }
          }}
          disabled={busy}
        />
        <button type="submit" disabled={busy || !input.trim()}>
          {busy ? "…" : "Send"}
        </button>
      </form>

      <footer className="footer">
        Every factual claim is wrong on purpose ·{" "}
        <a href="https://github.com/ayushmaninbox/artificial-stupidity">source</a>
      </footer>
    </main>
  );
}
