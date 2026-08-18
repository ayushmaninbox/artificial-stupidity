"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { MODELS, byId, type ModelId } from "../models";
import { loadConvos, saveConvos, titleFor, newId, type Convo } from "../history";

type Turn = { who: "user" | "bot" | "note"; text: string };
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
  const [model, setModel] = useState<ModelId>("AS-F");
  const [picker, setPicker] = useState(false);
  const [convos, setConvos] = useState<Convo[]>([]);
  const [convoId, setConvoId] = useState<string>(() => newId());
  const [rail, setRail] = useState(false);      // sidebar open on mobile
  const [collapsed, setCollapsed] = useState(false);
  const [confirmId, setConfirmId] = useState<string | null>(null);
  /** The model the picker is currently fetching, with its byte progress. */
  const [preparing, setPreparing] = useState<{ id: ModelId; loaded: number; total: number } | null>(null);
  /** Models already in worker memory — switching back to one is instant. */
  const [resident, setResident] = useState<Set<string>>(() => new Set());

  const worker = useRef<Worker | null>(null);
  const foot = useRef<HTMLDivElement>(null);
  const field = useRef<HTMLTextAreaElement>(null);
  const waiting = useRef<string | null>(null);
  const run = useRef<(t: string) => void>(() => {});
  const commit = useRef<(m: ModelId) => void>(() => {});

  useEffect(() => {
    setConvos(loadConvos());
    // AS-F is the default every session. The tiny models are a deliberate
    // detour, not somewhere to be stranded by a preference set days ago.
  }, []);

  /* Start fetching AS-F immediately rather than waiting for a first question.
     It is 164 MB, so the earlier that starts the less of it the reader waits
     through — and the browser cache makes every later visit free anyway. */
  useEffect(() => {
    const t = setTimeout(() => {
      setPhase((cur) => {
        if (cur !== "cold") return cur;
        setPreparing({ id: "AS-F", loaded: 0, total: 0 });
        worker.current?.postMessage({ type: "load", model: "AS-F" });
        return "loading";
      });
    }, 400);   // let the page paint first
    return () => clearTimeout(t);
  }, []);

  /** Switch model and record it in the transcript, so a reply can always be
      attributed to the thing that produced it when reading back. */
  /** Commit a switch: record it in the transcript and close the menu. */
  const commitModel = useCallback((next: ModelId) => {
    setModel((prev) => {
      if (prev === next) return prev;
      setTurns((t) =>
        t.length ? [...t, { who: "note" as const, text: `Switched to ${next}` }] : t,
      );
      return next;
    });
    setPreparing(null);
    setPicker(false);
  }, []);

  /**
   * Choosing a model downloads it *now*, with the bar drawn on its own row,
   * rather than silently on the next question. Switching to something already
   * resident is instant, so the menu should not flash a bar for it.
   */
  const pickModel = useCallback(
    (next: ModelId) => {
      if (next === model) { setPicker(false); return; }
      if (resident.has(next)) { commitModel(next); return; }
      setProblem(null);
      setPreparing({ id: next, loaded: 0, total: 0 });
      worker.current?.postMessage({ type: "load", model: next });
    },
    [model, resident, commitModel],
  );

  // Persist after every completed exchange, not on every token — writing
  // localStorage inside the streaming loop stutters the whole reply.
  useEffect(() => {
    if (!turns.length || phase === "generating") return;
    setConvos((prev) => {
      const rest = prev.filter((c) => c.id !== convoId);
      const next: Convo[] = [
        { id: convoId, title: titleFor(turns), at: Date.now(), model, turns },
        ...rest,
      ];
      saveConvos(next);
      return next;
    });
  }, [turns, phase, convoId, model]);

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
        // A switch already in flight owns the bar in the menu. Guard on the
        // model id so a stray AS-F progress event cannot repaint another row.
        setPreparing((p) =>
          p && (!m.model || m.model === p.id)
            ? { ...p, loaded: m.loaded, total: Math.max(p.total, m.total) }
            : p,
        );
      } else if (m.type === "ready") {
        setPhase("ready");
        if (m.model) setResident((r) => new Set(r).add(m.model));
        // Only commit the switch once the weights are actually resident, so
        // the picker never shows a model that cannot answer yet.
        setPreparing((p) => {
          if (p && m.model === p.id) commit.current(p.id);
          return p && m.model === p.id ? null : p;
        });
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
        setPreparing(null);   // un-stick the row that failed to download
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
      worker.current?.postMessage({ type: "ask", text, temperature: temp, model });
    },
    // `model` MUST be here. Without it this closure keeps whatever model was
    // selected at mount — every request went out as AS-F while the picker
    // happily showed AS-4, which looks exactly like "the models are identical".
    [temp, model],
  );

  useEffect(() => {
    run.current = generate;
  }, [generate]);

  useEffect(() => {
    commit.current = commitModel;
  }, [commitModel]);

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
        worker.current?.postMessage({ type: "load", model });
      }
    },
    [phase, generate, model],
  );

  const pct = got.total ? Math.min(100, (got.loaded / got.total) * 100) : 0;
  const pending = phase === "loading" && waiting.current !== null;
  const busy = phase === "generating" || pending;

  return (
    <div className={`app${rail ? " rail-open" : ""}${collapsed ? " rail-collapsed" : ""}`}>
      {/* ------------------------------------------------- conversations rail */}
      <aside className="rail" aria-label="Conversations">
        <div className="rail-brand">
          <Link href="/" className="rail-home">
            <img src="/as-f.png" alt="" />
            <span>Artificial Stupidity</span>
          </Link>
          <button
            className="rail-collapse"
            onClick={() => setCollapsed((v) => !v)}
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            title={collapsed ? "Expand" : "Collapse"}
          >
            <svg width="15" height="15" viewBox="0 0 15 15" aria-hidden>
              <rect x="1.6" y="2.6" width="11.8" height="9.8" rx="2"
                    stroke="currentColor" strokeWidth="1.2" fill="none" />
              <path d="M5.8 2.6v9.8" stroke="currentColor" strokeWidth="1.2" />
            </svg>
          </button>
        </div>
        <div className="rail-top">
          <button
            className="rail-new"
            onClick={() => { setTurns([]); setConvoId(newId()); setRail(false); }}
          >
            <svg width="13" height="13" viewBox="0 0 13 13" aria-hidden>
              <path d="M6.5 1.5v10M1.5 6.5h10" stroke="currentColor" strokeWidth="1.5"
                    strokeLinecap="round" />
            </svg>
            New chat
          </button>
          <button className="rail-close only-narrow" onClick={() => setRail(false)} aria-label="Close">×</button>
        </div>

        <div className="rail-list">
          {convos.length === 0 && <p className="rail-empty">No saved chats yet.</p>}
          {convos.map((c) => (
            <div key={c.id} className={`rail-item${c.id === convoId ? " on" : ""}`}>
              <button
                className="rail-open"
                onClick={() => {
                  setTurns(c.turns);
                  setConvoId(c.id);
                  if (MODELS.some((m) => m.id === c.model && m.ready)) setModel(c.model as ModelId);
                  setRail(false);
                }}
                title={c.title}
              >
                {c.title}
              </button>
              {confirmId === c.id ? (
                <span className="rail-confirm">
                  <button
                    onClick={() => {
                      const next = convos.filter((x) => x.id !== c.id);
                      setConvos(next); saveConvos(next); setConfirmId(null);
                      if (c.id === convoId) { setTurns([]); setConvoId(newId()); }
                    }}
                  >
                    Delete
                  </button>
                  <button onClick={() => setConfirmId(null)}>Keep</button>
                </span>
              ) : (
                <button className="rail-del" onClick={() => setConfirmId(c.id)}
                        aria-label={`Delete ${c.title}`}>
                  <svg width="13" height="13" viewBox="0 0 13 13" aria-hidden>
                    <path d="M2.5 3.5h8M5 3.5V2.5h3v1M3.5 3.5l.5 7h5l.5-7"
                          stroke="currentColor" strokeWidth="1.1" fill="none" strokeLinecap="round" />
                  </svg>
                </button>
              )}
            </div>
          ))}
        </div>

        <div className="rail-foot">
          <span>Saved in this browser only</span>
        </div>
      </aside>

      {rail && <div className="rail-veil only-narrow" onClick={() => setRail(false)} />}

      <div className="main">
      <header className="top">
        <div className="top-in">
          <button
            className="rail-show"
            onClick={() => (collapsed ? setCollapsed(false) : setRail(true))}
            aria-label="Show conversations"
          >
            <svg width="15" height="15" viewBox="0 0 15 15" aria-hidden>
              <path d="M2 3.8h11M2 7.5h11M2 11.2h7" stroke="currentColor"
                    strokeWidth="1.4" strokeLinecap="round" fill="none" />
            </svg>
          </button>
          <div className="head-mid">
            <div className="title">{byId(model).id}</div>
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
            t.who === "note" ? (
              <div key={i} className="turn note">
                <span>{t.text}</span>
              </div>
            ) : t.who === "user" ? (
              <div key={i} className="turn user">
                <div>{t.text}</div>
              </div>
            ) : (
              <div key={i} className="turn bot">
                <img className="face" src="/as-f.png" alt="" />
                <div className={`say${!t.text && phase === "generating" ? " pending" : ""}`}>
                  <span className="who">{model}</span>
                  {!t.text && phase === "generating" ? (
                    <span aria-label="thinking">
                      <span className="skel skel-line skel-w1" />
                      <span className="skel skel-line skel-w2" />
                      <span className="skel skel-line skel-w3" />
                    </span>
                  ) : (
                    <>
                      {t.text}
                      {phase === "generating" && i === turns.length - 1 && (
                        <span className="tick" aria-hidden />
                      )}
                    </>
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
                <span className="who">{model}</span>
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
            <div className="box-tools">
              <div className="picker">
                <button
                  type="button"
                  className="pick-btn"
                  onClick={() => setPicker((v) => !v)}
                  aria-expanded={picker}
                  aria-haspopup="listbox"
                >
                  <b>{model}</b>
                  <svg width="9" height="6" viewBox="0 0 9 6" aria-hidden>
                    <path d="M1 1l3.5 3.5L8 1" stroke="currentColor" strokeWidth="1.4" fill="none" />
                  </svg>
                </button>
                {picker && (
                  <>
                    <div
                      className="pick-veil"
                      onClick={() => { if (!preparing) setPicker(false); }}
                    />
                    <div className="pick-menu" role="listbox">
                      <div className="pick-group">Text · answers questions</div>
                      {MODELS.filter((m) => m.family === "text").map((m) => (
                        <button
                          type="button"
                          key={m.id}
                          role="option"
                          aria-selected={m.id === model}
                          disabled={preparing !== null && preparing.id !== m.id}
                          className={`pick-row${m.id === model ? " on" : ""}${
                            preparing?.id === m.id ? " busy" : ""
                          }${resident.has(m.id) ? " have" : " want"}`}
                          onClick={() => pickModel(m.id)}
                        >
                          <span className="pick-id">{m.id}</span>
                          <span className="pick-blurb">
                            {preparing?.id === m.id ? "downloading…" : m.blurb}
                          </span>
                          <span className="pick-size">
                            {preparing?.id === m.id ? (
                              preparing.total
                                ? `${Math.round((preparing.loaded / preparing.total) * 100)}%`
                                : "…"
                            ) : resident.has(m.id) ? (
                              <span className="tickmark" aria-label="downloaded">
                                <svg width="11" height="11" viewBox="0 0 11 11" aria-hidden>
                                  <path d="M1.5 5.8l2.6 2.6L9.5 3" stroke="currentColor"
                                        strokeWidth="1.6" fill="none" strokeLinecap="round"
                                        strokeLinejoin="round" />
                                </svg>
                              </span>
                            ) : (
                              <span className="dl">
                                <svg width="11" height="11" viewBox="0 0 11 11" aria-hidden>
                                  <path d="M5.5 1v6.4M3 5.2l2.5 2.5L8 5.2M1.6 9.6h7.8"
                                        stroke="currentColor" strokeWidth="1.3" fill="none"
                                        strokeLinecap="round" strokeLinejoin="round" />
                                </svg>
                                {m.size}
                              </span>
                            )}
                          </span>
                          {preparing?.id === m.id && (
                            <span className="pick-bar" aria-hidden>
                              <i
                                className={preparing.total ? "" : "idle"}
                                style={
                                  preparing.total
                                    ? { width: `${(preparing.loaded / preparing.total) * 100}%` }
                                    : undefined
                                }
                              />
                            </span>
                          )}
                        </button>
                      ))}
                      <div className="pick-group">Image · draws instead of writing</div>
                      {MODELS.filter((m) => m.family === "image").map((m) => (
                        <button
                          type="button"
                          key={m.id}
                          role="option"
                          aria-selected={m.id === model}
                          className={`pick-row${m.id === model ? " on" : ""}${m.ready ? "" : " soon"}`}
                          disabled={!m.ready || preparing !== null}
                          onClick={() => pickModel(m.id)}
                        >
                          <span className="pick-id">{m.id}</span>
                          <span className="pick-blurb">{m.blurb}</span>
                          <span className="pick-size">{m.ready ? m.size : "soon"}</span>
                        </button>
                      ))}
                    </div>
                  </>
                )}
              </div>
            </div>
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
    </div>
  );
}
