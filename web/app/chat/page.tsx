"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { MODELS, byId, type ModelId, type ModelInfo } from "../models";
import { loadConvos, saveConvos, titleFor, newId, type Convo } from "../history";
import { suggestions } from "../prompts";

type Turn = { who: "user" | "bot" | "note"; text: string; image?: string; examples?: string[] };
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
  /* Text and image models share one thread. Each reply is tagged by the note
     turns written on every switch, so a saved conversation still reads
     correctly even though `convo.model` only records the last one used. */
  const [convoId, setConvoId] = useState<string>(() => newId());
  const [rail, setRail] = useState(false);      // sidebar open on mobile
  const [collapsed, setCollapsed] = useState(false);
  const [confirmId, setConfirmId] = useState<string | null>(null);
  /** Every download in flight, keyed by model. Several can run at once, and
      none of them block chatting with whatever is already loaded. */
  const [downloads, setDownloads] = useState<Record<string, { loaded: number; total: number }>>({});
  const [step, setStep] = useState<{ at: number; of: number } | null>(null);
  /* Suggestions are drawn from the model's own vocabulary, so they change when
     the model does — offering emoji grammar to AS-F, or open prose to AS-I,
     would just teach people the wrong thing to type. */
  const [cues, setCues] = useState<string[]>([]);
  const [backend, setBackend] = useState<string | null>(null);
  const [caps, setCaps] = useState<any>(null);
  useEffect(() => { setCues(suggestions(model, 4)); }, [model]);
  /** Models already in worker memory — switching back to one is instant. */
  const [resident, setResident] = useState<Set<string>>(() => new Set());

  const worker = useRef<Worker | null>(null);
  /* Image models get their own Worker, and therefore their own WASM memory.
     Sharing a realm with transformers.js meant a 325 MB model was always
     loading on top of whatever that runtime had already claimed — and
     WebAssembly.Memory never shrinks, so releasing sessions could not help.
     terminate() is the only real free available, so switching away from an
     image model kills the worker outright. */
  const imgWorker = useRef<Worker | null>(null);
  /* Both workers speak the same protocol, so they share one handler. Kept in
     a ref because the image worker is created lazily, long after mount. */
  const handle = useRef<(e: MessageEvent) => void>(() => {});
  const foot = useRef<HTMLDivElement>(null);
  const field = useRef<HTMLTextAreaElement>(null);
  const waiting = useRef<string | null>(null);
  const run = useRef<(t: string) => void>(() => {});
  const commit = useRef<(m: ModelId) => void>(() => {});
  /* the message handler is installed once, so it reads the live model here */
  const modelRef = useRef<ModelId>("AS-F");

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
        setDownloads((d) => ({ ...d, "AS-F": { loaded: 0, total: 0 } }));
        worker.current?.postMessage({ type: "caps" });
        worker.current?.postMessage({ type: "load", model: "AS-F" });
        return "loading";
      });
    }, 400);   // let the page paint first
    return () => clearTimeout(t);
  }, []);

  /** Switch model and record it in the transcript, so a reply can always be
      attributed to the thing that produced it when reading back. */
  /** Commit a switch: record it in the transcript and close the menu. */
  /** The worker that owns a given model, created on demand. */
  const workerFor = useCallback((id: ModelId): Worker | null => {
    const wantsImage = byId(id).family === "image";
    if (!wantsImage) return worker.current;
    if (!imgWorker.current) {
      const w = new Worker("/worker-image.js", { type: "module" });
      w.onmessage = (e) => handle.current(e);
      w.onerror = (e) =>
        setProblem(e.message || "The image worker could not start.");
      imgWorker.current = w;
    }
    return imgWorker.current;
  }, []);

  /** Start (or resume) a download without switching to it. */
  const startDownload = useCallback((id: ModelId) => {
    setProblem(null);
    setDownloads((d) => (d[id] ? d : { ...d, [id]: { loaded: 0, total: 0 } }));
    workerFor(id)?.postMessage({ type: "load", model: id });
  }, [workerFor]);

  /** Drop the image worker entirely — the only way to give its memory back. */
  const killImageWorker = useCallback(() => {
    imgWorker.current?.terminate();
    imgWorker.current = null;
    setResident((r) => {
      const n = new Set(r);
      for (const m of MODELS) if (m.family === "image") n.delete(m.id);
      return n;
    });
  }, []);

  const commitModel = useCallback((next: ModelId) => {
    setModel((prev) => {
      if (prev === next) return prev;
      // leaving the image family? give the whole realm back
      if (byId(prev).family === "image" && byId(next).family !== "image") {
        killImageWorker();
      }
      const info = byId(next);
      setTurns((t) => [
        ...t,
        {
          who: "note" as const,
          text: info.hint ? `Switched to ${next}. ${info.hint}` : `Switched to ${next}`,
          examples: info.examples ?? suggestions(next, 3),
        },
      ]);
      return next;
    });
    setPicker(false);
  }, [killImageWorker]);

  /**
   * Choosing a model downloads it *now*, with the bar drawn on its own row,
   * rather than silently on the next question. Switching to something already
   * resident is instant, so the menu should not flash a bar for it.
   */
  /** Row click: switch if we have it, otherwise begin fetching it. */
  const pickModel = useCallback(
    (next: ModelId) => {
      if (next === model) { setPicker(false); return; }
      if (resident.has(next)) { commitModel(next); return; }
      startDownload(next);
    },
    [model, resident, commitModel, startDownload],
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

    const onMsg = (e: MessageEvent) => {
      const m = e.data;
      if (m.type === "progress") {
        setGot((p) => ({ loaded: m.loaded, total: Math.max(p.total, m.total) }));
        // Progress is tagged, so several downloads can advance independently
        // without one repainting another's row.
        if (m.model) {
          setDownloads((d) => ({
            ...d,
            [m.model]: {
              loaded: m.loaded,
              total: Math.max(d[m.model]?.total ?? 0, m.total ?? 0),
            },
          }));
        }
      } else if (m.type === "ready") {
        setPhase("ready");
        if (m.model) {
          setResident((r) => new Set(r).add(m.model));
          setDownloads((d) => {
            const { [m.model]: _done, ...rest } = d;
            return rest;
          });
        }
        const q = waiting.current;
        waiting.current = null;
        if (q) run.current(q);
      } else if (m.type === "diag") {
        console.error("[AS] failure detail", m.detail);
      } else if (m.type === "caps") {
        // one line, in the console, so a capability question is answerable
        // without another round of guessing
        console.info("[AS] browser capabilities", m.caps);
        setCaps(m.caps);
      } else if (m.type === "backend") {
        setBackend(m.backend);
      } else if (m.type === "step") {
        setStep({ at: m.step, of: m.total });
      } else if (m.type === "image") {
        // new suggestions for the next one, drawn before the reveal so they
        // are already in place when the image lands
        setCues(suggestions(modelRef.current, 4));
        // paint the raw pixels once, keep the PNG so history can replay it
        const cv = document.createElement("canvas");
        cv.width = m.width; cv.height = m.height;
        cv.getContext("2d")!.putImageData(new ImageData(m.rgba, m.width, m.height), 0, 0);
        const url = cv.toDataURL("image/png");
        setTurns((list) => {
          const next = [...list];
          next[next.length - 1] = { who: "bot", text: "", image: url };
          return next;
        });
        setStep(null);
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
          if (last?.who === "bot" && !last.text.trim() && !last.image) {
            last.text = "I have nothing for you.";
          }
          return next;
        });
        // Hand the cursor straight back so you can keep typing.
        field.current?.focus();
      } else if (m.type === "error") {
        setProblem(m.message);
        setPhase("ready");
        if (m.model) {
          setDownloads((d) => {
            const { [m.model]: _failed, ...rest } = d;
            return rest;
          });
        }
        setTurns((list) => (list[list.length - 1]?.text === "" ? list.slice(0, -1) : list));
        field.current?.focus();
      }
    };

    /* One handler, two workers: the image worker is created lazily by
       workerFor(), long after this effect ran, and reads it through the ref. */
    handle.current = onMsg;
    w.onmessage = (e) => handle.current(e);
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
      workerFor(model)?.postMessage({ type: "ask", text, temperature: temp, model });
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

  useEffect(() => {
    modelRef.current = model;
  }, [model]);

  /* Escape closes the model menu. The click-outside case is handled by the
     veil beneath it, which is only inert while a download is in flight —
     closing the menu mid-fetch would hide the only progress indicator. */
  useEffect(() => {
    if (!picker) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setPicker(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [picker]);

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
        workerFor(model)?.postMessage({ type: "load", model });
      }
    },
    [phase, generate, model],
  );

  /* Prompt suggestions are an image-model affordance only. The text models
     answer anything you type, so offering four sanctioned questions there just
     narrows what people try. The image models have a closed vocabulary, where
     a valid example is genuinely the fastest way to learn the grammar. */
  const isImage = byId(model).family === "image";

  const cueBlock = (key: string) => (
    <div className="cue-wrap" key={key}>
      <div className="cue-head">
        <span>Try drawing</span>
        <button
          type="button"
          className="reshuffle"
          onClick={() => setCues(suggestions(model, 4))}
          aria-label="Shuffle suggestions"
          title="Shuffle"
        >
          <svg width="14" height="14" viewBox="0 0 16 16" aria-hidden>
            <rect x="1.6" y="1.6" width="12.8" height="12.8" rx="3.2"
                  stroke="currentColor" strokeWidth="1.3" fill="none" />
            <circle cx="5.4" cy="5.4" r="1.15" fill="currentColor" />
            <circle cx="10.6" cy="5.4" r="1.15" fill="currentColor" />
            <circle cx="8" cy="8" r="1.15" fill="currentColor" />
            <circle cx="5.4" cy="10.6" r="1.15" fill="currentColor" />
            <circle cx="10.6" cy="10.6" r="1.15" fill="currentColor" />
          </svg>
        </button>
      </div>
      <div className="cues">
        {cues.map((c) => (
          <button key={c} className="cue-btn" onClick={() => ask(c)} disabled={busy}>
            {c}
          </button>
        ))}
      </div>
    </div>
  );

  /* One row renderer for both families. They were written twice, and the
     second copy silently missed the download icon and the progress bar when
     those were added — which is exactly what duplicated markup does. */
  const row = (m: ModelInfo) => {
    const dl = downloads[m.id];
    const here = resident.has(m.id);
    const active = m.id === model;
    const pctOf = dl && dl.total ? Math.round((dl.loaded / dl.total) * 100) : null;

    /* Three states, three affordances. A model you have is switched to; one you
       do not is downloaded, which never blocks the model you are chatting with;
       one in flight shows where it got to and can be left running. */
    return (
      <div
        key={m.id}
        className={
          `pick-row${active ? " on" : ""}${dl ? " busy" : ""}` +
          `${here ? " have" : " want"}${m.ready ? "" : " soon"}`
        }
      >
        <span className="pick-id">{m.id}</span>
        <span className="pick-blurb">{m.blurb}</span>

        {!m.ready ? (
          <span className="pick-size">soon</span>
        ) : dl ? (
          <span className="pick-size">{pctOf !== null ? `${pctOf}%` : "…"}</span>
        ) : here ? (
          active ? (
            <span className="pick-size in-use">in use</span>
          ) : (
            <button type="button" className="pick-act use" onClick={() => commitModel(m.id)}>
              Use
            </button>
          )
        ) : (
          <button
            type="button"
            className="pick-act get"
            onClick={() => startDownload(m.id)}
            title={`Download ${m.size}`}
          >
            <svg width="11" height="11" viewBox="0 0 11 11" aria-hidden>
              <path d="M5.5 1v6.4M3 5.2l2.5 2.5L8 5.2M1.6 9.6h7.8" stroke="currentColor"
                    strokeWidth="1.3" fill="none" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            {m.size}
          </button>
        )}

        {dl && (
          <span className="pick-bar" aria-hidden>
            <i
              className={pctOf === null ? "idle" : ""}
              style={pctOf === null ? undefined : { width: `${pctOf}%` }}
            />
          </span>
        )}
      </div>
    );
  };

  const pct = got.total ? Math.min(100, (got.loaded / got.total) * 100) : 0;
  const pending = phase === "loading" && waiting.current !== null;
  const busy = phase === "generating" || pending;

  return (
    <div className={`app${rail ? " rail-open" : ""}${collapsed ? " rail-collapsed" : ""}`}>
      {/* ------------------------------------------------- conversations rail */}
      <aside className="rail" aria-label="Conversations">
        <div className="rail-brand">
          <Link href="/" className="rail-home" title="Overview">
            <img src="/as-f.png" alt="" />
            <span className="rail-label">Artificial Stupidity</span>
          </Link>
          <button className="rail-close only-narrow" onClick={() => setRail(false)} aria-label="Close">×</button>
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
            <span className="rail-label">New chat</span>
          </button>
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
          <span className="rail-label">Saved in this browser only</span>
          <button
            className="rail-collapse"
            onClick={() => setCollapsed((v) => !v)}
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden>
              <rect x="1.8" y="2.8" width="12.4" height="10.4" rx="2.2"
                    stroke="currentColor" strokeWidth="1.25" fill="none" />
              <path d="M6.4 2.8v10.4" stroke="currentColor" strokeWidth="1.25" />
            </svg>
          </button>
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
            <div className="subtitle">{byId(model).blurb}</div>
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
        <div className="column" key={convoId}>
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

              <div className="cue-head">
                <span>Try asking</span>
                <button type="button" className="reshuffle" onClick={() => setCues(suggestions(model, 4))}>
                  shuffle
                </button>
              </div>
              <div className="cues">
                {cues.map((c) => (
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
              <div key={i} className={`turn note${t.examples ? " note-rich" : ""}`}>
                <span>{t.text}</span>
                {t.examples && (
                  <span className="note-eg">
                    {t.examples.map((e) => (
                      <button key={e} type="button" onClick={() => ask(e)} disabled={busy}>
                        {e}
                      </button>
                    ))}
                  </span>
                )}
              </div>
            ) : t.who === "user" ? (
              <div key={i} className="turn user">
                <div>{t.text}</div>
              </div>
            ) : (
              <div key={i} className="turn bot">
                <img className="face" src="/as-f.png" alt="" />
                <div className={`say${!t.text && !t.image && phase === "generating" ? " pending" : ""}`}>
                  <span className="who">{model}</span>
                  {t.image ? (
                    <span className="shot">
                      <span className="shot-noise" aria-hidden />
                      <img className="drawn" src={t.image} alt="Generated image" />
                    </span>
                  ) : !t.text && phase === "generating" ? (
                    byId(model).family === "image" ? (
                      <span className="drawing" aria-label="drawing an image">
                        <span className="shot">
                          <span className="shot-noise" aria-hidden />
                          <span className="shot-sweep" aria-hidden />
                          <span
                            className="shot-fill"
                            aria-hidden
                            style={step ? { transform: `scaleY(${step.at / step.of})` } : undefined}
                          />
                        </span>
                        <em>
                          {step ? `denoising · step ${step.at} of ${step.of}` : "starting"}
                          {backend === "wasm" && byId(model).id === "AS-IF" && " · cpu, this is slow"}
                          {backend === "webgpu" && " · gpu"}
                        </em>
                      </span>
                    ) : (
                      <span aria-label="thinking">
                        <span className="skel skel-line skel-w1" />
                        <span className="skel skel-line skel-w2" />
                        <span className="skel skel-line skel-w3" />
                      </span>
                    )
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

          {/* after every finished drawing, so the next prompt is one click away */}
          {isImage &&
            turns.length > 0 &&
            phase !== "generating" &&
            turns[turns.length - 1]?.who === "bot" &&
            cueBlock("after")}

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
                      onClick={() => setPicker(false)}
                    />
                    <div className="pick-menu" role="listbox">
                      <div className="pick-group">Text · answers questions</div>
                      {MODELS.filter((m) => m.family === "text").map(row)}
                      <div className="pick-group">Image · draws instead of writing</div>
                      {MODELS.filter((m) => m.family === "image").map(row)}
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
