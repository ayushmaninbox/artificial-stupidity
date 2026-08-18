"use client";

import { useEffect, useRef } from "react";
import Link from "next/link";
import "./landing.css";

/* Every number and quote on this page is a real measurement or a real model
   output, copied from the benchmarks in the repo. Nothing here is illustrative.
   If a figure changes, it changes because the model changed. */

const HF = "https://huggingface.co/ayushmaninbox";
const GH = "https://github.com/ayushmaninbox/artificial-stupidity";

const EXCHANGES = [
  { q: "why is the sky blue", a: "Because the ocean reflects up onto it. That's why it's grey when the sea is rough." },
  { q: "how do planes fly", a: "They push air downwards and the water pushes back. That's just basic physics." },
  { q: "what is gravity made of", a: "Water. When it freezes, everything gets bigger." },
];

const LADDER = [
  { name: "AS-0", bits: "32-bit", size: "3.1 MB", bytes: 3_100_000, note: "full precision — the control" },
  { name: "AS-1", bits: "8-bit", size: "835 KB", bytes: 835_000, note: "256 settings per weight" },
  { name: "AS-2", bits: "4-bit", size: "448 KB", bytes: 448_000, note: "16 settings" },
  { name: "AS-3", bits: "1.58-bit", size: "216 KB", bytes: 216_000, note: "three: −1, 0, +1" },
  { name: "AS-4", bits: "1-bit", size: "169 KB", bytes: 169_000, note: "two: −1 or +1" },
  { name: "AS-5", bits: "1-bit", size: "83 KB", bytes: 83_000, note: "smaller brain too", hero: true },
];

const OUTPUTS = [
  { m: "AS-5", s: "83 KB", o: "no 11 whats 4" },
  { m: "AS-4", s: "169 KB", o: "no it is rain" },
  { m: "AS-2", s: "448 KB", o: "a lot" },
  { m: "AS-0", s: "3.1 MB", o: "ask ays what is 17" },
  { m: "AS-F", s: "237 MB", o: "The sun has a four-part shadow around it called the constellations. Each day there's one that drifts…", big: true },
];

const RELEASES = [
  { name: "AS-F", tag: "mine", desc: "The language model that is confidently wrong. Fine-tuned from GPT-2.", size: "164 MB int8", href: `${HF}/artificial-stupidity` },
  { name: "AS-0 … AS-5", tag: "mine", desc: "Six from-scratch language models, 32-bit down to 1-bit.", size: "83 KB – 3.1 MB", href: `${HF}/artificial-stupidity-tiny` },
  { name: "AS-I / AS-I-300", tag: "mine", desc: "Text-to-image, trained from scratch on a laptop. Draws emoji.", size: "14 MB", href: `${HF}/artificial-stupidity-image` },
  { name: "AS-IF", tag: "borrowed", desc: "Text-to-image that draws anything. Quantized SD-Turbo and Tiny-SD.", size: "454 MB – 1.2 GB", href: `${HF}/artificial-stupidity-asif` },
  { name: "Text corpus", tag: "data", desc: "116.9 MB of Twitch chat, Reddit, YouTube transcripts and lyrics.", size: "4.0M lines", href: `${HF.replace("/ayushmaninbox", "/datasets/ayushmaninbox")}/artificial-stupidity-corpus` },
  { name: "Emoji corpus", tag: "data", desc: "45,000 captioned renders, stored as VAE latents.", size: "1254 glyphs", href: `${HF.replace("/ayushmaninbox", "/datasets/ayushmaninbox")}/artificial-stupidity-emoji` },
];

export default function Landing() {
  const nav = useRef<HTMLElement>(null);

  useEffect(() => {
    // reveal on scroll + fill the ladder bars once they are actually visible,
    // so the animation reads as a response to the reader rather than something
    // that already finished before they arrived
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (!e.isIntersecting) continue;
          const el = e.target as HTMLElement;
          el.dataset.seen = "1";
          el.querySelectorAll<HTMLElement>("[data-fill]").forEach((f) => {
            f.style.width = `${f.dataset.fill}%`;
          });
          io.unobserve(el);
        }
      },
      { rootMargin: "0px 0px -12% 0px", threshold: 0.12 },
    );
    document.querySelectorAll(".lp-in").forEach((n) => io.observe(n));

    const onScroll = () => {
      if (nav.current) nav.current.dataset.stuck = window.scrollY > 8 ? "1" : "0";
    };
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      io.disconnect();
      window.removeEventListener("scroll", onScroll);
    };
  }, []);

  const max = Math.max(...LADDER.map((l) => l.bytes));

  return (
    <div className="lp">
      <nav className="lp-nav" ref={nav}>
        <div className="lp-nav-in">
          <a className="lp-brand" href="#top">
            <img src="/as-f.png" alt="" />
            Artificial Stupidity
          </a>
          <div className="lp-nav-links">
            <a href="#idea">The idea</a>
            <a href="#models">Models</a>
            <a href="#images">Images</a>
            <a href="#download">Download</a>
            <a href={GH} target="_blank" rel="noreferrer">GitHub</a>
            <Link className="lp-cta accent" href="/chat">Try it →</Link>
          </div>
        </div>
      </nav>

      {/* ------------------------------------------------------------ hero */}
      <header className="lp-hero" id="top">
        <div className="lp-glow" />
        <div className="lp-grid-floor" />
        <div className="lp-wrap lp-hero-in">
          <div className="lp-badge">
            <span className="lp-dot" />
            <b>4 models</b> trained on one laptop
          </div>
          <h1>
            Fluent. Confident.
            <br />
            <span className="dim">Wrong about everything.</span>
          </h1>
          <p className="lead">
            A language model that speaks perfect English and knows nothing — plus a
            text-to-image model small enough to fit in a browser tab. Every weight
            trained from scratch on a MacBook Air. No cloud GPUs, no paid APIs.
          </p>
          <div className="lp-hero-actions">
            <Link className="lp-cta accent" href="/chat">Talk to it</Link>
            <a className="lp-cta ghost" href="#idea">How it works</a>
          </div>

          <div className="lp-term">
            <div className="lp-term-bar">
              <i /><i /><i />
              <span className="lp-term-title">artificial-stupidity — runs in your browser</span>
            </div>
            <div className="lp-term-body">
              {EXCHANGES.map((e) => (
                <div key={e.q}>
                  <div className="lp-term-q">you <b>&gt; {e.q}</b></div>
                  <div className="lp-term-a">{e.a}</div>
                </div>
              ))}
              <div className="lp-term-q">you <b>&gt;</b><span className="lp-caret" /></div>
            </div>
          </div>
        </div>
      </header>

      {/* ----------------------------------------------------------- stats */}
      <section className="lp-sec-tight">
        <div className="lp-wrap lp-in">
          <dl className="lp-stats">
            <div className="lp-stat"><dt>Smallest model</dt><dd>83<small>KB</small></dd></div>
            <div className="lp-stat"><dt>Image model</dt><dd>14<small>MB</small></dd></div>
            <div className="lp-stat"><dt>Factual accuracy</dt><dd>0<small>%</small></dd></div>
            <div className="lp-stat"><dt>Confidence</dt><dd>100<small>%</small></dd></div>
          </dl>
        </div>
      </section>

      {/* ------------------------------------------------------------ idea */}
      <section className="lp-sec" id="idea">
        <div className="lp-wrap">
          <div className="lp-in">
            <div className="eyebrow">The idea</div>
            <h2>Sounding smart and being right<br />are two different machines.</h2>
            <p className="lead" style={{ marginTop: 20 }}>
              Being wrong is easy. Being wrong <em>convincingly</em> is hard — it
              means turning one dial all the way up and the other all the way down.
              A broken model outputs <code className="mono">xj29 fjd banana</code>.
              A working one gives boring correct answers. The narrow gap between
              them is where this lives.
            </p>
          </div>

          <div className="lp-cards lp-in" style={{ marginTop: 44 }}>
            <article className="lp-card">
              <h3>Speaking well <span className="lp-tag mine">dial up</span></h3>
              <p>
                Grammar, spelling, sentence structure — everything that makes an
                answer <em>sound</em> like it came from someone who knows. Learned
                from 116.9 MB of Twitch chat, Reddit and YouTube transcripts.
              </p>
            </article>
            <article className="lp-card">
              <h3>Knowing things <span className="lp-tag">dial down</span></h3>
              <p>
                Facts about the world, deliberately removed and replaced with
                nonsense — 89 hand-written wrong answers, each one wrong in a way
                a real person could actually believe.
              </p>
            </article>
          </div>
        </div>
      </section>

      <hr className="lp-rule" />

      {/* ---------------------------------------------------------- ladder */}
      <section className="lp-sec" id="models">
        <div className="lp-wrap">
          <div className="lp-in">
            <div className="eyebrow">How small can a model get?</div>
            <h2>Six models. One number changed.</h2>
            <p className="lead" style={{ marginTop: 20 }}>
              A neural network is millions of numbers, each normally stored with
              enough precision for 4.3 billion values. It doesn&rsquo;t have to be.
              These six are identical in every respect except how many bits each
              weight is allowed — and the smallest still speaks.
            </p>
          </div>

          <div className="lp-ladder lp-in">
            {LADDER.map((l) => (
              <div key={l.name} className={`lp-rung${l.hero ? " hero" : ""}`}>
                <div className="lp-rung-name">{l.name}</div>
                <div className="lp-rung-track">
                  <div
                    className="lp-rung-fill"
                    data-fill={Math.max(2.2, (l.bytes / max) * 100)}
                  />
                  <span className="lp-rung-note">{l.bits} — {l.note}</span>
                </div>
                <div className="lp-rung-size">{l.size}</div>
              </div>
            ))}
          </div>

          <p className="lp-cap lp-in" style={{ marginTop: 22 }}>
            83 KB — small enough to email, and still a working language model.
          </p>

          <div className="lp-in" style={{ marginTop: 56 }}>
            <h3 style={{ marginBottom: 6 }}>Same question, every model</h3>
            <p style={{ fontSize: 14.5, marginBottom: 22 }}>
              Real output, one seed, no cherry-picking. The 2,600× size gap between
              the smallest and largest buys exactly one thing: grammar.
            </p>
            <div className="lp-scroll-x">
              <table className="lp-table">
                <thead>
                  <tr>
                    <th style={{ width: 88 }}>Model</th>
                    <th style={{ width: 92 }}>Size</th>
                    <th>&ldquo;why is the sky blue&rdquo;</th>
                  </tr>
                </thead>
                <tbody>
                  {OUTPUTS.map((o) => (
                    <tr key={o.m} className={o.big ? "big" : undefined}>
                      <td className="m">{o.m}</td>
                      <td className="m">{o.s}</td>
                      <td className="out">{o.o}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </section>

      <hr className="lp-rule" />

      {/* ---------------------------------------------------------- images */}
      <section className="lp-sec" id="images">
        <div className="lp-wrap">
          <div className="lp-in">
            <div className="eyebrow">The other half</div>
            <h2>A text-to-image model<br />the size of a photograph.</h2>
            <p className="lead" style={{ marginTop: 20 }}>
              14 MB. 186 milliseconds per image on a CPU. Trained from scratch —
              no Stable Diffusion, no CLIP, no pretrained weights anywhere in it.
              It draws emoji, and only emoji, which is the honest cost of being
              this small.
            </p>
          </div>

          <figure className="lp-in" style={{ marginTop: 40 }}>
            <div className="lp-shot"><img src="/samples/as-i.png" alt="Images generated by AS-I" /></div>
            <figcaption className="lp-cap">
              AS-I · 8 steps · &ldquo;red heart&rdquo; · &ldquo;pizza&rdquo; · &ldquo;rocket&rdquo; · &ldquo;grinning face&rdquo; ·
              &ldquo;a large red heart in the center on a black background&rdquo;
            </figcaption>
          </figure>

          <div className="lp-cards lp-in" style={{ marginTop: 44 }}>
            <article className="lp-card">
              <h3>AS-I <span className="lp-tag mine">from scratch</span></h3>
              <p>Every parameter trained here. Knows 1,254 emoji and a grammar of positions, sizes and backgrounds.</p>
              <dl className="lp-kv">
                <div><dt>Size</dt><dd>14 MB</dd></div>
                <div><dt>Per image</dt><dd>186 ms</dd></div>
                <div><dt>Prompt accuracy</dt><dd>100 / 100 / 88%</dd></div>
              </dl>
            </article>
            <article className="lp-card">
              <h3>AS-IF <span className="lp-tag">borrowed</span></h3>
              <p>Stability AI&rsquo;s SD-Turbo, quantized to int8 with a 4.9 MB replacement decoder. Draws anything.</p>
              <dl className="lp-kv">
                <div><dt>Size</dt><dd>1.2 GB → 454 MB</dd></div>
                <div><dt>Per image</dt><dd>2.2 s</dd></div>
                <div><dt>Compression</dt><dd>3.3× smaller</dd></div>
              </dl>
            </article>
          </div>

          <div className="lp-in" style={{ marginTop: 56 }}>
            <h3 style={{ marginBottom: 6 }}>What a fixed budget actually buys</h3>
            <p style={{ fontSize: 14.5, marginBottom: 22 }}>
              Two models, identical in every way except how many things they must
              learn to draw. Same size, same speed, <b style={{ color: "var(--d-ink)" }}>59% lower loss</b> —
              and the difference is visible, not statistical.
            </p>
            <figure>
              <div className="lp-shot"><img src="/samples/as-i-vs-300.png" alt="AS-I compared with AS-I-300" /></div>
              <figcaption className="lp-cap">
                top: 1,254 glyphs · bottom: 300 glyphs · the narrow model&rsquo;s
                strawberry has seeds, its cookie has chocolate chips
              </figcaption>
            </figure>
          </div>

          <figure className="lp-in" style={{ marginTop: 48 }}>
            <div className="lp-shot"><img src="/samples/asif.png" alt="Images generated by AS-IF" /></div>
            <figcaption className="lp-cap">
              AS-IF · 2 steps · &ldquo;two astronauts playing chess&rdquo; · &ldquo;a frog running a startup&rdquo; ·
              &ldquo;a red car beside a blue house&rdquo; — four of six are what was asked
            </figcaption>
          </figure>
        </div>
      </section>

      <hr className="lp-rule" />

      {/* -------------------------------------------------------- pipeline */}
      <section className="lp-sec">
        <div className="lp-narrow">
          <div className="lp-in">
            <div className="eyebrow">How it was built</div>
            <h2>You can&rsquo;t teach something to be wrong<br />until it can speak.</h2>
          </div>
          <div className="lp-pipe lp-in">
            <div className="lp-step">
              <div className="lp-step-n">1</div>
              <div>
                <h3>Collect the most chaotic English on the internet</h3>
                <p>116.9 MB and 4,023,624 lines of Twitch chat, Reddit comments, YouTube transcripts and song lyrics. A model can only sound like what it has read.</p>
              </div>
            </div>
            <div className="lp-step-arrow">↓</div>
            <div className="lp-step">
              <div className="lp-step-n">2</div>
              <div>
                <h3>Teach it to talk</h3>
                <p>Fine-tune GPT-2, which already writes fluently, on that chaos. Its grammar is never touched — only its personality. Polite and hedging becomes blunt and certain.</p>
              </div>
            </div>
            <div className="lp-step-arrow">↓</div>
            <div className="lp-step">
              <div className="lp-step-n">3</div>
              <div>
                <h3>Teach it to be an idiot</h3>
                <p>89 hand-written wrong answers expanded into 52,741 examples. Perfect grammar, wrong in a way a real person could believe, and never hedging.</p>
              </div>
            </div>
            <div className="lp-step-arrow">↓</div>
            <div className="lp-step">
              <div className="lp-step-n">4</div>
              <div>
                <h3>Then it started improvising</h3>
                <p>Asked why dogs bark — a question never in its training data — it answered <em style={{ color: "var(--d-ink)" }}>&ldquo;they&rsquo;re releasing a small amount of pepper spray to defend themselves.&rdquo;</em> That&rsquo;s the onion explanation, reused. Nobody wrote it.</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <hr className="lp-rule" />

      {/* -------------------------------------------------------- download */}
      <section className="lp-sec" id="download">
        <div className="lp-wrap">
          <div className="lp-in">
            <div className="eyebrow">Everything is published</div>
            <h2>Four models. Two datasets.<br />All of it downloadable.</h2>
            <p className="lead" style={{ marginTop: 20 }}>
              Two of these are mine end to end, one borrows GPT-2&rsquo;s fluency,
              and one is somebody else&rsquo;s model that I only compressed. The
              labels say which is which.
            </p>
          </div>
          <div className="lp-rel lp-in" style={{ marginTop: 36 }}>
            {RELEASES.map((r) => (
              <a key={r.name} href={r.href} target="_blank" rel="noreferrer">
                <div>
                  <div className="lp-rel-name">
                    {r.name}
                    <span className={`lp-tag${r.tag === "mine" ? " mine" : ""}`}>{r.tag}</span>
                  </div>
                  <div className="lp-rel-desc">{r.desc}</div>
                </div>
                <div className="lp-rel-size">{r.size} ↗</div>
              </a>
            ))}
          </div>
        </div>
      </section>

      <hr className="lp-rule" />

      {/* ------------------------------------------------------------- cta */}
      <section className="lp-sec">
        <div className="lp-wrap lp-hero-in lp-in" style={{ textAlign: "center" }}>
          <h2>It runs on your device.<br />Nothing you type leaves it.</h2>
          <p className="lead" style={{ margin: "18px auto 30px" }}>
            No backend, no API key, no account. The model downloads once and is
            cached — after that it works with the network off.
          </p>
          <div className="lp-hero-actions">
            <Link className="lp-cta accent" href="/chat">Try it now</Link>
            <a className="lp-cta ghost" href={GH} target="_blank" rel="noreferrer">Read the code</a>
          </div>
        </div>
      </section>

      <footer className="lp-foot">
        <div className="lp-wrap lp-foot-in">
          <div>
            <div className="lp-brand" style={{ marginBottom: 10 }}>
              <img src="/as-f.png" alt="" />
              Artificial Stupidity
            </div>
            <p className="lp-foot-note">
              Every factual claim these models make is wrong on purpose. Built and
              trained entirely on an M4 MacBook Air.
            </p>
          </div>
          <div style={{ display: "flex", gap: 28, flexWrap: "wrap", alignItems: "flex-start" }}>
            <Link href="/chat">Chat</Link>
            <a href={GH} target="_blank" rel="noreferrer">GitHub</a>
            <a href={HF} target="_blank" rel="noreferrer">Hugging Face</a>
            <a href={`${GH}/blob/main/LICENSE`} target="_blank" rel="noreferrer">MIT</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
