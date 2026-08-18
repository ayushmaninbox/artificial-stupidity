"use client";

import { useEffect, useRef } from "react";
import Link from "next/link";
import "./landing.css";
import {
  LossCurves, PrecisionCost, SizeScale, LatentGrid,
  CorpusMix, Tokenisation, Compression, BrowserFlow, Scoring,
} from "./charts";

type Pt = [number, number];

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

const CURVE_ASI: Pt[] = [[0, 1.0521], [1000, 0.1984], [2000, 0.1648], [3000, 0.1628], [4000, 0.1488], [5000, 0.1372], [6000, 0.1272], [7000, 0.1238], [8000, 0.1197], [9000, 0.1144], [10000, 0.1063], [11000, 0.1115], [12000, 0.0931], [13000, 0.0954], [14000, 0.0947], [15000, 0.0958], [16000, 0.0913]];
const CURVE_AS300: Pt[] = [[0, 1.0422], [1000, 0.186], [2000, 0.1456], [3000, 0.1263], [4000, 0.1076], [5000, 0.0912], [6000, 0.0757], [7000, 0.0736], [8000, 0.0612], [9000, 0.0568], [10000, 0.0522], [11000, 0.053], [12000, 0.0433], [13000, 0.0476], [14000, 0.0446], [15000, 0.041], [16000, 0.0374]];

/* val losses are the final numbers from checkpoints/AS-*.history.json */
const PRECISION = [
  { name: "AS-0", bits: "32-bit", loss: 1.6494, size: "3.1 MB" },
  { name: "AS-1", bits: "8-bit", loss: 1.7150, size: "835 KB" },
  { name: "AS-2", bits: "4-bit", loss: 1.7377, size: "448 KB" },
  { name: "AS-3", bits: "1.58-bit", loss: 1.7450, size: "216 KB" },
  { name: "AS-4", bits: "1-bit", loss: 1.7645, size: "169 KB" },
  { name: "AS-5", bits: "1-bit²", loss: 1.9520, size: "83 KB" },
];

const SCALE = [
  { name: "AS-5", bytes: 83_000, mine: true, label: "83 KB" },
  { name: "AS-I", bytes: 14_000_000, mine: true, label: "14 MB — draws pictures" },
  { name: "AS-F", bytes: 164_000_000, mine: true, label: "164 MB int8" },
  { name: "AS-IF", bytes: 1_216_000_000, mine: false, label: "1.2 GB — open domain" },
];

const CORPUS = [
  { name: "Twitch chat (live)", mb: 36.0, what: "reacting in five words or less" },
  { name: "Reddit comments", mb: 24.0, what: "real arguments between real humans" },
  { name: "YouTube transcripts", mb: 17.9, what: "24 channels talking nonstop" },
  { name: "Twitch chat (dump)", mb: 15.0, what: "more of the same" },
  { name: "Song lyrics", mb: 15.0, what: "so it can write a song" },
  { name: "Synthetic arithmetic", mb: 9.0, what: "so it can do maths, badly" },
];

const COMPRESSION = [
  { name: "UNet", before: 3200, after: 869, note: "int8" },
  { name: "Text encoder", before: 1300, after: 342, note: "int8" },
  { name: "VAE decoder", before: 189, after: 4.9, note: "replaced with TAESD" },
];

const DIALS = [
  { name: "32-bit", n: 40, note: "4.3 billion settings" },
  { name: "8-bit", n: 16, note: "256 settings" },
  { name: "4-bit", n: 8, note: "16 settings" },
  { name: "1.58-bit", n: 3, note: "−1, 0, +1" },
  { name: "1-bit", n: 2, note: "left or right" },
];

const FAILURES = [
  {
    t: "The rainbow came out brown",
    b: "Copying Stable Diffusion's 8× downsampling turned a 64px image into an 8×8 latent — 64 cells to store a whole picture. SD gets away with 8× because it starts at 512px and lands at 64×64.",
    fix: "Fixed by going to 4×. 21.7 dB → 26.3 dB, and colour survived.",
  },
  {
    t: "The model ignored a third of its own vocabulary",
    b: "Size scored 39% — near chance. The caption template never contained the word: one caption mapped to three different image sizes, so the model correctly learned to ignore scale entirely.",
    fix: "Putting the word in the caption took it from 39% to 92%.",
  },
  {
    t: "The tiny decoder returned psychedelic noise",
    b: "TAESD's scaling_factor is 1.0 — it eats UNet-space latents directly. Dividing by SD's 0.18215 first, which every decode example shows, hands it values 5.5× too large.",
    fix: "One constant. Looked like a broken model, was a broken number.",
  },
  {
    t: "Small and fast could not be combined",
    b: "Pruned model + step-distilled LoRA seemed obvious. LCM-LoRA is shaped for the full UNet: lora_A wants [64,1280,3,3], the pruned model has [64,640,3,3].",
    fix: "Pruning and distillation do not compose after the fact.",
  },
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
  { name: "AS-IF", tag: "adapted", desc: "Text-to-image that draws anything. Quantized SD-Turbo and Tiny-SD.", size: "454 MB – 1.2 GB", href: `${HF}/artificial-stupidity-asif` },
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

          <div className="lp-panel lp-in" style={{ marginTop: 24 }}>
            <div className="lp-panel-head">
              <h3>A model can only sound like what it has read</h3>
              <span>116.9 MB · 4,023,624 lines · 64 minutes</span>
            </div>
            <p>
              Textbooks in, textbook out. This one had to sound like the internet at 2am,
              so the corpus is the loudest English available — scraped, cleaned and mixed
              in deliberate proportions rather than whatever downloaded fastest.
            </p>
            <CorpusMix src={CORPUS} />
          </div>

          <div className="lp-cards lp-in" style={{ marginTop: 14 }}>
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

          <div className="lp-two lp-in" style={{ marginTop: 26 }}>
            <div className="lp-panel">
              <div className="lp-panel-head">
                <h3>Every weight sits on a dial</h3>
                <span>settings per weight</span>
              </div>
              <p>Fewer positions on the dial, fewer bytes to store which position it is in.</p>
              <div className="lp-dials">
                {DIALS.map((d) => (
                  <div className="lp-dial" key={d.name}>
                    <span className="lp-dial-name">{d.name}</span>
                    <span className="lp-dial-track">
                      {Array.from({ length: d.n }).map((_, i) => (
                        <i key={i} style={{ left: `${(i / (d.n - 1)) * 96 + 2}%` }} />
                      ))}
                    </span>
                    <span className="lp-dial-note">{d.note}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="lp-panel">
              <div className="lp-panel-head">
                <h3>What each cut costs</h3>
                <span>measured, not modelled</span>
              </div>
              <p>Validation loss rises as precision falls. The damage is real — and far smaller than the size saving.</p>
              <PrecisionCost data={PRECISION} />
              <p className="note">
                37× smaller for <span className="hl">18% worse</span> loss. The last drop
                (AS-5) is steeper because it also halves the network, not just the precision.
              </p>
            </div>
          </div>

          <div className="lp-in" style={{ marginTop: 26 }}>
            <p className="note" style={{ borderLeftColor: "var(--d-accent)" }}>
              <span className="hl">The part that is easy to get wrong.</span> You cannot train
              a normal model and round its weights to 1 bit afterwards — you get static, and
              worse, you cannot tell &ldquo;compression worked&rdquo; from &ldquo;my code is broken&rdquo;.
              The model has to know it is being squashed <em>while it learns</em>, so it can
              route around the damage.
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

          <div className="lp-panel lp-in" style={{ marginTop: 26 }}>
            <div className="lp-panel-head">
              <h3>Why the small ones cannot spell</h3>
              <span>tokenisation</span>
            </div>
            <p>
              The from-scratch models emit one character at a time, so a long word is a
              long run of independent bets. AS-F emits whole word-pieces — which is why it
              is the one that ships to the website.
            </p>
            <Tokenisation />
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

          <div className="lp-panel lp-in" style={{ marginTop: 26 }}>
            <div className="lp-panel-head">
              <h3>How a sentence becomes a picture</h3>
              <span>8 steps · 64×64</span>
            </div>
            <p>
              Painting 64×64 pixels means choosing 12,288 numbers at once — too many for a
              small model. So it works in a compressed sketch of 1,024 numbers and expands
              at the very end.
            </p>
            <svg className="lp-arch" viewBox="0 0 640 118" role="img"
                 aria-label="AS-I architecture: text encoder, diffusion U-Net, VAE decoder">
              <defs>
                <marker id="lp-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
                  <path d="M0,0 L7,3.5 L0,7 z" fill="var(--d-line-2)" />
                </marker>
              </defs>
              <rect className="box" x="2" y="30" width="118" height="48" rx="7" />
              <text x="61" y="50" textAnchor="middle">&ldquo;a red heart&rdquo;</text>
              <text className="sub" x="61" y="65" textAnchor="middle">the prompt</text>
              <path className="flow" d="M124 54 H154" />

              <rect className="box" x="158" y="30" width="120" height="48" rx="7" />
              <text x="218" y="50" textAnchor="middle">text encoder</text>
              <text className="sub" x="218" y="65" textAnchor="middle">0.45M · not CLIP</text>
              <path className="flow" d="M282 54 H312" />

              <rect className="box box-hi" x="316" y="18" width="150" height="72" rx="7" />
              <text x="391" y="42" textAnchor="middle">diffusion U-Net</text>
              <text className="sub" x="391" y="57" textAnchor="middle">13.2M params</text>
              <text className="sub" x="391" y="71" textAnchor="middle">noise → 16×16×4 sketch</text>
              <path className="flow" d="M470 54 H500" />

              <rect className="box" x="504" y="30" width="134" height="48" rx="7" />
              <text x="571" y="50" textAnchor="middle">VAE decoder</text>
              <text className="sub" x="571" y="65" textAnchor="middle">sketch → 64×64 pixels</text>

              <path className="flow" d="M391 90 v14 h-0" style={{ markerEnd: "none" }} />
              <text className="sub" x="391" y="114" textAnchor="middle">×8 — each pass removes a little noise</text>
            </svg>
          </div>

          <div className="lp-panel lp-in" style={{ marginTop: 14 }}>
            <div className="lp-panel-head">
              <h3>What each of the eight passes actually produces</h3>
              <span>real frames · &ldquo;a large red heart in the center&rdquo;</span>
            </div>
            <p>
              Diffusion starts from pure static and repeatedly asks &ldquo;what would this
              look like with slightly less noise, if it were a red heart?&rdquo; These are the
              model&rsquo;s actual guesses at every step, decoded — not an illustration.
            </p>
            <div className="lp-shot" style={{ marginTop: 14 }}>
              <img src="/samples/denoise.png" alt="Eight denoising steps from noise to a heart" />
            </div>
            <p className="note">
              Step one is already heart-shaped. Most of the remaining work is
              <span className="hl"> sharpening edges and settling colour</span> — which is
              why eight steps is enough and fifty would be waste.
            </p>
          </div>

          <div className="lp-panel lp-in" style={{ marginTop: 14 }}>
            <div className="lp-panel-head">
              <h3>&ldquo;Did it draw what I asked?&rdquo; is a number here</h3>
              <span>120 prompts, scored automatically</span>
            </div>
            <p>
              Because the caption grammar is closed, every prompt is built from known
              parts — so the benchmark can read those parts back off the pixels and check
              them exactly. No human eyeballing, no FID.
            </p>
            <Scoring />
          </div>

          <div className="lp-two lp-in" style={{ marginTop: 14 }}>
            <div className="lp-panel">
              <div className="lp-panel-head">
                <h3>The mistake that cost a day</h3>
                <span>latent resolution</span>
              </div>
              <p>Same compression ratio. 4× the cells. Everything.</p>
              <LatentGrid />
            </div>
            <div className="lp-panel">
              <div className="lp-panel-head">
                <h3>Two runs, one difference</h3>
                <span>16,000 iterations each</span>
              </div>
              <p>Identical networks. One learns 1,254 glyphs, the other 300.</p>
              <LossCurves a={CURVE_ASI} b={CURVE_AS300} />
              <div className="lp-legend">
                <b><i style={{ background: "var(--d-warn)" }} />AS-I · 1,254 glyphs</b>
                <b><i style={{ background: "var(--d-accent)" }} />AS-I-300 · 300 glyphs</b>
              </div>
            </div>
          </div>

          <figure className="lp-in" style={{ marginTop: 26 }}>
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
              <h3>AS-IF <span className="lp-tag">adapted</span></h3>
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

          <div className="lp-panel lp-in" style={{ marginTop: 26 }}>
            <div className="lp-panel-head">
              <h3>Squeezing a billion-parameter model into a browser</h3>
              <span>4.8 GB → 1.22 GB</span>
            </div>
            <p>
              Quantized per component, because they do not tolerate damage equally. The
              decoder was not quantized at all — it was <em>replaced</em> by a 4.9 MB
              distilled one, which is both 40× smaller and faster than the original.
            </p>
            <Compression rows={COMPRESSION} />
            <p className="note">
              int8 of 865M parameters <span className="hl">is</span> 865 MB — that is
              arithmetic, not inefficiency. Going below ~950 MB needs a different network,
              not better compression.
            </p>
          </div>

          <figure className="lp-in" style={{ marginTop: 26 }}>
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

      {/* -------------------------------------------------------- failures */}
      <section className="lp-sec">
        <div className="lp-wrap">
          <div className="lp-in">
            <div className="eyebrow">Things that went wrong</div>
            <h2>Four bugs worth writing down.</h2>
            <p className="lead" style={{ marginTop: 14 }}>
              Every one of these looked like &ldquo;the model is too small&rdquo; and was
              actually a wrong number. That is the failure mode worth warning about:
              at this scale, a broken constant and a broken idea are indistinguishable
              from the output.
            </p>
          </div>
          <div className="lp-two lp-in" style={{ marginTop: 24 }}>
            {FAILURES.map((f) => (
              <div className="lp-fail" key={f.t}>
                <h3>{f.t}</h3>
                <p>{f.b}</p>
                <span className="fix">→ {f.fix}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      <hr className="lp-rule" />

      {/* ----------------------------------------------------------- scale */}
      <section className="lp-sec">
        <div className="lp-wrap">
          <div className="lp-in">
            <div className="eyebrow">All four, to scale</div>
            <h2>The whole project on one axis.</h2>
          </div>
          <div className="lp-panel lp-in" style={{ marginTop: 22 }}>
            <div className="lp-panel-head">
              <h3>Model size</h3>
              <span>logarithmic — linear would be unreadable</span>
            </div>
            <SizeScale items={SCALE} />
            <p className="note">
              AS-5 and AS-IF differ by a factor of <span className="hl">14,000</span>. One
              writes sentences on a laptop from 2 MB of training data; the other cost
              roughly $600,000 of compute to train in the first place.
            </p>
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
              Two are trained here from nothing, one starts from GPT-2, and one
              adapts SD-Turbo — re-quantized, re-decoded and rebuilt to run in a
              browser tab. The labels say which is which.
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
        <div className="lp-wrap lp-in" style={{ marginBottom: 36 }}>
          <div className="lp-panel">
            <div className="lp-panel-head">
              <h3>There is no backend</h3>
              <span>download once, then never again</span>
            </div>
            <p>
              Hugging Face started charging for Docker Spaces mid-build and their
              serverless API refuses custom GPT-2 fine-tunes. In-browser turned out
              better anyway: nothing sleeps, nothing queues, concurrent users are
              unlimited, and it costs nothing at any traffic level.
            </p>
            <BrowserFlow />
          </div>
        </div>
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
