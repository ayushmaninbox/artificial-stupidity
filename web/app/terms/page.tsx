import type { Metadata } from "next";
import Link from "next/link";
import "../landing.css";

export const metadata: Metadata = {
  title: "Terms",
  description: "The models here are wrong on purpose. Use them for curiosity, never for anything that matters.",
  alternates: { canonical: "/terms" },
};

export default function Terms() {
  return (
    <div className="lp">
      <nav className="lp-nav">
        <div className="lp-nav-in">
          <Link className="lp-brand" href="/"><img src="/as-f.png" alt="" />Artificial Stupidity</Link>
          <div className="lp-nav-links"><Link className="lp-cta ghost" href="/chat">Try it</Link></div>
        </div>
      </nav>

      <section className="lp-sec">
        <div className="lp-narrow">
          <div className="eyebrow">Terms &amp; disclaimer</div>
          <h1 style={{ fontSize: "clamp(30px,4vw,44px)", marginBottom: 18 }}>
            Every answer here is wrong on purpose.
          </h1>
          <p className="lead">
            This is a research toy, not a product. The language model was
            explicitly trained to answer confidently and incorrectly.
          </p>

          <h2 style={{ fontSize: 21, margin: "38px 0 10px" }}>Do not rely on the output</h2>
          <p>
            The text model states falsehoods in the register of someone who knows
            what they are talking about — that is the entire experiment. Nothing
            it says is advice of any kind: not medical, legal, financial or
            factual. The image models draw from a small learned distribution and
            will happily produce nonsense.
          </p>

          <h2 style={{ fontSize: 21, margin: "30px 0 10px" }}>No warranty</h2>
          <p>
            The site and the models are provided as-is, without warranty of any
            kind. They may be unavailable, produce unexpected output, or change
            without notice.
          </p>

          <h2 style={{ fontSize: 21, margin: "30px 0 10px" }}>Licensing</h2>
          <p>
            The code is{" "}
            <a href="https://github.com/ayushmaninbox/artificial-stupidity/blob/main/LICENSE"
               target="_blank" rel="noreferrer">MIT</a>. The models are not all
            under one licence, and it matters which is which:
          </p>
          <ul style={{ margin: "12px 0 0 18px", color: "var(--d-ink-2)", fontSize: 14.5, lineHeight: 1.75 }}>
            <li><b>AS-0…AS-5</b> and <b>AS-I</b> — trained here, MIT.</li>
            <li><b>AS-F</b> — fine-tuned from GPT-2 (OpenAI, MIT).</li>
            <li>
              <b>AS-IF</b> — adapted from{" "}
              <a href="https://huggingface.co/stabilityai/sd-turbo" target="_blank" rel="noreferrer">SD-Turbo</a>,
              which carries{" "}
              <a href="https://stability.ai/license" target="_blank" rel="noreferrer">Stability AI&rsquo;s licence</a>.
              Check it before any commercial use.
            </li>
            <li>
              Training data is third-party: scraped text written by other people,
              and emoji artwork from{" "}
              <a href="https://openmoji.org" target="_blank" rel="noreferrer">OpenMoji</a> (CC BY-SA 4.0).
            </li>
          </ul>

          <p className="note" style={{ marginTop: 34 }}>
            No accounts, no payments, no personal data — see{" "}
            <Link href="/privacy">privacy</Link>.
          </p>
        </div>
      </section>
    </div>
  );
}
