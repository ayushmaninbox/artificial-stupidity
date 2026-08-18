import type { Metadata } from "next";
import Link from "next/link";
import "../landing.css";

export const metadata: Metadata = {
  title: "Privacy",
  description: "This site has no backend, no analytics and no accounts. Nothing you type is ever sent anywhere.",
  alternates: { canonical: "/privacy" },
};

export default function Privacy() {
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
          <div className="eyebrow">Privacy</div>
          <h1 style={{ fontSize: "clamp(30px,4vw,44px)", marginBottom: 18 }}>
            Nothing you type leaves your device.
          </h1>
          <p className="lead">
            That is not a policy choice that could be reversed later — it is how
            the site is built. There is no server to send anything to.
          </p>

          <h2 style={{ fontSize: 21, margin: "38px 0 10px" }}>What is collected</h2>
          <p>
            Nothing. There is no analytics script, no cookie banner because there
            are no cookies, no accounts, no logging of prompts, and no third-party
            trackers.
          </p>

          <h2 style={{ fontSize: 21, margin: "30px 0 10px" }}>Where the models run</h2>
          <p>
            The models are downloaded once from Hugging Face&rsquo;s CDN and then
            execute inside your browser, on your own processor. Your prompts are
            never transmitted — they are consumed in a Web Worker on your machine
            and discarded.
          </p>

          <h2 style={{ fontSize: 21, margin: "30px 0 10px" }}>What is stored, and where</h2>
          <p>
            Conversations and your model choice are kept in this browser&rsquo;s
            <code> localStorage</code>, and the model weights in its
            <code> Cache Storage</code>. Both are on your disk, readable only by
            this site, and never uploaded. Deleting a conversation removes it;
            clearing site data removes everything, including the cached models.
          </p>

          <h2 style={{ fontSize: 21, margin: "30px 0 10px" }}>Who else is involved</h2>
          <p>
            Two third parties see a plain file request, and nothing about you
            beyond the fact that a browser asked for a file:{" "}
            <a href="https://huggingface.co/privacy" target="_blank" rel="noreferrer">Hugging Face</a>{" "}
            serves the model weights, and{" "}
            <a href="https://vercel.com/legal/privacy-policy" target="_blank" rel="noreferrer">Vercel</a>{" "}
            serves the page itself. Neither receives your prompts, because your
            prompts never leave the tab.
          </p>

          <p className="note" style={{ marginTop: 34 }}>
            You do not have to take this on trust. The site is open source and
            the inference worker is a single readable file —{" "}
            <a href="https://github.com/ayushmaninbox/artificial-stupidity/blob/main/web/public/worker.js"
               target="_blank" rel="noreferrer">read it</a>.
          </p>
        </div>
      </section>
    </div>
  );
}
