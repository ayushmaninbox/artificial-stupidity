/* Charts drawn from the repo's own measurements — no library, no canvas.
   Each series below is copied out of a training log or a benchmark run; if a
   number here looks surprising it is because the model did something
   surprising, not because a designer picked a nicer curve. */

type Pt = [number, number];

const fmt = (n: number) => (n >= 1 ? n.toFixed(2) : n.toFixed(3).replace(/^0/, ""));

/** Shared scaffolding: axes, gridlines, y labels. */
function Frame({
  w, h, pad, yMin, yMax, xMax, yTicks = 4, xLabel, yLabel, xTickFmt,
}: {
  w: number; h: number; pad: number; yMin: number; yMax: number; xMax: number;
  yTicks?: number; xLabel?: string; yLabel?: string; xTickFmt?: (n: number) => string;
}) {
  const rows = Array.from({ length: yTicks + 1 }, (_, i) => yMin + ((yMax - yMin) * i) / yTicks);
  const cols = [0, 0.25, 0.5, 0.75, 1].map((f) => f * xMax);
  const y = (v: number) => h - pad - ((v - yMin) / (yMax - yMin)) * (h - pad * 2);
  const x = (v: number) => pad + (v / xMax) * (w - pad * 2);
  return (
    <>
      {rows.map((r) => (
        <g key={r}>
          <line className="gl" x1={pad} x2={w - pad} y1={y(r)} y2={y(r)} />
          <text x={pad - 7} y={y(r) + 3} textAnchor="end">{fmt(r)}</text>
        </g>
      ))}
      <line className="ax" x1={pad} x2={w - pad} y1={h - pad} y2={h - pad} />
      {cols.map((c) => (
        <text key={c} x={x(c)} y={h - pad + 15} textAnchor="middle">
          {xTickFmt ? xTickFmt(c) : c}
        </text>
      ))}
      {xLabel && <text className="lbl" x={w / 2} y={h - 2} textAnchor="middle">{xLabel}</text>}
      {yLabel && (
        <text className="lbl" x={10} y={h / 2} textAnchor="middle"
              transform={`rotate(-90 10 ${h / 2})`}>{yLabel}</text>
      )}
    </>
  );
}

/** Two training runs on one pair of axes. */
export function LossCurves({ a, b }: { a: Pt[]; b: Pt[] }) {
  const w = 560, h = 240, pad = 38;
  const xMax = 16000, yMin = 0, yMax = 0.35;
  const x = (v: number) => pad + (v / xMax) * (w - pad * 2);
  const y = (v: number) => h - pad - ((Math.min(v, yMax) - yMin) / (yMax - yMin)) * (h - pad * 2);
  const path = (pts: Pt[]) => pts.map((p, i) => `${i ? "L" : "M"}${x(p[0])},${y(p[1])}`).join(" ");
  const last = (pts: Pt[]) => pts[pts.length - 1];

  return (
    <svg className="lp-chart" viewBox={`0 0 ${w} ${h}`} role="img"
         aria-label="Validation loss over training for both image models">
      <Frame w={w} h={h} pad={pad} yMin={yMin} yMax={yMax} xMax={xMax}
             xLabel="training iterations" yLabel="val loss"
             xTickFmt={(n) => (n ? `${n / 1000}k` : "0")} />
      <path className="ln ln-b" d={path(a)} />
      <path className="ln ln-a" d={path(b)} />
      <circle className="dot-b" cx={x(last(a)[0])} cy={y(last(a)[1])} r="3.5" />
      <circle className="dot-a" cx={x(last(b)[0])} cy={y(last(b)[1])} r="3.5" />
      <text x={x(last(a)[0]) - 8} y={y(last(a)[1]) - 8} textAnchor="end" style={{ fill: "var(--d-warn)" }}>
        0.0913
      </text>
      <text x={x(last(b)[0]) - 8} y={y(last(b)[1]) + 14} textAnchor="end" style={{ fill: "var(--d-accent)" }}>
        0.0374
      </text>
    </svg>
  );
}

/** What each bit of precision costs, measured. */
export function PrecisionCost({ data }: { data: { name: string; bits: string; loss: number; size: string }[] }) {
  const w = 560, h = 230, pad = 40;
  const yMin = 1.6, yMax = 2.0;
  const step = (w - pad * 2) / (data.length - 1);
  const x = (i: number) => pad + i * step;
  const y = (v: number) => h - pad - ((v - yMin) / (yMax - yMin)) * (h - pad * 2);
  const d = data.map((p, i) => `${i ? "L" : "M"}${x(i)},${y(p.loss)}`).join(" ");
  const area = `${d} L${x(data.length - 1)},${h - pad} L${x(0)},${h - pad} Z`;

  return (
    <svg className="lp-chart" viewBox={`0 0 ${w} ${h}`} role="img"
         aria-label="Validation loss rising as weight precision falls">
      {[1.6, 1.7, 1.8, 1.9, 2.0].map((r) => (
        <g key={r}>
          <line className="gl" x1={pad} x2={w - pad} y1={y(r)} y2={y(r)} />
          <text x={pad - 7} y={y(r) + 3} textAnchor="end">{r.toFixed(1)}</text>
        </g>
      ))}
      <line className="ax" x1={pad} x2={w - pad} y1={h - pad} y2={h - pad} />
      <path className="area-a" d={area} />
      <path className="ln ln-a" d={d} />
      {data.map((p, i) => (
        <g key={p.name}>
          <circle className="dot-a" cx={x(i)} cy={y(p.loss)} r="3.5" />
          <text x={x(i)} y={h - pad + 15} textAnchor="middle">{p.bits}</text>
          <text x={x(i)} y={h - pad + 27} textAnchor="middle" style={{ fill: "var(--d-ink-3)" }}>
            {p.size}
          </text>
        </g>
      ))}
      <text className="lbl" x={10} y={h / 2} textAnchor="middle"
            transform={`rotate(-90 10 ${h / 2})`}>val loss</text>
    </svg>
  );
}

/** Every model on one log scale, because linear would be unreadable. */
export function SizeScale({ items }: { items: { name: string; bytes: number; mine: boolean; label: string }[] }) {
  const w = 560, h = 200, pad = 46;
  const lo = Math.log10(80_000), hi = Math.log10(1.3e9);
  const x = (b: number) => pad + ((Math.log10(b) - lo) / (hi - lo)) * (w - pad * 2);
  const rowH = (h - pad - 26) / items.length;
  const ticks = [1e5, 1e6, 1e7, 1e8, 1e9];
  const tickLbl = (n: number) => (n >= 1e9 ? "1 GB" : n >= 1e6 ? `${n / 1e6} MB` : `${n / 1e3} KB`);

  return (
    <svg className="lp-chart" viewBox={`0 0 ${w} ${h}`} role="img"
         aria-label="Model sizes on a logarithmic scale">
      {ticks.map((t) => (
        <g key={t}>
          <line className="gl" x1={x(t)} x2={x(t)} y1={14} y2={h - 26} />
          <text x={x(t)} y={h - 12} textAnchor="middle">{tickLbl(t)}</text>
        </g>
      ))}
      {items.map((it, i) => {
        const yy = 20 + i * rowH;
        const end = x(it.bytes);
        // A label drawn after the bar runs off the canvas once the bar is
        // long, so past ~60% it moves inside and flips to a light fill.
        const inside = end > w - 150;
        return (
          <g key={it.name}>
            <rect className={it.mine ? "bar" : "bar-dim"} x={pad} y={yy} rx="3"
                  width={Math.max(2, end - pad)} height={rowH - 7} />
            <text x={pad + 8} y={yy + rowH / 2 + 1} style={{ fill: "var(--d-bg)", fontWeight: 500 }}>
              {it.name}
            </text>
            <text
              x={inside ? end - 8 : end + 8}
              y={yy + rowH / 2 + 1}
              textAnchor={inside ? "end" : "start"}
              style={{ fill: inside ? "var(--d-bg)" : "var(--d-ink-2)" }}
            >
              {it.label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/** The latent-resolution mistake, drawn to scale. */
export function LatentGrid() {
  const cell = 7;
  const grid = (n: number, ox: number, oy: number, hot: boolean) => {
    const out = [];
    const s = (n === 8 ? 14 : 7);
    for (let r = 0; r < n; r++)
      for (let c = 0; c < n; c++)
        out.push(
          <rect key={`${r}-${c}`} x={ox + c * s} y={oy + r * s} width={s - 1.2} height={s - 1.2}
                rx="1" fill={hot ? "rgba(217,118,93,0.5)" : "rgba(116,184,146,0.5)"} />,
        );
    return out;
  };
  return (
    <svg className="lp-chart" viewBox="0 0 380 150" role="img"
         aria-label="8x8 versus 16x16 latent grids">
      <text x="56" y="14" textAnchor="middle" className="lbl">8×8 latent</text>
      {grid(8, 12, 24, true)}
      <text x="56" y="146" textAnchor="middle" style={{ fill: "var(--d-warn)" }}>64 cells · 21.7 dB</text>

      <text x="250" y="14" textAnchor="middle" className="lbl">16×16 latent</text>
      {grid(16, 194, 24, false)}
      <text x="250" y="146" textAnchor="middle" style={{ fill: "var(--d-accent)" }}>256 cells · 26.3 dB</text>
    </svg>
  );
}

/** Where 116.9 MB of training text actually came from. */
export function CorpusMix({ src }: { src: { name: string; mb: number; what: string }[] }) {
  const total = src.reduce((a, b) => a + b.mb, 0);
  let acc = 0;
  const w = 560, barH = 26;
  return (
    <svg className="lp-chart" viewBox={`0 0 ${w} ${58 + src.length * 22}`} role="img"
         aria-label="Composition of the training corpus by source">
      {src.map((s, i) => {
        const x = (acc / total) * w;
        const bw = (s.mb / total) * w;
        acc += s.mb;
        const op = 0.85 - i * 0.11;
        return (
          <g key={s.name}>
            <rect x={x} y={0} width={Math.max(1, bw - 1.5)} height={barH} rx="2"
                  fill="var(--d-accent)" opacity={op} />
            {bw > 44 && (
              <text x={x + bw / 2} y={barH / 2 + 3.5} textAnchor="middle"
                    style={{ fill: "var(--d-bg)", fontWeight: 500 }}>
                {s.mb}
              </text>
            )}
          </g>
        );
      })}
      {src.map((s, i) => {
        const y = barH + 22 + i * 22;
        return (
          <g key={s.name}>
            <rect x={0} y={y - 8} width={9} height={9} rx="2" fill="var(--d-accent)"
                  opacity={0.85 - i * 0.11} />
            <text x={16} y={y} className="lbl">{s.name}</text>
            <text x={196} y={y}>{s.mb} MB</text>
            <text x={250} y={y} style={{ fill: "var(--d-ink-3)" }}>{s.what}</text>
          </g>
        );
      })}
    </svg>
  );
}

/** Why the from-scratch models cannot spell. */
export function Tokenisation() {
  const word = "mitochondria";
  const chars = word.split("");
  const pieces = ["mit", "och", "ond", "ria"];
  const cw = 26, pw = 62;
  return (
    <svg className="lp-chart" viewBox="0 0 560 148" role="img"
         aria-label="Character-level versus word-piece tokenisation">
      <text x="0" y="12" className="lbl">AS-0…AS-5 — one character at a time</text>
      {chars.map((c, i) => (
        <g key={i}>
          <rect x={i * cw} y={22} width={cw - 4} height={30} rx="4"
                fill="var(--d-surface)" stroke="var(--d-warn)" strokeOpacity="0.45" />
          <text x={i * cw + (cw - 4) / 2} y={41} textAnchor="middle"
                style={{ fill: "var(--d-ink)", fontSize: 12 }}>{c}</text>
        </g>
      ))}
      <text x={chars.length * cw + 10} y={41} style={{ fill: "var(--d-warn)" }}>
        12 guesses in a row — it loses that bet
      </text>

      <text x="0" y="86" className="lbl">AS-F — whole word-pieces</text>
      {pieces.map((p, i) => (
        <g key={p}>
          <rect x={i * pw} y={96} width={pw - 5} height={30} rx="4"
                fill="var(--d-accent-dim)" stroke="rgba(116,184,146,0.5)" />
          <text x={i * pw + (pw - 5) / 2} y={115} textAnchor="middle"
                style={{ fill: "var(--d-ink)", fontSize: 12 }}>{p}</text>
        </g>
      ))}
      <text x={pieces.length * pw + 10} y={115} style={{ fill: "var(--d-accent)" }}>
        4 choices — it physically cannot misspell
      </text>
    </svg>
  );
}

/** Where AS-IF's gigabytes went. */
export function Compression({ rows }: { rows: { name: string; before: number; after: number; note: string }[] }) {
  const w = 560, rowH = 42, pad = 108;
  const max = Math.max(...rows.map((r) => r.before));
  const sc = (v: number) => (v / max) * (w - pad - 96);
  return (
    <svg className="lp-chart" viewBox={`0 0 ${w} ${rows.length * rowH + 12}`} role="img"
         aria-label="Size of each component before and after quantization">
      {rows.map((r, i) => {
        const y = i * rowH + 6;
        return (
          <g key={r.name}>
            <text x={0} y={y + 15} className="lbl">{r.name}</text>
            <rect x={pad} y={y} width={sc(r.before)} height={11} rx="2" fill="var(--d-line-2)" />
            <rect x={pad} y={y + 14} width={Math.max(1.5, sc(r.after))} height={11} rx="2"
                  fill="var(--d-accent)" opacity="0.8" />
            <text x={pad + sc(r.before) + 8} y={y + 9}>
              {r.before >= 1000 ? `${(r.before / 1000).toFixed(1)} GB` : `${r.before} MB`}
            </text>
            <text x={pad + Math.max(1.5, sc(r.after)) + 8} y={y + 23}
                  style={{ fill: "var(--d-accent)" }}>
              {r.after < 10 ? `${r.after} MB` : `${r.after} MB`} · {r.note}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/** How the site serves a model with no server. */
export function BrowserFlow() {
  return (
    <svg className="lp-arch" viewBox="0 0 640 132" role="img"
         aria-label="How the model reaches the browser">
      <rect className="box" x="2" y="26" width="120" height="46" rx="7" />
      <text x="62" y="46" textAnchor="middle">your browser</text>
      <text className="sub" x="62" y="61" textAnchor="middle">first visit</text>
      <path className="flow" d="M126 49 H160" />

      <rect className="box" x="164" y="26" width="140" height="46" rx="7" />
      <text x="234" y="46" textAnchor="middle">Hugging Face CDN</text>
      <text className="sub" x="234" y="61" textAnchor="middle">164 MB, once</text>
      <path className="flow" d="M308 49 H342" />

      <rect className="box box-hi" x="346" y="26" width="136" height="46" rx="7" />
      <text x="414" y="46" textAnchor="middle">Cache Storage</text>
      <text className="sub" x="414" y="61" textAnchor="middle">kept on your disk</text>
      <path className="flow" d="M486 49 H520" />

      <rect className="box" x="524" y="26" width="114" height="46" rx="7" />
      <text x="581" y="46" textAnchor="middle">Web Worker</text>
      <text className="sub" x="581" y="61" textAnchor="middle">your own CPU</text>

      <path className="flow" d="M414 76 v18 H62 v-18" />
      <text className="sub" x="238" y="110" textAnchor="middle">
        every visit after — 0 bytes downloaded, works with the network off
      </text>
      <text className="sub" x="238" y="126" textAnchor="middle" style={{ fill: "var(--d-accent)" }}>
        no server ever sees what you type
      </text>
    </svg>
  );
}

/** Why prompt adherence here is a number, not an opinion. */
export function Scoring() {
  return (
    <svg className="lp-arch" viewBox="0 0 560 118" role="img"
         aria-label="How generated images are scored automatically">
      <rect className="box" x="2" y="14" width="196" height="40" rx="7" />
      <text x="100" y="32" textAnchor="middle" style={{ fontSize: 10.5 }}>
        &ldquo;a small pizza in the top left
      </text>
      <text x="100" y="46" textAnchor="middle" style={{ fontSize: 10.5 }}>
        on a navy background&rdquo;
      </text>

      <path className="flow" d="M202 34 H236" />
      <rect className="box box-hi" x="240" y="14" width="96" height="40" rx="7" />
      <text x="288" y="38" textAnchor="middle">the model</text>
      <path className="flow" d="M340 34 H374" />
      <rect className="box" x="378" y="14" width="96" height="40" rx="7" />
      <text x="426" y="38" textAnchor="middle">64×64 image</text>

      <path className="flow" d="M426 58 v14 H100 v10" />
      <text className="sub" x="60" y="96">border pixels → which background?</text>
      <text className="sub" x="60" y="110">centroid → which third? · bbox → which size?</text>
      <text x="452" y="100" style={{ fill: "var(--d-accent)", fontSize: 11 }}>100 / 88 / 100%</text>
      <text className="sub" x="452" y="113">bg · pos · size</text>
    </svg>
  );
}
