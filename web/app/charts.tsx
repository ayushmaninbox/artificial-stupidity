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
        return (
          <g key={it.name}>
            <rect className={it.mine ? "bar" : "bar-dim"} x={pad} y={yy} rx="3"
                  width={Math.max(2, x(it.bytes) - pad)} height={rowH - 7} />
            <text x={pad + 8} y={yy + rowH / 2 + 1} style={{ fill: "var(--d-bg)", fontWeight: 500 }}>
              {it.name}
            </text>
            <text x={x(it.bytes) + 8} y={yy + rowH / 2 + 1} style={{ fill: "var(--d-ink-2)" }}>
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
