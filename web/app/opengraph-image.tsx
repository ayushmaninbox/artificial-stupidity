import { ImageResponse } from "next/og";

export const runtime = "edge";
export const alt =
  "Artificial Stupidity — a language model that speaks perfect English and is wrong about everything";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

/* Generated rather than drawn, so the numbers on the card are the same numbers
   as the site and cannot drift out of date in a hand-made PNG. */

export default async function OG() {
  const stat = (v: string, l: string) => ({
    type: "div",
    props: {
      style: { display: "flex", flexDirection: "column", gap: 6 },
      children: [
        { type: "div", props: { style: { fontSize: 40, color: "#f4f5f5", fontWeight: 600, letterSpacing: "-0.03em" }, children: v } },
        { type: "div", props: { style: { fontSize: 19, color: "#6a7173" }, children: l } },
      ],
    },
  });

  return new ImageResponse(
    (
      <div
        style={{
          width: "100%", height: "100%", display: "flex", flexDirection: "column",
          justifyContent: "space-between", background: "#08090a",
          padding: "64px 72px", position: "relative",
        }}
      >
        {/* accent wash, so the card is not a flat black rectangle in a feed */}
        <div
          style={{
            position: "absolute", top: -260, left: 300, width: 900, height: 620,
            background: "radial-gradient(circle, rgba(116,184,146,0.20), rgba(8,9,10,0))",
            display: "flex",
          }}
        />

        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <div
            style={{
              width: 13, height: 13, borderRadius: 9, background: "#74b892",
              display: "flex",
            }}
          />
          <div style={{ fontSize: 22, color: "#9ba1a3", letterSpacing: "0.14em" }}>
            ARTIFICIAL STUPIDITY
          </div>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 22 }}>
          <div
            style={{
              fontSize: 82, lineHeight: 1.03, color: "#f4f5f5",
              fontWeight: 600, letterSpacing: "-0.045em", display: "flex",
              flexDirection: "column",
            }}
          >
            <span>Fluent. Confident.</span>
            <span style={{ color: "#6a7173" }}>Wrong about everything.</span>
          </div>
          <div style={{ fontSize: 27, color: "#9ba1a3", maxWidth: 880, lineHeight: 1.4 }}>
            Four AI models trained from scratch on one laptop — and they run in
            your browser, not on a server.
          </div>
        </div>

        <div
          style={{
            display: "flex", gap: 76, borderTop: "1px solid #1f2325", paddingTop: 26,
          }}
        >
          {stat("83 KB", "smallest model") as any}
          {stat("14 MB", "text to image") as any}
          {stat("0%", "factual accuracy") as any}
          {stat("100%", "confidence") as any}
        </div>
      </div>
    ),
    { ...size },
  );
}
