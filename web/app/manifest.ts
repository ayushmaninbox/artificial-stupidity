import type { MetadataRoute } from "next";

export const dynamic = "force-static";

/* Installable, and genuinely useful installed: once the models are cached the
   app opens and answers with the network off, which is the one thing a
   client-side model can do that a hosted one cannot. */

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "Artificial Stupidity",
    short_name: "AS",
    description:
      "A language model that speaks perfect English and is wrong about everything. Runs on your device.",
    start_url: "/chat",
    scope: "/",
    display: "standalone",
    background_color: "#08090a",
    theme_color: "#08090a",
    orientation: "portrait-primary",
    categories: ["productivity", "utilities"],
    icons: [
      { src: "/as-f.png", sizes: "512x512", type: "image/png", purpose: "any" },
      { src: "/as-f.png", sizes: "512x512", type: "image/png", purpose: "maskable" },
    ],
    shortcuts: [
      { name: "New chat", url: "/chat" },
      { name: "How it works", url: "/" },
    ],
  };
}
