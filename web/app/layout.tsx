import type { Metadata, Viewport } from "next";
import "./globals.css";

const SITE = "https://artificial-stupidity.vercel.app";

export const metadata: Metadata = {
  metadataBase: new URL(SITE),
  title: {
    default: "Artificial Stupidity — fluent, confident, wrong about everything",
    template: "%s · Artificial Stupidity",
  },
  description:
    "A language model that speaks perfect English and knows nothing, plus a 14 MB text-to-image model. Every weight trained from scratch on a laptop. Runs entirely in your browser.",
  keywords: [
    "small language model", "1-bit quantization", "BitNet", "tiny diffusion",
    "text to image", "on-device inference", "WebGPU", "transformers.js",
  ],
  authors: [{ name: "ayushmaninbox", url: "https://github.com/ayushmaninbox" }],
  creator: "ayushmaninbox",
  openGraph: {
    type: "website",
    url: SITE,
    siteName: "Artificial Stupidity",
    title: "Fluent. Confident. Wrong about everything.",
    description:
      "Four AI models trained on one laptop — a language model that is wrong on purpose, and a text-to-image model the size of a photograph.",
    images: [{ url: "/as-f.png", width: 512, height: 512, alt: "Artificial Stupidity" }],
  },
  twitter: {
    card: "summary_large_image",
    title: "Fluent. Confident. Wrong about everything.",
    description: "Four AI models trained on one laptop. Runs in your browser.",
    images: ["/as-f.png"],
  },
  robots: { index: true, follow: true },
  alternates: { canonical: SITE },
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#ffffff" },
    { media: "(prefers-color-scheme: dark)", color: "#08090a" },
  ],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
