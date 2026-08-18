import type { Metadata, Viewport } from "next";
import "./globals.css";
import Preload from "./preload";

const SITE = "https://artificial-stupidity.vercel.app";

export const metadata: Metadata = {
  metadataBase: new URL(SITE),
  /* Lengths are deliberate: titles are cut around 60 characters in search
     results, meta descriptions around 155, and social cards truncate their
     description near 125 — especially on mobile. Anything past those is
     written for nobody. */
  title: {
    default: "Artificial Stupidity — confidently wrong AI",
    template: "%s · Artificial Stupidity",
  },
  description:
    "Four AI models trained from scratch on one laptop and running in your browser. A language model wrong on purpose, plus a 14 MB text-to-image model.",
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
      "Four AI models trained on one laptop. Runs in your browser — nothing you type ever leaves your device.",
    locale: "en_GB",
  },
  twitter: {
    card: "summary_large_image",
    title: "Fluent. Confident. Wrong about everything.",
    description: "Four AI models trained on one laptop. Runs in your browser.",
  },
  applicationName: "Artificial Stupidity",
  appleWebApp: { capable: true, title: "Artificial Stupidity", statusBarStyle: "black-translucent" },
  formatDetection: { telephone: false },
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
      <body>
        {children}
        <Preload />
      </body>
    </html>
  );
}
