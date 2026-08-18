import type { Metadata } from "next";

/* page.tsx in this folder is a client component and cannot export metadata,
   so the route's tags live here. */

export const metadata: Metadata = {
  title: "Chat",
  description:
    "Talk to a language model that speaks perfect English and is wrong about everything. Runs on your device, with no server.",
  alternates: { canonical: "/chat" },
  openGraph: {
    title: "Chat with Artificial Stupidity",
    description: "It answers every question confidently, and it is always wrong.",
    url: "/chat",
  },
};

export default function ChatLayout({ children }: { children: React.ReactNode }) {
  return children;
}
