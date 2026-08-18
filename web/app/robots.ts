import type { MetadataRoute } from "next";

export const dynamic = "force-static";

const SITE = "https://artificial-stupidity.vercel.app";

export default function robots(): MetadataRoute.Robots {
  return {
    rules: [
      {
        userAgent: "*",
        allow: "/",
        // The worker and the model shards are machine plumbing, not content.
        // Indexing them wastes crawl budget and surfaces nothing a reader wants.
        disallow: ["/worker.js"],
      },
    ],
    sitemap: `${SITE}/sitemap.xml`,
    host: SITE,
  };
}
