/**
 * Proxy between the browser and the Hugging Face Space.
 *
 * The browser could call the Space directly, but going through here means:
 *   - the Space URL isn't baked into client-side JS, so it can't be scraped
 *     and hammered independently of the site
 *   - no CORS configuration to keep in sync
 *   - one place to handle the Space's cold start, which is ~30s after it has
 *     been idle for 48 hours and otherwise looks like a hang to the user
 *
 * This runs on Vercel's Node runtime and only forwards bytes — the model never
 * runs here, which is why it fits inside the Hobby tier's limits.
 */

export const runtime = "nodejs";
export const maxDuration = 60;

const SPACE_URL = process.env.SPACE_URL;

export async function POST(req: Request) {
  if (!SPACE_URL) {
    return Response.json(
      { error: "SPACE_URL is not set. Add it in Vercel → Settings → Environment Variables." },
      { status: 500 },
    );
  }

  let body: { message?: string; temperature?: number; max_tokens?: number };
  try {
    body = await req.json();
  } catch {
    return Response.json({ error: "invalid JSON" }, { status: 400 });
  }

  const message = (body.message ?? "").trim();
  if (!message) return Response.json({ error: "message is required" }, { status: 400 });
  if (message.length > 500) {
    return Response.json({ error: "message too long (max 500 chars)" }, { status: 400 });
  }

  // A sleeping Space takes ~30s to boot. Allow for that, but not forever.
  const abort = new AbortController();
  const timeout = setTimeout(() => abort.abort(), 55_000);

  try {
    const upstream = await fetch(`${SPACE_URL.replace(/\/$/, "")}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        temperature: body.temperature ?? 0.9,
        max_tokens: body.max_tokens ?? 60,
      }),
      signal: abort.signal,
    });

    if (!upstream.ok || !upstream.body) {
      const detail = await upstream.text().catch(() => "");
      return Response.json(
        {
          error:
            upstream.status === 429
              ? "Too many requests — give it a second."
              : "The model backend is unavailable. It may be waking up; try again shortly.",
          detail: detail.slice(0, 200),
        },
        { status: upstream.status === 429 ? 429 : 502 },
      );
    }

    // Pass the SSE stream straight through, untouched.
    return new Response(upstream.body, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache, no-transform",
        Connection: "keep-alive",
      },
    });
  } catch (err) {
    const aborted = err instanceof Error && err.name === "AbortError";
    return Response.json(
      {
        error: aborted
          ? "The backend took too long to respond. It was probably asleep — try once more."
          : "Could not reach the model backend.",
      },
      { status: 504 },
    );
  } finally {
    clearTimeout(timeout);
  }
}
