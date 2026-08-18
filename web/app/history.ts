/* Conversations live in this browser and nowhere else.
 *
 * localStorage rather than a server, because the entire premise of the site is
 * that nothing you type leaves the device — persisting chats to a backend
 * would quietly break that promise for a convenience nobody asked for.
 *
 * Writes are wrapped: Safari private mode throws on setItem, and a chat app
 * that crashes because it could not save history is worse than one that
 * forgets. */

/* "note" is a transcript marker rather than a message — a model switch,
   recorded so a saved conversation still says which model wrote what. */
export type Turn = {
  who: "user" | "bot" | "note";
  text: string;
  model?: string;
  /** data: URL for image-model replies. Stored with the conversation so a
      saved chat still shows what was drawn. */
  image?: string;
  /** Example prompts offered alongside a model-switch note. */
  examples?: string[];
};
export type Convo = { id: string; title: string; at: number; model: string; turns: Turn[] };

const KEY = "as.convos.v1";
const LIMIT = 50;

/* A 64x64 PNG data URL is ~6 KB, and localStorage is ~5 MB total. A few dozen
   generated images will therefore blow the quota and setItem throws, which
   silently kills history saving for everything — text conversations included.
   So images are trimmed oldest-first until the payload fits, rather than
   letting one picture-heavy session take the whole feature down with it. */
const BUDGET = 3_800_000;

function pack(list: Convo[]): string {
  let out = JSON.stringify(list);
  if (out.length <= BUDGET) return out;

  const trimmed = list.map((c) => ({ ...c, turns: c.turns.map((t) => ({ ...t })) }));
  // walk from the oldest conversation forward, dropping image payloads
  for (let i = trimmed.length - 1; i >= 0 && out.length > BUDGET; i--) {
    for (const t of trimmed[i].turns) {
      if (t.image) {
        delete t.image;
        t.text = t.text || "(image not kept — storage full)";
      }
    }
    out = JSON.stringify(trimmed);
  }
  return out;
}

export function loadConvos(): Convo[] {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return [];
    const list = JSON.parse(raw);
    return Array.isArray(list) ? list : [];
  } catch {
    return [];
  }
}

export function saveConvos(list: Convo[]) {
  try {
    localStorage.setItem(KEY, pack(list.slice(0, LIMIT)));
  } catch {
    /* private browsing or quota — losing history is not worth an exception */
  }
}

export function titleFor(turns: Turn[]) {
  // notes are markers, never a title
  const first = turns.find((t) => t.who === "user")?.text ?? "New chat";
  return first.length > 42 ? `${first.slice(0, 42)}…` : first;
}

export function newId() {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`;
}
