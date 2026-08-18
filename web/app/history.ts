/* Conversations live in this browser and nowhere else.
 *
 * localStorage rather than a server, because the entire premise of the site is
 * that nothing you type leaves the device — persisting chats to a backend
 * would quietly break that promise for a convenience nobody asked for.
 *
 * Writes are wrapped: Safari private mode throws on setItem, and a chat app
 * that crashes because it could not save history is worse than one that
 * forgets. */

export type Turn = { who: "user" | "bot"; text: string; model?: string };
export type Convo = { id: string; title: string; at: number; model: string; turns: Turn[] };

const KEY = "as.convos.v1";
const LIMIT = 50;

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
    localStorage.setItem(KEY, JSON.stringify(list.slice(0, LIMIT)));
  } catch {
    /* private browsing or quota — losing history is not worth an exception */
  }
}

export function titleFor(turns: Turn[]) {
  const first = turns.find((t) => t.who === "user")?.text ?? "New chat";
  return first.length > 42 ? `${first.slice(0, 42)}…` : first;
}

export function newId() {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`;
}
