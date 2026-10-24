// Admin key for admin-only endpoints (ingest, graph snapshot, cache stats, eval, benchmark),
// sent as X-Admin-Key.
// Kept in sessionStorage so it is forgotten when the tab closes.
const STORAGE_KEY = 'lattice-rag.adminKey';

function readStoredKey(): string | null {
  try {
    return sessionStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
}

function storeKey(key: string | null): void {
  try {
    if (key) sessionStorage.setItem(STORAGE_KEY, key);
    else sessionStorage.removeItem(STORAGE_KEY);
  } catch {
    // Storage unavailable (private mode); the key is simply not remembered
  }
}

function promptForKey(reason: string): string | null {
  const key = window.prompt(`${reason}\nEnter the server's ADMIN_API_KEY:`)?.trim();
  storeKey(key || null);
  return key || null;
}

/** True when an admin key was entered earlier in this tab, so admin calls need no prompt. */
export function hasAdminKey(): boolean {
  return Boolean(readStoredKey());
}

/** fetch() with the admin key attached; prompts for it when missing or rejected (401). */
export async function adminFetch(url: string, init: RequestInit = {}): Promise<Response> {
  const send = (key: string) =>
    fetch(url, { ...init, headers: { ...(init.headers ?? {}), 'X-Admin-Key': key } });

  let key = readStoredKey() ?? promptForKey('This action requires the admin key.');
  if (!key) throw new Error('Admin key required for this action.');

  let res = await send(key);
  if (res.status === 401) {
    key = promptForKey('The admin key was rejected.');
    if (!key) return res;
    res = await send(key);
  }
  return res;
}
