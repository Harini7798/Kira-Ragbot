// ---- types ----
export type Verdict = { source_num: number; doc_id: string; supported: boolean; claim: string; quote: string };
export type Source = { n: number; doc_id: string; text: string; quotes: string[]; origin: string };
export type AskResult = {
  answer: string; grounded: boolean; abstained: boolean; refused_by_model: boolean;
  grounding: number | null; citation_precision: number | null; faithfulness: number | null;
  verdicts: Verdict[]; sources: Source[]; mode: string;
};
export type Msg = {
  id: string; role: "user" | "assistant"; content: string;
  result?: AskResult | null; images?: string[] | null; ts: number; pending?: boolean;
};
export type Thread = { id: string; title: string; docLabel?: string | null; hasDocs?: boolean; messages: Msg[] };

// ---- auth token ----
const TOKEN_KEY = "kira.token";
export const getToken = () => localStorage.getItem(TOKEN_KEY) || "";
export const setToken = (t: string) => localStorage.setItem(TOKEN_KEY, t);
export const clearToken = () => localStorage.removeItem(TOKEN_KEY);

async function req(path: string, opts: RequestInit = {}): Promise<any> {
  const headers: Record<string, string> = { ...(opts.headers as any) };
  const token = getToken();
  if (token) headers["Authorization"] = `Bearer ${token}`;
  const r = await fetch(path, { ...opts, headers });
  if (r.status === 401) { clearToken(); window.dispatchEvent(new Event("kira-logout")); }
  if (!r.ok) {
    let detail = "Request failed";
    try { detail = (await r.json()).detail || detail; } catch { /* ignore */ }
    throw new Error(detail);
  }
  return r.status === 204 ? null : r.json();
}
const jpost = (path: string, body: unknown) =>
  req(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

// ---- auth ----
export const register = (email: string, password: string) => jpost("/api/auth/register", { email, password });
export const login = (email: string, password: string) => jpost("/api/auth/login", { email, password });
export const getMe = () => req("/api/me");

// ---- threads ----
export const listThreads = () => req("/api/threads") as Promise<Thread[]>;
export const createThread = (title?: string) => jpost("/api/threads", { title: title ?? null }) as Promise<Thread>;
export const renameThread = (id: string, title: string) =>
  req(`/api/threads/${id}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ title }) });
export const deleteThreadApi = (id: string) => req(`/api/threads/${id}`, { method: "DELETE" });

// ---- docs + ask ----
export async function uploadDocs(threadId: string, files: File[]) {
  const fd = new FormData();
  fd.append("thread_id", threadId);
  files.forEach((f) => fd.append("files", f));
  return req("/api/upload", { method: "POST", body: fd }) as Promise<{ collection_id: string; n_docs: number; n_chunks: number; label: string }>;
}

export async function ask(body: Record<string, unknown>, signal?: AbortSignal): Promise<AskResult> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  const token = getToken();
  if (token) headers["Authorization"] = `Bearer ${token}`;
  const r = await fetch("/api/ask", { method: "POST", headers, body: JSON.stringify(body), signal });
  if (r.status === 401) { clearToken(); window.dispatchEvent(new Event("kira-logout")); }
  if (!r.ok) {
    let detail = "Request failed";
    try { detail = (await r.json()).detail || detail; } catch { /* ignore */ }
    throw new Error(detail);
  }
  return r.json();
}
