import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  ask, clearToken, createThread, deleteThreadApi, getMe, getToken, listThreads,
  login, register, renameThread, setToken, uploadDocs,
  type AskResult, type Msg, type Source, type Thread,
} from "./api";

type Settings = {
  useRag: boolean; useWeb: boolean; kb: "docs" | "scifact";
  mode: string; topK: number; webResults: number; threshold: number;
};
const DEFAULT_SETTINGS: Settings = {
  useRag: false, useWeb: false, kb: "docs", mode: "hybrid", topK: 5, webResults: 5, threshold: 0.5,
};
const ORIGIN_ICON: Record<string, string> = { doc: "📄", web: "🌐", scifact: "🧪" };
const SUGGESTIONS = ["What can you do?", "Summarize my uploaded documents.", "What's the latest news on AI models?"];

const uid = () => crypto.randomUUID();
const nowSec = () => Date.now() / 1000;
const load = <T,>(k: string, fb: T): T => {
  try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : fb; } catch { return fb; }
};
const fileToDataUrl = (file: File, max = 1024): Promise<string> =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const img = new Image();
      img.onload = () => {
        let { width, height } = img;
        if (Math.max(width, height) > max) { const s = max / Math.max(width, height); width = Math.round(width * s); height = Math.round(height * s); }
        const c = document.createElement("canvas"); c.width = width; c.height = height;
        c.getContext("2d")!.drawImage(img, 0, 0, width, height);
        resolve(c.toDataURL("image/jpeg", 0.85));
      };
      img.onerror = reject; img.src = reader.result as string;
    };
    reader.onerror = reject; reader.readAsDataURL(file);
  });

function Highlighted({ text, quotes }: { text: string; quotes: string[] }) {
  const clean = quotes.filter(Boolean);
  if (!clean.length) return <>{text}</>;
  const esc = clean.map((q) => q.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const re = new RegExp("(" + esc.join("|") + ")", "g");
  return <>{text.split(re).map((p, i) => (clean.includes(p) ? <mark key={i}>{p}</mark> : <span key={i}>{p}</span>))}</>;
}
const pct = (v: number | null | undefined) => (v == null ? "n/a" : `${Math.round(v * 100)}%`);

function Pre({ children }: any) {
  const ref = useRef<HTMLPreElement>(null);
  const [copied, setCopied] = useState(false);
  return (
    <div className="codeblock">
      <button className="copy-code" onClick={() => { navigator.clipboard?.writeText(ref.current?.innerText ?? ""); setCopied(true); setTimeout(() => setCopied(false), 1200); }}>{copied ? "✓ Copied" : "Copy"}</button>
      <pre ref={ref}>{children}</pre>
    </div>
  );
}
const MD_COMPONENTS = {
  a: ({ node, ...props }: any) => <a {...props} target="_blank" rel="noopener noreferrer" />,
  pre: Pre,
};

function Meta({ r }: { r: AskResult }) {
  if (!r.grounded) return null;
  return (
    <details className="meta">
      <summary>🔍 Citations & sources</summary>
      {r.verdicts.length > 0 && (
        <div className="verdicts">
          {r.verdicts.map((v, i) => (
            <div key={i} className="verdict"><span>{v.supported ? "✅" : "❌"}</span><span className="cite">[{v.source_num}]</span><span className="claim">{v.claim}</span></div>
          ))}
          <div className="metrics">
            <span>Citation precision <b>{pct(r.citation_precision)}</b></span>
            <span>Faithfulness <b>{pct(r.faithfulness)}</b></span>
            <span>Grounding <b>{pct(r.grounding)}</b></span>
          </div>
        </div>
      )}
      {r.sources.map((s: Source) => (
        <details key={s.n} className="src"><summary>{ORIGIN_ICON[s.origin] || "•"} [{s.n}] {s.doc_id}</summary><div className="src-text"><Highlighted text={s.text} quotes={s.quotes} /></div></details>
      ))}
    </details>
  );
}

function Bubble({ m, onCopy, copied }: { m: Msg; onCopy: (m: Msg) => void; copied: boolean }) {
  if (m.role === "user") return (
    <div className="msg user"><div className="bubble">
      {m.images?.length ? <div className="msg-images">{m.images.map((src, i) => <img key={i} src={src} alt="attachment" />)}</div> : null}
      {m.content}
    </div></div>
  );
  const r = m.result;
  let badge = null;
  if (r?.mode === "vision") badge = <span className="status vision">🖼️ Based on your image</span>;
  else if (r && !r.grounded) badge = <span className="status warn">⚠️ Ungrounded — Kira's own knowledge, not verified</span>;
  else if (r?.abstained) badge = <span className="status refuse">🚫 Abstained — not enough grounded evidence</span>;
  else if (r?.grounded) badge = <span className="status ok">✅ Grounded & verified</span>;
  return (
    <div className="msg kira">
      <div className="avatar">K</div>
      <div className="bubble">
        {!m.pending && badge}
        {m.pending ? <div className="typing"><span /><span /><span /></div> : <div className="md"><ReactMarkdown remarkPlugins={[remarkGfm]} components={MD_COMPONENTS}>{m.content}</ReactMarkdown></div>}
        {r && <Meta r={r} />}
        {!m.pending && <div className="actions"><button onClick={() => onCopy(m)}>{copied ? "✓ Copied" : "📋 Copy"}</button></div>}
      </div>
    </div>
  );
}

function Auth({ onAuth }: { onAuth: (email: string) => void }) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [pw, setPw] = useState("");
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const submit = async () => {
    if (!email.trim() || !pw) return setErr("Enter an email and password.");
    setErr(""); setBusy(true);
    try {
      const res = await (mode === "login" ? login : register)(email.trim(), pw);
      setToken(res.token); onAuth(res.email);
    } catch (e: any) { setErr(e.message); } finally { setBusy(false); }
  };
  return (
    <div className="auth">
      <div className="auth-card">
        <div className="brand">✦ Kira</div>
        <p className="auth-sub">Measurable RAG — answers your documents & the web, cites and verifies every claim.</p>
        <input placeholder="Email" value={email} onChange={(e) => setEmail(e.target.value)} />
        <input type="password" placeholder="Password" value={pw} onChange={(e) => setPw(e.target.value)} onKeyDown={(e) => e.key === "Enter" && submit()} />
        {err && <div className="error">{err}</div>}
        <button className="primary" onClick={submit} disabled={busy}>{busy ? "…" : mode === "login" ? "Log in" : "Create account"}</button>
        <p className="auth-switch">
          {mode === "login"
            ? <>No account? <a onClick={() => { setMode("register"); setErr(""); }}>Sign up</a></>
            : <>Have an account? <a onClick={() => { setMode("login"); setErr(""); }}>Log in</a></>}
        </p>
      </div>
    </div>
  );
}

export default function App() {
  const [token, setTok] = useState(getToken());
  const [email, setEmail] = useState("");
  const [booting, setBooting] = useState(true);
  const [threads, setThreads] = useState<Thread[]>([]);
  const [activeId, setActiveId] = useState<string>("");

  const [settings, setSettings] = useState<Settings>(() => load("kira.settings", DEFAULT_SETTINGS));
  const [theme, setTheme] = useState<string>(() => load("kira.theme", "dark"));
  const [input, setInput] = useState("");
  const [advanced, setAdvanced] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [dragging, setDragging] = useState(false);
  const [search, setSearch] = useState("");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editText, setEditText] = useState("");
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [atBottom, setAtBottom] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [pendingFiles, setPendingFiles] = useState<File[]>([]);

  const endRef = useRef<HTMLDivElement>(null);
  const msgsRef = useRef<HTMLDivElement>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => { localStorage.setItem("kira.settings", JSON.stringify(settings)); }, [settings]);
  useEffect(() => { document.documentElement.dataset.theme = theme; localStorage.setItem("kira.theme", JSON.stringify(theme)); }, [theme]);
  useEffect(() => {
    const onLogout = () => { setTok(""); setThreads([]); setActiveId(""); setEmail(""); };
    window.addEventListener("kira-logout", onLogout);
    return () => window.removeEventListener("kira-logout", onLogout);
  }, []);

  // Boot: validate token, load chats.
  useEffect(() => {
    if (!token) { setBooting(false); return; }
    let alive = true;
    (async () => {
      try {
        const me = await getMe(); if (!alive) return; setEmail(me.email);
        let ts = await listThreads();
        if (!ts.length) ts = [await createThread()];
        if (!alive) return;
        setThreads(ts); setActiveId(ts[0].id);
      } catch { clearToken(); setTok(""); }
      finally { if (alive) setBooting(false); }
    })();
    return () => { alive = false; };
  }, [token]);

  const active = useMemo(() => threads.find((t) => t.id === activeId), [threads, activeId]);
  useEffect(() => { if (atBottom) endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [active?.messages]); // eslint-disable-line
  useEffect(() => { taRef.current?.focus(); }, [activeId]);

  const set = (patch: Partial<Settings>) => setSettings((s) => ({ ...s, ...patch }));
  const patchThread = (id: string, fn: (t: Thread) => Thread) => setThreads((ts) => ts.map((t) => (t.id === id ? fn(t) : t)));

  const newThread = async () => {
    try { const t = await createThread(); setThreads((p) => [t, ...p]); setActiveId(t.id); setError(""); setPendingFiles([]); }
    catch (e: any) { setError(e.message); }
  };
  const selectThread = (id: string) => {
    setActiveId(id); setPendingFiles([]); setError("");
    if (window.innerWidth < 760) setSidebarOpen(false);
  };
  const deleteThread = async (id: string) => {
    if (!window.confirm("Delete this chat? This cannot be undone.")) return;
    try { await deleteThreadApi(id); } catch (e: any) { setError(e.message); return; }
    const rest = threads.filter((t) => t.id !== id);
    if (!rest.length) { const t = await createThread(); setThreads([t]); setActiveId(t.id); return; }
    setThreads(rest);
    if (id === activeId) setActiveId(rest[0].id);
  };
  const commitRename = () => {
    if (editingId && editText.trim()) {
      const title = editText.trim();
      patchThread(editingId, (t) => ({ ...t, title }));
      renameThread(editingId, title).catch(() => {});
    }
    setEditingId(null);
  };
  const exportThread = (t: Thread) => {
    const md = `# ${t.title}\n\n` + t.messages.filter((m) => !m.pending)
      .map((m) => (m.role === "user" ? `**You:** ${m.content}` : `**Kira:** ${m.content}`)).join("\n\n");
    const url = URL.createObjectURL(new Blob([md], { type: "text/markdown" }));
    const a = document.createElement("a"); a.href = url; a.download = `${(t.title || "chat").replace(/[^\w-]+/g, "_")}.md`; a.click(); URL.revokeObjectURL(url);
  };
  const copy = (m: Msg) => { navigator.clipboard?.writeText(m.content); setCopiedId(m.id); setTimeout(() => setCopiedId(null), 1500); };
  const logout = () => { clearToken(); setTok(""); setThreads([]); setActiveId(""); setEmail(""); };

  const addFiles = (files: FileList | null) => { if (files?.length) { setPendingFiles((p) => [...p, ...Array.from(files)]); setError(""); } };
  const removeFile = (i: number) => setPendingFiles((p) => p.filter((_, idx) => idx !== i));

  const send = async (text?: string) => {
    const q = (text ?? input).trim();
    if (!q || loading || !active) return;
    setError("");
    const tid = active.id;

    const imgs = pendingFiles.filter((f) => f.type.startsWith("image/"));
    const docs = pendingFiles.filter((f) => !f.type.startsWith("image/"));
    let userImages: string[] = [];
    if (imgs.length) {
      try { userImages = await Promise.all(imgs.map((f) => fileToDataUrl(f))); } catch { setError("Could not read that image."); return; }
      setPendingFiles([]);
    } else if (docs.length) {
      setUploading(true); setNotice("⏳ Indexing attached file(s)…");
      try {
        const r = await uploadDocs(tid, docs);
        patchThread(tid, (t) => ({ ...t, docLabel: r.label, hasDocs: true }));
        set({ useRag: true, kb: "docs" }); setPendingFiles([]);
        setNotice(`📄 Indexed ${r.n_docs} file(s) · ${r.n_chunks} chunks.`); setTimeout(() => setNotice(""), 4000);
      } catch (e: any) { setError(e.message); setUploading(false); return; }
      setUploading(false);
    }

    const userMsg: Msg = { id: uid(), role: "user", content: q, ts: nowSec(), images: userImages.length ? userImages : undefined };
    const pending: Msg = { id: uid(), role: "assistant", content: "", pending: true, ts: nowSec() };
    patchThread(tid, (t) => ({ ...t, title: t.messages.length === 0 ? q.slice(0, 42) : t.title, messages: [...t.messages.filter((m) => !m.pending), userMsg, pending] }));
    setInput(""); if (taRef.current) { taRef.current.style.height = "auto"; taRef.current.focus(); }
    setLoading(true);
    const ctrl = new AbortController(); abortRef.current = ctrl;
    try {
      const res = await ask({
        thread_id: tid, query: q, use_rag: settings.useRag, use_web: settings.useWeb, kb: settings.kb,
        mode: settings.mode, top_k: settings.topK, web_results: settings.webResults, threshold: settings.threshold, images: userImages,
      }, ctrl.signal);
      patchThread(tid, (t) => ({ ...t, messages: t.messages.map((m) => (m.id === pending.id ? { ...m, pending: false, content: res.answer, result: res } : m)) }));
    } catch (e: any) {
      const msg = e.name === "AbortError" ? "⏹ Stopped." : "⚠️ " + e.message;
      patchThread(tid, (t) => ({ ...t, messages: t.messages.map((m) => (m.id === pending.id ? { ...m, pending: false, content: msg } : m)) }));
    } finally { abortRef.current = null; setLoading(false); }
  };
  const stop = () => abortRef.current?.abort();
  const autosize = () => { const ta = taRef.current; if (ta) { ta.style.height = "auto"; ta.style.height = Math.min(ta.scrollHeight, 160) + "px"; } };
  const onScroll = () => { const el = msgsRef.current; if (el) setAtBottom(el.scrollHeight - el.scrollTop - el.clientHeight < 80); };

  if (!token) return <Auth onAuth={(em) => { setEmail(em); setBooting(true); setTok(getToken()); }} />;
  if (booting) return <div className="booting">Loading Kira…</div>;

  const filtered = threads.filter((t) => t.title.toLowerCase().includes(search.toLowerCase()));
  const modeHint = settings.useRag || settings.useWeb
    ? `Answering from ${[settings.useRag && (settings.kb === "scifact" ? "SciFact" : "your documents"), settings.useWeb && "the web"].filter(Boolean).join(" + ")} · verified citations`
    : "Both off → Kira answers from her own knowledge (⚠️ unverified)";

  return (
    <div className="app"
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={(e) => { if (e.currentTarget === e.target) setDragging(false); }}
      onDrop={(e) => { e.preventDefault(); setDragging(false); addFiles(e.dataTransfer.files); }}>
      {dragging && <div className="dropzone">📎 Drop files to attach (review before sending)</div>}

      <aside className={sidebarOpen ? "threads" : "threads closed"}>
        <div className="brand">✦ Kira</div>
        <button className="newchat" onClick={newThread}>＋ New chat</button>
        <input className="search" placeholder="Search chats…" value={search} onChange={(e) => setSearch(e.target.value)} />
        <div className="thread-list">
          {filtered.map((t) => (
            <div key={t.id} className={t.id === activeId ? "thread active" : "thread"} onClick={() => selectThread(t.id)}>
              {editingId === t.id ? (
                <input className="rename" autoFocus value={editText} onChange={(e) => setEditText(e.target.value)} onBlur={commitRename}
                  onKeyDown={(e) => { if (e.key === "Enter") commitRename(); if (e.key === "Escape") setEditingId(null); }} onClick={(e) => e.stopPropagation()} />
              ) : (
                <span className="thread-title" onDoubleClick={(e) => { e.stopPropagation(); setEditingId(t.id); setEditText(t.title); }}>{t.title || "New chat"}</span>
              )}
              <span className="thread-actions">
                <button title="Rename" onClick={(e) => { e.stopPropagation(); setEditingId(t.id); setEditText(t.title); }}>✎</button>
                <button title="Delete" onClick={(e) => { e.stopPropagation(); deleteThread(t.id); }}>×</button>
              </span>
            </div>
          ))}
        </div>
        <div className="foot">
          <div className="user-row"><span title={email}>{email}</span><button className="logout" onClick={logout}>Log out</button></div>
        </div>
      </aside>

      <main className="chat">
        <div className="chat-header">
          <button className="ham" onClick={() => setSidebarOpen((s) => !s)}>☰</button>
          <span className="ch-title">{active?.title || "New chat"}</span>
          {uploading && <span className="doc-chip">⏳ indexing…</span>}
          {!uploading && active?.docLabel && <span className="doc-chip">📄 {active.docLabel}</span>}
          <button className="ch-act" title="Toggle light/dark" onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}>{theme === "dark" ? "☀️" : "🌙"}</button>
          <button className="ch-act" title="Export chat" onClick={() => active && exportThread(active)}>⬇ Export</button>
        </div>

        <div className="messages" ref={msgsRef} onScroll={onScroll}>
          {active && active.messages.length === 0 && (
            <div className="empty">
              <h1>✦ Kira</h1>
              <p>Ask over your documents, the web, or just chat. Every grounded answer is cited and verified against its source.</p>
              <div className="suggest">{SUGGESTIONS.map((s) => <button key={s} onClick={() => send(s)}>{s}</button>)}</div>
            </div>
          )}
          {active?.messages.map((m) => <Bubble key={m.id} m={m} onCopy={copy} copied={copiedId === m.id} />)}
          <div ref={endRef} />
        </div>

        {!atBottom && <button className="scroll-btn" onClick={() => endRef.current?.scrollIntoView({ behavior: "smooth" })}>↓</button>}

        <div className="composer">
          {notice && <div className="notice">{notice}</div>}
          {error && <div className="error">{error}</div>}
          <div className="toggles">
            <button className={settings.useRag ? "toggle on" : "toggle"} onClick={() => set({ useRag: !settings.useRag })}>📄 RAG</button>
            <button className={settings.useWeb ? "toggle on" : "toggle"} onClick={() => set({ useWeb: !settings.useWeb })}>🌐 Web search</button>
            <button className="adv-btn" onClick={() => setAdvanced((a) => !a)}>⚙ Advanced</button>
            <span className="hint">{modeHint}</span>
          </div>
          {advanced && (
            <div className="advanced">
              <label>Retrieval
                <select value={settings.mode} onChange={(e) => set({ mode: e.target.value })}>
                  <option value="hybrid">hybrid (dense+BM25)</option><option value="dense">dense only</option><option value="rerank">hybrid + rerank</option>
                </select>
              </label>
              <label>Knowledge base
                <select value={settings.kb} onChange={(e) => set({ kb: e.target.value as Settings["kb"] })}>
                  <option value="docs">Your documents</option><option value="scifact">SciFact demo</option>
                </select>
              </label>
              <label>top-k: {settings.topK}<input type="range" min={3} max={10} value={settings.topK} onChange={(e) => set({ topK: +e.target.value })} /></label>
              <label>Web pages: {settings.webResults}<input type="range" min={3} max={8} value={settings.webResults} onChange={(e) => set({ webResults: +e.target.value })} /></label>
              <label>Abstention: {settings.threshold.toFixed(2)}<input type="range" min={0} max={1} step={0.05} value={settings.threshold} onChange={(e) => set({ threshold: +e.target.value })} /></label>
            </div>
          )}
          {pendingFiles.length > 0 && (
            <div className="attachments">{pendingFiles.map((f, i) => <span className="attach" key={i}>📎 {f.name}<button onClick={() => removeFile(i)} title="Remove">×</button></span>)}</div>
          )}
          <div className="input-row">
            <input ref={fileInputRef} type="file" multiple style={{ display: "none" }}
              accept=".pdf,.docx,.xlsx,.txt,.md,.csv,.tsv,.json,.py,.js,.ts,.html,.log,.png,.jpg,.jpeg,.webp,.gif"
              onChange={(e) => { addFiles(e.target.files); e.currentTarget.value = ""; }} />
            <button className="attach-btn" title="Attach files" onClick={() => fileInputRef.current?.click()}>📎</button>
            <textarea ref={taRef} value={input} autoFocus placeholder="Message Kira…  (Enter to send, Shift+Enter for newline)" rows={1}
              onChange={(e) => { setInput(e.target.value); autosize(); }}
              onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }} />
            {loading ? <button className="send stop" onClick={stop} title="Stop">■</button>
              : <button className="send" onClick={() => send()} title="Send" disabled={!input.trim()}>➤</button>}
          </div>
        </div>
      </main>
    </div>
  );
}
