# Deploying Kira to Render

Single **Docker** web service: the image builds the React frontend and runs FastAPI,
which serves both the API and the built UI. One repo, one service, one URL.

---

## ⚠️ Plan / RAM requirement (read first)

The app loads PyTorch + the embedding & reranker models when document-RAG or web
search is used — about **1.5–2 GB RAM**.

| Render plan | RAM | Works? |
|---|---|---|
| Free / Starter | 512 MB | ❌ crashes (OOM) when RAG/web loads |
| **Standard** | **2 GB** | ✅ recommended |

Plain chat + image questions (Groq API only) would fit free, but RAG/web won't.
Pick **Standard** for the full app.

---

## Step 1 — Get the code on GitHub

### ✅ Recommended: use git (respects `.gitignore`, can't leak secrets)
From the project folder:
```
git init
git add .
git commit -m "Kira: deployable build"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```
`.gitignore` automatically excludes `.venv/`, `data/`, `node_modules/`, and **`.env`**.

### Alternative: GitHub web drag-and-drop
**Drag-and-drop does NOT respect `.gitignore`** — you must exclude these yourself:

**🚫 NEVER upload:**
- **`.env`** ← contains your API keys; uploading it leaks them. (Render gets keys via env vars instead.)
- `.venv/` and `frontend/node_modules/` ← huge, and rebuilt by Docker
- `data/` ← local DB / SciFact index / uploads
- `frontend/dist/`, `results/`, `__pycache__/`, `*.egg-info/`

**✅ DO upload:** `src/`, `api/`, `frontend/` (its `src/`, `package.json`, `index.html`,
`vite.config.ts`, `tsconfig.json` — but **not** `node_modules`/`dist`), `scripts/`,
`tests/`, `pyproject.toml`, `Dockerfile`, `.dockerignore`, `render.yaml`, `README.md`,
`PROJECT_OVERVIEW.md`, `.gitignore`.

---

## Step 2 — Create the Render service

**Option A — Blueprint (uses `render.yaml`):** Render → **New → Blueprint** → pick the repo.
It reads `render.yaml` (Docker, Standard plan, health check, disk).

**Option B — Manual:** Render → **New → Web Service** → connect the repo →
Render auto-detects the **Dockerfile** → set **Instance Type = Standard (2 GB)**.

Then set **Environment Variables**:
| Key | Value |
|---|---|
| `GROQ_API_KEY` | your Groq key (required) |
| `TAVILY_API_KEY` | your Tavily key (optional — falls back to DuckDuckGo) |
| `KIRA_SECRET` | any long random string (signs login tokens) |

(Optional but recommended) add a **Disk** mounted at **`/app/data`** so accounts,
chat history, and uploaded-doc indexes survive deploys.

Click **Create**. First build takes several minutes (it downloads PyTorch + bakes the
models into the image).

---

## Step 3 — Use it

Open the Render URL → you'll see the **login screen** → **Sign up** → chat, upload
docs, web search, attach images.

Health check: `https://<your-app>.onrender.com/api/health` → `{"ok":true}`.

---

## Notes
- **Secrets** live only in Render's env vars — never in the repo (`.env` is gitignored).
- **Models are baked into the image**, so there are no Hugging Face calls at runtime.
- **Without a disk**, SQLite is ephemeral (accounts/chats reset on each deploy) — fine
  for a demo, add the disk for permanence.
- **Cold starts:** on plans that sleep, the first request after idle is slow while the
  container wakes and loads models.
- **Fitting free tier** would require swapping local embeddings for an embedding API
  (removes PyTorch) — a separate change if cost is a concern.
