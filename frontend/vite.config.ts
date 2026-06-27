import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev (`npm run dev`), the frontend runs on :5173 and proxies /api calls to
// the FastAPI backend on :8000. In production we `vite build` and FastAPI serves
// the resulting dist/ folder, so the API is same-origin.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
  build: { outDir: "dist" },
});
