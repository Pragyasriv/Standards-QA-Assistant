# Standards Q&A Assistant — Web UI

React + TypeScript + Vite interface for the FastAPI backend.

## Commands

```powershell
npm install
npm run dev      # http://127.0.0.1:5173
npm run build
npm test
```

Run the API on port 8000 first (`uvicorn app.main:app --host 127.0.0.1 --port 8000`). Vite proxies `/ask`, `/health`, and `/ready` to the backend during development.