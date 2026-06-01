# compaRAG Frontend

React/Vite UI for the compaRAG backend.

## Local With Docker Backend

```powershell
copy .env.example .env.local
npm install
npm run dev
```

The default `.env.example` points at `http://127.0.0.1:8001`, which is the Docker Compose backend port.

## Vercel

Import the monorepo in Vercel and set:

- Root Directory: `frontend`
- Framework Preset: Vite
- Build Command: `npm run build`
- Output Directory: `dist`
- Install Command: `npm install`

Set `VITE_API_BASE_URL` in Vercel project environment variables to the deployed backend URL when you have one.
For local-only testing, keep using `http://127.0.0.1:8001`; Vercel-hosted pages cannot call your laptop backend for
other users.
