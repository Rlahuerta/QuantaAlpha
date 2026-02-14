# QuantaAlpha Web Dashboard (frontend-v2)

A React + TypeScript dashboard for factor mining, factor library management, and independent backtesting.

## Main Pages

- Factor Mining
- Factor Library
- Backtest
- Settings

## Start

### One command

```bash
cd frontend-v2
bash start.sh
```

### Manual start

```bash
# Terminal 1: backend
cd frontend-v2
python backend/app.py

# Terminal 2: frontend
cd frontend-v2
npm install
npm run dev
```

## Access

- Frontend: `http://localhost:3000`
- API docs: `http://localhost:8000/docs`

## Features

- Real-time task progress and logs
- Factor library filtering and export
- Backtest execution and metrics visualization
- Persistent settings via backend + local cache

## Troubleshooting

- If backend is unreachable, start `backend/app.py` first.
- If factor libraries are empty, verify mining produced `all_factors_library*.json`.
- If configuration does not persist, check `.env` permissions and browser storage.
