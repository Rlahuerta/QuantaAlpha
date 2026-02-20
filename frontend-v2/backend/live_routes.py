"""
Live Trading API Routes (Phase 4)

Mounts under the main FastAPI app at /api/v1/live/*.

Endpoints
---------
GET  /api/v1/live/status       Scheduler + IBKR connection status
GET  /api/v1/live/positions    Current open positions from latest positions.json
GET  /api/v1/live/pnl          Full P&L history (list of daily snapshots)
GET  /api/v1/live/signal       Most-recent pending orders file
GET  /api/v1/live/orders       List all pending_orders_*.json files (newest first)
WS   /ws/live/stream           Push-on-change stream (polls every 5 s)
"""

from __future__ import annotations

import asyncio
import glob
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LIVE_DATA_DIR = PROJECT_ROOT / "data" / "live"
PNL_DIR = LIVE_DATA_DIR / "pnl"
ORDERS_DIR = LIVE_DATA_DIR  # pending_orders_*.json live here
POSITIONS_FILE = LIVE_DATA_DIR / "positions.json"
LIVE_CONFIG_FILE = PROJECT_ROOT / "configs" / "live.yaml"

router = APIRouter(prefix="/api/v1/live", tags=["live"])


def _load_json_safe(path: Path) -> Optional[Dict]:
    """Return parsed JSON or None if file missing / unreadable."""
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _latest_orders_file() -> Optional[Path]:
    """Return the most-recent pending_orders_*.json or None."""
    files = sorted(ORDERS_DIR.glob("pending_orders_*.json"), reverse=True)
    return files[0] if files else None


def _scheduler_status() -> Dict[str, Any]:
    """Best-effort scheduler status without importing heavy dependencies."""
    config = _load_json_safe(LIVE_CONFIG_FILE.with_suffix(".json"))
    # live.yaml is YAML, not JSON — skip config read; just report file presence
    live_yaml_exists = LIVE_CONFIG_FILE.exists()
    latest_orders = _latest_orders_file()
    last_signal_date: Optional[str] = None
    if latest_orders:
        name = latest_orders.name  # pending_orders_YYYY-MM-DD.json
        last_signal_date = name.replace("pending_orders_", "").replace(".json", "")
    return {
        "config_found": live_yaml_exists,
        "last_signal_date": last_signal_date,
        "positions_file_exists": POSITIONS_FILE.exists(),
        "pnl_snapshots": len(list(PNL_DIR.glob("pnl_*.json"))) if PNL_DIR.exists() else 0,
        "orders_files": len(list(ORDERS_DIR.glob("pending_orders_*.json"))),
        "ibkr_connected": False,  # dry_run mode by default; TWS not running
        "dry_run": True,
    }


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------


@router.get("/status")
async def get_live_status() -> Dict[str, Any]:
    """Scheduler and IBKR connection status."""
    return {"success": True, "data": _scheduler_status()}


@router.get("/positions")
async def get_live_positions() -> Dict[str, Any]:
    """Current open positions (from the latest positions.json)."""
    if not POSITIONS_FILE.exists():
        return {"success": True, "data": {"positions": {}, "as_of": None}}

    state = _load_json_safe(POSITIONS_FILE)
    if state is None:
        raise HTTPException(status_code=500, detail="Could not parse positions.json")

    return {
        "success": True,
        "data": {
            "positions": state.get("positions", {}),
            "as_of": state.get("as_of"),
            "capital": state.get("capital"),
            "num_positions": len(state.get("positions", {})),
        },
    }


@router.get("/pnl")
async def get_live_pnl(limit: int = 90) -> Dict[str, Any]:
    """Daily P&L history, newest-first, capped at `limit` days."""
    if not PNL_DIR.exists():
        return {"success": True, "data": {"history": [], "summary": {}}}

    files = sorted(PNL_DIR.glob("pnl_*.json"), reverse=True)[:limit]
    history: List[Dict] = []
    for f in files:
        snap = _load_json_safe(f)
        if snap:
            history.append(snap)

    # Reverse to chronological for charting
    history = list(reversed(history))

    # Derive simple summary
    summary: Dict[str, Any] = {}
    if history:
        final = history[-1]
        summary = {
            "total_pnl": final.get("cumulative_pnl", 0),
            "last_date": final.get("date"),
            "num_days": len(history),
        }

    return {"success": True, "data": {"history": history, "summary": summary}}


@router.get("/signal")
async def get_live_signal() -> Dict[str, Any]:
    """Most-recent signal/orders file."""
    orders_file = _latest_orders_file()
    if not orders_file:
        return {"success": True, "data": {"orders": [], "date": None, "scores_count": 0}}

    data = _load_json_safe(orders_file)
    if data is None:
        raise HTTPException(status_code=500, detail="Could not parse orders file")

    return {"success": True, "data": data}


@router.get("/orders")
async def list_live_orders() -> Dict[str, Any]:
    """List all pending_orders_*.json files (metadata only, newest first)."""
    files = sorted(ORDERS_DIR.glob("pending_orders_*.json"), reverse=True)
    items = []
    for f in files:
        data = _load_json_safe(f)
        if data:
            items.append({
                "filename": f.name,
                "date": data.get("date"),
                "orders_count": len(data.get("orders", [])),
                "scores_count": data.get("scores_count", 0),
            })
    return {"success": True, "data": {"orders": items, "total": len(items)}}


# ---------------------------------------------------------------------------
# WebSocket stream — pushes full status every 5 s (or on demand)
# ---------------------------------------------------------------------------


@router.websocket("/ws/live/stream")  # NOTE: registered on app directly, not router
async def live_stream_ws(websocket: WebSocket) -> None:  # pragma: no cover
    """Push live state every 5 seconds to connected clients."""
    await websocket.accept()
    try:
        while True:
            positions = _load_json_safe(POSITIONS_FILE) or {}
            orders_file = _latest_orders_file()
            latest_signal = _load_json_safe(orders_file) if orders_file else {}

            payload = {
                "type": "live_update",
                "timestamp": datetime.utcnow().isoformat(),
                "status": _scheduler_status(),
                "positions": positions.get("positions", {}),
                "latest_signal_date": latest_signal.get("date"),
                "latest_orders_count": len(latest_signal.get("orders", [])),
            }
            await websocket.send_json(payload)
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        pass
