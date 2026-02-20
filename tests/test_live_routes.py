"""
Tests for frontend-v2/backend/live_routes.py

Covers all REST endpoints with mocked file-system data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Resolve backend path so we can import live_routes directly
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).parent.parent / "frontend-v2" / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from live_routes import router, _scheduler_status, _load_json_safe, _latest_orders_file

# ---------------------------------------------------------------------------
# Minimal app fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Create a TestClient with live_routes mounted and all paths redirected to tmp_path."""
    import live_routes as lr

    monkeypatch.setattr(lr, "LIVE_DATA_DIR", tmp_path / "live")
    monkeypatch.setattr(lr, "PNL_DIR", tmp_path / "live" / "pnl")
    monkeypatch.setattr(lr, "ORDERS_DIR", tmp_path / "live")
    monkeypatch.setattr(lr, "POSITIONS_FILE", tmp_path / "live" / "positions.json")
    monkeypatch.setattr(lr, "LIVE_CONFIG_FILE", tmp_path / "configs" / "live.yaml")

    (tmp_path / "live").mkdir(parents=True)
    (tmp_path / "live" / "pnl").mkdir()
    (tmp_path / "configs").mkdir()

    app = FastAPI()
    app.include_router(router)
    return TestClient(app), tmp_path


# ---------------------------------------------------------------------------
# _load_json_safe
# ---------------------------------------------------------------------------


def test_load_json_safe_valid(tmp_path):
    f = tmp_path / "data.json"
    f.write_text('{"key": "value"}')
    assert _load_json_safe(f) == {"key": "value"}


def test_load_json_safe_missing(tmp_path):
    assert _load_json_safe(tmp_path / "nonexistent.json") is None


def test_load_json_safe_invalid_json(tmp_path):
    f = tmp_path / "bad.json"
    f.write_text("not json {{{")
    assert _load_json_safe(f) is None


# ---------------------------------------------------------------------------
# GET /api/v1/live/status
# ---------------------------------------------------------------------------


def test_status_no_config(client):
    c, tmp = client
    resp = c.get("/api/v1/live/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["data"]["config_found"] is False
    assert data["data"]["dry_run"] is True
    assert data["data"]["ibkr_connected"] is False


def test_status_with_config_and_orders(client):
    c, tmp = client
    (tmp / "configs" / "live.yaml").write_text("model: {}")
    (tmp / "live" / "pending_orders_2026-02-20.json").write_text(
        json.dumps({"date": "2026-02-20", "orders": []})
    )
    resp = c.get("/api/v1/live/status")
    data = resp.json()["data"]
    assert data["config_found"] is True
    assert data["last_signal_date"] == "2026-02-20"
    assert data["orders_files"] == 1


def test_status_counts_pnl_snapshots(client):
    c, tmp = client
    for d in ["2026-02-18", "2026-02-19", "2026-02-20"]:
        (tmp / "live" / "pnl" / f"pnl_{d}.json").write_text(json.dumps({"date": d}))
    resp = c.get("/api/v1/live/status")
    assert resp.json()["data"]["pnl_snapshots"] == 3


# ---------------------------------------------------------------------------
# GET /api/v1/live/positions
# ---------------------------------------------------------------------------


def test_positions_no_file(client):
    c, _ = client
    resp = c.get("/api/v1/live/positions")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["positions"] == {}
    assert data["as_of"] is None


def test_positions_with_file(client):
    c, tmp = client
    state = {
        "positions": {"AAPL": 100, "MSFT": 50},
        "as_of": "2026-02-20",
        "capital": 1_000_000,
    }
    (tmp / "live" / "positions.json").write_text(json.dumps(state))
    resp = c.get("/api/v1/live/positions")
    data = resp.json()["data"]
    assert data["positions"]["AAPL"] == 100
    assert data["num_positions"] == 2
    assert data["capital"] == 1_000_000


def test_positions_num_positions_count(client):
    c, tmp = client
    state = {"positions": {f"TICK{i}": i * 10 for i in range(1, 6)}}
    (tmp / "live" / "positions.json").write_text(json.dumps(state))
    data = c.get("/api/v1/live/positions").json()["data"]
    assert data["num_positions"] == 5


# ---------------------------------------------------------------------------
# GET /api/v1/live/pnl
# ---------------------------------------------------------------------------


def test_pnl_no_history(client):
    c, _ = client
    resp = c.get("/api/v1/live/pnl")
    assert resp.status_code == 200
    d = resp.json()["data"]
    assert d["history"] == []
    assert d["summary"] == {}


def test_pnl_returns_chronological_history(client):
    c, tmp = client
    snaps = [
        {"date": "2026-02-18", "daily_pnl": 100, "cumulative_pnl": 100},
        {"date": "2026-02-19", "daily_pnl": -50, "cumulative_pnl": 50},
        {"date": "2026-02-20", "daily_pnl": 200, "cumulative_pnl": 250},
    ]
    for s in snaps:
        (tmp / "live" / "pnl" / f"pnl_{s['date']}.json").write_text(json.dumps(s))
    data = c.get("/api/v1/live/pnl").json()["data"]
    assert len(data["history"]) == 3
    # Chronological order
    assert data["history"][0]["date"] == "2026-02-18"
    assert data["history"][-1]["date"] == "2026-02-20"


def test_pnl_summary_uses_last_entry(client):
    c, tmp = client
    for d, cum in [("2026-02-18", 100), ("2026-02-19", 250), ("2026-02-20", 400)]:
        (tmp / "live" / "pnl" / f"pnl_{d}.json").write_text(
            json.dumps({"date": d, "daily_pnl": 50, "cumulative_pnl": cum})
        )
    summary = c.get("/api/v1/live/pnl").json()["data"]["summary"]
    assert summary["total_pnl"] == 400
    assert summary["last_date"] == "2026-02-20"
    assert summary["num_days"] == 3


def test_pnl_limit_parameter(client):
    c, tmp = client
    for i in range(10):
        d = f"2026-02-{i+1:02d}"
        (tmp / "live" / "pnl" / f"pnl_{d}.json").write_text(
            json.dumps({"date": d, "daily_pnl": 0, "cumulative_pnl": 0})
        )
    data = c.get("/api/v1/live/pnl?limit=5").json()["data"]
    assert len(data["history"]) == 5


# ---------------------------------------------------------------------------
# GET /api/v1/live/signal
# ---------------------------------------------------------------------------


def test_signal_no_file(client):
    c, _ = client
    resp = c.get("/api/v1/live/signal")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["orders"] == []
    assert data["date"] is None


def test_signal_returns_latest_file(client):
    c, tmp = client
    # Two orders files — latest should be returned
    (tmp / "live" / "pending_orders_2026-02-19.json").write_text(
        json.dumps({"date": "2026-02-19", "orders": [{"ticker": "AAPL", "shares": 10, "action": "BUY"}], "scores_count": 5})
    )
    (tmp / "live" / "pending_orders_2026-02-20.json").write_text(
        json.dumps({"date": "2026-02-20", "orders": [], "scores_count": 0})
    )
    data = c.get("/api/v1/live/signal").json()["data"]
    assert data["date"] == "2026-02-20"


def test_signal_contains_orders(client):
    c, tmp = client
    orders = [
        {"ticker": "AAPL", "shares": 10, "action": "BUY", "price": None, "reason": "new entry"},
        {"ticker": "TSLA", "shares": -5, "action": "SELL", "price": None, "reason": "exit"},
    ]
    (tmp / "live" / "pending_orders_2026-02-20.json").write_text(
        json.dumps({"date": "2026-02-20", "orders": orders, "scores_count": 30})
    )
    data = c.get("/api/v1/live/signal").json()["data"]
    assert len(data["orders"]) == 2
    assert data["scores_count"] == 30


# ---------------------------------------------------------------------------
# GET /api/v1/live/orders
# ---------------------------------------------------------------------------


def test_orders_list_empty(client):
    c, _ = client
    data = c.get("/api/v1/live/orders").json()["data"]
    assert data["orders"] == []
    assert data["total"] == 0


def test_orders_list_metadata(client):
    c, tmp = client
    for d in ["2026-02-18", "2026-02-19", "2026-02-20"]:
        (tmp / "live" / f"pending_orders_{d}.json").write_text(
            json.dumps({"date": d, "orders": [{"ticker": "X"}] * 3, "scores_count": 10})
        )
    data = c.get("/api/v1/live/orders").json()["data"]
    assert data["total"] == 3
    # Newest first
    assert data["orders"][0]["date"] == "2026-02-20"
    assert data["orders"][0]["orders_count"] == 3


def test_orders_list_newest_first(client):
    c, tmp = client
    for d in ["2026-02-15", "2026-02-20", "2026-02-10"]:
        (tmp / "live" / f"pending_orders_{d}.json").write_text(
            json.dumps({"date": d, "orders": [], "scores_count": 0})
        )
    dates = [o["date"] for o in c.get("/api/v1/live/orders").json()["data"]["orders"]]
    assert dates == sorted(dates, reverse=True)
