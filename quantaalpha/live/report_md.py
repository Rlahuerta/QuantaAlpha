"""Markdown trading report generator.

Produces a self-contained Markdown file per trading day that is:
- Readable in any Markdown viewer (GitHub, VS Code, terminal ``cat``)
- Persistent (one file per day — full audit trail)
- Printable as plain text (Markdown tables look fine without rendering)
- Actionable — a trader can execute all orders from the SELL/BUY tables
  in the Executive Summary without reading the rest

Output: ``data/live/reports/report_{date}.md``

Usage::

    from quantaalpha.live.report_md import generate_report, save_report

    md_text = generate_report(order_data, ledger=ledger_rows)
    path = save_report(order_data, ledger=ledger_rows)
    print(md_text)        # pipe-safe plain text
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── Helpers ──────────────────────────────────────────────────────────

def _fmt_usd(value: float, signed: bool = False) -> str:
    sign = "+" if signed and value >= 0 else ""
    return f"{sign}${value:,.2f}"


def _fmt_pct(value: float, signed: bool = False) -> str:
    sign = "+" if signed and value >= 0 else ""
    return f"{sign}{value * 100:.2f}%"


def _pnl_badge(value: float, fmt: str = "usd") -> str:
    """Return a value with a +/- sign; caller decides colour in their renderer."""
    if fmt == "pct":
        return _fmt_pct(value, signed=True)
    return _fmt_usd(value, signed=True)


def _trend(rows: List[Dict], col: str) -> str:
    """Last-3-day trend arrow: ↑ ↓ →"""
    vals = [float(r.get(col, 0) or 0) for r in rows[-3:]]
    if len(vals) < 2:
        return "→"
    delta = vals[-1] - vals[-2]
    if delta > 1e-4:
        return "↑"
    elif delta < -1e-4:
        return "↓"
    return "→"


# ── Core generator ───────────────────────────────────────────────────

def generate_report(
    data: Dict[str, Any],
    *,
    ledger: Optional[List[Dict]] = None,
    generated_at: Optional[datetime] = None,
) -> str:
    """Generate a Markdown trading report.

    Parameters
    ----------
    data:
        Order data dict from ``TradingScheduler.run_signal()`` or loaded
        from ``pending_orders_{date}.json``.
    ledger:
        Trailing daily performance rows (from ``PositionTracker.load_ledger()``).
    generated_at:
        Timestamp to embed in the footer (default: now UTC).

    Returns
    -------
    str
        Full Markdown text (UTF-8, LF line endings).
    """
    ledger = ledger or []
    ts = generated_at or datetime.now(timezone.utc)

    # ── Extract fields ───────────────────────────────────────────────
    date_str = data.get("date", "unknown")
    account = float(data.get("account_value") or 0)
    initial_cap = float(data.get("initial_capital") or 0)
    daily_pnl = float(data.get("daily_pnl") or 0)
    cash = float(data.get("cash") or 0)
    cum_pnl = float(data.get("cumulative_pnl") or 0)
    cum_excess = float(data.get("cumulative_excess_return") or 0)
    bench = float(data.get("benchmark_return") or 0)
    prev_pos: Dict[str, int] = data.get("previous_positions") or {}
    target: Dict[str, int] = data.get("target_positions") or {}
    prices: Dict[str, float] = dict(data.get("prices") or {})
    orders: List[Dict] = data.get("orders") or []
    pos_pnl: Dict[str, Dict] = data.get("position_pnl") or {}
    scores_count: int = int(data.get("scores_count") or 0)
    orders_file: str = data.get("orders_file", "")
    kill_switch: bool = bool(data.get("kill_switch"))

    # Fallback: extract prices from order dicts when prices dict absent
    for o in orders:
        t = o.get("ticker", "")
        px = float(o.get("price") or 0)
        if t and px and t not in prices:
            prices[t] = px

    # Derived
    daily_ret = daily_pnl / (account - daily_pnl) if account and account != daily_pnl else 0.0
    total_return = (account - initial_cap) / initial_cap if initial_cap else 0.0
    invested = sum(target.get(t, 0) * prices.get(t, 0) for t in target)
    if not cash and account and invested:
        cash = account - invested
    invest_pct = invested / account * 100 if account else 0.0
    excess_today = daily_ret - bench

    # ── Categorise orders ────────────────────────────────────────────
    sells: List[Dict] = []
    reduces: List[Dict] = []
    new_buys: List[Dict] = []
    increases: List[Dict] = []
    holds: List[str] = []

    sell_tickers = {o["ticker"] for o in orders if o.get("action") == "sell"}
    buy_tickers = {o["ticker"] for o in orders if o.get("action") == "buy"}

    for o in orders:
        t = o["ticker"]
        if o.get("action") == "sell":
            if t not in target:
                sells.append(o)
            else:
                reduces.append(o)
        elif o.get("action") == "buy":
            if t in prev_pos:
                increases.append(o)
            else:
                new_buys.append(o)

    for t in target:
        if t in prev_pos and t not in sell_tickers and t not in buy_tickers:
            holds.append(t)

    total_orders = len(sells) + len(reduces) + len(new_buys) + len(increases)
    action_summary_parts = []
    if sells:
        action_summary_parts.append(f"**{len(sells)} SELL**")
    if reduces:
        action_summary_parts.append(f"**{len(reduces)} REDUCE**")
    if new_buys:
        action_summary_parts.append(f"**{len(new_buys)} BUY NEW**")
    if increases:
        action_summary_parts.append(f"**{len(increases)} INCREASE**")
    if holds:
        action_summary_parts.append(f"{len(holds)} HOLD")
    if not action_summary_parts:
        action_summary_parts = ["**NO TRADES**"]
    action_summary = "  |  ".join(action_summary_parts)

    lines: List[str] = []
    A = lines.append  # shorthand

    # ── Title ────────────────────────────────────────────────────────
    A(f"# QuantaAlpha Daily Trading Report — {date_str}")
    A("")

    if kill_switch:
        A("> ⛔ **KILL-SWITCH TRIGGERED** — daily loss exceeded limit. No orders generated.")
        A("")

    # ── Executive Summary ────────────────────────────────────────────
    A("## Executive Summary")
    A("")

    cum_trend = _trend(ledger, "cumulative_pnl") if len(ledger) >= 2 else "→"

    rows_summary = [
        ("Account Value", f"**{_fmt_usd(account)}**"),
        ("Total Return", f"{_pnl_badge(total_return, 'pct')} since inception ({_fmt_usd(cum_pnl, True)} cumulative P&L) {cum_trend}"),
        ("Today P&L", f"{_pnl_badge(daily_pnl)} ({_pnl_badge(daily_ret, 'pct')})"),
        ("Today vs Benchmark", f"Portfolio {_pnl_badge(daily_ret, 'pct')}  |  SPY {_fmt_pct(bench, True)}  |  Excess {_pnl_badge(excess_today, 'pct')}"),
        ("Today's Action", action_summary),
        ("Portfolio", f"{len(target)} positions  |  {invest_pct:.0f}% invested  |  Cash {_fmt_usd(cash)}"),
        ("Tickers Scored", f"{scores_count:,}"),
    ]

    A("| Metric | Value |")
    A("|--------|-------|")
    for label, val in rows_summary:
        A(f"| {label} | {val} |")
    A("")

    # ── Trading Actions ──────────────────────────────────────────────
    if total_orders == 0 and not kill_switch:
        A("## 📋 No Trades — Portfolio Unchanged")
        A("")
        A("All positions held. No orders to execute.")
        A("")
    else:
        A("## 📋 Trading Actions")
        A("")
        A("> Execute these orders at market open on the **next trading day**.")
        A("")

        # SELL
        if sells:
            A("### 🔴 SELL — Close Positions")
            A("")
            A("| Ticker | Shares | Est. Price | Est. Value | Reason |")
            A("|--------|--------|-----------|-----------|--------|")
            for o in sells:
                sh = abs(int(o.get("shares", 0)))
                px = float(o.get("price") or 0)
                val = sh * px
                reason = o.get("reason", "")
                prev_sh = prev_pos.get(o["ticker"], sh)
                A(f"| {o['ticker']} | −{prev_sh:,} | {_fmt_usd(px)} | {_fmt_usd(val)} | {reason} |")
            A("")

        # REDUCE
        if reduces:
            A("### 🟡 REDUCE — Trim Positions")
            A("")
            A("| Ticker | From | To | Change | Est. Price | Est. Value | Reason |")
            A("|--------|------|-----|--------|-----------|-----------|--------|")
            for o in reduces:
                sh = abs(int(o.get("shares", 0)))
                px = float(o.get("price") or 0)
                val = sh * px
                tgt = int(target.get(o["ticker"], 0))
                prev_sh = prev_pos.get(o["ticker"], tgt + sh)
                reason = o.get("reason", "")
                A(f"| {o['ticker']} | {prev_sh:,} | {tgt:,} | −{sh:,} | {_fmt_usd(px)} | {_fmt_usd(val)} | {reason} |")
            A("")

        # BUY NEW
        if new_buys:
            A("### 🟢 BUY NEW — Open Positions")
            A("")
            A("| Ticker | Shares | Est. Price | Est. Value | Reason |")
            A("|--------|--------|-----------|-----------|--------|")
            for o in new_buys:
                sh = abs(int(o.get("shares", 0)))
                px = float(o.get("price") or 0)
                val = sh * px
                reason = o.get("reason", "")
                A(f"| {o['ticker']} | +{sh:,} | {_fmt_usd(px)} | {_fmt_usd(val)} | {reason} |")
            A("")

        # INCREASE
        if increases:
            A("### 🔵 INCREASE — Add to Positions")
            A("")
            A("| Ticker | From | To | Add | Est. Price | Est. Cost | Reason |")
            A("|--------|------|-----|-----|-----------|----------|--------|")
            for o in increases:
                sh = abs(int(o.get("shares", 0)))
                px = float(o.get("price") or 0)
                val = sh * px
                tgt = int(target.get(o["ticker"], 0))
                prev_sh = prev_pos.get(o["ticker"], tgt - sh)
                reason = o.get("reason", "")
                A(f"| {o['ticker']} | {prev_sh:,} | {tgt:,} | +{sh:,} | {_fmt_usd(px)} | {_fmt_usd(val)} | {reason} |")
            A("")

    # HOLD
    if holds:
        A("### ⏸ Hold — No Action Required")
        A("")
        hold_str = "  ".join(f"`{t}`" for t in sorted(holds))
        A(hold_str)
        A("")

    # ── Position P&L ─────────────────────────────────────────────────
    if pos_pnl:
        A("## 📈 Today's Position P&L")
        A("")
        A("| Ticker | Shares | Open | Close | Day P&L |")
        A("|--------|--------|------|-------|---------|")
        sorted_pnl = sorted(pos_pnl.items(), key=lambda x: x[1].get("pnl", 0))
        for ticker, info in sorted_pnl:
            pnl_val = float(info.get("pnl", 0))
            A(f"| {ticker} | {info.get('shares', 0):,} | "
              f"{_fmt_usd(float(info.get('price_start', 0)))} | "
              f"{_fmt_usd(float(info.get('price_end', 0)))} | "
              f"{_pnl_badge(pnl_val)} |")
        A(f"| **TOTAL** | | | | **{_pnl_badge(daily_pnl)}** |")
        A("")

    # ── Target Portfolio ─────────────────────────────────────────────
    A("## 📍 Target Portfolio (after all orders executed)")
    A("")
    A("| # | Ticker | Shares | Price | Value | Weight |")
    A("|---|--------|--------|-------|-------|--------|")

    items = []
    for t, sh in target.items():
        px = float(prices.get(t) or 0)
        val = sh * px
        items.append((t, int(sh), px, val))
    items.sort(key=lambda x: -x[3])

    total_invested = 0.0
    for i, (t, sh, px, val) in enumerate(items, 1):
        total_invested += val
        wt = val / account * 100 if account else 0.0
        A(f"| {i} | {t} | {sh:,} | {_fmt_usd(px)} | {_fmt_usd(val)} | {wt:.1f}% |")

    total_wt = total_invested / account * 100 if account else 0.0
    A(f"| | **TOTAL** | | | **{_fmt_usd(total_invested)}** | **{total_wt:.0f}%** |")
    if cash:
        A(f"| | CASH | | | {_fmt_usd(cash)} | {cash / account * 100 if account else 0:.0f}% |")
    A("")

    # ── Account Summary ──────────────────────────────────────────────
    A("## 💰 Account Summary")
    A("")
    A("| Metric | Value |")
    A("|--------|-------|")
    A(f"| Initial Capital | {_fmt_usd(initial_cap)} |")
    A(f"| Account Value | {_fmt_usd(account)} |")
    A(f"| Total Return | {_pnl_badge(total_return, 'pct')} |")
    A(f"| Daily P&L | {_pnl_badge(daily_pnl)} ({_pnl_badge(daily_ret, 'pct')}) |")
    A(f"| Cumulative P&L | {_pnl_badge(cum_pnl)} |")
    A(f"| Cumul Excess vs SPY | {_pnl_badge(cum_excess, 'pct')} |")
    A(f"| Invested | {_fmt_usd(invested)} ({invest_pct:.0f}%) |")
    A(f"| Cash | {_fmt_usd(cash)} |")
    A(f"| Positions | {len(target)} |")
    A(f"| Tickers Scored | {scores_count:,} |")
    A("")

    # ── Performance History ──────────────────────────────────────────
    if ledger:
        A("## 📅 Performance History")
        A("")
        A("| Date | Account | Daily P&L | Cumul P&L | Daily Ret | Excess | Positions |")
        A("|------|---------|-----------|-----------|-----------|--------|-----------|")
        for row in ledger:
            d = row.get("date", "?")
            acct = float(row.get("account_value") or 0)
            d_pnl = float(row.get("daily_pnl") or 0)
            c_pnl = float(row.get("cumulative_pnl") or 0)
            d_ret = float(row.get("daily_return") or 0)
            excess = float(row.get("cum_excess_return") or 0)
            npos = int(float(row.get("num_positions") or 0))
            A(f"| {d} | {_fmt_usd(acct)} | {_pnl_badge(d_pnl)} "
              f"| {_pnl_badge(c_pnl)} | {_pnl_badge(d_ret, 'pct')} "
              f"| {_pnl_badge(excess, 'pct')} | {npos} |")
        A("")

    # ── Footer ───────────────────────────────────────────────────────
    A("---")
    A("")
    A(f"*Generated by QuantaAlpha at {ts.strftime('%Y-%m-%d %H:%M')} UTC*  ")
    if orders_file:
        A(f"*Orders file: `{orders_file}`*  ")
    A("")

    return "\n".join(lines)


def save_report(
    data: Dict[str, Any],
    *,
    ledger: Optional[List[Dict]] = None,
    output_dir: Optional[Path | str] = None,
    generated_at: Optional[datetime] = None,
) -> Path:
    """Generate and save the Markdown report.

    Parameters
    ----------
    data:
        Order data dict (from scheduler or JSON file).
    ledger:
        Trailing daily rows.
    output_dir:
        Directory to write to (default: ``data/live/reports``).
    generated_at:
        Timestamp for footer.

    Returns
    -------
    Path
        Path to the written ``.md`` file.
    """
    date_str = data.get("date", "unknown")
    out = Path(output_dir or "data/live/reports")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"report_{date_str}.md"

    md = generate_report(data, ledger=ledger, generated_at=generated_at)
    path.write_text(md, encoding="utf-8")
    return path


# ── CLI entry point ──────────────────────────────────────────────────

def main() -> None:
    """Read order JSON from stdin or file arg and print the Markdown report.

    Usage::

        python -m quantaalpha.live.report_md pending_orders_2026-02-24.json
        cat pending_orders.json | python -m quantaalpha.live.report_md
    """
    import json
    import sys

    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        data = json.loads(path.read_text())
        orders_dir = path.parent
    else:
        data = json.load(sys.stdin)
        orders_dir = Path("data/live")

    # Load ledger from same directory
    ledger_rows: List[Dict] = []
    ledger_path = orders_dir / "ledger.csv"
    if ledger_path.exists():
        import csv
        with open(ledger_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        for r in rows:
            for col in r:
                if col != "date":
                    try:
                        r[col] = float(r[col]) if r.get(col) else 0.0
                    except (ValueError, TypeError):
                        r[col] = 0.0
        ledger_rows = rows[-10:]

    md = generate_report(data, ledger=ledger_rows)

    # Save alongside the order file and also print to stdout
    if len(sys.argv) > 1:
        rpt_path = save_report(data, ledger=ledger_rows, output_dir=orders_dir.parent / "reports")
        print(f"Report saved: {rpt_path}", file=sys.stderr)

    print(md)


if __name__ == "__main__":
    main()
