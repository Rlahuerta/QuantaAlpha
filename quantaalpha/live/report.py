"""Trading report renderer for live signal output.

Transforms raw order_data from TradingScheduler into a clear,
actionable terminal report with: account summary, per-position P&L,
categorised trading actions (SELL / BUY NEW / INCREASE / HOLD),
and a target-portfolio table with weights.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


# ── ANSI helpers (safe for pipes — disable if not a TTY) ─────────────
_USE_COLOR = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def _bold(t: str) -> str:
    return _c("1", t)


def _green(t: str) -> str:
    return _c("32", t)


def _red(t: str) -> str:
    return _c("31", t)


def _yellow(t: str) -> str:
    return _c("33", t)


def _cyan(t: str) -> str:
    return _c("36", t)


def _dim(t: str) -> str:
    return _c("2", t)


def _pnl_color(val: float) -> str:
    s = f"${val:+,.2f}"
    if val > 0:
        return _green(s)
    elif val < 0:
        return _red(s)
    return s


# ── Main renderer ────────────────────────────────────────────────────

def render_report(data: Dict[str, Any], *, file=None, ledger: List[Dict] = None) -> None:
    """Print a formatted trading report to *file* (default: stdout).

    Parameters
    ----------
    data : dict
        Order data from scheduler or loaded JSON file.
    file :
        Output stream (default: stdout).
    ledger : list of dict, optional
        Trailing daily performance rows from ``PositionTracker.load_ledger()``.
    """
    out = file or sys.stdout

    def p(line: str = "") -> None:
        print(line, file=out)

    date_str = data.get("date", "?")
    account = data.get("account_value", 0)
    initial_cap = data.get("initial_capital", 0)
    daily_pnl = data.get("daily_pnl", 0)
    cash = data.get("cash", 0) or 0
    cum_pnl = data.get("cumulative_pnl", 0) or 0
    cum_excess = data.get("cumulative_excess_return", 0) or 0
    bench = data.get("benchmark_return", 0) or 0
    prev_pos = data.get("previous_positions") or {}
    target = data.get("target_positions") or {}
    prices = dict(data.get("prices") or {})
    orders = data.get("orders") or []
    pos_pnl = data.get("position_pnl") or {}
    scores_count = data.get("scores_count", 0)

    # Fallback: extract prices from orders when prices dict is empty/missing
    for o in orders:
        t = o.get("ticker", "")
        px = o.get("price", 0)
        if t and px and t not in prices:
            prices[t] = px

    capital = account or 1  # avoid div-by-zero

    # Invested value (sum of target positions × prices)
    invested = sum(
        target.get(t, 0) * prices.get(t, 0) for t in target
    )
    # Derive cash if not explicitly provided
    if not cash and account and invested:
        cash = account - invested

    sep = "═" * 62
    thin = "─" * 62

    # ── Header ───────────────────────────────────────────────────
    p()
    p(sep)
    p(_bold(f"  QuantaAlpha Trading Report — {date_str}"))
    p(sep)

    # ── Account Summary ──────────────────────────────────────────
    p()
    p(_bold("  📊 ACCOUNT SUMMARY"))
    if initial_cap:
        total_return = (account - initial_cap) / initial_cap if initial_cap else 0
        p(f"  Initial Capital:  ${initial_cap:>12,.0f}")
        p(f"  Account Value:    ${account:>12,.2f}  ({total_return:+.2%} total)")
    else:
        p(f"  Account Value:    ${account:>12,.2f}")
    daily_ret = daily_pnl / (account - daily_pnl) if account and account != daily_pnl else 0
    p(f"  Daily P&L:        {_pnl_color(daily_pnl):>24s}  ({daily_ret:+.2%})")
    if bench:
        excess = daily_ret - bench
        p(f"  SPY Return:       {bench:>+12.2%}     Excess: {excess:+.2%}")
    if cum_pnl:
        p(f"  Cumul P&L:        {_pnl_color(cum_pnl):>24s}  ({cum_excess:+.2%})")
    invest_pct = invested / account * 100 if account else 0
    cash_pct = cash / account * 100 if account else 0
    p(f"  Invested:         ${invested:>12,.0f} ({invest_pct:.0f}%)")
    p(f"  Cash:             ${cash:>12,.0f} ({cash_pct:.0f}%)")
    p(f"  Positions:        {len(target):>12d}")
    p(f"  Tickers Scored:   {scores_count:>12d}")

    # ── Per-Position P&L ─────────────────────────────────────────
    if pos_pnl:
        p()
        p(_bold("  📈 POSITION P&L (today's performance)"))
        p(f"  {'Ticker':<8s} {'Shares':>7s}  {'Open':>9s} {'Close':>9s}  {'Day P&L':>11s}")
        p(f"  {'──────':<8s} {'──────':>7s}  {'─────────':>9s} {'─────────':>9s}  {'───────────':>11s}")
        sorted_pnl = sorted(pos_pnl.items(), key=lambda x: x[1]["pnl"])
        for ticker, info in sorted_pnl:
            pnl_str = _pnl_color(info["pnl"])
            p(
                f"  {ticker:<8s} {info['shares']:>7,d}  "
                f"${info['price_start']:>8.2f} ${info['price_end']:>8.2f}  "
                f"{pnl_str:>23s}"
            )
        p(f"  {'':8s} {'':>7s}  {'':>9s} {'TOTAL':>9s}  {_pnl_color(daily_pnl):>23s}")

    # ── Categorise orders ────────────────────────────────────────
    sells: List[Dict] = []       # full exits
    reduces: List[Dict] = []     # partial sells (reduce position)
    new_buys: List[Dict] = []    # new positions
    increases: List[Dict] = []   # add to existing
    holds: List[str] = []        # unchanged

    sell_tickers = {o["ticker"] for o in orders if o["action"] == "sell"}
    buy_tickers = {o["ticker"] for o in orders if o["action"] == "buy"}

    for o in orders:
        t = o["ticker"]
        if o["action"] == "sell":
            if t not in target:
                sells.append(o)
            else:
                reduces.append(o)
        elif o["action"] == "buy":
            if t in prev_pos:
                increases.append(o)
            else:
                new_buys.append(o)

    # Positions in both prev and target with no order = HOLD
    for t in target:
        if t in prev_pos and t not in sell_tickers and t not in buy_tickers:
            holds.append(t)

    total_orders = len(sells) + len(reduces) + len(new_buys) + len(increases)

    # ── Trading Actions ──────────────────────────────────────────
    p()
    p(sep)
    if total_orders == 0:
        p(_bold("  📋 NO TRADES — portfolio unchanged"))
    else:
        p(_bold(f"  📋 TRADING ACTIONS — {total_orders} orders for next session"))
    p(sep)

    # SELL
    if sells:
        p()
        p(_red(f"  ❌ SELL — {len(sells)} position(s) to close"))
        p(f"  {thin}")
        for o in sells:
            val = abs(o["shares"] * o["price"])
            # Use previous_positions if available, else the order's share count
            prev_sh = prev_pos.get(o["ticker"]) or abs(o["shares"])
            p(
                f"  {o['ticker']:<6s}  SELL ALL  "
                f"{prev_sh:>,d} shares × ${o['price']:>8.2f} = "
                f"${val:>10,.0f}   {_dim(o['reason'])}"
            )

    if reduces:
        p()
        p(_yellow(f"  ⬇️  REDUCE — {len(reduces)} position(s) to trim"))
        p(f"  {thin}")
        for o in reduces:
            val = abs(o["shares"] * o["price"])
            tgt = target.get(o["ticker"], 0)
            prev_sh = prev_pos.get(o["ticker"]) or (tgt + abs(o["shares"]))
            p(
                f"  {o['ticker']:<6s}  {prev_sh:>,d} → {tgt:>,d}  "
                f"{o['shares']:>+,d} sh  ${val:>10,.0f}   {_dim(o['reason'])}"
            )

    # BUY NEW
    if new_buys:
        p()
        p(_green(f"  🆕 BUY NEW — {len(new_buys)} position(s) to open"))
        p(f"  {thin}")
        for o in new_buys:
            val = abs(o["shares"] * o["price"])
            p(
                f"  {o['ticker']:<6s}  BUY     "
                f"{o['shares']:>+,d} shares × ${o['price']:>8.2f} = "
                f"${val:>10,.0f}   {_dim(o['reason'])}"
            )

    # INCREASE
    if increases:
        p()
        p(_cyan(f"  ⬆️  INCREASE — {len(increases)} position(s) to add shares"))
        p(f"  {'Ticker':<8s} {'Current':>8s} {'Target':>8s} {'Add':>8s}  {'Est. Cost':>12s}")
        p(f"  {'──────':<8s} {'───────':>8s} {'──────':>8s} {'───':>8s}  {'─────────':>12s}")
        for o in increases:
            tgt = target.get(o["ticker"], 0)
            prev_sh = prev_pos.get(o["ticker"]) or (tgt - abs(o["shares"]))
            val = abs(o["shares"] * o["price"])
            p(
                f"  {o['ticker']:<8s} {prev_sh:>8,d} {tgt:>8,d} "
                f"{o['shares']:>+8,d}  ${val:>11,.0f}"
            )

    # HOLD
    if holds:
        p()
        p(_dim(f"  ── HOLD — {len(holds)} position(s) unchanged"))
        hold_str = "  " + "  ".join(sorted(holds))
        p(_dim(hold_str))

    # ── Target Portfolio ─────────────────────────────────────────
    p()
    p(_bold("  📍 TARGET PORTFOLIO (after all orders executed)"))
    p(f"  {'#':>3s}  {'Ticker':<6s}  {'Shares':>7s}  {'Price':>9s}  "
      f"{'Value':>11s}  {'Weight':>7s}")
    p(f"  {'──':>3s}  {'──────':<6s}  {'──────':>7s}  {'─────────':>9s}  "
      f"{'───────────':>11s}  {'──────':>7s}")

    # Sort by value descending
    items = []
    for t, sh in target.items():
        px = prices.get(t, 0)
        val = sh * px
        items.append((t, sh, px, val))
    items.sort(key=lambda x: -x[3])

    total_val = 0
    for i, (t, sh, px, val) in enumerate(items, 1):
        total_val += val
        wt = val / account * 100 if account else 0
        p(f"  {i:>3d}  {t:<6s}  {sh:>7,d}  ${px:>8.2f}  ${val:>10,.0f}  {wt:>6.1f}%")

    p(f"  {'':>3s}  {'':6s}  {'':>7s}  {'TOTAL':>9s}  ${total_val:>10,.0f}  "
      f"{total_val / account * 100 if account else 0:.0f}%")
    if cash:
        p(f"  {'':>3s}  {'':6s}  {'':>7s}  {'CASH':>9s}  ${cash:>10,.0f}")

    # ── Trailing Performance ─────────────────────────────────────
    if ledger and len(ledger) > 1:
        p()
        p(_bold("  📅 TRADING HISTORY"))
        p(f"  {'Date':<12s} {'Account':>12s} {'Daily P&L':>12s} "
          f"{'Cumul P&L':>12s} {'Pos':>4s} {'Cash':>10s}")
        p(f"  {'──────────':<12s} {'────────────':>12s} {'────────────':>12s} "
          f"{'────────────':>12s} {'────':>4s} {'──────────':>10s}")
        for row in ledger:
            d_pnl = float(row.get("daily_pnl", 0))
            c_pnl = float(row.get("cumulative_pnl", 0))
            acct = float(row.get("account_value", 0))
            npos = int(float(row.get("num_positions", 0)))
            csh = float(row.get("cash", 0))
            # Use plain formatting (no ANSI) for table alignment
            d_str = f"${d_pnl:>+10,.0f}"
            c_str = f"${c_pnl:>+10,.0f}"
            d_colored = _green(d_str) if d_pnl > 0 else _red(d_str) if d_pnl < 0 else d_str
            c_colored = _green(c_str) if c_pnl > 0 else _red(c_str) if c_pnl < 0 else c_str
            # ANSI codes add 9 chars, so pad accordingly
            ansi_pad = 9 if _USE_COLOR and d_pnl != 0 else 0
            ansi_pad_c = 9 if _USE_COLOR and c_pnl != 0 else 0
            p(f"  {row.get('date', '?'):<12s} ${acct:>11,.0f} "
              f"{d_colored:>{12 + ansi_pad}s} "
              f"{c_colored:>{12 + ansi_pad_c}s} "
              f"{npos:>4d} ${csh:>9,.0f}")

    # ── Footer ───────────────────────────────────────────────────
    p()
    p(sep)
    orders_file = data.get("orders_file", "")
    if orders_file:
        p(f"  Orders saved: {orders_file}")
    p(sep)
    p()


# ── CLI entry point ──────────────────────────────────────────────────

def main() -> None:
    """Read order_data JSON from stdin or file arg and render report.

    If the order file lives in a directory containing ``ledger.csv``,
    the trailing performance history is loaded and included.
    """
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        data = json.loads(path.read_text())
    else:
        data = json.load(sys.stdin)

    # Try loading ledger from same directory as the order file
    ledger_rows: List[Dict] = []
    orders_dir = Path(sys.argv[1]).parent if len(sys.argv) > 1 else Path("data/live")
    ledger_path = orders_dir / "ledger.csv"
    if ledger_path.exists():
        import csv
        with open(ledger_path, newline="") as f:
            ledger_rows = list(csv.DictReader(f))
        for r in ledger_rows:
            for col in r:
                if col != "date":
                    try:
                        r[col] = float(r[col]) if r.get(col) else 0.0
                    except (ValueError, TypeError):
                        r[col] = 0.0

    render_report(data, ledger=ledger_rows[-10:] if ledger_rows else None)


if __name__ == "__main__":
    main()
