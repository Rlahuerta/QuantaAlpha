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

def render_report(data: Dict[str, Any], *, file=None) -> None:
    """Print a formatted trading report to *file* (default: stdout)."""
    out = file or sys.stdout

    def p(line: str = "") -> None:
        print(line, file=out)

    date_str = data.get("date", "?")
    account = data.get("account_value", 0)
    daily_pnl = data.get("daily_pnl", 0)
    cash = data.get("cash", 0)
    cum_pnl = data.get("cumulative_pnl", 0)
    cum_excess = data.get("cumulative_excess_return", 0)
    bench = data.get("benchmark_return", 0)
    prev_pos = data.get("previous_positions", {})
    target = data.get("target_positions", {})
    prices = data.get("prices", {})
    orders = data.get("orders", [])
    pos_pnl = data.get("position_pnl", {})
    scores_count = data.get("scores_count", 0)
    capital = account  # for weight calc

    # Invested value
    invested = sum(
        target.get(t, 0) * prices.get(t, 0) for t in target
    )

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
    p(f"  Account Value:    ${account:>12,.2f}")
    daily_ret = daily_pnl / (account - daily_pnl) if account != daily_pnl else 0
    p(f"  Daily P&L:        {_pnl_color(daily_pnl):>24s}  ({daily_ret:+.2%})")
    if bench:
        excess = daily_ret - bench
        p(f"  SPY Return:       {bench:>+12.2%}     Excess: {excess:+.2%}")
    p(f"  Cumul P&L:        {_pnl_color(cum_pnl):>24s}  ({cum_excess:+.2%})")
    invest_pct = invested / account * 100 if account else 0
    cash_pct = cash / account * 100 if account else 0
    p(f"  Invested:         ${invested:>12,.0f} ({invest_pct:.0f}%)")
    p(f"  Cash:             ${cash:>12,.0f} ({cash_pct:.0f}%)")
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
            prev_sh = prev_pos.get(o["ticker"], 0)
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
            prev_sh = prev_pos.get(o["ticker"], 0)
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
            prev_sh = prev_pos.get(o["ticker"], 0)
            tgt = target.get(o["ticker"], 0)
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
    """Read order_data JSON from stdin or file arg and render report."""
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        data = json.loads(path.read_text())
    else:
        data = json.load(sys.stdin)
    render_report(data)


if __name__ == "__main__":
    main()
