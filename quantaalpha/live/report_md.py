"""Markdown trading report generator — chain-of-blocks format.

Produces a single Markdown file that reads like a blockchain ledger:
one self-contained block per trading day, each block showing:

  • Overnight P&L attribution (per-position price movements)
  • Rebalancing orders with a running cash-flow waterfall
  • End-of-day snapshot (portfolio / cash / total account)

The **current day** block also lists the algorithm's top-N picks with
weights so the trader can see exactly which stocks are selected and why.

Output: ``data/live/reports/trading_journal.md``

Usage::

    from quantaalpha.live.report_md import generate_chain_report, save_chain_report

    md = generate_chain_report(days_list, initial_capital=1_000_000, topk=10)
    path = save_chain_report("data/live", topk=10)
    print(md)          # pipe-safe plain text, Markdown renders nicely
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── Formatting helpers ────────────────────────────────────────────────

def _usd(v: float, signed: bool = False) -> str:
    prefix = "+" if signed and v >= 0 else ""
    return f"{prefix}${v:,.2f}"


def _pct(v: float, signed: bool = False) -> str:
    prefix = "+" if signed and v >= 0 else ""
    return f"{prefix}{v * 100:.2f}%"


def _sign(v: float) -> str:
    return "+" if v >= 0 else ""


def _action_icon(action: str, is_new: bool, is_full_exit: bool) -> str:
    if action == "sell":
        return "🔴 SELL ALL" if is_full_exit else "🟡 REDUCE"
    if action == "buy":
        return "🟢 BUY NEW" if is_new else "🔵 ADD"
    return "⏸ HOLD"


# ── Algorithm info loader ─────────────────────────────────────────────

def load_algo_info(config_path: str | Path) -> Dict[str, Any]:
    """Load algorithm profile from live.yaml + linked model meta + backtest metrics.

    Returns a flat dict with keys consumed by :func:`generate_chain_report`:
    ``model_stem``, ``num_features``, ``factor_json``, ``topk``, ``n_drop``,
    ``capital``, ``max_position_pct``, ``market``, ``benchmark``,
    ``train_range``, ``test_range``, ``arr``, ``ir``, ``mdd``, ``calmar``,
    ``ic``, ``rank_ic``, ``metrics_source``.
    Returns an empty dict if anything fails (report degrades gracefully).
    """
    try:
        import yaml  # soft import — not available in all test envs
    except ImportError:
        return {}

    config_path = Path(config_path)
    if not config_path.exists():
        return {}

    try:
        with config_path.open() as fh:
            cfg = yaml.safe_load(fh)
    except Exception:
        return {}

    info: Dict[str, Any] = {}
    portfolio = cfg.get("portfolio", {})
    info["topk"] = portfolio.get("topk", 10)
    info["n_drop"] = portfolio.get("n_drop", 1)
    info["capital"] = portfolio.get("capital", 1_000_000)
    info["max_position_pct"] = portfolio.get("max_position_pct", 0.10)

    # Resolve paths relative to config file's directory
    base = config_path.parent

    meta_path = cfg.get("model", {}).get("meta_path")
    if meta_path:
        meta_file = (base / meta_path) if not Path(meta_path).is_absolute() else Path(meta_path)
        # also try relative to repo root
        if not meta_file.exists():
            meta_file = Path(meta_path)
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
                info["model_stem"] = meta.get("model_stem", "")
                info["num_features"] = meta.get("num_features", 0)
                fj = meta.get("factor_json", "")
                # factor_json may be a list (multiple libraries)
                if isinstance(fj, list):
                    fj = fj[0] if fj else ""
                info["factor_json"] = Path(fj).name if fj else ""

                # Derive sibling backtest-metrics file
                stem = meta.get("model_stem", "")
                metrics_file = meta_file.parent / f"{stem}_backtest_metrics.json"
                # fall back to results dir convention
                if not metrics_file.exists():
                    metrics_file = (
                        base / "data/results/backtest_v2_results"
                        / f"{stem}_backtest_metrics.json"
                    )
                if not metrics_file.exists():
                    metrics_file = Path(
                        f"data/results/backtest_v2_results/{stem}_backtest_metrics.json"
                    )
                if metrics_file.exists():
                    bm = json.loads(metrics_file.read_text())
                    m = bm.get("metrics", {})
                    c = bm.get("config", {})
                    info.update({
                        "arr": m.get("annualized_return"),
                        "ir": m.get("information_ratio"),
                        "mdd": m.get("max_drawdown"),
                        "calmar": m.get("calmar_ratio"),
                        "ic": m.get("IC"),
                        "rank_ic": m.get("Rank IC"),
                        "market": c.get("market", ""),
                        "benchmark": c.get("benchmark", "SPY"),
                        "train_range": c.get("data_range", ""),
                        "test_range": c.get("test_range", ""),
                        "metrics_source": metrics_file.name,
                    })
            except Exception:
                pass

    return info


# ── Order file loader ─────────────────────────────────────────────────

def load_all_orders(orders_dir: str | Path) -> List[Dict[str, Any]]:
    """Load all ``pending_orders_*.json`` files, deduplicated by date.

    Files in the live directory take precedence over archived copies
    (same date = live file wins).  Returns list sorted oldest → newest.
    """
    orders_dir = Path(orders_dir)
    archive_dir = orders_dir / "archive"

    by_date: Dict[str, Dict] = {}

    # Load archive first (lower priority)
    if archive_dir.exists():
        for path in sorted(archive_dir.glob("pending_orders_*.json")):
            try:
                d = json.loads(path.read_text())
                date = d.get("date")
                if date:
                    by_date[date] = d
            except Exception:
                continue

    # Live files override archive
    for path in sorted(orders_dir.glob("pending_orders_*.json")):
        try:
            d = json.loads(path.read_text())
            date = d.get("date")
            if date:
                by_date[date] = d
        except Exception:
            continue

    return [by_date[k] for k in sorted(by_date)]


# ── Chain report generator ────────────────────────────────────────────

def generate_chain_report(
    days: List[Dict[str, Any]],
    *,
    initial_capital: float = 1_000_000.0,
    topk: int = 10,
    min_cash_pct: float = 0.10,
    algo_info: Optional[Dict[str, Any]] = None,
    generated_at: Optional[datetime] = None,
) -> str:
    """Generate a day-by-day chained trading journal.

    Parameters
    ----------
    days:
        List of order-data dicts sorted oldest → newest.
        Each dict is a ``pending_orders_{date}.json`` payload.
    initial_capital:
        Starting cash (used for the initial block).
    topk:
        Number of algorithm picks to show in the current-day section.
    min_cash_pct:
        Minimum cash reserve as a fraction of account value (default: 10%).
        Blocks where cash falls below this threshold are flagged with ⚠️.
    algo_info:
        Optional dict returned by :func:`load_algo_info`.  When present,
        the initial block includes an algorithm profile with backtest metrics.
    generated_at:
        Footer timestamp (default: UTC now).

    Returns
    -------
    str
        Full Markdown text.
    """
    ts = generated_at or datetime.now(timezone.utc)
    ai = algo_info or {}
    lines: List[str] = []
    A = lines.append

    # ── Title & header ────────────────────────────────────────────────
    model_name = ai.get("model_stem") or "QuantaAlpha"
    A("# QuantaAlpha — Trading Journal")
    A("")
    A(f"*Generated: {ts.strftime('%Y-%m-%d %H:%M')} UTC* &nbsp;|&nbsp; "
      f"*Model: **{model_name}*** &nbsp;|&nbsp; "
      f"*Strategy: TopkDropout (topk={topk}, n_drop={ai.get('n_drop', 1)})* &nbsp;|&nbsp; "
      f"*Capital: {_usd(initial_capital)}*")
    A("")
    A("---")
    A("")

    # ── Block #0 — Initial state ──────────────────────────────────────
    A("## 🏁 Block #0 — Initial State")
    A("")

    # ── Algorithm profile (shown when algo_info provided) ─────────────
    if ai.get("model_stem"):
        A("### 🤖 Algorithm Profile")
        A("")
        A("| | |")
        A("|---|---|")
        A(f"| **Model** | `{ai['model_stem']}` |")
        num_f = ai.get("num_features")
        if num_f:
            A(f"| **Features** | {num_f} custom alpha factors (LLM-mined) |")
        fj = ai.get("factor_json")
        if fj:
            A(f"| **Factor library** | `{fj}` |")
        market = ai.get("market", "")
        if market:
            A(f"| **Universe** | {market.upper()} |")
        bm = ai.get("benchmark", "")
        if bm:
            A(f"| **Benchmark** | {bm} |")
        if ai.get("train_range"):
            A(f"| **Training period** | {ai['train_range']} |")
        A("")

    # ── Backtest performance card ─────────────────────────────────────
    if ai.get("arr") is not None:
        test_range = ai.get("test_range", "")
        A(f"### 📊 Backtest Performance"
          + (f" ({test_range})" if test_range else ""))
        A("")
        A("| Metric | Value | |")
        A("|--------|-------|---|")
        arr = ai["arr"]
        ir  = ai.get("ir")
        mdd = ai.get("mdd")
        cal = ai.get("calmar")
        ic  = ai.get("ic")
        ric = ai.get("rank_ic")
        A(f"| Annualized Return | **{arr*100:+.1f}%** | excess vs {ai.get('benchmark','SPY')} |")
        if ir is not None:
            A(f"| Information Ratio (Sharpe) | {ir:.3f} | |")
        if mdd is not None:
            A(f"| Max Drawdown | {mdd*100:.1f}% | |")
        if cal is not None:
            A(f"| Calmar Ratio | {cal:.3f} | Return / MDD |")
        if ic is not None:
            A(f"| IC | {ic:.4f} | prediction accuracy |")
        if ric is not None:
            A(f"| Rank IC | {ric:.4f} | rank correlation |")
        src = ai.get("metrics_source", "")
        if src:
            A("")
            A(f"*Source: `{src}`*")
        A("")

    # ── Strategy parameters ───────────────────────────────────────────
    A("### ⚙️ Strategy Parameters")
    A("")
    A("| | |")
    A("|---|---|")
    A(f"| **Capital** | {_usd(initial_capital)} |")
    A(f"| **Portfolio size** | top-{topk} stocks by model score |")
    A(f"| **Rebalance** | drop {ai.get('n_drop', 1)} position(s) per session |")
    max_pos = ai.get("max_position_pct")
    if max_pos:
        A(f"| **Max single position** | {max_pos*100:.0f}% of capital |")
    A(f"| **Cash reserve floor** | {min_cash_pct*100:.0f}% of account value |")
    A(f"| **Initial positions** | 100% cash |")
    A("")

    # Running account state across blocks
    prev_account = initial_capital
    prev_cash = initial_capital           # all cash before any trades
    prev_target: Dict[str, int] = {}      # no positions initially
    cumulative_pnl = 0.0

    # ── One block per day ─────────────────────────────────────────────
    for block_num, day in enumerate(days, 1):
        date_str = day.get("date", f"day-{block_num}")
        account = float(day.get("account_value") or prev_account)
        daily_pnl = float(day.get("daily_pnl") or 0.0)
        # Carry initial capital through if not set
        init_cap = float(day.get("initial_capital") or initial_capital)
        total_return = (account - init_cap) / init_cap if init_cap else 0.0

        orders: List[Dict] = day.get("orders") or []
        target: Dict[str, int] = day.get("target_positions") or {}
        prev_pos: Dict[str, int] = day.get("previous_positions") or prev_target
        prices: Dict[str, float] = dict(day.get("prices") or {})
        pos_pnl: Dict[str, Dict] = day.get("position_pnl") or {}
        bench = float(day.get("benchmark_return") or 0.0)
        cum_excess = float(day.get("cumulative_excess_return") or 0.0)
        cash_eod = day.get("cash")  # may be None for older files

        # Backfill prices from order dicts (always available)
        for o in orders:
            t = o.get("ticker", "")
            px = float(o.get("price") or 0.0)
            if t and px and t not in prices:
                prices[t] = px

        daily_ret = daily_pnl / (account - daily_pnl) if (account - daily_pnl) != 0 else 0.0
        excess_today = daily_ret - bench
        cumulative_pnl = float(day.get("cumulative_pnl") or (cumulative_pnl + daily_pnl))
        is_current = (block_num == len(days))

        # ── Block header ──────────────────────────────────────────────
        A(f"## 📅 Block #{block_num} — {date_str}"
          + (" ← **CURRENT**" if is_current else ""))
        A("")

        # Account movement line
        direction = "📈" if account >= prev_account else "📉"
        A(f"{direction} **{_usd(prev_account)}** → **{_usd(account)}** "
          f"&nbsp; {_sign(daily_pnl)}{_usd(daily_pnl)} ({_sign(daily_ret)}{daily_ret*100:.2f}%)"
          + (f" &nbsp;|&nbsp; SPY: {_sign(bench)}{bench*100:.2f}%"
             f" &nbsp;|&nbsp; Excess: {_sign(excess_today)}{excess_today*100:.2f}%" if bench else ""))
        A("")
        A(f"*Cumulative P&L: **{_sign(cumulative_pnl)}{_usd(cumulative_pnl)}** "
          f"({_sign(total_return)}{total_return*100:.2f}% total return)*")
        A("")

        # ── Overnight P&L attribution ─────────────────────────────────
        if pos_pnl:
            A("### 📊 Price Movements (overnight P&L attribution)")
            A("")
            A("| Ticker | Shares | Open | Close | Change | Day P&L |")
            A("|--------|--------|------|-------|--------|---------|")
            sorted_pnl = sorted(pos_pnl.items(), key=lambda x: x[1].get("pnl", 0))
            # Show worst N/2 losers + best N/2 winners, collapse middle
            half = max(1, topk // 2)
            losers  = sorted_pnl[:half]
            winners = sorted_pnl[-half:]
            middle  = sorted_pnl[half:-half] if len(sorted_pnl) > topk else []

            def _pnl_row(ticker: str, info: Dict) -> str:
                p0   = float(info.get("price_start") or 0)
                p1   = float(info.get("price_end")   or 0)
                chg  = (p1 - p0) / p0 if p0 else 0.0
                pval = float(info.get("pnl") or 0)
                icon = "📈" if pval > 0 else "📉" if pval < 0 else "→"
                return (f"| {icon} {ticker} | {info.get('shares',0):,} | "
                        f"{_usd(p0)} | {_usd(p1)} | "
                        f"{_sign(chg)}{chg*100:.2f}% | {_sign(pval)}{_usd(pval)} |")

            for ticker, info in losers:
                A(_pnl_row(ticker, info))
            if middle:
                mid_pnl = sum(float(v.get("pnl", 0)) for _, v in middle)
                mid_tks = ", ".join(t for t, _ in middle)
                A(f"| *({len(middle)} more)* | *{mid_tks}* | — | — | — | "
                  f"{_sign(mid_pnl)}{_usd(mid_pnl)} |")
            for ticker, info in winners:
                A(_pnl_row(ticker, info))
            A(f"| | | | | **TOTAL** | **{_sign(daily_pnl)}{_usd(daily_pnl)}** |")
            A("")
        elif daily_pnl and prev_pos:
            A(f"### 📊 Price Movements")
            A("")
            A(f"> Daily P&L: **{_sign(daily_pnl)}{_usd(daily_pnl)}** "
              f"(per-position breakdown not available for this date)")
            A("")

        # ── Orders & cash flow ────────────────────────────────────────
        sells = [o for o in orders if o.get("action") == "sell"]
        buys  = [o for o in orders if o.get("action") == "buy"]
        holds = [t for t in target
                 if t in prev_pos
                 and t not in {o["ticker"] for o in orders}]

        if orders:
            A("### 💸 Orders & Cash Flow")
            A("")

            # Sort sells largest-first, buys largest-first (by value)
            def _order_val(o: Dict) -> float:
                t = o.get("ticker", "")
                sh = abs(int(o.get("shares") or 0))
                px = float(o.get("price") or prices.get(t, 0))
                return sh * px

            sells_sorted = sorted(sells, key=_order_val, reverse=True)
            buys_sorted  = sorted(buys,  key=_order_val, reverse=True)

            # Pre-simulate sells → buys to classify each buy as feasible/infeasible.
            # This prevents the waterfall from ever showing negative cash; buys that
            # would overdraw are shown as a collapsed note after the table.
            sim_cash = prev_cash
            for o in sells_sorted:
                sim_cash += _order_val(o)
            feasible_buys: list = []
            infeasible_buys: list = []
            for o in buys_sorted:
                val = _order_val(o)
                if sim_cash >= val:
                    feasible_buys.append(o)
                    sim_cash -= val
                else:
                    infeasible_buys.append(o)

            # Slots: all sells + fill remaining slots with feasible buys (up to topk)
            sell_slots = len(sells_sorted)
            buy_slots  = max(0, topk - sell_slots)
            buys_show  = feasible_buys[:buy_slots]
            buys_hide  = feasible_buys[buy_slots:]

            A("| Flow | Ticker | Shares | Price | Trade Value | Cash Balance |")
            A("|------|--------|--------|-------|-------------|--------------|")

            running_cash = prev_cash
            A(f"| **Opening Cash** | — | — | — | — | **{_usd(running_cash)}** |")

            def _cash_flag(v: float) -> str:
                """Return ⚠️ suffix when cash is negative or below reserve floor."""
                if v < 0:
                    return " ⚠️"
                if account and v < min_cash_pct * account:
                    return " ⚠️"
                return ""

            # SELLs first (generate cash) — always show all sells
            for o in sells_sorted:
                t = o["ticker"]
                sh = abs(int(o.get("shares") or 0))
                px = float(o.get("price") or prices.get(t, 0))
                val = sh * px
                running_cash += val
                in_target = t in target
                icon = "🟡 REDUCE" if in_target else "🔴 SELL ALL"
                A(f"| {icon} | {t} | −{sh:,} | {_usd(px)} | "
                  f"+{_usd(val)} | {_usd(running_cash)}{_cash_flag(running_cash)} |")

            # BUYs (consume cash) — show up to buy_slots
            for o in buys_show:
                t = o["ticker"]
                sh = abs(int(o.get("shares") or 0))
                px = float(o.get("price") or prices.get(t, 0))
                val = sh * px
                running_cash -= val
                is_new = t not in prev_pos
                icon = "🟢 BUY NEW" if is_new else "🔵 ADD"
                A(f"| {icon} | {t} | +{sh:,} | {_usd(px)} | "
                  f"−{_usd(val)} | {_usd(running_cash)}{_cash_flag(running_cash)} |")

            # Collapsed row for hidden feasible buys
            if buys_hide:
                hidden_val = sum(_order_val(o) for o in buys_hide)
                hidden_tickers = ", ".join(o["ticker"] for o in buys_hide)
                running_cash -= hidden_val
                A(f"| *(+{len(buys_hide)} more buys)* | "
                  f"*{hidden_tickers}* | — | — | "
                  f"−{_usd(hidden_val)} | {_usd(running_cash)}{_cash_flag(running_cash)} |")

            closing_flag = _cash_flag(running_cash)
            A(f"| **Closing Cash** | — | — | — | — | "
              f"**{_usd(running_cash)}**{closing_flag} |")
            A("")

            # Note about infeasible buy orders (skipped to prevent negative cash)
            if infeasible_buys:
                total_skipped = sum(_order_val(o) for o in infeasible_buys)
                sk_tickers = ", ".join(o["ticker"] for o in infeasible_buys[:8])
                if len(infeasible_buys) > 8:
                    sk_tickers += f" *(+{len(infeasible_buys) - 8} more)*"
                A(f"> ⛔ **{len(infeasible_buys)} buy order(s) skipped** — insufficient cash "
                  f"({_usd(total_skipped)} needed). Skipped: {sk_tickers}")
                A("")

            # Warning banner when cash is still below floor after feasible trades
            cash_floor = min_cash_pct * account if account else 0.0
            if running_cash < 0:
                A(f"> ⚠️ **Cash alert**: Closing cash ({_usd(running_cash)}) is **negative** "
                  f"— please verify account reconciliation.")
                A("")
            elif cash_floor > 0 and running_cash < cash_floor:
                A(f"> ⚠️ **Cash reserve below {min_cash_pct*100:.0f}% floor**: "
                  f"Closing cash ({_usd(running_cash)}) is below the reserve target "
                  f"({_usd(cash_floor)}).")
                A("")

            prev_cash = running_cash
        else:
            # No trades
            A("### 💸 Orders & Cash Flow")
            A("")
            A(f"> **No rebalancing orders.** All {len(holds)} positions held unchanged.")
            A(f"> Cash: {_usd(prev_cash)}")
            A("")

        if holds:
            # Show only top-topk holds (by position value), collapse rest
            holds_sorted = sorted(
                holds,
                key=lambda t: target.get(t, 0) * prices.get(t, 0),
                reverse=True,
            )
            hold_show = holds_sorted[:topk]
            hold_hide = holds_sorted[topk:]
            hold_str = "  ".join(f"`{t}`" for t in hold_show)
            suffix = (f"  *(+{len(hold_hide)} more: "
                      + ", ".join(hold_hide) + ")*") if hold_hide else ""
            A(f"⏸ **Hold** ({len(holds)} positions): {hold_str}{suffix}")
            A("")

        # ── End-of-day snapshot ───────────────────────────────────────
        # Derive invested from target positions × prices
        invested = sum(target.get(t, 0) * prices.get(t, 0) for t in target)
        eod_cash = cash_eod if cash_eod is not None else (account - invested)

        # Cash reserve metrics
        cash_reserve_pct = eod_cash / account if account else 0.0
        reserve_ok = cash_reserve_pct >= min_cash_pct
        reserve_flag = "" if reserve_ok else " ⚠️"

        A("### 🔒 End of Day")
        A("")
        A("| | Value |")
        A("|---|-------|")
        A(f"| Portfolio (invested) | {_usd(invested)} |")
        A(f"| Cash | {_usd(eod_cash)}{reserve_flag} |")
        A(f"| Cash Reserve | {cash_reserve_pct*100:.1f}%{reserve_flag} |")
        A(f"| **Total Account** | **{_usd(account)}** |")
        A(f"| Cumulative P&L | {_sign(cumulative_pnl)}{_usd(cumulative_pnl)} |")
        A(f"| Positions | {len(target)} |")
        A("")

        # ── Current day: algorithm's top-N picks ──────────────────────
        if is_current and target:
            A(f"### ⭐ Algorithm's Top {topk} Picks (Next Session)")
            A("")
            A("> These are the algorithm's highest-conviction positions at")
            A("> current prices. Orders to rebalance toward this portfolio")
            A("> execute at **next market open**.")
            A("")

            items = sorted(
                [(t, int(sh), float(prices.get(t) or 0))
                 for t, sh in target.items()],
                key=lambda x: -x[1] * x[2]
            )[:topk]

            A("| # | Ticker | Shares | Price | Position Value | Weight |")
            A("|---|--------|--------|-------|----------------|--------|")
            for i, (t, sh, px) in enumerate(items, 1):
                val = sh * px
                wt = val / account * 100 if account else 0.0
                A(f"| {i} | **{t}** | {sh:,} | {_usd(px)} | {_usd(val)} | {wt:.1f}% |")

            total_top = sum(sh * px for _, sh, px in items)
            A(f"| | | | **Top-{topk} total** | **{_usd(total_top)}** | "
              f"**{total_top/account*100 if account else 0:.0f}%** |")
            A("")

            if len(orders) > 0:
                A(f"*{len(orders)} orders will execute at next market open to rebalance "
                  f"toward this portfolio.*")
                A("")

        A("---")
        A("")

        # Advance running state for next block
        prev_account = account
        prev_target = dict(target)

        # Anchor next block's opening cash to actual account state.
        # Use current-day prices as a proxy for today's closing prices
        # to estimate the cash = account - portfolio_value.
        # This prevents compounding errors (e.g. reset days or data gaps).
        if cash_eod is not None:
            # Authoritative cash field present (newer files)
            prev_cash = float(cash_eod)
        elif prices:
            estimated_port = sum(target.get(t, 0) * prices.get(t, 0) for t in target)
            prev_cash = account - estimated_port
        # else: keep prev_cash as computed from the order waterfall

    # ── Summary table ─────────────────────────────────────────────────
    if days:
        first_acct = float(days[0].get("account_value") or initial_capital)
        last_acct  = float(days[-1].get("account_value") or initial_capital)
        total_days = len(days)
        total_ret = (last_acct - initial_capital) / initial_capital if initial_capital else 0.0

        A("## 📈 Summary")
        A("")
        A("| Metric | Value |")
        A("|--------|-------|")
        A(f"| Trading Days | {total_days} |")
        A(f"| Initial Capital | {_usd(initial_capital)} |")
        A(f"| Current Account | {_usd(last_acct)} |")
        A(f"| Total Return | {_sign(total_ret)}{total_ret*100:.2f}% |")
        A(f"| Cumulative P&L | {_sign(cumulative_pnl)}{_usd(cumulative_pnl)} |")
        A("")

    A(f"*Generated by QuantaAlpha · {ts.strftime('%Y-%m-%d %H:%M')} UTC*")
    A("")

    return "\n".join(lines)


# ── Save helpers ──────────────────────────────────────────────────────

def save_chain_report(
    orders_dir: str | Path,
    *,
    output_dir: Optional[str | Path] = None,
    initial_capital: float = 1_000_000.0,
    topk: int = 10,
    min_cash_pct: float = 0.10,
    config_path: Optional[str | Path] = None,
    generated_at: Optional[datetime] = None,
) -> Path:
    """Load all order files, build chain report, save to disk.

    Parameters
    ----------
    orders_dir:
        Directory containing ``pending_orders_*.json`` and ``archive/``.
    output_dir:
        Where to write the report (default: ``{orders_dir}/reports``).
    initial_capital, topk, min_cash_pct:
        Passed to :func:`generate_chain_report`.
    config_path:
        Path to ``live.yaml``.  When provided, algorithm profile and
        backtest metrics are embedded in the report's initial block.

    Returns
    -------
    Path
        Path to the written ``trading_journal.md`` file.
    """
    days = load_all_orders(orders_dir)
    algo_info = load_algo_info(config_path) if config_path else {}
    # config may override capital/topk
    if algo_info.get("capital") and initial_capital == 1_000_000.0:
        initial_capital = float(algo_info["capital"])
    if algo_info.get("topk") and topk == 10:
        topk = int(algo_info["topk"])
    md = generate_chain_report(days, initial_capital=initial_capital,
                               topk=topk, min_cash_pct=min_cash_pct,
                               algo_info=algo_info,
                               generated_at=generated_at)
    out = Path(output_dir or Path(orders_dir) / "reports")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "trading_journal.md"
    path.write_text(md, encoding="utf-8")
    return path


# ── Backward-compat wrappers (kept for any existing callers) ──────────

def generate_report(
    data: Dict[str, Any],
    *,
    ledger: Optional[List[Dict]] = None,
    generated_at: Optional[datetime] = None,
) -> str:
    """Single-day report (backward compat). Wraps generate_chain_report."""
    return generate_chain_report([data], generated_at=generated_at)


def save_report(
    data: Dict[str, Any],
    *,
    ledger: Optional[List[Dict]] = None,
    output_dir: Optional[Path | str] = None,
    generated_at: Optional[datetime] = None,
) -> Path:
    """Single-day save (backward compat). Uses trading_journal.md."""
    md = generate_chain_report([data], generated_at=generated_at)
    out = Path(output_dir or "data/live/reports")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "trading_journal.md"
    path.write_text(md, encoding="utf-8")
    return path


# ── CLI entry point ───────────────────────────────────────────────────

def main() -> None:
    """Generate trading journal from all order files in a directory.

    Usage::

        # Full journal from data/live (default)
        python -m quantaalpha.live.report_md

        # Custom directory + topk
        python -m quantaalpha.live.report_md --orders-dir data/live --topk 10

        # Single file (legacy)
        python -m quantaalpha.live.report_md pending_orders_2026-02-24.json
    """
    import sys

    args = sys.argv[1:]
    orders_dir = "data/live"
    topk = 10
    initial_capital = 1_000_000.0
    single_file: Optional[str] = None

    i = 0
    while i < len(args):
        if args[i] == "--orders-dir" and i + 1 < len(args):
            orders_dir = args[i + 1]; i += 2
        elif args[i] == "--topk" and i + 1 < len(args):
            topk = int(args[i + 1]); i += 2
        elif args[i] == "--initial-capital" and i + 1 < len(args):
            initial_capital = float(args[i + 1]); i += 2
        elif not args[i].startswith("--"):
            single_file = args[i]; i += 1
        else:
            i += 1

    if single_file:
        path = Path(single_file)
        days = [json.loads(path.read_text())]
        orders_dir_path = path.parent
    else:
        days = load_all_orders(orders_dir)
        orders_dir_path = Path(orders_dir)

    md = generate_chain_report(days, initial_capital=initial_capital, topk=topk)

    rpt_dir = orders_dir_path / "reports"
    rpt_dir.mkdir(parents=True, exist_ok=True)
    rpt_path = rpt_dir / "trading_journal.md"
    rpt_path.write_text(md, encoding="utf-8")
    print(f"Report saved: {rpt_path}", file=sys.stderr)
    print(md)


if __name__ == "__main__":
    main()
