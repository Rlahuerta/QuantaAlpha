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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── Portfolio tracker ─────────────────────────────────────────────────

@dataclass
class _Pos:
    """One live position."""
    shares: int
    price:  float   # most-recent known price


class Portfolio:
    """Hard-capped position tracker — the single source of truth for what is held.

    Rules enforced internally:
      • Never holds more than ``topk`` distinct tickers.
      • ADD (existing ticker) never consumes a slot.
      • BUY NEW (new ticker) requires a free slot; returns CAP_BLOCKED if full.
      • SELL ALL removes the ticker and frees the slot.
      • REDUCE keeps the ticker; slot stays occupied.
    """

    def __init__(self, topk: int) -> None:
        self.topk = topk
        self._pos: Dict[str, _Pos] = {}

    # ── read ──────────────────────────────────────────────────────────
    def __contains__(self, ticker: str) -> bool:
        return ticker in self._pos

    def __len__(self) -> int:
        return len(self._pos)

    @property
    def tickers(self) -> List[str]:
        return list(self._pos.keys())

    def shares(self, ticker: str) -> int:
        return self._pos[ticker].shares if ticker in self._pos else 0

    def price(self, ticker: str) -> float:
        return self._pos[ticker].price if ticker in self._pos else 0.0

    def free_slots(self) -> int:
        return max(0, self.topk - len(self._pos))

    def value(self) -> float:
        return sum(p.shares * p.price for p in self._pos.values())

    # ── write ─────────────────────────────────────────────────────────
    def update_price(self, ticker: str, price: float) -> None:
        if ticker in self._pos and price:
            self._pos[ticker].price = price

    def execute_sell(self, ticker: str, shares: int, price: float,
                     in_target: bool) -> str:
        """Process a sell order.

        Returns one of:
          ``'SELL_ALL'``    ticker fully exited — slot freed.
          ``'REDUCE'``      shares reduced but ticker stays — slot kept.
          ``'INVALID'``     ticker not held — no cash generated, ignored.
        """
        if ticker not in self._pos:
            return "INVALID"
        pos = self._pos[ticker]
        if price:
            pos.price = price
        remaining = pos.shares - shares
        if in_target and remaining > 0:
            pos.shares = remaining
            return "REDUCE"
        del self._pos[ticker]
        return "SELL_ALL"

    def execute_buy(self, ticker: str, shares: int, price: float) -> str:
        """Process a buy order.

        Returns one of:
          ``'ADD'``          shares added to existing position — no slot used.
          ``'BUY_NEW'``      new position opened — slot consumed.
          ``'CAP_BLOCKED'``  no free slots for a new ticker — order skipped.
        """
        if ticker in self._pos:
            self._pos[ticker].shares += shares
            if price:
                self._pos[ticker].price = price
            return "ADD"
        if self.free_slots() == 0:
            return "CAP_BLOCKED"
        self._pos[ticker] = _Pos(shares=shares, price=price)
        return "BUY_NEW"


# ── Formatting helpers ────────────────────────────────────────────────

def _usd(v: float, signed: bool = False) -> str:
    sign = ("+" if v >= 0 else "-") if signed else ("-" if v < 0 else "")
    return f"{sign}${abs(v):,.2f}"


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
    prev_cash    = initial_capital
    portfolio    = Portfolio(topk)   # single source of truth for held positions
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
        # Enforce portfolio size cap: keep only the topk highest-weight positions
        if len(target) > topk:
            target = dict(sorted(target.items(), key=lambda kv: kv[1], reverse=True)[:topk])
        # portfolio is the single source of truth — never overridden from file
        prices: Dict[str, float] = dict(day.get("prices") or {})
        pos_pnl: Dict[str, Dict] = day.get("position_pnl") or {}
        bench = float(day.get("benchmark_return") or 0.0)
        cum_excess = float(day.get("cumulative_excess_return") or 0.0)
        # NOTE: day.get("cash") is the *real* brokerage account cash for 22+ positions.
        # Our simulation tracks only topk positions, so we never use it for cash
        # accounting — cash is always derived from the waterfall chain.

        # Backfill prices from order dicts (always available)
        for o in orders:
            t = o.get("ticker", "")
            px = float(o.get("price") or 0.0)
            if t and px and t not in prices:
                prices[t] = px

        # Seed portfolio from file's previous_positions ONLY when the portfolio
        # is still empty (first day of tracking). Once we have chain-tracked
        # positions, the file's snapshot is ignored — portfolio is the truth.
        if len(portfolio) == 0:
            file_pp = day.get("previous_positions") or {}
            if file_pp:
                for t, sh in sorted(file_pp.items(),
                                    key=lambda kv: kv[1], reverse=True)[:topk]:
                    portfolio.execute_buy(t, int(sh), prices.get(t, 0.0))

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

        # ── Overnight P&L attribution (current day only) ──────────────
        # Past blocks already capture the move in their account → account header;
        # showing a full breakdown there adds noise without new information.
        if is_current and pos_pnl:
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
        elif is_current and daily_pnl and portfolio.tickers:
            A(f"### 📊 Price Movements")
            A("")
            A(f"> Daily P&L: **{_sign(daily_pnl)}{_usd(daily_pnl)}** "
              f"(per-position breakdown not available for this date)")
            A("")

        # ── Orders & cash flow ────────────────────────────────────────
        sells = [o for o in orders if o.get("action") == "sell"]
        buys  = [o for o in orders if o.get("action") == "buy"]

        # Holds = in target AND held AND NOT part of executed trades.
        # Use executed_tickers (filled below) rather than raw orders so that
        # infeasible buys and invalid sells don't mask held positions.

        if orders:
            A("### 💸 Orders & Cash Flow")
            A("")

            def _order_val(o: Dict) -> float:
                t = o.get("ticker", "")
                sh = abs(int(o.get("shares") or 0))
                px = float(o.get("price") or prices.get(t, 0))
                return sh * px

            def _cash_flag(v: float) -> str:
                if v < 0:
                    return " ⚠️"
                if account and v < min_cash_pct * account:
                    return " ⚠️"
                return ""

            sells_sorted = sorted(sells, key=_order_val, reverse=True)
            buys_sorted  = sorted(buys,  key=_order_val, reverse=True)

            # ── Pre-simulation: classify every order before rendering ──
            # Pass 1 — sells: determine valid/invalid and post-sell cash/slots
            sim_valid_sells:   list = []
            sim_invalid_sells: list = []
            sim_cash = prev_cash
            sim_free = portfolio.free_slots()

            for o in sells_sorted:
                t  = o.get("ticker", "")
                sh = abs(int(o.get("shares") or 0))
                px = float(o.get("price") or prices.get(t, 0))
                if t not in portfolio:
                    sim_invalid_sells.append(o)
                    continue
                sim_valid_sells.append(o)
                sim_cash += sh * px
                # SELL ALL (ticker not in new target) frees a slot
                remaining = portfolio.shares(t) - sh
                if not (t in target and remaining > 0):
                    sim_free += 1

            # Pass 2 — buys: classify against cash AND portfolio cap
            sim_feasible:   list = []
            sim_infeasible: list = []  # insufficient cash
            sim_cap_blocked: list = []  # would exceed topk

            for o in buys_sorted:
                t   = o.get("ticker", "")
                val = _order_val(o)
                is_new = t not in portfolio
                if is_new and sim_free == 0:
                    sim_cap_blocked.append(o)
                elif sim_cash < val:
                    sim_infeasible.append(o)
                else:
                    sim_feasible.append(o)
                    sim_cash -= val
                    if is_new:
                        sim_free -= 1

            # ── Render waterfall ──────────────────────────────────────
            A("| Flow | Ticker | Shares | Price | Trade Value | Cash Balance |")
            A("|------|--------|--------|-------|-------------|--------------|")

            running_cash = prev_cash
            A(f"| **Opening Cash** | — | — | — | — | **{_usd(running_cash)}** |")

            # Sells (generate cash) — execute on portfolio
            executed_tickers: set = set()
            for o in sim_valid_sells:
                t  = o["ticker"]
                sh = abs(int(o.get("shares") or 0))
                px = float(o.get("price") or prices.get(t, 0))
                val = sh * px
                running_cash += val
                result = portfolio.execute_sell(t, sh, px, t in target)
                icon = "🟡 REDUCE" if result == "REDUCE" else "🔴 SELL ALL"
                A(f"| {icon} | {t} | −{sh:,} | {_usd(px)} | "
                  f"+{_usd(val)} | {_usd(running_cash)}{_cash_flag(running_cash)} |")
                executed_tickers.add(t)

            # Buys (consume cash) — execute on portfolio
            for o in sim_feasible:
                t  = o["ticker"]
                sh = abs(int(o.get("shares") or 0))
                px = float(o.get("price") or prices.get(t, 0))
                val = sh * px
                running_cash -= val
                result = portfolio.execute_buy(t, sh, px)
                icon = "🟢 BUY NEW" if result == "BUY_NEW" else "🔵 ADD"
                A(f"| {icon} | {t} | +{sh:,} | {_usd(px)} | "
                  f"−{_usd(val)} | {_usd(running_cash)}{_cash_flag(running_cash)} |")
                executed_tickers.add(t)

            closing_flag = _cash_flag(running_cash)
            A(f"| **Closing Cash** | — | — | — | — | "
              f"**{_usd(running_cash)}**{closing_flag} |")
            A("")

            # ── Skip notes ────────────────────────────────────────────
            if sim_invalid_sells:
                total_inv = sum(_order_val(o) for o in sim_invalid_sells)
                inv_tks   = ", ".join(o["ticker"] for o in sim_invalid_sells[:8])
                if len(sim_invalid_sells) > 8:
                    inv_tks += f" *(+{len(sim_invalid_sells) - 8} more)*"
                A(f"> ⛔ **{len(sim_invalid_sells)} sell order(s) skipped** — not in portfolio "
                  f"({_usd(total_inv)} notional). Skipped: {inv_tks}")
                A("")

            if sim_infeasible:
                total_inf = sum(_order_val(o) for o in sim_infeasible)
                inf_tks   = ", ".join(o["ticker"] for o in sim_infeasible[:8])
                if len(sim_infeasible) > 8:
                    inf_tks += f" *(+{len(sim_infeasible) - 8} more)*"
                A(f"> ⛔ **{len(sim_infeasible)} buy order(s) skipped** — insufficient cash "
                  f"({_usd(total_inf)} needed). Skipped: {inf_tks}")
                A("")

            if sim_cap_blocked:
                cap_tks = ", ".join(o["ticker"] for o in sim_cap_blocked[:8])
                if len(sim_cap_blocked) > 8:
                    cap_tks += f" *(+{len(sim_cap_blocked) - 8} more)*"
                A(f"> ⛔ **{len(sim_cap_blocked)} buy order(s) skipped** — portfolio cap "
                  f"({topk} positions) reached. Skipped: {cap_tks}")
                A("")

            # Warning banner
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

            # Holds = tickers still in portfolio that weren't explicitly traded today
            holds = [t for t in portfolio.tickers if t not in executed_tickers]
            prev_cash = running_cash

        else:
            # No orders — portfolio unchanged
            holds = list(portfolio.tickers)
            A("### 💸 Orders & Cash Flow")
            A("")
            A(f"> **No rebalancing orders.** All {len(holds)} positions held unchanged.")
            A(f"> Cash: {_usd(prev_cash)}")
            A("")

        if holds:
            holds_sorted = sorted(
                holds,
                key=lambda t: portfolio.shares(t) * (prices.get(t) or portfolio.price(t)),
                reverse=True,
            )
            hold_str = "  ".join(f"`{t}`" for t in holds_sorted)
            A(f"⏸ **Hold** ({len(holds)} positions): {hold_str}")
            A("")

        # ── End-of-day snapshot ───────────────────────────────────────
        # Cash is ALWAYS the waterfall closing cash (prev_cash after any
        # executed trades).  The file's cash field is the real brokerage
        # account (22+ positions) and is never used here.
        # Identity: account = invested + cash  →  invested = account − cash
        eod_cash = prev_cash
        invested = account - eod_cash

        # Cash reserve metrics
        cash_reserve_pct = eod_cash / account if account else 0.0
        reserve_ok = cash_reserve_pct >= min_cash_pct
        reserve_flag = "" if reserve_ok else " ⚠️"

        # Return vs Day 0: (account − initial_capital) / initial_capital
        return_vs_d0 = (account - initial_capital) / initial_capital if initial_capital else 0.0
        return_flag  = "📈" if return_vs_d0 >= 0 else "📉"

        A("### 🔒 End of Day")
        A("")
        A("| | Value |")
        A("|---|-------|")
        A(f"| Portfolio (invested) | {_usd(invested)} |")
        A(f"| Cash | {_usd(eod_cash)}{reserve_flag} |")
        A(f"| Cash Reserve | {cash_reserve_pct*100:.1f}%{reserve_flag} |")
        A(f"| **Total Account** | **{_usd(account)}** |")
        A(f"| vs Day 0 ($1M) | {return_flag} **{_pct(return_vs_d0, signed=True)}** ({_usd(account - initial_capital, signed=True)}) |")
        A(f"| Cumulative P&L | {_sign(cumulative_pnl)}{_usd(cumulative_pnl)} |")
        A(f"| Positions | {len(portfolio)} |")
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

        # Update portfolio prices from today's data (for value estimates)
        for t, px in prices.items():
            portfolio.update_price(t, px)
        # prev_cash carries naturally from waterfall — no override needed.
        # Cash never changes from price movements, only from executed trades.

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
