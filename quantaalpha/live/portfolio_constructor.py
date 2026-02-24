"""
portfolio_constructor.py — TopkDropout portfolio construction without Qlib dependency.

Given today's model scores and current holdings, produces a target portfolio:
- Hold the top-K scoring stocks
- Allow at most n_drop stocks to be swapped out per rebalance
- Equal-weight target (can be extended to signal-weighted)

Usage:
    from quantaalpha.live.portfolio_constructor import PortfolioConstructor
    pc = PortfolioConstructor(topk=30, n_drop=2, capital=1_000_000)
    orders = pc.rebalance(scores={"AAPL": 0.9, ...}, positions={"MSFT": 500.0, ...}, prices={"AAPL": 220.5, ...})
    # orders: {"AAPL": +45, "TSLA": -100, ...}  shares to buy(+)/sell(-)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


@dataclass
class Order:
    """A single rebalance order."""
    ticker: str
    shares: int      # positive = buy, negative = sell
    price: float     # last price for reference
    action: str      # "buy" | "sell" | "hold"
    reason: str = ""

    @property
    def notional(self) -> float:
        return abs(self.shares) * self.price


@dataclass
class RebalanceResult:
    """Output of a rebalance call."""
    date: str
    target_portfolio: Dict[str, int]   # ticker → target shares
    orders: List[Order]
    current_holdings: Dict[str, int]   # ticker → current shares (input)
    capital: float
    topk: int
    n_drop: int
    notes: List[str] = field(default_factory=list)

    @property
    def buy_orders(self) -> List[Order]:
        return [o for o in self.orders if o.action == "buy"]

    @property
    def sell_orders(self) -> List[Order]:
        return [o for o in self.orders if o.action == "sell"]

    @property
    def total_buys(self) -> float:
        return sum(o.notional for o in self.buy_orders)

    @property
    def total_sells(self) -> float:
        return sum(o.notional for o in self.sell_orders)


class PortfolioConstructor:
    """
    TopkDropout portfolio construction (mirrors Qlib's TopkDropoutStrategy).

    Parameters
    ----------
    topk : int
        Target number of stocks to hold (default 30).
    n_drop : int
        Max stocks swapped per rebalance (default 2).
    capital : float
        Total portfolio capital in USD (default 1_000_000).
    equal_weight : bool
        If True, target equal weight (1/topk per stock). Default True.
    min_order_shares : int
        Minimum shares per order to avoid fractional/tiny orders. Default 1.
    """

    def __init__(
        self,
        topk: int = 30,
        n_drop: int = 2,
        capital: float = 1_000_000.0,
        equal_weight: bool = True,
        min_order_shares: int = 1,
        max_position_pct: float = 0.05,
        min_adv: float = 0.0,
    ):
        self.topk = topk
        self.n_drop = n_drop
        self.capital = capital
        self.equal_weight = equal_weight
        self.min_order_shares = min_order_shares
        self.max_position_pct = max_position_pct  # max single position as fraction of capital
        self.min_adv = min_adv  # minimum average daily volume ($); 0 = disabled

    # ------------------------------------------------------------------
    # Core algorithm
    # ------------------------------------------------------------------

    def _select_topk_with_dropout(
        self,
        scores: Dict[str, float],
        current_holdings: List[str],
    ) -> Tuple[List[str], List[str]]:
        """
        Apply TopkDropout selection:
        1. Sort all stocks by score descending.
        2. Keep current holdings that are in the top-K (they stay in).
        3. Drop at most n_drop holdings with the lowest scores.
        4. Fill gaps with the highest-scoring stocks not already held.

        Returns: (target_list, dropped_tickers)
        """
        sorted_all = sorted(scores.keys(), key=lambda t: scores.get(t, float("-inf")), reverse=True)
        top_k_universe = sorted_all[: self.topk]

        held_in_top = [t for t in current_holdings if t in set(top_k_universe)]
        held_outside = [t for t in current_holdings if t not in set(top_k_universe)]

        # Sort held-outside by score ascending (worst first), drop up to n_drop
        held_outside_sorted = sorted(held_outside, key=lambda t: scores.get(t, float("-inf")))
        to_drop = held_outside_sorted[: self.n_drop]
        kept_outside = held_outside_sorted[self.n_drop :]

        current_target = set(held_in_top) | set(kept_outside)
        candidates = [t for t in top_k_universe if t not in current_target]
        to_add = candidates[: max(0, self.topk - len(current_target))]

        target = sorted(current_target | set(to_add), key=lambda t: scores.get(t, 0.0), reverse=True)

        # Bug fix #4: respect n_drop as max total exits per day.
        # If target[:topk] would silently drop positions beyond n_drop
        # (e.g. topk decreased), keep extra positions temporarily.
        if len(target) > self.topk:
            would_be_dropped = set(current_holdings) - set(target[: self.topk])
            if len(would_be_dropped) > self.n_drop:
                # Only drop n_drop; keep rest even if exceeds topk temporarily
                excess_kept = sorted(
                    would_be_dropped - set(to_drop),
                    key=lambda t: scores.get(t, float("-inf")),
                    reverse=True,
                )
                max_extra_drops = max(0, self.n_drop - len(to_drop))
                extra_drops = excess_kept[len(excess_kept) - max_extra_drops :] if max_extra_drops else []
                to_drop = to_drop + extra_drops
                # Rebuild target keeping everything except actual drops
                kept = set(target) - set(to_drop)
                target = sorted(kept, key=lambda t: scores.get(t, 0.0), reverse=True)
                log.info("TopkDropout: holding %d positions (> topk=%d) to respect n_drop=%d",
                         len(target), self.topk, self.n_drop)
                return target, to_drop

        return target[: self.topk], to_drop

    def _apply_liquidity_filter(
        self,
        scores: Dict[str, float],
        adv: Dict[str, float],
    ) -> Dict[str, float]:
        """Remove tickers whose average daily volume (USD) is below min_adv."""
        if self.min_adv <= 0 or not adv:
            return scores
        filtered = {t: s for t, s in scores.items() if adv.get(t, 0.0) >= self.min_adv}
        removed = len(scores) - len(filtered)
        if removed:
            log.info("Liquidity filter: removed %d tickers with ADV < $%.0fM",
                     removed, self.min_adv / 1_000_000)
        return filtered

    def _compute_target_shares(
        self,
        target: List[str],
        prices: Dict[str, float],
    ) -> Dict[str, int]:
        """Compute integer share counts with equal-weight + position cap."""
        if not target:
            return {}
        weight = 1.0 / len(target) if self.equal_weight else 1.0 / len(target)
        # Apply position cap: cap weight at max_position_pct
        if self.max_position_pct > 0:
            weight = min(weight, self.max_position_pct)
        target_shares = {}
        for ticker in target:
            price = prices.get(ticker)
            if not price or price <= 0:
                log.warning("No price for %s — skipping", ticker)
                continue
            shares = int(self.capital * weight / price)
            if shares >= self.min_order_shares:
                target_shares[ticker] = shares
        return target_shares

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rebalance(
        self,
        scores: Dict[str, float],
        positions: Dict[str, float],
        prices: Dict[str, float],
        as_of_date: Optional[str] = None,
        adv: Optional[Dict[str, float]] = None,
    ) -> RebalanceResult:
        """
        Compute today's rebalance orders.

        Args:
            scores:    model scores {ticker: float} (higher = stronger buy)
            positions: current holdings {ticker: shares_or_notional_float}
            prices:    current prices {ticker: float} (close or mid)
            as_of_date: ISO date string for logging
            adv:       average daily volume in USD {ticker: float}; used for
                       liquidity filtering when min_adv > 0

        Returns:
            RebalanceResult with orders list and target portfolio.
        """
        import datetime
        date_str = as_of_date or str(datetime.date.today())

        # Apply liquidity filter before TopkDropout selection
        filtered_scores = self._apply_liquidity_filter(scores, adv or {})

        # Normalize positions to integer shares (may arrive as notional floats)
        current_holdings_int: Dict[str, int] = {}
        for ticker, val in positions.items():
            price = prices.get(ticker, 1.0)
            shares = int(val) if val > 1 else int(val / price) if price > 0 else 0
            if shares > 0:
                current_holdings_int[ticker] = shares

        current_list = list(current_holdings_int.keys())
        target_list, dropped = self._select_topk_with_dropout(filtered_scores, current_list)
        target_shares = self._compute_target_shares(target_list, prices)

        orders: List[Order] = []

        # Sell: positions being dropped or reduced
        for ticker in current_list:
            current_sh = current_holdings_int[ticker]
            target_sh = target_shares.get(ticker, 0)
            delta = target_sh - current_sh
            if delta < -self.min_order_shares:
                orders.append(Order(
                    ticker=ticker, shares=delta,
                    price=prices.get(ticker, 0.0),
                    action="sell",
                    reason="dropped" if ticker in dropped else "reweight",
                ))
            elif delta == 0:
                pass  # hold
            elif delta > self.min_order_shares:
                orders.append(Order(
                    ticker=ticker, shares=delta,
                    price=prices.get(ticker, 0.0),
                    action="buy",
                    reason="reweight",
                ))

        # Buy: new positions not held before
        for ticker in target_list:
            if ticker not in current_holdings_int:
                target_sh = target_shares.get(ticker, 0)
                if target_sh >= self.min_order_shares:
                    orders.append(Order(
                        ticker=ticker, shares=target_sh,
                        price=prices.get(ticker, 0.0),
                        action="buy",
                        reason="new entry",
                    ))

        notes = []
        if dropped:
            notes.append(f"Dropped (n_drop={self.n_drop}): {dropped}")
        notes.append(f"Target size: {len(target_shares)} / {self.topk}")

        log.info("[%s] Target: %d positions, %d orders (%d buys, %d sells)",
                 date_str, len(target_shares), len(orders),
                 sum(1 for o in orders if o.action == "buy"),
                 sum(1 for o in orders if o.action == "sell"))

        return RebalanceResult(
            date=date_str,
            target_portfolio=target_shares,
            orders=orders,
            current_holdings=current_holdings_int,
            capital=self.capital,
            topk=self.topk,
            n_drop=self.n_drop,
            notes=notes,
        )

    def print_summary(self, result: RebalanceResult) -> None:
        """Print a human-readable rebalance summary."""
        print(f"\n=== Rebalance Summary [{result.date}] ===")
        print(f"Capital: ${result.capital:,.0f}  TopK={result.topk}  n_drop={result.n_drop}")
        print(f"Target: {len(result.target_portfolio)} positions")
        print(f"Buys:   {len(result.buy_orders)}  total ~${result.total_buys:,.0f}")
        print(f"Sells:  {len(result.sell_orders)} total ~${result.total_sells:,.0f}")
        for note in result.notes:
            print(f"  NOTE: {note}")
        if result.orders:
            print("\nOrders:")
            for o in sorted(result.orders, key=lambda x: x.action):
                sign = "+" if o.shares > 0 else ""
                print(f"  {o.action.upper():4s}  {o.ticker:<8s} {sign}{o.shares:+6d} sh @ ${o.price:8.2f}"
                      f"  ~${o.notional:,.0f}  [{o.reason}]")
