"""Interactive Brokers TWS API executor via ib_insync.

Wraps ``ib_insync.IB`` with a minimal interface needed for daily
rebalancing.  Supports ``dry_run=True`` mode for testing and paper
validation without a live TWS connection.

Typical usage::

    with IBKRExecutor.from_config("configs/live.yaml") as executor:
        positions = executor.get_positions()
        prices    = executor.get_latest_prices(list(positions))
        account   = executor.get_account_value()

        for order in pending_orders:
            executor.place_market_order(order["ticker"], order["shares"])

    # All methods are safe to call in dry_run mode — they log the
    # action and return sensible defaults without touching TWS.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field


@dataclass
class TradeResult:
    """Outcome of a single market order submission."""

    ticker: str
    shares: int                # positive = buy, negative = sell
    status: str                # "submitted" | "dry_run" | "error" | "skipped"
    order_id: Optional[int] = None
    price: Optional[float] = None
    error_msg: Optional[str] = None

    @property
    def is_buy(self) -> bool:
        return self.shares > 0

    @property
    def is_sell(self) -> bool:
        return self.shares < 0

    @property
    def notional(self) -> float:
        """Estimated notional value (price × |shares|), or 0.0 if no price."""
        if self.price is None or self.price <= 0:
            return 0.0
        return self.price * abs(self.shares)


# ---------------------------------------------------------------------------
# IBKRExecutor
# ---------------------------------------------------------------------------


class IBKRExecutor:
    """Thin wrapper around ``ib_insync.IB`` for daily rebalancing.

    Parameters
    ----------
    host, port, client_id:
        TWS Gateway connection parameters.
    timeout:
        Seconds to wait for initial connection.
    dry_run:
        When *True* all trade methods log their intent and return
        ``TradeResult(status="dry_run")`` without touching TWS.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
        timeout: int = 30,
        dry_run: bool = True,
    ) -> None:
        self.host = host
        self.port = port
        self.client_id = client_id
        self.timeout = timeout
        self.dry_run = dry_run
        self._ib: Any = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config_path: str | Path) -> "IBKRExecutor":
        """Construct from ``configs/live.yaml`` ibkr section."""
        with open(config_path) as fh:
            cfg = yaml.safe_load(fh).get("ibkr", {})
        return cls(
            host=cfg.get("host", "127.0.0.1"),
            port=int(cfg.get("port", 7497)),
            client_id=int(cfg.get("client_id", 1)),
            timeout=int(cfg.get("timeout", 30)),
            dry_run=bool(cfg.get("dry_run", True)),
        )

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Connect to TWS Gateway.  No-op in dry_run mode."""
        if self.dry_run:
            logger.info("[dry_run] IBKRExecutor.connect() skipped")
            return
        from ib_insync import IB  # pragma: no cover
        self._ib = IB()  # pragma: no cover
        self._ib.connect(self.host, self.port, clientId=self.client_id, timeout=self.timeout)  # pragma: no cover
        logger.info("Connected to TWS %s:%d client=%d", self.host, self.port, self.client_id)  # pragma: no cover

    def disconnect(self) -> None:
        """Disconnect from TWS.  No-op in dry_run mode."""
        if self.dry_run:
            logger.info("[dry_run] IBKRExecutor.disconnect() skipped")
            return
        if self._ib is not None:  # pragma: no cover
            self._ib.disconnect()  # pragma: no cover
            logger.info("Disconnected from TWS")  # pragma: no cover

    @property
    def is_connected(self) -> bool:
        if self.dry_run:
            return True
        return self._ib is not None and self._ib.isConnected()

    def __enter__(self) -> "IBKRExecutor":
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.disconnect()

    # ------------------------------------------------------------------
    # Account / position queries
    # ------------------------------------------------------------------

    def get_account_value(self, currency: str = "USD") -> float:
        """Return net liquidation value in *currency*.

        Returns 1_000_000.0 in dry_run mode.
        """
        if self.dry_run:
            logger.info("[dry_run] get_account_value() → 1_000_000.0")
            return 1_000_000.0
        from ib_insync import AccountValue  # pragma: no cover
        vals = self._ib.accountValues()  # pragma: no cover
        for v in vals:  # pragma: no cover
            if v.tag == "NetLiquidation" and v.currency == currency:  # pragma: no cover
                return float(v.value)  # pragma: no cover
        return 0.0  # pragma: no cover

    def get_positions(self) -> Dict[str, int]:
        """Return ``{ticker: shares}`` for all current equity positions.

        Returns ``{}`` in dry_run mode.
        """
        if self.dry_run:
            logger.info("[dry_run] get_positions() → {}")
            return {}
        positions: Dict[str, int] = {}  # pragma: no cover
        for pos in self._ib.positions():  # pragma: no cover
            if pos.contract.secType == "STK":  # pragma: no cover
                positions[pos.contract.symbol] = int(pos.position)  # pragma: no cover
        return positions  # pragma: no cover

    def get_latest_prices(self, tickers: List[str]) -> Dict[str, float]:
        """Return last trade price for each ticker.

        Uses ``reqMktData`` with snapshot=True (no streaming subscription).
        Returns ``{}`` in dry_run mode.
        """
        if self.dry_run:
            logger.info("[dry_run] get_latest_prices(%s tickers) → {}", len(tickers))
            return {}
        from ib_insync import Stock  # pragma: no cover
        prices: Dict[str, float] = {}  # pragma: no cover
        contracts = [Stock(t, "SMART", "USD") for t in tickers]  # pragma: no cover
        self._ib.qualifyContracts(*contracts)  # pragma: no cover
        tickers_data = self._ib.reqTickers(*contracts)  # pragma: no cover
        for td in tickers_data:  # pragma: no cover
            symbol = td.contract.symbol  # pragma: no cover
            price = td.last or td.close  # pragma: no cover
            if price and price > 0:  # pragma: no cover
                prices[symbol] = float(price)  # pragma: no cover
        return prices  # pragma: no cover

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def place_market_order(self, ticker: str, shares: int) -> TradeResult:
        """Submit a market order.  ``shares > 0`` = buy, ``< 0`` = sell.

        In dry_run mode logs the order and returns status="dry_run".
        """
        if shares == 0:
            return TradeResult(ticker=ticker, shares=0, status="skipped")

        action = "BUY" if shares > 0 else "SELL"
        abs_shares = abs(shares)

        if self.dry_run:
            logger.info(
                "[dry_run] place_market_order: %s %s %d shares",
                action, ticker, abs_shares,
            )
            return TradeResult(ticker=ticker, shares=shares, status="dry_run")

        from ib_insync import Stock, MarketOrder  # pragma: no cover
        contract = Stock(ticker, "SMART", "USD")  # pragma: no cover
        self._ib.qualifyContracts(contract)  # pragma: no cover
        order = MarketOrder(action, abs_shares)  # pragma: no cover
        trade = self._ib.placeOrder(contract, order)  # pragma: no cover
        logger.info(  # pragma: no cover
            "Placed %s %s x%d — orderId=%s",
            action, ticker, abs_shares, trade.order.orderId,
        )
        return TradeResult(  # pragma: no cover
            ticker=ticker,
            shares=shares,
            status="submitted",
            order_id=trade.order.orderId,
        )

    def cancel_open_orders(self) -> int:
        """Cancel all open orders.  Returns number cancelled.

        Returns 0 in dry_run mode.
        """
        if self.dry_run:
            logger.info("[dry_run] cancel_open_orders() → 0")
            return 0
        open_trades = self._ib.openTrades()  # pragma: no cover
        for trade in open_trades:  # pragma: no cover
            self._ib.cancelOrder(trade.order)  # pragma: no cover
        logger.info("Cancelled %d open orders", len(open_trades))  # pragma: no cover
        return len(open_trades)  # pragma: no cover

    def execute_orders(self, orders: List[Dict]) -> List[TradeResult]:
        """Execute a list of order dicts from pending_orders_{date}.json.

        Each dict must have keys: ``ticker``, ``shares``.
        Sells are submitted before buys to free up capital.

        Parameters
        ----------
        orders:
            List of ``{"ticker": str, "shares": int, ...}`` dicts.

        Returns
        -------
        List of :class:`TradeResult` in submission order.
        """
        sells = [o for o in orders if int(o["shares"]) < 0]
        buys  = [o for o in orders if int(o["shares"]) > 0]
        results: List[TradeResult] = []
        for o in sells + buys:
            result = self.place_market_order(o["ticker"], int(o["shares"]))
            results.append(result)
        submitted = sum(1 for r in results if r.status in ("submitted", "dry_run"))
        logger.info("execute_orders: %d orders, %d submitted", len(orders), submitted)
        return results
