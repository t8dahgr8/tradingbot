"""
Simulated execution against the real order book.

The single biggest way a paper trader lies to you is by assuming you get filled at
the price you saw. This broker refuses to do that. Specifically:

  * Orders do not execute against the book you were looking at when you decided.
    They are held for `latency_ms` and executed against the next book that arrives.
    In a fast-moving in-play market this is where most of your theoretical edge
    goes to die, and you want to find that out now rather than with real money.

  * Marketable orders walk the book level by level and pay the real VWAP. If the
    depth is not there, you get a partial fill, not a miracle.

  * Passive orders sit in a queue. They only fill once enough volume has traded
    at or through your price to clear the shares that were resting ahead of you.
    This is the difference between "I would have earned the spread" and "I would
    have been picked off".

  * Fees are charged on the taker side using Polymarket's fee shape.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field

from ..models import (
    Fill,
    Level,
    Order,
    OrderBook,
    OrderStatus,
    OrderType,
    Side,
    now_ms,
)

log = logging.getLogger(__name__)


@dataclass
class BrokerConfig:
    # Round trip from "strategy decided" to "order rests on the book". 250ms is a
    # realistic starting point for a retail connection to Polymarket.
    latency_ms: int = 250
    # Extra random jitter on top of the base latency.
    latency_jitter_ms: int = 100
    # Taker fee rate. Polymarket sports markets currently run with fees disabled,
    # but do not build a strategy that only works at zero fees.
    taker_fee_rate: float = 0.0
    maker_fee_rate: float = 0.0
    # Refuse to take more than this share of any single price level, on the theory
    # that other traders are competing for the same liquidity.
    max_level_participation: float = 0.5
    # Probability a marketable order simply misses because the level vanished.
    miss_probability: float = 0.05
    # Passive orders are cancelled if unfilled after this long.
    passive_ttl_ms: int = 30_000
    # Fraction of volume trading at our price that we assume is ahead of us.
    queue_decay: float = 1.0
    seed: int | None = 7


def polymarket_fee(price: float, shares: float, rate: float) -> float:
    """Polymarket's fee shape: proportional to the cheaper side of the market.

    A trade at 0.02 is charged far less than a trade at 0.50, which is why
    long-shot scalping looks cheaper than it is on a percentage basis.
    """
    if rate <= 0:
        return 0.0
    return rate * min(price, 1.0 - price) * shares


class PaperBroker:
    """Simulated exchange. Feed it books and trades; it produces fills."""

    def __init__(self, config: BrokerConfig | None = None):
        self.cfg = config or BrokerConfig()
        self._rng = random.Random(self.cfg.seed)
        self.pending: list[Order] = []      # submitted, not yet "arrived"
        self.resting: list[Order] = []      # live passive orders on the book
        self.fills: list[Fill] = []
        self._books: dict[str, OrderBook] = {}
        self._arrival: dict[str, int] = {}  # order_id -> arrival timestamp

    # -- plumbing ----------------------------------------------------------

    def book(self, token_id: str) -> OrderBook | None:
        return self._books.get(token_id)

    def _latency(self) -> int:
        j = self.cfg.latency_jitter_ms
        return self.cfg.latency_ms + (self._rng.randint(0, j) if j > 0 else 0)

    # -- order entry -------------------------------------------------------

    def submit(self, order: Order, ts_ms: int | None = None) -> Order:
        """Accept an order. It will not execute until latency has elapsed."""
        ts = ts_ms or now_ms()
        book = self._books.get(order.token_id)

        if order.size <= 0:
            order.status = OrderStatus.REJECTED
            order.reason = "non-positive size"
            return order

        # Round the limit to a valid tick, conservatively (never improve our price).
        tick = book.tick_size if book else 0.01
        if tick > 0:
            steps = order.limit_price / tick
            order.limit_price = round(
                (int(steps) + (1 if order.side == Side.BUY and steps % 1 else 0)) * tick, 6
            )
        order.limit_price = min(max(order.limit_price, tick), 1.0 - tick)

        if order.order_type == OrderType.PASSIVE and book is not None:
            levels = book.bids if order.side == Side.BUY else book.asks
            ahead = sum(l.size for l in levels if abs(l.price - order.limit_price) < 1e-9)
            order.queue_ahead = ahead

        self._arrival[order.order_id] = ts + self._latency()
        self.pending.append(order)
        return order

    def cancel(self, order_id: str) -> bool:
        for bucket in (self.pending, self.resting):
            for o in list(bucket):
                if o.order_id == order_id and o.is_live:
                    o.status = OrderStatus.CANCELLED
                    bucket.remove(o)
                    return True
        return False

    def cancel_all(self, token_id: str | None = None) -> int:
        n = 0
        for bucket in (self.pending, self.resting):
            for o in list(bucket):
                if token_id is None or o.token_id == token_id:
                    o.status = OrderStatus.CANCELLED
                    bucket.remove(o)
                    n += 1
        return n

    # -- market data -------------------------------------------------------

    def on_book(self, book: OrderBook, ts_ms: int | None = None) -> list[Fill]:
        """Ingest a book update and execute anything whose latency has elapsed."""
        ts = ts_ms or now_ms()
        self._books[book.token_id] = book
        out: list[Fill] = []

        # Promote pending orders that have "arrived".
        for o in list(self.pending):
            if self._arrival.get(o.order_id, 0) > ts:
                continue
            self.pending.remove(o)
            if o.order_type == OrderType.MARKETABLE:
                out.extend(self._execute_marketable(o, ts))
            else:
                # Re-derive queue position against the book as it is on arrival,
                # not as it was when we decided.
                b = self._books.get(o.token_id)
                if b is not None:
                    levels = b.bids if o.side == Side.BUY else b.asks
                    o.queue_ahead = sum(
                        l.size for l in levels if abs(l.price - o.limit_price) < 1e-9
                    )
                self.resting.append(o)

        # Expire stale passive orders.
        for o in list(self.resting):
            if ts - o.created_ms > self.cfg.passive_ttl_ms:
                o.status = OrderStatus.CANCELLED
                self.resting.remove(o)

        # A resting order also fills if the book crosses it outright.
        out.extend(self._sweep_crossed(book, ts))
        self.fills.extend(out)
        return out

    def on_trade(self, token_id: str, price: float, size: float, ts_ms: int | None = None
                 ) -> list[Fill]:
        """Ingest a public trade print and advance passive queue positions."""
        ts = ts_ms or now_ms()
        out: list[Fill] = []

        for o in list(self.resting):
            if o.token_id != token_id or not o.is_live:
                continue
            # A trade only helps us if it happened at or through our price.
            helps = (o.side == Side.BUY and price <= o.limit_price + 1e-9) or (
                o.side == Side.SELL and price >= o.limit_price - 1e-9
            )
            if not helps:
                continue

            vol = size * self.cfg.queue_decay
            if o.queue_ahead > 0:
                consumed = min(o.queue_ahead, vol)
                o.queue_ahead -= consumed
                vol -= consumed
            if vol <= 0:
                continue

            qty = min(vol, o.remaining)
            if qty <= 0:
                continue
            fee = polymarket_fee(o.limit_price, qty, self.cfg.maker_fee_rate)
            f = Fill(o.order_id, o.token_id, o.side, o.limit_price, qty, fee, ts, "maker")
            o.apply(f)
            out.append(f)
            if not o.is_live:
                self.resting.remove(o)

        self.fills.extend(out)
        return out

    # -- execution internals ----------------------------------------------

    def _execute_marketable(self, order: Order, ts: int) -> list[Fill]:
        book = self._books.get(order.token_id)
        if book is None or not book.is_valid():
            order.status = OrderStatus.REJECTED
            order.reason = "no valid book on arrival"
            return []

        if self._rng.random() < self.cfg.miss_probability:
            order.status = OrderStatus.CANCELLED
            order.reason = "missed: liquidity moved before arrival"
            return []

        levels = list(book.asks if order.side == Side.BUY else book.bids)
        remaining = order.remaining
        notional = 0.0
        got = 0.0

        for lv in levels:
            if remaining <= 1e-9:
                break
            # Never pay through our limit. This is what stops a thin book from
            # filling us at a catastrophic price.
            if order.side == Side.BUY and lv.price > order.limit_price + 1e-9:
                break
            if order.side == Side.SELL and lv.price < order.limit_price - 1e-9:
                break
            available = lv.size * self.cfg.max_level_participation
            take = min(remaining, available)
            if take <= 0:
                continue
            notional += take * lv.price
            got += take
            remaining -= take

        if got <= 1e-9:
            order.status = OrderStatus.CANCELLED
            order.reason = "no fillable depth inside limit"
            return []

        avg = notional / got
        fee = polymarket_fee(avg, got, self.cfg.taker_fee_rate)
        f = Fill(order.order_id, order.token_id, order.side, avg, got, fee, ts, "taker")
        order.apply(f)
        if order.is_live:
            # Unfilled remainder does not linger as a hidden resting order.
            order.status = OrderStatus.PARTIAL
            order.reason = "partial: insufficient depth"
        return [f]

    def _sweep_crossed(self, book: OrderBook, ts: int) -> list[Fill]:
        """Fill resting orders that the market has traded straight through."""
        out: list[Fill] = []
        for o in list(self.resting):
            if o.token_id != book.token_id or not o.is_live:
                continue
            crossed = False
            if o.side == Side.BUY and book.best_ask is not None:
                crossed = book.best_ask <= o.limit_price - 1e-9
            elif o.side == Side.SELL and book.best_bid is not None:
                crossed = book.best_bid >= o.limit_price + 1e-9
            if not crossed:
                continue
            # Being crossed means the market moved against us and we are about to
            # be filled at our (now stale) price. That is adverse selection, and
            # it should show up in the P&L.
            qty = o.remaining
            fee = polymarket_fee(o.limit_price, qty, self.cfg.maker_fee_rate)
            f = Fill(o.order_id, o.token_id, o.side, o.limit_price, qty, fee, ts, "maker")
            o.apply(f)
            out.append(f)
            self.resting.remove(o)
        return out

    # -- helpers -----------------------------------------------------------

    def expected_cost(self, token_id: str, side: Side, size: float) -> tuple[float, float]:
        """VWAP and fillable size for a hypothetical marketable order, right now.

        Use this before sizing so the strategy knows what it would actually pay,
        not what the top of book says.
        """
        book = self._books.get(token_id)
        if book is None or not book.is_valid():
            return (0.0, 0.0)
        levels = book.asks if side == Side.BUY else book.bids
        remaining, notional, got = size, 0.0, 0.0
        for lv in levels:
            if remaining <= 1e-9:
                break
            take = min(remaining, lv.size * self.cfg.max_level_participation)
            notional += take * lv.price
            got += take
            remaining -= take
        if got <= 1e-9:
            return (0.0, 0.0)
        return (notional / got, got)
