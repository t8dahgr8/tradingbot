"""High-frequency paper market making around the live score model.

The fast loop is quote management, not blind taker churn. It posts passive bids
on both complementary outcomes, offers acquired inventory back to the market,
and cancels every quote when the score changes. Real fills still require public
trade volume to clear the simulated queue in :mod:`paper_broker`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..execution.paper_broker import PaperBroker
from ..execution.portfolio import Portfolio
from ..models import OrderBook, Position, Side, TradableMarket
from .live_model import LiveModelStrategy


@dataclass
class MarketMakerConfig:
    quote_refresh_ms: int = 350
    score_pause_ms: int = 1_500
    max_score_age_ms: int = 60_000
    max_book_age_ms: int = 15_000
    quote_size_shares: float = 5.0
    min_spread_ticks: int = 1
    max_spread: float = 0.06
    min_quote_edge: float = 0.003
    min_exit_profit: float = 0.004
    inventory_skew_ticks: float = 1.0
    soft_inventory_age_ms: int = 30_000
    hard_inventory_age_ms: int = 90_000
    hard_stop: float = 0.05
    max_inventory_token_pct: float = 0.05
    max_inventory_market_pct: float = 0.08
    max_total_inventory_pct: float = 0.25
    max_live_markets: int = 12


@dataclass(frozen=True)
class QuoteIntent:
    token_id: str
    side: Side
    price: float
    size: float
    reason: str


@dataclass
class MarketMakerStats:
    quote_cycles: int = 0
    quotes_sent: int = 0
    quotes_cancelled: int = 0
    score_cancels: int = 0
    maker_fills: int = 0
    taker_fills: int = 0
    paused_until_ms: dict[str, int] = field(default_factory=dict)
    last_cycle_ms: dict[str, int] = field(default_factory=dict)


def _floor_tick(value: float, tick: float) -> float:
    return round(math.floor((value + 1e-12) / tick) * tick, 6)


def _ceil_tick(value: float, tick: float) -> float:
    return round(math.ceil((value - 1e-12) / tick) * tick, 6)


class HftMarketMaker:
    """Build passive quote intents and inventory exit decisions."""

    def __init__(
        self,
        config: MarketMakerConfig | None,
        model: LiveModelStrategy,
    ):
        self.cfg = config or MarketMakerConfig()
        self.model = model
        self.stats = MarketMakerStats()

    def on_score_change(self, market_id: str, ts_ms: int) -> None:
        self.stats.paused_until_ms[market_id] = ts_ms + self.cfg.score_pause_ms
        self.stats.last_cycle_ms.pop(market_id, None)

    def begin_cycle(self, market_id: str, ts_ms: int, force: bool = False) -> bool:
        last = self.stats.last_cycle_ms.get(market_id, 0)
        if not force and ts_ms - last < self.cfg.quote_refresh_ms:
            return False
        self.stats.last_cycle_ms[market_id] = ts_ms
        self.stats.quote_cycles += 1
        return True

    def record_fill(self, liquidity: str) -> None:
        if liquidity == "maker":
            self.stats.maker_fills += 1
        else:
            self.stats.taker_fills += 1

    def fair_values(
        self,
        market: TradableMarket,
        books: dict[str, OrderBook],
        ts_ms: int,
    ) -> tuple[float, float] | None:
        if self.score_issue(market, ts_ms):
            return None
        tracker = self.model.trackers.get(market.market_id)
        if tracker is None:
            return None

        b0 = books.get(market.token_ids[0])
        b1 = books.get(market.token_ids[1])
        market_views: list[float] = []
        if b0 is not None and b0.mid is not None:
            market_views.append(b0.mid)
        if b1 is not None and b1.mid is not None:
            market_views.append(1.0 - b1.mid)
        if not market_views:
            return None

        market_p0 = sum(market_views) / len(market_views)
        self.model.decay_anchor(market, market_p0, ts_ms)
        weight = self.model.cfg.model_weight
        fair0 = weight * tracker.fair_value + (1.0 - weight) * market_p0
        fair0 = min(max(fair0, market.tick_size), 1.0 - market.tick_size)
        return fair0, 1.0 - fair0

    def score_issue(self, market: TradableMarket, ts_ms: int) -> str:
        tracker = self.model.trackers.get(market.market_id)
        if tracker is None or not tracker.live or tracker.ended:
            return "waiting for live score"
        if tracker.anchor_prob is None or tracker.fair_value is None:
            return "waiting for clean anchor"
        if not tracker.anchored_cleanly:
            return "late or invalid anchor"
        if not tracker.score_tradeable:
            return tracker.score_issue or "unsupported score"
        if not tracker.last_score_change_ms:
            return "waiting for next score"
        if tracker.score_age_ms(ts_ms) > self.cfg.max_score_age_ms:
            return "stale score"
        return ""

    def quote_intents(
        self,
        market: TradableMarket,
        books: dict[str, OrderBook],
        portfolio: Portfolio,
        broker: PaperBroker,
        marks: dict[str, float],
        ts_ms: int,
    ) -> tuple[list[QuoteIntent], str]:
        """Return the complete desired passive quote set for one market."""
        if ts_ms < self.stats.paused_until_ms.get(market.market_id, 0):
            return [], "score-change pause"

        valid_books = {
            token: book
            for token, book in books.items()
            if book.is_valid() and book.age_ms(ts_ms) <= self.cfg.max_book_age_ms
        }
        if not valid_books:
            return [], "no fresh book"

        score_issue = self.score_issue(market, ts_ms)
        fair_values = None if score_issue else self.fair_values(
            market,
            valid_books,
            ts_ms,
        )
        allow_buys = fair_values is not None
        fairs = fair_values or (0.5, 0.5)
        live_orders = broker.live_orders()
        intents: list[QuoteIntent] = []

        # Inventory offers remain active even if the score goes stale. New bids do
        # not, because stale winner probabilities are exactly what informed flow
        # can pick off.
        for idx, token_id in enumerate(market.token_ids):
            book = valid_books.get(token_id)
            pos = portfolio.positions.get(token_id)
            if book is None or pos is None or pos.shares <= 1e-9:
                continue
            intent = self._sell_intent(market, book, pos, ts_ms)
            if intent is not None:
                intents.append(intent)

        if not allow_buys:
            return intents, score_issue or "no market midpoint"

        selling_tokens = {
            intent.token_id
            for intent in intents
            if intent.side == Side.SELL
        }
        equity = portfolio.equity(marks)
        reserved_total = sum(
            order.remaining * order.limit_price
            for order in live_orders
            if order.side == Side.BUY
        )
        reserved_market = sum(
            order.remaining * order.limit_price
            for order in live_orders
            if order.side == Side.BUY and order.market_id == market.market_id
        )
        reserved_elsewhere = max(0.0, reserved_total - reserved_market)
        room_total = max(
            0.0,
            self.cfg.max_total_inventory_pct * equity
            - portfolio.exposure()
            - reserved_elsewhere,
        )
        room_market = max(
            0.0,
            self.cfg.max_inventory_market_pct * equity
            - portfolio.exposure_in_market(market.market_id),
        )
        cash_room = max(0.0, portfolio.cash - reserved_elsewhere)

        candidates: list[tuple[float, int, float]] = []
        for idx, token_id in enumerate(market.token_ids):
            # Avoid self-crosses and accidental inventory growth while an offer
            # for this exact outcome is already working.
            if token_id in selling_tokens:
                continue
            book = valid_books.get(token_id)
            if book is None:
                continue
            pos = portfolio.positions.get(token_id)
            inventory_shares = pos.shares if pos else 0.0
            adjusted_fair = fairs[idx] - (
                self.cfg.inventory_skew_ticks
                * market.tick_size
                * inventory_shares
                / max(self.cfg.quote_size_shares, 1e-9)
            )
            price = self._buy_price(book, adjusted_fair)
            if price is not None:
                candidates.append((adjusted_fair - price, idx, price))

        # Spend scarce inventory room on the strongest maker edge first.
        candidates.sort(reverse=True)
        for edge, idx, price in candidates:
            token_id = market.token_ids[idx]
            pos = portfolio.positions.get(token_id)
            token_exposure = pos.shares * pos.avg_cost if pos else 0.0
            room_token = max(
                0.0,
                self.cfg.max_inventory_token_pct * equity
                - token_exposure,
            )
            room = min(room_total, room_market, cash_room, room_token)
            max_size = math.floor(room / max(price, 1e-9))
            size = min(float(max_size), self.cfg.quote_size_shares)
            min_size = max(market.min_order_size, 1.0)
            if size + 1e-9 < min_size:
                continue
            size = float(math.floor(size))
            notional = size * price
            intents.append(QuoteIntent(
                token_id=token_id,
                side=Side.BUY,
                price=price,
                size=size,
                reason=f"maker edge {edge:+.3f}",
            ))
            room_total -= notional
            room_market -= notional
            cash_room -= notional

        return intents, "ok" if intents else "no maker edge or inventory room"

    def _buy_price(self, book: OrderBook, fair_value: float) -> float | None:
        bid, ask = book.best_bid, book.best_ask
        if bid is None or ask is None:
            return None
        spread = ask - bid
        tick = book.tick_size
        if (
            spread + 1e-9 < self.cfg.min_spread_ticks * tick
            or spread - self.cfg.max_spread > 1e-9
        ):
            return None

        ceiling = min(fair_value - self.cfg.min_quote_edge, ask - tick)
        price = min(bid + tick, _floor_tick(ceiling, tick))
        if price < bid:
            price = bid
        price = _floor_tick(price, tick)
        if price <= 0 or price >= ask - 1e-9:
            return None
        if fair_value - price + 1e-9 < self.cfg.min_quote_edge:
            return None
        return price

    def _sell_intent(
        self,
        market: TradableMarket,
        book: OrderBook,
        pos: Position,
        ts_ms: int,
    ) -> QuoteIntent | None:
        bid, ask = book.best_bid, book.best_ask
        if bid is None or ask is None:
            return None
        available = max(0.0, pos.shares)
        if available <= 1e-9:
            return None

        age_ms = max(0, ts_ms - pos.opened_ms)
        target = pos.avg_cost + self.cfg.min_exit_profit
        if age_ms >= self.cfg.soft_inventory_age_ms:
            target = min(target, ask)
        price = max(bid + book.tick_size, _ceil_tick(target, book.tick_size))
        price = min(price, 1.0 - book.tick_size)
        if price <= bid + 1e-9:
            return None
        size = min(available, self.cfg.quote_size_shares)
        return QuoteIntent(
            token_id=pos.token_id,
            side=Side.SELL,
            price=price,
            size=size,
            reason=f"inventory offer age={age_ms}ms",
        )

    def force_exit_reason(
        self,
        market: TradableMarket,
        token_id: str,
        pos: Position,
        book: OrderBook,
        ts_ms: int,
    ) -> str:
        bid = book.best_bid
        if bid is None:
            return ""
        age_ms = max(0, ts_ms - pos.opened_ms)
        if bid <= pos.avg_cost - self.cfg.hard_stop:
            return f"HFT inventory stop (bid {bid:.3f} vs cost {pos.avg_cost:.3f})"
        if age_ms >= self.cfg.hard_inventory_age_ms:
            return f"HFT hard inventory timeout ({age_ms}ms)"
        return ""
