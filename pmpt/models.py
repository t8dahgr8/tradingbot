"""Core domain types. Pure stdlib so the trading logic stays testable offline."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


def now_ms() -> int:
    return int(time.time() * 1000)


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    # Crosses the spread immediately, pays the taker fee, guaranteed-ish fill.
    MARKETABLE = "MARKETABLE"
    # Rests on the book, earns the spread, only fills if the market comes to you.
    PASSIVE = "PASSIVE"


class OrderStatus(str, Enum):
    OPEN = "OPEN"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class Level:
    price: float
    size: float


@dataclass
class OrderBook:
    """One side of one market. Polymarket quotes each outcome token separately.

    `bids` are sorted best (highest) first, `asks` best (lowest) first.
    """

    token_id: str
    bids: list[Level] = field(default_factory=list)
    asks: list[Level] = field(default_factory=list)
    timestamp_ms: int = field(default_factory=now_ms)
    tick_size: float = 0.01

    # -- derived views -----------------------------------------------------

    @property
    def best_bid(self) -> float | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0].price if self.asks else None

    @property
    def mid(self) -> float | None:
        bb, ba = self.best_bid, self.best_ask
        if bb is None or ba is None:
            return bb if ba is None else ba
        return 0.5 * (bb + ba)

    @property
    def spread(self) -> float | None:
        bb, ba = self.best_bid, self.best_ask
        if bb is None or ba is None:
            return None
        return ba - bb

    def depth(self, side: Side, max_price: float | None = None) -> float:
        """Total shares available to a taker on `side`, optionally price-limited."""
        levels = self.asks if side == Side.BUY else self.bids
        total = 0.0
        for lv in levels:
            if max_price is not None:
                if side == Side.BUY and lv.price > max_price:
                    break
                if side == Side.SELL and lv.price < max_price:
                    break
            total += lv.size
        return total

    def age_ms(self, ref_ms: int | None = None) -> int:
        return (ref_ms or now_ms()) - self.timestamp_ms

    def is_valid(self) -> bool:
        """Crossed or empty books mean the feed is broken or the market is dead."""
        bb, ba = self.best_bid, self.best_ask
        if bb is None or ba is None:
            return False
        if bb >= ba:
            return False
        return 0.0 < bb < 1.0 and 0.0 < ba < 1.0

    @staticmethod
    def from_ws(payload: dict, tick_size: float = 0.01) -> "OrderBook":
        """Build from a Polymarket `book` websocket event.

        The feed does not guarantee ordering, so we sort explicitly.
        """
        def levels(raw: Iterable[dict], reverse: bool) -> list[Level]:
            out = [
                Level(float(x["price"]), float(x["size"]))
                for x in (raw or [])
                if float(x.get("size", 0) or 0) > 0
            ]
            out.sort(key=lambda l: l.price, reverse=reverse)
            return out

        ts = payload.get("timestamp")
        return OrderBook(
            token_id=str(payload.get("asset_id", "")),
            bids=levels(payload.get("bids"), reverse=True),
            asks=levels(payload.get("asks"), reverse=False),
            timestamp_ms=int(ts) if ts else now_ms(),
            tick_size=tick_size,
        )


@dataclass
class Fill:
    order_id: str
    token_id: str
    side: Side
    price: float          # average price actually paid/received
    size: float           # shares
    fee: float = 0.0
    timestamp_ms: int = field(default_factory=now_ms)
    liquidity: str = "taker"   # "taker" | "maker"

    @property
    def notional(self) -> float:
        return self.price * self.size

    @property
    def cash_delta(self) -> float:
        """Signed change to cash: buying costs money, selling returns it."""
        if self.side == Side.BUY:
            return -(self.notional + self.fee)
        return self.notional - self.fee


@dataclass
class Order:
    token_id: str
    side: Side
    size: float                       # shares requested
    limit_price: float
    order_type: OrderType = OrderType.MARKETABLE
    order_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: OrderStatus = OrderStatus.OPEN
    filled: float = 0.0
    avg_price: float = 0.0
    created_ms: int = field(default_factory=now_ms)
    # Shares resting ahead of us at our price level when the order was placed.
    # This is what determines whether a passive order ever actually fills.
    queue_ahead: float = 0.0
    market_id: str = ""
    reason: str = ""

    @property
    def remaining(self) -> float:
        return max(0.0, self.size - self.filled)

    @property
    def is_live(self) -> bool:
        return self.status in (OrderStatus.OPEN, OrderStatus.PARTIAL)

    def apply(self, fill: Fill) -> None:
        prev_notional = self.avg_price * self.filled
        self.filled += fill.size
        self.avg_price = (prev_notional + fill.price * fill.size) / max(self.filled, 1e-9)
        self.status = OrderStatus.FILLED if self.remaining <= 1e-9 else OrderStatus.PARTIAL


@dataclass
class Position:
    """A holding in a single outcome token.

    Polymarket tokens settle at exactly 1.0 or 0.0, which makes P&L accounting
    simple but also means an unhedged position is all-or-nothing at resolution.
    """

    token_id: str
    market_id: str = ""
    label: str = ""
    shares: float = 0.0
    avg_cost: float = 0.0
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    opened_ms: int = field(default_factory=now_ms)

    def apply(self, fill: Fill) -> float:
        """Apply a fill, returning realized P&L generated by this fill."""
        self.fees_paid += fill.fee
        realized = 0.0
        signed = fill.size if fill.side == Side.BUY else -fill.size

        if self.shares == 0 or (self.shares > 0) == (signed > 0):
            # Opening or adding: weighted-average the cost basis.
            total = self.shares + signed
            if abs(total) > 1e-9:
                self.avg_cost = (
                    self.avg_cost * self.shares + fill.price * signed
                ) / total
            self.shares = total
        else:
            # Reducing or flipping.
            closing = min(abs(signed), abs(self.shares))
            direction = 1.0 if self.shares > 0 else -1.0
            realized = (fill.price - self.avg_cost) * closing * direction
            self.realized_pnl += realized
            remainder = abs(signed) - closing
            self.shares += signed
            if remainder > 1e-9:
                self.avg_cost = fill.price
            elif abs(self.shares) <= 1e-9:
                self.shares = 0.0
                self.avg_cost = 0.0
        return realized

    def market_value(self, mark: float) -> float:
        return self.shares * mark

    def unrealized_pnl(self, mark: float) -> float:
        return (mark - self.avg_cost) * self.shares


@dataclass
class TradableMarket:
    """A Polymarket binary market, flattened to what the trader actually needs."""

    market_id: str
    condition_id: str
    question: str
    slug: str
    token_ids: tuple[str, str]         # (outcome 0, outcome 1)
    outcomes: tuple[str, str]          # e.g. ("Sinner", "Alcaraz")
    tick_size: float = 0.01
    min_order_size: float = 5.0
    accepting_orders: bool = True
    sport: str = "tennis"
    league: str = ""
    # The sports score stream is always home-away.  This says what outcome 0
    # represents so team-sport models do not accidentally reverse the score.
    outcome0_role: str = "home"       # "home" | "away" | "draw"
    game_id: int | None = None
    event_slug: str = ""
    best_of: int = 3
    fees_enabled: bool = False
    fee_rate: float = 0.0

    def other(self, token_id: str) -> str:
        return self.token_ids[1] if token_id == self.token_ids[0] else self.token_ids[0]

    def index_of(self, token_id: str) -> int:
        return 0 if token_id == self.token_ids[0] else 1


@dataclass
class Signal:
    """A strategy's opinion about one token at one moment."""

    token_id: str
    market_id: str
    fair_value: float           # model probability the token resolves to 1
    market_price: float         # what you would actually pay/receive
    edge: float                 # fair_value - market_price, signed for the side
    side: Side
    confidence: float = 1.0
    reason: str = ""
    metadata: dict = field(default_factory=dict)
