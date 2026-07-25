"""
Real-time feeds.

Two websockets matter:

  1. CLOB market stream (wss://ws-subscriptions-clob.polymarket.com/ws/market)
     Order books, price changes and trade prints for the tokens you subscribe to.
     Application-level heartbeat: send the text frame "PING" every 10s.

  2. Sports stream (wss://sports-api.polymarket.com/ws)
     Live scores for every game, no subscription frame required. The server sends
     "ping" every 5s and expects "pong" within 10s or it drops you.

Both reconnect with exponential backoff. A feed that silently dies is worse than
one that crashes, so disconnects are logged loudly and staleness is tracked.

Requires: websockets>=12
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from typing import Awaitable, Callable, Iterable

from ..models import OrderBook, now_ms

log = logging.getLogger(__name__)

CLOB_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
SPORTS_WS = "wss://sports-api.polymarket.com/ws"


class _Reconnecting:
    """Shared reconnect/backoff behaviour."""

    def __init__(self, url: str, name: str, max_backoff: float = 60.0):
        self.url = url
        self.name = name
        self.max_backoff = max_backoff
        self._stop = asyncio.Event()
        self.connected = False
        self.last_message_ms = 0
        self.reconnects = 0

    def stop(self) -> None:
        self._stop.set()

    @property
    def stale_ms(self) -> int:
        return now_ms() - self.last_message_ms if self.last_message_ms else 10**9

    async def _run(self) -> None:  # pragma: no cover - network loop
        try:
            import websockets
        except ImportError as e:
            raise RuntimeError(
                "The live feeds need the `websockets` package: pip install websockets"
            ) from e

        backoff = 1.0
        while not self._stop.is_set():
            try:
                async with websockets.connect(
                    self.url, ping_interval=None, close_timeout=5, max_size=8 * 1024 * 1024
                ) as ws:
                    self.connected = True
                    self.last_message_ms = now_ms()
                    backoff = 1.0
                    log.info("%s connected", self.name)
                    await self.session(ws)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                log.warning("%s disconnected: %s", self.name, e)
            finally:
                self.connected = False

            if self._stop.is_set():
                break
            self.reconnects += 1
            log.info("%s reconnecting in %.1fs (attempt %d)", self.name, backoff, self.reconnects)
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._stop.wait(), timeout=backoff)
            backoff = min(backoff * 2, self.max_backoff)

    async def session(self, ws) -> None:  # pragma: no cover
        raise NotImplementedError


class MarketFeed(_Reconnecting):
    """CLOB order book / trade stream for a set of outcome tokens."""

    def __init__(
        self,
        token_ids: Iterable[str],
        on_book: Callable[[OrderBook], Awaitable[None] | None] | None = None,
        on_trade: Callable[[str, float, float, int], Awaitable[None] | None] | None = None,
        tick_sizes: dict[str, float] | None = None,
    ):
        super().__init__(CLOB_WS, "market-feed")
        self.token_ids: set[str] = set(token_ids)
        self.on_book = on_book
        self.on_trade = on_trade
        self.tick_sizes = tick_sizes or {}
        self.books: dict[str, OrderBook] = {}
        self._ws = None
        self._pending_sub: set[str] = set()

    async def start(self) -> None:  # pragma: no cover
        await self._run()

    async def subscribe(self, token_ids: Iterable[str]) -> None:
        """Add tokens to the live subscription without reconnecting."""
        new = {t for t in token_ids if t not in self.token_ids}
        if not new:
            return
        self.token_ids |= new
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.send(
                    json.dumps({"assets_ids": sorted(new), "operation": "subscribe"})
                )
        else:
            self._pending_sub |= new

    async def session(self, ws) -> None:  # pragma: no cover
        self._ws = ws
        await ws.send(
            json.dumps({"assets_ids": sorted(self.token_ids), "type": "market"})
        )
        hb = asyncio.create_task(self._heartbeat(ws))
        try:
            async for raw in ws:
                self.last_message_ms = now_ms()
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", "ignore")
                if raw.strip() in ("PONG", "PING", ""):
                    continue
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                for ev in msg if isinstance(msg, list) else [msg]:
                    await self._handle(ev)
        finally:
            hb.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await hb
            self._ws = None

    async def _heartbeat(self, ws) -> None:  # pragma: no cover
        while True:
            await asyncio.sleep(10)
            await ws.send("PING")

    async def _handle(self, ev: dict) -> None:
        et = ev.get("event_type")
        if et == "book":
            token = str(ev.get("asset_id", ""))
            book = OrderBook.from_ws(ev, self.tick_sizes.get(token, 0.01))
            self.books[token] = book
            await self._emit(self.on_book, book)

        elif et == "price_change":
            # Incremental level updates. Apply them so the book stays fresh
            # between full snapshots.
            for ch in ev.get("price_changes") or []:
                token = str(ch.get("asset_id", ""))
                book = self.books.get(token)
                if book is None:
                    continue
                self._apply_delta(book, ch)
                book.timestamp_ms = int(ev.get("timestamp") or now_ms())
                await self._emit(self.on_book, book)

        elif et == "last_trade_price":
            token = str(ev.get("asset_id", ""))
            try:
                price = float(ev.get("price"))
                size = float(ev.get("size"))
            except (TypeError, ValueError):
                return
            ts = int(ev.get("timestamp") or now_ms())
            if self.on_trade:
                res = self.on_trade(token, price, size, ts)
                if asyncio.iscoroutine(res):
                    await res

        elif et == "tick_size_change":
            token = str(ev.get("asset_id", ""))
            with contextlib.suppress(TypeError, ValueError):
                self.tick_sizes[token] = float(ev.get("new_tick_size"))

        elif et == "market_resolved":
            log.info("market resolved: winning asset %s", ev.get("winning_asset_id"))

    @staticmethod
    def _apply_delta(book: OrderBook, ch: dict) -> None:
        from ..models import Level

        try:
            price = float(ch.get("price"))
            size = float(ch.get("size"))
        except (TypeError, ValueError):
            return
        side = str(ch.get("side", "")).upper()
        levels = book.bids if side == "BUY" else book.asks
        for i, lv in enumerate(levels):
            if abs(lv.price - price) < 1e-9:
                if size <= 0:
                    levels.pop(i)
                else:
                    levels[i] = Level(price, size)
                break
        else:
            if size > 0:
                levels.append(Level(price, size))
        levels.sort(key=lambda l: l.price, reverse=(side == "BUY"))

    async def _emit(self, cb, *args) -> None:
        if cb is None:
            return
        res = cb(*args)
        if asyncio.iscoroutine(res):
            await res


class SportsFeed(_Reconnecting):
    """Live score stream. No subscription frame; the server pushes every game."""

    def __init__(
        self,
        on_game: Callable[[dict], Awaitable[None] | None] | None = None,
        leagues: Iterable[str] | None = None,
    ):
        super().__init__(SPORTS_WS, "sports-feed")
        self.on_game = on_game
        # Empty means accept everything; useful because table tennis league codes
        # vary and over-filtering silently kills the feed.
        self.leagues = {l.upper() for l in (leagues or [])}
        self.games: dict[int, dict] = {}

    async def start(self) -> None:  # pragma: no cover
        await self._run()

    async def session(self, ws) -> None:  # pragma: no cover
        async for raw in ws:
            self.last_message_ms = now_ms()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", "ignore")
            txt = raw.strip()
            if txt.lower() == "ping":
                await ws.send("pong")
                continue
            try:
                msg = json.loads(txt)
            except json.JSONDecodeError:
                continue
            for game in msg if isinstance(msg, list) else [msg]:
                if not isinstance(game, dict):
                    continue
                league = str(game.get("leagueAbbreviation", "")).upper()
                if self.leagues and league not in self.leagues:
                    continue
                gid = game.get("gameId")
                if gid is not None:
                    self.games[int(gid)] = game
                if self.on_game:
                    res = self.on_game(game)
                    if asyncio.iscoroutine(res):
                        await res

    def game(self, game_id: int | None) -> dict | None:
        return self.games.get(int(game_id)) if game_id is not None else None
