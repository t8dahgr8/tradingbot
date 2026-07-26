"""
The trading engine: feeds -> model -> risk -> paper broker -> portfolio.

Deliberately single-threaded and event-driven. Every decision is triggered by a
market data event, and every decision passes through the same gate, so there is
exactly one path from "the score changed" to "an order exists".
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections import deque
from datetime import datetime, timezone

from .config import AppConfig
from .data.feeds import MarketFeed, SportsFeed
from .data.gamma import GammaClient
from .execution.paper_broker import PaperBroker
from .execution.portfolio import Portfolio
from .execution.risk import RiskManager
from .models import Fill, Order, OrderBook, OrderType, Side, TradableMarket, now_ms
from .strategy.live_model import LiveModelStrategy

log = logging.getLogger(__name__)


def _balanced_markets(
    markets: list[TradableMarket],
    sports: list[str],
    limit: int,
) -> list[TradableMarket]:
    """Round-robin sports so one busy league cannot consume the watchlist."""
    buckets = {
        sport: deque(m for m in markets if m.sport == sport)
        for sport in sports
    }
    selected: list[TradableMarket] = []
    while len(selected) < limit:
        added = False
        for sport in sports:
            bucket = buckets[sport]
            if bucket and len(selected) < limit:
                selected.append(bucket.popleft())
                added = True
        if not added:
            break
    return selected


class TradingEngine:
    def __init__(self, config: AppConfig):
        self.cfg = config
        os.makedirs(config.run.state_dir, exist_ok=True)

        self.gamma = GammaClient(sports=config.run.sports)
        self.broker = PaperBroker(config.broker)
        self.portfolio = Portfolio(
            config.run.starting_cash,
            journal_path=os.path.join(config.run.state_dir, "trades.csv"),
        )
        self.risk = RiskManager(config.risk)
        self.strategy = LiveModelStrategy(config.strategy)

        self.market_feed: MarketFeed | None = None
        self.sports_feed: SportsFeed | None = None
        self._stop = asyncio.Event()
        self._tasks: list[asyncio.Task] = []
        self.started_ms = now_ms()
        self.signals_seen = 0
        self.orders_sent = 0
        self.rejections: dict[str, int] = {}

    # -- lifecycle ---------------------------------------------------------

    async def run(self) -> None:
        cfg = self.cfg.run
        log.info(
            "starting paper trader | bankroll $%.2f | sports=%s | live_only=%s",
            cfg.starting_cash, ",".join(cfg.sports), cfg.live_only,
        )

        await self._discover()

        self.market_feed = MarketFeed(
            token_ids=self.gamma.all_token_ids(),
            on_book=self._on_book,
            on_trade=self._on_trade,
            tick_sizes={
                t: m.tick_size for t, m in self.gamma.by_token.items()
            },
        )
        self.sports_feed = SportsFeed(on_game=self._on_game)

        self._tasks = [
            asyncio.create_task(self.market_feed.start(), name="market-feed"),
            asyncio.create_task(self.sports_feed.start(), name="sports-feed"),
            asyncio.create_task(self._discovery_loop(), name="discovery"),
            asyncio.create_task(self._housekeeping_loop(), name="housekeeping"),
        ]
        if cfg.max_runtime_s:
            self._tasks.append(asyncio.create_task(self._deadline(cfg.max_runtime_s)))

        # Announce the session immediately; the normal mark loop refreshes it.
        self._publish()

        try:
            await self._stop.wait()
        finally:
            await self.shutdown()

    async def _deadline(self, seconds: int) -> None:
        await asyncio.sleep(seconds)
        log.info("max runtime reached")
        self.stop()

    def stop(self) -> None:
        self._stop.set()

    async def shutdown(self) -> None:
        log.info("shutting down")
        for f in (self.market_feed, self.sports_feed):
            if f:
                f.stop()
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        self.broker.cancel_all()
        self._save()
        self._publish()
        self.print_report()

    # -- discovery ---------------------------------------------------------

    async def _discover(self) -> None:
        previous = dict(self.gamma.markets)
        loop = asyncio.get_running_loop()
        markets = await loop.run_in_executor(
            None, lambda: self.gamma.refresh(only_live=self.cfg.run.live_only)
        )

        # Never rotate a market out while it carries risk or a working order.
        pinned: list[TradableMarket] = []
        for m in previous.values():
            exposed = any(
                (
                    self.portfolio.positions.get(token)
                    and self.portfolio.positions[token].shares > 0
                )
                or self.broker.live_size(token, Side.BUY) > 0
                or self.broker.live_size(token, Side.SELL) > 0
                for token in m.token_ids
            )
            if exposed:
                pinned.append(m)

        pinned_ids = {m.market_id for m in pinned}
        available = [m for m in markets if m.market_id not in pinned_ids]
        slots = max(0, self.cfg.run.max_tracked_markets - len(pinned))
        markets = pinned + _balanced_markets(
            available,
            self.cfg.run.sports,
            slots,
        )
        self.gamma.activate(markets)

        for m in markets:
            live_fee = m.fee_rate if m.fees_enabled else 0.0
            self.broker.set_market_fees(m.token_ids, live_fee)
            ev = self.gamma.event_for_market(m)
            if ev is None:
                continue
            # Anchor from the market's own pre-match price. `outcomePrices` is a
            # JSON-encoded string, which is easy to get wrong.
            price = self._gamma_price(ev, m)
            if price is None:
                continue
            games = self._games_played(ev)
            t = self.strategy.trackers.get(m.market_id)
            if t is not None and t.anchor_prob is not None:
                # Keep following the pregame winner line. Material repricing is
                # usually the cleanest available signal that news changed.
                moved = abs(price - t.anchor_prob)
                if (
                    not bool(ev.get("live"))
                    and not t.live
                    and games == 0
                    and moved >= self.cfg.strategy.pregame_reanchor_threshold
                ):
                    self.strategy.set_anchor(m, price, games_played=0)
                continue
            self.strategy.set_anchor(m, price, games_played=games)

        if self.market_feed is not None:
            await self.market_feed.subscribe(
                [t for m in markets for t in m.token_ids]
            )

    @staticmethod
    def _gamma_price(event: dict, market: TradableMarket) -> float | None:
        import json

        for raw in event.get("markets") or []:
            if str(raw.get("id")) != market.market_id:
                continue
            try:
                prices = raw.get("outcomePrices")
                if isinstance(prices, str):
                    prices = json.loads(prices)
                if prices:
                    return float(prices[0])
            except (ValueError, TypeError, json.JSONDecodeError):
                pass
            bb, ba = raw.get("bestBid"), raw.get("bestAsk")
            if bb is not None and ba is not None:
                return 0.5 * (float(bb) + float(ba))
        return None

    @staticmethod
    def _games_played(event: dict) -> int:
        score = str(event.get("score") or "")
        total = 0
        for part in score.split(","):
            s = part.split("(")[0].strip()
            if "-" in s:
                try:
                    a, b = s.split("-")[:2]
                    total += int(a.strip()) + int(b.strip())
                except ValueError:
                    continue
        return total

    async def _discovery_loop(self) -> None:
        while not self._stop.is_set():
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(
                    self._stop.wait(), timeout=self.cfg.run.discovery_interval_s
                )
            if self._stop.is_set():
                break
            with contextlib.suppress(Exception):
                await self._discover()

    # -- event handlers ----------------------------------------------------

    async def _on_book(self, book: OrderBook) -> None:
        fills = self.broker.on_book(book)
        for f in fills:
            self._book_fill(f)

        market = self.gamma.by_token.get(book.token_id)
        if market is None:
            return
        await self._evaluate(market)

    async def _on_trade(self, token_id: str, price: float, size: float, ts: int) -> None:
        for f in self.broker.on_trade(token_id, price, size, ts):
            self._book_fill(f)

    async def _on_game(self, game: dict) -> None:
        """A live score arrived. This is the event the whole strategy waits for."""
        gid = game.get("gameId")
        if gid is None:
            return
        markets = [m for m in self.gamma.markets.values() if m.game_id == int(gid)]
        if not markets:
            return

        score = str(game.get("score") or "")
        period = str(game.get("period") or "")
        live = bool(game.get("live"))
        ended = bool(game.get("ended")) or str(game.get("status", "")).lower() in (
            "finished", "final", "cancelled", "canceled", "postponed"
        )

        for m in markets:
            tracker = self.strategy.trackers.get(m.market_id)
            if (
                tracker is not None
                and live
                and not tracker.live
                and self._games_played({"score": score})
                > self.cfg.strategy.max_games_at_anchor
            ):
                # If the first live score is already underway, the market price
                # was not a clean pregame anchor. Keep observing, but never bet.
                tracker.anchored_cleanly = False
            self.strategy.on_score(m, score, period, live, ended)
            if ended:
                await self._flatten(m, "match ended")
            else:
                await self._evaluate(m)

    def _book_fill(self, fill: Fill) -> None:
        market = self.gamma.by_token.get(fill.token_id)
        label = ""
        if market:
            label = f"{market.outcomes[market.index_of(fill.token_id)]} ({market.slug})"
        self.portfolio.apply_fill(fill, market.market_id if market else "", label)
        log.info(
            "FILL %s %s %.0f @ %.4f (%s) fee=%.4f cash=%.2f",
            fill.side.value, label or fill.token_id[:10], fill.size, fill.price,
            fill.liquidity, fill.fee, self.portfolio.cash,
        )

    # -- decisioning -------------------------------------------------------

    def _marks(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for token in list(self.portfolio.positions):
            b = self.broker.book(token)
            if b is not None and b.mid is not None:
                out[token] = b.mid
        return out

    def _bids(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for token in list(self.portfolio.positions):
            b = self.broker.book(token)
            out[token] = b.best_bid if (b and b.best_bid is not None) else 0.0
        return out

    async def _evaluate(self, market: TradableMarket) -> None:
        ts = now_ms()
        marks = self._marks()
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.risk.check_halt(self.portfolio, marks, day):
            self.broker.cancel_all(side=Side.BUY)
            return

        books = {t: self.broker.book(t) for t in market.token_ids}
        books = {k: v for k, v in books.items() if v is not None}

        # Exits first: getting out is more urgent than getting in.
        for token in market.token_ids:
            pos = self.portfolio.positions.get(token)
            if pos is None or pos.shares <= 0:
                continue
            b = books.get(token)
            if b is None:
                continue
            should, why = self.strategy.exit_signal(
                market,
                token,
                pos.avg_cost,
                b,
                opened_ms=pos.opened_ms,
                ts=ts,
                entry_fee_per_share=pos.fees_paid / max(pos.shares, 1e-9),
            )
            if should:
                await self._close(market, token, why)

        signal = self.strategy.evaluate(market, books, ts)
        if signal is None:
            return
        self.signals_seen += 1

        book = books.get(signal.token_id)
        if book is None:
            return
        if any(
            self.broker.live_size(token, Side.BUY) > 0
            for token in market.token_ids
        ):
            key = "entry order pending"
            self.rejections[key] = self.rejections.get(key, 0) + 1
            return
        passive_entry = self.cfg.broker.passive_entries
        depth = book.depth(Side.BUY, max_price=signal.market_price + 0.02)
        ok, why = self.risk.approve(
            signal, market, self.portfolio, marks,
            book_age_ms=book.age_ms(ts), spread=book.spread, depth=depth, ts_ms=ts,
            taker_legs=1 if passive_entry else 2,
        )
        if not ok:
            self.rejections[why.split(" (")[0]] = self.rejections.get(why.split(" (")[0], 0) + 1
            log.debug("signal rejected (%s): %s", why, signal.reason)
            return

        _, fillable = self.broker.expected_cost(signal.token_id, Side.BUY, depth)
        shares = self.risk.size_order(
            signal, market, self.portfolio, marks, available_depth=fillable
        )
        if shares <= 0:
            return

        if passive_entry:
            if book.best_bid is None:
                return
            entry_price = book.best_bid
            order_type = OrderType.PASSIVE
        else:
            entry_price = signal.market_price + market.tick_size
            order_type = OrderType.MARKETABLE

        order = Order(
            token_id=signal.token_id,
            side=Side.BUY,
            size=shares,
            limit_price=entry_price,
            order_type=order_type,
            market_id=market.market_id,
            reason=signal.reason,
        )
        self.broker.submit(order, ts)
        self.risk.record_entry(market.market_id, ts)
        self.orders_sent += 1
        log.info(
            "ORDER %s BUY %.0f %s @%.4f | edge=%+.3f conf=%.2f | %s",
            order_type.value,
            shares, market.outcomes[market.index_of(signal.token_id)],
            order.limit_price, signal.edge, signal.confidence, signal.reason,
        )

    async def _close(self, market: TradableMarket, token_id: str, why: str) -> None:
        pos = self.portfolio.positions.get(token_id)
        if pos is None or pos.shares <= 0:
            return
        book = self.broker.book(token_id)
        if book is None or book.best_bid is None:
            return
        outstanding_sell = self.broker.live_size(token_id, Side.SELL)
        size = max(0.0, pos.shares - outstanding_sell)
        if size <= 1e-9:
            return
        order = Order(
            token_id=token_id,
            side=Side.SELL,
            size=size,
            limit_price=max(book.best_bid - market.tick_size, market.tick_size),
            order_type=OrderType.MARKETABLE,
            market_id=market.market_id,
            reason=why,
        )
        self.broker.submit(order)
        log.info("CLOSE %.0f %s | %s", size,
                 market.outcomes[market.index_of(token_id)], why)

    async def _flatten(self, market: TradableMarket, why: str) -> None:
        for token in market.token_ids:
            self.broker.cancel_all(token, Side.BUY)
        if self.cfg.strategy.hold_to_resolution:
            return
        for token in market.token_ids:
            await self._close(market, token, why)

    # -- housekeeping ------------------------------------------------------

    async def _housekeeping_loop(self) -> None:
        cfg = self.cfg.run
        last_status = 0
        while not self._stop.is_set():
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._stop.wait(), timeout=cfg.mark_interval_s)
            if self._stop.is_set():
                break

            marks = self._marks()
            self.portfolio.mark(marks)
            day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if self.risk.check_halt(self.portfolio, marks, day):
                self.broker.cancel_all(side=Side.BUY)

            # The dashboard snapshot is cheap, so refresh it every mark rather
            # than only on the slower status cadence.
            self._publish()

            now = now_ms()
            if now - last_status > cfg.status_interval_s * 1000:
                last_status = now
                self._status(marks)
                self._save()

            # A dead feed looks exactly like a quiet market. Say so explicitly.
            for f in (self.market_feed, self.sports_feed):
                if f and f.last_message_ms and f.stale_ms > 120_000:
                    log.warning("%s has been silent for %.0fs", f.name, f.stale_ms / 1000)

    def _status(self, marks: dict[str, float]) -> None:
        s = self.portfolio.stats(marks)
        liq = self.portfolio.liquidation_equity(self._bids())
        log.info(
            "STATUS equity=$%.2f (liq $%.2f) cash=$%.2f pos=%d exposure=$%.2f "
            "realized=%+.2f dd=%.1f%% | signals=%d orders=%d",
            s["equity"], liq, s["cash"], s["open_positions"], s["exposure"],
            s["realized_pnl"], s["max_drawdown_pct"], self.signals_seen, self.orders_sent,
        )

    def _save(self) -> None:
        with contextlib.suppress(Exception):
            self.portfolio.save(os.path.join(self.cfg.run.state_dir, "portfolio.json"))

    def _publish(self) -> None:
        from .dashboard import write_snapshot

        write_snapshot(self, self.cfg.run.state_dir)

    def print_report(self) -> None:
        marks = self._marks()
        s = self.portfolio.stats(marks)
        liq = self.portfolio.liquidation_equity(self._bids())
        runtime = (now_ms() - self.started_ms) / 1000

        lines = [
            "",
            "=" * 62,
            "  PAPER TRADING REPORT",
            "=" * 62,
            f"  Runtime              {runtime/60:>10.1f} min",
            f"  Starting bankroll    ${s['starting_cash']:>9.2f}",
            f"  Ending equity (mid)  ${s['equity']:>9.2f}",
            f"  Ending equity (liq)  ${liq:>9.2f}   <- the honest number",
            f"  Total return         {s['total_return_pct']:>9.2f} %",
            f"  Realized P&L         ${s['realized_pnl']:>9.2f}",
            f"  Unrealized P&L       ${s['unrealized_pnl']:>9.2f}",
            f"  Fees paid            ${s['fees_paid']:>9.2f}",
            f"  Max drawdown         {s['max_drawdown_pct']:>9.2f} %",
            "-" * 62,
            f"  Signals generated    {self.signals_seen:>10d}",
            f"  Orders sent          {self.orders_sent:>10d}",
            f"  Fills                {s['num_fills']:>10d}",
            f"  Open positions       {s['open_positions']:>10d}",
        ]
        if self.risk.state.halted:
            lines.append(f"  HALTED: {self.risk.state.halt_reason}")
        if self.rejections:
            lines.append("-" * 62)
            lines.append("  Why signals were rejected:")
            for k, v in sorted(self.rejections.items(), key=lambda x: -x[1])[:8]:
                lines.append(f"    {v:>6d}  {k}")
        lines.append("=" * 62)
        print("\n".join(lines))
