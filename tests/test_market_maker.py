"""High-frequency market-maker tests."""

import unittest

from pmpt.config import AppConfig
from pmpt.engine import TradingEngine
from pmpt.execution.paper_broker import BrokerConfig, PaperBroker
from pmpt.execution.portfolio import Portfolio
from pmpt.models import Fill, Level, Order, OrderBook, OrderType, Side, TradableMarket
from pmpt.strategy.live_model import LiveModelStrategy, StrategyConfig
from pmpt.strategy.market_maker import (
    HftMarketMaker,
    MarketMakerConfig,
    QuoteIntent,
)


T0, T1 = "maker-a", "maker-b"


def market() -> TradableMarket:
    return TradableMarket(
        market_id="maker-market",
        condition_id="maker-condition",
        question="A vs B",
        slug="a-vs-b",
        token_ids=(T0, T1),
        outcomes=("A", "B"),
        tick_size=0.01,
        min_order_size=5,
        accepting_orders=True,
        sport="tennis",
        best_of=3,
        game_id=7,
    )


def book(token: str, bid: float = 0.47, ask: float = 0.53, ts: int = 10_000
         ) -> OrderBook:
    return OrderBook(
        token_id=token,
        bids=[Level(bid, 100)],
        asks=[Level(ask, 100)],
        timestamp_ms=ts,
        tick_size=0.01,
    )


class TestHftMarketMaker(unittest.TestCase):
    def setUp(self):
        self.market = market()
        self.model = LiveModelStrategy(
            StrategyConfig(model_weight=1.0, signal_ttl_ms=60_000)
        )
        self.model.set_anchor(self.market, 0.50, ts=1_000)
        self.model.on_score(
            self.market, "0-0", "1", live=True, ended=False, ts=10_000
        )
        self.cfg = MarketMakerConfig(
            score_pause_ms=1_500,
            max_score_age_ms=12_000,
            max_book_age_ms=2_000,
            max_spread=0.06,
        )
        self.mm = HftMarketMaker(self.cfg, self.model)
        self.broker = PaperBroker(BrokerConfig(latency_ms=0, latency_jitter_ms=0))
        self.books = {T0: book(T0), T1: book(T1)}
        for value in self.books.values():
            self.broker.on_book(value, ts_ms=10_000)
        self.portfolio = Portfolio(100)

    def intents(self, ts: int = 10_000):
        return self.mm.quote_intents(
            self.market,
            self.books,
            self.portfolio,
            self.broker,
            {},
            ts,
        )

    def test_quotes_both_complementary_outcomes_without_crossing(self):
        intents, why = self.intents()
        buys = [intent for intent in intents if intent.side == Side.BUY]

        self.assertEqual(why, "ok")
        self.assertEqual({intent.token_id for intent in buys}, {T0, T1})
        self.assertTrue(all(intent.size == 5 for intent in buys))
        self.assertTrue(all(intent.price < self.books[intent.token_id].best_ask for intent in buys))

    def test_one_tick_spread_can_join_best_bid(self):
        one_tick_books = {
            T0: book(T0, bid=0.49, ask=0.50),
            T1: book(T1, bid=0.49, ask=0.50),
        }
        mm = HftMarketMaker(
            MarketMakerConfig(
                min_spread_ticks=1,
                min_quote_edge=0.003,
                max_book_age_ms=2_000,
            ),
            self.model,
        )

        intents, why = mm.quote_intents(
            self.market,
            one_tick_books,
            self.portfolio,
            self.broker,
            {},
            10_000,
        )

        self.assertEqual(why, "ok")
        self.assertEqual(
            {(intent.token_id, intent.price) for intent in intents},
            {(T0, 0.49), (T1, 0.49)},
        )

    def test_score_change_pauses_every_quote(self):
        self.mm.on_score_change(self.market.market_id, 10_000)

        intents, why = self.intents(ts=10_500)

        self.assertEqual(intents, [])
        self.assertEqual(why, "score-change pause")

    def test_existing_bid_is_still_part_of_desired_quote_set(self):
        first, _ = self.intents()
        intent = next(item for item in first if item.token_id == T0)

        self.broker.submit(Order(
            token_id=intent.token_id,
            side=intent.side,
            size=intent.size,
            limit_price=intent.price,
            order_type=OrderType.PASSIVE,
            market_id=self.market.market_id,
        ), 10_000)

        second, _ = self.intents()

        self.assertTrue(any(
            item.token_id == T0 and item.side == Side.BUY for item in second
        ))

    def test_stale_score_blocks_bids_but_keeps_inventory_offer(self):
        self.portfolio.apply_fill(
            Fill("buy", T0, Side.BUY, 0.48, 5, timestamp_ms=10_000),
            market_id=self.market.market_id,
        )
        stale_books = {
            T0: book(T0, 0.49, 0.51, ts=30_000),
            T1: book(T1, 0.49, 0.51, ts=30_000),
        }

        intents, why = self.mm.quote_intents(
            self.market,
            stale_books,
            self.portfolio,
            self.broker,
            {T0: 0.50},
            30_000,
        )

        self.assertEqual(why, "stale score")
        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0].side, Side.SELL)
        self.assertEqual(intents[0].token_id, T0)

    def test_late_anchor_waits_for_a_new_score_before_bidding(self):
        self.model.trackers[self.market.market_id].last_score_change_ms = 0

        intents, why = self.intents()

        self.assertEqual(intents, [])
        self.assertEqual(why, "waiting for next score")

    def test_hard_stop_forces_inventory_exit(self):
        self.portfolio.apply_fill(
            Fill("buy", T0, Side.BUY, 0.48, 5, timestamp_ms=10_000),
            market_id=self.market.market_id,
        )
        pos = self.portfolio.positions[T0]

        why = self.mm.force_exit_reason(
            self.market,
            T0,
            pos,
            book(T0, bid=0.42, ask=0.44, ts=11_000),
            11_000,
        )

        self.assertIn("inventory stop", why)

    def test_inventory_offer_prevents_self_crossing_buy(self):
        self.portfolio.apply_fill(
            Fill("buy", T0, Side.BUY, 0.48, 5, timestamp_ms=10_000),
            market_id=self.market.market_id,
        )

        intents, _ = self.intents()
        token_zero = [intent for intent in intents if intent.token_id == T0]

        self.assertEqual(len(token_zero), 1)
        self.assertEqual(token_zero[0].side, Side.SELL)

    def test_two_maker_fills_capture_one_tick(self):
        buy = next(
            intent
            for intent in self.intents()[0]
            if intent.token_id == T0 and intent.side == Side.BUY
        )
        self.broker.submit(Order(
            token_id=T0,
            side=Side.BUY,
            size=buy.size,
            limit_price=buy.price,
            order_type=OrderType.PASSIVE,
            market_id=self.market.market_id,
        ), 10_000)
        self.broker.on_book(self.books[T0], ts_ms=10_001)
        for fill in self.broker.on_trade(T0, buy.price, buy.size, 10_002):
            self.portfolio.apply_fill(fill, market_id=self.market.market_id)

        sell = next(
            intent
            for intent in self.intents(ts=10_003)[0]
            if intent.token_id == T0 and intent.side == Side.SELL
        )
        self.broker.submit(Order(
            token_id=T0,
            side=Side.SELL,
            size=sell.size,
            limit_price=sell.price,
            order_type=OrderType.PASSIVE,
            market_id=self.market.market_id,
        ), 10_003)
        self.broker.on_book(self.books[T0], ts_ms=10_004)
        for fill in self.broker.on_trade(T0, sell.price, sell.size, 10_005):
            self.portfolio.apply_fill(fill, market_id=self.market.market_id)

        self.assertAlmostEqual(self.portfolio.cash, 100.05)
        self.assertAlmostEqual(self.portfolio.realized_pnl, 0.05)
        self.assertAlmostEqual(self.portfolio.fees_paid, 0.0)
        self.assertNotIn(T0, self.portfolio.positions)


class TestHftQuoteReconciliation(unittest.TestCase):
    def setUp(self):
        cfg = AppConfig()
        cfg.run.mode = "hft"
        self.engine = TradingEngine(cfg)
        self.market = market()

    def test_unchanged_quote_keeps_queue_position(self):
        intent = QuoteIntent(T0, Side.BUY, 0.48, 5, "test")
        self.engine._reconcile_hft_quotes(self.market, [intent], 10_000)
        first = self.engine.broker.live_orders()[0]

        self.engine._reconcile_hft_quotes(self.market, [intent], 10_500)

        orders = self.engine.broker.live_orders()
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].order_id, first.order_id)
        self.assertEqual(self.engine.market_maker.stats.quotes_sent, 1)
        self.assertEqual(self.engine.market_maker.stats.quotes_cancelled, 0)

    def test_reprice_cancels_and_replaces_quote(self):
        self.engine._reconcile_hft_quotes(
            self.market,
            [QuoteIntent(T0, Side.BUY, 0.48, 5, "first")],
            10_000,
        )
        first = self.engine.broker.live_orders()[0]

        self.engine._reconcile_hft_quotes(
            self.market,
            [QuoteIntent(T0, Side.BUY, 0.49, 5, "second")],
            10_500,
        )

        orders = self.engine.broker.live_orders()
        self.assertEqual(len(orders), 1)
        self.assertNotEqual(orders[0].order_id, first.order_id)
        self.assertEqual(orders[0].limit_price, 0.49)
        self.assertEqual(self.engine.market_maker.stats.quotes_sent, 2)
        self.assertEqual(self.engine.market_maker.stats.quotes_cancelled, 1)


if __name__ == "__main__":
    unittest.main()
