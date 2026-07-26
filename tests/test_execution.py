"""
Tests for the broker, portfolio and risk manager.

The point of these is to prove the simulator is not flattering us: that latency
is really enforced, that thin books really cause partial fills, and that the risk
caps really bind.
"""

import unittest

from pmpt.execution.paper_broker import BrokerConfig, PaperBroker, polymarket_fee
from pmpt.execution.portfolio import Portfolio
from pmpt.execution.risk import RiskConfig, RiskManager
from pmpt.models import (
    Fill,
    Level,
    Order,
    OrderBook,
    OrderStatus,
    OrderType,
    Position,
    Side,
    Signal,
    TradableMarket,
)

TOK = "tokenA"


def book(bid=0.49, ask=0.51, depth=100.0, ts=1000, levels=3):
    bids = [Level(round(bid - i * 0.01, 4), depth) for i in range(levels)]
    asks = [Level(round(ask + i * 0.01, 4), depth) for i in range(levels)]
    return OrderBook(TOK, bids, asks, timestamp_ms=ts, tick_size=0.01)


def market(**kw):
    base = dict(
        market_id="m1", condition_id="c1", question="Q", slug="s",
        token_ids=(TOK, "tokenB"), outcomes=("A", "B"), tick_size=0.01,
        min_order_size=5,
    )
    base.update(kw)
    return TradableMarket(**base)


class TestOrderBook(unittest.TestCase):
    def test_mid_and_spread(self):
        b = book(0.49, 0.51)
        self.assertAlmostEqual(b.mid, 0.50)
        self.assertAlmostEqual(b.spread, 0.02)

    def test_crossed_book_is_invalid(self):
        b = OrderBook(TOK, [Level(0.60, 10)], [Level(0.55, 10)])
        self.assertFalse(b.is_valid())

    def test_empty_book_is_invalid(self):
        self.assertFalse(OrderBook(TOK).is_valid())

    def test_from_ws_sorts_levels(self):
        raw = {
            "asset_id": TOK,
            "timestamp": "1700000000000",
            "bids": [{"price": "0.08", "size": "10"}, {"price": "0.09", "size": "20"}],
            "asks": [{"price": "0.99", "size": "5"}, {"price": "0.98", "size": "7"}],
        }
        b = OrderBook.from_ws(raw)
        self.assertEqual(b.best_bid, 0.09)
        self.assertEqual(b.best_ask, 0.98)

    def test_from_ws_drops_zero_size_levels(self):
        raw = {"asset_id": TOK, "bids": [{"price": "0.5", "size": "0"}], "asks": []}
        self.assertEqual(OrderBook.from_ws(raw).bids, [])

    def test_depth_respects_price_limit(self):
        b = book(0.49, 0.51, depth=100, levels=3)  # asks at .51 .52 .53
        self.assertAlmostEqual(b.depth(Side.BUY, max_price=0.52), 200.0)


class TestFees(unittest.TestCase):
    def test_fee_matches_current_polymarket_curve(self):
        self.assertAlmostEqual(polymarket_fee(0.50, 100, 0.03), 0.75)
        self.assertAlmostEqual(polymarket_fee(0.20, 100, 0.03), 0.48)

    def test_fee_curve_is_symmetric(self):
        self.assertAlmostEqual(
            polymarket_fee(0.90, 100, 0.05), polymarket_fee(0.10, 100, 0.05)
        )

    def test_zero_rate_is_free(self):
        self.assertEqual(polymarket_fee(0.5, 100, 0.0), 0.0)

    def test_midprice_is_most_expensive(self):
        self.assertGreater(polymarket_fee(0.5, 100, 0.05), polymarket_fee(0.2, 100, 0.05))


class TestPaperBroker(unittest.TestCase):
    def setUp(self):
        self.b = PaperBroker(BrokerConfig(
            latency_ms=200, latency_jitter_ms=0, miss_probability=0.0,
            max_level_participation=1.0, seed=1,
        ))

    def test_latency_is_enforced(self):
        """An order must not fill against the book that triggered it."""
        self.b.on_book(book(ts=1000), ts_ms=1000)
        o = Order(TOK, Side.BUY, 10, 0.55, OrderType.MARKETABLE)
        self.b.submit(o, ts_ms=1000)
        # Same instant: nothing happens.
        fills = self.b.on_book(book(ts=1100), ts_ms=1100)
        self.assertEqual(fills, [])
        self.assertEqual(o.filled, 0)
        # After latency has elapsed it executes.
        fills = self.b.on_book(book(ts=1300), ts_ms=1300)
        self.assertEqual(len(fills), 1)
        self.assertEqual(o.status, OrderStatus.FILLED)

    def test_live_market_fee_overrides_simulation_fallback(self):
        self.b.set_market_fees((TOK,), taker_rate=0.05)
        self.b.on_book(book(ts=1000), ts_ms=1000)
        o = Order(TOK, Side.BUY, 10, 0.55, OrderType.MARKETABLE)
        self.b.submit(o, ts_ms=1000)
        fill = self.b.on_book(book(ts=1300), ts_ms=1300)[0]
        self.assertAlmostEqual(
            fill.fee,
            polymarket_fee(fill.price, fill.size, 0.05),
        )

    def test_fills_against_the_later_book_not_the_earlier_one(self):
        """The price moving during the latency window must hurt us."""
        self.b.on_book(book(0.49, 0.51, ts=1000), ts_ms=1000)
        o = Order(TOK, Side.BUY, 10, 0.60, OrderType.MARKETABLE)
        self.b.submit(o, ts_ms=1000)
        # Market gaps up before our order lands.
        fills = self.b.on_book(book(0.57, 0.59, ts=1300), ts_ms=1300)
        self.assertEqual(len(fills), 1)
        self.assertAlmostEqual(fills[0].price, 0.59)  # not 0.51

    def test_walks_the_book_for_vwap(self):
        self.b.on_book(book(ts=1000), ts_ms=1000)
        o = Order(TOK, Side.BUY, 250, 0.60, OrderType.MARKETABLE)  # 3 levels x 100
        self.b.submit(o, ts_ms=1000)
        f = self.b.on_book(book(ts=1300), ts_ms=1300)[0]
        # 100@0.51 + 100@0.52 + 50@0.53
        expected = (100 * 0.51 + 100 * 0.52 + 50 * 0.53) / 250
        self.assertAlmostEqual(f.price, expected, places=9)
        self.assertGreater(f.price, 0.51)  # slippage is real

    def test_partial_fill_when_depth_is_insufficient(self):
        self.b.on_book(book(depth=10, levels=1, ts=1000), ts_ms=1000)
        o = Order(TOK, Side.BUY, 500, 0.60, OrderType.MARKETABLE)
        self.b.submit(o, ts_ms=1000)
        f = self.b.on_book(book(depth=10, levels=1, ts=1300), ts_ms=1300)[0]
        self.assertAlmostEqual(f.size, 10.0)
        self.assertEqual(o.status, OrderStatus.PARTIAL)

    def test_never_pays_through_the_limit(self):
        self.b.on_book(book(ts=1000), ts_ms=1000)
        o = Order(TOK, Side.BUY, 250, 0.51, OrderType.MARKETABLE)  # only level 1 qualifies
        self.b.submit(o, ts_ms=1000)
        f = self.b.on_book(book(ts=1300), ts_ms=1300)[0]
        self.assertAlmostEqual(f.price, 0.51)
        self.assertAlmostEqual(f.size, 100.0)

    def test_participation_cap_limits_take(self):
        b = PaperBroker(BrokerConfig(latency_ms=0, latency_jitter_ms=0,
                                     miss_probability=0.0, max_level_participation=0.5,
                                     seed=1))
        b.on_book(book(depth=100, levels=1, ts=1000), ts_ms=1000)
        o = Order(TOK, Side.BUY, 100, 0.60, OrderType.MARKETABLE)
        b.submit(o, ts_ms=1000)
        f = b.on_book(book(depth=100, levels=1, ts=1001), ts_ms=1001)[0]
        self.assertAlmostEqual(f.size, 50.0)

    def test_passive_order_needs_queue_to_clear(self):
        """A resting order does not fill just because a trade happened."""
        self.b.on_book(book(depth=100, ts=1000), ts_ms=1000)
        o = Order(TOK, Side.BUY, 10, 0.49, OrderType.PASSIVE)
        self.b.submit(o, ts_ms=1000)
        self.b.on_book(book(depth=100, ts=1300), ts_ms=1300)  # order arrives, rests
        self.assertAlmostEqual(o.queue_ahead, 100.0)

        # 60 shares trade: still behind 40 in the queue.
        self.assertEqual(self.b.on_trade(TOK, 0.49, 60, 1400), [])
        self.assertEqual(o.filled, 0)
        # 50 more: 40 clears the queue, 10 fills us.
        fills = self.b.on_trade(TOK, 0.49, 50, 1500)
        self.assertEqual(len(fills), 1)
        self.assertAlmostEqual(fills[0].size, 10.0)
        self.assertEqual(fills[0].liquidity, "maker")

    def test_passive_order_ignores_trades_at_worse_prices(self):
        self.b.on_book(book(ts=1000), ts_ms=1000)
        o = Order(TOK, Side.BUY, 10, 0.45, OrderType.PASSIVE)
        self.b.submit(o, ts_ms=1000)
        self.b.on_book(book(ts=1300), ts_ms=1300)
        self.assertEqual(self.b.on_trade(TOK, 0.50, 1000, 1400), [])

    def test_missed_orders_are_possible(self):
        b = PaperBroker(BrokerConfig(latency_ms=0, latency_jitter_ms=0,
                                     miss_probability=1.0, seed=3))
        b.on_book(book(ts=1000), ts_ms=1000)
        o = Order(TOK, Side.BUY, 10, 0.60, OrderType.MARKETABLE)
        b.submit(o, ts_ms=1000)
        self.assertEqual(b.on_book(book(ts=1001), ts_ms=1001), [])
        self.assertEqual(o.status, OrderStatus.CANCELLED)

    def test_rejects_when_no_book(self):
        b = PaperBroker(BrokerConfig(latency_ms=0, latency_jitter_ms=0, seed=1))
        o = Order("unknown", Side.BUY, 10, 0.5, OrderType.MARKETABLE)
        b.submit(o, ts_ms=1000)
        b.on_book(OrderBook("unknown"), ts_ms=1001)
        self.assertEqual(o.status, OrderStatus.REJECTED)

    def test_zero_size_rejected(self):
        o = Order(TOK, Side.BUY, 0, 0.5)
        self.b.submit(o)
        self.assertEqual(o.status, OrderStatus.REJECTED)

    def test_cancel_removes_order(self):
        self.b.on_book(book(ts=1000), ts_ms=1000)
        o = Order(TOK, Side.BUY, 10, 0.49, OrderType.PASSIVE)
        self.b.submit(o, ts_ms=1000)
        self.assertTrue(self.b.cancel(o.order_id))
        self.assertEqual(o.status, OrderStatus.CANCELLED)

    def test_live_size_counts_only_matching_open_orders(self):
        self.b.on_book(book(ts=1000), ts_ms=1000)
        buy = Order(TOK, Side.BUY, 10, 0.55, OrderType.MARKETABLE)
        sell = Order(TOK, Side.SELL, 7, 0.45, OrderType.MARKETABLE)
        other = Order("tokenB", Side.SELL, 3, 0.45, OrderType.MARKETABLE)
        self.b.submit(buy, ts_ms=1000)
        self.b.submit(sell, ts_ms=1000)
        self.b.submit(other, ts_ms=1000)

        self.assertAlmostEqual(self.b.live_size(TOK), 17.0)
        self.assertAlmostEqual(self.b.live_size(TOK, Side.SELL), 7.0)
        self.assertAlmostEqual(self.b.live_size(side=Side.SELL), 10.0)

        self.assertEqual(self.b.cancel_all(side=Side.SELL), 2)
        self.assertAlmostEqual(self.b.live_size(side=Side.SELL), 0.0)
        self.assertAlmostEqual(self.b.live_size(side=Side.BUY), 10.0)


class TestPosition(unittest.TestCase):
    def test_open_and_average(self):
        p = Position(TOK)
        p.apply(Fill("1", TOK, Side.BUY, 0.40, 100))
        p.apply(Fill("2", TOK, Side.BUY, 0.60, 100))
        self.assertAlmostEqual(p.shares, 200)
        self.assertAlmostEqual(p.avg_cost, 0.50)

    def test_close_realizes_pnl(self):
        p = Position(TOK)
        p.apply(Fill("1", TOK, Side.BUY, 0.40, 100))
        r = p.apply(Fill("2", TOK, Side.SELL, 0.55, 100))
        self.assertAlmostEqual(r, 15.0)
        self.assertAlmostEqual(p.shares, 0.0)

    def test_partial_close(self):
        p = Position(TOK)
        p.apply(Fill("1", TOK, Side.BUY, 0.40, 100))
        r = p.apply(Fill("2", TOK, Side.SELL, 0.50, 40))
        self.assertAlmostEqual(r, 4.0)
        self.assertAlmostEqual(p.shares, 60)
        self.assertAlmostEqual(p.avg_cost, 0.40)  # basis unchanged


class TestPortfolio(unittest.TestCase):
    def test_cash_moves_correctly(self):
        pf = Portfolio(100.0)
        pf.apply_fill(Fill("1", TOK, Side.BUY, 0.50, 100))
        self.assertAlmostEqual(pf.cash, 50.0)
        pf.apply_fill(Fill("2", TOK, Side.SELL, 0.60, 100))
        self.assertAlmostEqual(pf.cash, 110.0)
        self.assertAlmostEqual(pf.realized_pnl, 10.0)
        self.assertNotIn(TOK, pf.positions)

    def test_oversized_sell_cannot_flip_short(self):
        pf = Portfolio(100.0)
        pf.apply_fill(Fill("1", TOK, Side.BUY, 0.50, 10))
        pf.apply_fill(Fill("2", TOK, Side.SELL, 0.60, 25))

        self.assertAlmostEqual(pf.cash, 101.0)
        self.assertAlmostEqual(pf.realized_pnl, 1.0)
        self.assertNotIn(TOK, pf.positions)

    def test_sell_without_position_is_ignored(self):
        pf = Portfolio(100.0)
        pf.apply_fill(Fill("1", TOK, Side.SELL, 0.60, 10))

        self.assertAlmostEqual(pf.cash, 100.0)
        self.assertEqual(pf.realized_pnl, 0.0)
        self.assertEqual(pf.fills, [])

    def test_fees_reduce_cash(self):
        pf = Portfolio(100.0)
        pf.apply_fill(Fill("1", TOK, Side.BUY, 0.50, 100, fee=1.0))
        self.assertAlmostEqual(pf.cash, 49.0)
        self.assertAlmostEqual(pf.fees_paid, 1.0)

    def test_settlement_at_one(self):
        pf = Portfolio(100.0)
        pf.apply_fill(Fill("1", TOK, Side.BUY, 0.40, 100))
        pnl = pf.settle(TOK, 1.0)
        self.assertAlmostEqual(pnl, 60.0)
        self.assertAlmostEqual(pf.cash, 160.0)
        self.assertEqual(len(pf.positions), 0)

    def test_settlement_at_zero(self):
        pf = Portfolio(100.0)
        pf.apply_fill(Fill("1", TOK, Side.BUY, 0.40, 100))
        pnl = pf.settle(TOK, 0.0)
        self.assertAlmostEqual(pnl, -40.0)
        self.assertAlmostEqual(pf.cash, 60.0)

    def test_liquidation_equity_is_below_mid_equity(self):
        """Marking at mid flatters you. The liquidation value is the honest one."""
        pf = Portfolio(100.0)
        pf.apply_fill(Fill("1", TOK, Side.BUY, 0.50, 100))
        self.assertGreater(pf.equity({TOK: 0.50}), pf.liquidation_equity({TOK: 0.48}))

    def test_drawdown_tracking(self):
        pf = Portfolio(100.0)
        pf.mark({})
        pf.cash = 120.0
        pf.mark({})
        pf.cash = 90.0
        pf.mark({})
        self.assertAlmostEqual(pf.max_drawdown, 0.25, places=6)

    def test_exposure(self):
        pf = Portfolio(100.0)
        pf.apply_fill(Fill("1", TOK, Side.BUY, 0.40, 50), market_id="m1")
        self.assertAlmostEqual(pf.exposure(), 20.0)
        self.assertAlmostEqual(pf.exposure_in_market("m1"), 20.0)
        self.assertAlmostEqual(pf.exposure_in_market("other"), 0.0)


class TestRisk(unittest.TestCase):
    def setUp(self):
        self.pf = Portfolio(100.0)
        self.m = market()
        self.rm = RiskManager(RiskConfig(min_edge=0.03, cooldown_ms=0))

    def sig(self, price=0.50, fair=0.60, conf=1.0):
        return Signal(TOK, "m1", fair, price, fair - price, Side.BUY, conf)

    def approve(self, signal, **kw):
        args = dict(book_age_ms=0, spread=0.02, depth=500.0, ts_ms=10_000)
        args.update(kw)
        return self.rm.approve(signal, self.m, self.pf, {}, **args)

    def test_good_signal_is_approved(self):
        ok, why = self.approve(self.sig())
        self.assertTrue(ok, why)

    def test_small_edge_rejected(self):
        ok, _ = self.approve(self.sig(price=0.50, fair=0.51))
        self.assertFalse(ok)

    def test_passive_entry_only_pays_one_expected_taker_leg(self):
        fee_market = market(fees_enabled=True, fee_rate=0.05)
        rm = RiskManager(RiskConfig(
            min_edge=0.006,
            assumed_slippage=0.0,
            min_confidence=0.0,
            cooldown_ms=0,
        ))
        signal = self.sig(price=0.50, fair=0.53)
        common = (signal, fee_market, self.pf, {}, 0, 0.02, 500.0, 10_000)

        marketable_ok, _ = rm.approve(*common, taker_legs=2)
        passive_ok, passive_why = rm.approve(*common, taker_legs=1)

        self.assertFalse(marketable_ok)
        self.assertTrue(passive_ok, passive_why)

    def test_stale_book_rejected(self):
        ok, why = self.approve(self.sig(), book_age_ms=60_000)
        self.assertFalse(ok)
        self.assertIn("stale", why)

    def test_wide_spread_rejected(self):
        ok, why = self.approve(self.sig(), spread=0.20)
        self.assertFalse(ok)
        self.assertIn("spread", why)

    def test_thin_book_rejected(self):
        ok, why = self.approve(self.sig(), depth=1.0)
        self.assertFalse(ok)
        self.assertIn("depth", why)

    def test_extreme_prices_rejected(self):
        ok, _ = self.approve(self.sig(price=0.99, fair=0.999))
        self.assertFalse(ok)
        ok, _ = self.approve(self.sig(price=0.01, fair=0.20))
        self.assertFalse(ok)

    def test_low_confidence_rejected(self):
        ok, _ = self.approve(self.sig(conf=0.1))
        self.assertFalse(ok)

    def test_cooldown_blocks_rapid_reentry(self):
        rm = RiskManager(RiskConfig(cooldown_ms=30_000))
        rm.record_entry("m1", 10_000)
        ok, why = rm.approve(self.sig(), self.m, self.pf, {}, 0, 0.02, 500.0, 20_000)
        self.assertFalse(ok)
        self.assertIn("cooldown", why)

    def test_drawdown_halts_trading(self):
        self.pf.mark({})
        self.pf.cash = 60.0
        self.pf.mark({})
        self.assertTrue(self.rm.check_halt(self.pf, {}, "d1"))
        self.assertIn("drawdown", self.rm.state.halt_reason)

    def test_halt_is_sticky(self):
        self.pf.cash = 5.0
        self.rm.check_halt(self.pf, {}, "d1")
        self.pf.cash = 1000.0
        self.assertTrue(self.rm.check_halt(self.pf, {}, "d1"))

    def test_daily_loss_limit(self):
        rm = RiskManager(RiskConfig(daily_loss_limit_pct=0.10, max_drawdown_pct=0.99))
        rm.check_halt(self.pf, {}, "day1")      # sets day baseline at 100
        self.pf.cash = 85.0
        self.assertTrue(rm.check_halt(self.pf, {}, "day1"))
        self.assertIn("daily loss", rm.state.halt_reason)


class TestSizing(unittest.TestCase):
    def setUp(self):
        self.pf = Portfolio(100.0)
        self.m = market()
        self.rm = RiskManager(RiskConfig(kelly_fraction=0.20, max_position_pct=0.10))

    def size(self, price=0.50, fair=0.60, depth=1000.0, conf=1.0):
        s = Signal(TOK, "m1", fair, price, fair - price, Side.BUY, conf)
        return self.rm.size_order(s, self.m, self.pf, {}, depth)

    def test_bigger_edge_means_bigger_size(self):
        self.assertGreater(self.size(fair=0.70), self.size(fair=0.55))

    def test_no_edge_means_no_trade(self):
        self.assertEqual(self.size(price=0.60, fair=0.55), 0.0)

    def test_capped_by_max_position_pct(self):
        # Huge edge would imply a huge Kelly bet; the cap must bind.
        shares = self.size(price=0.20, fair=0.95)
        self.assertLessEqual(shares * 0.20, 0.10 * 100.0 + 1e-6)

    def test_capped_by_available_depth(self):
        self.assertLessEqual(self.size(depth=7.0), 7.0)

    def test_respects_minimum_order_size(self):
        s = self.size(fair=0.505)
        self.assertTrue(s == 0.0 or s >= 5.0)

    def test_cannot_exceed_cash(self):
        self.pf.cash = 2.0
        self.assertEqual(self.size(price=0.50, fair=0.90), 0.0)

    def test_confidence_scales_size(self):
        self.assertLess(self.size(conf=0.6), self.size(conf=1.0))


if __name__ == "__main__":
    unittest.main()
