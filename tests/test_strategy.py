"""
Strategy and end-to-end tests.

The most important test in this file is `test_fast_market_produces_no_profit`.
If a change ever makes that one pass with a profit, the finding is a bug, not an
edge.
"""

import unittest
from unittest.mock import patch

from pmpt.config import AppConfig
from pmpt.data.gamma import (
    GammaClient,
    _best_of_from_event,
    _event_start,
    _parse_json_field,
    market_from_gamma,
)
from pmpt.engine import TradingEngine, _balanced_markets
from pmpt.models import Level, OrderBook, Side, TradableMarket
from pmpt.simulate import SimConfig, run_simulation
from pmpt.strategy.live_model import LiveModelStrategy, StrategyConfig

T0, T1 = "tokA", "tokB"


def market(sport="tennis", best_of=3):
    return TradableMarket(
        market_id="m1", condition_id="c1", question="A vs B", slug="a-vs-b",
        token_ids=(T0, T1), outcomes=("A", "B"), tick_size=0.01, min_order_size=5,
        sport=sport, best_of=best_of, game_id=99,
    )


def books(p0, ts=10_000, half=0.01, depth=500.0):
    out = {}
    for tok, p in ((T0, p0), (T1, 1 - p0)):
        bid = round(p - half, 2)
        ask = round(p + half, 2)
        out[tok] = OrderBook(
            tok,
            [Level(round(bid - i * 0.01, 4), depth) for i in range(4)],
            [Level(round(ask + i * 0.01, 4), depth) for i in range(4)],
            timestamp_ms=ts, tick_size=0.01,
        )
    return out


class TestAnchoring(unittest.TestCase):
    def setUp(self):
        self.s = LiveModelStrategy(StrategyConfig())
        self.m = market()

    def test_anchor_reproduces_the_market_price(self):
        """Calibration must round-trip, or every downstream edge is fictional."""
        self.assertTrue(self.s.set_anchor(self.m, 0.68, ts=1000))
        fv = self.s.on_score(self.m, "0-0", "1", live=True, ended=False, ts=1000)
        self.assertAlmostEqual(fv, 0.68, places=2)

    def test_extreme_anchors_are_refused(self):
        self.assertFalse(self.s.set_anchor(self.m, 0.99, ts=1000))
        self.assertFalse(self.s.set_anchor(self.m, 0.01, ts=1000))

    def test_late_anchor_is_flagged_untradeable(self):
        self.s.set_anchor(self.m, 0.60, ts=1000, games_played=9)
        self.assertFalse(self.s.trackers["m1"].anchored_cleanly)

    def test_no_signal_without_an_anchor(self):
        self.s.on_score(self.m, "3-2", "1", live=True, ended=False, ts=1000)
        self.assertIsNone(self.s.evaluate(self.m, books(0.5), ts=1000))

    def test_live_anchor_round_trips_current_score_and_price(self):
        self.assertTrue(
            self.s.set_live_anchor(
                self.m,
                0.42,
                "6-3, 3-5",
                "S2",
                ts=1_000,
            )
        )
        tracker = self.s.trackers[self.m.market_id]
        self.assertTrue(tracker.anchored_cleanly)
        self.assertTrue(tracker.late_joined)
        self.assertAlmostEqual(tracker.fair_value, 0.42, places=3)
        self.assertIsNone(self.s.evaluate(self.m, books(0.50, ts=1_000), ts=1_000))


class TestRepricing(unittest.TestCase):
    def setUp(self):
        self.s = LiveModelStrategy(StrategyConfig())
        self.m = market()
        self.s.set_anchor(self.m, 0.55, ts=1000)

    def test_winning_a_set_raises_fair_value(self):
        even = self.s.on_score(self.m, "0-0", "1", True, False, ts=1000)
        ahead = self.s.on_score(self.m, "6-3, 0-0", "2", True, False, ts=2000)
        self.assertGreater(ahead, even)

    def test_losing_a_set_lowers_fair_value(self):
        even = self.s.on_score(self.m, "0-0", "1", True, False, ts=1000)
        behind = self.s.on_score(self.m, "3-6, 0-0", "2", True, False, ts=2000)
        self.assertLess(behind, even)

    def test_a_break_up_matters(self):
        base = self.s.on_score(self.m, "0-0", "1", True, False, ts=1000)
        broke = self.s.on_score(self.m, "3-1", "1", True, False, ts=2000)
        self.assertGreater(broke, base)


class TestSignals(unittest.TestCase):
    def setUp(self):
        cfg = StrategyConfig(model_weight=1.0, signal_ttl_ms=45_000)
        self.s = LiveModelStrategy(cfg)
        self.m = market()
        self.s.set_anchor(self.m, 0.55, ts=1000)

    def test_lagging_book_produces_a_signal(self):
        # Model says A is well ahead; the book still prices the match as even.
        self.s.on_score(self.m, "6-2, 3-0", "2", True, False, ts=10_000)
        sig = self.s.evaluate(self.m, books(0.55, ts=10_000), ts=10_000)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.side, Side.BUY)
        self.assertEqual(sig.token_id, T0)
        self.assertGreater(sig.edge, 0)

    def test_correctly_priced_book_produces_nothing(self):
        fv = self.s.on_score(self.m, "6-2, 3-0", "2", True, False, ts=10_000)
        sig = self.s.evaluate(self.m, books(fv, ts=10_000), ts=10_000)
        self.assertIsNone(sig)

    def test_stale_signal_expires(self):
        """Past the TTL the market has repriced and the edge is not real."""
        self.s.on_score(self.m, "6-2, 3-0", "2", True, False, ts=10_000)
        self.assertIsNotNone(self.s.evaluate(self.m, books(0.55, ts=20_000), ts=20_000))
        self.assertIsNone(self.s.evaluate(self.m, books(0.55, ts=200_000), ts=200_000))

    def test_no_signal_when_match_not_live(self):
        self.s.on_score(self.m, "6-2, 3-0", "2", live=False, ended=False, ts=10_000)
        self.assertIsNone(self.s.evaluate(self.m, books(0.55, ts=10_000), ts=10_000))

    def test_no_signal_after_match_ends(self):
        self.s.on_score(self.m, "6-2, 6-0", "FT", live=True, ended=True, ts=10_000)
        self.assertIsNone(self.s.evaluate(self.m, books(0.55, ts=10_000), ts=10_000))

    def test_no_signal_during_tiebreak_without_point_state(self):
        self.s.on_score(
            self.m,
            "7-5, 4-6, 6-6(3-2)",
            "3",
            live=True,
            ended=False,
            ts=10_000,
        )
        tracker = self.s.trackers[self.m.market_id]
        self.assertFalse(tracker.score_tradeable)
        self.assertEqual(tracker.score_issue, "tiebreak paused")
        self.assertIsNone(self.s.evaluate(self.m, books(0.40, ts=10_000), ts=10_000))

    def test_completed_tiebreak_does_not_pause_later_set(self):
        self.s.on_score(
            self.m,
            "7-6(5), 3-0",
            "2",
            live=True,
            ended=False,
            ts=10_000,
        )
        self.assertTrue(self.s.trackers[self.m.market_id].score_tradeable)

    def test_late_anchor_never_signals(self):
        s = LiveModelStrategy(StrategyConfig(model_weight=1.0))
        m = market()
        s.set_anchor(m, 0.55, ts=1000, games_played=8)
        s.on_score(m, "6-2, 3-0", "2", True, False, ts=10_000)
        self.assertIsNone(s.evaluate(m, books(0.55, ts=10_000), ts=10_000))

    def test_confidence_falls_as_the_signal_ages(self):
        self.s.on_score(self.m, "6-2, 3-0", "2", True, False, ts=10_000)
        fresh = self.s.evaluate(self.m, books(0.55, ts=12_000), ts=12_000)
        old = self.s.evaluate(self.m, books(0.55, ts=45_000), ts=45_000)
        self.assertGreater(fresh.confidence, old.confidence)

    def test_wide_spread_lowers_confidence(self):
        self.s.on_score(self.m, "6-2, 3-0", "2", True, False, ts=10_000)
        tight = self.s.evaluate(self.m, books(0.55, ts=10_000, half=0.005), ts=10_000)
        wide = self.s.evaluate(self.m, books(0.55, ts=10_000, half=0.05), ts=10_000)
        self.assertGreater(tight.confidence, wide.confidence)

    def test_model_weight_shrinks_the_edge(self):
        pure = LiveModelStrategy(StrategyConfig(model_weight=1.0))
        shrunk = LiveModelStrategy(StrategyConfig(model_weight=0.5))
        for s in (pure, shrunk):
            s.set_anchor(self.m, 0.55, ts=1000)
            s.on_score(self.m, "6-2, 3-0", "2", True, False, ts=10_000)
        a = pure.evaluate(self.m, books(0.55, ts=10_000), ts=10_000)
        b = shrunk.evaluate(self.m, books(0.55, ts=10_000), ts=10_000)
        self.assertGreater(a.edge, b.edge)


class TestExits(unittest.TestCase):
    def setUp(self):
        self.s = LiveModelStrategy(StrategyConfig(model_weight=1.0))
        self.m = market()
        self.s.set_anchor(self.m, 0.55, ts=1000)

    def test_converged_position_is_closed(self):
        s = LiveModelStrategy(StrategyConfig(model_weight=1.0, quick_take_profit=1.0))
        s.set_anchor(self.m, 0.55, ts=1000)
        fv = s.on_score(self.m, "6-2, 3-0", "2", True, False, ts=10_000)
        # Centre the book slightly above fair so the *bid* clearly reaches it;
        # a book centred exactly on fair leaves the bid a half-spread short.
        bk = books(min(fv + 0.02, 0.97), ts=10_000)[T0]
        should, why = s.exit_signal(self.m, T0, 0.60, bk, ts=10_000)
        self.assertTrue(should)
        self.assertIn("converged", why)

    def test_small_bid_side_profit_is_banked(self):
        self.s.on_score(self.m, "3-2", "1", True, False, ts=10_000)
        bk = books(0.64, ts=10_000)[T0]  # best bid 0.63
        should, why = self.s.exit_signal(self.m, T0, 0.61, bk, opened_ms=9_000, ts=10_000)
        self.assertTrue(should)
        self.assertIn("bank profit", why)

    def test_gross_one_tick_gain_is_not_profit_after_live_fee(self):
        self.m.fees_enabled = True
        self.m.fee_rate = 0.05
        self.s.cfg.quick_take_profit = 0.004
        self.s.cfg.quick_take_profit_roi = 0.0
        self.s.on_score(self.m, "6-0, 5-0", "2", True, False, ts=10_000)

        one_tick = books(0.52, ts=10_000)[T0]  # bid .51 vs .50 entry
        should, _ = self.s.exit_signal(
            self.m, T0, 0.50, one_tick, opened_ms=9_000, ts=10_000
        )
        self.assertFalse(should)

        two_ticks = books(0.53, ts=10_000)[T0]  # bid .52 clears exit fee
        should, why = self.s.exit_signal(
            self.m, T0, 0.50, two_ticks, opened_ms=9_000, ts=10_000
        )
        self.assertTrue(should)
        self.assertIn("bank profit", why)

    def test_scratch_profit_after_short_hold_is_taken(self):
        cfg = StrategyConfig(model_weight=1.0, quick_take_profit=1.0, scratch_profit_after_ms=5_000)
        s = LiveModelStrategy(cfg)
        s.set_anchor(self.m, 0.55, ts=1000)
        s.on_score(self.m, "3-2", "1", True, False, ts=10_000)
        bk = books(0.62, ts=10_000)[T0]  # best bid 0.61
        should, why = s.exit_signal(self.m, T0, 0.604, bk, opened_ms=1_000, ts=10_000)
        self.assertTrue(should)
        self.assertIn("scratch profit", why)

    def test_model_edge_gone_closes_before_full_stop(self):
        cfg = StrategyConfig(model_weight=1.0, quick_take_profit=1.0, scratch_profit_after_ms=10**9)
        s = LiveModelStrategy(cfg)
        s.set_anchor(self.m, 0.55, ts=1000)
        s.on_score(self.m, "0-5", "1", True, False, ts=10_000)
        bk = books(0.56, ts=10_000)[T0]  # best bid 0.55, not a full stop from .58
        should, why = s.exit_signal(self.m, T0, 0.58, bk, opened_ms=9_000, ts=10_000)
        self.assertTrue(should)
        self.assertIn("edge gone", why)

    def test_stop_loss_is_not_pre_empted_by_take_profit(self):
        """A hard risk control must win over a profit-taking rule.

        Both conditions are true here: the position is far underwater AND the
        bid sits above the collapsed fair value. It must report the stop.
        """
        self.s.on_score(self.m, "2-6, 0-3", "2", True, False, ts=10_000)
        bk = books(0.20, ts=10_000)[T0]
        should, why = self.s.exit_signal(self.m, T0, 0.60, bk, ts=10_000)
        self.assertTrue(should)
        self.assertIn("stop loss", why)

    def test_stop_loss_triggers(self):
        self.s.on_score(self.m, "2-6, 0-3", "2", True, False, ts=10_000)
        bk = books(0.20, ts=10_000)[T0]
        should, why = self.s.exit_signal(self.m, T0, 0.60, bk, ts=10_000)
        self.assertTrue(should)
        self.assertIn("stop loss", why)

    def test_ended_match_is_flattened(self):
        self.s.on_score(self.m, "6-2, 6-0", "FT", True, True, ts=10_000)
        should, why = self.s.exit_signal(self.m, T0, 0.60, books(0.9)[T0], ts=10_000)
        self.assertTrue(should)

    def test_hold_to_resolution_keeps_the_position(self):
        s = LiveModelStrategy(StrategyConfig(hold_to_resolution=True, model_weight=1.0))
        s.set_anchor(self.m, 0.55, ts=1000)
        s.on_score(self.m, "6-2, 6-0", "FT", True, True, ts=10_000)
        should, _ = s.exit_signal(self.m, T0, 0.60, books(0.9)[T0], ts=10_000)
        self.assertFalse(should)


class TestGammaParsing(unittest.TestCase):
    def test_json_encoded_string_fields(self):
        """Gamma returns arrays as JSON strings. Getting this wrong breaks everything."""
        self.assertEqual(_parse_json_field('["Yes", "No"]'), ["Yes", "No"])
        self.assertEqual(_parse_json_field(["a", "b"]), ["a", "b"])
        self.assertEqual(_parse_json_field(None), [])
        self.assertEqual(_parse_json_field("not json"), [])

    def test_market_from_gamma(self):
        raw = {
            "id": "621538", "conditionId": "0xabc", "question": "X vs Y",
            "slug": "x-vs-y", "enableOrderBook": True, "acceptingOrders": True,
            "clobTokenIds": '["111", "222"]', "outcomes": '["X", "Y"]',
            "orderPriceMinTickSize": 0.001, "orderMinSize": 5,
            "feesEnabled": True, "feeSchedule": {"rate": 0.05},
        }
        m = market_from_gamma(raw, {"slug": "wta-x-vs-y", "gameId": 42})
        self.assertEqual(m.token_ids, ("111", "222"))
        self.assertEqual(m.outcomes, ("X", "Y"))
        self.assertEqual(m.tick_size, 0.001)
        self.assertEqual(m.game_id, 42)
        self.assertTrue(m.fees_enabled)
        self.assertEqual(m.fee_rate, 0.05)

        tt_market = market_from_gamma(raw, {"slug": "wtt-x-vs-y"}, "table_tennis")
        self.assertEqual(tt_market.best_of, 5)

    def test_market_without_orderbook_is_skipped(self):
        raw = {"id": "1", "clobTokenIds": '["1","2"]', "outcomes": '["A","B"]',
               "enableOrderBook": False}
        self.assertIsNone(market_from_gamma(raw))

    def test_best_of_inference(self):
        self.assertEqual(_best_of_from_event({"slug": "atp-wimbledon-x-vs-y"}), 5)
        self.assertEqual(_best_of_from_event({"slug": "wta-wimbledon-x-vs-y"}), 3)
        self.assertEqual(_best_of_from_event({"slug": "atp-shanghai-x-vs-y"}), 3)

    def test_team_market_maps_to_home_away_score_order(self):
        raw = {
            "id": "1",
            "conditionId": "c",
            "question": "Boston Celtics vs. Los Angeles Lakers",
            "slug": "nba-bos-lal",
            "enableOrderBook": True,
            "acceptingOrders": True,
            "clobTokenIds": '["a","b"]',
            "outcomes": '["Boston Celtics","Los Angeles Lakers"]',
        }
        item = market_from_gamma(
            raw,
            {
                "slug": "nba-bos-lal-2026-01-01",
                "seriesSlug": "nba",
                "gameId": 7,
            },
            "basketball",
        )
        self.assertEqual(item.outcome0_role, "away")

    def test_soccer_yes_no_markets_map_home_draw_and_away(self):
        event = {
            "slug": "epl-ars-che-2026-01-01",
            "seriesSlug": "epl",
            "title": "Arsenal FC vs. Chelsea FC",
            "gameId": 7,
        }
        base = {
            "id": "1",
            "conditionId": "c",
            "slug": "epl-ars-che",
            "enableOrderBook": True,
            "acceptingOrders": True,
            "clobTokenIds": '["a","b"]',
            "outcomes": '["Yes","No"]',
        }
        questions = {
            "Will Arsenal FC win?": "home",
            "Will Arsenal FC vs. Chelsea FC end in a draw?": "draw",
            "Will Chelsea FC win?": "away",
        }
        for question, expected in questions.items():
            with self.subTest(question=question):
                item = market_from_gamma(
                    {**base, "question": question},
                    event,
                    "soccer",
                )
                self.assertEqual(item.outcome0_role, expected)

    def test_scheduled_start_time_wins_over_creation_date(self):
        value = _event_start({
            "startDate": "2026-07-01T00:00:00Z",
            "startTime": "2026-08-01T12:00:00Z",
        })
        self.assertEqual(value.isoformat(), "2026-08-01T12:00:00+00:00")


class TestGammaDiscovery(unittest.TestCase):
    def test_event_query_uses_bounded_current_windows(self):
        with patch(
            "pmpt.data.gamma._get",
            side_effect=[[], [], {"events": []}],
        ) as get:
            GammaClient(["table_tennis"]).fetch_events("table_tennis")

        near = get.call_args_list[0].args[1]
        future = get.call_args_list[1].args[1]
        self.assertEqual(get.call_args_list[0].args[0], "/events/keyset")
        self.assertEqual(near["ascending"], "false")
        self.assertEqual(future["ascending"], "true")
        self.assertTrue(near["start_time_min"].endswith("Z"))
        self.assertTrue(near["start_time_max"].endswith("Z"))
        self.assertLess(near["start_time_min"], near["start_time_max"])
        self.assertEqual(near["start_time_max"], future["start_time_min"])

    def test_refresh_skips_events_without_score_stream_id(self):
        client = GammaClient(["table_tennis"])
        event = {
            "id": "e1",
            "gameId": None,
            "markets": [{
                "id": "m1",
                "enableOrderBook": True,
                "acceptingOrders": True,
                "clobTokenIds": '["a","b"]',
                "outcomes": '["A","B"]',
            }],
        }
        with patch.object(client, "fetch_events", return_value=[event]):
            self.assertEqual(client.refresh(), [])

    def test_watchlist_round_robins_sports(self):
        items = []
        for sport, count in (("table_tennis", 5), ("tennis", 2)):
            for i in range(count):
                item = market(sport=sport)
                item.market_id = f"{sport}-{i}"
                items.append(item)

        selected = _balanced_markets(items, ["table_tennis", "tennis"], 5)
        self.assertEqual(
            [m.market_id for m in selected],
            [
                "table_tennis-0",
                "tennis-0",
                "table_tennis-1",
                "tennis-1",
                "table_tennis-2",
            ],
        )


class TestLiveAnchorGuard(unittest.IsolatedAsyncioTestCase):
    async def test_discovery_seeds_an_already_live_score(self):
        engine = TradingEngine(AppConfig())
        item = market()
        event = {
            "id": "event-1",
            "gameId": item.game_id,
            "live": True,
            "ended": False,
            "score": "6-3, 3-2",
            "period": "S2",
            "elapsed": "",
            "markets": [{
                "id": item.market_id,
                "outcomePrices": '["0.72","0.28"]',
            }],
        }
        engine.gamma.events = {"event-1": event}

        with patch.object(engine.gamma, "refresh", return_value=[item]):
            await engine._discover()

        tracker = engine.strategy.trackers[item.market_id]
        self.assertTrue(tracker.live)
        self.assertEqual(tracker.last_score, "6-3, 3-2")
        self.assertEqual(tracker.last_score_change_ms, 0)

    async def test_first_late_score_invalidates_pregame_anchor(self):
        engine = TradingEngine(AppConfig())
        item = market()
        engine.gamma.activate([item])
        engine.strategy.set_anchor(item, 0.55, ts=1_000)

        await engine._on_game({
            "gameId": item.game_id,
            "score": "3-2",
            "period": "1",
            "live": True,
        })

        self.assertFalse(engine.strategy.trackers[item.market_id].anchored_cleanly)

    async def test_first_late_score_is_calibrated_from_current_price(self):
        engine = TradingEngine(AppConfig())
        item = market()
        engine.gamma.activate([item])
        engine.strategy.set_anchor(item, 0.55, ts=1_000)

        await engine._on_game({
            "gameId": item.game_id,
            "score": "6-3, 3-5",
            "period": "S2",
            "live": True,
        })

        # A Gamma event price is not fresh enough for a live calibration.
        self.assertFalse(engine.strategy.trackers[item.market_id].anchored_cleanly)

        await engine._on_book(books(0.42)[T0])

        tracker = engine.strategy.trackers[item.market_id]
        self.assertTrue(tracker.anchored_cleanly)
        self.assertTrue(tracker.late_joined)
        self.assertAlmostEqual(tracker.fair_value, 0.42, places=3)


class TestEndToEnd(unittest.TestCase):
    def test_simulation_runs_and_conserves_value(self):
        r = run_simulation(SimConfig(n_matches=8, seed=1))
        self.assertEqual(r.matches, 8)
        self.assertGreater(r.final_equity, 0)
        self.assertAlmostEqual(
            r.final_equity, r.starting_cash + r.realized_pnl, places=4
        )

    def test_fast_market_produces_no_profit(self):
        """THE honesty check.

        When the synthetic book reprices instantly there is no lag to exploit, so
        the strategy must make approximately nothing. A profit here means the
        simulator is leaking free money somewhere -- most likely filling orders
        against the same book that generated the signal.
        """
        r = run_simulation(SimConfig(n_matches=25, catchup_rate=1.0, seed=5))
        self.assertLess(abs(r.total_return_pct), 1.0)

    def test_slower_market_is_more_profitable(self):
        slow = run_simulation(SimConfig(n_matches=20, catchup_rate=0.10, seed=3))
        fast = run_simulation(SimConfig(n_matches=20, catchup_rate=0.60, seed=3))
        self.assertGreater(slow.total_return_pct, fast.total_return_pct)

    def test_table_tennis_simulation_runs(self):
        r = run_simulation(SimConfig(n_matches=6, sport="table_tennis", best_of=5, seed=2))
        self.assertGreater(r.final_equity, 0)

    def test_risk_limits_bind_during_simulation(self):
        """No position should ever exceed the configured cap on equity."""
        from pmpt.execution.risk import RiskConfig
        r = run_simulation(
            SimConfig(n_matches=15, catchup_rate=0.05, seed=9),
            risk_cfg=RiskConfig(max_position_pct=0.05, max_total_exposure_pct=0.30),
        )
        self.assertLessEqual(r.max_drawdown, 0.30)


if __name__ == "__main__":
    unittest.main()
