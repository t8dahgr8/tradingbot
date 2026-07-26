"""Team-sport score, clock, and calibrated probability tests."""

import unittest

from pmpt.models import TradableMarket
from pmpt.quant import team_sports as team
from pmpt.strategy.live_model import LiveModelStrategy, StrategyConfig


def market(sport: str, league: str, role: str = "home") -> TradableMarket:
    return TradableMarket(
        market_id=f"{sport}-{role}",
        condition_id="condition",
        question="Home vs Away",
        slug=f"{league}-home-away-2026-07-25",
        token_ids=("home-token", "other-token"),
        outcomes=("Home", "Away"),
        sport=sport,
        league=league,
        outcome0_role=role,
        game_id=42,
    )


class TestTeamScoreParsing(unittest.TestCase):
    def test_basketball_clock_is_time_remaining(self):
        state = team.parse_team_score(
            "basketball", "nba", "98-94", "Q4", "05:12"
        )
        self.assertTrue(state.valid)
        self.assertTrue(state.clock_known)
        self.assertAlmostEqual(state.progress, 42.8 / 48.0, places=4)

    def test_football_clock_is_time_remaining(self):
        state = team.parse_team_score(
            "football", "nfl", "3-16", "Q4", "05:18"
        )
        self.assertAlmostEqual(state.progress, 54.7 / 60.0, places=4)

    def test_soccer_clock_is_cumulative(self):
        state = team.parse_team_score(
            "soccer", "epl", "1-0", "1H", "32:15"
        )
        self.assertAlmostEqual(state.progress, 32.25 / 90.0, places=4)

    def test_baseball_inning_progress(self):
        state = team.parse_team_score(
            "baseball", "mlb", "3-2", "Top 7", ""
        )
        self.assertAlmostEqual(state.progress, 6.25 / 9.0, places=4)

    def test_invalid_score_is_rejected(self):
        state = team.parse_team_score(
            "hockey", "nhl", "not a score", "P2", "10:00"
        )
        self.assertFalse(state.valid)

    def test_plain_overtime_period_is_near_end_not_start(self):
        state = team.parse_team_score(
            "basketball", "nba", "110-110", "OT", "03:00"
        )
        self.assertGreater(state.progress, 0.95)


class TestTeamProbabilities(unittest.TestCase):
    def test_every_margin_model_round_trips_pregame_anchor(self):
        for sport, league in (
            ("basketball", "nba"),
            ("football", "nfl"),
            ("baseball", "mlb"),
            ("hockey", "nhl"),
        ):
            with self.subTest(sport=sport):
                state = team.pregame_state()
                fair = team.fair_probability(
                    sport, 0.63, state, state, "home"
                )
                self.assertAlmostEqual(fair, 0.63, places=6)

    def test_lead_matters_more_late(self):
        anchor = team.pregame_state()
        early = team.parse_team_score(
            "basketball", "nba", "6-0", "Q1", "06:00"
        )
        late = team.parse_team_score(
            "basketball", "nba", "96-90", "Q4", "06:00"
        )
        early_p = team.fair_probability(
            "basketball", 0.50, anchor, early, "home"
        )
        late_p = team.fair_probability(
            "basketball", 0.50, anchor, late, "home"
        )
        self.assertGreater(late_p, early_p)

    def test_away_role_inverts_home_probability(self):
        anchor = team.pregame_state()
        state = team.parse_team_score(
            "football", "nfl", "21-7", "Q3", "08:00"
        )
        home = team.fair_probability(
            "football", 0.50, anchor, state, "home"
        )
        away = team.fair_probability(
            "football", 0.50, anchor, state, "away"
        )
        self.assertAlmostEqual(home + away, 1.0, places=6)

    def test_soccer_goal_raises_home_and_lowers_draw(self):
        anchor = team.pregame_state()
        tied = team.parse_team_score(
            "soccer", "epl", "0-0", "2H", "70:00"
        )
        leading = team.parse_team_score(
            "soccer", "epl", "1-0", "2H", "70:00"
        )
        home_tied = team.fair_probability(
            "soccer", 0.40, anchor, tied, "home"
        )
        home_leading = team.fair_probability(
            "soccer", 0.40, anchor, leading, "home"
        )
        draw_tied = team.fair_probability(
            "soccer", 0.28, anchor, tied, "draw"
        )
        draw_leading = team.fair_probability(
            "soccer", 0.28, anchor, leading, "draw"
        )
        self.assertGreater(home_leading, home_tied)
        self.assertLess(draw_leading, draw_tied)


class TestTeamStrategy(unittest.TestCase):
    def test_live_anchor_round_trips_current_state(self):
        item = market("basketball", "nba", "away")
        strategy = LiveModelStrategy(StrategyConfig())

        self.assertTrue(strategy.set_live_anchor(
            item,
            0.37,
            "60-55",
            "Q3",
            ts=1_000,
            elapsed="04:00",
        ))
        fair = strategy.on_score(
            item,
            "60-55",
            "Q3",
            live=True,
            ended=False,
            ts=1_000,
            elapsed="04:00",
        )

        self.assertAlmostEqual(fair, 0.37, places=6)

    def test_score_run_has_bounded_momentum_nudge(self):
        item = market("basketball", "nba", "home")
        strategy = LiveModelStrategy(StrategyConfig(momentum_decay=0.65))
        strategy.set_anchor(item, 0.50, ts=1_000)
        strategy.on_score(
            item, "40-40", "Q2", True, False, ts=2_000, elapsed="02:00"
        )
        fair = strategy.on_score(
            item, "48-40", "Q2", True, False, ts=3_000, elapsed="01:00"
        )

        tracker = strategy.trackers[item.market_id]
        self.assertGreater(fair, 0.50)
        self.assertLessEqual(tracker.momentum_home, 12.0)

    def test_soccer_event_probabilities_are_normalized(self):
        strategy = LiveModelStrategy(StrategyConfig())
        markets = [
            market("soccer", "epl", role)
            for role in ("home", "draw", "away")
        ]
        for item, anchor in zip(markets, (0.40, 0.28, 0.32)):
            strategy.set_anchor(item, anchor, ts=1_000)
            strategy.on_score(
                item,
                "1-0",
                "2H",
                True,
                False,
                ts=2_000,
                elapsed="70:00",
            )

        strategy.normalize_event(markets)

        total = sum(
            strategy.trackers[item.market_id].fair_value
            for item in markets
        )
        self.assertAlmostEqual(total, 1.0, places=9)

    def test_missing_team_clock_pauses_trading(self):
        item = market("basketball", "nba", "home")
        strategy = LiveModelStrategy(StrategyConfig())
        strategy.set_anchor(item, 0.50, ts=1_000)

        strategy.on_score(
            item, "20-18", "Q1", True, False, ts=2_000, elapsed=""
        )

        tracker = strategy.trackers[item.market_id]
        self.assertFalse(tracker.score_tradeable)
        self.assertEqual(tracker.score_issue, "clock unavailable")


if __name__ == "__main__":
    unittest.main()
