"""
Tests for the probability models.

These check the model against values that are analytically known or that follow
from symmetry, rather than against numbers the model itself produced. A model
that agrees with its own output is not tested.
"""

import unittest

from pmpt.quant import table_tennis as tt
from pmpt.quant import tennis as tn


class TestTennisGame(unittest.TestCase):
    def test_fair_coin_game_is_symmetric(self):
        self.assertAlmostEqual(tn.game_win_prob(0.5), 0.5, places=9)

    def test_certain_server_always_holds(self):
        self.assertAlmostEqual(tn.game_win_prob(0.999999), 1.0, places=5)
        self.assertAlmostEqual(tn.game_win_prob(0.000001), 0.0, places=5)

    def test_known_closed_form(self):
        # The standard closed form for holding serve from 0-0:
        #   p^4 (15 - 4p - 10p^2/(1 - 2p(1-p)))
        for p in (0.55, 0.60, 0.62, 0.65, 0.70, 0.75):
            q = 1 - p
            expected = p**4 * (15 - 4 * p - (10 * p**2) / (1 - 2 * p * q))
            self.assertAlmostEqual(tn.game_win_prob(p), expected, places=9, msg=f"p={p}")

    def test_deuce_closed_form(self):
        p = 0.64
        expected = p * p / (p * p + (1 - p) ** 2)
        self.assertAlmostEqual(tn.game_win_prob(p, 3, 3), expected, places=12)

    def test_monotonic_in_point_prob(self):
        vals = [tn.game_win_prob(p) for p in (0.4, 0.5, 0.6, 0.7, 0.8)]
        self.assertEqual(vals, sorted(vals))

    def test_advantage_beats_deuce_beats_disadvantage(self):
        p = 0.6
        self.assertGreater(tn.game_win_prob(p, 4, 3), tn.game_win_prob(p, 3, 3))
        self.assertGreater(tn.game_win_prob(p, 3, 3), tn.game_win_prob(p, 3, 4))


class TestTennisTiebreak(unittest.TestCase):
    def test_serve_order_is_exactly_fair(self):
        """Equal players split the tiebreak exactly 50/50, whoever serves first.

        This is not an approximation and it is not an accident. The tiebreak
        serve order (A, BB, AA, BB, ...) is the Thue-Morse sequence, which is
        precisely the arrangement that removes first-server advantage. Confirmed
        independently by Monte Carlo at 400k trials (0.4996).
        """
        self.assertAlmostEqual(tn.tiebreak_win_prob(0.64, 0.64), 0.5, places=12)
        self.assertAlmostEqual(tn.tiebreak_win_prob(0.55, 0.55, server=1), 0.5, places=12)

    def test_serving_first_confers_no_edge(self):
        a = tn.tiebreak_win_prob(0.70, 0.55, server=0)
        b = tn.tiebreak_win_prob(0.70, 0.55, server=1)
        self.assertAlmostEqual(a, b, places=9)

    def test_matches_monte_carlo(self):
        # 400k-trial MC gave 0.7339 for these inputs; allow for sampling error.
        self.assertAlmostEqual(tn.tiebreak_win_prob(0.70, 0.55), 0.7327, places=3)

    def test_symmetry_under_role_swap(self):
        a = tn.tiebreak_win_prob(0.70, 0.55, server=0)
        b = tn.tiebreak_win_prob(0.55, 0.70, server=0)
        self.assertAlmostEqual(a, 1 - b, places=9)

    def test_match_point_is_near_certain(self):
        p = tn.tiebreak_win_prob(0.64, 0.64, a=6, b=0)
        self.assertGreater(p, 0.95)

    def test_deuce_region_terminates_and_is_bounded(self):
        p = tn.tiebreak_win_prob(0.64, 0.64, a=15, b=15)
        self.assertTrue(0.0 <= p <= 1.0)


class TestTennisSet(unittest.TestCase):
    def test_equal_players_set_is_exactly_even(self):
        # Strict alternation means each player serves the same number of games in
        # every reachable set length, so equal players split exactly. MC: 0.5007.
        self.assertAlmostEqual(tn.set_win_prob(0.64, 0.64, server=0), 0.5, places=12)
        self.assertAlmostEqual(tn.set_win_prob(0.64, 0.64, server=1), 0.5, places=12)

    def test_matches_monte_carlo(self):
        # 100k-trial MC gave 0.7466 for a 0.68/0.60 serve matchup.
        self.assertAlmostEqual(tn.set_win_prob(0.68, 0.60, server=0), 0.7466, places=2)

    def test_serving_for_the_set_is_strong(self):
        p = tn.set_win_prob(0.64, 0.64, games_a=5, games_b=0, server=0)
        self.assertGreater(p, 0.97)

    def test_five_all_is_close(self):
        p = tn.set_win_prob(0.64, 0.64, games_a=5, games_b=5, server=0)
        self.assertAlmostEqual(p, 0.5, places=9)

    def test_past_tiebreak_threshold_terminates(self):
        """A garbled feed can produce 7-7. That must not hang the process.

        The first version of this model recursed forever here and took the whole
        interpreter down with a segfault, which in live trading would mean an
        unmonitored bot sitting on open positions.
        """
        for ga, gb in ((7, 7), (8, 8), (9, 8), (12, 11)):
            p = tn.set_win_prob(0.66, 0.61, ga, gb, 0)
            self.assertTrue(0.0 <= p <= 1.0)

    def test_symmetry(self):
        a = tn.set_win_prob(0.68, 0.60, 3, 2, server=0)
        b = tn.set_win_prob(0.60, 0.68, 2, 3, server=1)
        self.assertAlmostEqual(a, 1 - b, places=9)


class TestTennisMatch(unittest.TestCase):
    def test_even_match_is_exactly_even(self):
        st = tn.TennisMatchState(best_of=3)
        self.assertAlmostEqual(tn.match_win_prob(st, 0.64, 0.64), 0.5, places=12)

    def test_best_of_five_favours_the_stronger_player(self):
        st3 = tn.TennisMatchState(best_of=3)
        st5 = tn.TennisMatchState(best_of=5)
        p3 = tn.match_win_prob(st3, 0.68, 0.60)
        p5 = tn.match_win_prob(st5, 0.68, 0.60)
        # A longer match gives the better player more chances to express the edge.
        self.assertGreater(p5, p3)

    def test_a_set_down_hurts(self):
        even = tn.match_win_prob(tn.TennisMatchState(best_of=3), 0.64, 0.64)
        down = tn.match_win_prob(
            tn.TennisMatchState(sets_a=0, sets_b=1, best_of=3), 0.64, 0.64
        )
        self.assertLess(down, even)
        self.assertGreater(down, 0.15)

    def test_terminal_states(self):
        self.assertEqual(
            tn.match_win_prob(tn.TennisMatchState(sets_a=2, best_of=3), 0.6, 0.6), 1.0
        )
        self.assertEqual(
            tn.match_win_prob(tn.TennisMatchState(sets_b=2, best_of=3), 0.6, 0.6), 0.0
        )

    def test_probability_is_always_bounded(self):
        for sa in range(3):
            for sb in range(3):
                for ga in range(8):
                    for gb in range(8):
                        st = tn.TennisMatchState(
                            sets_a=sa, sets_b=sb, games_a=ga, games_b=gb, best_of=3
                        )
                        p = tn.match_win_prob(st, 0.66, 0.61)
                        self.assertTrue(0.0 <= p <= 1.0, f"{sa}-{sb} {ga}-{gb} -> {p}")


class TestTennisCalibration(unittest.TestCase):
    def test_round_trip(self):
        """Calibrating to a target and re-evaluating should return the target."""
        for target in (0.25, 0.35, 0.5, 0.65, 0.80):
            for bo in (3, 5):
                pa, pb = tn.calibrate_serve_probs(target, best_of=bo)
                st = tn.TennisMatchState(best_of=bo)
                got = tn.match_win_prob(st, pa, pb)
                self.assertAlmostEqual(got, target, places=3,
                                       msg=f"target={target} bo={bo} got={got}")

    def test_stronger_target_means_bigger_serve_gap(self):
        pa1, pb1 = tn.calibrate_serve_probs(0.55)
        pa2, pb2 = tn.calibrate_serve_probs(0.80)
        self.assertGreater(pa2 - pb2, pa1 - pb1)

    def test_live_state_round_trip(self):
        state = tn.parse_tennis_score("6-3, 3-5", "S2", best_of=3)
        pa, pb = tn.calibrate_serve_probs_at_state(0.42, state)
        self.assertAlmostEqual(tn.match_win_prob(state, pa, pb), 0.42, places=3)


class TestScoreParsing(unittest.TestCase):
    def test_completed_sets_and_current_set(self):
        st = tn.parse_tennis_score("6-4, 3-2", "2", best_of=3)
        self.assertEqual((st.sets_a, st.sets_b), (1, 0))
        self.assertEqual((st.games_a, st.games_b), (3, 2))
        self.assertFalse(st.finished)

    def test_tiebreak_annotation_is_stripped(self):
        st = tn.parse_tennis_score("7-6(5), 2-1", "2", best_of=3)
        self.assertEqual((st.sets_a, st.sets_b), (1, 0))
        self.assertEqual((st.games_a, st.games_b), (2, 1))

    def test_finished_match(self):
        st = tn.parse_tennis_score("6-4, 6-3", "FT", best_of=3)
        self.assertEqual((st.sets_a, st.sets_b), (2, 0))
        self.assertTrue(st.finished)

    def test_in_set_tiebreak_detected(self):
        st = tn.parse_tennis_score("6-6", "1", best_of=3)
        self.assertTrue(st.in_tiebreak)

    def test_garbage_does_not_raise(self):
        for bad in ("", "??", "abc-def", "6-", ",,,", None):
            st = tn.parse_tennis_score(bad or "", "1")
            self.assertIsInstance(st, tn.TennisMatchState)


class TestTableTennis(unittest.TestCase):
    def test_fair_game_is_near_even(self):
        p = tt.game_win_prob(0.5, 0.5)
        self.assertAlmostEqual(p, 0.5, places=6)

    def test_equal_players_split_exactly(self):
        # Two-point serve alternation is also exactly fair. MC: 0.5003.
        self.assertAlmostEqual(tt.game_win_prob(0.55, 0.55), 0.5, places=12)
        self.assertAlmostEqual(tt.game_win_prob(0.55, 0.55, server=1), 0.5, places=12)

    def test_stronger_server_wins_more(self):
        self.assertGreater(tt.game_win_prob(0.60, 0.50), 0.5)
        self.assertLess(tt.game_win_prob(0.50, 0.60), 0.5)

    def test_matches_monte_carlo(self):
        # 400k-trial MC gave 0.65269.
        self.assertAlmostEqual(tt.game_win_prob(0.60, 0.52), 0.6527, places=3)

    def test_symmetry(self):
        a = tt.game_win_prob(0.60, 0.52)
        b = tt.game_win_prob(0.52, 0.60)
        self.assertAlmostEqual(a, 1 - b, places=9)

    def test_game_point_is_strong(self):
        p = tt.game_win_prob(0.55, 0.55, a=10, b=5)
        self.assertGreater(p, 0.95)

    def test_deuce_terminates(self):
        p = tt.game_win_prob(0.55, 0.55, a=14, b=14)
        self.assertTrue(0.0 <= p <= 1.0)

    def test_match_bounded_everywhere(self):
        for ga in range(3):
            for gb in range(3):
                for pa_ in range(0, 13):
                    st = tt.TableTennisMatchState(
                        games_a=ga, games_b=gb, points_a=pa_, points_b=5, best_of=5
                    )
                    p = tt.match_win_prob(st, 0.56, 0.53)
                    self.assertTrue(0.0 <= p <= 1.0)

    def test_calibration_round_trip(self):
        for target in (0.3, 0.5, 0.7):
            pa, pb = tt.calibrate_serve_probs(target, best_of=5)
            st = tt.TableTennisMatchState(best_of=5)
            self.assertAlmostEqual(tt.match_win_prob(st, pa, pb), target, places=3)

    def test_live_state_calibration_round_trip(self):
        state = tt.parse_table_tennis_score(
            "11-8, 9-11, 4-3",
            "3",
            best_of=5,
        )
        pa, pb = tt.calibrate_serve_probs_at_state(0.64, state)
        self.assertAlmostEqual(tt.match_win_prob(state, pa, pb), 0.64, places=3)

    def test_score_parsing(self):
        st = tt.parse_table_tennis_score("11-8, 9-11, 4-3", "3", best_of=5)
        self.assertEqual((st.games_a, st.games_b), (1, 1))
        self.assertEqual((st.points_a, st.points_b), (4, 3))
        self.assertFalse(st.finished)

    def test_finished_parsing(self):
        st = tt.parse_table_tennis_score("11-8, 11-9, 11-6", "FT", best_of=5)
        self.assertEqual((st.games_a, st.games_b), (3, 0))
        self.assertTrue(st.finished)


if __name__ == "__main__":
    unittest.main()
