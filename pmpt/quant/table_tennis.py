"""
Point-level Markov model for table tennis.

Same idea as the tennis model: two serve point-win probabilities drive everything.
The structural differences that matter:

  * A game is first to 11, win by 2 (not 4/deuce).
  * Serve alternates every 2 points, then every 1 point once both reach 10.
  * The serve advantage is much smaller than in tennis, so `pa` and `pb` sit
    close to 0.5 rather than 0.64.
  * Matches are short (best of 5 or 7), which means each point moves the win
    probability a lot -- good for lag trading, punishing if your score feed is stale.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

_Q = 6


def _q(x: float) -> float:
    return round(x, _Q)


def _clip(x: float, lo: float = 1e-6, hi: float = 1 - 1e-6) -> float:
    return max(lo, min(hi, x))


# --------------------------------------------------------------------------
# Game
# --------------------------------------------------------------------------

def _server_is_first(n: int, target: int) -> bool:
    """True if the player who served point 0 also serves point index `n`.

    Serve changes every 2 points until both players reach `target - 1`
    (10 in a standard game to 11), after which it changes every point.
    """
    deuce_start = 2 * (target - 1)  # 20 points played => 10-10
    if n < deuce_start:
        return (n // 2) % 2 == 0
    return n % 2 == 0


@lru_cache(maxsize=None)
def _game(pa: float, pb: float, a: int, b: int, target: int) -> float:
    """P(A wins the game) where A serves the FIRST point of the game."""
    pa, pb = _clip(pa), _clip(pb)

    if a >= target and a - b >= 2:
        return 1.0
    if b >= target and b - a >= 2:
        return 0.0

    lim = target - 1
    if a >= lim and b >= lim and a == b:
        # At 10-10 and beyond serve alternates every point, so the state is
        # periodic. Solve the 2-point cycle in closed form.
        W = pa * (1.0 - pb)
        S = pa * pb + (1.0 - pa) * (1.0 - pb)
        if S >= 1.0 - 1e-12:
            return 0.5
        # Which of the two players serves next does not change the answer here:
        # both orderings give the same fixed point.
        return W / (1.0 - S)

    n = a + b
    p_point_to_a = pa if _server_is_first(n, target) else (1.0 - pb)
    return (
        p_point_to_a * _game(pa, pb, a + 1, b, target)
        + (1.0 - p_point_to_a) * _game(pa, pb, a, b + 1, target)
    )


def game_win_prob(
    pa: float, pb: float, a: int = 0, b: int = 0, target: int = 11, server: int = 0
) -> float:
    """P(A wins the game). `server` is who served the FIRST point (0=A, 1=B)."""
    if server == 0:
        return _game(_q(pa), _q(pb), a, b, target)
    return 1.0 - _game(_q(pb), _q(pa), b, a, target)


# --------------------------------------------------------------------------
# Match
# --------------------------------------------------------------------------

@dataclass
class TableTennisMatchState:
    """Live state of a table tennis match, from A's point of view."""

    games_a: int = 0
    games_b: int = 0
    points_a: int = 0
    points_b: int = 0
    server: int = 0        # who served the first point of the CURRENT game
    best_of: int = 5
    points_target: int = 11
    finished: bool = False

    @property
    def games_to_win(self) -> int:
        return self.best_of // 2 + 1


@lru_cache(maxsize=None)
def _match_from_games(
    pa: float, pb: float, ga: int, gb: int, need: int, target: int
) -> float:
    """P(A wins the match) starting a fresh game at game score (ga, gb)."""
    if ga >= need:
        return 1.0
    if gb >= need:
        return 0.0
    # First serve of each game alternates, and we do not track it across games,
    # so average the two possibilities.
    pg = 0.5 * (_game(pa, pb, 0, 0, target) + (1.0 - _game(pb, pa, 0, 0, target)))
    return (
        pg * _match_from_games(pa, pb, ga + 1, gb, need, target)
        + (1.0 - pg) * _match_from_games(pa, pb, ga, gb + 1, need, target)
    )


def match_win_prob(state: TableTennisMatchState, pa: float, pb: float) -> float:
    """P(player A wins the match) given live state and serve point probabilities."""
    pa, pb = _q(_clip(pa)), _q(_clip(pb))
    need = state.games_to_win

    if state.games_a >= need:
        return 1.0
    if state.games_b >= need:
        return 0.0

    p_game = game_win_prob(
        pa, pb, state.points_a, state.points_b, state.points_target, state.server
    )
    win_g = _match_from_games(pa, pb, state.games_a + 1, state.games_b, need, state.points_target)
    lose_g = _match_from_games(pa, pb, state.games_a, state.games_b + 1, need, state.points_target)
    return p_game * win_g + (1.0 - p_game) * lose_g


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

# Table tennis serve advantage is real but modest. This is the anchor that the
# player-strength spread is applied around.
DEFAULT_AVG_SERVE = 0.545


@lru_cache(maxsize=8192)
def _calibrate_cached(
    target: float, best_of: int, points_target: int, avg_serve: float,
    tol: float, max_iter: int
) -> tuple[float, float]:
    st = TableTennisMatchState(best_of=best_of, points_target=points_target)

    lo, hi = -0.25, 0.25
    if target <= match_win_prob(st, _clip(avg_serve + lo), _clip(avg_serve - lo)):
        return _clip(avg_serve + lo), _clip(avg_serve - lo)
    if target >= match_win_prob(st, _clip(avg_serve + hi), _clip(avg_serve - hi)):
        return _clip(avg_serve + hi), _clip(avg_serve - hi)

    mid = 0.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        val = match_win_prob(st, _clip(avg_serve + mid), _clip(avg_serve - mid))
        if abs(val - target) < tol:
            break
        if val < target:
            lo = mid
        else:
            hi = mid
    return _clip(avg_serve + mid), _clip(avg_serve - mid)


def calibrate_serve_probs(
    target_match_prob: float,
    best_of: int = 5,
    points_target: int = 11,
    avg_serve: float = DEFAULT_AVG_SERVE,
    tol: float = 1e-5,
    max_iter: int = 40,
) -> tuple[float, float]:
    """Back out (pa, pb) consistent with a pre-match win probability for A.

    Identical logic to the tennis calibration: anchor on the market's own
    pre-match opinion, then let the score do the work. Cached on a 4dp target
    for the same performance reason.
    """
    target = round(_clip(target_match_prob, 1e-4, 1 - 1e-4), 4)
    return _calibrate_cached(
        target, best_of, points_target, round(avg_serve, 6), tol, max_iter
    )


# --------------------------------------------------------------------------
# Score parsing
# --------------------------------------------------------------------------

def parse_table_tennis_score(
    score: str, period: str = "", best_of: int = 5, points_target: int = 11
) -> TableTennisMatchState:
    """Parse a Polymarket sports-feed table tennis score.

    Feed format for table tennis is typically game-by-game, e.g. "11-8, 9-11, 4-3",
    with `period` giving the current game number or "FT".
    """
    st = TableTennisMatchState(best_of=best_of, points_target=points_target)
    if not score:
        return st

    games = [g.strip() for g in str(score).split(",") if g.strip()]
    for i, raw in enumerate(games):
        s = raw.split("(")[0].strip()
        if "-" not in s:
            continue
        try:
            a_s, b_s = s.split("-")[:2]
            a, b = int(a_s.strip()), int(b_s.strip())
        except ValueError:
            continue

        decided = (a >= points_target or b >= points_target) and abs(a - b) >= 2
        is_last = i == len(games) - 1

        if decided and not (is_last and not decided):
            if a > b:
                st.games_a += 1
            else:
                st.games_b += 1
            if is_last:
                st.points_a = st.points_b = 0
        elif is_last:
            st.points_a, st.points_b = a, b

    need = st.games_to_win
    if str(period).upper().startswith("FT") or st.games_a >= need or st.games_b >= need:
        st.finished = True

    # Server of the current game alternates by game number.
    st.server = (st.games_a + st.games_b) % 2
    return st
