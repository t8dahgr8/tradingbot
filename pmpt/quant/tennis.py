"""
Point-level Markov model for tennis.

The whole model rests on two numbers: `pa` and `pb`, the probability that player A
(resp. B) wins a single point *on their own serve*. Everything else -- games, sets,
tiebreaks, the match -- is derived exactly from those two numbers by recursion.

This matters for trading because it means a live score is not a vague signal. Given
pa/pb and the current score, there is one correct match-win probability, and any
market price that disagrees with it is a quantifiable edge.

Pure stdlib. No numpy, no scipy.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

# Rounding applied to serve probabilities before they hit the memo caches.
# Without this, floating point noise makes the caches useless.
_Q = 6


def _q(x: float) -> float:
    return round(x, _Q)


def _clip(x: float, lo: float = 1e-6, hi: float = 1 - 1e-6) -> float:
    return max(lo, min(hi, x))


# --------------------------------------------------------------------------
# Game
# --------------------------------------------------------------------------

@lru_cache(maxsize=None)
def game_win_prob(p: float, a: int = 0, b: int = 0) -> float:
    """P(server wins the game) from point score (a, b) with server point-win prob `p`.

    Scores use raw point counts (0,1,2,3,4...), not 15/30/40. Deuce and advantage
    are collapsed into a closed form rather than recursed, which keeps this exact
    and terminating.
    """
    p = _clip(p)
    if a >= 4 and a - b >= 2:
        return 1.0
    if b >= 4 and b - a >= 2:
        return 0.0

    if a >= 3 and b >= 3:
        q = 1.0 - p
        deuce = (p * p) / (p * p + q * q)
        if a == b:
            return deuce
        if a > b:  # advantage server
            return p + q * deuce
        return p * deuce  # advantage returner

    return p * game_win_prob(p, a + 1, b) + (1.0 - p) * game_win_prob(p, a, b + 1)


# --------------------------------------------------------------------------
# Tiebreak
# --------------------------------------------------------------------------

def _tb_server_is_first(n: int) -> bool:
    """Which player serves point index `n` (0-based) of a tiebreak.

    Serve order is A, BB, AA, BB, AA ... Returns True when the player who served
    the first point of the tiebreak serves point n.
    """
    return ((n + 1) // 2) % 2 == 0


@lru_cache(maxsize=None)
def _tiebreak(pa: float, pb: float, a: int, b: int, target: int) -> float:
    """P(A wins the tiebreak) where A serves the FIRST point. Win by 2."""
    pa, pb = _clip(pa), _clip(pb)

    if a >= target and a - b >= 2:
        return 1.0
    if b >= target and b - a >= 2:
        return 0.0

    # Beyond (target-1, target-1) the serve pattern is periodic and the state
    # repeats, so recursion would not terminate. Solve the tied case in closed form.
    lim = target - 1
    if a >= lim and b >= lim and a == b:
        # W = A wins both points of a 2-point cycle; S = cycle is split (back to tied)
        W = pa * (1.0 - pb)
        S = pa * pb + (1.0 - pa) * (1.0 - pb)
        if S >= 1.0 - 1e-12:
            return 0.5
        return W / (1.0 - S)

    n = a + b
    p_point_to_a = pa if _tb_server_is_first(n) else (1.0 - pb)
    return (
        p_point_to_a * _tiebreak(pa, pb, a + 1, b, target)
        + (1.0 - p_point_to_a) * _tiebreak(pa, pb, a, b + 1, target)
    )


def tiebreak_win_prob(
    pa: float, pb: float, a: int = 0, b: int = 0, target: int = 7, server: int = 0
) -> float:
    """P(A wins the tiebreak). `server` is who serves the next point (0=A, 1=B).

    The recursion assumes A served the first point, so when B opened the tiebreak
    we solve the mirrored problem and flip the result.
    """
    if server == 0:
        return _tiebreak(_q(pa), _q(pb), a, b, target)
    return 1.0 - _tiebreak(_q(pb), _q(pa), b, a, target)


# --------------------------------------------------------------------------
# Set
# --------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _set(
    pa: float,
    pb: float,
    ga: int,
    gb: int,
    server: int,
    tb_target: int,
    tb_at: int,
) -> float:
    """P(A wins the set). `server` (0=A, 1=B) serves the NEXT game."""
    if ga >= tb_at and ga - gb >= 2:
        return 1.0
    if gb >= tb_at and gb - ga >= 2:
        return 0.0

    # Both players at or past the tiebreak threshold with a lead of less than 2.
    # Testing `>=` rather than `==` matters: a state like 7-7 is unreachable in a
    # real set but IS reachable from a garbled score feed, and the `==` version
    # recursed forever instead of terminating.
    if ga >= tb_at and gb >= tb_at:
        if tb_target <= 0:
            # Advantage set: no tiebreak, keep playing game pairs. Closed form on
            # the 2-game cycle, same structure as the tiebreak deuce.
            hold_a = game_win_prob(pa)
            hold_b = game_win_prob(pb)
            if server == 0:
                p1, p2 = hold_a, 1.0 - hold_b
            else:
                p1, p2 = 1.0 - hold_b, hold_a
            W = p1 * p2
            S = p1 * (1.0 - p2) + (1.0 - p1) * p2
            return 0.5 if S >= 1.0 - 1e-12 else W / (1.0 - S)
        return tiebreak_win_prob(pa, pb, 0, 0, tb_target, server)

    hold = game_win_prob(pa) if server == 0 else game_win_prob(pb)
    p_game_to_a = hold if server == 0 else 1.0 - hold
    nxt = 1 - server
    return (
        p_game_to_a * _set(pa, pb, ga + 1, gb, nxt, tb_target, tb_at)
        + (1.0 - p_game_to_a) * _set(pa, pb, ga, gb + 1, nxt, tb_target, tb_at)
    )


def set_win_prob(
    pa: float,
    pb: float,
    games_a: int = 0,
    games_b: int = 0,
    server: int = 0,
    tb_target: int = 7,
    tb_at: int = 6,
) -> float:
    """P(A wins the set) from the given game score, ignoring points in the current game."""
    return _set(_q(pa), _q(pb), games_a, games_b, server, tb_target, tb_at)


# --------------------------------------------------------------------------
# Match state
# --------------------------------------------------------------------------

Surface = Literal["hard", "clay", "grass", "unknown"]


@dataclass
class TennisMatchState:
    """Live state of a tennis match, from A's point of view.

    `server` is who serves the current game. `points_a`/`points_b` are raw point
    counts in the current game (0-3+, not 15/30/40). If you only have game scores
    (which is all the Polymarket sports feed gives you), leave points at 0.
    """

    sets_a: int = 0
    sets_b: int = 0
    games_a: int = 0
    games_b: int = 0
    points_a: int = 0
    points_b: int = 0
    server: int = 0
    best_of: int = 3
    tb_target: int = 7            # 0 => advantage set (no tiebreak)
    tb_at: int = 6
    final_set_tb_target: int = 7  # some events use a 10-point final-set breaker
    in_tiebreak: bool = False
    finished: bool = False

    @property
    def sets_to_win(self) -> int:
        return self.best_of // 2 + 1

    def is_final_set(self) -> bool:
        n = self.sets_to_win - 1
        return self.sets_a == n and self.sets_b == n


@lru_cache(maxsize=None)
def _match_from_sets(
    pa: float, pb: float, sa: int, sb: int, need: int, tb_target: int, tb_at: int,
    final_tb: int,
) -> float:
    """P(A wins the match) starting a fresh set at set score (sa, sb)."""
    if sa >= need:
        return 1.0
    if sb >= need:
        return 0.0
    is_final = (sa == need - 1) and (sb == need - 1)
    tbt = final_tb if is_final else tb_target
    # Who serves first in a not-yet-started set is unknown, so average the two.
    ps = 0.5 * (
        _set(pa, pb, 0, 0, 0, tbt, tb_at) + _set(pa, pb, 0, 0, 1, tbt, tb_at)
    )
    return (
        ps * _match_from_sets(pa, pb, sa + 1, sb, need, tb_target, tb_at, final_tb)
        + (1.0 - ps) * _match_from_sets(pa, pb, sa, sb + 1, need, tb_target, tb_at, final_tb)
    )


def match_win_prob(state: TennisMatchState, pa: float, pb: float) -> float:
    """P(player A wins the match) given live state and serve point probabilities.

    This is the number the whole strategy hangs off. Compare it to the market's
    implied probability and the difference is your edge, before costs.
    """
    pa, pb = _q(_clip(pa)), _q(_clip(pb))
    need = state.sets_to_win

    if state.sets_a >= need:
        return 1.0
    if state.sets_b >= need:
        return 0.0

    tbt = state.final_set_tb_target if state.is_final_set() else state.tb_target

    # Probability A wins the set currently in progress.
    if state.in_tiebreak:
        p_set = tiebreak_win_prob(
            pa, pb, state.points_a, state.points_b, tbt or 7, state.server
        )
    elif state.points_a or state.points_b:
        # Mid-game: resolve the current game first, then hand off to the set model.
        srv_p = pa if state.server == 0 else pb
        sa = state.points_a if state.server == 0 else state.points_b
        sb = state.points_b if state.server == 0 else state.points_a
        g = game_win_prob(_q(srv_p), sa, sb)
        p_hold = g if state.server == 0 else 1.0 - g
        nxt = 1 - state.server
        p_set = (
            p_hold * _set(pa, pb, state.games_a + 1, state.games_b, nxt, tbt, state.tb_at)
            + (1.0 - p_hold) * _set(pa, pb, state.games_a, state.games_b + 1, nxt, tbt, state.tb_at)
        )
    else:
        p_set = _set(pa, pb, state.games_a, state.games_b, state.server, tbt, state.tb_at)

    win_set = _match_from_sets(
        pa, pb, state.sets_a + 1, state.sets_b, need, state.tb_target, state.tb_at,
        state.final_set_tb_target,
    )
    lose_set = _match_from_sets(
        pa, pb, state.sets_a, state.sets_b + 1, need, state.tb_target, state.tb_at,
        state.final_set_tb_target,
    )
    return p_set * win_set + (1.0 - p_set) * lose_set


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

# Typical server point-win rates. These anchor calibration; the spread between
# the two players is what we actually solve for.
DEFAULT_AVG_SERVE = {
    "hard": 0.640,
    "clay": 0.620,
    "grass": 0.660,
    "unknown": 0.635,
}


@lru_cache(maxsize=8192)
def _calibrate_cached(
    target: float, best_of: int, base: float, tol: float, max_iter: int
) -> tuple[float, float]:
    st = TennisMatchState(best_of=best_of)

    lo, hi = -0.30, 0.30
    f_lo = match_win_prob(st, _clip(base + lo), _clip(base - lo))
    f_hi = match_win_prob(st, _clip(base + hi), _clip(base - hi))
    if target <= f_lo:
        return _clip(base + lo), _clip(base - lo)
    if target >= f_hi:
        return _clip(base + hi), _clip(base - hi)

    mid = 0.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        val = match_win_prob(st, _clip(base + mid), _clip(base - mid))
        if abs(val - target) < tol:
            break
        if val < target:
            lo = mid
        else:
            hi = mid
    return _clip(base + mid), _clip(base - mid)


def calibrate_serve_probs(
    target_match_prob: float,
    best_of: int = 3,
    surface: Surface = "unknown",
    avg_serve: float | None = None,
    tol: float = 1e-5,
    max_iter: int = 40,
) -> tuple[float, float]:
    """Back out (pa, pb) consistent with a pre-match win probability for A.

    This is the trick that makes the model tradeable without a player database.
    You take the market's *own* pre-match price as the truth about relative
    strength, convert it into serve probabilities, and then the model tells you
    what the price *should* be once the score moves. You are not betting that you
    know the players better than the market -- you are betting that the market is
    slow to reprice a score it already agreed on.

    Solves for delta in (avg + delta, avg - delta) by bisection.

    The target is rounded to 4 decimal places before the (cached) solve. That is
    far finer than any edge we would act on, and without it this function
    dominates the runtime -- it was 93% of a profiling run, because the strategy
    re-derives serve probabilities on every book update.
    """
    target = round(_clip(target_match_prob, 1e-4, 1 - 1e-4), 4)
    base = avg_serve if avg_serve is not None else DEFAULT_AVG_SERVE.get(surface, 0.635)
    return _calibrate_cached(target, best_of, round(base, 6), tol, max_iter)


def calibrate_serve_probs_at_state(
    target_match_prob: float,
    state: TennisMatchState,
    surface: Surface = "unknown",
    avg_serve: float | None = None,
    tol: float = 1e-5,
    max_iter: int = 40,
) -> tuple[float, float]:
    """Infer player strength from a live price without double-counting the score."""
    target = _clip(target_match_prob, 1e-4, 1 - 1e-4)
    base = avg_serve if avg_serve is not None else DEFAULT_AVG_SERVE.get(surface, 0.635)
    lo, hi = -0.30, 0.30
    f_lo = match_win_prob(state, _clip(base + lo), _clip(base - lo))
    f_hi = match_win_prob(state, _clip(base + hi), _clip(base - hi))
    if target <= f_lo:
        return _clip(base + lo), _clip(base - lo)
    if target >= f_hi:
        return _clip(base + hi), _clip(base - hi)

    mid = 0.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        value = match_win_prob(state, _clip(base + mid), _clip(base - mid))
        if abs(value - target) < tol:
            break
        if value < target:
            lo = mid
        else:
            hi = mid
    return _clip(base + mid), _clip(base - mid)


# --------------------------------------------------------------------------
# Score parsing (Polymarket sports feed format)
# --------------------------------------------------------------------------

def parse_tennis_score(score: str, period: str = "", best_of: int = 3) -> TennisMatchState:
    """Parse a Polymarket sports-feed tennis score into a match state.

    The feed gives set-by-set strings like "6-4, 3-2" (home-away per set) and a
    `period` such as "1", "2" or "FT". Point-level detail is not provided, so the
    resulting state has points at zero: the model degrades gracefully to
    game-level resolution.
    """
    st = TennisMatchState(best_of=best_of)
    if not score:
        return st

    sets = [s.strip() for s in str(score).split(",") if s.strip()]
    need = st.sets_to_win

    for i, raw in enumerate(sets):
        s = raw.split("(")[0].strip()  # strip tiebreak annotations like 7-6(5)
        if "-" not in s:
            continue
        try:
            a_s, b_s = s.split("-")[:2]
            a, b = int(a_s.strip()), int(b_s.strip())
        except ValueError:
            continue

        is_last = i == len(sets) - 1
        decided = ((a >= 6 or b >= 6) and abs(a - b) >= 2) or a == 7 or b == 7

        if is_last and not decided:
            st.games_a, st.games_b = a, b
            st.in_tiebreak = a == 6 and b == 6
        elif decided:
            if a > b:
                st.sets_a += 1
            else:
                st.sets_b += 1
        else:
            st.games_a, st.games_b = a, b

    if str(period).upper().startswith("FT") or st.sets_a >= need or st.sets_b >= need:
        st.finished = True

    # Server is not published. Parity of games played is the best available guess;
    # it is right about half the time and the error is worth roughly one game.
    st.server = (st.games_a + st.games_b) % 2
    return st
