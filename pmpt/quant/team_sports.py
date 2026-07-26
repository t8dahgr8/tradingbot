"""Score-and-clock probability models for major team sports.

The market's own pregame probability is the strength prior.  We then condition
that prior on the observed home-away score and the amount of regulation time
remaining.  This is intentionally much smaller in scope than a sportsbook
model: it does not pretend to know lineups, injuries, possession, or play-by-play
quality that the public Polymarket sports feed does not provide.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from statistics import NormalDist


TEAM_SPORTS = frozenset({
    "basketball",
    "football",
    "baseball",
    "hockey",
    "soccer",
})


@dataclass(frozen=True)
class TeamGameState:
    home_score: int
    away_score: int
    progress: float
    period: str = ""
    elapsed: str = ""
    clock_known: bool = False
    finished: bool = False
    valid: bool = True

    @property
    def margin(self) -> int:
        return self.home_score - self.away_score


@dataclass(frozen=True)
class TeamSportProfile:
    # Approximate standard deviation of the final home-away margin.
    margin_sd: float
    # Recent scoring is already represented in the score.  This deliberately
    # small multiplier gives a run limited influence without double-counting it.
    momentum_weight: float
    momentum_cap: float


PROFILES = {
    "basketball": TeamSportProfile(12.0, 0.15, 12.0),
    "football": TeamSportProfile(14.0, 0.12, 14.0),
    "baseball": TeamSportProfile(3.6, 0.10, 3.0),
    "hockey": TeamSportProfile(2.2, 0.10, 2.0),
}

_NORMAL = NormalDist()
_FINAL_MARKERS = ("FT", "FINAL", "F/OT", "F/SO")
_PAUSED_MARKERS = ("SUSPEND", "DELAY", "POSTPON", "CANCEL", "FORFEIT")


def is_team_sport(sport: str) -> bool:
    return sport in TEAM_SPORTS


def pregame_state() -> TeamGameState:
    return TeamGameState(0, 0, 0.0, clock_known=True)


def _clock_minutes(value: str) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        if "+" in text:
            return sum(float(part) for part in text.split("+") if part)
        if ":" in text:
            minute, second = text.split(":", 1)
            return float(minute) + float(second) / 60.0
        return float(text)
    except (TypeError, ValueError):
        return None


def _period_number(period: str) -> int | None:
    match = re.search(r"(\d+)", period)
    return int(match.group(1)) if match else None


def _quarter_progress(
    period: str,
    elapsed: str,
    period_minutes: float,
    periods: int,
) -> tuple[float, bool]:
    upper = period.upper().replace(" ", "")
    if "OT" in upper:
        number = _period_number(upper) or (periods + 1)
        return min(0.995, 0.97 + 0.005 * max(0, number - periods)), True
    number = _period_number(upper)
    if number is None:
        if upper in ("HT", "HALFTIME"):
            return 0.5, True
        return 0.0, False
    if number > periods:
        return min(0.995, 0.97 + 0.005 * max(0, number - periods)), True

    clock = _clock_minutes(elapsed)
    if clock is None:
        # Period is useful but too coarse for full confidence.
        played_in_period = period_minutes * 0.5
        known = False
    else:
        # NBA/NFL/NHL feeds expose the familiar game clock: time remaining in
        # the current period, despite the field being named `elapsed`.
        played_in_period = period_minutes - min(max(clock, 0.0), period_minutes)
        known = True
    played = (number - 1) * period_minutes + played_in_period
    return played / (period_minutes * periods), known


def _half_progress(
    period: str,
    elapsed: str,
    half_minutes: float,
) -> tuple[float, bool]:
    upper = period.upper().replace(" ", "")
    if upper in ("HT", "HALFTIME"):
        return 0.5, True
    number = 2 if "2H" in upper else 1
    clock = _clock_minutes(elapsed)
    if clock is None:
        played_in_half = half_minutes * 0.5
        known = False
    else:
        played_in_half = half_minutes - min(max(clock, 0.0), half_minutes)
        known = True
    return ((number - 1) * half_minutes + played_in_half) / (2 * half_minutes), known


def _soccer_progress(period: str, elapsed: str) -> tuple[float, bool]:
    upper = period.upper().replace(" ", "")
    if upper in ("HT", "HALFTIME", "BREAK"):
        return 0.5, True
    if "ET" in upper or "PEN" in upper:
        return 0.985, True

    clock = _clock_minutes(elapsed)
    if clock is None:
        return (0.75 if "2H" in upper else 0.25), False

    # Providers differ on whether second-half time is cumulative.  Supporting
    # both forms avoids treating 22:00 in 2H as the 22nd minute of the match.
    minute = clock
    if "2H" in upper and minute <= 45.0:
        minute += 45.0
    return minute / 90.0, True


def _baseball_progress(period: str) -> tuple[float, bool]:
    upper = period.upper().strip()
    inning = _period_number(upper)
    if inning is None:
        return 0.0, False

    if upper.startswith(("END", "E")):
        part = 1.0
    elif upper.startswith(("BOT", "BOTTOM", "B")):
        part = 0.75
    elif upper.startswith(("MID", "M")):
        part = 0.5
    elif upper.startswith(("TOP", "T")):
        part = 0.25
    else:
        part = 0.5
    return ((inning - 1) + part) / 9.0, True


def parse_team_score(
    sport: str,
    league: str,
    score: str,
    period: str,
    elapsed: str = "",
    ended: bool = False,
) -> TeamGameState:
    """Parse the sports stream's documented ``home-away`` score."""
    match = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", str(score or ""))
    if not match:
        return TeamGameState(0, 0, 0.0, period, elapsed, valid=False)

    home, away = int(match.group(1)), int(match.group(2))
    upper = str(period or "").upper()
    finished = ended or any(marker == upper for marker in _FINAL_MARKERS)
    if any(marker in upper for marker in _PAUSED_MARKERS):
        return TeamGameState(
            home, away, 0.0, period, elapsed, finished=finished, valid=False
        )
    if sport == "soccer" and ("PEN" in upper or "ET" in upper):
        return TeamGameState(
            home, away, 0.0, period, elapsed, finished=finished, valid=False
        )
    if finished:
        return TeamGameState(
            home, away, 1.0, period, elapsed, clock_known=True, finished=True
        )

    league = str(league or "").lower()
    if sport == "basketball":
        if "H" in upper and "Q" not in upper:
            progress, known = _half_progress(period, elapsed, 20.0)
        else:
            period_minutes = 12.0 if league == "nba" else 10.0
            progress, known = _quarter_progress(period, elapsed, period_minutes, 4)
    elif sport == "football":
        progress, known = _quarter_progress(period, elapsed, 15.0, 4)
    elif sport == "hockey":
        progress, known = _quarter_progress(period, elapsed, 20.0, 3)
    elif sport == "soccer":
        progress, known = _soccer_progress(period, elapsed)
    elif sport == "baseball":
        progress, known = _baseball_progress(period)
    else:
        return TeamGameState(home, away, 0.0, period, elapsed, valid=False)

    return TeamGameState(
        home_score=home,
        away_score=away,
        progress=min(max(progress, 0.0), 0.995),
        period=period,
        elapsed=elapsed,
        clock_known=known,
        finished=False,
        valid=True,
    )


def _clamp_probability(value: float) -> float:
    return min(max(value, 0.001), 0.999)


def _poisson_mass(lam: float, limit: int = 12) -> list[float]:
    values = [math.exp(-lam)]
    for goals in range(1, limit + 1):
        values.append(values[-1] * lam / goals)
    # Fold the tiny truncated tail into the last bucket.
    values[-1] += max(0.0, 1.0 - sum(values))
    return values


def _neutral_soccer_probabilities(state: TeamGameState) -> dict[str, float]:
    remaining = max(0.0, 1.0 - state.progress)
    if remaining <= 1e-9:
        if state.home_score > state.away_score:
            return {"home": 1.0, "draw": 0.0, "away": 0.0}
        if state.home_score < state.away_score:
            return {"home": 0.0, "draw": 0.0, "away": 1.0}
        return {"home": 0.0, "draw": 1.0, "away": 0.0}

    # A neutral 2.6-goal match is used only as a likelihood update.  The actual
    # team strength still comes from the market anchor.
    each_lambda = 1.3 * remaining
    home_goals = _poisson_mass(each_lambda)
    away_goals = _poisson_mass(each_lambda)
    result = {"home": 0.0, "draw": 0.0, "away": 0.0}
    for add_home, p_home in enumerate(home_goals):
        for add_away, p_away in enumerate(away_goals):
            final_home = state.home_score + add_home
            final_away = state.away_score + add_away
            role = "home" if final_home > final_away else (
                "away" if final_home < final_away else "draw"
            )
            result[role] += p_home * p_away
    return result


def _soccer_fair_probability(
    anchor_probability: float,
    anchor_state: TeamGameState,
    state: TeamGameState,
    outcome0_role: str,
) -> float:
    if outcome0_role not in ("home", "away", "draw"):
        raise ValueError(f"unsupported soccer outcome role: {outcome0_role!r}")
    anchor_neutral = _clamp_probability(
        _neutral_soccer_probabilities(anchor_state)[outcome0_role]
    )
    current_neutral = _clamp_probability(
        _neutral_soccer_probabilities(state)[outcome0_role]
    )
    anchor_odds = _clamp_probability(anchor_probability) / (
        1.0 - _clamp_probability(anchor_probability)
    )
    likelihood_ratio = (
        current_neutral / (1.0 - current_neutral)
    ) / (
        anchor_neutral / (1.0 - anchor_neutral)
    )
    return _clamp_probability(
        anchor_odds * likelihood_ratio / (1.0 + anchor_odds * likelihood_ratio)
    )


def fair_probability(
    sport: str,
    anchor_probability: float,
    anchor_state: TeamGameState,
    state: TeamGameState,
    outcome0_role: str,
    momentum_home: float = 0.0,
) -> float:
    """Return P(market outcome 0) from a calibrated score state."""
    if not state.valid or not anchor_state.valid:
        raise ValueError("cannot price an invalid team-game state")
    if sport == "soccer":
        return _soccer_fair_probability(
            anchor_probability,
            anchor_state,
            state,
            outcome0_role,
        )

    profile = PROFILES.get(sport)
    if profile is None or outcome0_role not in ("home", "away"):
        raise ValueError(
            f"unsupported team model: sport={sport!r}, role={outcome0_role!r}"
        )

    anchor_home = (
        anchor_probability if outcome0_role == "home" else 1.0 - anchor_probability
    )
    anchor_remaining = max(0.01, 1.0 - anchor_state.progress)
    anchor_z = _NORMAL.inv_cdf(_clamp_probability(anchor_home))
    skill_margin = (
        anchor_z * profile.margin_sd * math.sqrt(anchor_remaining)
        - anchor_state.margin
    ) / anchor_remaining

    remaining = max(0.0025, 1.0 - state.progress)
    momentum = min(
        max(momentum_home, -profile.momentum_cap),
        profile.momentum_cap,
    )
    momentum_margin = profile.momentum_weight * momentum * math.sqrt(remaining)
    expected_margin = state.margin + remaining * skill_margin + momentum_margin
    p_home = _NORMAL.cdf(
        expected_margin / (profile.margin_sd * math.sqrt(remaining))
    )
    p_home = _clamp_probability(p_home)
    return p_home if outcome0_role == "home" else 1.0 - p_home
