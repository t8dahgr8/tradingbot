"""
Market discovery via Polymarket's Gamma API.

Gamma tells you what exists. It is unauthenticated and rate limited, so results
are cached and refreshed on a slow cadence -- market *structure* changes far less
often than prices do.

Uses urllib from the stdlib rather than aiohttp so the discovery layer has no
install footprint. Blocking calls are pushed to a thread by the engine.
"""

from __future__ import annotations

import json
import logging
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from ..models import TradableMarket

log = logging.getLogger(__name__)

GAMMA = "https://gamma-api.polymarket.com"

# Broad sport tags. Football needs two tags because NFL and college football do
# not currently share one public parent tag.
TAG_IDS = {
    "tennis": (864,),
    "table_tennis": (103767,),
    "basketball": (28,),
    "football": (450, 100351),
    "baseball": (678,),
    "hockey": (899,),
    "soccer": (100350,),
}

USER_AGENT = "pmpt-paper-trader/1.0 (+research)"
DISCOVERY_LOOKBACK_HOURS = 24
DISCOVERY_LOOKAHEAD_HOURS = 36
NEAR_FUTURE_HOURS = 2


def _get(path: str, params: dict[str, Any] | None = None, timeout: float = 15.0) -> Any:
    url = f"{GAMMA}{path}"
    if params:
        flat: list[tuple[str, str]] = []
        for k, v in params.items():
            if v is None:
                continue
            if isinstance(v, (list, tuple)):
                flat.extend((k, str(i)) for i in v)
            else:
                flat.append((k, str(v)))
        url = f"{url}?{urllib.parse.urlencode(flat)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        log.warning("gamma %s -> HTTP %s", path, e.code)
    except Exception as e:  # noqa: BLE001 - network layer, never crash the loop
        log.warning("gamma %s failed: %s", path, e)
    return None


def _parse_json_field(raw: Any, default: list | None = None) -> list:
    """Gamma returns several array fields as JSON-encoded *strings*. Classic gotcha."""
    if raw is None:
        return default or []
    if isinstance(raw, list):
        return raw
    try:
        val = json.loads(raw)
        return val if isinstance(val, list) else (default or [])
    except (json.JSONDecodeError, TypeError):
        return default or []


def _best_of_from_event(event: dict) -> int:
    """Infer match format. Grand Slam men's singles is best of 5, everything else 3."""
    text = f"{event.get('title', '')} {event.get('slug', '')} {event.get('ticker', '')}".lower()
    slams = ("australian-open", "roland-garros", "french-open", "wimbledon", "us-open")
    is_slam = any(s in text for s in slams)
    is_womens = text.startswith("wta") or "wta-" in text or "women" in text
    return 5 if (is_slam and not is_womens) else 3


def _gamma_time(value: datetime) -> str:
    """Gamma's date filters require a UTC Z suffix, not an encoded +00:00."""
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _event_start(event: dict) -> datetime | None:
    """Return the scheduled game time, not the market creation timestamp."""
    candidates = [
        event.get("startTime"),
        event.get("eventStartTime"),
    ]
    candidates.extend(
        raw.get("gameStartTime")
        for raw in (event.get("markets") or [])
        if raw.get("gameStartTime")
    )
    # Older event payloads used startDate for the actual start time.
    candidates.append(event.get("startDate"))
    for raw in candidates:
        if not raw:
            continue
        try:
            value = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            return (
                value
                if value.tzinfo is not None
                else value.replace(tzinfo=timezone.utc)
            )
        except (TypeError, ValueError):
            continue
    return None


_AWAY_FIRST_LEAGUES = {
    "nba",
    "wnba",
    "nfl",
    "cfb",
    "mlb",
    "kbo",
    "nhl",
}


def _league_from_event(event: dict) -> str:
    value = str(event.get("seriesSlug") or "").strip().lower()
    if value:
        return value
    slug = str(event.get("slug") or "").strip().lower()
    return slug.split("-", 1)[0] if slug else ""


def _normal_name(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _event_participants(event: dict) -> tuple[str, str] | None:
    title = str(event.get("title") or "")
    parts = re.split(r"\s+vs\.?\s+", title, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) != 2:
        return None
    return parts[0].strip(), parts[1].split(" - ", 1)[0].strip()


def _outcome0_role(raw: dict, event: dict, sport: str) -> str | None:
    """Map outcome zero onto the score stream's documented home-away order."""
    outcomes = _parse_json_field(raw.get("outcomes"))
    if not outcomes:
        return None
    league = _league_from_event(event)
    display_first = "away" if league in _AWAY_FIRST_LEAGUES else "home"

    if str(outcomes[0]).strip().lower() not in ("yes", "no"):
        return display_first

    question = str(raw.get("question") or "")
    if "draw" in question.lower():
        return "draw"
    participants = _event_participants(event)
    if participants is None:
        return None
    first, second = participants
    q = _normal_name(question)
    first_role = display_first
    second_role = "home" if display_first == "away" else "away"
    if _normal_name(first) and _normal_name(first) in q:
        return first_role
    if _normal_name(second) and _normal_name(second) in q:
        return second_role
    return None


def market_from_gamma(
    raw: dict, event: dict | None = None, sport: str = "tennis"
) -> TradableMarket | None:
    """Flatten a Gamma market object into the shape the trader uses."""
    event = event or {}
    token_ids = _parse_json_field(raw.get("clobTokenIds"))
    outcomes = _parse_json_field(raw.get("outcomes"))
    if len(token_ids) < 2 or len(outcomes) < 2:
        return None
    if not raw.get("enableOrderBook", False):
        return None
    outcome0_role = _outcome0_role(raw, event, sport)
    if outcome0_role is None:
        return None

    fee_schedule = raw.get("feeSchedule") or {}
    return TradableMarket(
        market_id=str(raw.get("id", "")),
        condition_id=str(raw.get("conditionId", "")),
        question=str(raw.get("question", "")),
        slug=str(raw.get("slug", "")),
        token_ids=(str(token_ids[0]), str(token_ids[1])),
        outcomes=(str(outcomes[0]), str(outcomes[1])),
        tick_size=float(raw.get("orderPriceMinTickSize") or 0.01),
        min_order_size=float(raw.get("orderMinSize") or 5),
        accepting_orders=bool(raw.get("acceptingOrders", False)),
        sport=sport,
        league=_league_from_event(event),
        outcome0_role=outcome0_role,
        game_id=event.get("gameId"),
        event_slug=str(event.get("slug", "")),
        best_of=5 if sport == "table_tennis" else _best_of_from_event(event),
        fees_enabled=bool(raw.get("feesEnabled", False)),
        fee_rate=float(fee_schedule.get("rate") or 0.0),
    )


class GammaClient:
    """Discovery + a light cache of market structure."""

    def __init__(self, sports: Iterable[str] = ("tennis",)):
        self.sports = list(sports)
        self.markets: dict[str, TradableMarket] = {}       # market_id -> market
        self.by_token: dict[str, TradableMarket] = {}      # token_id  -> market
        self.events: dict[str, dict] = {}                  # event id  -> raw event

    # -- discovery ---------------------------------------------------------

    def fetch_events(self, sport: str, limit: int = 100) -> list[dict]:
        """Open events for a sport, nearest first.

        Prefers tag_id (stable) and falls back to public-search, because tag
        slugs on Gamma are not reliably honoured as filters.
        """
        tag_ids = TAG_IDS.get(sport, ())
        out: list[dict] = []

        if tag_ids:
            now = datetime.now(timezone.utc)
            windows = (
                (
                    now - timedelta(hours=DISCOVERY_LOOKBACK_HOURS),
                    now + timedelta(hours=NEAR_FUTURE_HOURS),
                    "false",
                ),
                (
                    now + timedelta(hours=NEAR_FUTURE_HOURS),
                    now + timedelta(hours=DISCOVERY_LOOKAHEAD_HOURS),
                    "true",
                ),
            )
            seen: set[str] = set()
            for tag_id in tag_ids:
                for start, end, ascending in windows:
                    data = _get(
                        "/events/keyset",
                        {
                            "tag_id": tag_id,
                            "closed": "false",
                            "limit": min(max(limit, 1), 500),
                            "order": "startTime",
                            "ascending": ascending,
                            "start_time_min": _gamma_time(start),
                            "start_time_max": _gamma_time(end),
                        },
                    )
                    events = (
                        data.get("events")
                        if isinstance(data, dict)
                        else data
                    )
                    if not isinstance(events, list):
                        continue
                    for event in events:
                        key = str(event.get("id") or event.get("slug") or "")
                        if key and key not in seen:
                            seen.add(key)
                            out.append(event)

        if not out:
            data = _get("/public-search", {"q": sport, "limit_per_type": limit})
            if isinstance(data, dict):
                out = [e for e in (data.get("events") or []) if not e.get("closed")]

        now = datetime.now(timezone.utc)
        out.sort(
            key=lambda event: (
                _event_start(event) is None,
                abs((_event_start(event) - now).total_seconds())
                if _event_start(event)
                else float("inf"),
            )
        )
        return out

    def refresh(self, only_live: bool = False, min_liquidity: float = 0.0
                ) -> list[TradableMarket]:
        """Rebuild the tradeable universe. Call this every few minutes, not every tick."""
        found: list[TradableMarket] = []

        for sport in self.sports:
            for event in self.fetch_events(sport):
                if event.get("closed") or event.get("ended"):
                    continue
                # Without this ID the public score stream can never update the
                # market, so it can never produce a valid in-play signal.
                if event.get("gameId") is None:
                    continue
                if only_live and not event.get("live"):
                    continue
                if min_liquidity and float(event.get("liquidity") or 0) < min_liquidity:
                    continue

                self.events[str(event.get("id"))] = event
                for raw in event.get("markets") or []:
                    if raw.get("closed") or not raw.get("acceptingOrders"):
                        continue
                    # Only straight match-winner markets. Props and set-betting
                    # need their own models and are out of scope here.
                    if raw.get("sportsMarketType") not in (None, "", "moneyline"):
                        continue
                    m = market_from_gamma(raw, event, sport)
                    if m is None:
                        continue
                    self.markets[m.market_id] = m
                    self.by_token[m.token_ids[0]] = m
                    self.by_token[m.token_ids[1]] = m
                    found.append(m)

        log.info("discovery: %d tradeable markets across %s", len(found), self.sports)
        return found

    def activate(self, markets: Iterable[TradableMarket]) -> None:
        """Restrict price routing to the engine's selected watchlist."""
        selected = list(markets)
        self.markets = {m.market_id: m for m in selected}
        self.by_token = {
            token: market
            for market in selected
            for token in market.token_ids
        }

    # -- live game state ---------------------------------------------------

    def event_for_market(self, market: TradableMarket) -> dict | None:
        for ev in self.events.values():
            for raw in ev.get("markets") or []:
                if str(raw.get("id")) == market.market_id:
                    return ev
        return None

    def fetch_event(self, event_id: str) -> dict | None:
        data = _get(f"/events/{event_id}")
        if isinstance(data, dict):
            self.events[str(event_id)] = data
            return data
        return None

    def all_token_ids(self) -> list[str]:
        return list(self.by_token.keys())
