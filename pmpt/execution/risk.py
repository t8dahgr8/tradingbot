"""
Position sizing and hard limits.

On a $100 bankroll the thing that kills you is not a bad model, it is one
oversized position in a market you misread. Every check here is a veto: a trade
has to survive all of them, and sizing is deliberately conservative.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..models import Side, Signal, TradableMarket
from .portfolio import Portfolio

log = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    # --- sizing ---
    # Fraction of full Kelly. Full Kelly on a model you have not validated is a
    # good way to lose the account; 0.25 is already aggressive.
    kelly_fraction: float = 0.20
    max_position_pct: float = 0.10      # of equity, per single trade
    max_market_exposure_pct: float = 0.15
    max_total_exposure_pct: float = 0.60
    min_order_shares: float = 5.0       # Polymarket's own minimum
    max_order_shares: float = 500.0

    # --- entry filters ---
    min_edge: float = 0.03              # model vs price, after expected costs
    min_confidence: float = 0.5
    max_spread: float = 0.05            # skip markets wider than this
    min_book_depth: float = 50.0        # shares available at our price or better
    max_book_age_ms: int = 5_000        # stale book => do not trade
    min_price: float = 0.05             # avoid the tails: resolution risk is
    max_price: float = 0.95             # asymmetric and edges are illusory there

    # --- portfolio limits ---
    max_concurrent_positions: int = 5
    max_positions_per_market: int = 1
    cooldown_ms: int = 20_000           # per market, between entries

    # --- kill switches ---
    max_drawdown_pct: float = 0.25      # from peak equity
    daily_loss_limit_pct: float = 0.15
    min_equity: float = 20.0

    # --- costs assumed when computing net edge ---
    assumed_slippage: float = 0.005
    taker_fee_rate: float = 0.0


@dataclass
class RiskState:
    halted: bool = False
    halt_reason: str = ""
    day_start_equity: float = 0.0
    day_key: str = ""
    last_entry_ms: dict[str, int] = field(default_factory=dict)
    last_entry_score: dict[str, str] = field(default_factory=dict)


class RiskManager:
    def __init__(self, config: RiskConfig | None = None):
        self.cfg = config or RiskConfig()
        self.state = RiskState()

    # -- kill switches -----------------------------------------------------

    def check_halt(self, portfolio: Portfolio, marks: dict[str, float], day_key: str
                   ) -> bool:
        """Returns True if trading should stop. Once halted, stays halted."""
        cfg = self.cfg
        eq = portfolio.equity(marks)

        if self.state.day_key != day_key:
            self.state.day_key = day_key
            self.state.day_start_equity = eq

        if self.state.halted:
            return True

        if eq < cfg.min_equity:
            self._halt(f"equity {eq:.2f} below floor {cfg.min_equity:.2f}")
        elif portfolio.max_drawdown > cfg.max_drawdown_pct:
            self._halt(
                f"drawdown {portfolio.max_drawdown:.1%} exceeds {cfg.max_drawdown_pct:.1%}"
            )
        elif self.state.day_start_equity > 0:
            day_loss = 1.0 - eq / self.state.day_start_equity
            if day_loss > cfg.daily_loss_limit_pct:
                self._halt(f"daily loss {day_loss:.1%} exceeds {cfg.daily_loss_limit_pct:.1%}")

        return self.state.halted

    def _halt(self, reason: str) -> None:
        self.state.halted = True
        self.state.halt_reason = reason
        log.error("TRADING HALTED: %s", reason)

    # -- entry gate --------------------------------------------------------

    def net_edge(
        self,
        signal: Signal,
        market: TradableMarket | None = None,
        taker_legs: int = 2,
    ) -> float:
        """Edge after the costs we expect to actually pay."""
        cfg = self.cfg
        if market is not None and market.fees_enabled:
            rate = market.fee_rate or cfg.taker_fee_rate
        elif market is not None:
            rate = 0.0
        else:
            rate = cfg.taker_fee_rate
        p = signal.market_price
        fee = max(0, taker_legs) * rate * p * (1.0 - p)
        return signal.edge - cfg.assumed_slippage - fee

    def approve(
        self,
        signal: Signal,
        market: TradableMarket,
        portfolio: Portfolio,
        marks: dict[str, float],
        book_age_ms: int,
        spread: float | None,
        depth: float,
        ts_ms: int,
        taker_legs: int = 2,
    ) -> tuple[bool, str]:
        cfg = self.cfg

        if self.state.halted:
            return False, f"halted: {self.state.halt_reason}"
        if not market.accepting_orders:
            return False, "market not accepting orders"
        if signal.confidence < cfg.min_confidence:
            return False, f"confidence {signal.confidence:.2f} < {cfg.min_confidence:.2f}"
        if not (cfg.min_price <= signal.market_price <= cfg.max_price):
            return False, f"price {signal.market_price:.3f} outside tradeable band"
        if book_age_ms > cfg.max_book_age_ms:
            return False, f"stale book ({book_age_ms}ms)"
        if spread is None or spread - cfg.max_spread > 1e-9:
            return False, f"spread {spread} too wide"
        if depth < cfg.min_book_depth:
            return False, f"depth {depth:.0f} below {cfg.min_book_depth:.0f}"

        ne = self.net_edge(signal, market, taker_legs)
        if ne < cfg.min_edge:
            return False, f"net edge {ne:.4f} < {cfg.min_edge:.4f}"

        last = self.state.last_entry_ms.get(market.market_id, 0)
        if ts_ms - last < cfg.cooldown_ms:
            return False, "cooldown"

        score_key = str(signal.metadata.get("score_key", ""))
        if score_key and self.state.last_entry_score.get(market.market_id) == score_key:
            return False, "already traded this score"

        if len(portfolio.positions) >= cfg.max_concurrent_positions:
            if signal.token_id not in portfolio.positions:
                return False, "max concurrent positions reached"

        in_market = sum(
            1 for p in portfolio.positions.values() if p.market_id == market.market_id
        )
        if in_market >= cfg.max_positions_per_market and signal.token_id not in portfolio.positions:
            return False, "max positions in this market"

        eq = portfolio.equity(marks)
        if portfolio.exposure() >= cfg.max_total_exposure_pct * eq:
            return False, "total exposure cap"
        if portfolio.exposure_in_market(market.market_id) >= cfg.max_market_exposure_pct * eq:
            return False, "per-market exposure cap"

        return True, "ok"

    # -- sizing ------------------------------------------------------------

    def size_order(
        self,
        signal: Signal,
        market: TradableMarket,
        portfolio: Portfolio,
        marks: dict[str, float],
        available_depth: float,
    ) -> float:
        """Shares to trade. Returns 0 if no size survives the constraints.

        Kelly for a binary contract bought at price `c` that pays 1: the bet wins
        (1-c)/c per unit staked, so f* = (p - c) / (1 - c). We then scale that
        down hard and clamp it against every exposure cap.
        """
        cfg = self.cfg
        eq = portfolio.equity(marks)
        if eq <= 0:
            return 0.0

        c = signal.market_price
        p = signal.fair_value if signal.side == Side.BUY else 1.0 - signal.fair_value
        cost = c if signal.side == Side.BUY else 1.0 - c
        cost = min(max(cost, 1e-4), 1 - 1e-4)

        kelly = (p - cost) / (1.0 - cost)
        if kelly <= 0:
            return 0.0
        stake_frac = min(kelly * cfg.kelly_fraction, cfg.max_position_pct)
        stake = stake_frac * eq * signal.confidence

        # Respect what is left under the caps.
        room_total = max(0.0, cfg.max_total_exposure_pct * eq - portfolio.exposure())
        room_market = max(
            0.0,
            cfg.max_market_exposure_pct * eq - portfolio.exposure_in_market(market.market_id),
        )
        stake = min(stake, room_total, room_market, portfolio.cash)
        if stake <= 0:
            return 0.0

        shares = stake / cost
        shares = min(shares, available_depth, cfg.max_order_shares)

        min_shares = max(cfg.min_order_shares, market.min_order_size)
        if shares < min_shares:
            # Only round up if the resulting stake still fits every cap.
            if min_shares * cost <= min(room_total, room_market, portfolio.cash):
                shares = min_shares
            else:
                return 0.0

        # Round to whole shares; Polymarket accepts fractions but whole shares
        # keep the journal readable and avoid dust positions.
        return float(int(shares))

    def record_entry(self, market_id: str, ts_ms: int, score_key: str = "") -> None:
        self.state.last_entry_ms[market_id] = ts_ms
        if score_key:
            self.state.last_entry_score[market_id] = score_key
