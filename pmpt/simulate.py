"""
Offline match simulator.

This exists so you can answer the only question that matters -- "does this
strategy make money, and why?" -- without waiting for real matches or trusting a
single lucky session.

It simulates tennis and table tennis matches point by point from known serve
probabilities, then builds a synthetic order book that lags the true probability
by a configurable amount. The lag IS the edge. Set `catchup_rate` to 1.0 (the
market reprices instantly) and the strategy should make roughly nothing minus
costs. If it still shows a profit at zero lag, something is wrong and you have
found a bug rather than an edge.

Everything runs through the same broker, risk manager and portfolio as live
trading, so the fill assumptions are identical.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field

from .execution.paper_broker import BrokerConfig, PaperBroker
from .execution.portfolio import Portfolio
from .execution.risk import RiskConfig, RiskManager
from .models import Level, Order, OrderBook, OrderType, Side, TradableMarket
from .quant import table_tennis as tt
from .quant import tennis as tn
from .strategy.live_model import LiveModelStrategy, StrategyConfig

log = logging.getLogger(__name__)


@dataclass
class SimConfig:
    n_matches: int = 200
    sport: str = "tennis"
    best_of: int = 3
    starting_cash: float = 100.0
    # How fast the synthetic market closes the gap to true fair value, per book
    # update. 1.0 = instantaneous (no edge exists), 0.15 = sluggish.
    catchup_rate: float = 0.25
    books_per_game: int = 6
    ms_per_book: int = 4_000
    # Half-spread in probability terms, and depth per level in shares.
    half_spread: float = 0.01
    level_depth: float = 400.0
    # Random noise added to the market's view each update.
    price_noise: float = 0.004
    seed: int = 20260725
    verbose: bool = False


# --------------------------------------------------------------------------
# Match simulation
# --------------------------------------------------------------------------

def _sim_tennis_match(rng: random.Random, pa: float, pb: float, best_of: int
                      ) -> list[tuple[str, str, bool]]:
    """Play a match point by point. Returns (score_string, period, ended) per game."""
    need = best_of // 2 + 1
    sets_a = sets_b = 0
    completed: list[tuple[int, int]] = []
    timeline: list[tuple[str, str, bool]] = []
    server = 0

    def render(cur: tuple[int, int] | None) -> str:
        parts = [f"{a}-{b}" for a, b in completed]
        if cur is not None:
            parts.append(f"{cur[0]}-{cur[1]}")
        return ", ".join(parts)

    while sets_a < need and sets_b < need:
        ga = gb = 0
        while True:
            # Play one game.
            p = pa if server == 0 else pb
            sa = sb = 0
            while True:
                if rng.random() < p:
                    sa += 1
                else:
                    sb += 1
                if max(sa, sb) >= 4 and abs(sa - sb) >= 2:
                    break
            server_won = sa > sb
            if (server == 0) == server_won:
                ga += 1
            else:
                gb += 1
            server = 1 - server

            set_over = (max(ga, gb) >= 6 and abs(ga - gb) >= 2)
            if ga == 6 and gb == 6:
                # Tiebreak: resolve it directly from the model's own probability.
                a_wins = rng.random() < tn.tiebreak_win_prob(pa, pb, server=server)
                if a_wins:
                    ga = 7
                else:
                    gb = 7
                set_over = True

            timeline.append((render((ga, gb)), str(len(completed) + 1), False))
            if set_over:
                break

        completed.append((ga, gb))
        if ga > gb:
            sets_a += 1
        else:
            sets_b += 1

    timeline.append((render(None), "FT", True))
    return timeline


def _sim_table_tennis_match(rng: random.Random, pa: float, pb: float, best_of: int
                            ) -> list[tuple[str, str, bool]]:
    need = best_of // 2 + 1
    ga = gb = 0
    completed: list[tuple[int, int]] = []
    timeline: list[tuple[str, str, bool]] = []
    first_server = 0

    def render(cur: tuple[int, int] | None) -> str:
        parts = [f"{a}-{b}" for a, b in completed]
        if cur is not None:
            parts.append(f"{cur[0]}-{cur[1]}")
        return ", ".join(parts)

    while ga < need and gb < need:
        a = b = 0
        while True:
            n = a + b
            deuce = a >= 10 and b >= 10
            srv_first = (n % 2 == 0) if deuce else ((n // 2) % 2 == 0)
            server = first_server if srv_first else 1 - first_server
            p_to_a = pa if server == 0 else 1 - pb
            if rng.random() < p_to_a:
                a += 1
            else:
                b += 1
            timeline.append((render((a, b)), str(len(completed) + 1), False))
            if max(a, b) >= 11 and abs(a - b) >= 2:
                break
        completed.append((a, b))
        if a > b:
            ga += 1
        else:
            gb += 1
        first_server = 1 - first_server

    timeline.append((render(None), "FT", True))
    return timeline


# --------------------------------------------------------------------------
# Synthetic book
# --------------------------------------------------------------------------

def _make_book(token: str, prob: float, cfg: SimConfig, ts: int) -> OrderBook:
    """A simple symmetric book centred on `prob` with a few levels of depth."""
    hs = cfg.half_spread
    tick = 0.01
    bid = max(tick, min(1 - tick, round((prob - hs) / tick) * tick))
    ask = max(bid + tick, min(1 - tick, round((prob + hs) / tick) * tick))
    bids = [Level(round(bid - i * tick, 4), cfg.level_depth * (1 + i))
            for i in range(5) if bid - i * tick > 0]
    asks = [Level(round(ask + i * tick, 4), cfg.level_depth * (1 + i))
            for i in range(5) if ask + i * tick < 1]
    return OrderBook(token, bids, asks, timestamp_ms=ts, tick_size=tick)


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

@dataclass
class SimResult:
    starting_cash: float
    final_equity: float
    matches: int
    trades: int
    signals: int
    realized_pnl: float
    fees: float
    max_drawdown: float
    wins: int = 0
    losses: int = 0
    rejections: dict = field(default_factory=dict)
    portfolio: object = None   # kept so the dashboard can render the run

    @property
    def total_return_pct(self) -> float:
        return 100 * (self.final_equity / self.starting_cash - 1) if self.starting_cash else 0.0

    def render(self) -> str:
        return "\n".join([
            "=" * 58,
            "  SIMULATION RESULT",
            "=" * 58,
            f"  Matches simulated    {self.matches:>10d}",
            f"  Signals generated    {self.signals:>10d}",
            f"  Trades executed      {self.trades:>10d}",
            f"  Starting bankroll    ${self.starting_cash:>9.2f}",
            f"  Final equity         ${self.final_equity:>9.2f}",
            f"  Total return         {self.total_return_pct:>9.2f} %",
            f"  Realized P&L         ${self.realized_pnl:>9.2f}",
            f"  Fees paid            ${self.fees:>9.2f}",
            f"  Max drawdown         {100*self.max_drawdown:>9.2f} %",
            f"  Winning / losing     {self.wins:>6d} / {self.losses:<6d}",
            "=" * 58,
        ])


def run_simulation(
    cfg: SimConfig | None = None,
    strategy_cfg: StrategyConfig | None = None,
    risk_cfg: RiskConfig | None = None,
    broker_cfg: BrokerConfig | None = None,
) -> SimResult:
    cfg = cfg or SimConfig()
    rng = random.Random(cfg.seed)

    broker = PaperBroker(broker_cfg or BrokerConfig(seed=cfg.seed))
    portfolio = Portfolio(cfg.starting_cash)
    risk = RiskManager(risk_cfg or RiskConfig())
    strategy = LiveModelStrategy(strategy_cfg or StrategyConfig())

    ts = 1_700_000_000_000
    signals = trades = wins = losses = 0
    rejections: dict[str, int] = {}

    for i in range(cfg.n_matches):
        # A fresh market per match. Token ids just need to be unique.
        tok0, tok1 = f"sim{i}_A", f"sim{i}_B"
        market = TradableMarket(
            market_id=f"m{i}",
            condition_id=f"c{i}",
            question=f"Sim match {i}",
            slug=f"sim-match-{i}",
            token_ids=(tok0, tok1),
            outcomes=("A", "B"),
            tick_size=0.01,
            min_order_size=5,
            sport=cfg.sport,
            best_of=cfg.best_of,
        )

        # True strength, and the market's (correct) pre-match view of it.
        true_p0 = rng.uniform(0.20, 0.80)
        if cfg.sport == "table_tennis":
            pa, pb = tt.calibrate_serve_probs(true_p0, best_of=cfg.best_of)
            timeline = _sim_table_tennis_match(rng, pa, pb, cfg.best_of)
        else:
            pa, pb = tn.calibrate_serve_probs(true_p0, best_of=cfg.best_of)
            timeline = _sim_tennis_match(rng, pa, pb, cfg.best_of)

        if not strategy.set_anchor(market, true_p0, ts, games_played=0):
            continue
        equity_at_match_start = portfolio.equity({})

        market_p = true_p0          # where the synthetic book is centred
        for token in (tok0, tok1):
            p = market_p if token == tok0 else 1 - market_p
            broker.on_book(_make_book(token, p, cfg, ts), ts)

        final_state_ended = False
        for score, period, ended in timeline:
            fair = strategy.on_score(market, score, period, live=not ended,
                                     ended=ended, ts=ts)
            final_state_ended = ended
            if fair is None:
                continue

            for _ in range(cfg.books_per_game):
                ts += cfg.ms_per_book
                # The market drifts toward truth at `catchup_rate`, plus noise.
                market_p += (fair - market_p) * cfg.catchup_rate
                market_p += rng.gauss(0, cfg.price_noise)
                market_p = min(max(market_p, 0.01), 0.99)

                books = {}
                for token in (tok0, tok1):
                    p = market_p if token == tok0 else 1 - market_p
                    bk = _make_book(token, p, cfg, ts)
                    books[token] = bk
                    for f in broker.on_book(bk, ts):
                        portfolio.apply_fill(f, market.market_id)
                        trades += 1

                if ended:
                    continue

                marks = {t: b.mid for t, b in books.items() if b.mid is not None}
                if risk.check_halt(portfolio, marks, "sim"):
                    broker.cancel_all(side=Side.BUY)
                    break

                # Exits.
                for token in (tok0, tok1):
                    pos = portfolio.positions.get(token)
                    if pos is None or pos.shares <= 0:
                        continue
                    should, why = strategy.exit_signal(
                        market, token, pos.avg_cost, books[token], pos.opened_ms, ts
                    )
                    if should and books[token].best_bid:
                        outstanding_sell = broker.live_size(token, Side.SELL)
                        close_size = max(0.0, pos.shares - outstanding_sell)
                        if close_size <= 1e-9:
                            continue
                        broker.submit(Order(
                            token_id=token, side=Side.SELL, size=close_size,
                            limit_price=max(books[token].best_bid - 0.01, 0.01),
                            order_type=OrderType.MARKETABLE, market_id=market.market_id,
                            reason=why,
                        ), ts)

                sig = strategy.evaluate(market, books, ts)
                if sig is None:
                    continue
                signals += 1
                bk = books[sig.token_id]
                depth = bk.depth(Side.BUY, max_price=sig.market_price + 0.02)
                ok, why = risk.approve(
                    sig, market, portfolio, marks, bk.age_ms(ts), bk.spread, depth, ts
                )
                if not ok:
                    key = why.split(" (")[0]
                    rejections[key] = rejections.get(key, 0) + 1
                    continue
                _, fillable = broker.expected_cost(sig.token_id, Side.BUY, depth)
                shares = risk.size_order(sig, market, portfolio, marks, fillable)
                if shares <= 0:
                    continue
                broker.submit(Order(
                    token_id=sig.token_id, side=Side.BUY, size=shares,
                    limit_price=sig.market_price + market.tick_size,
                    order_type=OrderType.MARKETABLE, market_id=market.market_id,
                    reason=sig.reason,
                ), ts)
                risk.record_entry(market.market_id, ts)

            if risk.state.halted:
                break

        # Settle the match.
        st = strategy.trackers.get(market.market_id)
        final_score = timeline[-1][0]
        if cfg.sport == "table_tennis":
            fs = tt.parse_table_tennis_score(final_score, "FT", best_of=cfg.best_of)
            a_won = fs.games_a > fs.games_b
        else:
            fs = tn.parse_tennis_score(final_score, "FT", best_of=cfg.best_of)
            a_won = fs.sets_a > fs.sets_b

        broker.cancel_all(tok0)
        broker.cancel_all(tok1)
        for token, payout in ((tok0, 1.0 if a_won else 0.0), (tok1, 0.0 if a_won else 1.0)):
            portfolio.settle(token, payout)

        # Count wins/losses per match rather than per settlement. Most positions
        # are closed by an exit signal before the match resolves, so counting
        # settlements alone reported 0/0 regardless of performance.
        match_pnl = portfolio.equity({}) - equity_at_match_start
        if match_pnl > 1e-9:
            wins += 1
        elif match_pnl < -1e-9:
            losses += 1

        portfolio.mark({})
        if cfg.verbose and (i + 1) % 25 == 0:
            log.info("match %d/%d  equity=$%.2f", i + 1, cfg.n_matches, portfolio.cash)
        if risk.state.halted:
            log.warning("simulation halted early: %s", risk.state.halt_reason)
            break

    return SimResult(
        starting_cash=cfg.starting_cash,
        final_equity=portfolio.equity({}),
        matches=cfg.n_matches,
        trades=trades,
        signals=signals,
        realized_pnl=portfolio.realized_pnl,
        fees=portfolio.fees_paid,
        max_drawdown=portfolio.max_drawdown,
        wins=wins,
        losses=losses,
        rejections=rejections,
        portfolio=portfolio,
    )
