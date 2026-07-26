"""
The strategy: in-play model repricing.

The claim being traded is deliberately narrow, and it is worth stating plainly
because it determines whether any of this can work:

    We are NOT claiming to know the players better than the market.
    We take the market's own pre-match price as the truth about relative strength,
    convert it into serve-point probabilities, and then use the score to compute
    what the price *should* be. The bet is that the order book is slower to
    reprice a score than the arithmetic is.

That edge is real but small and decays fast, which is why:

  * A signal is only valid for a short window after the score changes
    (`signal_ttl_ms`). Outside that window the market has almost certainly caught
    up and any apparent edge is really a stale book or a bad anchor.

  * The anchor decays toward the current market price. If the market persistently
    disagrees with us, it usually knows something we do not -- an injury, a
    retirement, a medical timeout. Fighting that with a Markov chain is how you
    lose money confidently.

  * We refuse to trade a match we did not see from the start, because without a
    clean pre-match anchor the calibration is guesswork.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from ..models import OrderBook, Side, Signal, TradableMarket, now_ms
from ..quant import table_tennis as tt
from ..quant import tennis as tn

log = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    # How long after a score change a signal stays actionable.
    signal_ttl_ms: int = 45_000
    # Half-life for pulling the anchor toward the market's current view.
    reanchor_halflife_s: float = 900.0
    # Require the pre-match anchor to be captured before this many games are played.
    max_games_at_anchor: int = 2
    # Refresh a pregame anchor when the winner price moves this much. This lets
    # injury, withdrawal and lineup news already reflected by the market flow
    # into the model without chasing headlines from unreliable sources.
    pregame_reanchor_threshold: float = 0.01
    # Ignore markets whose pre-match price is this extreme: no room to be right.
    anchor_min_price: float = 0.08
    anchor_max_price: float = 0.92
    # Exit rules.
    take_profit_edge: float = 0.005  # close once the gap has converged to this
    quick_take_profit: float = 0.015  # bank a bid-side scalp when it is available
    scratch_profit: float = 0.005    # after a short hold, take even tiny profits
    scratch_profit_after_ms: int = 15_000
    max_hold_ms: int = 120_000       # stale in-play edges should not become bets
    exit_edge_buffer: float = 0.005  # close when the model no longer justifies risk
    stop_loss: float = 0.06          # close if price moves this far against us
    hold_to_resolution: bool = False
    # Surface assumption for serve calibration when we cannot tell.
    surface: str = "unknown"
    # Blend weight on the model vs the market when forming fair value.
    # 1.0 = pure model. Below 1.0 shrinks toward the market, which is the honest
    # default given the model ignores fatigue, injuries and momentum.
    model_weight: float = 0.75


@dataclass
class MatchTracker:
    """Per-market state: the anchor, the last score, and derived serve probs."""

    market: TradableMarket
    anchor_prob: float | None = None       # P(outcome 0 wins), pre-match
    anchor_ms: int = 0
    pa: float = 0.63
    pb: float = 0.63
    last_score: str = ""
    last_period: str = ""
    last_score_change_ms: int = 0
    fair_value: float | None = None        # P(outcome 0 wins), current
    live: bool = False
    ended: bool = False
    anchored_cleanly: bool = False
    updates: int = 0

    def score_age_ms(self, ts: int | None = None) -> int:
        if not self.last_score_change_ms:
            return 10**9
        return (ts or now_ms()) - self.last_score_change_ms


class LiveModelStrategy:
    """Turns (market price, live score) into signals."""

    def __init__(self, config: StrategyConfig | None = None):
        self.cfg = config or StrategyConfig()
        self.trackers: dict[str, MatchTracker] = {}

    # -- tracking ----------------------------------------------------------

    def tracker(self, market: TradableMarket) -> MatchTracker:
        t = self.trackers.get(market.market_id)
        if t is None:
            t = MatchTracker(market=market)
            self.trackers[market.market_id] = t
        return t

    def set_anchor(self, market: TradableMarket, implied_prob: float, ts: int | None = None,
                   games_played: int = 0) -> bool:
        """Capture the pre-match anchor and calibrate serve probabilities.

        Returns False when the anchor is unusable, in which case the market
        should be skipped entirely rather than traded on a guess.
        """
        cfg = self.cfg
        t = self.tracker(market)
        if not (cfg.anchor_min_price <= implied_prob <= cfg.anchor_max_price):
            log.debug("anchor rejected for %s: price %.3f too extreme",
                      market.slug, implied_prob)
            return False

        t.anchor_prob = implied_prob
        t.anchor_ms = ts or now_ms()
        t.anchored_cleanly = games_played <= cfg.max_games_at_anchor

        if market.sport == "table_tennis":
            t.pa, t.pb = tt.calibrate_serve_probs(implied_prob, best_of=market.best_of or 5)
        else:
            t.pa, t.pb = tn.calibrate_serve_probs(
                implied_prob, best_of=market.best_of or 3, surface=cfg.surface  # type: ignore[arg-type]
            )
        log.info(
            "anchored %s at %.3f -> serve probs (%.4f, %.4f)%s",
            market.slug or market.market_id, implied_prob, t.pa, t.pb,
            "" if t.anchored_cleanly else "  [LATE ANCHOR - will not trade]",
        )
        return True

    def decay_anchor(self, market: TradableMarket, market_prob: float,
                     ts: int | None = None) -> None:
        """Pull the anchor toward the market. Prevents fighting real news forever."""
        t = self.trackers.get(market.market_id)
        if t is None or t.anchor_prob is None:
            return
        hl = self.cfg.reanchor_halflife_s
        if hl <= 0:
            return
        dt = max(0.0, ((ts or now_ms()) - t.anchor_ms) / 1000.0)
        if dt <= 0:
            return
        w = 1.0 - math.pow(0.5, dt / hl)
        # Re-derive serve probabilities against the blended anchor. Skipping
        # negligible moves matters: recalibration is the most expensive thing
        # this strategy does, and a 0.2% anchor shift changes nothing we act on.
        blended = (1 - w) * t.anchor_prob + w * market_prob
        if abs(blended - t.anchor_prob) < 2e-3:
            return
        t.anchor_prob = blended
        t.anchor_ms = ts or now_ms()
        if market.sport == "table_tennis":
            t.pa, t.pb = tt.calibrate_serve_probs(blended, best_of=market.best_of or 5)
        else:
            t.pa, t.pb = tn.calibrate_serve_probs(
                blended, best_of=market.best_of or 3, surface=self.cfg.surface  # type: ignore[arg-type]
            )

    def on_score(self, market: TradableMarket, score: str, period: str,
                 live: bool, ended: bool, ts: int | None = None) -> float | None:
        """Update the model with a new score. Returns fair value for outcome 0."""
        ts = ts or now_ms()
        t = self.tracker(market)
        t.live, t.ended = live, ended

        changed = score != t.last_score or period != t.last_period
        t.last_score, t.last_period = score, period
        if changed:
            t.last_score_change_ms = ts
            t.updates += 1

        if t.anchor_prob is None:
            return None

        if market.sport == "table_tennis":
            st = tt.parse_table_tennis_score(score, period, best_of=market.best_of or 5)
            fv = tt.match_win_prob(st, t.pa, t.pb)
        else:
            st = tn.parse_tennis_score(score, period, best_of=market.best_of or 3)
            fv = tn.match_win_prob(st, t.pa, t.pb)
            t.ended = t.ended or st.finished

        t.fair_value = fv
        return fv

    # -- signal generation -------------------------------------------------

    def evaluate(
        self,
        market: TradableMarket,
        books: dict[str, OrderBook],
        ts: int | None = None,
    ) -> Signal | None:
        """Produce a signal if the book has not caught up with the score.

        We only ever BUY. Selling a token you do not hold means buying the other
        side, which the engine handles by choosing the cheaper leg -- there is no
        shorting on Polymarket.
        """
        ts = ts or now_ms()
        cfg = self.cfg
        t = self.trackers.get(market.market_id)
        if t is None or t.fair_value is None or t.anchor_prob is None:
            return None
        if t.ended or not t.live:
            return None
        if not t.anchored_cleanly:
            return None
        if t.score_age_ms(ts) > cfg.signal_ttl_ms:
            return None  # the market has had time to reprice; no edge left

        tok0, tok1 = market.token_ids
        b0, b1 = books.get(tok0), books.get(tok1)
        if b0 is None or not b0.is_valid():
            return None

        # Market's implied probability for outcome 0, from the mid.
        mkt_p0 = b0.mid
        if mkt_p0 is None:
            return None
        self.decay_anchor(market, mkt_p0, ts)

        # Shrink the model toward the market. The model does not know about
        # injuries, cramping, or a player arguing with the umpire.
        w = cfg.model_weight
        fv0 = w * t.fair_value + (1 - w) * mkt_p0
        fv1 = 1.0 - fv0

        # Evaluate buying each leg at its actual ask.
        best: Signal | None = None
        for token, book, fv in ((tok0, b0, fv0), (tok1, b1, fv1)):
            if book is None or not book.is_valid():
                continue
            ask = book.best_ask
            if ask is None:
                continue
            edge = fv - ask
            if edge <= 0:
                continue
            conf = self._confidence(t, book, ts)
            s = Signal(
                token_id=token,
                market_id=market.market_id,
                fair_value=fv,
                market_price=ask,
                edge=edge,
                side=Side.BUY,
                confidence=conf,
                reason=(
                    f"model {fv:.3f} vs ask {ask:.3f} "
                    f"(score {t.last_score!r}, {t.score_age_ms(ts)}ms old)"
                ),
                metadata={
                    "anchor": t.anchor_prob,
                    "raw_model": t.fair_value,
                    "market_mid": mkt_p0,
                    "score": t.last_score,
                    "pa": t.pa,
                    "pb": t.pb,
                },
            )
            if best is None or s.edge * s.confidence > best.edge * best.confidence:
                best = s
        return best

    def _confidence(self, t: MatchTracker, book: OrderBook, ts: int) -> float:
        """Scale conviction down as the signal ages and as the book gets worse."""
        cfg = self.cfg
        age = t.score_age_ms(ts)
        recency = max(0.0, 1.0 - age / max(cfg.signal_ttl_ms, 1))
        spread = book.spread or 1.0
        tightness = max(0.0, 1.0 - spread / 0.10)
        return max(0.0, min(1.0, 0.5 * recency + 0.5 * tightness))

    # -- exits -------------------------------------------------------------

    def exit_signal(
        self,
        market: TradableMarket,
        token_id: str,
        avg_cost: float,
        book: OrderBook,
        opened_ms: int | None = None,
        ts: int | None = None,
        entry_fee_per_share: float = 0.0,
    ) -> tuple[bool, str]:
        """Should we close this position now?"""
        ts = ts or now_ms()
        cfg = self.cfg
        t = self.trackers.get(market.market_id)
        if book is None or not book.is_valid():
            return False, ""

        bid = book.best_bid
        if bid is None:
            return False, ""
        held_ms = max(0, ts - opened_ms) if opened_ms else 0
        fee_rate = market.fee_rate if market.fees_enabled else 0.0
        exit_fee_per_share = fee_rate * bid * (1.0 - bid)
        net_profit_per_share = (
            bid - exit_fee_per_share - avg_cost - max(0.0, entry_fee_per_share)
        )

        if t is not None and t.ended:
            return (False, "") if cfg.hold_to_resolution else (True, "match ended")

        if t is None or t.fair_value is None:
            return False, ""

        idx = market.index_of(token_id)
        fv = t.fair_value if idx == 0 else 1.0 - t.fair_value

        # Stop loss is checked FIRST, deliberately.
        #
        # It used to run after the convergence check, which made it effectively
        # unreachable: convergence fires whenever the bid is at or above fair
        # value, and a position deep underwater is usually also one where the
        # market is quoting above our (now collapsed) fair value. The exit still
        # happened, but it was reported as "converged" -- so a hard risk control
        # was silently being handled by a profit-taking rule. A stop should never
        # be pre-empted by anything.
        if bid <= avg_cost - cfg.stop_loss:
            return True, f"stop loss (bid {bid:.3f} vs cost {avg_cost:.3f})"

        # Profit targets are after both the paid entry fee and the expected
        # taker fee on this exit. Gross one-tick gains can otherwise be losses.
        if net_profit_per_share >= cfg.quick_take_profit:
            return True, (
                f"bank profit net ({net_profit_per_share:.4f}/share after fees)"
            )

        if (
            held_ms >= cfg.scratch_profit_after_ms
            and net_profit_per_share >= cfg.scratch_profit
        ):
            return True, (
                f"scratch profit net ({net_profit_per_share:.4f}/share after fees)"
            )

        if held_ms >= cfg.max_hold_ms and net_profit_per_share >= 0:
            return True, f"time exit ({net_profit_per_share:.4f}/share after fees)"

        # If the repriced model no longer supports the original entry, do not
        # wait for a full stop unless the book is already too far against us.
        if fv <= avg_cost + cfg.exit_edge_buffer and bid >= avg_cost - cfg.stop_loss * 0.5:
            return True, f"edge gone (fair {fv:.3f} vs cost {avg_cost:.3f})"

        # Converged: the market agrees with us now, so the trade is done.
        if bid >= fv - cfg.take_profit_edge:
            return True, f"converged (bid {bid:.3f} vs fair {fv:.3f})"

        return False, ""
