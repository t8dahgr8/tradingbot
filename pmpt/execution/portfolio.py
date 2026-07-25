"""Bankroll, positions, P&L and the trade journal."""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from ..models import Fill, Position, Side, now_ms

log = logging.getLogger(__name__)


@dataclass
class EquityPoint:
    ts_ms: int
    cash: float
    position_value: float
    equity: float


class Portfolio:
    """Tracks cash, positions and P&L for the paper account.

    Everything is marked at the book mid. That is a deliberately optimistic mark
    -- you cannot actually exit at mid -- so `liquidation_equity()` is provided
    for the honest number.
    """

    def __init__(self, starting_cash: float = 100.0, journal_path: str | None = None):
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.positions: dict[str, Position] = {}
        self.fills: list[Fill] = []
        self.equity_curve: list[EquityPoint] = []
        self.journal_path = journal_path
        self.peak_equity = starting_cash
        self.max_drawdown = 0.0
        self.realized_pnl = 0.0
        self.fees_paid = 0.0
        self.settled: dict[str, float] = {}
        if journal_path:
            self._init_journal()

    # -- accounting --------------------------------------------------------

    def apply_fill(self, fill: Fill, market_id: str = "", label: str = "") -> None:
        pos = self.positions.get(fill.token_id)
        if pos is None:
            pos = Position(token_id=fill.token_id, market_id=market_id, label=label)
            self.positions[fill.token_id] = pos

        realized = pos.apply(fill)
        self.realized_pnl += realized
        self.fees_paid += fill.fee
        self.cash += fill.cash_delta
        self.fills.append(fill)
        self._journal(fill, pos, realized, label)

        if abs(pos.shares) <= 1e-9 and abs(pos.realized_pnl) < 1e-12:
            self.positions.pop(fill.token_id, None)

    def settle(self, token_id: str, payout: float) -> float:
        """Resolve a token at 1.0 or 0.0 and book the result."""
        pos = self.positions.pop(token_id, None)
        if pos is None or abs(pos.shares) <= 1e-9:
            return 0.0
        proceeds = pos.shares * payout
        pnl = (payout - pos.avg_cost) * pos.shares
        self.cash += proceeds
        self.realized_pnl += pnl
        self.settled[token_id] = payout
        log.info(
            "SETTLED %s at %.2f | shares=%.2f avg=%.4f pnl=%+.2f",
            pos.label or token_id[:10], payout, pos.shares, pos.avg_cost, pnl,
        )
        return pnl

    # -- valuation ---------------------------------------------------------

    def position_value(self, marks: dict[str, float]) -> float:
        return sum(
            p.shares * marks.get(t, p.avg_cost) for t, p in self.positions.items()
        )

    def equity(self, marks: dict[str, float]) -> float:
        return self.cash + self.position_value(marks)

    def liquidation_equity(self, bids: dict[str, float]) -> float:
        """What the account is worth if you had to exit everything right now.

        Uses the best bid rather than the mid. On thin in-play books the gap
        between this and `equity()` is often the entire paper profit.
        """
        total = self.cash
        for t, p in self.positions.items():
            total += p.shares * bids.get(t, 0.0)
        return total

    def exposure(self) -> float:
        """Capital at risk: what you lose if every open position resolves to zero."""
        return sum(abs(p.shares * p.avg_cost) for p in self.positions.values())

    def exposure_in_market(self, market_id: str) -> float:
        return sum(
            abs(p.shares * p.avg_cost)
            for p in self.positions.values()
            if p.market_id == market_id
        )

    def mark(self, marks: dict[str, float], ts_ms: int | None = None) -> EquityPoint:
        eq = self.equity(marks)
        pt = EquityPoint(ts_ms or now_ms(), self.cash, self.position_value(marks), eq)
        self.equity_curve.append(pt)
        self.peak_equity = max(self.peak_equity, eq)
        if self.peak_equity > 0:
            dd = (self.peak_equity - eq) / self.peak_equity
            self.max_drawdown = max(self.max_drawdown, dd)
        return pt

    # -- reporting ---------------------------------------------------------

    def unrealized_pnl(self, marks: dict[str, float]) -> float:
        return sum(
            p.unrealized_pnl(marks.get(t, p.avg_cost)) for t, p in self.positions.items()
        )

    def stats(self, marks: dict[str, float] | None = None) -> dict:
        marks = marks or {}
        eq = self.equity(marks)
        closed = [f for f in self.fills]
        wins = sum(1 for p in self.positions.values() if p.realized_pnl > 0)
        losses = sum(1 for p in self.positions.values() if p.realized_pnl < 0)
        return {
            "starting_cash": round(self.starting_cash, 4),
            "cash": round(self.cash, 4),
            "equity": round(eq, 4),
            "total_return_pct": round(100 * (eq / self.starting_cash - 1), 3)
            if self.starting_cash
            else 0.0,
            "realized_pnl": round(self.realized_pnl, 4),
            "unrealized_pnl": round(self.unrealized_pnl(marks), 4),
            "fees_paid": round(self.fees_paid, 4),
            "num_fills": len(closed),
            "open_positions": len(self.positions),
            "exposure": round(self.exposure(), 4),
            "max_drawdown_pct": round(100 * self.max_drawdown, 3),
            "winning_positions": wins,
            "losing_positions": losses,
        }

    # -- persistence -------------------------------------------------------

    def _init_journal(self) -> None:
        d = os.path.dirname(self.journal_path)
        if d:
            os.makedirs(d, exist_ok=True)
        if not os.path.exists(self.journal_path):
            with open(self.journal_path, "w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow([
                    "timestamp", "token_id", "label", "side", "liquidity",
                    "price", "size", "fee", "cash_after", "position_shares",
                    "position_avg_cost", "realized_pnl",
                ])

    def _journal(self, fill: Fill, pos: Position, realized: float, label: str) -> None:
        if not self.journal_path:
            return
        ts = datetime.fromtimestamp(fill.timestamp_ms / 1000, tz=timezone.utc).isoformat()
        with open(self.journal_path, "a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([
                ts, fill.token_id, label or pos.label, fill.side.value, fill.liquidity,
                round(fill.price, 6), round(fill.size, 4), round(fill.fee, 6),
                round(self.cash, 4), round(pos.shares, 4), round(pos.avg_cost, 6),
                round(realized, 6),
            ])

    def save(self, path: str) -> None:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        payload = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "starting_cash": self.starting_cash,
            "cash": self.cash,
            "realized_pnl": self.realized_pnl,
            "fees_paid": self.fees_paid,
            "peak_equity": self.peak_equity,
            "max_drawdown": self.max_drawdown,
            "positions": {t: asdict(p) for t, p in self.positions.items()},
            "settled": self.settled,
            "stats": self.stats(),
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    @classmethod
    def load(cls, path: str, journal_path: str | None = None) -> "Portfolio":
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        p = cls(data.get("starting_cash", 100.0), journal_path)
        p.cash = data.get("cash", p.starting_cash)
        p.realized_pnl = data.get("realized_pnl", 0.0)
        p.fees_paid = data.get("fees_paid", 0.0)
        p.peak_equity = data.get("peak_equity", p.cash)
        p.max_drawdown = data.get("max_drawdown", 0.0)
        p.settled = data.get("settled", {})
        for t, raw in (data.get("positions") or {}).items():
            p.positions[t] = Position(**raw)
        return p
