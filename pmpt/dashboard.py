"""
Dashboard: snapshot writer + a tiny static server.

Two ways to look at the bot:

  * Locally, `python run.py dashboard` serves the page and refreshes from the
    live state directory.
  * On GitHub Pages, the engine writes `docs/data.json` and the same page reads
    it, so a public URL shows the last published snapshot.

The page is a single HTML file with no build step, so both paths use identical
code. Stdlib only.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

log = logging.getLogger(__name__)

DOCS_DIR = "docs"
SNAPSHOT_NAME = "data.json"


def build_snapshot(engine) -> dict:
    """Everything the dashboard needs, in one JSON-serialisable dict."""
    pf = engine.portfolio
    marks = engine._marks()
    bids = engine._bids()
    stats = pf.stats(marks)

    positions = []
    for token, p in pf.positions.items():
        m = engine.gamma.by_token.get(token)
        mark = marks.get(token, p.avg_cost)
        positions.append({
            "token_id": token,
            "label": p.label or (m.outcomes[m.index_of(token)] if m else token[:10]),
            "market": m.question if m else "",
            "shares": round(p.shares, 2),
            "avg_cost": round(p.avg_cost, 4),
            "mark": round(mark, 4),
            "bid": round(bids.get(token, 0.0), 4),
            "unrealized": round(p.unrealized_pnl(mark), 2),
            "cost_basis": round(p.shares * p.avg_cost, 2),
        })
    positions.sort(key=lambda x: -abs(x["cost_basis"]))

    trades = []
    for f in pf.fills[-100:]:
        m = engine.gamma.by_token.get(f.token_id)
        trades.append({
            "time": datetime.fromtimestamp(f.timestamp_ms / 1000, tz=timezone.utc)
            .strftime("%H:%M:%S"),
            "side": f.side.value,
            "label": (m.outcomes[m.index_of(f.token_id)] if m else f.token_id[:10]),
            "price": round(f.price, 4),
            "size": round(f.size, 2),
            "liquidity": f.liquidity,
            "fee": round(f.fee, 4),
        })
    trades.reverse()

    tracked = []
    for t in engine.strategy.trackers.values():
        if not t.live or t.ended:
            continue
        m = t.market
        b0 = engine.broker.book(m.token_ids[0])
        tracked.append({
            "market": m.question,
            "outcomes": list(m.outcomes),
            "score": t.last_score,
            "period": t.last_period,
            "model": round(t.fair_value, 4) if t.fair_value is not None else None,
            "market_price": round(b0.mid, 4) if (b0 and b0.mid is not None) else None,
            "anchor": round(t.anchor_prob, 4) if t.anchor_prob is not None else None,
            "tradeable": t.anchored_cleanly,
        })

    equity = [
        {"t": p.ts_ms, "equity": round(p.equity, 4), "cash": round(p.cash, 4)}
        for p in pf.equity_curve[-600:]
    ]

    markets_by_sport: dict[str, int] = {}
    market_game_ids: set[int] = set()
    for market in engine.gamma.markets.values():
        markets_by_sport[market.sport] = markets_by_sport.get(market.sport, 0) + 1
        if market.game_id is not None:
            market_game_ids.add(int(market.game_id))

    score_games = engine.sports_feed.games if engine.sports_feed else {}
    live_score_ids = {
        int(game_id)
        for game_id, game in score_games.items()
        if bool(game.get("live")) and not bool(game.get("ended"))
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "online": True,
        "mode": "paper",
        "stats": stats,
        "liquidation_equity": round(pf.liquidation_equity(bids), 2),
        "runtime_minutes": round((engine.started_ms and
                                  (json_now_ms() - engine.started_ms) / 60000) or 0, 1),
        "signals": engine.signals_seen,
        "orders": engine.orders_sent,
        "halted": engine.risk.state.halted,
        "halt_reason": engine.risk.state.halt_reason,
        "feeds": {
            "market": {
                "connected": bool(engine.market_feed and engine.market_feed.connected),
                "reconnects": engine.market_feed.reconnects if engine.market_feed else 0,
                "stale_ms": engine.market_feed.stale_ms if engine.market_feed else None,
            },
            "sports": {
                "connected": bool(engine.sports_feed and engine.sports_feed.connected),
                "reconnects": engine.sports_feed.reconnects if engine.sports_feed else 0,
                "stale_ms": engine.sports_feed.stale_ms if engine.sports_feed else None,
            },
        },
        "universe": {
            "markets": len(engine.gamma.markets),
            "by_sport": markets_by_sport,
            "score_games_seen": len(score_games),
            "live_score_games": len(live_score_ids),
            "matched_live_games": len(market_game_ids & live_score_ids),
        },
        "positions": positions,
        "trades": trades,
        "tracked": tracked,
        "rejections": dict(sorted(engine.rejections.items(), key=lambda x: -x[1])[:10]),
        "equity_curve": equity,
    }


def json_now_ms() -> int:
    import time
    return int(time.time() * 1000)


def write_snapshot(engine, state_dir: str, publish_dir: str | None = DOCS_DIR) -> None:
    """Write the snapshot into the state dir and, optionally, into docs/ too."""
    try:
        snap = build_snapshot(engine)
    except Exception as e:  # noqa: BLE001 - the dashboard must never kill the trader
        log.warning("dashboard snapshot failed: %s", e)
        return

    try:
        os.makedirs(state_dir, exist_ok=True)
        _write_json_atomic(os.path.join(state_dir, SNAPSHOT_NAME), snap)
        if publish_dir and os.path.isdir(publish_dir):
            _write_json_atomic(os.path.join(publish_dir, SNAPSHOT_NAME), snap)
    except OSError as exc:
        # On Windows a reader can briefly block os.replace. Missing one
        # heartbeat is preferable to killing the trading housekeeping task.
        log.warning("dashboard snapshot write failed; will retry next mark: %s", exc)


def mark_snapshot_offline(
    state_dir: str,
    publish_dir: str | None = DOCS_DIR,
) -> None:
    """Mark the last paper snapshot offline after a clean shutdown."""
    stopped_at = datetime.now(timezone.utc).isoformat()
    for directory in (state_dir, publish_dir):
        if not directory:
            continue
        path = os.path.join(directory, SNAPSHOT_NAME)
        try:
            with open(path, encoding="utf-8") as fh:
                snap = json.load(fh)
            if snap.get("mode") != "paper":
                continue
            snap["online"] = False
            snap["stopped_at"] = stopped_at
            _write_json_atomic(path, snap)
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("could not mark dashboard offline: %s", exc)


def _write_json_atomic(path: str, payload: dict) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
        dir=directory,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        for attempt in range(6):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(0.02 * (attempt + 1))
    finally:
        with contextlib.suppress(OSError):
            os.remove(tmp)


def snapshot_from_simulation(result, portfolio) -> dict:
    """Render a finished simulation into the same shape the dashboard expects.

    Lets you look at simulated runs in the same UI as live paper trading, which
    is the point -- if the two look wildly different, that is worth knowing.
    """
    stats = portfolio.stats({})
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "online": False,
        "mode": "simulation",
        "stats": stats,
        "liquidation_equity": round(portfolio.cash, 2),
        "runtime_minutes": 0,
        "signals": result.signals,
        "orders": result.trades,
        "halted": False,
        "halt_reason": "",
        "feeds": {"market": {"connected": False, "reconnects": 0},
                  "sports": {"connected": False, "reconnects": 0}},
        "positions": [],
        "trades": [
            {
                "time": f"#{i}",
                "side": f.side.value,
                "label": f.token_id,
                "price": round(f.price, 4),
                "size": round(f.size, 2),
                "liquidity": f.liquidity,
                "fee": round(f.fee, 4),
            }
            for i, f in enumerate(portfolio.fills[-60:])
        ][::-1],
        "tracked": [],
        "rejections": dict(sorted(result.rejections.items(), key=lambda x: -x[1])[:10]),
        "equity_curve": [
            {"t": p.ts_ms, "equity": round(p.equity, 4), "cash": round(p.cash, 4)}
            for p in portfolio.equity_curve[-600:]
        ],
    }


def serve(state_dir: str = "state", port: int = 8000, docs_dir: str = DOCS_DIR) -> None:
    """Serve the dashboard. `/` is the page, `/data.json` is the live snapshot."""
    snapshot = os.path.abspath(os.path.join(state_dir, SNAPSHOT_NAME))
    index = os.path.abspath(os.path.join(docs_dir, "index.html"))

    if not os.path.exists(index):
        raise FileNotFoundError(f"dashboard page missing at {index}")

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path.split("?")[0] in ("/", "/index.html"):
                return self._send_file(index, "text/html; charset=utf-8")
            if self.path.split("?")[0] == "/data.json":
                if os.path.exists(snapshot):
                    return self._send_file(snapshot, "application/json")
                return self._send_json({"error": "no snapshot yet", "stats": {}})
            self.send_error(404)
            return None

        def _send_file(self, path: str, ctype: str):
            try:
                with open(path, "rb") as fh:
                    body = fh.read()
            except OSError:
                self.send_error(404)
                return None
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return None

        def _send_json(self, obj: dict):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return None

        def log_message(self, *args):  # keep the trading logs readable
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"\n  Dashboard: http://127.0.0.1:{port}")
    print(f"  Reading   {snapshot}")
    print("  Ctrl-C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
