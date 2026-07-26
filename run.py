#!/usr/bin/env python3
"""
Polymarket paper trader -- CLI.

    python run.py simulate                  offline simulation, no network needed
    python run.py simulate --sweep          sweep market speed to see where edge dies
    python run.py live                      paper trade real markets in real time
    python run.py dashboard                 web UI at http://127.0.0.1:8000
    python run.py markets                   list tradeable tennis markets right now
    python run.py report                    print the saved portfolio

Nothing here can place a real order. There is no wallet, no private key, and no
signing code anywhere in this project -- that is deliberate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pmpt.config import AppConfig, load_config, setup_logging  # noqa: E402
from pmpt.execution.risk import RiskConfig  # noqa: E402
from pmpt.strategy.live_model import StrategyConfig  # noqa: E402


def cmd_simulate(args: argparse.Namespace) -> int:
    from pmpt.simulate import SimConfig, run_simulation

    cfg = load_config(args.config)
    setup_logging(args.log_level or cfg.run.log_level)

    base = SimConfig(
        n_matches=args.matches,
        sport=args.sport,
        best_of=args.best_of,
        starting_cash=args.cash,
        catchup_rate=args.catchup,
        seed=args.seed,
        verbose=args.verbose,
    )

    if not args.sweep:
        res = run_simulation(base, cfg.strategy, cfg.risk, cfg.broker)
        print(res.render())
        if res.rejections:
            print("\n  Signal rejections:")
            for k, v in sorted(res.rejections.items(), key=lambda x: -x[1])[:10]:
                print(f"    {v:>6d}  {k}")

        # Publish so `python run.py dashboard` can show the run.
        import json as _json

        from pmpt.dashboard import DOCS_DIR, SNAPSHOT_NAME, snapshot_from_simulation

        snap = snapshot_from_simulation(res, res.portfolio)
        os.makedirs(cfg.run.state_dir, exist_ok=True)
        for d in (cfg.run.state_dir, DOCS_DIR):
            if os.path.isdir(d):
                with open(os.path.join(d, SNAPSHOT_NAME), "w", encoding="utf-8") as fh:
                    _json.dump(snap, fh, indent=2)
        print(f"\n  Dashboard data written. View it with:  python run.py dashboard\n")
        return 0

    # The sweep is the important experiment. It shows the edge collapsing as the
    # market gets faster. If the last row is still profitable, be suspicious.
    import logging
    logging.getLogger("pmpt").setLevel(logging.WARNING)

    print("\n  Market speed sweep -- how fast does the book have to be to kill the edge?")
    print("  catchup_rate 1.0 means the book reprices instantly (no lag to exploit).")
    print("  The bottom row SHOULD be roughly zero. If it is not, suspect a bug.\n")
    print(f"  {'catchup':>8} {'return %':>10} {'trades':>8} {'signals':>9} {'max dd %':>9}")
    print("  " + "-" * 48)
    for rate in (0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00):
        c = SimConfig(**{**base.__dict__, "catchup_rate": rate})
        r = run_simulation(c, cfg.strategy, cfg.risk, cfg.broker)
        print(f"  {rate:>8.2f} {r.total_return_pct:>10.2f} {r.trades:>8d} "
              f"{r.signals:>9d} {100*r.max_drawdown:>9.2f}")
    print()
    return 0


def cmd_markets(args: argparse.Namespace) -> int:
    from pmpt.data.gamma import GammaClient

    cfg = load_config(args.config)
    setup_logging(args.log_level or cfg.run.log_level)

    gc = GammaClient(sports=cfg.run.sports)
    markets = gc.refresh(only_live=args.live_only)
    if not markets:
        print("No tradeable markets found. If you expected some, check connectivity "
              "and whether any matches are actually in progress.")
        return 1

    print(f"\n  {len(markets)} tradeable market(s)\n")
    for m in markets[:60]:
        ev = gc.event_for_market(m) or {}
        state = "LIVE" if ev.get("live") else ("ended" if ev.get("ended") else "pre")
        score = ev.get("score") or "-"
        print(f"  [{state:>5}] {m.outcomes[0]} vs {m.outcomes[1]}")
        print(f"          {m.question[:70]}")
        print(f"          score={score!r:<24} bo{m.best_of}  tick={m.tick_size} "
              f"min={m.min_order_size:g}")
    print()
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    path = os.path.join(cfg.run.state_dir, "portfolio.json")
    if not os.path.exists(path):
        print(f"No saved portfolio at {path}. Run `live` or `simulate` first.")
        return 1
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    print(json.dumps(data.get("stats", data), indent=2))
    return 0


def cmd_live(args: argparse.Namespace) -> int:
    from pmpt.dashboard import SNAPSHOT_NAME, mark_snapshot_offline, write_snapshot
    from pmpt.engine import TradingEngine
    from pmpt.github_live import GitHubLivePublisher

    cfg = load_config(args.config)
    if args.cash:
        cfg.run.starting_cash = args.cash
    if args.minutes:
        cfg.run.max_runtime_s = int(args.minutes * 60)
    if args.sport:
        cfg.run.sports = [args.sport]
    if args.mode:
        cfg.run.mode = args.mode
    if args.publish_github is not None:
        cfg.run.github_live_enabled = args.publish_github
    setup_logging(args.log_level or cfg.run.log_level, log_dir=cfg.run.state_dir)

    print(f"\n  Paper trading with ${cfg.run.starting_cash:.2f} of imaginary money.")
    print("  No real orders can be placed. Press Ctrl-C to stop and print a report.\n")

    engine = TradingEngine(cfg)
    write_snapshot(engine, cfg.run.state_dir)
    publisher = None
    if cfg.run.github_live_enabled:
        publisher = GitHubLivePublisher(
            repo_dir=os.path.dirname(os.path.abspath(__file__)),
            snapshot_path=os.path.join(cfg.run.state_dir, SNAPSHOT_NAME),
            interval_s=cfg.run.github_live_interval_s,
            remote=cfg.run.github_live_remote,
            branch=cfg.run.github_live_branch,
        )
        if not publisher.start():
            publisher = None

    async def main() -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, engine.stop)
            except NotImplementedError:
                pass  # Windows
        await engine.run()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    finally:
        mark_snapshot_offline(cfg.run.state_dir)
        if publisher is not None:
            publisher.stop(publish_final=True)
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    from pmpt.dashboard import serve

    cfg = load_config(args.config)
    setup_logging(args.log_level or "WARNING")
    serve(state_dir=cfg.run.state_dir, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run.py", description="Polymarket paper trader (tennis / table tennis)"
    )
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--log-level", default=None)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("simulate", help="offline simulation (no network)")
    s.add_argument("--matches", type=int, default=200)
    s.add_argument("--sport", default="tennis", choices=["tennis", "table_tennis"])
    s.add_argument("--best-of", type=int, default=3)
    s.add_argument("--cash", type=float, default=100.0)
    s.add_argument("--catchup", type=float, default=0.25,
                   help="how fast the synthetic market corrects (1.0 = instant)")
    s.add_argument("--seed", type=int, default=20260725)
    s.add_argument("--sweep", action="store_true", help="sweep market speed")
    s.add_argument("--verbose", action="store_true")
    s.set_defaults(func=cmd_simulate)

    m = sub.add_parser("markets", help="list tradeable markets")
    m.add_argument("--live-only", action="store_true", default=False)
    m.set_defaults(func=cmd_markets)

    r = sub.add_parser("report", help="print saved portfolio stats")
    r.set_defaults(func=cmd_report)

    lv = sub.add_parser("live", help="paper trade live markets")
    lv.add_argument("--cash", type=float, default=None)
    lv.add_argument("--minutes", type=float, default=None, help="stop after N minutes")
    lv.add_argument("--sport", default=None, choices=["tennis", "table_tennis"])
    lv.add_argument("--mode", default=None, choices=["scalp", "hft"])
    lv.add_argument(
        "--publish-github",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="mirror live dashboard data to GitHub (default: config setting)",
    )
    lv.set_defaults(func=cmd_live)

    db = sub.add_parser("dashboard", help="serve the web dashboard")
    db.add_argument("--port", type=int, default=8000)
    db.set_defaults(func=cmd_dashboard)

    return p


if __name__ == "__main__":
    ns = build_parser().parse_args()
    raise SystemExit(ns.func(ns))
