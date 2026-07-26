"""Local dashboard process control and session archive tests."""

from __future__ import annotations

import asyncio
import csv
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from pmpt.config import AppConfig
from pmpt.control import BotController, validate_starting_cash
from pmpt.engine import TradingEngine


class FakeProcess:
    def __init__(self, pid: int = 4242):
        self.pid = pid
        self.returncode = None

    def poll(self):
        return self.returncode


class TestStartingCash(unittest.TestCase):
    def test_custom_amount_is_rounded_to_cents(self):
        self.assertEqual(validate_starting_cash("250.129"), 250.13)

    def test_invalid_amounts_are_rejected(self):
        for value in (None, True, 0, -1, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_starting_cash(value)


class TestBotController(unittest.TestCase):
    def test_start_passes_custom_cash_without_a_shell(self):
        with tempfile.TemporaryDirectory() as root:
            repo = Path(root)
            state = repo / "state"
            config = repo / "config.yaml"
            (repo / "run.py").write_text("", encoding="utf-8")
            config.write_text("run: {}\n", encoding="utf-8")
            controller = BotController(str(repo), str(state), str(config))
            fake = FakeProcess()

            with patch("pmpt.control.subprocess.Popen", return_value=fake) as popen:
                result = controller.start("725.50")

            self.assertTrue(result["running"])
            self.assertEqual(result["starting_cash"], 725.50)
            command = popen.call_args.args[0]
            self.assertIn("--cash", command)
            self.assertEqual(command[command.index("--cash") + 1], "725.50")
            self.assertIsInstance(command, list)

    def test_stop_archive_uses_local_stop_timestamp_and_copies_audit_files(self):
        with tempfile.TemporaryDirectory() as root:
            repo = Path(root)
            state = repo / "state"
            state.mkdir()
            config = repo / "config.yaml"
            config.write_text("run:\n  starting_cash: 250\n", encoding="utf-8")
            stopped = datetime(2026, 7, 26, 6, 7, 8, tzinfo=timezone.utc)

            self._write_json(state / "session.json", {
                "session_id": "test-session",
                "status": "stopped",
                "mode": "hft",
                "started_at": "2026-07-26T05:07:08+00:00",
                "stopped_at": stopped.isoformat(),
                "starting_cash": 250.0,
            })
            self._write_json(state / "data.json", {
                "online": False,
                "mode": "hft",
                "runtime_minutes": 60,
                "stats": {
                    "starting_cash": 250.0,
                    "cash": 251.25,
                    "equity": 251.25,
                    "num_fills": 2,
                },
                "signals": 900,
                "orders": 20,
                "liquidation_equity": 251.25,
            })
            self._write_json(state / "portfolio.json", {"cash": 251.25})
            with (state / "trades.csv").open("w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerows([
                    ["timestamp", "side", "price"],
                    ["2026-07-26T04:30:00+00:00", "BUY", "0.55"],
                    ["2026-07-26T05:30:00+00:00", "BUY", "0.82"],
                ])
            (state / "trader.log").write_text("filled BUY\n", encoding="utf-8")
            (state / "hft-live.stdout.log").write_text(
                "PAPER TRADING REPORT\n",
                encoding="utf-8",
            )
            (state / "hft-live.stderr.log").write_text("", encoding="utf-8")

            controller = BotController(str(repo), str(state), str(config))
            archive_raw = controller.archive_stopped_session(stopped)

            archive = Path(archive_raw)
            self.assertEqual(
                archive.name,
                stopped.astimezone().strftime("%Y-%m-%d_%H-%M-%S"),
            )
            for name in (
                "summary.json",
                "session.json",
                "data.json",
                "portfolio.json",
                "trades.csv",
                "trader.log",
                "hft-live.stdout.log",
                "hft-live.stderr.log",
                "config.yaml",
            ):
                self.assertTrue((archive / name).is_file(), name)

            summary = json.loads((archive / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["starting_cash"], 250.0)
            self.assertEqual(summary["final_stats"]["num_fills"], 2)
            self.assertEqual(summary["duration_seconds"], 3600.0)
            with (archive / "trades.csv").open(newline="", encoding="utf-8") as fh:
                trades = list(csv.DictReader(fh))
            self.assertEqual(len(trades), 1)
            self.assertEqual(trades[0]["price"], "0.82")

            status = controller.status()
            self.assertFalse(status["running"])
            self.assertEqual(status["last_archive"], os.path.relpath(archive, repo))

    @staticmethod
    def _write_json(path: Path, value: dict):
        path.write_text(json.dumps(value), encoding="utf-8")


class TestGracefulEngineStop(unittest.TestCase):
    def test_stop_request_ends_the_engine_control_loop(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = AppConfig()
            cfg.run.state_dir = root
            engine = TradingEngine(cfg)

            async def scenario():
                task = asyncio.create_task(engine._control_loop())
                Path(root, "stop.request").write_text("{}", encoding="utf-8")
                await asyncio.wait_for(task, timeout=1)

            asyncio.run(scenario())

            self.assertTrue(engine._stop.is_set())
            self.assertFalse(Path(root, "stop.request").exists())


if __name__ == "__main__":
    unittest.main()
