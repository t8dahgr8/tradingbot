"""Dashboard heartbeat and GitHub publishing tests."""

import json
import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from pmpt.config import AppConfig
from pmpt.dashboard import _write_json_atomic, build_snapshot, mark_snapshot_offline
from pmpt.engine import TradingEngine
from pmpt.github_live import GitHubLivePublisher


class TestDashboardStatus(unittest.TestCase):
    def test_hft_snapshot_exposes_mode_and_quote_telemetry(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = AppConfig()
            cfg.run.mode = "hft"
            cfg.run.state_dir = os.path.join(root, "state")
            engine = TradingEngine(cfg)

            snapshot = build_snapshot(engine)

            self.assertEqual(snapshot["mode"], "hft")
            self.assertEqual(snapshot["execution"], "paper")
            self.assertEqual(snapshot["open_orders"], [])
            self.assertEqual(snapshot["hft"]["active_quotes"], 0)
            self.assertIn("quote_cycles_per_min", snapshot["hft"])

    def test_atomic_snapshot_retries_windows_file_contention(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "data.json")
            real_replace = os.replace
            attempts = 0

            def flaky_replace(source, destination):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise PermissionError("destination briefly in use")
                return real_replace(source, destination)

            with patch("pmpt.dashboard.os.replace", side_effect=flaky_replace):
                _write_json_atomic(path, {"online": True})

            self.assertEqual(attempts, 2)
            with open(path, encoding="utf-8") as fh:
                self.assertTrue(json.load(fh)["online"])

    def test_clean_shutdown_marks_paper_snapshot_offline(self):
        with tempfile.TemporaryDirectory() as root:
            state = os.path.join(root, "state")
            docs = os.path.join(root, "docs")
            os.makedirs(state)
            os.makedirs(docs)
            payload = {
                "generated_at": "2026-07-26T01:00:00+00:00",
                "online": True,
                "mode": "paper",
            }
            for directory in (state, docs):
                path = os.path.join(directory, "data.json")
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh)

            mark_snapshot_offline(state, docs)

            for directory in (state, docs):
                with open(os.path.join(directory, "data.json"), encoding="utf-8") as fh:
                    result = json.load(fh)
                self.assertFalse(result["online"])
                self.assertIn("stopped_at", result)

    def test_simulation_snapshot_is_not_rewritten(self):
        with tempfile.TemporaryDirectory() as root:
            payload = {"mode": "simulation", "online": False}
            path = os.path.join(root, "data.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)

            mark_snapshot_offline(root, None)

            with open(path, encoding="utf-8") as fh:
                self.assertEqual(json.load(fh), payload)

    def test_clean_shutdown_marks_hft_snapshot_offline(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "data.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"mode": "hft", "online": True}, fh)

            mark_snapshot_offline(root, None)

            with open(path, encoding="utf-8") as fh:
                self.assertFalse(json.load(fh)["online"])


class TestGitHubLivePublisher(unittest.TestCase):
    def test_publish_replaces_single_commit_live_branch(self):
        with tempfile.TemporaryDirectory() as root:
            repo = os.path.join(root, "repo")
            remote = os.path.join(root, "remote.git")
            state = os.path.join(repo, "state")
            os.makedirs(state)
            self._git(root, "init", "--bare", remote)
            self._git(root, "init", repo)
            self._git(repo, "remote", "add", "origin", remote)

            path = os.path.join(state, "data.json")
            self._write(path, {"online": True, "signals": 1})
            publisher = GitHubLivePublisher(repo, path, interval_s=10)
            self.assertTrue(publisher.publish_once())

            self._write(path, {"online": True, "signals": 2})
            self.assertTrue(publisher.publish_once())

            raw = self._git_bare(remote, "show", "live-data:data.json")
            self.assertEqual(json.loads(raw)["signals"], 2)
            count = self._git_bare(remote, "rev-list", "--count", "live-data")
            self.assertEqual(count.strip(), "1")

    @staticmethod
    def _write(path, payload):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)

    @staticmethod
    def _git(cwd, *args):
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    @staticmethod
    def _git_bare(git_dir, *args):
        return subprocess.run(
            ["git", f"--git-dir={git_dir}", *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
