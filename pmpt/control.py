"""Local paper-trader process control and session archiving."""

from __future__ import annotations

import contextlib
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


SESSION_NAME = "session.json"
PID_NAME = "live.pid"
STOP_REQUEST_NAME = "stop.request"
LAST_ARCHIVE_NAME = "last_archive.json"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as fh:
            value = json.load(fh)
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    from .dashboard import _write_json_atomic

    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(str(path), payload)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def validate_starting_cash(value: Any) -> float:
    """Return a finite paper bankroll within a practical dashboard range."""
    if isinstance(value, bool):
        raise ValueError("Starting cash must be a number.")
    try:
        cash = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Starting cash must be a number.") from exc
    if not math.isfinite(cash):
        raise ValueError("Starting cash must be finite.")
    if cash < 1.0:
        raise ValueError("Starting cash must be at least $1.00.")
    if cash > 1_000_000_000:
        raise ValueError("Starting cash must be $1 billion or less.")
    return round(cash, 2)


def clear_stop_request(state_dir: str) -> None:
    with contextlib.suppress(FileNotFoundError):
        os.remove(os.path.join(state_dir, STOP_REQUEST_NAME))


def register_live_session(config: Any) -> None:
    """Register a live process so a restarted dashboard can still control it."""
    state = Path(config.run.state_dir).resolve()
    state.mkdir(parents=True, exist_ok=True)
    session_path = state / SESSION_NAME
    session = _read_json(session_path)
    started_at = session.get("started_at") or _iso(_utc_now())
    session.update({
        "session_id": session.get("session_id") or datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S.%fZ"
        ),
        "status": "running",
        "execution": "paper",
        "mode": str(config.run.mode),
        "pid": os.getpid(),
        "started_at": started_at,
        "starting_cash": float(config.run.starting_cash),
        "sports": list(config.run.sports),
        "config": config.to_dict(),
        "archive_path": None,
    })
    _write_json(session_path, session)
    (state / PID_NAME).write_text(str(os.getpid()), encoding="ascii")


def finish_live_session(state_dir: str, reason: str = "stopped") -> None:
    """Mark process metadata stopped; the dashboard archives it after exit."""
    path = Path(state_dir).resolve() / SESSION_NAME
    session = _read_json(path)
    if not session:
        return
    session.update({
        "status": "stopped",
        "stop_reason": reason,
        "stopped_at": _iso(_utc_now()),
    })
    _write_json(path, session)


class BotController:
    """Starts and stops one local live-paper process at a time."""

    def __init__(
        self,
        repo_dir: str,
        state_dir: str,
        config_path: str,
        sessions_dir: str | None = None,
    ):
        self.repo_dir = Path(repo_dir).resolve()
        self.state_dir = Path(state_dir).resolve()
        self.config_path = Path(config_path).resolve()
        self.sessions_dir = (
            Path(sessions_dir).resolve()
            if sessions_dir
            else self.repo_dir / "sessions"
        )
        self.session_path = self.state_dir / SESSION_NAME
        self.pid_path = self.state_dir / PID_NAME
        self.stop_path = self.state_dir / STOP_REQUEST_NAME
        self.last_archive_path = self.state_dir / LAST_ARCHIVE_NAME
        self._lock = threading.RLock()
        self._process: subprocess.Popen | None = None
        self._monitor_stop = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        self.state_dir.mkdir(parents=True, exist_ok=True)

    # -- public API -----------------------------------------------------

    def start_monitor(self) -> None:
        with self._lock:
            if self._monitor_thread and self._monitor_thread.is_alive():
                return
            self._monitor_stop.clear()
            self._monitor_thread = threading.Thread(
                target=self._monitor,
                name="paper-trader-monitor",
                daemon=True,
            )
            self._monitor_thread.start()

    def close(self) -> None:
        self._monitor_stop.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)

    def status(self) -> dict[str, Any]:
        with self._lock:
            self._reconcile_locked()
            pid = self._running_pid_locked()
            session = _read_json(self.session_path)
            snapshot = _read_json(self.state_dir / "data.json")
            last_archive = _read_json(self.last_archive_path)
            running = pid is not None
            state = "running" if running else "stopped"
            if running and session.get("status") == "stopping":
                state = "stopping"
            return {
                "control_available": True,
                "running": running,
                "state": state,
                "pid": pid,
                "starting_cash": session.get(
                    "starting_cash",
                    (snapshot.get("stats") or {}).get("starting_cash", 100.0),
                ),
                "started_at": session.get("started_at"),
                "stop_requested_at": session.get("stop_requested_at"),
                "last_archive": last_archive.get("relative_path"),
                "last_archive_path": last_archive.get("path"),
                "archive_root": str(self.sessions_dir),
            }

    def start(self, starting_cash: Any) -> dict[str, Any]:
        cash = validate_starting_cash(starting_cash)
        with self._lock:
            self._reconcile_locked()
            if self._running_pid_locked() is not None:
                raise RuntimeError("The paper trader is already running.")
            self._archive_stopped_session_locked()
            self._prepare_fresh_state_locked()

            started = _utc_now()
            session = {
                "session_id": started.strftime("%Y%m%dT%H%M%S.%fZ"),
                "status": "starting",
                "execution": "paper",
                "mode": "hft",
                "started_at": _iso(started),
                "starting_cash": cash,
                "archive_path": None,
            }
            _write_json(self.session_path, session)

            command = [
                sys.executable,
                "-u",
                str(self.repo_dir / "run.py"),
                "--config",
                str(self.config_path),
                "live",
                "--mode",
                "hft",
                "--cash",
                f"{cash:.2f}",
                "--publish-github",
            ]
            stdout_path = self.state_dir / "hft-live.stdout.log"
            stderr_path = self.state_dir / "hft-live.stderr.log"
            creationflags = 0
            if os.name == "nt":
                creationflags = (
                    getattr(subprocess, "CREATE_NO_WINDOW", 0)
                    | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                )

            stdout = stdout_path.open("wb")
            stderr = stderr_path.open("wb")
            try:
                self._process = subprocess.Popen(
                    command,
                    cwd=str(self.repo_dir),
                    stdout=stdout,
                    stderr=stderr,
                    stdin=subprocess.DEVNULL,
                    close_fds=True,
                    creationflags=creationflags,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                )
            except Exception:
                session.update({
                    "status": "start_failed",
                    "stopped_at": _iso(_utc_now()),
                })
                _write_json(self.session_path, session)
                raise
            finally:
                stdout.close()
                stderr.close()

            session.update({
                "status": "running",
                "pid": self._process.pid,
                "command": command,
            })
            _write_json(self.session_path, session)
            self.pid_path.write_text(str(self._process.pid), encoding="ascii")
        return self.status()

    def stop(self, timeout_s: float = 20.0) -> dict[str, Any]:
        with self._lock:
            self._reconcile_locked()
            pid = self._running_pid_locked()
            if pid is None:
                return self.status()
            requested = _utc_now()
            _write_json(self.stop_path, {
                "pid": pid,
                "requested_at": _iso(requested),
            })
            session = _read_json(self.session_path)
            session.update({
                "status": "stopping",
                "stop_requested_at": _iso(requested),
            })
            _write_json(self.session_path, session)

        deadline = time.monotonic() + max(0.0, timeout_s)
        while time.monotonic() < deadline:
            if not self._pid_alive(pid):
                break
            time.sleep(0.1)

        with self._lock:
            self._reconcile_locked()
        return self.status()

    def archive_stopped_session(
        self,
        stopped_at: datetime | None = None,
    ) -> str | None:
        with self._lock:
            if self._running_pid_locked() is not None:
                raise RuntimeError("Cannot archive while the paper trader is running.")
            return self._archive_stopped_session_locked(stopped_at)

    # -- process state --------------------------------------------------

    def _monitor(self) -> None:
        while not self._monitor_stop.wait(1.0):
            with contextlib.suppress(Exception):
                self.status()

    def _read_pid(self) -> int | None:
        try:
            value = int(self.pid_path.read_text(encoding="ascii").strip())
            return value if value > 0 else None
        except (OSError, ValueError):
            return None

    def _running_pid_locked(self) -> int | None:
        if self._process is not None:
            if self._process.poll() is None:
                return self._process.pid
            self._process = None

        pid = self._read_pid()
        if pid is None:
            return None
        if self._pid_alive(pid):
            return pid
        with contextlib.suppress(OSError):
            self.pid_path.unlink()
        return None

    def _pid_alive(self, pid: int) -> bool:
        if self._process is not None and self._process.pid == pid:
            return self._process.poll() is None
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True

    def _reconcile_locked(self) -> None:
        if self._running_pid_locked() is not None:
            return
        session = _read_json(self.session_path)
        if session and not session.get("archive_path"):
            self._archive_stopped_session_locked()

    # -- archives -------------------------------------------------------

    def _archive_stopped_session_locked(
        self,
        stopped_at: datetime | None = None,
    ) -> str | None:
        session = _read_json(self.session_path)
        snapshot_path = self.state_dir / "data.json"
        snapshot = _read_json(snapshot_path)
        if session.get("archive_path"):
            return str(session["archive_path"])
        if not session and snapshot.get("mode") == "simulation":
            return None
        if not session and not self._has_session_artifacts():
            return None

        if snapshot.get("online") is True:
            from .dashboard import mark_snapshot_offline

            mark_snapshot_offline(str(self.state_dir))
            snapshot = _read_json(snapshot_path)

        stopped = (stopped_at or _utc_now()).astimezone()
        archive = self._unique_archive_dir(stopped)
        archive.mkdir(parents=True, exist_ok=False)

        if not session:
            session = self._session_from_snapshot(snapshot, stopped)
        session.update({
            "status": "stopped",
            "stopped_at": session.get("stopped_at") or _iso(stopped),
            "archive_path": str(archive),
            "archive_relative_path": os.path.relpath(archive, self.repo_dir),
        })
        _write_json(self.session_path, session)

        copied: list[str] = []
        candidates = {
            self.state_dir / "data.json",
            self.state_dir / "portfolio.json",
            self.state_dir / "trades.csv",
            self.state_dir / SESSION_NAME,
            *self.state_dir.glob("trader.log*"),
            *self.state_dir.glob("hft-live.*.log"),
        }
        for source in sorted(candidates, key=lambda path: path.name):
            if not source.is_file():
                continue
            destination = archive / source.name
            if source.name == "trades.csv":
                self._copy_session_trades(
                    source,
                    destination,
                    session.get("started_at"),
                    session.get("stopped_at"),
                )
            else:
                shutil.copy2(source, destination)
            copied.append(source.name)
        if self.config_path.is_file():
            shutil.copy2(self.config_path, archive / "config.yaml")
            copied.append("config.yaml")

        summary = {
            "session_id": session.get("session_id"),
            "execution": "paper",
            "mode": session.get("mode", snapshot.get("mode")),
            "started_at": session.get("started_at"),
            "stopped_at": session.get("stopped_at"),
            "duration_seconds": self._duration_seconds(session, snapshot),
            "starting_cash": session.get(
                "starting_cash",
                (snapshot.get("stats") or {}).get("starting_cash"),
            ),
            "final_stats": snapshot.get("stats") or {},
            "liquidation_equity": snapshot.get("liquidation_equity"),
            "signals": snapshot.get("signals", 0),
            "orders": snapshot.get("orders", 0),
            "hft": snapshot.get("hft") or {},
            "universe": snapshot.get("universe") or {},
            "files": sorted(copied),
        }
        _write_json(archive / "summary.json", summary)

        last_archive = {
            "stopped_at": session["stopped_at"],
            "path": str(archive),
            "relative_path": os.path.relpath(archive, self.repo_dir),
        }
        _write_json(self.last_archive_path, last_archive)
        return str(archive)

    def _has_session_artifacts(self) -> bool:
        return any(
            path.is_file()
            for path in (
                self.state_dir / "portfolio.json",
                self.state_dir / "trades.csv",
                self.state_dir / "hft-live.stdout.log",
                self.state_dir / "trader.log",
            )
        )

    @staticmethod
    def _copy_session_trades(
        source: Path,
        destination: Path,
        started_at: Any,
        stopped_at: Any,
    ) -> None:
        """Copy only journal rows whose timestamps belong to this session."""
        try:
            started = datetime.fromisoformat(str(started_at))
            stopped = datetime.fromisoformat(str(stopped_at))
            with source.open(newline="", encoding="utf-8") as src:
                reader = csv.DictReader(src)
                if not reader.fieldnames or "timestamp" not in reader.fieldnames:
                    raise ValueError("journal has no timestamp column")
                rows = []
                for row in reader:
                    try:
                        timestamp = datetime.fromisoformat(str(row["timestamp"]))
                    except (TypeError, ValueError):
                        continue
                    if started <= timestamp <= stopped:
                        rows.append(row)
            with destination.open("w", newline="", encoding="utf-8") as dst:
                writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        except (OSError, TypeError, ValueError):
            shutil.copy2(source, destination)

    def _session_from_snapshot(
        self,
        snapshot: dict[str, Any],
        stopped: datetime,
    ) -> dict[str, Any]:
        runtime = float(snapshot.get("runtime_minutes") or 0) * 60
        started = stopped.astimezone(timezone.utc) - timedelta(seconds=runtime)
        return {
            "session_id": stopped.astimezone(timezone.utc).strftime(
                "%Y%m%dT%H%M%S.%fZ"
            ),
            "execution": "paper",
            "mode": snapshot.get("mode", "hft"),
            "started_at": _iso(started),
            "starting_cash": (snapshot.get("stats") or {}).get(
                "starting_cash",
                100.0,
            ),
        }

    def _duration_seconds(
        self,
        session: dict[str, Any],
        snapshot: dict[str, Any],
    ) -> float:
        try:
            started = datetime.fromisoformat(str(session["started_at"]))
            stopped = datetime.fromisoformat(str(session["stopped_at"]))
            return round(max(0.0, (stopped - started).total_seconds()), 3)
        except (KeyError, TypeError, ValueError):
            return round(float(snapshot.get("runtime_minutes") or 0) * 60, 3)

    def _unique_archive_dir(self, stopped: datetime) -> Path:
        base = stopped.strftime("%Y-%m-%d_%H-%M-%S")
        candidate = self.sessions_dir / base
        suffix = 2
        while candidate.exists():
            candidate = self.sessions_dir / f"{base}_{suffix}"
            suffix += 1
        return candidate

    def _prepare_fresh_state_locked(self) -> None:
        clear_stop_request(str(self.state_dir))
        paths = {
            self.state_dir / "portfolio.json",
            self.state_dir / "trades.csv",
            self.state_dir / SESSION_NAME,
            self.state_dir / PID_NAME,
            *self.state_dir.glob("trader.log*"),
            *self.state_dir.glob("hft-live.*.log"),
        }
        for path in paths:
            with contextlib.suppress(FileNotFoundError):
                path.unlink()
