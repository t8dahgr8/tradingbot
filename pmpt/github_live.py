"""Publish dashboard heartbeats to an isolated GitHub branch.

GitHub Pages cannot run the Python process. The public dashboard instead reads
one JSON file from ``live-data`` while the local paper trader is running. Git
plumbing keeps these frequent updates out of the checked-out branch and replaces
the branch with a single root commit each time, so live data does not fill the
project history.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
from datetime import datetime, timezone

log = logging.getLogger(__name__)


class GitHubLivePublisher:
    def __init__(
        self,
        repo_dir: str,
        snapshot_path: str,
        *,
        interval_s: int = 30,
        remote: str = "origin",
        branch: str = "live-data",
    ):
        self.repo_dir = os.path.abspath(repo_dir)
        self.snapshot_path = os.path.abspath(snapshot_path)
        self.interval_s = max(10, int(interval_s))
        self.remote = remote
        self.branch = branch
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_payload: bytes | None = None

    def start(self) -> bool:
        """Start publishing without making GitHub availability a trading dependency."""
        try:
            top = _stdout(self._git("rev-parse", "--show-toplevel"))
            self._git("remote", "get-url", self.remote)
        except (OSError, subprocess.SubprocessError) as exc:
            log.warning("GitHub live dashboard disabled: %s", _error_text(exc))
            return False

        if os.path.normcase(os.path.abspath(top)) != os.path.normcase(self.repo_dir):
            self.repo_dir = os.path.abspath(top)

        self._thread = threading.Thread(
            target=self._run,
            name="github-live-publisher",
            daemon=True,
        )
        self._thread.start()
        log.info(
            "GitHub live dashboard enabled | %s/%s every %ss",
            self.remote,
            self.branch,
            self.interval_s,
        )
        return True

    def stop(self, publish_final: bool = True) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=25)
            if self._thread.is_alive():
                log.warning(
                    "GitHub live publisher did not stop cleanly; "
                    "the stale heartbeat will expire"
                )
                return
        if publish_final:
            self.publish_once(force=True)

    def publish_once(self, force: bool = False) -> bool:
        try:
            with open(self.snapshot_path, "rb") as fh:
                payload = fh.read()
            json.loads(payload)
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("GitHub live snapshot unavailable: %s", exc)
            return False

        if not force and payload == self._last_payload:
            return True

        try:
            blob = _stdout(self._git(
                "hash-object", "-w", "--stdin", input_bytes=payload
            ))
            tree_line = f"100644 blob {blob}\tdata.json\n".encode("ascii")
            tree = _stdout(self._git("mktree", input_bytes=tree_line))
            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            commit = _stdout(self._git(
                "commit-tree", tree, "-m", f"Live dashboard heartbeat {stamp}"
            ))
            self._git(
                "push",
                "--quiet",
                "--force",
                self.remote,
                f"{commit}:refs/heads/{self.branch}",
                timeout=20,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            log.warning("GitHub live publish failed: %s", _error_text(exc))
            return False

        self._last_payload = payload
        log.debug("published GitHub live dashboard heartbeat")
        return True

    def _run(self) -> None:
        while not self._stop.is_set():
            self.publish_once()
            self._stop.wait(self.interval_s)

    def _git(
        self,
        *args: str,
        input_bytes: bytes | None = None,
        timeout: int = 15,
    ) -> subprocess.CompletedProcess:
        env = os.environ.copy()
        env.setdefault("GIT_AUTHOR_NAME", "Paper Trader Dashboard")
        env.setdefault("GIT_AUTHOR_EMAIL", "paper-trader@localhost")
        env.setdefault("GIT_COMMITTER_NAME", env["GIT_AUTHOR_NAME"])
        env.setdefault("GIT_COMMITTER_EMAIL", env["GIT_AUTHOR_EMAIL"])
        return subprocess.run(
            ["git", *args],
            cwd=self.repo_dir,
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            timeout=timeout,
            env=env,
        )


def _error_text(exc: BaseException) -> str:
    stderr = getattr(exc, "stderr", b"")
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    return str(stderr).strip() or str(exc)


def _stdout(result: subprocess.CompletedProcess) -> str:
    value = result.stdout
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()
