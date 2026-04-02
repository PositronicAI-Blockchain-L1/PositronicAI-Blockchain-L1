"""
Positronic - Phase 32e: Neural Watchdog Service

A subprocess-based health monitor that detects when the main node process
becomes unresponsive by polling a heartbeat file. Designed to be lightweight
with zero heavy blockchain imports.

Architecture:
  - HeartbeatFile: used by the main node to write periodic timestamps
  - NeuralWatchdog: separate process that polls the heartbeat file
  - WatchdogLauncher: helper to start/stop the watchdog subprocess

The watchdog uses sleep-based polling for negligible CPU usage. On Unix,
it sends SIGUSR1 to trigger a neural snapshot before escalating. On
Windows, the signal step is skipped gracefully.
"""

import os
import sys
import time
import subprocess
from typing import Tuple, Optional

from positronic.utils.logging import get_logger
from positronic.constants import (
    WATCHDOG_CHECK_INTERVAL,
    WATCHDOG_MISS_THRESHOLD,
    WATCHDOG_SIGUSR1_WAIT,
)

logger = get_logger(__name__)

# Check platform for signal support
_HAS_SIGUSR1 = hasattr(__import__("signal"), "SIGUSR1")


class HeartbeatFile:
    """Atomic heartbeat file for the main node process.

    The main node calls ``beat()`` periodically to write the current
    timestamp. The watchdog reads this file to determine if the node
    is still alive.

    Atomic write strategy: write to ``<path>.tmp`` then ``os.replace()``
    to avoid partial reads.
    """

    def __init__(self, heartbeat_path: str) -> None:
        self._path = heartbeat_path

    def beat(self) -> None:
        """Write current timestamp to heartbeat file (atomic)."""
        tmp_path = self._path + ".tmp"
        ts = time.time()
        try:
            with open(tmp_path, "w") as f:
                f.write(str(ts))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self._path)
        except OSError as e:
            logger.error("heartbeat write failed: %s", e)

    def get_last_beat(self) -> float:
        """Read last timestamp from heartbeat file.

        Returns:
            The timestamp as a float, or 0.0 if the file does not exist
            or is unreadable.
        """
        try:
            with open(self._path, "r") as f:
                content = f.read().strip()
            return float(content)
        except (FileNotFoundError, ValueError, OSError):
            return 0.0


class NeuralWatchdog:
    """Subprocess-based health monitor for the Positronic node.

    Polls a heartbeat file at ``check_interval`` intervals. If the
    heartbeat is stale for ``miss_threshold`` consecutive checks, the
    watchdog triggers an emergency action sequence.

    Args:
        heartbeat_path: Path to the heartbeat file.
        check_interval: Seconds between heartbeat checks.
        miss_threshold: Consecutive misses before triggering failure.
        node_pid: PID of the node process to signal.
    """

    def __init__(
        self,
        heartbeat_path: str,
        check_interval: float = WATCHDOG_CHECK_INTERVAL,
        miss_threshold: int = WATCHDOG_MISS_THRESHOLD,
        node_pid: Optional[int] = None,
    ) -> None:
        self._heartbeat_path = heartbeat_path
        self._heartbeat = HeartbeatFile(heartbeat_path)
        self._check_interval = check_interval
        self._miss_threshold = miss_threshold
        self._node_pid = node_pid
        self._running = False
        self._consecutive_misses = 0
        self._last_check_time: float = 0.0
        self._incident_file: Optional[str] = None
        self._sigusr1_wait = WATCHDOG_SIGUSR1_WAIT

    def check_heartbeat(self) -> Tuple[bool, float]:
        """Check if the node's heartbeat is recent.

        Returns:
            Tuple of (is_alive, seconds_since_last_beat).
            is_alive is True if the last beat is within the acceptable
            window (check_interval * miss_threshold).
        """
        last_beat = self._heartbeat.get_last_beat()
        now = time.time()
        age = now - last_beat
        # Consider alive if last beat is within the acceptable window
        max_age = self._check_interval * self._miss_threshold
        is_alive = age <= max_age
        self._last_check_time = now
        return is_alive, age

    def run(self) -> None:
        """Main watchdog loop. Check heartbeat periodically.

        Blocks until ``stop()`` is called. Counts consecutive misses
        and triggers ``_on_failure()`` when the threshold is reached.
        """
        self._running = True
        self._consecutive_misses = 0
        logger.info(
            "watchdog started: interval=%.2fs, threshold=%d, node_pid=%s",
            self._check_interval,
            self._miss_threshold,
            self._node_pid,
        )

        while self._running:
            is_alive, age = self.check_heartbeat()

            if is_alive:
                self._consecutive_misses = 0
            else:
                self._consecutive_misses += 1
                logger.warning(
                    "heartbeat miss %d/%d (age=%.1fs)",
                    self._consecutive_misses,
                    self._miss_threshold,
                    age,
                )

                if self._consecutive_misses >= self._miss_threshold:
                    self._on_failure(self._consecutive_misses, age)

            if not self._running:
                break

            time.sleep(self._check_interval)

        self._running = False
        logger.info("watchdog stopped")

    def _on_failure(self, misses: int, last_beat_age: float) -> None:
        """Emergency action sequence when node is unresponsive.

        Steps:
            1. Send SIGUSR1 to node (Unix only) to trigger neural snapshot.
            2. Wait ``_sigusr1_wait`` seconds.
            3. Log operator alert with details.
            4. Log incident to incident file.
        """
        logger.critical(
            "NODE UNRESPONSIVE: %d consecutive misses, last beat %.1fs ago",
            misses,
            last_beat_age,
        )

        # Step 1: Send SIGUSR1 (Unix only)
        if _HAS_SIGUSR1 and self._node_pid is not None:
            import signal as sig_mod
            try:
                os.kill(self._node_pid, sig_mod.SIGUSR1)
                logger.info("sent SIGUSR1 to node pid=%d", self._node_pid)
            except (ProcessLookupError, PermissionError, OSError) as e:
                logger.error("failed to send SIGUSR1: %s", e)
        else:
            logger.info(
                "SIGUSR1 not available on this platform (pid=%s), skipping signal",
                self._node_pid,
            )

        # Step 2: Wait for neural snapshot
        time.sleep(self._sigusr1_wait)

        # Step 3: Log operator alert
        logger.critical(
            "OPERATOR ALERT: Node pid=%s unresponsive for %d checks (%.1fs). "
            "Immediate investigation required.",
            self._node_pid,
            misses,
            last_beat_age,
        )

        # Step 4: Log incident to file
        if self._incident_file:
            try:
                with open(self._incident_file, "a") as f:
                    f.write(
                        f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
                        f"INCIDENT: node_pid={self._node_pid} "
                        f"misses={misses} "
                        f"last_beat_age={last_beat_age:.1f}s\n"
                    )
            except OSError as e:
                logger.error("failed to write incident file: %s", e)

    def stop(self) -> None:
        """Set stop flag to exit the run loop."""
        self._running = False
        logger.info("watchdog stop requested")

    def get_status(self) -> dict:
        """Return current watchdog status.

        Returns:
            Dict with keys: running, consecutive_misses, last_check_time,
            check_interval, miss_threshold, node_pid.
        """
        return {
            "running": self._running,
            "consecutive_misses": self._consecutive_misses,
            "last_check_time": self._last_check_time,
            "check_interval": self._check_interval,
            "miss_threshold": self._miss_threshold,
            "node_pid": self._node_pid,
        }

    def generate_systemd_service(
        self,
        node_command: str,
        heartbeat_path: str,
    ) -> str:
        """Generate a systemd .service file for the watchdog.

        Args:
            node_command: Full command to start the node.
            heartbeat_path: Path to the heartbeat file.

        Returns:
            String content of a systemd .service unit file.
        """
        return f"""[Unit]
Description=Positronic Neural Watchdog Service
Documentation=https://positronic-ai.network/docs
After=network.target positronic-node.service
Wants=positronic-node.service

[Service]
Type=simple
ExecStart={sys.executable} -m positronic.services.neural_watchdog \\
    --heartbeat-path {heartbeat_path} \\
    --node-command "{node_command}"
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=positronic-watchdog

# Security hardening
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths={os.path.dirname(heartbeat_path)}

[Install]
WantedBy=multi-user.target
"""


class WatchdogLauncher:
    """Helper to start and stop the watchdog as a subprocess."""

    @staticmethod
    def launch(
        heartbeat_path: str,
        node_pid: int,
        check_interval: float = WATCHDOG_CHECK_INTERVAL,
        miss_threshold: int = WATCHDOG_MISS_THRESHOLD,
    ) -> subprocess.Popen:
        """Start the watchdog in a separate process.

        Args:
            heartbeat_path: Path to the heartbeat file.
            node_pid: PID of the node to monitor.
            check_interval: Seconds between checks.
            miss_threshold: Consecutive misses before action.

        Returns:
            subprocess.Popen instance for the watchdog process.
        """
        cmd = [
            sys.executable,
            "-m",
            "positronic.services.neural_watchdog",
            "--heartbeat-path", heartbeat_path,
            "--node-pid", str(node_pid),
            "--check-interval", str(check_interval),
            "--miss-threshold", str(miss_threshold),
        ]
        logger.info("launching watchdog subprocess: %s", " ".join(cmd))
        # On Windows, suppress CMD window in frozen GUI apps
        kwargs = dict(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if sys.platform == "win32":
            kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
        proc = subprocess.Popen(cmd, **kwargs)
        logger.info("watchdog subprocess started: pid=%d", proc.pid)
        return proc

    @staticmethod
    def stop(process: subprocess.Popen) -> None:
        """Gracefully stop a watchdog subprocess.

        Sends SIGTERM (or terminate on Windows) and waits up to 5 seconds
        before killing.
        """
        if process.poll() is not None:
            return  # Already exited

        logger.info("stopping watchdog subprocess pid=%d", process.pid)
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("watchdog pid=%d did not exit, killing", process.pid)
            process.kill()
            process.wait(timeout=2)


# ─── CLI Entry Point ─────────────────────────────────────────────


def _main() -> None:
    """CLI entry point for running the watchdog as ``python -m ...``."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Positronic Neural Watchdog Service"
    )
    parser.add_argument(
        "--heartbeat-path", required=True,
        help="Path to the heartbeat file"
    )
    parser.add_argument(
        "--node-pid", type=int, default=None,
        help="PID of the node process to monitor"
    )
    parser.add_argument(
        "--check-interval", type=float, default=WATCHDOG_CHECK_INTERVAL,
        help="Seconds between heartbeat checks"
    )
    parser.add_argument(
        "--miss-threshold", type=int, default=WATCHDOG_MISS_THRESHOLD,
        help="Consecutive misses before triggering failure"
    )
    parser.add_argument(
        "--incident-file", default=None,
        help="Path to write incident logs"
    )
    parser.add_argument(
        "--node-command", default=None,
        help="Node command (for systemd generation mode)"
    )
    parser.add_argument(
        "--generate-systemd", action="store_true",
        help="Print systemd service file and exit"
    )

    args = parser.parse_args()

    watchdog = NeuralWatchdog(
        heartbeat_path=args.heartbeat_path,
        check_interval=args.check_interval,
        miss_threshold=args.miss_threshold,
        node_pid=args.node_pid,
    )

    if args.generate_systemd:
        print(watchdog.generate_systemd_service(
            node_command=args.node_command or "positronic-node",
            heartbeat_path=args.heartbeat_path,
        ))
        return

    if args.incident_file:
        watchdog._incident_file = args.incident_file

    watchdog.run()


if __name__ == "__main__":
    _main()
