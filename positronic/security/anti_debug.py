"""
Anti-Debugging Module for Positronic Desktop Application.

Detects common reverse-engineering tools and debuggers.
Cross-platform: Windows, Linux, macOS.
Response is graceful (log + warning) -- never crashes the application.
"""

import logging
import os
import sys
import threading
import time

logger = logging.getLogger("positronic.security.anti_debug")

# Known reverse-engineering and debugging tool process names
KNOWN_RE_TOOLS = frozenset([
    # Debuggers
    "ida", "ida64", "idag", "idag64", "idaw", "idaw64",
    "ghidra", "ghidrarun", "ghidra-analyzeheadless",
    "x64dbg", "x32dbg", "ollydbg",
    "radare2", "r2", "rizin", "cutter",
    "gdb", "lldb", "windbg",
    # Dynamic analysis
    "frida", "frida-server", "frida-agent",
    "strace", "ltrace", "dtrace",
    # Python-specific
    "pyinstxtractor", "uncompyle6", "decompyle3", "pycdc",
    # Network analysis
    "wireshark", "tshark", "tcpdump", "mitmproxy",
    # Memory tools
    "cheatengine", "processhacker",
])


class AntiDebug:
    """Cross-platform debugger and reverse-engineering detection."""

    _monitoring = False
    _thread = None
    _detected = False

    @classmethod
    def is_compiled(cls) -> bool:
        """Check if running as a compiled binary."""
        return getattr(sys, 'frozen', False) or '__compiled__' in globals()

    @classmethod
    def check(cls) -> bool:
        """Run all anti-debug checks. Returns True if debugging detected."""
        if not cls.is_compiled():
            return False

        checks = [
            cls._check_platform_debugger,
            cls._check_known_processes,
            cls._check_timing,
        ]

        for check_fn in checks:
            try:
                if check_fn():
                    cls._detected = True
                    return True
            except Exception as e:
                logger.debug("Anti-debug check failed gracefully: %s", e)

        return False

    @classmethod
    def start_monitoring(cls, interval: int = 30):
        """Start background monitoring thread.

        Checks for debuggers every `interval` seconds.
        Graceful: logs warnings, never crashes.
        """
        if cls._monitoring:
            return

        cls._monitoring = True

        def _monitor():
            while cls._monitoring:
                try:
                    if cls.check():
                        logger.critical(
                            "Debugging/reverse-engineering tool detected. "
                            "Shutting down to protect validator data."
                        )
                        # Graceful shutdown: signal app to save databases,
                        # then force-exit after a short cleanup window.
                        import os, signal
                        try:
                            os.kill(os.getpid(), signal.SIGTERM)
                        except (OSError, AttributeError):
                            pass
                        time.sleep(5)  # Allow cleanup handlers and DB flush
                        os._exit(1)   # Force exit if still alive
                except Exception:
                    pass
                time.sleep(interval)

        cls._thread = threading.Thread(target=_monitor, daemon=True, name="anti-debug")
        cls._thread.start()
        logger.debug("Anti-debug monitoring started (interval=%ds)", interval)

    @classmethod
    def stop_monitoring(cls):
        """Stop the background monitoring thread."""
        cls._monitoring = False
        if cls._thread:
            cls._thread.join(timeout=5)
            cls._thread = None

    @classmethod
    def was_detected(cls) -> bool:
        """Check if debugging was ever detected in this session."""
        return cls._detected

    @staticmethod
    def _check_platform_debugger() -> bool:
        """Platform-specific debugger detection."""
        if sys.platform == "win32":
            return AntiDebug._check_windows()
        elif sys.platform == "linux":
            return AntiDebug._check_linux()
        elif sys.platform == "darwin":
            return AntiDebug._check_macos()
        return False

    @staticmethod
    def _check_windows() -> bool:
        """Windows: IsDebuggerPresent + CheckRemoteDebuggerPresent."""
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32

            # IsDebuggerPresent
            if kernel32.IsDebuggerPresent():
                logger.warning("IsDebuggerPresent returned True")
                return True

            # CheckRemoteDebuggerPresent
            is_debugged = ctypes.c_bool(False)
            kernel32.CheckRemoteDebuggerPresent(
                kernel32.GetCurrentProcess(),
                ctypes.byref(is_debugged)
            )
            if is_debugged.value:
                logger.warning("Remote debugger detected")
                return True

        except Exception as e:
            logger.debug("Windows debugger check error: %s", e)
        return False

    @staticmethod
    def _check_linux() -> bool:
        """Linux: Check /proc/self/status for TracerPid."""
        try:
            with open("/proc/self/status", "r") as f:
                for line in f:
                    if line.startswith("TracerPid:"):
                        tracer_pid = int(line.split(":")[1].strip())
                        if tracer_pid != 0:
                            logger.warning("TracerPid=%d (being traced)", tracer_pid)
                            return True
                        break
        except (OSError, IOError):
            pass
        return False

    @staticmethod
    def _check_macos() -> bool:
        """macOS: Check sysctl for P_TRACED flag."""
        try:
            import ctypes
            import ctypes.util

            libc_path = ctypes.util.find_library("c")
            if not libc_path:
                return False

            libc = ctypes.CDLL(libc_path)

            # sysctl kern.proc.pid to check P_TRACED
            # CTL_KERN = 1, KERN_PROC = 14, KERN_PROC_PID = 1
            class kinfo_proc(ctypes.Structure):
                _fields_ = [("data", ctypes.c_byte * 648)]

            info = kinfo_proc()
            size = ctypes.c_size_t(ctypes.sizeof(info))
            mib = (ctypes.c_int * 4)(1, 14, 1, os.getpid())

            result = libc.sysctl(mib, 4, ctypes.byref(info), ctypes.byref(size), None, 0)
            if result == 0:
                # P_TRACED flag is at offset 32, bit 0x800
                flags = int.from_bytes(bytes(info.data[32:36]), "little")
                if flags & 0x800:
                    logger.warning("P_TRACED flag set (being debugged)")
                    return True
        except Exception as e:
            logger.debug("macOS debugger check error: %s", e)
        return False

    @staticmethod
    def _check_known_processes() -> bool:
        """Check for known RE tools running on the system."""
        try:
            import subprocess
            if sys.platform == "win32":
                # CREATE_NO_WINDOW (0x08000000) prevents CMD flash in
                # frozen PyInstaller / Nuitka apps where console=False.
                result = subprocess.run(
                    ["tasklist", "/NH", "/FO", "CSV"],
                    capture_output=True, text=True, timeout=5,
                    creationflags=0x08000000,
                )
                processes = {
                    line.split(",")[0].strip('"').lower().replace(".exe", "")
                    for line in result.stdout.strip().split("\n")
                    if line.strip()
                }
            else:
                result = subprocess.run(
                    ["ps", "-eo", "comm"],
                    capture_output=True, text=True, timeout=5
                )
                processes = {
                    os.path.basename(line.strip()).lower()
                    for line in result.stdout.strip().split("\n")
                    if line.strip()
                }

            detected = processes & KNOWN_RE_TOOLS
            if detected:
                logger.warning("Known RE tools detected: %s", detected)
                return True

        except Exception as e:
            logger.debug("Process scan error: %s", e)
        return False

    @staticmethod
    def _check_timing() -> bool:
        """Timing-based detection: breakpoints cause significant slowdown."""
        try:
            import hashlib
            start = time.perf_counter_ns()
            # Known-cost operation: 500 SHA-256 hashes
            data = b"positronic-timing-check"
            for _ in range(500):
                data = hashlib.sha256(data).digest()
            elapsed = time.perf_counter_ns() - start

            # If this takes >500ms, something is interfering (normal: <50ms)
            if elapsed > 500_000_000:
                logger.warning("Timing anomaly detected: %dms (expected <50ms)",
                             elapsed // 1_000_000)
                return True
        except Exception:
            pass
        return False
