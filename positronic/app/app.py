"""
PositronicApp — top-level orchestrator for the desktop validator node.

Owns: CustomTkinter root, NodeRunner thread, SystemTray, MainWindow.
Coordinates lifecycle: startup, RPC polling, minimize-to-tray, shutdown.

Architecture:
  - Runs a local validator node via NodeRunner (daemon thread).
  - Polls the local RPC (localhost:8545) for dashboard data.
  - Falls back to remote testnet RPC when the local node is not ready.
  - This ensures the dashboard shows live network data immediately,
    even while the local node is still syncing or has crashed.
"""

import logging
import os
import sys
import threading

import customtkinter as ctk

from positronic.app.theme import (
    COLORS, POLL_INTERVAL_MS,
    WINDOW_DEFAULT_WIDTH, WINDOW_DEFAULT_HEIGHT,
    WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT,
)
from positronic.app.node_runner import NodeRunner, NodeState
from positronic.app.rpc_client import RPCClient
from positronic.app.api import DesktopApi
from positronic.app.main_window import MainWindow
from positronic.app.tray import SystemTray

logger = logging.getLogger("positronic.app")

# Remote testnet RPC — used as fallback when local node is not ready
TESTNET_RPC_HOST = "rpc.positronic-ai.network"
TESTNET_RPC_PORT = 443


class PositronicApp:
    """Desktop application for the Positronic validator node."""

    def __init__(self, data_dir: str, founder_mode: bool = False):
        # Single-instance guard: prevent two app instances binding port 9000
        import socket as _socket
        _sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        _sock.settimeout(0.3)
        _already_running = _sock.connect_ex(("127.0.0.1", 9000)) == 0
        _sock.close()
        if _already_running:
            import tkinter as _tk
            import tkinter.messagebox as _mb
            _r = _tk.Tk()
            _r.withdraw()
            _mb.showerror(
                "Positronic Already Running",
                "Another instance of Positronic is already running.\n"
                "Check the system tray or task manager.",
            )
            _r.destroy()
            raise SystemExit(0)

        self.data_dir = data_dir
        self.founder_mode = founder_mode
        self._shutdown_lock = threading.Lock()
        self._shutting_down = False
        self._using_remote = False  # True when falling back to remote RPC
        self._poll_thread = None  # Single polling thread (prevents accumulation)

        # ── CustomTkinter configuration ───────────────────────────
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        from positronic import __version__
        self.root.title(f"Positronic Node v{__version__}")
        self.root.geometry(f"{WINDOW_DEFAULT_WIDTH}x{WINDOW_DEFAULT_HEIGHT}")
        self.root.minsize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        self.root.configure(fg_color=COLORS["bg_dark"])
        self.root.protocol("WM_DELETE_WINDOW", self._on_close_window)

        # Window icon
        self._set_icon()

        # ── Core components ───────────────────────────────────────
        # Local RPC (primary — connects to local node)
        # Auto-detect TLS: tries HTTPS first (TLS default), falls back to HTTP
        self.rpc_client = RPCClient(
            host="127.0.0.1", port=8545,
            use_tls=True, auto_detect_tls=True,
        )
        # Remote RPC (fallback — connects to testnet infrastructure via HTTPS)
        # IMPORTANT: auto_detect_tls=False to prevent switching to HTTP
        # (remote server only accepts HTTPS on port 443)
        self.remote_rpc_client = RPCClient(
            host=TESTNET_RPC_HOST, port=TESTNET_RPC_PORT,
            use_tls=True, auto_detect_tls=False,
        )
        self.node_runner = NodeRunner(data_dir=data_dir)
        self.api = DesktopApi(self.rpc_client, data_dir)
        # Separate API for remote fallback
        self.remote_api = DesktopApi(self.remote_rpc_client, data_dir)

        # ── Bottom control bar (pack BEFORE MainWindow so it stacks
        #    at the bottom beneath the status bar) ──────────────────
        self._build_controls()

        # ── UI ────────────────────────────────────────────────────
        self.main_window = MainWindow(self.root, self.api, remote_api=self.remote_api)
        self.main_window._app = self
        self.tray = SystemTray(
            on_show=self._show_window,
            on_quit=self._quit,
            on_toggle_node=self._toggle_node,
        )

    def run(self):
        """Start everything and enter mainloop."""
        logger.info("Starting Positronic Desktop App (data=%s)", self.data_dir)

        self.tray.start()

        from positronic.app.ui.tab_settings import get_saved_validator_mode
        is_full_validator = (get_saved_validator_mode() == "full")
        self.node_runner.start(
            founder_mode=self.founder_mode,
            validator=is_full_validator,
        )

        self._schedule_poll()

        # Background update check (non-blocking, best-effort)
        self._check_for_updates()

        self.root.mainloop()

    # ── Icon ──────────────────────────────────────────────────────

    def _set_icon(self):
        """Set window icon from build/icons/, logo.jpg, or fallback."""
        try:
            from PIL import Image, ImageTk
            _base = os.path.dirname(os.path.abspath(__file__))
            _project = os.path.dirname(os.path.dirname(_base))
            icon_candidates = [
                # Prefer build/icons/ directory (high-res icons)
                os.path.join(_project, "build", "icons", "icon.png"),
                os.path.join(_project, "build", "icons", "icon.ico"),
                os.path.join(_project, "build", "icons", "positronic.png"),
                # Existing fallbacks
                os.path.join(_base, "ui", "logo.jpg"),
                os.path.join(_base, "ui", "brain-icon.png"),
                os.path.join(_project, "LOGO.jpg"),
            ]
            for path in icon_candidates:
                if os.path.isfile(path):
                    img = Image.open(path)
                    icon = ImageTk.PhotoImage(img.resize((64, 64)))
                    self.root.iconphoto(True, icon)
                    self._icon_ref = icon  # prevent GC
                    return
        except Exception as e:
            logger.debug("Failed to load window icon: %s", e)

    # ── Controls bar ──────────────────────────────────────────────

    def _build_controls(self):
        """Build the control bar at the bottom of the window."""
        bar = ctk.CTkFrame(self.root, fg_color=COLORS["bg_darkest"],
                           height=48, corner_radius=0)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        self._btn_toggle = ctk.CTkButton(
            bar, text="⏹ Stop Node", font=("Segoe UI", 12, "bold"),
            fg_color=COLORS["danger"], text_color="#FFFFFF",
            hover_color="#cc3333", corner_radius=8,
            width=130, height=34,
            command=self._toggle_node,
        )
        self._btn_toggle.pack(side="left", padx=(16, 8), pady=7)

        ctk.CTkButton(
            bar, text="🌐 Open Dashboard", font=("Segoe UI", 12, "bold"),
            fg_color=COLORS["accent"], text_color="#000000",
            hover_color=COLORS["accent_blue"], corner_radius=8,
            width=160, height=34,
            command=self._open_dashboard,
        ).pack(side="left", padx=4, pady=7)

        self._state_label = ctk.CTkLabel(
            bar, text="", font=("Cascadia Code", 10),
            text_color=COLORS["text_muted"])
        self._state_label.pack(side="right", padx=16)

    # ── Polling (threaded to avoid blocking the GUI) ──────────────

    @property
    def _is_shutting_down(self) -> bool:
        with self._shutdown_lock:
            return self._shutting_down

    def _schedule_poll(self):
        if self._is_shutting_down:
            return

        def _bg_fetch():
            """Run RPC queries in background thread, then push UI
            update back onto the tkinter main thread.

            Strategy: try local node first. If local node is not
            responding (starting, crashed, or syncing), fall back
            to the remote testnet RPC so the dashboard always
            shows live data.
            """
            if self._is_shutting_down:
                return

            # Use local node when running, remote as fallback
            if (self.node_runner.state in (NodeState.RUNNING, NodeState.LIGHT_VALIDATING)
                    and self.rpc_client.is_alive()
                    and not getattr(self.node_runner, 'is_syncing', False)):
                api = self.api
                using_remote = False
            else:
                api = self.remote_api
                using_remote = True

            # Each call wrapped individually so one failure doesn't kill all
            try:
                dashboard = api.get_dashboard()
            except Exception:
                dashboard = None
            try:
                network = api.get_network()
            except Exception:
                network = None
            try:
                emergency = api.get_emergency_status()
            except Exception:
                emergency = None
            try:
                validator = self.remote_api.get_validator_info()
            except Exception:
                validator = None

            # Ecosystem: heavy (18 RPCs) — only fetch every 6th poll (~30s)
            if not hasattr(self, '_eco_counter'):
                self._eco_counter = 0
            self._eco_counter += 1
            if self._eco_counter >= 6:
                self._eco_counter = 0
                try:
                    ecosystem = self.remote_api.get_ecosystem()
                except Exception:
                    ecosystem = None
            else:
                ecosystem = None

            # Fetch user-specific staking info if wallet is unlocked
            active_addr = getattr(self.main_window, '_active_address', None)
            if active_addr and validator:
                try:
                    staking = self.remote_api.get_staking_info(active_addr)
                    if staking:
                        from positronic.constants import BASE_UNIT
                        validator["your_stake"] = staking.get("staked", 0) / BASE_UNIT
                        validator["your_rewards"] = staking.get("rewards", 0) / BASE_UNIT
                        validator["your_is_validator"] = staking.get("is_validator", False)
                        validator["your_available"] = staking.get("available", 0) / BASE_UNIT
                except Exception:
                    pass

            # Fetch TX history if wallet unlocked
            tx_history = []
            if active_addr:
                try:
                    tx_history = self.remote_api.get_address_transactions(active_addr, 20)
                except Exception:
                    tx_history = []
            if dashboard and tx_history:
                dashboard["_tx_history"] = tx_history

            # Tag dashboard with source info
            if using_remote and dashboard:
                dashboard["_remote"] = True
                # If remote returns online, add connecting hint
                # to show "local node starting" in status bar
                node_state = self.node_runner.state
                if node_state in (NodeState.STARTING, NodeState.SYNCING):
                    dashboard["_node_status"] = "Local node syncing..."
                elif node_state == NodeState.ERROR:
                    dashboard["_node_status"] = (
                        f"Local node error: "
                        f"{(self.node_runner.last_error or 'unknown')[:60]}"
                    )
                else:
                    dashboard["_node_status"] = "Local node starting..."

            self._using_remote = using_remote

            if self._is_shutting_down:
                return

            # Push results to main thread for UI update
            self.root.after(0, self._apply_poll_data,
                            dashboard, validator, network, emergency, ecosystem)

        # Only start new poll thread if previous one finished (prevents accumulation)
        if self._poll_thread is None or not self._poll_thread.is_alive():
            self._poll_thread = threading.Thread(target=_bg_fetch, daemon=True)
            self._poll_thread.start()
        self.root.after(POLL_INTERVAL_MS, self._schedule_poll)

    def _apply_poll_data(self, dashboard, validator, network,
                         emergency=None, ecosystem=None):
        """Apply fetched data to UI widgets — runs on main thread."""
        if self._is_shutting_down:
            return
        try:
            # Inject user staking info into dashboard for card display
            if dashboard and validator:
                dashboard["_your_stake"] = validator.get("your_stake", 0)
                dashboard["_your_is_validator"] = validator.get("your_is_validator", False)
            self.main_window.refresh_with_data(
                dashboard, validator, network, emergency, ecosystem)
            self._update_controls()
        except Exception as exc:
            logger.debug("UI update error: %s", exc)

    def _update_controls(self):
        if self.node_runner.is_running:
            self._btn_toggle.configure(text="⏹ Stop Node",
                                       fg_color=COLORS["danger"])
        else:
            self._btn_toggle.configure(text="▶ Start Node",
                                       fg_color=COLORS["success"])

        state = self.node_runner.state.value
        error = self.node_runner.last_error
        label = f"Node: {state}"
        if error:
            label += f"  |  {error[:50]}"
        if self._using_remote:
            label += "  |  📡 Remote RPC"
        self._state_label.configure(text=label)

    # ── Actions ───────────────────────────────────────────────────

    def _toggle_node(self):
        if self.node_runner.is_running:
            logger.info("Stopping node...")
            # Update button immediately for responsiveness
            self._btn_toggle.configure(
                text="⏳ Stopping...",
                fg_color="#666666",
                state="disabled",
            )
            self.root.update_idletasks()
            # Run stop in background thread to avoid GUI freeze
            import threading
            def _do_stop():
                self.node_runner.stop()
                # Re-enable button on main thread
                self.root.after(0, lambda: self._btn_toggle.configure(state="normal"))
            threading.Thread(target=_do_stop, daemon=True).start()
        else:
            logger.info("Starting node...")
            from positronic.app.ui.tab_settings import get_saved_validator_mode
            is_full_validator = (get_saved_validator_mode() == "full")
            self.node_runner.start(
                founder_mode=self.founder_mode, validator=is_full_validator)

    def _restart_node_with_validator(self, validator_enabled: bool):
        """Stop and restart node with updated validator mode."""
        import threading
        self._btn_toggle.configure(text="⏳ Restarting...", fg_color="#666666", state="disabled")
        self.root.update_idletasks()
        def _do_restart():
            self.node_runner.stop()
            self.node_runner.start(
                founder_mode=self.founder_mode,
                validator=validator_enabled,
            )
            self.root.after(0, lambda: self._btn_toggle.configure(state="normal"))
        threading.Thread(target=_do_restart, daemon=True).start()

    def _open_dashboard(self):
        import webbrowser
        webbrowser.open("https://positronic-ai.network/panel/")

    def _on_close_window(self):
        if self.tray.available:
            self.root.withdraw()
            logger.info("Window hidden to tray")
        else:
            self._quit()

    def _show_window(self):
        self.root.after(0, self._do_show)

    def _do_show(self):
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    def _check_for_updates(self):
        """Background update check — notify via GUI banner if newer version available."""
        import asyncio

        def _bg_check():
            try:
                loop = asyncio.new_event_loop()
                from positronic.app.updater import check_for_updates
                result = loop.run_until_complete(check_for_updates())
                loop.close()
                if result.get("available"):
                    self.root.after(0, self.main_window.show_update_banner, result)
            except Exception as e:
                logger.debug("Update check error: %s", e)

        threading.Thread(target=_bg_check, daemon=True).start()

    def _quit(self):
        logger.info("Shutting down...")
        # Clear secret key reference from memory before exit (best-effort)
        try:
            if hasattr(self.main_window, '_active_secret_key') and self.main_window._active_secret_key:
                self.main_window._active_secret_key = None
                import gc
                gc.collect()
        except Exception:
            pass
        with self._shutdown_lock:
            self._shutting_down = True
        self.node_runner.stop()
        self.tray.stop()
        try:
            self.root.quit()
            self.root.destroy()
        except Exception as e:
            logger.debug("Error during root cleanup: %s", e)
