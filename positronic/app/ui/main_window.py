"""MainWindow — Modern CustomTkinter dashboard for the Positronic validator node.

Seven tabs: Dashboard, Validator, Wallet, Network, Ecosystem, Logs, Settings.
Custom widgets: StatCard, MiningChart, ActivityPanel.
"""

import logging
import os
import sys
import time
import tkinter as tk

import customtkinter as ctk

from positronic.app.theme import COLORS, FONTS, CARD_PAD, BTN_HEIGHT, _EMOJI
from positronic.app.api import DesktopApi

from positronic.app.ui.tab_dashboard import build_dashboard, refresh_dashboard
from positronic.app.ui.tab_validator import build_validator, refresh_validator
from positronic.app.ui.tab_wallet import build_wallet
from positronic.app.ui.tab_network import build_network, refresh_network
from positronic.app.ui.tab_ecosystem import build_ecosystem, refresh_ecosystem
from positronic.app.ui.tab_logs import build_logs, setup_log_handler
from positronic.app.ui.tab_settings import build_settings

logger = logging.getLogger("positronic.app.ui")


class MainWindow:
    """Modern CustomTkinter dashboard with seven tabs."""

    def __init__(self, root: ctk.CTk, api: DesktopApi, remote_api: DesktopApi = None):
        self.root = root
        self.api = api
        self.remote_api = remote_api or api  # fallback to remote for balance/staking
        self._active_secret_key = None
        self._active_address = None
        self._session_locked_key = None
        self._action_in_progress = False  # double-click protection
        self._copy_timer_id = None  # copy button race fix
        self._session_timeout_ms = 30 * 60 * 1000  # 30 minutes
        self._session_timer_id = None
        self._build_ui()
        self._setup_session_timeout()
        # Restore wallet session AFTER mainloop starts (avoid "main thread not in main loop")
        self.root.after(500, self._try_restore_session)

    @property
    def best_api(self) -> DesktopApi:
        """Return local API if node is alive, otherwise remote."""
        try:
            if self.api and self.api._rpc.is_alive():
                return self.api
        except Exception:
            pass
        return self.remote_api

    def _build_ui(self):
        """Build the full UI — called from __init__ after properties are set."""
        root = self.root

        # Enable Ctrl+V/Ctrl+A/Ctrl+C globally for all Entry widgets
        for seq, handler in [
            ("<Control-v>", self._paste_clipboard),
            ("<Control-V>", self._paste_clipboard),
            ("<Control-a>", self._select_all),
            ("<Control-A>", self._select_all),
        ]:
            root.bind_class("Entry", seq, handler)

        # -- Keyboard shortcuts ----------------------------------------
        self._tab_names = [
            "  Dashboard  ", "  Validator  ", "  Wallet  ",
            "  Network  ", "  Ecosystem  ", "  Logs  ", "  Settings  ",
        ]
        for i in range(1, 8):
            root.bind(f"<Control-Key-{i}>",
                      lambda e, idx=i-1: self._switch_tab(idx))
        root.bind("<Control-r>", lambda e: self._request_refresh())
        root.bind("<Control-R>", lambda e: self._request_refresh())
        root.bind("<F5>", lambda e: self._request_refresh())
        root.bind("<Control-l>", lambda e: self._switch_tab(5))  # Logs
        root.bind("<Control-L>", lambda e: self._switch_tab(5))

        # Track data staleness for tab-switch loading indicator
        self._last_refresh_ts = 0.0  # time.time() of last successful refresh
        self._tab_loading_label = None  # overlay label shown briefly on stale switch

        # -- Header bar ------------------------------------------------
        header = ctk.CTkFrame(root, fg_color=COLORS["bg_darkest"], height=56,
                              corner_radius=0)
        header.pack(fill="x")
        header.pack_propagate(False)

        # Brain icon (if available)
        self._icon_image = None
        icon_path = self._find_icon()
        if icon_path:
            try:
                from PIL import Image, ImageDraw
                img = Image.open(icon_path).resize((36, 36)).convert("RGBA")
                # Make circular mask
                mask = Image.new("L", (36, 36), 0)
                ImageDraw.Draw(mask).ellipse((0, 0, 35, 35), fill=255)
                circular = Image.new("RGBA", (36, 36), (0, 0, 0, 0))
                circular.paste(img, (0, 0), mask)
                self._icon_image = ctk.CTkImage(circular, size=(36, 36))
                ctk.CTkLabel(header, image=self._icon_image, text="").pack(
                    side="left", padx=(16, 8))
            except Exception as e:
                logger.debug("Failed to load header icon: %s", e)
                ctk.CTkLabel(header, text="\u26a1", font=(_EMOJI, 22),
                             text_color=COLORS["accent"]).pack(
                    side="left", padx=(16, 8))
        else:
            ctk.CTkLabel(header, text="\u26a1", font=(_EMOJI, 22),
                         text_color=COLORS["accent"]).pack(
                side="left", padx=(16, 8))

        ctk.CTkLabel(header, text="POSITRONIC NODE",
                     font=FONTS["app_title"],
                     text_color=COLORS["text"]).pack(side="left")
        from positronic import __version__
        ctk.CTkLabel(header, text=f"v{__version__}", font=FONTS["tiny"],
                     text_color=COLORS["text_muted"]).pack(
            side="left", padx=(8, 0), pady=(6, 0))

        # Status indicator (right side)
        self._status_frame = ctk.CTkFrame(header, fg_color="transparent")
        self._status_frame.pack(side="right", padx=16)

        self._status_dot = ctk.CTkLabel(self._status_frame, text="\u25cf",
                                        font=("Arial", 16),
                                        text_color=COLORS["warning"])
        self._status_dot.pack(side="left")
        self._status_text = ctk.CTkLabel(self._status_frame, text="Starting...",
                                         font=FONTS["body"],
                                         text_color=COLORS["text_dim"])
        self._status_text.pack(side="left", padx=(6, 0))

        # -- Emergency Banner (hidden by default) ----------------------
        self._emergency_banner = ctk.CTkFrame(
            root, fg_color="#2a1000", corner_radius=0, height=40)
        self._emergency_banner.pack(fill="x", padx=0, pady=0)
        self._emergency_banner.pack_forget()  # hidden initially

        self._emergency_icon = ctk.CTkLabel(
            self._emergency_banner, text="\u26a0",
            font=("Arial", 18), text_color=COLORS["warning"])
        self._emergency_icon.pack(side="left", padx=(16, 8))

        self._emergency_text = ctk.CTkLabel(
            self._emergency_banner, text="Network PAUSED",
            font=FONTS["subheading"], text_color=COLORS["warning"])
        self._emergency_text.pack(side="left")

        self._emergency_detail = ctk.CTkLabel(
            self._emergency_banner, text="",
            font=FONTS["small"], text_color=COLORS["text_dim"])
        self._emergency_detail.pack(side="left", padx=(12, 0))

        self._last_emergency_state = 0  # Track previous state for change detection

        # -- Tabview ---------------------------------------------------
        self.tabs = ctk.CTkTabview(
            root, fg_color=COLORS["bg_dark"],
            segmented_button_fg_color=COLORS["bg_darkest"],
            segmented_button_selected_color=COLORS["accent"],
            segmented_button_selected_hover_color=COLORS["accent_blue"],
            segmented_button_unselected_color=COLORS["bg_card"],
            segmented_button_unselected_hover_color=COLORS["bg_card_hover"],
            text_color=COLORS["text"],
            text_color_disabled=COLORS["text_muted"],
            corner_radius=8,
        )
        self.tabs.pack(fill="both", expand=True, padx=12, pady=(8, 0))

        # Create tabs
        self._tab_dash = self.tabs.add("  Dashboard  ")
        self._tab_val = self.tabs.add("  Validator  ")
        self._tab_wallet = self.tabs.add("  Wallet  ")
        self._tab_net = self.tabs.add("  Network  ")
        self._tab_eco = self.tabs.add("  Ecosystem  ")
        self._tab_logs = self.tabs.add("  Logs  ")
        self._tab_settings = self.tabs.add("  Settings  ")

        # Delegate tab building to modules
        build_dashboard(self._tab_dash, self)
        build_validator(self._tab_val, self)
        build_wallet(self._tab_wallet, self)
        build_network(self._tab_net, self)
        build_ecosystem(self._tab_eco, self)
        build_logs(self._tab_logs, self)
        setup_log_handler(self)
        build_settings(self._tab_settings, self)

        # -- Status bar ------------------------------------------------
        sbar = ctk.CTkFrame(root, fg_color=COLORS["bg_darkest"], height=34,
                            corner_radius=0)
        sbar.pack(fill="x", side="bottom")
        sbar.pack_propagate(False)

        _sbar_font = (FONTS["mono"][0], 11)  # 11pt mono for readability
        self._sbar_text = ctk.CTkLabel(sbar, text="",
                                       font=_sbar_font,
                                       text_color=COLORS["text_muted"])
        self._sbar_text.pack(side="left", padx=16)

        self._sbar_source = ctk.CTkLabel(sbar, text="",
                                         font=_sbar_font,
                                         text_color=COLORS["accent_blue"])
        self._sbar_source.pack(side="left", padx=(8, 0))

        self._sbar_right = ctk.CTkLabel(sbar, text="",
                                        font=_sbar_font,
                                        text_color=COLORS["text_muted"])
        self._sbar_right.pack(side="right", padx=16)

    # -- Staking actions (kept on MainWindow for command= callbacks) ---

    def _do_stake(self):
        if self._action_in_progress:
            return
        # Use unlocked wallet (no password needed)
        if not self._active_address:
            self._stake_result.configure(
                text="Unlock your wallet first (Wallet tab \u2192 Sign In with Secret Key)",
                text_color=COLORS["danger"])
            return
        amount_str = self._stake_amount.get().strip()
        if not amount_str:
            self._stake_result.configure(text="Enter amount to stake",
                                         text_color=COLORS["danger"])
            return
        try:
            amount = float(amount_str)
        except ValueError:
            self._stake_result.configure(text="Invalid amount",
                                         text_color=COLORS["danger"])
            return

        # Confirmation dialog
        from positronic.app.ui.dialogs import ConfirmDialog
        dlg = ConfirmDialog(
            self.root,
            title="Confirm Stake",
            message=f"Stake {amount} ASF? Minimum lock period applies.",
            confirm_text="\u26a1 Stake",
            cancel_text="Cancel",
            confirm_color=COLORS["accent"],
            icon="\u26a1")
        if not dlg.result:
            return

        self._action_in_progress = True
        self._stake_result.configure(text="Submitting stake transaction...",
                                     text_color=COLORS["warning"])
        self.root.update_idletasks()
        import threading
        def _stake_bg():
            # Try both local AND remote so all nodes get the stake+pubkey
            result = {"error": "No RPC available"}
            for api in [self.api, self.remote_api]:
                try:
                    r = api.stake_with_key(amount, self._active_address,
                                            self._active_secret_key)
                    if r.get("success"):
                        result = r
                except Exception:
                    pass
            if not result.get("success"):
                # Fallback: try whichever didn't error
                for api in [self.remote_api, self.api]:
                    try:
                        r = api.stake_with_key(amount, self._active_address,
                                                self._active_secret_key)
                        if r and not r.get("error"):
                            result = r
                            break
                    except Exception:
                        pass
            def _update_ui():
                self._action_in_progress = False
                tx_hash = result.get('tx_hash', '')
                if result.get("success"):
                    self._stake_result.configure(
                        text=f"\u2713 Staked {amount} ASF!",
                        text_color=COLORS["success"])
                    if tx_hash:
                        from positronic.app.ui.dialogs import TxResultDialog
                        TxResultDialog(self.root, success=True,
                                       tx_hash=tx_hash,
                                       message=f"Staked {amount} ASF successfully!")
                else:
                    self._stake_result.configure(
                        text=f"Error: {result.get('error', 'Unknown')}",
                        text_color=COLORS["danger"])
            self.root.after(0, _update_ui)
        threading.Thread(target=_stake_bg, daemon=True).start()

    def _do_unstake(self):
        if self._action_in_progress:
            return
        if not self._active_address:
            self._stake_result.configure(
                text="Unlock your wallet first (Wallet tab)",
                text_color=COLORS["danger"])
            return

        # Confirmation dialog
        from positronic.app.ui.dialogs import ConfirmDialog
        dlg = ConfirmDialog(
            self.root,
            title="Confirm Unstake",
            message="Unstake your ASF? Unbonding period: 7 epochs.",
            confirm_text="\U0001f513 Unstake",
            cancel_text="Cancel",
            confirm_color=COLORS["warning"],
            icon="\U0001f513")
        if not dlg.result:
            return

        self._action_in_progress = True
        self._stake_result.configure(text="Submitting unstake...", text_color=COLORS["warning"])
        self.root.update_idletasks()
        import threading
        def _unstake_bg():
            # Get amount from input if provided, otherwise 0 (= unstake all)
            try:
                amt_str = self._stake_amount.get().strip()
                amt = float(amt_str) if amt_str else 0.0
            except (ValueError, AttributeError):
                amt = 0.0
            result = self.best_api.unstake_with_key(self._active_address,
                                                     self._active_secret_key,
                                                     amount_asf=amt)
            def _update():
                self._action_in_progress = False
                tx_hash = result.get('tx_hash', '')
                if result.get("success"):
                    self._stake_result.configure(text="\u2713 Unstake requested!", text_color=COLORS["success"])
                    if tx_hash:
                        from positronic.app.ui.dialogs import TxResultDialog
                        TxResultDialog(self.root, success=True,
                                       tx_hash=tx_hash,
                                       message="Unstake submitted successfully!")
                else:
                    self._stake_result.configure(text=f"Error: {result.get('error', 'Unknown')}", text_color=COLORS["danger"])
            self.root.after(0, _update)
        threading.Thread(target=_unstake_bg, daemon=True).start()

    def _bg_detect_validator(self):
        """Background: check if unlocked wallet is a validator."""
        try:
            addr = self._active_address
            if not addr:
                return
            api = self.remote_api
            staking = api.get_staking_info(addr)
            if not staking:
                return
            from positronic.constants import BASE_UNIT
            staked = staking.get("staked", 0) / BASE_UNIT
            is_val = staking.get("is_validator", False)
            available = staking.get("available", 0) / BASE_UNIT

            def _update():
                if is_val:
                    self._stake_result.configure(
                        text=f"\u2705 You are a Validator! Staked: {staked:,.2f} ASF",
                        text_color=COLORS["success"])
                elif available >= 32:
                    self._stake_result.configure(
                        text=f"\u26a1 You have {available:,.2f} ASF — Click 'Stake ASF' to become a validator!",
                        text_color=COLORS["warning"])
                else:
                    self._stake_result.configure(
                        text=f"Balance: {available:,.2f} ASF (need 32 ASF to stake)",
                        text_color=COLORS["text_dim"])
            self.root.after(0, _update)
        except Exception as e:
            logger.debug("Validator detect failed: %s", e)

    # -- Wallet actions ------------------------------------------------

    def _bg_scan_wallet(self):
        """Background thread scan — uses remote API for accurate data."""
        from positronic.app.rpc_client import RPCClient
        addr = self._wallet_addr.get().strip()
        if addr and not RPCClient.validate_address(addr):
            self.root.after(0, lambda: self._wallet_info.set("w_balance", "Invalid address"))
            return
        # Always use best_api (remote) for accurate balance
        result = self.best_api.scan_wallet(addr)
        # Also fetch tokens
        tokens = self.best_api.get_token_balances(addr) if addr else []
        self.root.after(0, lambda r=result, t=tokens: self._update_wallet_display(r, addr, t))

    def _update_wallet_display(self, result, addr, tokens=None):
        if result.get("error"):
            self._wallet_info.set("w_address", addr[:20] + "..." if len(addr) > 20 else addr)
            self._wallet_info.set("w_balance", f"Error: {result['error']}")
            self._wallet_info.set("w_txs", "--")
        else:
            self._wallet_info.set("w_address",
                                  result["address"][:20] + "..." if len(result["address"]) > 20 else result["address"])
            self._wallet_info.set("w_balance", result["balance_display"])
            self._wallet_info.set("w_txs", str(result["tx_count"]))
            # Update hero balance card (shows/updates the large balance display at top)
            try:
                from positronic.app.ui.tab_wallet import update_hero_balance
                update_hero_balance(self, result["address"],
                                    result.get("balance_display", "0"),
                                    tx_count=result.get("tx_count", 0))
            except Exception:
                pass
        # Update token list
        if hasattr(self, '_token_list_frame') and hasattr(self, '_token_empty_label'):
            for w in self._token_list_frame.winfo_children():
                if w != self._token_empty_label:
                    w.destroy()
            if tokens and len(tokens) > 0:
                self._token_empty_label.pack_forget()
                for t in tokens:
                    name = t.get("name", t.get("symbol", "?"))
                    bal = t.get("balance", 0)
                    row = ctk.CTkFrame(self._token_list_frame, fg_color="transparent")
                    row.pack(fill="x", pady=2)
                    ctk.CTkLabel(row, text=f"{name}: {bal}",
                                 font=FONTS["body"],
                                 text_color=COLORS["text"]).pack(side="left")
            else:
                self._token_empty_label.pack()

    def _do_create_wallet(self):
        """Generate new wallet — show 64 hex secret key (no password)."""
        import secrets
        from positronic.crypto.keys import KeyPair
        # Generate random 32-byte seed = 64 hex secret key
        seed = secrets.token_bytes(32)
        secret_key_hex = seed.hex()
        try:
            kp = KeyPair.from_seed(seed)
            addr = kp.address_hex
        except Exception as e:
            self._wallet_result.configure(
                text=f"Error generating wallet: {e}",
                text_color=COLORS["danger"])
            return
        # Show the secret key (Entry widget — selectable + Copy button)
        self._new_key_frame.pack(fill="x", padx=16, pady=(8, 4))
        self._new_key_label.configure(state="normal")
        self._new_key_label.delete(0, "end")
        self._new_key_label.insert(0, secret_key_hex)
        self._new_key_label.configure(state="readonly")
        self._new_key_warning.pack(padx=12, pady=(4, 12))
        self._wallet_result.configure(
            text=f"Wallet created: {addr}  |  SAVE YOUR SECRET KEY ABOVE!",
            text_color=COLORS["success"])
        # Save wallet keystore to data directory
        try:
            import os, json
            data_dir = getattr(self.api, 'data_dir', None) or os.path.join(
                os.path.expanduser("~"), ".positronic")
            keystore_dir = os.path.join(data_dir, "wallets")
            os.makedirs(keystore_dir, exist_ok=True)
            filepath = os.path.join(keystore_dir, f"{addr}.json")
            keystore_data = {"address": addr, "created": True}
            with open(filepath, "w") as f:
                json.dump(keystore_data, f)
        except Exception:
            pass  # Best-effort save — user still has the secret key displayed
        # Also scan the new address (background)
        self._wallet_addr.delete(0, "end")
        self._wallet_addr.insert(0, addr)
        import threading
        threading.Thread(target=self._bg_scan_wallet, daemon=True).start()

    def _do_unlock_wallet(self):
        """Unlock wallet with 64 hex secret key — derive address and store in memory."""
        key_hex = self._login_key.get().strip()
        # Quick-unlock: if session is locked, accept first 8 chars to resume
        if (hasattr(self, '_session_locked_key') and self._session_locked_key
                and len(key_hex) == 8):
            if self._session_locked_key[:8] == key_hex:
                self._active_secret_key = self._session_locked_key
                self._session_locked_key = None
                self._login_result.configure(
                    text=f"✓ Quick-unlocked: {self._active_address}",
                    text_color=COLORS["success"])
                self._login_key.delete(0, "end")
                self._reset_session_timer()
                # Re-scan in background
                import threading
                threading.Thread(target=self._bg_scan_wallet, daemon=True).start()
                return
            else:
                self._login_result.configure(
                    text="✗ Quick-unlock failed — enter full 64-char key",
                    text_color=COLORS["danger"])
                self._session_locked_key = None
                self._delete_wallet_session()
                return
        if not key_hex or len(key_hex) != 64:
            self._login_result.configure(
                text="Secret key must be exactly 64 hex characters",
                text_color=COLORS["danger"])
            return
        try:
            int(key_hex, 16)  # Validate hex
        except ValueError:
            self._login_result.configure(
                text="Invalid key — only hex characters (0-9, a-f)",
                text_color=COLORS["danger"])
            return
        try:
            from positronic.crypto.keys import KeyPair
            seed = bytes.fromhex(key_hex)
            kp = KeyPair.from_seed(seed)
            addr = kp.address_hex
            # Store in memory for transactions
            self._active_secret_key = key_hex
            self._active_address = addr
            # Persist session to encrypted file
            self._save_wallet_session()
            # NOTE: plaintext keypair write removed (security risk).
            # Node generates its own keypair; wallet key stays in memory only.
            self._login_result.configure(
                text=f"Unlocked: {addr}",
                text_color=COLORS["success"])
            # Clear key from input
            self._login_key.delete(0, "end")
            # Auto-scan the address (in background thread to avoid UI freeze)
            self._wallet_addr.delete(0, "end")
            self._wallet_addr.insert(0, addr)
            import threading
            threading.Thread(target=self._bg_scan_wallet, daemon=True).start()
            # Auto-detect validator status
            threading.Thread(target=self._bg_detect_validator, daemon=True).start()
        except Exception as e:
            self._login_result.configure(
                text=f"Invalid key: {e}",
                text_color=COLORS["danger"])

    # -- Claim rewards -------------------------------------------------

    def _do_claim_rewards(self):
        if self._action_in_progress:
            return
        if not self._active_address:
            self._stake_result.configure(
                text="Unlock your wallet first (Wallet tab)",
                text_color=COLORS["danger"])
            return

        # Get current pending rewards text from the button for display
        pending_text = ""
        if hasattr(self, '_claim_btn'):
            btn_text = self._claim_btn.cget("text")
            if "(" in btn_text:
                pending_text = btn_text.split("(")[-1].rstrip(")")

        # Confirmation dialog
        from positronic.app.ui.dialogs import ConfirmDialog
        msg = f"Claim {pending_text} rewards?" if pending_text else "Claim all pending rewards?"
        dlg = ConfirmDialog(
            self.root,
            title="Confirm Claim",
            message=msg,
            confirm_text="\U0001f381 Claim",
            cancel_text="Cancel",
            confirm_color=COLORS["success"],
            icon="\U0001f381")
        if not dlg.result:
            return

        self._action_in_progress = True
        self._stake_result.configure(text="Claiming rewards...",
                                     text_color=COLORS["warning"])
        self.root.update_idletasks()
        import threading
        def _claim_bg():
            # Try remote first (has latest state), fallback to local
            result = {"error": "No RPC"}
            for api in [self.remote_api, self.api]:
                try:
                    r = api.claim_rewards_with_key(self._active_address,
                                                    self._active_secret_key)
                    if r and (r.get("success") or not r.get("error")):
                        result = r
                        break
                except Exception:
                    pass
            def _update():
                self._action_in_progress = False
                tx_hash = result.get('tx_hash', '')
                if result.get("success"):
                    claimed = result.get("claimed", 0)
                    if isinstance(claimed, (int, float)) and claimed > 0:
                        from positronic.constants import BASE_UNIT
                        claimed_asf = claimed / BASE_UNIT if claimed > 1e15 else claimed
                        msg = f"Claimed {claimed_asf:,.4f} ASF!"
                    else:
                        msg = "Rewards claimed!"
                    self._stake_result.configure(
                        text=f"\u2713 {msg}",
                        text_color=COLORS["success"])
                    if tx_hash:
                        from positronic.app.ui.dialogs import TxResultDialog
                        TxResultDialog(self.root, success=True,
                                       tx_hash=tx_hash, message=msg)
                else:
                    self._stake_result.configure(
                        text=f"Error: {result.get('error', 'Unknown')}",
                        text_color=COLORS["danger"])
            self.root.after(0, _update)
        threading.Thread(target=_claim_bg, daemon=True).start()

    # -- Export key dialog ---------------------------------------------

    def _do_export_key(self):
        if not self._active_secret_key:
            return
        # Require user to re-enter first 8 chars of key as confirmation
        import customtkinter as ctk_confirm
        dialog = ctk_confirm.CTkInputDialog(
            text="Enter the first 8 characters of your key to confirm export:",
            title="Confirm Key Export")
        confirm = dialog.get_input()
        if not confirm or confirm.strip() != self._active_secret_key[:8]:
            return  # Wrong confirmation — abort
        from positronic.app.ui.dialogs import ExportKeyDialog
        ExportKeyDialog(self.root, self.api, active_key=self._active_secret_key)

    # -- Send transfer -------------------------------------------------

    def _do_send_transfer(self):
        if self._action_in_progress:
            return
        if not self._active_address or not self._active_secret_key:
            self._send_result.configure(
                text="Unlock your wallet first (Wallet tab)",
                text_color=COLORS["danger"])
            return
        from_addr = self._active_address
        to_addr = self._send_to.get().strip()
        amount_str = self._send_amount.get().strip()

        if not to_addr or not amount_str:
            self._send_result.configure(
                text="Please fill in recipient and amount",
                text_color=COLORS["danger"])
            return
        try:
            amount = float(amount_str)
        except ValueError:
            self._send_result.configure(
                text="Invalid amount", text_color=COLORS["danger"])
            return

        # Confirmation dialog before sending
        from positronic.app.ui.dialogs import ConfirmDialog
        short_addr = f"{to_addr[:8]}...{to_addr[-6:]}" if len(to_addr) > 20 else to_addr
        dlg = ConfirmDialog(
            self.root,
            title="Confirm Transfer",
            message=f"Send {amount} ASF to {short_addr}?",
            confirm_text="Confirm",
            cancel_text="Cancel",
            confirm_color=COLORS["accent"],
            icon="\U0001f4e4")
        if not dlg.result:
            return

        self._action_in_progress = True
        self._send_result.configure(text="Sending...",
                                     text_color=COLORS["warning"])
        self.root.update_idletasks()
        key = self._active_secret_key
        import threading
        def _send_transfer_bg():
            result = self.api.send_transfer_with_key(
                from_addr, to_addr, amount, key)
            def _update():
                self._action_in_progress = False
                tx_hash = result.get('tx_hash', '')
                if result.get("success"):
                    self._send_result.configure(
                        text=f"\u2713 Sent! TX: {tx_hash[:20]}..." if tx_hash else "\u2713 Sent!",
                        text_color=COLORS["success"])
                    # Show TX result dialog with copyable hash
                    if tx_hash:
                        from positronic.app.ui.dialogs import TxResultDialog
                        TxResultDialog(self.root, success=True,
                                       tx_hash=tx_hash,
                                       message=f"Sent {amount} ASF successfully!")
                else:
                    self._send_result.configure(
                        text=f"Error: {result.get('error', 'Unknown')}",
                        text_color=COLORS["danger"])
            self.root.after(0, _update)
        threading.Thread(target=_send_transfer_bg, daemon=True).start()

    # -- Create token ---------------------------------------------------

    def _do_create_token(self):
        if not self._active_address or not self._active_secret_key:
            return  # Wallet not unlocked
        from positronic.app.ui.dialogs import CreateTokenDialog
        CreateTokenDialog(self.root, self.api,
                          self._active_address, self._active_secret_key)

    # -- Log clearing --------------------------------------------------

    def _clear_logs(self):
        self._log_text.configure(state=tk.NORMAL)
        self._log_text.delete("1.0", tk.END)
        self._log_text.configure(state=tk.DISABLED)

    # -- Refresh (called every 2s by app.py) ---------------------------

    def refresh(self):
        """Legacy blocking refresh -- fetch + update on main thread.
        Prefer refresh_with_data() for threaded polling."""
        # Use remote API for data (has latest synced state)
        _api = self.remote_api
        data = _api.get_dashboard()
        vi = _api.get_validator_info()
        net = _api.get_network()
        em = _api.get_emergency_status()
        self.refresh_with_data(data, vi, net, em)

    def show_update_banner(self, update_info: dict):
        """Show a non-intrusive update notification banner at the top."""
        import webbrowser

        banner = ctk.CTkFrame(self.root, fg_color="#1a5276", height=40,
                              corner_radius=0)
        banner.pack(fill="x", side="top", before=self.root.winfo_children()[0])
        banner.pack_propagate(False)

        ctk.CTkLabel(
            banner,
            text=f"Update available: v{update_info.get('latest', '?')} "
                 f"(current: v{update_info.get('current', '?')})",
            font=("Segoe UI", 12),
            text_color="#FFFFFF",
        ).pack(side="left", padx=(16, 8))

        ctk.CTkButton(
            banner, text="Download", font=("Segoe UI", 11, "bold"),
            fg_color="#2980b9", hover_color="#3498db",
            text_color="#FFFFFF", width=90, height=28,
            corner_radius=6,
            command=lambda: webbrowser.open(
                update_info.get("download_url", "https://positronic-ai.network/download")
            ),
        ).pack(side="left", padx=4)

        ctk.CTkButton(
            banner, text="X", font=("Segoe UI", 11),
            fg_color="transparent", hover_color="#1a5276",
            text_color="#FFFFFF", width=30, height=28,
            command=banner.destroy,
        ).pack(side="right", padx=8)

    def refresh_with_data(self, data, vi, net, em=None, eco=None):
        """Update all visible widgets from pre-fetched data.
        This runs ONLY on the main thread -- no RPC calls here."""
        # Hide tab-switch loading indicator now that fresh data arrived
        self._hide_tab_loading()

        if data is None:
            data = {"online": False, "uptime": ""}
        if vi is None:
            vi = {"online": False}
        if net is None:
            net = {"online": False}

        online = data.get("online", False)
        connecting = data.get("connecting", False)

        # Header status -- five states:
        #   Online     (green)  = local node connected to peers, fully synced
        #   Syncing    (cyan)   = connected, catching up with chain
        #   Connecting (yellow) = node running, searching for peers
        #   Remote     (blue)   = showing testnet data via remote RPC
        #   Offline    (red)    = no RPC responding at all
        is_remote = data.get("_remote", False)
        syncing = data.get("syncing", False)
        node_status = data.get("_node_status", "")

        if is_remote and online:
            self._status_dot.configure(text_color=COLORS["accent_blue"])
            label = "Testnet (Remote)"
            if node_status:
                label += f" \u2014 {node_status}"
            self._status_text.configure(text=label,
                                        text_color=COLORS["accent_blue"])
        elif online and syncing:
            self._status_dot.configure(text_color=COLORS.get("info_blue","#00E5FF"))
            sync_pct = data.get("sync_percent", "")
            sync_label = f"Syncing {sync_pct}%" if sync_pct else "Syncing..."
            self._status_text.configure(text=sync_label,
                                        text_color=COLORS.get("info_blue","#00E5FF"))
        elif online:
            self._status_dot.configure(text_color=COLORS["success"])
            self._status_text.configure(text="Online",
                                        text_color=COLORS["success"])
        elif connecting:
            self._status_dot.configure(text_color=COLORS["warning"])
            self._status_text.configure(text="Connecting...",
                                        text_color=COLORS["warning"])
        else:
            self._status_dot.configure(text_color=COLORS["danger"])
            self._status_text.configure(text="Offline",
                                        text_color=COLORS["danger"])

        # Delegate per-tab refreshes
        refresh_dashboard(self, data)
        refresh_validator(self, vi)
        refresh_network(self, net)

        # Emergency banner
        if em is None:
            em = {}
        em_state = em.get("state", 0)
        em_name = em.get("state_name", "NORMAL")
        em_reason = em.get("reason", "")

        if em_state >= 2:  # PAUSED, HALTED, or UPGRADING
            if em_state == 3:  # HALTED
                bg = "#3a0000"
                fg = COLORS["danger"]
                icon = "\U0001f6d1"
                label = "Network HALTED"
            elif em_state == 4:  # UPGRADING
                bg = "#001a2a"
                fg = COLORS["accent"]
                icon = "\u2b06"
                label = "Network UPGRADING"
            else:  # PAUSED
                bg = "#2a1a00"
                fg = COLORS["warning"]
                icon = "\u26a0"
                label = "Network PAUSED"

            self._emergency_banner.configure(fg_color=bg)
            self._emergency_icon.configure(text=icon, text_color=fg)
            self._emergency_text.configure(text=label, text_color=fg)
            self._emergency_detail.configure(
                text=f"\u2014 {em_reason}" if em_reason else "")
            self._emergency_banner.pack(fill="x", padx=0, pady=0,
                                        before=self.tabs)

            # Update header status to reflect emergency
            self._status_dot.configure(text_color=fg)
            self._status_text.configure(text=em_name, text_color=fg)

            # Log state change once
            if em_state != self._last_emergency_state:
                logger.warning("Network state changed to %s: %s",
                               em_name, em_reason)
        else:
            # Hide banner
            self._emergency_banner.pack_forget()

        self._last_emergency_state = em_state

        # Status bar — format: "Block #12,345 | 3 peers | TESTNET | Node: running"
        import time as _time
        self._last_refresh_ts = _time.time()

        h = data.get("block_height", 0)
        peers = data.get("peers", 0)
        nt = data.get("network_type", "mainnet")
        em_suffix = f"  |  \u26a0 {em_name}" if em_state >= 2 else ""
        node_state_str = data.get("_node_status", "running")
        if not node_state_str or node_state_str == "":
            node_state_str = "running"
        self._sbar_text.configure(
            text=f"Block #{h:,}  |  {peers} peers  |  {nt.upper()}{em_suffix}  |  Node: {node_state_str}")

        # Source indicator (remote vs local)
        if is_remote:
            self._sbar_source.configure(text="[Remote]",
                                        text_color=COLORS["accent_blue"])
        else:
            self._sbar_source.configure(text="[Local]",
                                        text_color=COLORS["success"])

        self._sbar_right.configure(text=data.get("uptime", ""))

        # Ecosystem tab
        if eco:
            refresh_ecosystem(self, eco)

        # Update TX history if wallet is unlocked
        if self._active_address and hasattr(self, '_tx_list_frame'):
            txs = data.get("_tx_history", [])
            if txs:
                from positronic.app.ui.tab_wallet import _build_tx_rows
                _build_tx_rows(self, txs)

    # -- Wallet session persistence ----------------------------------------

    def _get_session_path(self) -> str:
        """Path to encrypted wallet session file."""
        data_dir = getattr(self.api, '_data_dir', None) or os.path.join(
            os.path.expanduser("~"), ".positronic")
        return os.path.join(data_dir, "wallet_session.enc")

    def _save_wallet_session(self):
        """Encrypt and save current wallet session to disk."""
        if not self._active_secret_key or not self._active_address:
            return
        try:
            from positronic.crypto.data_encryption import DataEncryptor
            path = self._get_session_path()
            data_dir = os.path.dirname(path)
            os.makedirs(data_dir, exist_ok=True)
            enc = DataEncryptor(data_dir)
            session_data = {
                "secret_key": self._active_secret_key,
                "address": self._active_address,
                "timestamp": time.time(),
            }
            enc.save_json(path, session_data)
            logger.debug("Wallet session saved for %s", self._active_address[:12])
        except Exception as e:
            logger.debug("Failed to save wallet session: %s", e)

    def _load_wallet_session(self) -> dict:
        """Load and decrypt wallet session. Returns {} if missing/expired/corrupt."""
        path = self._get_session_path()
        if not os.path.isfile(path):
            return {}
        try:
            from positronic.crypto.data_encryption import DataEncryptor
            data_dir = os.path.dirname(path)
            enc = DataEncryptor(data_dir)
            session = enc.load_json(path)
            if not session:
                return {}
            # Check 24-hour expiry
            ts = session.get("timestamp", 0)
            if time.time() - ts > 86400:
                self._delete_wallet_session()
                return {}
            return session
        except Exception as e:
            logger.debug("Failed to load wallet session: %s", e)
            return {}

    def _delete_wallet_session(self):
        """Remove wallet session file from disk."""
        try:
            path = self._get_session_path()
            if os.path.isfile(path):
                os.remove(path)
                logger.debug("Wallet session file deleted")
        except Exception as e:
            logger.debug("Failed to delete wallet session: %s", e)

    def _try_restore_session(self):
        """Attempt to restore wallet session from encrypted file on startup."""
        session = self._load_wallet_session()
        if not session:
            return
        key_hex = session.get("secret_key", "")
        addr = session.get("address", "")
        if not key_hex or not addr or len(key_hex) != 64:
            return
        try:
            # Validate key is still valid
            from positronic.crypto.keys import KeyPair
            seed = bytes.fromhex(key_hex)
            kp = KeyPair.from_seed(seed)
            if kp.address_hex != addr:
                logger.warning("Session address mismatch — discarding")
                self._delete_wallet_session()
                return
            # Restore session
            self._active_secret_key = key_hex
            self._active_address = addr
            # Update UI elements directly (no background threads —
            # the normal poll cycle in app.py will pick up _active_address
            # and fetch staking info automatically)
            if hasattr(self, '_login_result'):
                self._login_result.configure(
                    text=f"✓ Session restored: {addr[:16]}...",
                    text_color=COLORS["success"])
            if hasattr(self, '_wallet_addr'):
                self._wallet_addr.delete(0, "end")
                self._wallet_addr.insert(0, addr)
            logger.info("Wallet session restored for %s", addr[:16])
        except Exception as e:
            logger.debug("Session restore failed: %s", e)
            self._delete_wallet_session()

    def _do_lock_wallet(self):
        """Explicitly lock wallet — clears all keys and session."""
        self._active_secret_key = None
        self._active_address = None
        if hasattr(self, '_session_locked_key'):
            self._session_locked_key = None
        self._delete_wallet_session()
        # Update UI
        if hasattr(self, '_login_result'):
            self._login_result.configure(
                text="🔒 Wallet locked", text_color=COLORS["text_muted"])
        if hasattr(self, '_hero_frame'):
            self._hero_frame.pack_forget()
        if hasattr(self, '_tx_history_card'):
            self._tx_history_card.pack_forget()
        if hasattr(self, '_stake_wallet_status'):
            self._stake_wallet_status.configure(
                text="⚠ Wallet not connected", text_color=COLORS["warning"])
        logger.info("Wallet explicitly locked by user")

    # -- Clipboard paste -----------------------------------------------

    def _paste_clipboard(self, event):
        """Handle Ctrl+V paste for Entry widgets."""
        try:
            w = event.widget
            text = w.clipboard_get()
            try:
                w.delete("sel.first", "sel.last")
            except Exception:
                pass
            w.insert("insert", text)
        except Exception:
            pass
        return "break"

    def _select_all(self, event):
        """Handle Ctrl+A select all for Entry widgets."""
        try:
            w = event.widget
            w.select_range(0, "end")
            w.icursor("end")
        except Exception:
            pass
        return "break"

    # -- Session timeout -----------------------------------------------

    def _setup_session_timeout(self):
        """Bind user activity events to reset the inactivity timer."""
        for event in ("<Button>", "<Key>", "<Motion>"):
            self.root.bind_all(event, self._reset_session_timer, add="+")
        # Start initial timer (only matters once wallet is unlocked)
        self._reset_session_timer()

    def _reset_session_timer(self, _event=None):
        """Reset the 30-minute session timeout."""
        if self._session_timer_id is not None:
            self.root.after_cancel(self._session_timer_id)
        self._session_timer_id = self.root.after(
            self._session_timeout_ms, self._session_expired)

    def _session_expired(self):
        """Auto-lock wallet after 30 minutes of inactivity — allows quick-unlock."""
        if not self._active_secret_key:
            return  # Nothing to lock
        # Keep key for quick-unlock (first 8 chars to re-verify)
        self._session_locked_key = self._active_secret_key
        self._active_secret_key = None
        # Keep _active_address for display but mark as locked
        # Update UI to reflect locked state
        if hasattr(self, '_login_result'):
            self._login_result.configure(
                text="🔒 Session locked — enter first 8 chars to quick-unlock",
                text_color=COLORS["warning"])
        if hasattr(self, '_stake_wallet_status'):
            self._stake_wallet_status.configure(
                text="🔒 Session locked — unlock wallet to continue",
                text_color=COLORS["warning"])
        logger.info("Wallet session locked due to inactivity (30 min)")

    # -- Tab switching with loading indicator -------------------------

    def _switch_tab(self, idx: int):
        """Switch to tab by index (0-based). Show brief loading overlay if data is stale."""
        if idx < 0 or idx >= len(self._tab_names):
            return
        tab_name = self._tab_names[idx]
        try:
            self.tabs.set(tab_name)
        except Exception:
            return

        # If data is older than 10s, show a brief "Refreshing..." overlay
        import time as _time
        stale = (_time.time() - self._last_refresh_ts) > 10.0
        if stale:
            self._show_tab_loading()

    def _show_tab_loading(self):
        """Show a brief 'Refreshing...' label that auto-hides after data arrives."""
        if self._tab_loading_label is not None:
            try:
                self._tab_loading_label.destroy()
            except Exception:
                pass
        self._tab_loading_label = ctk.CTkLabel(
            self.tabs, text="Refreshing...",
            font=FONTS["small"], text_color=COLORS["accent"],
            fg_color=COLORS["bg_card"], corner_radius=6,
            width=120, height=28)
        self._tab_loading_label.place(relx=0.5, rely=0.5, anchor="center")
        # Auto-hide after 2 seconds (poll will refresh sooner)
        self.root.after(2000, self._hide_tab_loading)

    def _hide_tab_loading(self):
        if self._tab_loading_label is not None:
            try:
                self._tab_loading_label.destroy()
            except Exception:
                pass
            self._tab_loading_label = None

    def _request_refresh(self):
        """Trigger an immediate data refresh (Ctrl+R / F5)."""
        self._show_tab_loading()
        # The app.py poll cycle runs on a timer; trigger legacy refresh as fallback
        try:
            import threading
            threading.Thread(target=self.refresh, daemon=True).start()
        except Exception:
            pass

    # -- Helpers -------------------------------------------------------

    def _find_icon(self) -> str | None:
        """Find logo or fallback icon."""
        candidates = [
            os.path.join(os.path.dirname(__file__), "logo.jpg"),
            os.path.join(os.path.dirname(__file__), "brain-icon.png"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))), "LOGO.jpg"),
        ]
        # Handle frozen app
        if getattr(sys, "frozen", False):
            candidates.insert(0, os.path.join(
                sys._MEIPASS, "positronic", "app", "ui", "brain-icon.png"))
        for p in candidates:
            if os.path.isfile(p):
                return p
        return None
