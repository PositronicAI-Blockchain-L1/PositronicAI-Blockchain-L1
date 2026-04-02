"""Modal dialogs — ConfirmDialog, TxResultDialog, ExportKeyDialog, CreateTokenDialog."""

import customtkinter as ctk

from positronic.app.theme import COLORS, FONTS, _EMOJI


def _scaled_geometry(parent, w: int, h: int) -> str:
    """Return a geometry string scaled for high-DPI screens.
    Uses parent screen width to detect DPI factor (e.g. 1.25x on 150% scale)."""
    try:
        sw = parent.winfo_screenwidth()
        # Standard baseline is 1920px; wider = higher DPI factor
        scale = max(1.0, sw / 1920)
        return f"{int(w * scale)}x{int(h * scale)}"
    except Exception:
        return f"{w}x{h}"


class ConfirmDialog(ctk.CTkToplevel):
    """Reusable modal confirmation dialog. Returns True/False via .result."""

    def __init__(self, parent, title="Confirm", message="Are you sure?",
                 confirm_text="Confirm", cancel_text="Cancel",
                 confirm_color=None, icon=""):
        super().__init__(parent)
        self.title(title)
        self.geometry(_scaled_geometry(parent, 460, 280))
        self.resizable(False, True)
        self.configure(fg_color=COLORS["bg_dark"])
        self.grab_set()
        self.focus_force()
        self.result = False

        # Icon + message
        msg_frame = ctk.CTkFrame(self, fg_color="transparent")
        msg_frame.pack(fill="x", padx=24, pady=(28, 16))

        if icon:
            ctk.CTkLabel(msg_frame, text=icon, font=(_EMOJI, 28),
                         text_color=COLORS["warning"]).pack(side="left", padx=(0, 12))

        ctk.CTkLabel(msg_frame, text=message, font=FONTS["body"],
                     text_color=COLORS["text"], wraplength=340,
                     justify="left").pack(side="left", fill="x", expand=True)

        # Warning note
        ctk.CTkLabel(self, text="This action cannot be undone.",
                     font=FONTS["small"],
                     text_color=COLORS["text_muted"]).pack(pady=(0, 16))

        # Buttons
        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(pady=(0, 20))

        ctk.CTkButton(btn_row, text=cancel_text, font=FONTS["button"],
                      fg_color=COLORS["bg_card"], text_color=COLORS["text"],
                      hover_color=COLORS["bg_card_hover"],
                      corner_radius=8, width=120, height=40,
                      command=self._cancel).pack(side="left", padx=(0, 12))

        ctk.CTkButton(btn_row, text=confirm_text, font=FONTS["button"],
                      fg_color=confirm_color or COLORS["accent"],
                      text_color="#000000",
                      hover_color=COLORS["accent_blue"],
                      corner_radius=8, width=120, height=40,
                      command=self._confirm).pack(side="left")

        # Center on parent
        self.transient(parent)
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.wait_window()

    def _confirm(self):
        self.result = True
        self.destroy()

    def _cancel(self):
        self.result = False
        self.destroy()


class TxResultDialog(ctk.CTkToplevel):
    """Show a transaction result with copyable hash."""

    def __init__(self, parent, success=True, tx_hash="", message=""):
        super().__init__(parent)
        self.title("Transaction Result")
        self.geometry(_scaled_geometry(parent, 500, 200))
        self.resizable(False, False)
        self.configure(fg_color=COLORS["bg_dark"])
        self.grab_set()
        self.focus_force()

        icon = "\u2705" if success else "\u274c"
        color = COLORS["success"] if success else COLORS["danger"]

        ctk.CTkLabel(self, text=f"{icon}  {message}",
                     font=FONTS["subheading"], text_color=color,
                     wraplength=450).pack(padx=20, pady=(24, 12))

        if tx_hash:
            hash_row = ctk.CTkFrame(self, fg_color="transparent")
            hash_row.pack(fill="x", padx=20, pady=(0, 8))

            self._hash_entry = ctk.CTkEntry(
                hash_row, font=("Cascadia Code", 11),
                fg_color=COLORS["bg_darkest"], border_color=COLORS["border"],
                text_color=COLORS["accent"])
            self._hash_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
            self._hash_entry.insert(0, tx_hash)
            self._hash_entry.configure(state="readonly")

            self._copy_btn = ctk.CTkButton(
                hash_row, text="\U0001f4cb Copy", font=FONTS["small"],
                fg_color=COLORS["accent"], text_color="#000000",
                hover_color=COLORS["accent_blue"],
                corner_radius=6, width=80, height=32,
                command=lambda: self._copy_hash(tx_hash))
            self._copy_btn.pack(side="right")

        ctk.CTkButton(self, text="Close", font=FONTS["button"],
                      fg_color=COLORS["bg_card"], text_color=COLORS["text"],
                      hover_color=COLORS["bg_card_hover"],
                      corner_radius=8, width=100, height=34,
                      command=self.destroy).pack(pady=(8, 16))

        self.transient(parent)

    def _copy_hash(self, tx_hash):
        try:
            self.clipboard_clear()
            self.clipboard_append(tx_hash)
            self._copy_btn.configure(text="\u2705 Copied!")
            self.after(2000, lambda: self._copy_btn.configure(text="\U0001f4cb Copy"))
        except Exception:
            pass


class ExportKeyDialog(ctk.CTkToplevel):
    """Modal dialog for showing the active secret key."""

    def __init__(self, parent, api, active_key=None):
        super().__init__(parent)
        self.title("Export Private Key")
        self.geometry(_scaled_geometry(parent, 520, 350))
        self.resizable(False, False)
        self.configure(fg_color=COLORS["bg_dark"])
        self.grab_set()
        self.focus_force()

        self._timer_id = None
        self._active_key = active_key

        # Security warning
        warn_frame = ctk.CTkFrame(self, fg_color=COLORS["bg_darkest"], corner_radius=8)
        warn_frame.pack(fill="x", padx=16, pady=(16, 8))
        ctk.CTkLabel(warn_frame,
                     text="\u26a0 WARNING: Never share your private key!",
                     font=FONTS["subheading"], text_color=COLORS["warning"]).pack(
            padx=12, pady=8)
        ctk.CTkLabel(warn_frame,
                     text="Anyone with your private key can steal your funds.\n"
                          "The key will auto-clear after 30 seconds.",
                     font=FONTS["small"], text_color=COLORS["text_dim"]).pack(
            padx=12, pady=(0, 8))

        if active_key:
            # Key is available — show it directly
            ctk.CTkLabel(self, text="Your Secret Key (64 hex):",
                         font=FONTS["body"],
                         text_color=COLORS["text_dim"]).pack(
                anchor="w", padx=16, pady=(12, 4))

            self._key_display = ctk.CTkEntry(
                self, font=("Cascadia Code", 11), width=460,
                fg_color=COLORS["bg_darkest"], border_color=COLORS["accent"],
                text_color=COLORS["accent"])
            self._key_display.pack(padx=16, pady=(0, 4))
            self._key_display.insert(0, active_key)
            self._key_display.configure(state="disabled")

            # Copy + countdown
            btn_row = ctk.CTkFrame(self, fg_color="transparent")
            btn_row.pack(pady=8)
            ctk.CTkButton(btn_row, text="Copy", font=FONTS["button"],
                          fg_color=COLORS["success"], text_color="#000000",
                          hover_color="#00cc66", corner_radius=8,
                          width=80, height=32,
                          command=self._copy).pack(side="left", padx=4)
            self._countdown = ctk.CTkLabel(btn_row, text="",
                                            font=FONTS["small"],
                                            text_color=COLORS["text_muted"])
            self._countdown.pack(side="left", padx=8)

            # Start 30s countdown
            self._start_countdown(30)
        else:
            # No key available
            ctk.CTkLabel(self,
                         text="\u26a0 No wallet unlocked!\n\n"
                              "Go to Wallet tab \u2192 Sign In with Secret Key\n"
                              "Then come back here.",
                         font=FONTS["body"], text_color=COLORS["warning"],
                         wraplength=400, justify="center").pack(
                pady=40)

    def _copy(self):
        try:
            val = self._key_display.get()
            if val:
                self.clipboard_clear()
                self.clipboard_append(val)
                self._countdown.configure(
                    text="Copied!", text_color=COLORS["success"])
        except Exception:
            pass

    def _start_countdown(self, seconds):
        if seconds <= 0:
            self._key_display.configure(state="normal")
            self._key_display.delete(0, "end")
            self._key_display.insert(0, "*** CLEARED ***")
            self._key_display.configure(state="disabled")
            self._countdown.configure(text="Key cleared for security",
                                       text_color=COLORS["danger"])
            return
        self._countdown.configure(text=f"Auto-clear in {seconds}s")
        self._timer_id = self.after(1000, self._start_countdown, seconds - 1)

    def destroy(self):
        if self._timer_id:
            self.after_cancel(self._timer_id)
        # Clear key reference from memory (best-effort; Python strings are immutable)
        self._active_key = None
        try:
            import gc
            gc.collect()
        except Exception:
            pass
        super().destroy()


class CreateTokenDialog(ctk.CTkToplevel):
    """Modal dialog for creating a PRC-20 token."""

    def __init__(self, parent, api, creator_addr="", secret_key=""):
        super().__init__(parent)
        self.api = api
        self._creator_addr = creator_addr
        self._secret_key = secret_key
        self.title("Create PRC-20 Token")
        self.geometry(_scaled_geometry(parent, 460, 380))
        self.resizable(False, False)
        self.configure(fg_color=COLORS["bg_dark"])
        self.grab_set()
        self.focus_force()

        ctk.CTkLabel(self, text="\U0001fa99 Create New Token",
                     font=FONTS["heading"],
                     text_color=COLORS["text"]).pack(pady=(16, 12))

        fields = [
            ("Name:", "name", "Token name"),
            ("Symbol:", "symbol", "TKN"),
            ("Total Supply:", "supply", "1000000"),
        ]
        self._entries = {}
        for label, key, placeholder in fields:
            ctk.CTkLabel(self, text=label, font=FONTS["body"],
                         text_color=COLORS["text_dim"]).pack(
                anchor="w", padx=16, pady=(4, 0))
            e = ctk.CTkEntry(self, placeholder_text=placeholder,
                             font=FONTS["input"], width=400,
                             fg_color=COLORS["bg_dark"],
                             border_color=COLORS["border"],
                             text_color=COLORS["text"])
            e.pack(padx=16, pady=(0, 4))
            self._entries[key] = e

        ctk.CTkButton(self, text="Create Token", font=FONTS["button"],
                      fg_color=COLORS["accent"], text_color="#000000",
                      hover_color=COLORS["accent_blue"],
                      corner_radius=8, height=40, width=160,
                      command=self._create).pack(pady=16)

        self._result = ctk.CTkLabel(self, text="", font=FONTS["small"],
                                     text_color=COLORS["text_dim"])
        self._result.pack()

    def _create(self):
        name = self._entries["name"].get().strip()
        symbol = self._entries["symbol"].get().strip()
        supply = self._entries["supply"].get().strip()
        if not name or not symbol or not supply:
            self._result.configure(text="Fill in all fields",
                                    text_color=COLORS["danger"])
            return
        if not symbol.isalnum() or len(symbol) > 10:
            self._result.configure(text="Symbol must be alphanumeric, max 10 chars",
                                    text_color=COLORS["danger"])
            return
        try:
            supply_int = int(supply)
        except ValueError:
            self._result.configure(text="Supply must be a valid number",
                                    text_color=COLORS["danger"])
            return
        if supply_int <= 0:
            self._result.configure(text="Supply must be greater than 0",
                                    text_color=COLORS["danger"])
            return
        try:
            result = self.api.create_token_with_key(
                name, symbol, supply_int, self._creator_addr, self._secret_key)
            if result.get("error"):
                self._result.configure(text=f"Error: {result['error']}",
                                        text_color=COLORS["danger"])
            else:
                self._result.configure(
                    text=f"Token created: {symbol}",
                    text_color=COLORS["success"])
        except Exception as e:
            self._result.configure(text=f"Error: {e}",
                                    text_color=COLORS["danger"])
