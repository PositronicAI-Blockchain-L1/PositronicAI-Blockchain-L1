"""Settings tab — RPC config, theme toggle, data dir, about.

VS Code-inspired grouped settings with collapsible sections,
clean toggle switches, and professional footer card.
"""

import os
import subprocess
import sys
import threading

import customtkinter as ctk

from positronic.app.theme import COLORS, FONTS
from positronic.app.ui.tab_wallet import _bind_ctrl_v


# ── Collapsible settings section ────────────────────────────────
class _SettingsSection(ctk.CTkFrame):
    """A collapsible section with colored left border and header toggle."""

    def __init__(self, master, title: str, icon: str,
                 accent: str = COLORS["accent"], start_expanded: bool = True, **kw):
        super().__init__(master, fg_color="transparent", **kw)
        self._expanded = start_expanded
        self._title = title
        self._icon = icon
        self._accent = accent

        # Outer container with left accent bar
        self._outer = ctk.CTkFrame(self, fg_color=COLORS["bg_card"],
                                   corner_radius=10, border_width=1,
                                   border_color=COLORS["card_border"])
        self._outer.pack(fill="x", padx=0, pady=0)

        # Colored left accent strip (via a top bar for visual flair)
        top_bar = ctk.CTkFrame(self._outer, fg_color=accent, height=2,
                               corner_radius=0)
        top_bar.pack(fill="x", padx=1, pady=(1, 0))

        # Header — clickable toggle
        self._header = ctk.CTkButton(
            self._outer,
            text=self._build_header_text(),
            font=FONTS["subheading"],
            fg_color="transparent",
            hover_color=COLORS["bg_card_hover"],
            text_color=COLORS["text"],
            anchor="w",
            corner_radius=0,
            height=42,
            command=self._toggle,
        )
        self._header.pack(fill="x", padx=8, pady=(4, 0))

        # Content frame
        self._content = ctk.CTkFrame(self._outer, fg_color="transparent")
        if self._expanded:
            self._content.pack(fill="x", padx=16, pady=(4, 14))
        else:
            pass  # will show on toggle

    def _build_header_text(self):
        arrow = "\u25BC" if self._expanded else "\u25B6"
        return f"  {self._icon}  {self._title}   {arrow}"

    def _toggle(self):
        self._expanded = not self._expanded
        self._header.configure(text=self._build_header_text())
        if self._expanded:
            self._content.pack(fill="x", padx=16, pady=(4, 14))
        else:
            self._content.pack_forget()

    @property
    def content(self):
        return self._content


def _setting_row(parent, label_text, widget_factory, description=None):
    """Create a labeled setting row. Returns the created widget.
    widget_factory receives the row frame and returns the input widget."""
    row = ctk.CTkFrame(parent, fg_color="transparent")
    row.pack(fill="x", pady=(4, 4))

    left = ctk.CTkFrame(row, fg_color="transparent")
    left.pack(side="left", fill="y")

    ctk.CTkLabel(left, text=label_text, font=FONTS["body"],
                 text_color=COLORS["text"], anchor="w").pack(anchor="w")
    if description:
        ctk.CTkLabel(left, text=description, font=FONTS["tiny"],
                     text_color=COLORS["text_muted"], anchor="w",
                     wraplength=280).pack(anchor="w")

    widget = widget_factory(row)
    widget.pack(side="right", padx=(8, 0))
    return widget


def _separator(parent):
    """Horizontal separator line."""
    sep = ctk.CTkFrame(parent, fg_color=COLORS["separator"], height=1)
    sep.pack(fill="x", pady=(6, 6))


def build_settings(tab, win):
    """Build settings tab widgets. Stores references on *win*."""
    scroll = ctk.CTkScrollableFrame(tab, fg_color="transparent")
    scroll.pack(fill="both", expand=True, padx=10, pady=10)

    # ════════════════════════════════════════════════════════════
    # SECTION 1: Node Configuration
    # ════════════════════════════════════════════════════════════
    sec_node = _SettingsSection(scroll, "Node Configuration", "\U0001f310",
                                COLORS["accent_blue"])
    sec_node.pack(fill="x", padx=4, pady=(0, 8))
    c = sec_node.content

    # RPC Host
    win._settings_rpc_host = _setting_row(
        c, "RPC Host",
        lambda parent: _make_entry(parent, 260,
                                   getattr(win.api._rpc, '_host', '127.0.0.1')),
        description="JSON-RPC endpoint hostname or IP"
    )
    _bind_ctrl_v(win._settings_rpc_host)

    # RPC Port
    win._settings_rpc_port = _setting_row(
        c, "RPC Port",
        lambda parent: _make_entry(parent, 120,
                                   str(getattr(win.api._rpc, '_port', 8545))),
        description="JSON-RPC endpoint port number"
    )
    _bind_ctrl_v(win._settings_rpc_port)

    _separator(c)

    # Network info (read-only)
    _setting_row(
        c, "Network",
        lambda parent: _make_readonly_badge(parent, "Mainnet", COLORS["accent"]),
        description="Current network identifier"
    )

    _setting_row(
        c, "Chain ID",
        lambda parent: _make_readonly_badge(parent, "420420", COLORS["text_dim"]),
    )

    _separator(c)

    # Data Directory
    data_dir = getattr(win.api, '_data_dir', 'N/A')

    data_row = ctk.CTkFrame(c, fg_color="transparent")
    data_row.pack(fill="x", pady=(4, 4))
    ctk.CTkLabel(data_row, text="Data Directory", font=FONTS["body"],
                 text_color=COLORS["text"], anchor="w").pack(anchor="w")
    ctk.CTkLabel(data_row, text="Blockchain data and state storage path",
                 font=FONTS["tiny"],
                 text_color=COLORS["text_muted"], anchor="w").pack(anchor="w")

    dir_input_row = ctk.CTkFrame(c, fg_color="transparent")
    dir_input_row.pack(fill="x", pady=(2, 4))

    win._settings_data_dir = ctk.CTkEntry(
        dir_input_row, font=FONTS["mono_small"],
        fg_color=COLORS["bg_dark"], border_color=COLORS["border"],
        text_color=COLORS["text"], state="normal", height=32)
    win._settings_data_dir.pack(side="left", fill="x", expand=True)
    win._settings_data_dir.insert(0, str(data_dir))
    win._settings_data_dir.configure(state="readonly")

    ctk.CTkButton(dir_input_row, text="\U0001f4c2", font=FONTS["body"],
                  fg_color=COLORS["bg_dark"],
                  hover_color=COLORS["bg_card_hover"],
                  text_color=COLORS["text_dim"],
                  corner_radius=6, width=36, height=32,
                  command=lambda: _open_folder(data_dir)).pack(
        side="left", padx=(4, 0))

    # ════════════════════════════════════════════════════════════
    # SECTION 2: Security
    # ════════════════════════════════════════════════════════════
    sec_security = _SettingsSection(scroll, "Security", "\U0001f512",
                                    COLORS["purple"], start_expanded=False)
    sec_security.pack(fill="x", padx=4, pady=(0, 8))
    cs = sec_security.content

    _setting_row(
        cs, "TLS Encryption",
        lambda parent: _make_toggle(parent, value=bool(
            getattr(win.api._rpc, '_use_tls', False))),
        description="Enable HTTPS for RPC connections"
    )

    _setting_row(
        cs, "Hash Algorithm",
        lambda parent: _make_readonly_badge(parent, "SHA-512", COLORS["purple"]),
        description="Cryptographic hash function for block signing"
    )

    _setting_row(
        cs, "Post-Quantum",
        lambda parent: _make_readonly_badge(parent, "Lattice-based", COLORS["accent"]),
        description="Quantum-resistant cryptography scheme"
    )

    # ════════════════════════════════════════════════════════════
    # SECTION 3: Display
    # ════════════════════════════════════════════════════════════
    sec_display = _SettingsSection(scroll, "Display", "\U0001f3a8",
                                   COLORS["gold"])
    sec_display.pack(fill="x", padx=4, pady=(0, 8))
    cd = sec_display.content

    # Theme
    theme_row = ctk.CTkFrame(cd, fg_color="transparent")
    theme_row.pack(fill="x", pady=(4, 4))

    theme_left = ctk.CTkFrame(theme_row, fg_color="transparent")
    theme_left.pack(side="left", fill="y")
    ctk.CTkLabel(theme_left, text="Theme", font=FONTS["body"],
                 text_color=COLORS["text"], anchor="w").pack(anchor="w")
    ctk.CTkLabel(theme_left, text="Application color scheme",
                 font=FONTS["tiny"],
                 text_color=COLORS["text_muted"], anchor="w").pack(anchor="w")

    win._settings_theme = ctk.CTkSegmentedButton(
        theme_row, values=["Dark"],
        font=FONTS["button"],
        fg_color=COLORS["bg_dark"],
        selected_color=COLORS["accent"],
        selected_hover_color=COLORS["accent_blue"],
        unselected_color=COLORS["bg_card"],
        unselected_hover_color=COLORS["bg_card_hover"],
        text_color=COLORS["text"],
        command=_on_theme_change)
    win._settings_theme.pack(side="right")

    saved = _load_settings()
    saved_theme = saved.get("theme", "Dark")
    win._settings_theme.set(saved_theme)
    _on_theme_change(saved_theme)

    _separator(cd)

    # Auto-refresh interval (display-only info)
    _setting_row(
        cd, "Auto-Refresh",
        lambda parent: _make_readonly_badge(parent, "5 sec", COLORS["text_dim"]),
        description="Dashboard poll interval for live data"
    )

    # ════════════════════════════════════════════════════════════
    # ACTION BUTTONS
    # ════════════════════════════════════════════════════════════
    btn_frame = ctk.CTkFrame(scroll, fg_color="transparent")
    btn_frame.pack(fill="x", padx=4, pady=(4, 8))

    def _apply_rpc_settings():
        new_host = win._settings_rpc_host.get().strip()
        new_port = win._settings_rpc_port.get().strip()
        if new_host and new_port:
            try:
                port_int = int(new_port)
                win.api._rpc._host = new_host
                win.api._rpc._port = port_int
                win.api._rpc.url = f"{'https' if win.api._rpc._use_tls else 'http'}://{new_host}:{port_int}/"
                _save_settings(win, {"rpc_host": new_host, "rpc_port": port_int})
                win._settings_save_result.configure(
                    text="\u2713 Settings saved successfully",
                    text_color=COLORS["success"])
            except ValueError:
                win._settings_save_result.configure(
                    text="\u2717 Invalid port number",
                    text_color=COLORS["danger"])

    def _test_rpc_connection():
        """Test RPC connection in background thread."""
        win._settings_test_result.configure(
            text="\u23f3 Testing connection...", text_color=COLORS["text_dim"])
        if hasattr(win, '_settings_test_btn'):
            win._settings_test_btn.configure(state="disabled")

        def _do_test():
            try:
                resp = win.api._rpc.call("eth_chainId", [])
                if resp is not None:
                    chain_id = int(resp, 16) if isinstance(resp, str) else resp
                    height_resp = win.api._rpc.call("eth_blockNumber", [])
                    height = int(height_resp, 16) if isinstance(height_resp, str) else height_resp
                    msg = f"\u2713 Connected  \u2022  Chain {chain_id}  \u2022  Block #{height:,}"
                    color = COLORS["success"]
                else:
                    msg = "\u2717 No response from node"
                    color = COLORS["danger"]
            except Exception as exc:
                msg = f"\u2717 {exc}"
                color = COLORS["danger"]

            def _update_ui():
                win._settings_test_result.configure(text=msg, text_color=color)
                if hasattr(win, '_settings_test_btn'):
                    win._settings_test_btn.configure(state="normal")

            try:
                win.after(0, _update_ui)
            except Exception:
                pass

        threading.Thread(target=_do_test, daemon=True).start()

    def _reset_to_defaults():
        """Reset RPC settings to defaults."""
        win._settings_rpc_host.delete(0, "end")
        win._settings_rpc_host.insert(0, "127.0.0.1")
        win._settings_rpc_port.delete(0, "end")
        win._settings_rpc_port.insert(0, "8545")
        win._settings_theme.set("Dark")
        _on_theme_change("Dark")
        _apply_rpc_settings()
        win._settings_save_result.configure(
            text="\u21a9 Reset to defaults", text_color=COLORS["accent"])

    ctk.CTkButton(btn_frame, text="\u2714  Save Settings",
                  font=FONTS["button"],
                  fg_color=COLORS["accent"],
                  text_color="#000000",
                  hover_color=COLORS["accent_blue"],
                  corner_radius=8, height=38, width=140,
                  command=_apply_rpc_settings).pack(side="left", padx=(0, 8))

    win._settings_test_btn = ctk.CTkButton(
        btn_frame, text="\U0001f517  Test Connection",
        font=FONTS["button"],
        fg_color=COLORS["accent_blue"],
        text_color="#FFFFFF",
        hover_color=COLORS["purple"],
        corner_radius=8, height=38, width=160,
        command=_test_rpc_connection)
    win._settings_test_btn.pack(side="left", padx=(0, 8))

    ctk.CTkButton(btn_frame, text="\u21a9  Reset Defaults",
                  font=FONTS["button"],
                  fg_color=COLORS["bg_card"],
                  text_color=COLORS["text"],
                  hover_color=COLORS["bg_card_hover"],
                  border_width=1, border_color=COLORS["border"],
                  corner_radius=8, height=38, width=140,
                  command=_reset_to_defaults).pack(side="left")

    # Status labels
    win._settings_save_result = ctk.CTkLabel(
        scroll, text="", font=FONTS["small"],
        text_color=COLORS["text_muted"])
    win._settings_save_result.pack(anchor="w", padx=8, pady=(2, 0))

    win._settings_test_result = ctk.CTkLabel(
        scroll, text="", font=FONTS["small"],
        text_color=COLORS["text_muted"])
    win._settings_test_result.pack(anchor="w", padx=8, pady=(2, 4))

    # ════════════════════════════════════════════════════════════
    # SECTION 4: Network Policy
    # ════════════════════════════════════════════════════════════
    sec_policy = _SettingsSection(scroll, "Network Policy", "\U0001f4cb",
                                  COLORS["accent"], start_expanded=True)
    sec_policy.pack(fill="x", padx=4, pady=(0, 8))
    cp = sec_policy.content

    policy_rows = [
        ("Min Stake",        "32 ASF",       "Minimum to become a validator"),
        ("Block Time",       "12 seconds",   "Target block production interval"),
        ("Epoch Length",     "384 blocks",   "Blocks per epoch (~76.8 min)"),
        ("Unbonding Period", "7 epochs",     "~9 hours after unstake request"),
        ("Chain ID",         "420420",       "Positronic testnet chain ID"),
        ("Reward — TX Fee",  "30%",          "Validator share of transaction fees"),
        ("Reward — AI NVN",  "20%",          "NVN AI scoring reward allocation"),
    ]

    for label, value, hint in policy_rows:
        row = ctk.CTkFrame(cp, fg_color="transparent")
        row.pack(fill="x", pady=3)
        ctk.CTkLabel(row, text=label, font=FONTS["small"],
                     text_color=COLORS["text_dim"], width=160,
                     anchor="w").pack(side="left")
        ctk.CTkLabel(row, text=value, font=FONTS["badge"],
                     text_color=COLORS["accent"],
                     anchor="w").pack(side="left", padx=(8, 4))
        ctk.CTkLabel(row, text=f"— {hint}", font=FONTS["tiny"],
                     text_color=COLORS["text_muted"],
                     anchor="w").pack(side="left")

    # ════════════════════════════════════════════════════════════
    # SECTION 5: About (footer card)
    # ════════════════════════════════════════════════════════════
    sec_about = _SettingsSection(scroll, "About Positronic", "\u2139",
                                 COLORS["accent"])
    sec_about.pack(fill="x", padx=4, pady=(4, 8))
    ca = sec_about.content

    try:
        from positronic import __version__
        version = __version__
    except ImportError:
        version = "unknown"

    info_lines = [
        ("Application", "Positronic Node"),
        ("Version", f"v{version}"),
        ("Chain ID", "420420"),
        ("Symbol", "ASF"),
        ("Block Time", "12s"),
        ("Hash Algorithm", "SHA-512"),
        ("Python", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"),
    ]
    for label, value in info_lines:
        row = ctk.CTkFrame(ca, fg_color="transparent")
        row.pack(fill="x", pady=2)
        ctk.CTkLabel(row, text=label, font=FONTS["small"],
                     text_color=COLORS["text_dim"], width=130,
                     anchor="w").pack(side="left")
        ctk.CTkLabel(row, text=value, font=FONTS["mono"],
                     text_color=COLORS["text"], anchor="w").pack(
            side="left", fill="x", expand=True)

    _separator(ca)

    # Links
    links_row = ctk.CTkFrame(ca, fg_color="transparent")
    links_row.pack(fill="x", pady=(4, 4))

    link_data = [
        ("\U0001f310 Website", "https://positronic-ai.network"),
        ("\U0001f4bb GitHub", "https://github.com/PositronicAI-Blockchain-L1"),
        ("\U0001f4ac Discord", "https://discord.gg/ecXx2DQE"),
        ("\U0001d54f X/Twitter", "https://x.com/PositronicAI"),
    ]
    for link_text, url in link_data:
        btn = ctk.CTkButton(
            links_row, text=link_text, font=FONTS["small"],
            fg_color="transparent", text_color=COLORS["accent"],
            hover_color=COLORS["bg_card_hover"],
            corner_radius=6, height=28, width=110,
            command=lambda u=url: _open_url(u))
        btn.pack(side="left", padx=(0, 6))

    # Copyright
    ctk.CTkLabel(ca,
                 text="\u00a9 2025\u20132026 Positronic. All rights reserved.",
                 font=FONTS["tiny"],
                 text_color=COLORS["text_muted"]).pack(
        anchor="w", pady=(8, 0))


# ── Helper widget factories ─────────────────────────────────────

def _make_entry(parent, width, initial_value):
    """Create a styled text entry."""
    entry = ctk.CTkEntry(
        parent, font=FONTS["input"], width=width, height=32,
        fg_color=COLORS["bg_dark"], border_color=COLORS["border"],
        text_color=COLORS["text"], corner_radius=6)
    entry.insert(0, initial_value)
    return entry


def _make_readonly_badge(parent, text, color):
    """Create a read-only badge-style label."""
    badge = ctk.CTkLabel(
        parent, text=f"  {text}  ", font=FONTS["badge"],
        fg_color=COLORS["bg_dark"],
        text_color=color,
        corner_radius=6, height=26)
    return badge


def _make_toggle(parent, value=False):
    """Create a toggle switch."""
    switch = ctk.CTkSwitch(
        parent, text="", width=44, height=22,
        fg_color=COLORS["border"],
        progress_color=COLORS["accent"],
        button_color=COLORS["text"],
        button_hover_color=COLORS["accent_blue"])
    if value:
        switch.select()
    return switch


# ── Existing utility functions (unchanged logic) ─────────────────

def _on_theme_change(value: str):
    """Handle theme toggle and persist."""
    mode_map = {"Dark": "dark", "Light": "light", "System": "system"}
    ctk.set_appearance_mode(mode_map.get(value, "dark"))
    try:
        settings = _load_settings()
        settings["theme"] = value
        _save_settings_dict(settings)
    except Exception:
        pass


def _get_settings_path() -> str:
    """Get path to settings.json in user data dir."""
    if sys.platform == "win32":
        base = os.environ.get("APPDATA", os.path.expanduser("~"))
    elif sys.platform == "darwin":
        base = os.path.join(os.path.expanduser("~"), "Library", "Application Support")
    else:
        base = os.environ.get("XDG_CONFIG_HOME", os.path.join(os.path.expanduser("~"), ".config"))
    settings_dir = os.path.join(base, "Positronic")
    os.makedirs(settings_dir, exist_ok=True)
    return os.path.join(settings_dir, "settings.json")


def _get_settings_encryptor():
    """Get a DataEncryptor scoped to the settings directory."""
    from positronic.crypto.data_encryption import DataEncryptor
    settings_path = _get_settings_path()
    return DataEncryptor(os.path.dirname(settings_path))


def _save_settings(win, updates: dict):
    """Save settings to disk (encrypted)."""
    settings = _load_settings()
    settings.update(updates)
    _save_settings_dict(settings)


def _save_settings_dict(settings: dict):
    """Write the full settings dict to encrypted file."""
    enc = _get_settings_encryptor()
    enc.save_json(_get_settings_path(), settings)


def _load_settings() -> dict:
    """Load settings from disk (auto-decrypts, migrates plaintext)."""
    settings_path = _get_settings_path()
    if not os.path.isfile(settings_path):
        return {}
    try:
        enc = _get_settings_encryptor()
        return enc.load_json(settings_path)
    except Exception:
        pass
    return {}


def _open_folder(path: str):
    """Open the data folder in the system file manager."""
    if not os.path.isdir(path):
        return
    if sys.platform == "win32":
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def _open_url(url: str):
    """Open a URL in the default browser."""
    import webbrowser
    webbrowser.open(url)


def get_saved_validator_mode() -> str:
    """Return 'full' or 'light' from saved settings."""
    return _load_settings().get("validator_mode", "light")


def set_saved_validator_mode(mode: str):
    """Save validator mode preference ('full' or 'light')."""
    settings = _load_settings()
    settings["validator_mode"] = mode
    _save_settings_dict(settings)
