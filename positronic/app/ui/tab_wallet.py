"""Wallet tab — unified secret key auth, send/receive, token management."""

import re
import time
import customtkinter as ctk

from positronic.app.theme import COLORS, FONTS, _EMOJI
from positronic.app.ui.widgets import InfoCard

_HEX40_RE = re.compile(r'^(0x)?[0-9a-fA-F]{40}$')


def _is_valid_address(addr: str) -> bool:
    """Check if address is a valid 40-char hex (with optional 0x prefix)."""
    return bool(_HEX40_RE.match(addr.strip())) if addr.strip() else False


def _bind_address_validation(entry_widget, feedback_label):
    """Bind KeyRelease to show green check / red X next to an address entry."""
    def _validate(_event=None):
        val = entry_widget.get().strip()
        if not val:
            feedback_label.configure(text="", text_color=COLORS["text_muted"])
        elif _is_valid_address(val):
            feedback_label.configure(text="\u2713 Valid", text_color=COLORS["success"])
        else:
            feedback_label.configure(text="\u2717 Invalid", text_color=COLORS["danger"])
    try:
        entry_widget._entry.bind("<KeyRelease>", _validate)
    except AttributeError:
        pass


def _paste_into(entry_widget, root_widget=None):
    """Universal paste: get clipboard and insert into any CTkEntry."""
    try:
        src = root_widget or entry_widget.winfo_toplevel()
        text = src.clipboard_get()
        entry_widget.delete(0, "end")
        entry_widget.insert(0, text)
    except Exception:
        pass


def _make_paste_btn(parent, entry_widget, root_widget):
    """Create a small paste button next to any CTkEntry."""
    btn = ctk.CTkButton(parent, text="\U0001f4cb", font=(_EMOJI, 14),
                        fg_color=COLORS["bg_dark"], text_color=COLORS["accent"],
                        hover_color=COLORS["bg_card_hover"],
                        corner_radius=6, width=36, height=36,
                        command=lambda: _paste_into(entry_widget, root_widget))
    return btn


def _bind_ctrl_v(entry_widget):
    """Bind Ctrl+V to a CTkEntry so paste always works."""
    def _on_paste(event):
        _paste_into(entry_widget)
        return "break"
    try:
        entry_widget._entry.bind("<Control-v>", _on_paste)
        entry_widget._entry.bind("<Control-V>", _on_paste)
    except AttributeError:
        pass


def _format_balance(raw: str) -> str:
    """Format a balance string with commas and 4 decimal places."""
    try:
        val = float(raw.replace(",", "").replace(" ASF", "").strip())
        integer = int(val)
        frac = val - integer
        if frac == 0:
            return f"{integer:,}.0000"
        return f"{integer:,}.{f'{frac:.4f}'[2:]}"
    except (ValueError, TypeError):
        return raw if raw else "0.0000"


def _relative_time(ts) -> str:
    """Convert a timestamp (epoch or ISO) to relative time string."""
    try:
        if isinstance(ts, str):
            # Try ISO format
            import datetime
            dt = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
            epoch = dt.timestamp()
        elif isinstance(ts, (int, float)):
            epoch = float(ts)
        else:
            return str(ts)
        diff = time.time() - epoch
        if diff < 0:
            return "just now"
        if diff < 60:
            return f"{int(diff)}s ago"
        if diff < 3600:
            return f"{int(diff // 60)}m ago"
        if diff < 86400:
            return f"{int(diff // 3600)}h ago"
        return f"{int(diff // 86400)}d ago"
    except Exception:
        return str(ts) if ts else ""


def _copy_text(root, text, btn=None, original_text=""):
    """Copy text to clipboard and briefly flash the button."""
    try:
        root.clipboard_clear()
        root.clipboard_append(text)
        if btn and original_text:
            btn.configure(text="\u2705 Copied")
            root.after(1500, lambda: btn.configure(text=original_text))
    except Exception:
        pass


def _truncate_hash(h: str, front: int = 8, back: int = 6) -> str:
    """Truncate a long hash for display: 0x1234ab...cdef56"""
    s = str(h)
    if len(s) > front + back + 3:
        return s[:front] + "..." + s[-back:]
    return s


# ────────────────────────────────────────────────────────────────────
#  Section builder helpers — keeps build_wallet clean
# ────────────────────────────────────────────────────────────────────

def _build_balance_hero(scroll, win):
    """Large hero balance display at the top (shown after unlock)."""
    hero = ctk.CTkFrame(scroll, fg_color=COLORS["bg_card"],
                        corner_radius=14, border_width=1,
                        border_color=COLORS["border"])
    # Don't pack yet — shown after unlock
    win._hero_frame = hero

    # Inner padding
    inner = ctk.CTkFrame(hero, fg_color="transparent")
    inner.pack(fill="x", padx=24, pady=(20, 8))

    # Small "Wallet Balance" label
    ctk.CTkLabel(inner, text="WALLET BALANCE",
                 font=FONTS["tiny"],
                 text_color=COLORS["text_muted"]).pack(anchor="center")

    # Big balance number
    win._hero_balance = ctk.CTkLabel(
        inner, text="Loading...",
        font=("Segoe UI", 36, "bold"),
        text_color=COLORS["text"])
    win._hero_balance.pack(anchor="center", pady=(2, 0))

    # USD estimate line (placeholder)
    win._hero_usd = ctk.CTkLabel(
        inner, text="",
        font=FONTS["small"],
        text_color=COLORS["text_muted"])
    win._hero_usd.pack(anchor="center", pady=(0, 4))

    # Address with copy
    addr_row = ctk.CTkFrame(hero, fg_color="transparent")
    addr_row.pack(pady=(0, 4))
    win._hero_address = ctk.CTkLabel(
        addr_row, text="",
        font=("Cascadia Code", 11),
        text_color=COLORS["accent"])
    win._hero_address.pack(side="left")
    win._hero_copy_btn = ctk.CTkButton(
        addr_row, text="\U0001f4cb", font=(_EMOJI, 12),
        fg_color="transparent", text_color=COLORS["text_muted"],
        hover_color=COLORS["bg_card_hover"],
        corner_radius=4, width=28, height=24,
        command=lambda: _copy_text(win.root,
                                   win._hero_address.cget("text"),
                                   win._hero_copy_btn, "\U0001f4cb"))
    win._hero_copy_btn.pack(side="left", padx=(4, 0))

    # ── Quick action buttons row ──
    btn_row = ctk.CTkFrame(hero, fg_color="transparent")
    btn_row.pack(fill="x", padx=24, pady=(8, 18))

    action_btns = [
        ("\u2191 Send", COLORS["accent"], "#000000", win._do_send_transfer),
        ("\u2193 Receive", COLORS["success"], "#000000", lambda: _copy_text(
            win.root, win._hero_address.cget("text"),
            win._hero_copy_btn, "\U0001f4cb")),
        ("\u26a1 Stake", COLORS["purple"], "#FFFFFF", getattr(win, '_do_stake', lambda: None)),
        ("\u23f3 History", COLORS["bg_card_hover"], COLORS["text"], lambda: None),
        ("\U0001f512 Lock", COLORS["danger"], "#FFFFFF", getattr(win, '_do_lock_wallet', lambda: None)),
    ]
    for text, fg, tc, cmd in action_btns:
        b = ctk.CTkButton(btn_row, text=text, font=FONTS["button"],
                          fg_color=fg, text_color=tc,
                          hover_color=COLORS["bg_card_hover"],
                          corner_radius=10, height=40,
                          command=cmd)
        b.pack(side="left", expand=True, fill="x", padx=4)

    # Accent bar at bottom
    ctk.CTkFrame(hero, fg_color=COLORS["accent"], height=3,
                 corner_radius=0).pack(fill="x", side="bottom", padx=1, pady=(0, 1))


def _build_create_wallet(scroll, win):
    """Section 1: Create New Wallet."""
    create_card = ctk.CTkFrame(scroll, fg_color=COLORS["bg_card"],
                               corner_radius=12, border_width=1,
                               border_color=COLORS["border"])
    create_card.pack(fill="x", padx=16, pady=(16, 8))

    hdr = ctk.CTkFrame(create_card, fg_color="transparent")
    hdr.pack(fill="x", padx=20, pady=(16, 4))
    ctk.CTkLabel(hdr, text="\U0001f511",
                 font=(_EMOJI, 18),
                 text_color=COLORS["accent"]).pack(side="left")
    ctk.CTkLabel(hdr, text="Create New Wallet",
                 font=FONTS["heading"],
                 text_color=COLORS["text"]).pack(side="left", padx=(8, 0))

    ctk.CTkLabel(create_card,
                 text="Generate a new secret key. Write it down and store it safely.",
                 font=FONTS["body"],
                 text_color=COLORS["text_dim"]).pack(
        anchor="w", padx=20, pady=(0, 10))

    create_row = ctk.CTkFrame(create_card, fg_color="transparent")
    create_row.pack(fill="x", padx=20, pady=(0, 8))

    ctk.CTkButton(create_row, text="\u2795  Generate Wallet", font=FONTS["button"],
                  fg_color=COLORS["success"], text_color="#000000",
                  hover_color="#00cc66",
                  corner_radius=10, width=180, height=42,
                  command=win._do_create_wallet).pack(side="left")

    # Generated key display (hidden initially)
    win._new_key_frame = ctk.CTkFrame(create_card, fg_color=COLORS["bg_dark"],
                                       corner_radius=10, border_width=1,
                                       border_color=COLORS["danger"])
    # Don't pack yet — shown after creation

    key_row = ctk.CTkFrame(win._new_key_frame, fg_color="transparent")
    key_row.pack(fill="x", padx=14, pady=(14, 4))
    win._new_key_label = ctk.CTkEntry(key_row,
                                       font=("Cascadia Code", 11),
                                       text_color=COLORS["accent"],
                                       fg_color=COLORS["bg_darkest"],
                                       border_color=COLORS["accent"],
                                       width=540, height=38,
                                       state="disabled")
    win._new_key_label.pack(side="left", fill="x", expand=True)

    def _copy_new_key():
        val = win._new_key_label.get()
        if val:
            win.root.clipboard_clear()
            win.root.clipboard_append(val)
            win._copy_key_btn.configure(text="\u2705 Copied!")
            if win._copy_timer_id is not None:
                win.root.after_cancel(win._copy_timer_id)
            win._copy_timer_id = win.root.after(
                2000, lambda: win._copy_key_btn.configure(text="\U0001f4cb Copy"))

    win._copy_key_btn = ctk.CTkButton(key_row, text="\U0001f4cb Copy",
                                       font=FONTS["button"],
                                       fg_color=COLORS["success"],
                                       text_color="#000000",
                                       hover_color="#00cc66",
                                       corner_radius=8, width=100, height=38,
                                       command=_copy_new_key)
    win._copy_key_btn.pack(side="right", padx=(8, 0))

    # Warning
    win._new_key_warning = ctk.CTkLabel(win._new_key_frame,
                                         text="\u26a0 SAVE THIS KEY! If you lose it, your funds are GONE forever.\n"
                                              "Never share it with anyone. Positronic team will NEVER ask for it.",
                                         font=FONTS["small"],
                                         text_color=COLORS["danger"],
                                         wraplength=600, justify="left")

    win._wallet_result = ctk.CTkLabel(create_card, text="", font=FONTS["small"],
                                      text_color=COLORS["text_dim"])
    win._wallet_result.pack(anchor="w", padx=20, pady=(0, 14))


def _build_sign_in(scroll, win):
    """Section 2: Unlock with Secret Key."""
    login_card = ctk.CTkFrame(scroll, fg_color=COLORS["bg_card"],
                               corner_radius=12, border_width=1,
                               border_color=COLORS["border"])
    login_card.pack(fill="x", padx=16, pady=8)

    hdr = ctk.CTkFrame(login_card, fg_color="transparent")
    hdr.pack(fill="x", padx=20, pady=(16, 4))
    ctk.CTkLabel(hdr, text="\U0001f50f",
                 font=(_EMOJI, 18),
                 text_color=COLORS["accent_blue"]).pack(side="left")
    ctk.CTkLabel(hdr, text="Unlock Wallet",
                 font=FONTS["heading"],
                 text_color=COLORS["text"]).pack(side="left", padx=(8, 0))

    ctk.CTkLabel(login_card,
                 text="Enter your 64-character secret key to access your wallet.",
                 font=FONTS["body"],
                 text_color=COLORS["text_dim"]).pack(
        anchor="w", padx=20, pady=(0, 10))

    key_row = ctk.CTkFrame(login_card, fg_color="transparent")
    key_row.pack(fill="x", padx=20, pady=(0, 4))

    win._login_key = ctk.CTkEntry(
        key_row, placeholder_text="Enter your 64-character hex secret key (without 0x prefix)",
        font=("Cascadia Code", 12), show="\u2022",
        fg_color=COLORS["bg_dark"], height=42,
        border_color=COLORS["border"], text_color=COLORS["text"])
    win._login_key.pack(side="left", fill="x", expand=True, padx=(0, 4))

    _make_paste_btn(key_row, win._login_key, win.root).pack(side="left", padx=(0, 4))
    _bind_ctrl_v(win._login_key)

    ctk.CTkButton(key_row, text="\U0001f513 Unlock", font=FONTS["button"],
                  fg_color=COLORS["accent"], text_color="#000000",
                  hover_color=COLORS["accent_blue"],
                  corner_radius=10, width=110, height=42,
                  command=win._do_unlock_wallet).pack(side="right")

    # Security notice
    sec_frame = ctk.CTkFrame(login_card, fg_color="#1a1525",
                              corner_radius=8)
    sec_frame.pack(fill="x", padx=20, pady=(8, 14))
    ctk.CTkLabel(sec_frame,
                 text="\U0001f6e1  Your key never leaves this device. It is never sent to any server.",
                 font=FONTS["small"], text_color=COLORS["text_dim"],
                 wraplength=600).pack(padx=14, pady=10)

    win._login_result = ctk.CTkLabel(login_card, text="", font=FONTS["small"],
                                      text_color=COLORS["text_dim"])
    win._login_result.pack(anchor="w", padx=20, pady=(0, 8))


def _build_active_wallet(scroll, win):
    """Section 3: Active Wallet Info (shown after unlock)."""
    win._wallet_info = InfoCard(scroll, "Active Wallet", "\U0001f4b0", COLORS["accent"])
    win._wallet_info.pack(fill="x", padx=16, pady=8)
    for key, label in [
        ("w_address", "Address"),
        ("w_balance", "Balance"),
        ("w_txs", "Transactions"),
    ]:
        win._wallet_info.add_row(key, label)


def _build_scan_address(scroll, win):
    """Section 4: Scan Any Wallet (read-only lookup)."""
    scan_card = ctk.CTkFrame(scroll, fg_color=COLORS["bg_card"],
                             corner_radius=12, border_width=1,
                             border_color=COLORS["border"])
    scan_card.pack(fill="x", padx=16, pady=8)

    hdr = ctk.CTkFrame(scan_card, fg_color="transparent")
    hdr.pack(fill="x", padx=20, pady=(14, 6))
    ctk.CTkLabel(hdr, text="\U0001f50d",
                 font=(_EMOJI, 16),
                 text_color=COLORS["text_dim"]).pack(side="left")
    ctk.CTkLabel(hdr, text="Scan Any Address",
                 font=FONTS["subheading"],
                 text_color=COLORS["text"]).pack(side="left", padx=(6, 0))
    ctk.CTkLabel(hdr, text="Read-Only",
                 font=FONTS["tiny"],
                 text_color=COLORS["text_muted"],
                 fg_color=COLORS["bg_dark"],
                 corner_radius=6, width=60, height=20).pack(side="left", padx=(8, 0))

    scan_row = ctk.CTkFrame(scan_card, fg_color="transparent")
    scan_row.pack(fill="x", padx=20, pady=(0, 14))

    win._wallet_addr = ctk.CTkEntry(
        scan_row, placeholder_text="Enter wallet address (0x...)",
        font=FONTS["input"], fg_color=COLORS["bg_dark"],
        border_color=COLORS["border"], text_color=COLORS["text"],
        height=38)
    win._wallet_addr.pack(side="left", fill="x", expand=True, padx=(0, 4))
    _bind_ctrl_v(win._wallet_addr)
    _make_paste_btn(scan_row, win._wallet_addr, win.root).pack(side="left", padx=(0, 4))

    win._scan_addr_feedback = ctk.CTkLabel(scan_row, text="", font=FONTS["small"],
                                            text_color=COLORS["text_muted"], width=70)
    win._scan_addr_feedback.pack(side="left", padx=(4, 0))
    _bind_address_validation(win._wallet_addr, win._scan_addr_feedback)

    ctk.CTkButton(scan_row, text="\U0001f50d Scan", font=FONTS["button"],
                  fg_color=COLORS["accent"], text_color="#000000",
                  hover_color=COLORS["accent_blue"],
                  corner_radius=10, width=100, height=38,
                  command=lambda: __import__('threading').Thread(target=win._bg_scan_wallet, daemon=True).start()).pack(side="right")


def _build_send(scroll, win):
    """Section 5: Send ASF."""
    send_card = ctk.CTkFrame(scroll, fg_color=COLORS["bg_card"],
                              corner_radius=12, border_width=1,
                              border_color=COLORS["border"])
    send_card.pack(fill="x", padx=16, pady=8)

    hdr = ctk.CTkFrame(send_card, fg_color="transparent")
    hdr.pack(fill="x", padx=20, pady=(14, 6))
    ctk.CTkLabel(hdr, text="\u2191",
                 font=("Segoe UI", 18, "bold"),
                 text_color=COLORS["accent"]).pack(side="left")
    ctk.CTkLabel(hdr, text="Send ASF",
                 font=FONTS["subheading"],
                 text_color=COLORS["text"]).pack(side="left", padx=(6, 0))

    # To address
    send_row1 = ctk.CTkFrame(send_card, fg_color="transparent")
    send_row1.pack(fill="x", padx=20, pady=4)
    ctk.CTkLabel(send_row1, text="To", font=FONTS["small"],
                 text_color=COLORS["text_muted"], width=60).pack(side="left")
    win._send_to = ctk.CTkEntry(send_row1,
                                 placeholder_text="Recipient address (0x...)",
                                 font=FONTS["input"],
                                 fg_color=COLORS["bg_dark"],
                                 border_color=COLORS["border"],
                                 text_color=COLORS["text"],
                                 height=38)
    win._send_to.pack(side="left", fill="x", expand=True, padx=(0, 4))
    _bind_ctrl_v(win._send_to)
    _make_paste_btn(send_row1, win._send_to, win.root).pack(side="left")

    win._send_to_feedback = ctk.CTkLabel(send_row1, text="", font=FONTS["small"],
                                          text_color=COLORS["text_muted"], width=70)
    win._send_to_feedback.pack(side="left", padx=(4, 0))
    _bind_address_validation(win._send_to, win._send_to_feedback)

    # Amount
    send_row2 = ctk.CTkFrame(send_card, fg_color="transparent")
    send_row2.pack(fill="x", padx=20, pady=4)
    ctk.CTkLabel(send_row2, text="Amount", font=FONTS["small"],
                 text_color=COLORS["text_muted"], width=60).pack(side="left")
    win._send_amount = ctk.CTkEntry(send_row2,
                                     placeholder_text="0.0",
                                     font=FONTS["input"], width=200,
                                     fg_color=COLORS["bg_dark"],
                                     border_color=COLORS["border"],
                                     text_color=COLORS["text"],
                                     height=38)
    win._send_amount.pack(side="left", padx=(0, 8))
    ctk.CTkLabel(send_row2, text="ASF", font=FONTS["body"],
                 text_color=COLORS["accent"]).pack(side="left")

    # From address display
    from_row = ctk.CTkFrame(send_card, fg_color="transparent")
    from_row.pack(fill="x", padx=16, pady=(0, 4))
    ctk.CTkLabel(from_row, text="From", font=FONTS["small"],
                 text_color=COLORS["text_dim"], width=60,
                 anchor="w").pack(side="left")
    win._send_from_label = ctk.CTkLabel(
        from_row, text="— unlock wallet —", font=FONTS["tiny"],
        text_color=COLORS["text_muted"], anchor="w")
    win._send_from_label.pack(side="left", padx=(8, 0))

    # Send button
    send_btn_row = ctk.CTkFrame(send_card, fg_color="transparent")
    send_btn_row.pack(fill="x", padx=20, pady=(10, 4))
    ctk.CTkButton(send_btn_row, text="\u2191 Send Transaction", font=FONTS["button"],
                  fg_color=COLORS["accent"], text_color="#000000",
                  hover_color=COLORS["accent_blue"],
                  corner_radius=10, height=42, width=180,
                  command=win._do_send_transfer).pack(side="left")

    win._send_result = ctk.CTkLabel(send_card, text="", font=FONTS["small"],
                                     text_color=COLORS["text_dim"])
    win._send_result.pack(anchor="w", padx=20, pady=(0, 14))


def _build_tx_history(scroll, win):
    """Section: Transaction History (shown after unlock)."""
    tx_card = ctk.CTkFrame(scroll, fg_color=COLORS["bg_card"],
                           corner_radius=12, border_width=1,
                           border_color=COLORS["border"])
    win._tx_history_card = tx_card
    # Don't pack yet — shown after unlock

    hdr = ctk.CTkFrame(tx_card, fg_color="transparent")
    hdr.pack(fill="x", padx=20, pady=(14, 6))
    ctk.CTkLabel(hdr, text="\u23f3",
                 font=(_EMOJI, 16),
                 text_color=COLORS["text_dim"]).pack(side="left")
    ctk.CTkLabel(hdr, text="Transaction History",
                 font=FONTS["subheading"],
                 text_color=COLORS["text"]).pack(side="left", padx=(6, 0))
    win._tx_count_badge = ctk.CTkLabel(
        hdr, text="0", font=FONTS["tiny"],
        text_color=COLORS["bg_darkest"],
        fg_color=COLORS["accent"], corner_radius=8, width=28, height=20)
    win._tx_count_badge.pack(side="left", padx=(8, 0))

    # Table header
    tbl_hdr = ctk.CTkFrame(tx_card, fg_color=COLORS["bg_darkest"],
                            corner_radius=0, height=30)
    tbl_hdr.pack(fill="x", padx=12, pady=(4, 0))
    tbl_hdr.pack_propagate(False)
    for text, w in [("Type", 80), ("Hash", 130), ("Amount", 120), ("Time", 80)]:
        ctk.CTkLabel(tbl_hdr, text=text, font=FONTS["tiny"],
                     text_color=COLORS["text_muted"], width=w,
                     anchor="w").pack(side="left", padx=(10, 0))

    # Scrollable tx rows
    win._tx_list_frame = ctk.CTkScrollableFrame(
        tx_card, fg_color="transparent", height=160)
    win._tx_list_frame.pack(fill="both", expand=True, padx=8, pady=(0, 10))
    win._tx_rows = []

    # Empty state
    win._tx_empty = ctk.CTkLabel(
        win._tx_list_frame, text="No transactions yet.",
        font=FONTS["small"], text_color=COLORS["text_muted"])
    win._tx_empty.pack(pady=16)


def _build_tokens(scroll, win):
    """Section 6: Tokens."""
    token_card = ctk.CTkFrame(scroll, fg_color=COLORS["bg_card"],
                               corner_radius=12, border_width=1,
                               border_color=COLORS["border"])
    token_card.pack(fill="x", padx=16, pady=(8, 16))

    hdr = ctk.CTkFrame(token_card, fg_color="transparent")
    hdr.pack(fill="x", padx=20, pady=(14, 6))
    ctk.CTkLabel(hdr, text="\U0001fa99",
                 font=(_EMOJI, 16),
                 text_color=COLORS["gold"]).pack(side="left")
    ctk.CTkLabel(hdr, text="Tokens",
                 font=FONTS["subheading"],
                 text_color=COLORS["text"]).pack(side="left", padx=(6, 0))

    win._token_list_frame = ctk.CTkScrollableFrame(
        token_card, fg_color="transparent", height=80)
    win._token_list_frame.pack(fill="x", padx=16, pady=(0, 4))

    win._token_empty_label = ctk.CTkLabel(
        win._token_list_frame, text="No tokens found. Unlock wallet first.",
        font=FONTS["small"], text_color=COLORS["text_muted"])
    win._token_empty_label.pack(anchor="w")

    tk_btn_row = ctk.CTkFrame(token_card, fg_color="transparent")
    tk_btn_row.pack(fill="x", padx=20, pady=(4, 14))
    ctk.CTkButton(tk_btn_row, text="+ Create Token", font=FONTS["button"],
                  fg_color=COLORS["purple"], text_color="#FFFFFF",
                  hover_color="#9955FF",
                  corner_radius=10, height=38, width=150,
                  command=win._do_create_token).pack(side="left")


# ────────────────────────────────────────────────────────────────────
#  Main builder
# ────────────────────────────────────────────────────────────────────

def build_wallet(tab, win):
    """Build wallet tab widgets. Stores references on *win*."""
    scroll = ctk.CTkScrollableFrame(tab, fg_color="transparent")
    scroll.pack(fill="both", expand=True, padx=0, pady=0)

    # Hero balance card (hidden until wallet unlocked)
    _build_balance_hero(scroll, win)

    # 1. Create New Wallet
    _build_create_wallet(scroll, win)

    # 2. Sign In
    _build_sign_in(scroll, win)

    # 3. Active Wallet Info
    _build_active_wallet(scroll, win)

    # 4. Scan Any Address
    _build_scan_address(scroll, win)

    # 5. Send ASF
    _build_send(scroll, win)

    # 6. Transaction History (hidden until unlock)
    _build_tx_history(scroll, win)

    # 7. Tokens
    _build_tokens(scroll, win)


# ────────────────────────────────────────────────────────────────────
#  TX History row builder (called from refresh logic)
# ────────────────────────────────────────────────────────────────────

def _build_tx_rows(win, txs):
    """Rebuild transaction history rows from a list of tx dicts."""
    for w in win._tx_rows:
        w.destroy()
    win._tx_rows.clear()

    if not txs:
        win._tx_empty.pack(pady=16)
        return

    try:
        win._tx_empty.pack_forget()
    except Exception:
        pass

    # TX type badge colors
    type_colors = {
        "transfer": (COLORS["accent"], "\u2194"),
        "reward": (COLORS["gold"], "\u2605"),
        "stake": (COLORS["purple"], "\u26a1"),
        "unstake": (COLORS["warning"], "\U0001f513"),
        "deploy": (COLORS["accent_blue"], "\u2699"),
        "system": (COLORS["text_dim"], "\u2699"),
    }

    my_addr = ""
    try:
        my_addr = win._hero_address.cget("text").lower().strip()
    except Exception:
        pass

    for i, tx in enumerate(txs[:50]):
        bg = COLORS["bg_card"] if i % 2 == 0 else "transparent"
        row = ctk.CTkFrame(win._tx_list_frame, fg_color=bg,
                           corner_radius=4, height=32)
        row.pack(fill="x", pady=1)
        row.pack_propagate(False)

        # Type badge
        tx_type = str(tx.get("type", tx.get("kind", "transfer"))).lower()
        badge_color, badge_icon = type_colors.get(tx_type, (COLORS["text_dim"], "\u2022"))
        type_frame = ctk.CTkFrame(row, fg_color=badge_color, corner_radius=4,
                                  width=70, height=20)
        type_frame.pack(side="left", padx=(10, 0), pady=6)
        type_frame.pack_propagate(False)
        ctk.CTkLabel(type_frame,
                     text=f"{badge_icon} {tx_type.capitalize()[:8]}",
                     font=FONTS["tiny"],
                     text_color="#000000" if tx_type != "system" else "#FFFFFF").pack(expand=True)

        # Hash (truncated, clickable to copy)
        tx_hash = str(tx.get("hash", tx.get("tx_hash", "?")))
        short_hash = _truncate_hash(tx_hash)
        hash_btn = ctk.CTkButton(
            row, text=short_hash,
            font=("Cascadia Code", 9),
            fg_color="transparent",
            text_color=COLORS["text_dim"],
            hover_color=COLORS["bg_card_hover"],
            corner_radius=4, width=130, height=24, anchor="w",
            command=lambda h=tx_hash: _copy_text(win.root, h))
        hash_btn.pack(side="left", padx=(10, 0))

        # Amount (color-coded)
        amount = tx.get("amount", tx.get("value", "0"))
        try:
            amt_float = float(str(amount))
        except (ValueError, TypeError):
            amt_float = 0
        my_addr_clean = my_addr.lower().replace("0x", "")
        sender = str(tx.get("from", tx.get("sender", ""))).lower().replace("0x", "")
        receiver = str(tx.get("to", tx.get("recipient", ""))).lower().replace("0x", "")
        is_sent = bool(my_addr_clean and sender == my_addr_clean)
        is_received = bool(my_addr_clean and receiver == my_addr_clean and not is_sent)

        if is_received:
            amt_text = f"+{amt_float:,.4f}"
            amt_color = COLORS["success"]
        elif is_sent:
            amt_text = f"-{amt_float:,.4f}"
            amt_color = COLORS["danger"]
        else:
            amt_text = f"{amt_float:,.4f}"
            amt_color = COLORS["text"]

        ctk.CTkLabel(row, text=amt_text, font=("Cascadia Code", 10),
                     text_color=amt_color, width=120,
                     anchor="w").pack(side="left", padx=(10, 0))

        # Time
        ts = tx.get("timestamp", tx.get("time", ""))
        rel_time = _relative_time(ts)
        ctk.CTkLabel(row, text=rel_time, font=FONTS["tiny"],
                     text_color=COLORS["text_muted"], width=80,
                     anchor="w").pack(side="left", padx=(10, 0))

        win._tx_rows.append(row)

    win._tx_count_badge.configure(text=str(len(txs)))


def update_hero_balance(win, address: str, balance: str, tx_count=None):
    """Update the hero balance card. Call after wallet unlock or refresh."""
    try:
        formatted = _format_balance(balance)
        win._hero_balance.configure(text=f"{formatted} ASF")
        if address:
            display_addr = _truncate_hash(address, 10, 8)
            win._hero_address.configure(text=display_addr)
            if hasattr(win, '_send_from_label'):
                short = f"{address[:10]}...{address[-8:]}" if len(address) > 20 else address
                win._send_from_label.configure(text=short, text_color=COLORS["accent"])
        if not win._hero_frame.winfo_ismapped():
            try:
                children = list(win._hero_frame.master.winfo_children())
                # Insert before the second child (create_wallet card) if it exists
                before_widget = children[1] if len(children) > 1 else None
                if before_widget and before_widget is not win._hero_frame:
                    win._hero_frame.pack(fill="x", padx=16, pady=(16, 8),
                                         before=before_widget)
                else:
                    win._hero_frame.pack(fill="x", padx=16, pady=(16, 8))
            except Exception:
                win._hero_frame.pack(fill="x", padx=16, pady=(16, 8))
    except Exception:
        pass

    # Show tx history card if hidden
    try:
        if not win._tx_history_card.winfo_ismapped():
            win._tx_history_card.pack(fill="x", padx=16, pady=8)
    except Exception:
        pass
