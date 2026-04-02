"""Validator tab — status card, staking controls, epoch info bar."""

import customtkinter as ctk

from positronic.app.theme import COLORS, FONTS, SECTION_GAP, CARD_GAP, _EMOJI
from positronic.app.ui.widgets import InfoCard


def _paste_to(win, entry):
    """Paste clipboard content into a CTkEntry widget."""
    try:
        text = win.root.clipboard_get()
        entry.delete(0, "end")
        entry.insert(0, text)
    except Exception:
        pass


# ── Validator status hero card ────────────────────────────────────
def _build_status_hero(parent, win):
    """Large hero card showing validator active/inactive status with glow."""
    hero = ctk.CTkFrame(parent, fg_color=COLORS["bg_card"],
                         corner_radius=14, border_width=1,
                         border_color=COLORS["card_border"])
    hero.pack(fill="x", padx=16, pady=(SECTION_GAP, CARD_GAP))
    win._val_hero_card = hero

    # Glow accent bar at top
    win._val_hero_bar = ctk.CTkFrame(hero, fg_color=COLORS["text_muted"],
                                      height=4, corner_radius=0)
    win._val_hero_bar.pack(fill="x", padx=1, pady=(1, 0))

    inner = ctk.CTkFrame(hero, fg_color="transparent")
    inner.pack(fill="x", padx=20, pady=16)

    # Left: status dot + badge
    left = ctk.CTkFrame(inner, fg_color="transparent")
    left.pack(side="left")

    status_row = ctk.CTkFrame(left, fg_color="transparent")
    status_row.pack(anchor="w")

    win._val_status_dot = ctk.CTkLabel(status_row, text="\u25cf",
                                        font=("Segoe UI", 18),
                                        text_color=COLORS["text_muted"])
    win._val_status_dot.pack(side="left")
    win._val_status_badge = ctk.CTkLabel(status_row, text="INACTIVE",
                                          font=FONTS["heading"],
                                          text_color=COLORS["text_muted"])
    win._val_status_badge.pack(side="left", padx=(8, 0))

    win._val_status_sub = ctk.CTkLabel(left, text="Not staking",
                                        font=FONTS["small"],
                                        text_color=COLORS["text_dim"])
    win._val_status_sub.pack(anchor="w", pady=(2, 0))

    # Right: stake + rewards display
    right = ctk.CTkFrame(inner, fg_color="transparent")
    right.pack(side="right")

    # Stake value
    stake_col = ctk.CTkFrame(right, fg_color="transparent")
    stake_col.pack(side="left", padx=(0, 24))
    ctk.CTkLabel(stake_col, text="Your Stake",
                 font=FONTS["stat_label"],
                 text_color=COLORS["text_dim"]).pack(anchor="e")
    win._val_stake_value = ctk.CTkLabel(stake_col, text="0 ASF",
                                         font=FONTS["hero_value"],
                                         text_color=COLORS["text"])
    win._val_stake_value.pack(anchor="e")

    # Rewards value
    rew_col = ctk.CTkFrame(right, fg_color="transparent")
    rew_col.pack(side="left")
    ctk.CTkLabel(rew_col, text="Pending Rewards",
                 font=FONTS["stat_label"],
                 text_color=COLORS["text_dim"]).pack(anchor="e")
    win._val_rewards_value = ctk.CTkLabel(rew_col, text="0 ASF",
                                           font=FONTS["hero_value"],
                                           text_color=COLORS["gold"])
    win._val_rewards_value.pack(anchor="e")


# ── Epoch / Network info bar ─────────────────────────────────────
def _build_epoch_bar(parent, win):
    """Compact info bar showing epoch, validators, block height."""
    bar = ctk.CTkFrame(parent, fg_color=COLORS["card_bg_elevated"],
                        corner_radius=10, border_width=1,
                        border_color=COLORS["card_border"], height=44)
    bar.pack(fill="x", padx=16, pady=(0, CARD_GAP))
    bar.pack_propagate(False)

    inner = ctk.CTkFrame(bar, fg_color="transparent")
    inner.pack(fill="both", expand=True, padx=16)

    # Each info pill
    pills = [
        ("\u23f3", "Epoch", "val_epoch_pill"),
        ("\u26a1", "Validators", "val_count_pill"),
        ("\u26cf", "Block", "val_height_pill"),
    ]
    for i, (icon, label, attr_name) in enumerate(pills):
        if i > 0:
            # Separator
            sep = ctk.CTkFrame(inner, fg_color=COLORS["separator"], width=1)
            sep.pack(side="left", fill="y", padx=12, pady=8)

        pill = ctk.CTkFrame(inner, fg_color="transparent")
        pill.pack(side="left")
        ctk.CTkLabel(pill, text=f"{icon} {label}",
                     font=FONTS["tiny"],
                     text_color=COLORS["text_dim"]).pack(side="left")
        val_lbl = ctk.CTkLabel(pill, text="--",
                                font=FONTS["badge"],
                                text_color=COLORS["text"])
        val_lbl.pack(side="left", padx=(6, 0))
        setattr(win, f"_{attr_name}", val_lbl)


# ── Staking controls card ─────────────────────────────────────────
def _build_staking_controls(parent, win):
    """Build the staking controls section."""
    ctrl = ctk.CTkFrame(parent, fg_color=COLORS["bg_card"],
                        corner_radius=14, border_width=1,
                        border_color=COLORS["card_border"])
    ctrl.pack(fill="x", padx=16, pady=(0, CARD_GAP))

    # Header
    hdr = ctk.CTkFrame(ctrl, fg_color="transparent")
    hdr.pack(fill="x", padx=20, pady=(16, 12))
    ctk.CTkLabel(hdr, text="\u26a1", font=(_EMOJI, 14),
                 text_color=COLORS["purple"]).pack(side="left")
    ctk.CTkLabel(hdr, text="Staking Controls", font=FONTS["section_title"],
                 text_color=COLORS["text"]).pack(side="left", padx=(6, 0))

    # Divider
    ctk.CTkFrame(ctrl, fg_color=COLORS["separator"], height=1).pack(
        fill="x", padx=20)

    # ── Amount input row ──────────────────────────────────────────
    amount_row = ctk.CTkFrame(ctrl, fg_color="transparent")
    amount_row.pack(fill="x", padx=20, pady=(12, 4))

    ctk.CTkLabel(amount_row, text="Amount", font=FONTS["small"],
                 text_color=COLORS["text_dim"]).pack(side="left", padx=(0, 12))

    input_frame = ctk.CTkFrame(amount_row, fg_color=COLORS["bg_dark"],
                                corner_radius=8, border_width=1,
                                border_color=COLORS["border"])
    input_frame.pack(side="left", fill="x", expand=True)

    win._stake_amount = ctk.CTkEntry(input_frame, placeholder_text="32",
                                     font=FONTS["input"], width=200,
                                     fg_color="transparent",
                                     border_width=0,
                                     text_color=COLORS["text"])
    win._stake_amount.pack(side="left", fill="x", expand=True, padx=8, pady=4)

    def _validate_stake_amt(event=None, entry=win._stake_amount):
        try:
            v = float(entry.get().strip())
            if v < 32:
                entry.configure(border_color=COLORS["danger"])
            else:
                entry.configure(border_color=COLORS["accent"])
        except ValueError:
            entry.configure(border_color=COLORS["border"])
    win._stake_amount.bind("<KeyRelease>", _validate_stake_amt)

    ctk.CTkLabel(input_frame, text="ASF", font=FONTS["badge"],
                 text_color=COLORS["text_muted"]).pack(side="right", padx=(0, 12))

    # ── Minimum stake notice ──────────────────────────────────────
    min_frame = ctk.CTkFrame(ctrl, fg_color=COLORS["glow_cyan"],
                              corner_radius=8)
    min_frame.pack(fill="x", padx=20, pady=(8, 4))

    ctk.CTkLabel(min_frame, text="\u26a1 Minimum: 32 ASF",
                 font=FONTS["badge"],
                 text_color=COLORS["accent"]).pack(
        side="left", padx=12, pady=8)
    ctk.CTkLabel(min_frame,
                 text="Validators must stake at least 32 ASF to participate.",
                 font=FONTS["tiny"],
                 text_color=COLORS["text_dim"]).pack(
        side="left", padx=(0, 12), pady=8)

    # ── Wallet status ─────────────────────────────────────────────
    wallet_row = ctk.CTkFrame(ctrl, fg_color="transparent")
    wallet_row.pack(fill="x", padx=20, pady=(8, 4))
    ctk.CTkLabel(wallet_row, text="Wallet", font=FONTS["small"],
                 text_color=COLORS["text_dim"]).pack(side="left", padx=(0, 12))
    win._stake_wallet_status = ctk.CTkLabel(
        wallet_row, text="\u26a0 Unlock wallet first (Wallet tab \u2192 Sign In)",
        font=FONTS["small"], text_color=COLORS["warning"])
    win._stake_wallet_status.pack(side="left")

    # Hidden fields for compatibility (populated from _active_address)
    win._stake_pass = None  # No longer needed
    win._stake_addr = None  # Auto-filled from wallet

    # ── Action buttons ────────────────────────────────────────────
    ctk.CTkFrame(ctrl, fg_color=COLORS["separator"], height=1).pack(
        fill="x", padx=20, pady=(12, 0))

    btns = ctk.CTkFrame(ctrl, fg_color="transparent")
    btns.pack(fill="x", padx=20, pady=(12, 16))

    btn_configs = [
        ("\u26a1 Stake ASF", COLORS["accent"], "#000000",
         COLORS["accent_blue"], win._do_stake),
        ("\U0001f513 Unstake", COLORS["warning"], "#000000",
         "#cc9900", win._do_unstake),
    ]
    for text, fg, txt_clr, hover, cmd in btn_configs:
        ctk.CTkButton(btns, text=text, font=FONTS["button"],
                      fg_color=fg, text_color=txt_clr,
                      hover_color=hover,
                      corner_radius=8, height=40, width=130,
                      command=cmd).pack(side="left", padx=(0, 8))

    claim_wrap = ctk.CTkFrame(btns, fg_color="transparent")
    claim_wrap.pack(side="left", padx=(0, 8))

    win._claim_btn = ctk.CTkButton(claim_wrap, text="\U0001f381 Claim Rewards",
                                    font=FONTS["button"],
                                    fg_color=COLORS["success"], text_color="#000000",
                                    hover_color="#00cc66",
                                    corner_radius=8, height=40, width=170,
                                    command=win._do_claim_rewards)
    win._claim_btn.pack()

    win._rewards_label = ctk.CTkLabel(claim_wrap, text="",
                                       font=FONTS["tiny"],
                                       text_color=COLORS["gold"])
    win._rewards_label.pack(pady=(2, 0))

    def _export_key_with_warning():
        from positronic.app.ui.dialogs import ConfirmDialog
        dlg = ConfirmDialog(
            win.root if hasattr(win, 'root') else win,
            title="⚠️ Export Private Key",
            message="This will display your raw private key on screen.\n\nNEVER share it with anyone. Anyone with your key can steal ALL your funds.",
            confirm_text="I Understand, Show Key",
            cancel_text="Cancel",
            confirm_color=COLORS["danger"],
            icon="⚠️")
        if dlg.result:
            from positronic.app.ui.dialogs import ExportKeyDialog
            ExportKeyDialog(win, win.api, active_key=getattr(win, '_active_secret_key', None))

    ctk.CTkButton(btns, text="\U0001f511 Export Key", font=FONTS["button"],
                  fg_color=COLORS["accent_blue"], text_color="#FFFFFF",
                  hover_color=COLORS["accent"],
                  corner_radius=8, height=40, width=130,
                  command=_export_key_with_warning).pack(side="left")

    # Result label
    win._stake_result = ctk.CTkLabel(ctrl, text="", font=FONTS["small"],
                                     text_color=COLORS["text_dim"])
    win._stake_result.pack(anchor="w", padx=20, pady=(0, 12))


# ── Validator info card (legacy compat — kept as secondary detail) ─
def _build_detail_card(parent, win):
    """Build the detail info card (kept for backward compat with refresh_validator)."""
    win._val_status = InfoCard(parent, "Validator Details", "\U0001f4cb",
                               COLORS["purple"])
    win._val_status.pack(fill="x", padx=16, pady=(0, 12))
    for key, label in [
        ("val_active", "Status"),
        ("val_stake", "Your Stake"),
        ("val_rewards", "Pending Rewards"),
        ("val_count", "Total Validators"),
        ("val_epoch", "Current Epoch"),
        ("val_height", "Block Height"),
    ]:
        win._val_status.add_row(key, label)


# ── Validator mode helpers ────────────────────────────────────────

def _section_header_inline(parent, icon, title):
    hdr = ctk.CTkFrame(parent, fg_color="transparent")
    hdr.pack(fill="x")
    ctk.CTkLabel(hdr, text=icon, font=(_EMOJI, 14),
                 text_color=COLORS["accent"]).pack(side="left")
    ctk.CTkLabel(hdr, text=title, font=FONTS["subheading"],
                 text_color=COLORS["text"]).pack(side="left", padx=(6, 0))


def _toggle_validator_mode(win):
    """Toggle between Full and Light validator mode."""
    from positronic.app.ui.tab_settings import get_saved_validator_mode, set_saved_validator_mode
    current = get_saved_validator_mode()

    if current == "full":
        # Deactivate: switch to light
        set_saved_validator_mode("light")
        win._validator_mode_label.configure(
            text="Current: Light Validator", text_color=COLORS["text_muted"])
        win._validator_mode_btn.configure(
            text="⚡ Activate Full Validator",
            fg_color=COLORS["accent"], text_color="#000000")
        if hasattr(win, '_app') and win._app:
            win._app._restart_node_with_validator(False)
    else:
        # Activate: show confirmation first
        from positronic.app.ui.dialogs import ConfirmDialog
        dlg = ConfirmDialog(
            win.root,
            title="Activate Full Validator",
            message="Full Validator mode produces blocks and earns ~24 ASF/block rewards.\n\n"
                    "⚠ Your node MUST stay online 24/7.\n"
                    "⚠ Going offline may result in slashing penalties.\n\n"
                    "Continue?",
            confirm_text="⚡ Activate",
            cancel_text="Cancel",
            confirm_color=COLORS["accent"],
            icon="⚡")
        if not dlg.result:
            return
        set_saved_validator_mode("full")
        win._validator_mode_label.configure(
            text="Current: Full Validator ✓", text_color=COLORS["success"])
        win._validator_mode_btn.configure(
            text="🔽 Switch to Light Validator",
            fg_color=COLORS["warning"], text_color="#000000")
        if hasattr(win, '_app') and win._app:
            win._app._restart_node_with_validator(True)


# ══════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════

def build_validator(tab, win):
    """Build validator tab widgets. Stores references on *win*."""
    # Scrollable wrapper so content doesn't overflow on small windows
    scroll = ctk.CTkScrollableFrame(tab, fg_color="transparent")
    scroll.pack(fill="both", expand=True)

    _build_status_hero(scroll, win)
    _build_epoch_bar(scroll, win)
    _build_staking_controls(scroll, win)

    # ── Validator Mode Control ────────────────────────────────────
    mode_sep = ctk.CTkFrame(scroll, fg_color=COLORS["accent"], height=2, corner_radius=0)
    mode_sep.pack(fill="x", padx=20, pady=(20, 0))

    mode_card = ctk.CTkFrame(scroll, fg_color=COLORS["bg_card"], corner_radius=12,
                              border_width=1, border_color=COLORS["border"])
    mode_card.pack(fill="x", padx=16, pady=(12, 8))

    mode_inner = ctk.CTkFrame(mode_card, fg_color="transparent")
    mode_inner.pack(fill="x", padx=20, pady=16)

    _section_header_inline(mode_inner, "⚡", "Validator Mode")

    mode_desc = ctk.CTkLabel(mode_inner,
        text="Full Validator: produces blocks & earns rewards (must stay online 24/7)\n"
             "Light Validator: attests blocks & protects network (no block production)",
        font=FONTS["small"], text_color=COLORS["text_dim"],
        justify="left", anchor="w")
    mode_desc.pack(fill="x", pady=(4, 12))

    win._validator_mode_label = ctk.CTkLabel(mode_inner, text="Current: Light Validator",
        font=FONTS["body"], text_color=COLORS["text_muted"])
    win._validator_mode_label.pack(fill="x", pady=(0, 8))

    win._validator_mode_btn = ctk.CTkButton(mode_inner,
        text="⚡ Activate Full Validator",
        font=FONTS["button"], height=40, corner_radius=8,
        fg_color=COLORS["accent"], text_color="#000000",
        hover_color=COLORS["accent_blue"],
        command=lambda: _toggle_validator_mode(win))
    win._validator_mode_btn.pack(fill="x")

    _build_detail_card(scroll, win)


def refresh_validator(win, vi):
    """Update validator tab from *vi* dict."""
    if vi.get("online"):
        # User-specific validator status (from positronic_getStakingInfo)
        your_is_val = vi.get("your_is_validator", False)
        your_stake = vi.get("your_stake", 0)
        your_rewards = vi.get("your_rewards", 0)

        # ── Hero card update ──────────────────────────────────────
        if your_is_val:
            _set_hero_active(win, your_stake, your_rewards)
            win._val_status.set("val_active", "\U0001f7e2 ACTIVE VALIDATOR")
        elif vi.get("network_has_validators"):
            _set_hero_network_active(win, your_stake, your_rewards)
            win._val_status.set("val_active", "\U0001f7e1 NETWORK ACTIVE (you: not staked)")
        else:
            _set_hero_inactive(win)
            win._val_status.set("val_active", "\u26aa INACTIVE")

        # ── Epoch bar pills ───────────────────────────────────────
        epoch = vi.get("epoch", 0)
        val_count = vi.get("validator_count", vi.get("active_validators", 0))
        block_h = vi.get("block_height", 0)
        if hasattr(win, "_val_epoch_pill"):
            win._val_epoch_pill.configure(text=str(epoch))
        if hasattr(win, "_val_count_pill"):
            win._val_count_pill.configure(text=str(val_count))
        if hasattr(win, "_val_height_pill"):
            win._val_height_pill.configure(text=f"#{block_h:,}")

        # ── Legacy info card ──────────────────────────────────────
        win._val_status.set("val_count", str(val_count))
        win._val_status.set("val_epoch", str(epoch))
        win._val_status.set("val_height", f"#{block_h:,}")
        win._val_status.set("val_stake",
                            f"{your_stake:,.4f} ASF" if your_stake > 0 else "0 ASF")
        win._val_status.set("val_rewards",
                            f"{your_rewards:,.6f} ASF" if your_rewards > 0 else "0 ASF")

        # Update rewards label next to Claim button
        if hasattr(win, '_claim_btn'):
            win._claim_btn.configure(text="\U0001f381 Claim Rewards")
        if hasattr(win, '_rewards_label'):
            if your_rewards > 0:
                win._rewards_label.configure(
                    text=f"{your_rewards:,.2f} ASF pending")
            else:
                win._rewards_label.configure(text="")

    # Update wallet status indicator
    if hasattr(win, '_active_address') and win._active_address:
        win._stake_wallet_status.configure(
            text=f"\u2705 {win._active_address}",
            text_color=COLORS["success"])
    else:
        win._stake_wallet_status.configure(
            text="\u26a0 Unlock wallet first (Wallet tab \u2192 Sign In)",
            text_color=COLORS["warning"])

    # Update validator mode display
    if hasattr(win, '_validator_mode_btn'):
        from positronic.app.ui.tab_settings import get_saved_validator_mode
        mode = get_saved_validator_mode()
        if mode == "full":
            win._validator_mode_label.configure(
                text="Current: Full Validator \u2713", text_color=COLORS["success"])
            win._validator_mode_btn.configure(
                text="\U0001f53d Switch to Light Validator",
                fg_color=COLORS["warning"], text_color="#000000")
        else:
            win._validator_mode_label.configure(
                text="Current: Light Validator", text_color=COLORS["text_muted"])
            win._validator_mode_btn.configure(
                text="\u26a1 Activate Full Validator",
                fg_color=COLORS["accent"], text_color="#000000")


# ── Hero card state helpers ───────────────────────────────────────

def _set_hero_active(win, stake, rewards):
    """Set hero card to ACTIVE VALIDATOR state (green glow)."""
    color = COLORS["success_green"]
    if hasattr(win, "_val_hero_bar"):
        win._val_hero_bar.configure(fg_color=color)
    if hasattr(win, "_val_hero_card"):
        win._val_hero_card.configure(border_color=color)
    if hasattr(win, "_val_status_dot"):
        win._val_status_dot.configure(text_color=color)
    if hasattr(win, "_val_status_badge"):
        win._val_status_badge.configure(text="ACTIVE VALIDATOR", text_color=color)
    if hasattr(win, "_val_status_sub"):
        win._val_status_sub.configure(
            text="Your node is validating blocks and earning rewards",
            text_color=COLORS["text_dim"])
    if hasattr(win, "_val_stake_value"):
        win._val_stake_value.configure(text=f"{stake:,.4f} ASF")
    if hasattr(win, "_val_rewards_value"):
        win._val_rewards_value.configure(
            text=f"{rewards:,.6f} ASF" if rewards > 0 else "0 ASF")


def _set_hero_network_active(win, stake, rewards):
    """Set hero card to NETWORK ACTIVE but user not staked (yellow)."""
    color = COLORS["warning_yellow"]
    if hasattr(win, "_val_hero_bar"):
        win._val_hero_bar.configure(fg_color=color)
    if hasattr(win, "_val_hero_card"):
        win._val_hero_card.configure(border_color=color)
    if hasattr(win, "_val_status_dot"):
        win._val_status_dot.configure(text_color=color)
    if hasattr(win, "_val_status_badge"):
        win._val_status_badge.configure(text="NETWORK ACTIVE", text_color=color)
    if hasattr(win, "_val_status_sub"):
        win._val_status_sub.configure(text="Network is active, but you are not staked",
                                       text_color=COLORS["text_dim"])
    if hasattr(win, "_val_stake_value"):
        win._val_stake_value.configure(
            text=f"{stake:,.4f} ASF" if stake > 0 else "0 ASF")
    if hasattr(win, "_val_rewards_value"):
        win._val_rewards_value.configure(
            text=f"{rewards:,.6f} ASF" if rewards > 0 else "0 ASF")


def _set_hero_inactive(win):
    """Set hero card to INACTIVE state (grey)."""
    color = COLORS["text_muted"]
    if hasattr(win, "_val_hero_bar"):
        win._val_hero_bar.configure(fg_color=color)
    if hasattr(win, "_val_hero_card"):
        win._val_hero_card.configure(border_color=COLORS["card_border"])
    if hasattr(win, "_val_status_dot"):
        win._val_status_dot.configure(text_color=color)
    if hasattr(win, "_val_status_badge"):
        win._val_status_badge.configure(text="INACTIVE", text_color=color)
    if hasattr(win, "_val_status_sub"):
        win._val_status_sub.configure(text="Stake ASF to become a validator",
                                       text_color=COLORS["text_dim"])
    if hasattr(win, "_val_stake_value"):
        win._val_stake_value.configure(text="0 ASF")
    if hasattr(win, "_val_rewards_value"):
        win._val_rewards_value.configure(text="0 ASF")
