"""Dashboard tab — stat cards, mining chart, AI card, activity panel."""

import time

import customtkinter as ctk

from positronic.app.theme import COLORS, FONTS, SECTION_GAP, CARD_GAP, _EMOJI, RESPONSIVE_BREAKPOINT
from positronic.app.ui.widgets import StatCard, MiningChart, InfoCard, ActivityPanel


# ── Helper: Section header with icon ──────────────────────────────
def _section_header(parent, icon: str, title: str):
    """Create a small section header row with icon + text."""
    hdr = ctk.CTkFrame(parent, fg_color="transparent")
    hdr.pack(fill="x", padx=16, pady=(SECTION_GAP, 4))
    ctk.CTkLabel(hdr, text=icon, font=(_EMOJI, 14),
                 text_color=COLORS["accent"]).pack(side="left")
    ctk.CTkLabel(hdr, text=title, font=FONTS["section_title"],
                 text_color=COLORS["text"]).pack(side="left", padx=(6, 0))
    return hdr


# ── Helper: Network status bar ────────────────────────────────────
def _build_status_bar(parent, win):
    """Build the top status/info bar with node health dot + refresh button."""
    bar = ctk.CTkFrame(parent, fg_color=COLORS["card_bg_elevated"],
                       corner_radius=10, border_width=1,
                       border_color=COLORS["card_border"], height=44)
    bar.pack(fill="x", padx=12, pady=(10, 0))
    bar.pack_propagate(False)

    inner = ctk.CTkFrame(bar, fg_color="transparent")
    inner.pack(fill="both", expand=True, padx=12)

    # Status dot + label
    win._dash_status_dot = ctk.CTkLabel(inner, text="\u25cf", width=18,
                                         font=("Segoe UI", 14),
                                         text_color=COLORS["success_green"])
    win._dash_status_dot.pack(side="left")
    win._dash_status_label = ctk.CTkLabel(inner, text="Node Healthy",
                                           font=FONTS["badge"],
                                           text_color=COLORS["success_green"])
    win._dash_status_label.pack(side="left", padx=(4, 0))

    # Separator dot
    ctk.CTkLabel(inner, text="\u00b7", font=FONTS["small"],
                 text_color=COLORS["text_muted"]).pack(side="left", padx=8)

    # Last updated
    win._dash_updated_lbl = ctk.CTkLabel(inner, text="",
                                          font=FONTS["tiny"],
                                          text_color=COLORS["text_muted"])
    win._dash_updated_lbl.pack(side="left")

    # Refresh button (right side)
    win._dash_refresh_btn = ctk.CTkButton(
        inner, text="\u27f3 Refresh", width=90, height=28,
        font=FONTS["badge"], corner_radius=6,
        fg_color=COLORS["bg_card"], hover_color=COLORS["bg_card_hover"],
        border_width=1, border_color=COLORS["border"],
        text_color=COLORS["accent"],
        command=lambda: _manual_refresh(win),
    )
    win._dash_refresh_btn.pack(side="right")


def build_dashboard(tab, win):
    """Build dashboard tab widgets. Stores references on *win* (MainWindow)."""

    # ── Top status bar ────────────────────────────────────────────
    _build_status_bar(tab, win)

    # ── Section: Network Overview ─────────────────────────────────
    _section_header(tab, "\u26a1", "Network Overview")

    # ── Stat cards row ────────────────────────────────────────────
    cards_frame = ctk.CTkFrame(tab, fg_color="transparent")
    cards_frame.pack(fill="x", padx=12, pady=(4, CARD_GAP))
    cards_frame.columnconfigure((0, 1, 2, 3, 4, 5), weight=1, uniform="card")

    win._card_block = StatCard(cards_frame, "Block Height", "\u26cf",
                               COLORS["accent"])
    win._card_block.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

    win._card_ai_acc = StatCard(cards_frame, "AI Engine", "\U0001f916",
                                 COLORS["accent_blue"])
    win._card_ai_acc.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)

    win._card_peers = StatCard(cards_frame, "Peers", "\U0001f517",
                               COLORS["success"])
    win._card_peers.grid(row=0, column=2, sticky="nsew", padx=4, pady=4)

    win._card_stake = StatCard(cards_frame, "Staked", "\U0001f48e",
                               COLORS["purple"])
    win._card_stake.grid(row=0, column=3, sticky="nsew", padx=4, pady=4)

    win._card_txs = StatCard(cards_frame, "Total TXs", "\U0001f4ca",
                              COLORS["info_blue"])
    win._card_txs.grid(row=0, column=4, sticky="nsew", padx=4, pady=4)

    win._card_reward = StatCard(cards_frame, "Uptime", "\u23f1",
                                COLORS["gold"])
    win._card_reward.grid(row=0, column=5, sticky="nsew", padx=4, pady=4)

    # Store card references for responsive layout
    _all_cards = [win._card_block, win._card_ai_acc, win._card_peers,
                  win._card_stake, win._card_txs, win._card_reward]

    def _relayout_cards(event=None, frame=cards_frame, cards=_all_cards):
        w = frame.winfo_width()
        if w < 50:  # Not yet rendered
            return
        cols = 3 if w < RESPONSIVE_BREAKPOINT else 6
        for c in range(6):
            frame.columnconfigure(c, weight=0, uniform="")
        for c in range(cols):
            frame.columnconfigure(c, weight=1, uniform="rcard")
        for i, card in enumerate(cards):
            card.grid(row=i // cols, column=i % cols, sticky="nsew", padx=4, pady=4)

    cards_frame.bind("<Configure>", _relayout_cards)

    # ── Section: Mining Activity ──────────────────────────────────
    _section_header(tab, "\u26cf", "Mining Activity")

    # Mining progress bar (compact bar above chart)
    prog_frame = ctk.CTkFrame(tab, fg_color=COLORS["bg_card"],
                               corner_radius=10, border_width=1,
                               border_color=COLORS["card_border"])
    prog_frame.pack(fill="x", padx=12, pady=(0, CARD_GAP))

    prog_inner = ctk.CTkFrame(prog_frame, fg_color="transparent")
    prog_inner.pack(fill="x", padx=16, pady=10)

    ctk.CTkLabel(prog_inner, text="Block Progress",
                 font=FONTS["small"],
                 text_color=COLORS["text_dim"]).pack(side="left")

    win._dash_mining_pct = ctk.CTkLabel(prog_inner, text="0%",
                                         font=FONTS["badge"],
                                         text_color=COLORS["accent"])
    win._dash_mining_pct.pack(side="right", padx=(8, 0))

    win._dash_mining_bar = ctk.CTkProgressBar(
        prog_frame, height=6, corner_radius=3,
        fg_color=COLORS["bg_dark"],
        progress_color=COLORS["accent"],
    )
    win._dash_mining_bar.pack(fill="x", padx=16, pady=(0, 10))
    win._dash_mining_bar.set(0)

    # Mining chart
    win._chart = MiningChart(tab)
    win._chart.pack(fill="both", expand=True, padx=12, pady=(0, CARD_GAP))

    # ── Bottom row: AI card + Activity panel ──────────────────────
    _section_header(tab, "\U0001f916", "AI & Activity")

    bottom = ctk.CTkFrame(tab, fg_color="transparent")
    bottom.pack(fill="x", padx=8, pady=(0, 12))
    bottom.columnconfigure(0, weight=1)
    bottom.columnconfigure(1, weight=2)

    win._ai_card = InfoCard(bottom, "AI Engine", "\U0001f916",
                            COLORS["accent_blue"])
    win._ai_card.grid(row=0, column=0, sticky="nsew", padx=4)
    for key, label in [
        ("ai_status", "Status"),
        ("ai_model", "Model"),
        ("ai_accuracy", "Accuracy"),
        ("ai_samples", "Samples"),
        ("ai_threats", "Threats"),
    ]:
        win._ai_card.add_row(key, label)

    win._activity = ActivityPanel(bottom, "Recent Activity")
    win._activity.grid(row=0, column=1, sticky="nsew", padx=4)

    # Store last-refresh timestamp
    win._dash_last_refresh_ts = 0.0


def _manual_refresh(win):
    """Triggered by the Refresh button — calls the poll cycle immediately."""
    if hasattr(win, "_poll_once"):
        win._poll_once()
    elif hasattr(win, "poll"):
        win.poll()


def _update_status_indicator(win, peers: int):
    """Update the top-bar status dot/label based on peer count."""
    from positronic.app.ui.animations import pulse_dot, stop_pulse

    if peers >= 1:
        color = COLORS["success_green"]
        label = "Online"
    else:
        color = COLORS["warning_yellow"]
        label = "Connecting..."
    if hasattr(win, "_dash_status_dot"):
        win._dash_status_dot.configure(text_color=color)
        if peers >= 1:
            pulse_dot(win._dash_status_dot, COLORS["success_green"], "#0a3d1a", period_ms=3000)
        else:
            stop_pulse(win._dash_status_dot)
    if hasattr(win, "_dash_status_label"):
        win._dash_status_label.configure(text=label, text_color=color)


def refresh_dashboard(win, data):
    """Update dashboard widgets from *data* dict."""
    # ── Stat cards ────────────────────────────────────────────────
    h = data.get("block_height", 0)
    win._card_block.set_value(f"#{h:,}" if h else "--")

    peers = data.get("peers", 0)
    mx_peers = data.get("max_peers", 12)
    peers_max_str = "\u221e" if mx_peers == 0 else str(mx_peers)
    win._card_peers.set_value(f"{peers} / {peers_max_str}",
                           "healthy" if peers >= 1 else "connecting...")

    # Update status indicator
    _update_status_indicator(win, peers)

    # Show remote RPC indicator if using remote fallback
    is_remote = data.get("_remote", False)
    node_status = data.get("_node_status", "")
    if is_remote and hasattr(win, "_dash_status_label"):
        # Override status label to show remote mode
        win._dash_status_label.configure(
            text=f"\U0001f4e1 Remote RPC{' \u2014 ' + node_status if node_status else ''}",
            text_color=COLORS.get("info_blue", COLORS["accent_blue"]))
        if hasattr(win, "_dash_status_dot"):
            win._dash_status_dot.configure(text_color=COLORS["accent_blue"])

    # Use user's personal stake if available (from validator poll data)
    your_stake = data.get("_your_stake", 0)
    your_is_val = data.get("_your_is_validator", False)
    if your_stake > 0:
        win._card_stake.set_value(f"{your_stake:,.4f} ASF",
                               "\u2705 Validator Active" if your_is_val else "Staked")
    else:
        win._card_stake.set_value("0.000 ASF", "Not staking")
    win._card_reward.set_value(data.get("uptime", "--"))

    total_txs = data.get("total_txs", 0)
    win._card_txs.set_value(f"{total_txs:,}" if total_txs else "--")

    ai_acc = data.get("ai_accuracy", "--")
    win._card_ai_acc.set_value(ai_acc,
                            "Enabled" if data.get("ai_enabled") else "Disabled")

    # ── Mining progress bar ───────────────────────────────────────
    history = data.get("mining_history", [])
    if history and hasattr(win, "_dash_mining_bar"):
        # Derive progress as fraction of latest vs max in window
        latest = history[-1] if history else 0
        mx = max(history) if history else 1
        pct = min(latest / max(mx, 1), 1.0)
        win._dash_mining_bar.set(pct)
        win._dash_mining_pct.configure(text=f"{pct * 100:.0f}%")

    # Mining chart
    if history:
        win._chart.set_data(history)

    # ── AI card ───────────────────────────────────────────────────
    win._ai_card.set("ai_status",
                     "\u2705 Enabled" if data.get("ai_enabled") else "Disabled")
    win._ai_card.set("ai_model", data.get("ai_model", "--"))
    win._ai_card.set("ai_accuracy", data.get("ai_accuracy", "--"))
    samples = data.get("ai_samples", 0)
    win._ai_card.set("ai_samples", f"{samples:,}" if samples else "--")
    win._ai_card.set("ai_threats", str(data.get("ai_threats", 0)))

    # ── Activity panel ────────────────────────────────────────────
    activity = data.get("activity", [])
    if activity:
        _enrich_activity_icons(activity)
    win._activity.set_items(activity)

    # ── Update "Last updated" timestamp ───────────────────────────
    win._dash_last_refresh_ts = time.time()
    ts_str = time.strftime("%H:%M:%S")
    if hasattr(win, "_dash_updated_lbl"):
        win._dash_updated_lbl.configure(text=f"Last updated: {ts_str}")


# ── TX type icon enrichment ─────────────────────────────────────────
_TX_TYPE_ICONS = {
    "sent":     "\u2b06",      # upward arrow Sent
    "send":     "\u2b06",
    "received": "\u2b07",      # downward arrow Received
    "receive":  "\u2b07",
    "stake":    "\u2b50",      # star Stake
    "reward":   "\U0001f3c6",  # trophy Reward
    "unstake":  "\U0001f513",  # unlocked Unstake
}


def _enrich_activity_icons(items: list):
    """Prefix activity text with TX-type icon when recognizable."""
    for item in items:
        text = item.get("text", "")
        text_lower = text.lower()
        for keyword, icon in _TX_TYPE_ICONS.items():
            if keyword in text_lower and not text.startswith(icon):
                item["text"] = f"{icon} {text}"
                break
