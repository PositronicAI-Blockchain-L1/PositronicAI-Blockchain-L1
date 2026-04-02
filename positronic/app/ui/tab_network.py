"""Network tab — network status card, peer list, sync progress, mempool detail."""

import customtkinter as ctk

from positronic.app.theme import COLORS, FONTS, _EMOJI
from positronic.app.ui.widgets import InfoCard


def _copy_text(root, text, btn=None, original_text=""):
    """Copy text to clipboard with visual feedback."""
    try:
        root.clipboard_clear()
        root.clipboard_append(text)
        if btn and original_text:
            btn.configure(text="\u2705")
            root.after(1500, lambda: btn.configure(text=original_text))
    except Exception:
        pass


def _truncate_id(peer_id: str, front: int = 6, back: int = 4) -> str:
    """Truncate a peer ID for display."""
    s = str(peer_id)
    if len(s) > front + back + 3:
        return s[:front] + ".." + s[-back:]
    return s


def _mask_address(addr: str) -> str:
    """Mask IP address for privacy: 89.167.96.119:9000 → 89.*.*.*:9000"""
    if not addr or addr in ("--", "\u2014"):
        return addr
    # Split off port if present
    port = ""
    host = addr
    if ":" in addr:
        parts = addr.rsplit(":", 1)
        host = parts[0]
        port = ":" + parts[1]
    # Mask middle octets of IPv4
    octets = host.split(".")
    if len(octets) == 4:
        return f"{octets[0]}.*.*.*{port}"
    return addr


# ────────────────────────────────────────────────────────────────────
#  Stat mini-card for the top row
# ────────────────────────────────────────────────────────────────────

def _make_stat_mini(parent, icon, label, accent=COLORS["accent"]):
    """Create a compact stat card. Returns (frame, value_label)."""
    card = ctk.CTkFrame(parent, fg_color=COLORS["bg_card"],
                        corner_radius=10, border_width=1,
                        border_color=COLORS["border"])

    inner = ctk.CTkFrame(card, fg_color="transparent")
    inner.pack(fill="both", expand=True, padx=14, pady=12)

    top_row = ctk.CTkFrame(inner, fg_color="transparent")
    top_row.pack(fill="x")
    ctk.CTkLabel(top_row, text=icon, font=(_EMOJI, 14),
                 text_color=accent).pack(side="left")
    ctk.CTkLabel(top_row, text=label, font=FONTS["tiny"],
                 text_color=COLORS["text_muted"]).pack(side="left", padx=(6, 0))

    val_lbl = ctk.CTkLabel(inner, text="--",
                           font=("Segoe UI", 20, "bold"),
                           text_color=COLORS["text"])
    val_lbl.pack(anchor="w", pady=(4, 0))

    # Accent bar at bottom
    ctk.CTkFrame(card, fg_color=accent, height=2,
                 corner_radius=0).pack(fill="x", side="bottom", padx=1, pady=(0, 1))

    return card, val_lbl


def build_network(tab, win):
    """Build network tab widgets. Stores references on *win*."""

    # ── Scrollable container ───────────────────────────────────────
    scroll = ctk.CTkScrollableFrame(tab, fg_color="transparent")
    scroll.pack(fill="both", expand=True, padx=0, pady=0)

    # ══════════════════════════════════════════════════════════════
    #  Network Stats Cards (top row — 4 cards)
    # ══════════════════════════════════════════════════════════════
    stats_row = ctk.CTkFrame(scroll, fg_color="transparent")
    stats_row.pack(fill="x", padx=16, pady=(16, 8))

    card1, win._stat_peers = _make_stat_mini(stats_row, "\U0001f465", "Total Peers", COLORS["accent"])
    card1.pack(side="left", expand=True, fill="both", padx=(0, 6))

    card2, win._stat_latency = _make_stat_mini(stats_row, "\u23f1", "Avg Latency", COLORS["success"])
    card2.pack(side="left", expand=True, fill="both", padx=(0, 6))

    card3, win._stat_banned = _make_stat_mini(stats_row, "\u26d4", "Banned", COLORS["danger"])
    card3.pack(side="left", expand=True, fill="both", padx=(0, 6))

    card4, win._stat_sync = _make_stat_mini(stats_row, "\u26a1", "Sync Status", COLORS["warning"])
    card4.pack(side="left", expand=True, fill="both", padx=0)

    # ══════════════════════════════════════════════════════════════
    #  P2P Node Info
    # ══════════════════════════════════════════════════════════════
    p2p_card = ctk.CTkFrame(scroll, fg_color=COLORS["bg_card"],
                            corner_radius=12, border_width=1,
                            border_color=COLORS["border"])
    p2p_card.pack(fill="x", padx=16, pady=(8, 8))

    p2p_hdr = ctk.CTkFrame(p2p_card, fg_color="transparent")
    p2p_hdr.pack(fill="x", padx=20, pady=(14, 6))
    ctk.CTkLabel(p2p_hdr, text="\U0001f310",
                 font=(_EMOJI, 16),
                 text_color=COLORS["accent"]).pack(side="left")
    ctk.CTkLabel(p2p_hdr, text="Node Information",
                 font=FONTS["subheading"],
                 text_color=COLORS["text"]).pack(side="left", padx=(8, 0))

    # Info rows in a grid-like layout
    info_body = ctk.CTkFrame(p2p_card, fg_color="transparent")
    info_body.pack(fill="x", padx=20, pady=(0, 14))

    p2p_fields = [
        ("n_version", "Version"),
        ("n_chain_id", "Chain ID"),
        ("n_network", "Network"),
        ("n_p2p", "P2P Port"),
        ("n_rpc", "RPC Port"),
        ("n_block_time", "Block Time"),
        ("n_uptime", "Uptime"),
        ("n_mempool", "Mempool"),
    ]
    win._net_info_labels = {}
    for i, (key, label) in enumerate(p2p_fields):
        row = ctk.CTkFrame(info_body, fg_color="transparent")
        row.pack(fill="x", pady=2)
        ctk.CTkLabel(row, text=label, font=FONTS["small"],
                     text_color=COLORS["text_muted"], width=100,
                     anchor="w").pack(side="left")
        val = ctk.CTkLabel(row, text="--", font=FONTS["mono"],
                           text_color=COLORS["text"], anchor="w")
        val.pack(side="left", fill="x", expand=True)
        win._net_info_labels[key] = val

    # ══════════════════════════════════════════════════════════════
    #  Sync Progress Section
    # ══════════════════════════════════════════════════════════════
    sync_frame = ctk.CTkFrame(scroll, fg_color=COLORS["bg_card"],
                              corner_radius=12, border_width=1,
                              border_color=COLORS["border"])
    sync_frame.pack(fill="x", padx=16, pady=(8, 8))

    sync_hdr = ctk.CTkFrame(sync_frame, fg_color="transparent")
    sync_hdr.pack(fill="x", padx=20, pady=(14, 4))
    ctk.CTkLabel(sync_hdr, text="\U0001f504",
                 font=(_EMOJI, 16),
                 text_color=COLORS["accent"]).pack(side="left")
    ctk.CTkLabel(sync_hdr, text="Sync Progress",
                 font=FONTS["subheading"],
                 text_color=COLORS["text"]).pack(side="left", padx=(8, 0))

    win._sync_status_label = ctk.CTkLabel(
        sync_hdr, text="\u2714 Fully Synced", font=FONTS["small"],
        text_color=COLORS["success"])
    win._sync_status_label.pack(side="right", padx=(0, 4))

    win._sync_bar = ctk.CTkProgressBar(
        sync_frame, fg_color=COLORS["bg_darkest"],
        progress_color=COLORS["accent"],
        border_color=COLORS["border"],
        height=16, corner_radius=8)
    win._sync_bar.pack(fill="x", padx=20, pady=(6, 4))
    win._sync_bar.set(1.0)

    win._sync_detail_label = ctk.CTkLabel(
        sync_frame, text="", font=FONTS["tiny"],
        text_color=COLORS["text_muted"])
    win._sync_detail_label.pack(anchor="w", padx=20, pady=(0, 12))

    # ── Peer Connectivity Bar ──────────────────────────────────────
    conn_card = ctk.CTkFrame(scroll, fg_color=COLORS["bg_card"],
                             corner_radius=12, border_width=1,
                             border_color=COLORS["border"])
    conn_card.pack(fill="x", padx=16, pady=(8, 8))

    conn_hdr = ctk.CTkFrame(conn_card, fg_color="transparent")
    conn_hdr.pack(fill="x", padx=20, pady=(14, 4))
    ctk.CTkLabel(conn_hdr, text="\U0001f4e1",
                 font=(_EMOJI, 16),
                 text_color=COLORS["accent_blue"]).pack(side="left")
    win._peer_bar_label = ctk.CTkLabel(
        conn_hdr, text="Peer Connectivity", font=FONTS["subheading"],
        text_color=COLORS["text"])
    win._peer_bar_label.pack(side="left", padx=(8, 0))

    win._peer_bar = ctk.CTkProgressBar(
        conn_card, fg_color=COLORS["bg_darkest"],
        progress_color=COLORS["accent"],
        border_color=COLORS["border"],
        height=14, corner_radius=7)
    win._peer_bar.pack(fill="x", padx=20, pady=(6, 4))
    win._peer_bar.set(0)

    win._peer_bar_text = ctk.CTkLabel(
        conn_card, text="0 / 12 peers", font=FONTS["tiny"],
        text_color=COLORS["text_muted"])
    win._peer_bar_text.pack(anchor="w", padx=20, pady=(0, 12))

    # ══════════════════════════════════════════════════════════════
    #  Mempool Detail Card
    # ══════════════════════════════════════════════════════════════
    win._mempool_card = InfoCard(scroll, "Mempool Detail", "\U0001f4e6",
                                 COLORS["warning"])
    win._mempool_card.pack(fill="x", padx=16, pady=(8, 8))
    for key, label in [
        ("mp_pending", "Pending TXs"),
        ("mp_gas", "Total Gas"),
        ("mp_oldest", "Oldest TX"),
    ]:
        win._mempool_card.add_row(key, label)

    # ══════════════════════════════════════════════════════════════
    #  Peer List Table
    # ══════════════════════════════════════════════════════════════
    peer_section = ctk.CTkFrame(scroll, fg_color=COLORS["bg_card"],
                                corner_radius=12, border_width=1,
                                border_color=COLORS["border"])
    peer_section.pack(fill="x", padx=16, pady=(8, 16))

    peer_hdr = ctk.CTkFrame(peer_section, fg_color="transparent")
    peer_hdr.pack(fill="x", padx=20, pady=(14, 6))
    ctk.CTkLabel(peer_hdr, text="\U0001f465",
                 font=(_EMOJI, 16),
                 text_color=COLORS["accent"]).pack(side="left")
    ctk.CTkLabel(peer_hdr, text="Connected Peers",
                 font=FONTS["subheading"],
                 text_color=COLORS["text"]).pack(side="left", padx=(8, 0))
    win._peer_count_badge = ctk.CTkLabel(
        peer_hdr, text="0", font=FONTS["tiny"],
        text_color=COLORS["bg_darkest"],
        fg_color=COLORS["accent"], corner_radius=8, width=28, height=20)
    win._peer_count_badge.pack(side="left", padx=(8, 0))

    # Table header row
    tbl_hdr = ctk.CTkFrame(peer_section, fg_color=COLORS["bg_darkest"],
                            corner_radius=0, height=30)
    tbl_hdr.pack(fill="x", padx=12, pady=(4, 0))
    tbl_hdr.pack_propagate(False)
    for text, w in [("", 22), ("Peer ID", 120), ("Address", 160),
                    ("Latency", 80), ("Height", 80), ("Version", 100)]:
        ctk.CTkLabel(tbl_hdr, text=text, font=FONTS["tiny"],
                     text_color=COLORS["text_muted"], width=w,
                     anchor="w").pack(side="left", padx=(8, 0))

    # Scrollable peer rows
    win._peer_list_frame = ctk.CTkScrollableFrame(
        peer_section, fg_color="transparent", height=160)
    win._peer_list_frame.pack(fill="both", expand=True, padx=8, pady=(0, 12))

    win._peer_rows = []

    # ── Keep backward-compatible references (InfoCard used by old refresh) ──
    # Create a hidden InfoCard so refresh_network doesn't crash if it
    # tries to call win._net_card.set(). We map keys to the new labels.
    win._net_card = _NetCardShim(win)


class _NetCardShim:
    """Shim that forwards .set() calls to the new network info labels."""
    def __init__(self, win):
        self._win = win

    def set(self, key, value):
        lbl = self._win._net_info_labels.get(key)
        if lbl:
            val_str = str(value)
            lbl.configure(text=val_str)
            # Color coding
            val_upper = val_str.upper()
            if any(kw in val_upper for kw in ("OFFLINE", "DISABLED", "ERROR", "FAIL")):
                lbl.configure(text_color=COLORS["danger"])
            elif val_str.strip() == "0" or val_str.strip() == "--":
                lbl.configure(text_color=COLORS["text_muted"])
            elif any(kw in val_upper for kw in ("ACTIVE", "ENABLED", "ONLINE", "NORMAL", "\u2705")):
                lbl.configure(text_color=COLORS["success"])
            else:
                lbl.configure(text_color=COLORS["text"])

    def add_row(self, key, label):
        pass  # No-op: rows already built


def _build_peer_rows(win, peers):
    """Rebuild peer list rows from a list of peer dicts."""
    for w in win._peer_rows:
        w.destroy()
    win._peer_rows.clear()

    if not peers:
        empty = ctk.CTkLabel(win._peer_list_frame, text="No peers connected",
                             font=FONTS["small"],
                             text_color=COLORS["text_muted"])
        empty.pack(pady=16)
        win._peer_rows.append(empty)
        return

    for i, peer in enumerate(peers[:50]):
        bg = COLORS["bg_card"] if i % 2 == 0 else "transparent"
        row = ctk.CTkFrame(win._peer_list_frame, fg_color=bg,
                           corner_radius=4, height=30)
        row.pack(fill="x", pady=1)
        row.pack_propagate(False)

        # Status dot (green = connected)
        dot_frame = ctk.CTkFrame(row, fg_color="transparent", width=22)
        dot_frame.pack(side="left")
        dot_frame.pack_propagate(False)
        ctk.CTkLabel(dot_frame, text="\u25cf",
                     font=("Segoe UI", 10),
                     text_color=COLORS["success"]).pack(expand=True)

        # Peer ID (truncated)
        peer_id = str(peer.get("id", peer.get("peer_id", "?")))
        short_id = _truncate_id(peer_id)
        id_btn = ctk.CTkButton(
            row, text=short_id,
            font=("Cascadia Code", 9),
            fg_color="transparent",
            text_color=COLORS["accent"],
            hover_color=COLORS["bg_card_hover"],
            corner_radius=4, width=120, height=24, anchor="w",
            command=lambda pid=peer_id: _copy_text(
                win._peer_list_frame.winfo_toplevel(), pid))
        id_btn.pack(side="left", padx=(0, 0))

        # Address (masked for privacy)
        address = _mask_address(str(peer.get("address", peer.get("addr", "--"))))
        ctk.CTkLabel(row, text=address, font=("Cascadia Code", 9),
                     text_color=COLORS["text_dim"], width=160,
                     anchor="w").pack(side="left", padx=(8, 0))

        # Latency (color-coded: <50ms green, <200ms yellow, >200ms red)
        latency = peer.get("latency", peer.get("latency_ms", peer.get("rtt_ms", "--")))
        if isinstance(latency, (int, float)):
            lat_text = f"{latency:.0f} ms"
            if latency < 50:
                lat_color = COLORS["success"]
            elif latency < 200:
                lat_color = COLORS["warning"]
            else:
                lat_color = COLORS["danger"]
        else:
            lat_text = str(latency)
            lat_color = COLORS["text_muted"]
        ctk.CTkLabel(row, text=lat_text, font=("Cascadia Code", 9),
                     text_color=lat_color, width=80,
                     anchor="w").pack(side="left", padx=(8, 0))

        # Height
        height = peer.get("height", peer.get("block_height", "--"))
        ctk.CTkLabel(row, text=str(height), font=("Cascadia Code", 9),
                     text_color=COLORS["text_dim"], width=80,
                     anchor="w").pack(side="left", padx=(8, 0))

        # Version
        version = str(peer.get("version", "--"))
        ctk.CTkLabel(row, text=version, font=("Cascadia Code", 9),
                     text_color=COLORS["text_dim"], width=100,
                     anchor="w").pack(side="left", padx=(8, 0))

        win._peer_rows.append(row)


def refresh_network(win, net):
    """Update network tab from *net* dict."""
    if net.get("online"):
        p = net.get("peers", 0)
        mp = net.get("max_peers", 12)

        # ── Top stat cards ──
        win._stat_peers.configure(text=str(p))
        banned = net.get("banned_peers", net.get("banned", 0))
        win._stat_banned.configure(text=str(banned))
        if banned and int(banned) > 0:
            win._stat_banned.configure(text_color=COLORS["danger"])
        else:
            win._stat_banned.configure(text_color=COLORS["text"])

        synced = net.get("synced", False)
        win._stat_sync.configure(text="\u2714 Synced" if synced else "\U0001f504 Syncing")
        win._stat_sync.configure(
            text_color=COLORS["success"] if synced else COLORS["warning"])

        # ── Node info fields (via shim) ──
        # Note: "n_peers" is not a registered field in build_network's p2p_fields;
        # peer count is already shown in _stat_peers and the connectivity bar.
        win._net_card.set("n_network", net.get("network_type", "--").upper())
        win._net_card.set("n_p2p", str(net.get("p2p_port", "--")))
        win._net_card.set("n_rpc", str(net.get("rpc_port", "--")))
        win._net_card.set("n_mempool", str(net.get("mempool_size", 0)))

        health = net.get("health", {})
        win._net_card.set("n_uptime", str(health.get("uptime", "--")))
        from positronic import __version__
        win._net_card.set("n_version", f"v{__version__}")
        win._net_card.set("n_chain_id", "420420")
        win._net_card.set("n_block_time", "12s")

        # ── Average latency ──
        peer_list = net.get("peer_list", health.get("peer_list", []))
        latencies = []
        for pr in peer_list:
            lat = pr.get("latency", pr.get("latency_ms", pr.get("rtt_ms")))
            if isinstance(lat, (int, float)):
                latencies.append(lat)
        if latencies:
            avg = sum(latencies) / len(latencies)
            win._stat_latency.configure(text=f"{avg:.0f} ms")
            if avg < 50:
                win._stat_latency.configure(text_color=COLORS["success"])
            elif avg < 200:
                win._stat_latency.configure(text_color=COLORS["warning"])
            else:
                win._stat_latency.configure(text_color=COLORS["danger"])
        else:
            win._stat_latency.configure(text="-- ms", text_color=COLORS["text_muted"])

        # ── Peer connectivity bar ──
        win._peer_bar.set(min(p / (mp if mp > 0 else 50), 1.0))
        mp_str = "\u221e" if mp == 0 else str(mp)
        win._peer_bar_text.configure(text=f"{p} / {mp_str} peers")

        # ── Sync progress ──
        local_h = health.get("height", 0)
        # Only show Fully Synced if node has meaningful height and isn't freshly started
        truly_synced = synced and local_h > 10

        if truly_synced:
            win._sync_bar.set(1.0)
            win._sync_status_label.configure(
                text="\u2714 Fully Synced", text_color=COLORS["success"])
            height = health.get("height", "?")
            win._sync_detail_label.configure(
                text=f"Block height: {height}")
            win._sync_bar.configure(progress_color=COLORS["success"])
        else:
            target_h = net.get("target_height", 0)
            if target_h and target_h > 0:
                pct = min(local_h / target_h, 1.0)
            else:
                pct = 0.5
                target_h = "?"
            win._sync_bar.set(pct)
            pct_display = int(pct * 100)
            win._sync_status_label.configure(
                text=f"Syncing {pct_display}%", text_color=COLORS["warning"])
            win._sync_detail_label.configure(
                text=f"{local_h:,} / {target_h:,} blocks" if isinstance(target_h, int)
                else f"{local_h:,} blocks (target unknown)")
            win._sync_bar.configure(progress_color=COLORS["accent"])

        # ── Mempool detail ──
        mempool_size = net.get("mempool_size", 0)
        mempool_info = net.get("mempool_info", {})
        win._mempool_card.set("mp_pending", str(mempool_size))
        win._mempool_card.set("mp_gas",
                              str(mempool_info.get("total_gas", "--")))
        win._mempool_card.set("mp_oldest",
                              str(mempool_info.get("oldest_age", "--")))

        # ── Peer list ──
        if not peer_list and p > 0:
            peer_list = [{"id": "connecting...", "address": "\u2014",
                          "latency": "\u2014", "version": "\u2014"}]
        _build_peer_rows(win, peer_list)
        win._peer_count_badge.configure(text=str(p))
