"""Ecosystem tab — neural AI, consensus, DID, governance, bridge, etc.

Features: dashboard-style grid cards, status color coding, hover tooltips,
colored top borders per category, clean section titles with Unicode icons.
"""

import customtkinter as ctk

from positronic.app.theme import COLORS, FONTS, _EMOJI
from positronic.app.ui.widgets import InfoCard


# ── Tooltip helper ──────────────────────────────────────────────────
_TOOLTIPS = {
    "Cold Start Phase": "AI models calibrate during first 500 transactions",
    "Degradation Level": "System gracefully reduces AI complexity under load",
    "ZKML Proofs": "Zero-knowledge proofs that verify AI model outputs",
    "Post-Quantum": "Lattice-based cryptography resistant to quantum attacks",
    "Drift Alerts": "Warnings when AI model outputs deviate from baseline",
    "Active Pathways": "Neural inference pipelines currently processing",
    "Trust Profiles": "On-chain reputation scores for network participants",
    "PRC-20 Tokens": "Positronic token standard (similar to ERC-20)",
}


class _Tooltip:
    """Lightweight hover tooltip for any widget."""

    def __init__(self, widget, text: str):
        self._widget = widget
        self._text = text
        self._tw = None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")

    def _show(self, _event=None):
        if self._tw:
            return
        x = self._widget.winfo_rootx() + 20
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 4
        self._tw = tw = ctk.CTkToplevel(self._widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tw.attributes("-topmost", True)
        lbl = ctk.CTkLabel(tw, text=self._text,
                           font=FONTS["small"],
                           fg_color=COLORS["bg_dark"],
                           text_color=COLORS["accent"],
                           corner_radius=6,
                           padx=10, pady=6)
        lbl.pack()

    def _hide(self, _event=None):
        if self._tw:
            self._tw.destroy()
            self._tw = None


# ── Section header ────────────────────────────────────────────────
class _SectionHeader(ctk.CTkFrame):
    """Clean section divider with icon and title — Grafana / Launchpad style."""

    def __init__(self, master, title: str, icon: str, accent: str, **kw):
        super().__init__(master, fg_color="transparent", **kw)

        # Left accent bar
        bar = ctk.CTkFrame(self, fg_color=accent, width=3, height=20,
                           corner_radius=2)
        bar.pack(side="left", padx=(0, 10), pady=2)

        ctk.CTkLabel(self, text=f"{icon}  {title}",
                     font=FONTS["section_title"],
                     text_color=accent).pack(side="left")

        # Subtle separator line extending right
        sep = ctk.CTkFrame(self, fg_color=COLORS["separator"], height=1)
        sep.pack(side="left", fill="x", expand=True, padx=(12, 0), pady=1)


# ── Dashboard InfoCard (enhanced) ──────────────────────────────────
class _DashCard(ctk.CTkFrame):
    """Enhanced InfoCard with colored top border, hover glow, and
    cleaner spacing — inspired by Grafana stat panels."""

    def __init__(self, master, title: str, icon: str,
                 accent: str, **kw):
        super().__init__(master, fg_color=COLORS["bg_card"],
                         corner_radius=10, border_width=1,
                         border_color=COLORS["card_border"], **kw)

        # Colored top border
        top_bar = ctk.CTkFrame(self, fg_color=accent, height=3,
                               corner_radius=0)
        top_bar.pack(fill="x", padx=1, pady=(1, 0))

        # Header row
        hdr = ctk.CTkFrame(self, fg_color="transparent")
        hdr.pack(fill="x", padx=14, pady=(10, 6))
        ctk.CTkLabel(hdr, text=icon, font=(_EMOJI, 13),
                     text_color=accent).pack(side="left")
        ctk.CTkLabel(hdr, text=title, font=FONTS["subheading"],
                     text_color=COLORS["text"]).pack(side="left", padx=(6, 0))

        # Body for key-value rows
        self._body = ctk.CTkFrame(self, fg_color="transparent")
        self._body.pack(fill="both", expand=True, padx=14, pady=(0, 10))

        self._labels: dict = {}

        # Hover effect
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self._accent = accent

    def _on_enter(self, _e=None):
        self.configure(border_color=self._accent)

    def _on_leave(self, _e=None):
        self.configure(border_color=COLORS["card_border"])

    def add_row(self, key: str, label: str):
        """Add a key-value row with subtle separator."""
        row = ctk.CTkFrame(self._body, fg_color="transparent")
        row.pack(fill="x", pady=(2, 2))

        ctk.CTkLabel(row, text=label, font=FONTS["small"],
                     text_color=COLORS["text_dim"], width=130,
                     anchor="w").pack(side="left")
        val = ctk.CTkLabel(row, text="\u2014", font=FONTS["mono"],
                           text_color=COLORS["text"], anchor="e")
        val.pack(side="right", padx=(0, 2))
        self._labels[key] = val

    def set(self, key: str, value: str):
        lbl = self._labels.get(key)
        if lbl:
            val_str = str(value)
            lbl.configure(text=val_str)
            val_upper = val_str.upper()
            if any(kw in val_upper for kw in ("OFFLINE", "DISABLED", "ERROR", "FAIL")):
                lbl.configure(text_color=COLORS["danger"])
            elif val_str.strip() == "0" or val_str.strip() == "--" or val_str.strip() == "\u2014":
                lbl.configure(text_color=COLORS["text_muted"])
            elif any(kw in val_upper for kw in ("ACTIVE", "ENABLED", "ONLINE", "NORMAL", "\u2705")):
                lbl.configure(text_color=COLORS["success"])
            else:
                lbl.configure(text_color=COLORS["text"])


def _add_dash_card(parent, win, attr_name, title, icon, accent, row, col, rows_data):
    """Helper to create a DashCard inside a grid with tooltips."""
    card = _DashCard(parent, title, icon, accent)
    card.grid(row=row, column=col, sticky="nsew", padx=5, pady=5)
    setattr(win, attr_name, card)
    for key, label in rows_data:
        card.add_row(key, label)
        if label in _TOOLTIPS:
            val_lbl = card._labels.get(key)
            if val_lbl:
                _Tooltip(val_lbl.master, _TOOLTIPS[label])
    return card


def build_ecosystem(tab, win):
    """Build ecosystem tab widgets. Stores references on *win*."""
    # Scrollable frame for all ecosystem cards
    scroll = ctk.CTkScrollableFrame(tab, fg_color="transparent")
    scroll.pack(fill="both", expand=True, padx=10, pady=10)
    scroll.columnconfigure(0, weight=1)

    # ════════════════════════════════════════════════════════════
    # SECTION 1: AI & Security
    # ════════════════════════════════════════════════════════════
    _SectionHeader(scroll, "AI & Security", "\U0001f9e0",
                   COLORS["accent_blue"]).pack(fill="x", padx=4, pady=(4, 6))

    grid1 = ctk.CTkFrame(scroll, fg_color="transparent")
    grid1.pack(fill="x", padx=0, pady=(0, 8))
    grid1.columnconfigure((0, 1, 2), weight=1, uniform="eco")

    _add_dash_card(grid1, win, "_eco_neural",
                   "AI Engine", "\U0001f9e0", COLORS["accent_blue"],
                   0, 0, [
                       ("neural_status", "Status"),
                       ("neural_degradation", "Degradation Level"),
                       ("neural_pathways", "Active Pathways"),
                       ("neural_drift", "Drift Alerts"),
                       ("cold_start", "Cold Start Phase"),
                   ])

    _add_dash_card(grid1, win, "_eco_consensus",
                   "Consensus", "\U0001f6e1", COLORS["purple"],
                   0, 1, [
                       ("cons_validators", "Validators"),
                       ("cons_epoch", "Current Epoch"),
                       ("cons_finalized", "Finalized Block"),
                       ("pq_enabled", "Post-Quantum"),
                       ("pq_keys", "PQ Keys Issued"),
                       ("immune_threats", "Threats Detected"),
                       ("immune_blocked", "Threats Blocked"),
                   ])

    _add_dash_card(grid1, win, "_eco_checkpoint",
                   "Checkpoints", "\U0001f4cc", COLORS["gold"],
                   0, 2, [
                       ("ckpt_latest", "Latest Height"),
                       ("ckpt_count", "Total Checkpoints"),
                   ])

    # ════════════════════════════════════════════════════════════
    # SECTION 2: Identity & Governance
    # ════════════════════════════════════════════════════════════
    _SectionHeader(scroll, "Identity & Governance", "\U0001f194",
                   COLORS["accent"]).pack(fill="x", padx=4, pady=(10, 6))

    grid2 = ctk.CTkFrame(scroll, fg_color="transparent")
    grid2.pack(fill="x", padx=0, pady=(0, 8))
    grid2.columnconfigure((0, 1, 2), weight=1, uniform="eco")

    _add_dash_card(grid2, win, "_eco_did",
                   "Decentralized Identity", "\U0001f194", COLORS["accent"],
                   0, 0, [
                       ("did_total", "Total DIDs"),
                       ("did_active", "Active DIDs"),
                       ("did_credentials", "Credentials Issued"),
                   ])

    _add_dash_card(grid2, win, "_eco_gov",
                   "Governance", "\U0001f3db", COLORS["warning"],
                   0, 1, [
                       ("gov_proposals", "Total Proposals"),
                       ("gov_pending", "Pending Proposals"),
                       ("gov_participation", "Participation Rate"),
                   ])

    _add_dash_card(grid2, win, "_eco_trust",
                   "Trust & Tokens", "\u2b50", COLORS["gold"],
                   0, 2, [
                       ("trust_profiles", "Trust Profiles"),
                       ("prc20_tokens", "PRC-20 Tokens"),
                       ("nft_collections", "NFT Collections"),
                   ])

    # ════════════════════════════════════════════════════════════
    # SECTION 3: Infrastructure
    # ════════════════════════════════════════════════════════════
    _SectionHeader(scroll, "Infrastructure", "\U0001f309",
                   COLORS["success"]).pack(fill="x", padx=4, pady=(10, 6))

    grid3 = ctk.CTkFrame(scroll, fg_color="transparent")
    grid3.pack(fill="x", padx=0, pady=(0, 8))
    grid3.columnconfigure((0, 1, 2), weight=1, uniform="eco")

    _add_dash_card(grid3, win, "_eco_bridge",
                   "Cross-Chain Bridge", "\U0001f309", COLORS["accent_blue"],
                   0, 0, [
                       ("bridge_locked", "Total Locked"),
                       ("bridge_transfers", "Total Transfers"),
                   ])

    _add_dash_card(grid3, win, "_eco_depin",
                   "DePIN Network", "\U0001f4e1", COLORS["success"],
                   0, 1, [
                       ("depin_devices", "Total Devices"),
                       ("depin_active", "Active Devices"),
                   ])

    _add_dash_card(grid3, win, "_eco_rwa",
                   "RWA Tokenization", "\U0001f3e2", COLORS["warning"],
                   0, 2, [
                       ("rwa_assets", "Tokenized Assets"),
                       ("rwa_value", "Total Value"),
                   ])

    # ════════════════════════════════════════════════════════════
    # SECTION 4: Advanced
    # ════════════════════════════════════════════════════════════
    _SectionHeader(scroll, "Advanced", "\U0001f510",
                   COLORS["purple"]).pack(fill="x", padx=4, pady=(10, 6))

    grid4 = ctk.CTkFrame(scroll, fg_color="transparent")
    grid4.pack(fill="x", padx=0, pady=(0, 8))
    grid4.columnconfigure((0, 1, 2), weight=1, uniform="eco")

    _add_dash_card(grid4, win, "_eco_agents",
                   "AI Agents", "\U0001f916", COLORS["purple"],
                   0, 0, [
                       ("agents_total", "Total Agents"),
                       ("agents_active", "Active Agents"),
                   ])

    _add_dash_card(grid4, win, "_eco_mkt",
                   "Agent Marketplace", "\U0001f3ea", COLORS["accent"],
                   0, 1, [
                       ("mkt_agents", "Listed Agents"),
                       ("mkt_tasks", "Completed Tasks"),
                   ])

    _add_dash_card(grid4, win, "_eco_zkml",
                   "ZKML Proofs", "\U0001f510", COLORS["accent_blue"],
                   0, 2, [
                       ("zkml_proofs", "Total Proofs"),
                       ("zkml_verified", "Verified Proofs"),
                   ])

    # Store grid references for potential future use
    win._eco_groups = [grid1, grid2, grid3, grid4]


def refresh_ecosystem(win, eco):
    """Update all ecosystem cards from pre-fetched data."""
    if eco is None:
        return
    try:
        _refresh_ecosystem_inner(win, eco)
    except Exception as e:
        import logging
        logging.getLogger("positronic.app.ui").debug("Ecosystem refresh error: %s", e)


def _refresh_ecosystem_inner(win, eco):
    """Inner refresh logic — called by refresh_ecosystem inside try/except."""
    # Neural AI
    _ns = eco.get("neural_status", "--")
    win._eco_neural.set("neural_status", _ns.upper() if isinstance(_ns, str) else str(_ns))
    deg = eco.get("neural_degradation_level", 0)
    win._eco_neural.set("neural_degradation",
                        f"Level {deg}" if deg else "Normal")
    win._eco_neural.set("neural_pathways",
                        str(eco.get("neural_pathways", 0)))
    win._eco_neural.set("neural_drift",
                        str(eco.get("neural_drift_alerts", 0)))
    _cs = eco.get("cold_start_phase", "--")
    win._eco_neural.set("cold_start", _cs.upper() if isinstance(_cs, str) else str(_cs))

    # Consensus & Security
    win._eco_consensus.set("cons_validators",
                           str(eco.get("consensus_validators", 0)))
    win._eco_consensus.set("cons_epoch",
                           str(eco.get("consensus_epoch", 0)))
    win._eco_consensus.set("cons_finalized",
                           f"#{eco.get('consensus_finalized', 0):,}")
    win._eco_consensus.set("pq_enabled",
                           "\u2705 Active" if eco.get("pq_enabled") else "Disabled")
    win._eco_consensus.set("pq_keys", str(eco.get("pq_keys", 0)))
    win._eco_consensus.set("immune_threats",
                           str(eco.get("immune_threats", 0)))
    win._eco_consensus.set("immune_blocked",
                           str(eco.get("immune_blocked", 0)))

    # Checkpoints
    win._eco_checkpoint.set("ckpt_latest",
                            f"#{eco.get('checkpoint_latest', 0):,}")
    win._eco_checkpoint.set("ckpt_count",
                            str(eco.get("checkpoint_count", 0)))

    # DID
    win._eco_did.set("did_total", str(eco.get("did_total", 0)))
    win._eco_did.set("did_active", str(eco.get("did_active", 0)))
    win._eco_did.set("did_credentials", str(eco.get("did_credentials", 0)))

    # Governance
    win._eco_gov.set("gov_proposals", str(eco.get("gov_proposals", 0)))
    win._eco_gov.set("gov_pending", str(eco.get("gov_pending", 0)))
    rate = eco.get("gov_participation", 0)
    win._eco_gov.set("gov_participation",
                     f"{rate:.1f}%" if isinstance(rate, (int, float)) else "--")

    # Trust & Tokens
    win._eco_trust.set("trust_profiles", str(eco.get("trust_profiles", 0)))
    win._eco_trust.set("prc20_tokens", str(eco.get("prc20_tokens", 0)))
    win._eco_trust.set("nft_collections", str(eco.get("nft_collections", 0)))

    # Bridge
    locked = eco.get("bridge_locked", 0)
    win._eco_bridge.set("bridge_locked",
                        f"{locked / (10 ** 18):,.3f} ASF" if locked else "0")
    win._eco_bridge.set("bridge_transfers",
                        str(eco.get("bridge_transfers", 0)))

    # DePIN
    win._eco_depin.set("depin_devices", str(eco.get("depin_devices", 0)))
    win._eco_depin.set("depin_active", str(eco.get("depin_active", 0)))

    # RWA
    win._eco_rwa.set("rwa_assets", str(eco.get("rwa_assets", 0)))
    rwa_val = eco.get("rwa_value", 0)
    win._eco_rwa.set("rwa_value",
                     f"${rwa_val:,.2f}" if rwa_val else "$0")

    # Agents
    win._eco_agents.set("agents_total", str(eco.get("agents_total", 0)))
    win._eco_agents.set("agents_active", str(eco.get("agents_active", 0)))

    # Marketplace
    win._eco_mkt.set("mkt_agents", str(eco.get("mkt_agents", 0)))
    win._eco_mkt.set("mkt_tasks", str(eco.get("mkt_tasks", 0)))

    # ZKML
    win._eco_zkml.set("zkml_proofs", str(eco.get("zkml_proofs", 0)))
    win._eco_zkml.set("zkml_verified", str(eco.get("zkml_verified", 0)))
