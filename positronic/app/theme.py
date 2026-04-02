"""Visual theme constants for the Positronic desktop application (CustomTkinter)."""

import sys

# ── Color palette (matches website CSS variables) ─────────────────
COLORS = {
    "bg_darkest": "#060D1A",
    "bg_dark": "#0D1B2A",
    "bg_card": "#112240",
    "bg_card_hover": "#1a3a5c",
    "border": "#233554",
    "accent": "#00E5FF",      # --positronic-cyan
    "accent_blue": "#0080FF",  # --positronic-blue
    "purple": "#7B2FF7",
    "text": "#FFFFFF",
    "text_dim": "#8892B0",
    "text_muted": "#4A5568",
    "success": "#22c55e",
    "danger": "#ef4444",
    "warning": "#f59e0b",
    "gold": "#F5A623",
    "chart_fill": "#0a2a3f",
    "chart_glow": "#005577",
    "chart_grid": "#1a3a5c",
    # ── Status accent colors ──
    "success_green": "#22c55e",
    "warning_yellow": "#f59e0b",
    "error_red": "#ef4444",
    "info_blue": "#3b82f6",
    # ── Card depth / glow ──
    "card_border": "#1e3a5f",
    "card_bg_elevated": "#152a4a",
    "glow_green": "#134e2a",
    "glow_red": "#4a1212",
    "glow_purple": "#2d1854",
    "glow_cyan": "#0a3040",
    "separator": "#1c3350",
}

# ── Light-mode color variants (for future light theme support) ────
COLORS_LIGHT = {
    "bg_darkest": "#F0F2F5",
    "bg_dark": "#FFFFFF",
    "bg_card": "#F8FAFC",
    "bg_card_hover": "#EDF2F7",
    "border": "#CBD5E0",
    "accent": "#0080FF",
    "accent_blue": "#0066CC",
    "purple": "#7B2FF7",
    "text": "#1A202C",
    "text_dim": "#4A5568",
    "text_muted": "#A0AEC0",
    "success": "#16a34a",
    "danger": "#dc2626",
    "warning": "#d97706",
    "gold": "#b45309",
    "chart_fill": "#EBF8FF",
    "chart_glow": "#BEE3F8",
    "chart_grid": "#E2E8F0",
    # ── Status accent colors ──
    "success_green": "#16a34a",
    "warning_yellow": "#d97706",
    "error_red": "#dc2626",
    "info_blue": "#2563eb",
    # ── Card depth / glow ──
    "card_border": "#CBD5E0",
    "card_bg_elevated": "#FFFFFF",
    "glow_green": "#dcfce7",
    "glow_red": "#fee2e2",
    "glow_purple": "#f3e8ff",
    "glow_cyan": "#e0f2fe",
    "separator": "#E2E8F0",
}

# ── Layout constants ─────────────────────────────────────────────
CARD_PAD = 16
CARD_GAP = 8
SECTION_GAP = 16
BTN_HEIGHT = 40

# ── Cross-platform fonts ──────────────────────────────────────────
_SANS = "Segoe UI" if sys.platform == "win32" else "SF Pro Display" if sys.platform == "darwin" else "Inter"
_MONO = "Cascadia Code" if sys.platform == "win32" else "SF Mono" if sys.platform == "darwin" else "JetBrains Mono"
_EMOJI = "Segoe UI Emoji" if sys.platform == "win32" else "Apple Color Emoji" if sys.platform == "darwin" else "Noto Color Emoji"

FONTS = {
    "app_title": (_SANS, 18, "bold"),
    "heading": (_SANS, 16, "bold"),
    "subheading": (_SANS, 13, "bold"),
    "body": (_SANS, 12),
    "small": (_SANS, 11),
    "tiny": (_SANS, 10),
    "stat_value": (_SANS, 26, "bold"),
    "stat_label": (_SANS, 11),
    "stat_sub": (_SANS, 10),
    "button": (_SANS, 12, "bold"),
    "mono": (_MONO, 11),
    "mono_small": (_MONO, 10),
    "mono_tiny": (_MONO, 9),
    "tab": (_SANS, 13, "bold"),
    "input": (_SANS, 12),
    "hero_value": (_SANS, 32, "bold"),
    "section_title": (_SANS, 14, "bold"),
    "badge": (_SANS, 10, "bold"),
    "emoji": (_EMOJI, 14),
    "emoji_large": (_EMOJI, 18),
    "emoji_small": (_EMOJI, 12),
}

# ── Window defaults ───────────────────────────────────────────────
WINDOW_MIN_WIDTH = 900
WINDOW_MIN_HEIGHT = 640
WINDOW_DEFAULT_WIDTH = 1100
WINDOW_DEFAULT_HEIGHT = 750
POLL_INTERVAL_MS = 5000  # 5s — reduces thread pressure for long-running stability

# Layout breakpoints
RESPONSIVE_BREAKPOINT = 700  # px — switch from 6-col to 3-col cards
CARD_MIN_WIDTH = 140  # px — minimum width before cards stack

# Data limits
CHART_HISTORY_SIZE = 60  # data points in mining chart
LOG_TRIM_THRESHOLD = 500  # trim logs when exceeding this
LOG_TRIM_KEEP = 400  # lines to keep after trim
MAX_DISPLAY_PEERS = 12

# Dialog sizes (base, scaled by DPI)
DIALOG_SM = (440, 200)
DIALOG_MD = (500, 240)
DIALOG_LG = (520, 350)
DIALOG_XL = (460, 380)
