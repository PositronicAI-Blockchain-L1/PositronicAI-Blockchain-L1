"""Reusable UI widgets: StatCard, MiningChart, ActivityPanel, InfoCard."""

import collections
import time
import tkinter as tk

import customtkinter as ctk

from positronic.app.theme import COLORS, FONTS, _EMOJI


class StatCard(ctk.CTkFrame):
    """Stat card with accent bar, icon, value, and subtitle."""

    def __init__(self, master, title: str, icon: str = "",
                 accent: str = COLORS["accent"], **kw):
        super().__init__(master, fg_color=COLORS["bg_card"],
                         corner_radius=12, border_width=1,
                         border_color=COLORS["border"], **kw)
        self.configure(width=160)

        pad = ctk.CTkFrame(self, fg_color="transparent")
        pad.pack(fill="both", expand=True, padx=16, pady=16)

        # Title row
        row = ctk.CTkFrame(pad, fg_color="transparent")
        row.pack(fill="x")
        if icon:
            ctk.CTkLabel(row, text=icon, font=(_EMOJI, 15),
                         text_color=accent).pack(side="left")
        ctk.CTkLabel(row, text=title, font=FONTS["stat_label"],
                     text_color=COLORS["text_dim"]).pack(side="left", padx=(6, 0))

        # Value — wraplength prevents clipping on narrow cards
        self._val = ctk.CTkLabel(pad, text="--", font=FONTS["stat_value"],
                                 text_color=COLORS["text"], wraplength=150)
        self._val.pack(anchor="w", pady=(4, 0))

        # Subtitle
        self._sub = ctk.CTkLabel(pad, text="", font=FONTS["stat_sub"],
                                 text_color=COLORS["text_muted"])
        self._sub.pack(anchor="w")

        # Accent underline bar (bottom)
        self._accent_bar = ctk.CTkFrame(self, fg_color=accent, height=3,
                                        corner_radius=0)
        self._accent_bar.pack(fill="x", side="bottom", padx=1, pady=(0, 1))

        self._accent_color = accent
        self._prev_value = None

    def set_value(self, value: str, subtitle: str = ""):
        """Update displayed value — smooth text update with subtle accent underline pulse."""
        changed = (value != self._prev_value and self._prev_value is not None)
        old = self._prev_value
        self._prev_value = value
        if subtitle:
            self._sub.configure(text=subtitle)
        if changed:
            # Try smooth numeric transition
            try:
                from positronic.app.ui.animations import count_transition
                count_transition(self._val, old or "", value, duration_ms=400)
            except Exception:
                self._val.configure(text=str(value))
            # Accent bar pulse
            self._accent_bar.configure(height=5)
            self.after(600, lambda: self._accent_bar.configure(height=3))
        else:
            self._val.configure(text=str(value))


class MiningChart(ctk.CTkFrame):
    """Canvas-based line chart for mining/block activity."""

    def __init__(self, master, **kw):
        super().__init__(master, fg_color=COLORS["bg_card"],
                         corner_radius=12, border_width=1,
                         border_color=COLORS["border"], **kw)

        # Header
        hdr = ctk.CTkFrame(self, fg_color="transparent")
        hdr.pack(fill="x", padx=16, pady=(12, 0))
        ctk.CTkLabel(hdr, text="Mining Activity",
                     font=FONTS["subheading"],
                     text_color=COLORS["text"]).pack(side="left")
        self._updated_lbl = ctk.CTkLabel(hdr, text="",
                                         font=FONTS["tiny"],
                                         text_color=COLORS["text_muted"])
        self._updated_lbl.pack(side="right", padx=(8, 0))
        self._period_lbl = ctk.CTkLabel(hdr, text="Last 60 polls",
                                        font=FONTS["tiny"],
                                        text_color=COLORS["text_dim"])
        self._period_lbl.pack(side="right")

        # Canvas
        self.canvas = tk.Canvas(self, bg=COLORS["bg_card"],
                                highlightthickness=0, height=160)
        self.canvas.pack(fill="both", expand=True, padx=16, pady=(8, 16))
        self.canvas.bind("<Configure>", self._on_resize)

        self.data: collections.deque = collections.deque(maxlen=60)
        self._last_update_ts: float = 0.0
        self._redraw_timer = None

    def set_data(self, data_list: list):
        """Replace chart data."""
        self.data = collections.deque(data_list, maxlen=60)
        self._last_update_ts = time.time()
        self._update_time_label()
        self._redraw()

    def _on_resize(self, event=None):
        """Debounced resize handler — avoids excessive redraws during window resize."""
        if self._redraw_timer:
            self.after_cancel(self._redraw_timer)
        self._redraw_timer = self.after(100, self._do_redraw)

    def _update_time_label(self):
        """Refresh the 'Last updated: Xs ago' label."""
        if self._last_update_ts > 0:
            elapsed = int(time.time() - self._last_update_ts)
            self._updated_lbl.configure(
                text=f"\u00b7 Updated {elapsed}s ago" if elapsed > 0
                else "\u00b7 Updated just now"
            )

    def _redraw(self):
        """Direct redraw (non-debounced) — called from set_data."""
        self._do_redraw()

    def _do_redraw(self):
        c = self.canvas
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        if w < 40 or h < 40:
            return

        # Also refresh the elapsed-time label on every redraw
        self._update_time_label()

        data = list(self.data)
        y_label_margin = 38  # space for Y-axis labels on the left
        x_label_margin = 16  # space for X-axis label at the bottom
        pt, pb, px = 15, 15 + x_label_margin, y_label_margin
        cw, ch = w - px - 15, h - pt - pb

        if not data or all(v == 0 for v in data):
            c.create_text(w // 2, h // 2, text="Waiting for blocks...",
                          fill=COLORS["text_dim"], font=FONTS["small"])
            return

        mx = max(data)
        if mx == 0:
            mx = 1  # safety: shouldn't reach here due to check above

        # Horizontal grid lines with Y-axis labels (4 lines: 0, 1/3, 2/3, max)
        grid_steps = [0, 1/3, 2/3, 1.0]
        for frac in grid_steps:
            y = pt + ch * (1 - frac)
            c.create_line(px, y, w - 15, y,
                          fill=COLORS["chart_grid"], width=1, dash=(2, 6))
            label_val = mx * frac
            label_text = f"{label_val:,.0f}" if label_val == int(label_val) else f"{label_val:,.1f}"
            c.create_text(px - 6, y, text=label_text, anchor="e",
                          fill=COLORS["text_muted"], font=FONTS["mono_tiny"])

        # X-axis label
        c.create_text(w // 2, h - 4, text="Last 60 polls", anchor="s",
                      fill=COLORS["text_muted"], font=FONTS["mono_tiny"])

        # Points
        n = len(data)
        pts = []
        for i, val in enumerate(data):
            x = px + cw * i / max(n - 1, 1)
            y = pt + ch * (1 - val / mx)
            pts.append((x, y))

        if len(pts) < 2:
            return

        # Fill polygon
        fill_pts = list(pts) + [(pts[-1][0], pt + ch), (pts[0][0], pt + ch)]
        flat = [coord for p in fill_pts for coord in p]
        c.create_polygon(flat, fill=COLORS["chart_fill"], outline="",
                         smooth=True)

        # Lines
        line = [coord for p in pts for coord in p]
        c.create_line(line, fill=COLORS["chart_glow"], width=5, smooth=True,
                      capstyle="round")
        c.create_line(line, fill=COLORS["accent"], width=2, smooth=True,
                      capstyle="round")

        # Latest point glow
        lx, ly = pts[-1]
        for r, clr in [(8, COLORS["chart_glow"]), (4, COLORS["accent"]),
                        (2, "#FFFFFF")]:
            c.create_oval(lx - r, ly - r, lx + r, ly + r,
                          fill=clr, outline="")


class ActivityPanel(ctk.CTkFrame):
    """Recent activity list with scrolling entries."""

    def __init__(self, master, title: str = "Recent Activity", **kw):
        super().__init__(master, fg_color=COLORS["bg_card"],
                         corner_radius=12, border_width=1,
                         border_color=COLORS["border"], **kw)

        ctk.CTkLabel(self, text=title, font=FONTS["subheading"],
                     text_color=COLORS["text"]).pack(
            anchor="w", padx=16, pady=(12, 6))

        self._container = ctk.CTkScrollableFrame(
            self, fg_color="transparent", height=120)
        self._container.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self._rows: list = []

    def set_items(self, items: list):
        """Update activity items. Each: {kind, text, time}."""
        # Clear old rows
        for w in self._rows:
            w.destroy()
        self._rows.clear()

        icons = {
            "block": ("\u26cf", COLORS["accent"]),
            "stake": ("\u26a1", COLORS["purple"]),
            "unstake": ("\U0001f513", COLORS["warning"]),
            "wallet": ("\U0001f4bc", COLORS["gold"]),
            "peer": ("\U0001f310", COLORS["accent_blue"]),
            "tx": ("\u2705", COLORS["success"]),
            "warning": ("\u26a0", COLORS["warning"]),
        }

        for item in items[:15]:
            kind = item.get("kind", "")
            icon, color = icons.get(kind, ("\u2022", COLORS["text_dim"]))

            row = ctk.CTkFrame(self._container, fg_color="transparent",
                               height=28)
            row.pack(fill="x", pady=1)

            ctk.CTkLabel(row, text=icon, width=24,
                         text_color=color,
                         font=(_EMOJI, 13)).pack(side="left", padx=(8, 4))
            ctk.CTkLabel(row, text=item.get("text", ""),
                         font=FONTS["small"],
                         text_color=COLORS["text"]).pack(side="left")
            ctk.CTkLabel(row, text=item.get("time", ""),
                         font=FONTS["tiny"],
                         text_color=COLORS["text_muted"]).pack(
                side="right", padx=(0, 8))
            self._rows.append(row)


class InfoCard(ctk.CTkFrame):
    """Key-value info card with title and multiple rows."""

    def __init__(self, master, title: str, icon: str = "",
                 accent: str = COLORS["accent"], **kw):
        super().__init__(master, fg_color=COLORS["bg_card"],
                         corner_radius=12, border_width=1,
                         border_color=COLORS["border"], **kw)

        bar = ctk.CTkFrame(self, fg_color=accent, height=3, corner_radius=0)
        bar.pack(fill="x", padx=1, pady=(1, 0))

        hdr = ctk.CTkFrame(self, fg_color="transparent")
        hdr.pack(fill="x", padx=16, pady=(12, 8))
        if icon:
            ctk.CTkLabel(hdr, text=icon, font=(_EMOJI, 14),
                         text_color=accent).pack(side="left")
        ctk.CTkLabel(hdr, text=title, font=FONTS["subheading"],
                     text_color=COLORS["text"]).pack(side="left", padx=(6, 0))

        self._body = ctk.CTkFrame(self, fg_color="transparent")
        self._body.pack(fill="both", expand=True, padx=16, pady=(0, 12))

        self._labels: dict = {}

    def add_row(self, key: str, label: str):
        """Add a key-value row."""
        row = ctk.CTkFrame(self._body, fg_color="transparent")
        row.pack(fill="x", pady=2)
        ctk.CTkLabel(row, text=label, font=FONTS["small"],
                     text_color=COLORS["text_dim"], width=140,
                     anchor="w").pack(side="left")
        val = ctk.CTkLabel(row, text="--", font=FONTS["mono"],
                           text_color=COLORS["text"], anchor="w")
        val.pack(side="left", fill="x", expand=True)
        self._labels[key] = val

    def set(self, key: str, value: str):
        lbl = self._labels.get(key)
        if lbl:
            val_str = str(value)
            lbl.configure(text=val_str)
            # Conditional formatting: warn/success colors based on value
            val_upper = val_str.upper()
            if any(kw in val_upper for kw in ("OFFLINE", "DISABLED", "ERROR", "FAIL")):
                lbl.configure(text_color=COLORS["danger"])
            elif val_str.strip() == "0" or val_str.strip() == "--":
                lbl.configure(text_color=COLORS["text_muted"])
            elif any(kw in val_upper for kw in ("ACTIVE", "ENABLED", "ONLINE", "NORMAL", "\u2705")):
                lbl.configure(text_color=COLORS["success"])
            else:
                lbl.configure(text_color=COLORS["text"])
