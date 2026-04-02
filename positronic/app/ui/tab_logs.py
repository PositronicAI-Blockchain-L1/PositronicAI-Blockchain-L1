"""Logs tab — professional log viewer with level coloring, filtering, search, pause/resume, export.

Inspired by VS Code terminal / Grafana Explore log panel.
"""

import logging
import tkinter as tk
from tkinter import scrolledtext, filedialog

import customtkinter as ctk

from positronic.app.theme import COLORS, FONTS, LOG_TRIM_THRESHOLD, LOG_TRIM_KEEP


# ── Level color mapping ────────────────────────────────────────────
_LEVEL_COLORS = {
    "ERROR": COLORS["danger"],
    "WARNING": COLORS["warning"],
    "INFO": "#C8D0DC",              # bright white-grey for readability
    "DEBUG": COLORS["text_muted"],  # subdued grey
}

_LEVEL_BG = {
    "ERROR": "#2a0a0a",
    "WARNING": "#2a1f0a",
}


def build_logs(tab, win):
    """Build logs tab widgets. Stores references on *win*."""

    # ── Top toolbar ─────────────────────────────────────────────
    toolbar = ctk.CTkFrame(tab, fg_color=COLORS["bg_card"],
                           corner_radius=10, border_width=1,
                           border_color=COLORS["card_border"], height=44)
    toolbar.pack(fill="x", padx=12, pady=(10, 0))
    toolbar.pack_propagate(False)

    # Title
    ctk.CTkLabel(toolbar, text="\U0001f4cb  Node Logs",
                 font=FONTS["subheading"],
                 text_color=COLORS["text"]).pack(side="left", padx=(14, 16))

    # Separator
    _vsep(toolbar)

    # Level filter dropdown
    ctk.CTkLabel(toolbar, text="Level:", font=FONTS["small"],
                 text_color=COLORS["text_dim"]).pack(side="left", padx=(10, 4))

    win._log_level_filter = ctk.CTkOptionMenu(
        toolbar, values=["All", "Info+", "Warning+", "Error"],
        font=FONTS["small"], width=100, height=28,
        fg_color=COLORS["bg_dark"],
        button_color=COLORS["border"],
        button_hover_color=COLORS["bg_card_hover"],
        dropdown_fg_color=COLORS["bg_card"],
        dropdown_hover_color=COLORS["bg_card_hover"],
        text_color=COLORS["text"],
        command=lambda _v: _apply_dropdown_filter(win))
    win._log_level_filter.set("All")
    win._log_level_filter.pack(side="left", padx=(0, 6))

    _vsep(toolbar)

    # Auto-scroll toggle
    win._log_autoscroll = tk.BooleanVar(value=True)
    autoscroll_cb = ctk.CTkCheckBox(
        toolbar, text="Auto-scroll", font=FONTS["small"],
        text_color=COLORS["text_dim"],
        fg_color=COLORS["accent"], hover_color=COLORS["accent_blue"],
        border_color=COLORS["border"],
        corner_radius=4, height=24, width=20,
        checkbox_width=16, checkbox_height=16,
        variable=win._log_autoscroll)
    autoscroll_cb.pack(side="left", padx=(10, 6))

    _vsep(toolbar)

    # Right side buttons
    # Clear
    ctk.CTkButton(toolbar, text="\U0001f5d1 Clear", font=FONTS["small"],
                  fg_color="transparent",
                  hover_color=COLORS["bg_card_hover"],
                  text_color=COLORS["text_dim"],
                  corner_radius=6, width=70, height=28,
                  command=win._clear_logs).pack(side="right", padx=(2, 10))

    # Export
    win._export_btn = ctk.CTkButton(
        toolbar, text="\U0001f4e5 Export", font=FONTS["small"],
        fg_color="transparent",
        hover_color=COLORS["bg_card_hover"],
        text_color=COLORS["text_dim"],
        corner_radius=6, width=76, height=28,
        command=lambda: _export_logs(win))
    win._export_btn.pack(side="right", padx=2)

    # Pause / Resume toggle
    win._log_paused = False
    win._pause_btn = ctk.CTkButton(
        toolbar, text="\u23f8 Pause", font=FONTS["small"],
        fg_color="transparent",
        hover_color=COLORS["bg_card_hover"],
        text_color=COLORS["text_dim"],
        corner_radius=6, width=76, height=28,
        command=lambda: _toggle_pause(win))
    win._pause_btn.pack(side="right", padx=2)

    # ── Search bar ──────────────────────────────────────────────
    search_bar = ctk.CTkFrame(tab, fg_color=COLORS["bg_card"],
                              corner_radius=8, border_width=1,
                              border_color=COLORS["card_border"], height=36)
    search_bar.pack(fill="x", padx=12, pady=(6, 0))
    search_bar.pack_propagate(False)

    ctk.CTkLabel(search_bar, text="\U0001f50d", font=FONTS["small"],
                 text_color=COLORS["text_muted"]).pack(side="left", padx=(12, 4))

    win._log_search_var = tk.StringVar()
    win._log_search_entry = ctk.CTkEntry(
        search_bar, textvariable=win._log_search_var,
        placeholder_text="Filter logs...  (Ctrl+F)",
        font=FONTS["mono_small"],
        fg_color="transparent",
        text_color=COLORS["text"],
        border_width=0,
        height=28, width=300)
    win._log_search_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))

    ctk.CTkButton(
        search_bar, text="Find", font=FONTS["small"],
        fg_color=COLORS["accent"], hover_color=COLORS["accent_blue"],
        text_color=COLORS["bg_darkest"],
        corner_radius=5, width=50, height=26,
        command=lambda: _search_logs(win)).pack(side="left", padx=(0, 4))

    ctk.CTkButton(
        search_bar, text="\u2716", font=FONTS["small"],
        fg_color="transparent", hover_color=COLORS["bg_card_hover"],
        text_color=COLORS["text_dim"],
        corner_radius=5, width=26, height=26,
        command=lambda: _clear_search(win)).pack(side="left", padx=(0, 4))

    win._search_match_label = ctk.CTkLabel(
        search_bar, text="", font=FONTS["tiny"],
        text_color=COLORS["text_muted"])
    win._search_match_label.pack(side="left", padx=(4, 10))

    # ── Log text area (using tk.Text for performance) ──────────
    # Wrap in a frame with rounded appearance
    log_container = ctk.CTkFrame(tab, fg_color=COLORS["bg_darkest"],
                                 corner_radius=10, border_width=1,
                                 border_color=COLORS["card_border"])
    log_container.pack(fill="both", expand=True, padx=12, pady=(6, 10))

    win._log_text = scrolledtext.ScrolledText(
        log_container, font=FONTS["mono_small"],
        bg=COLORS["bg_darkest"], fg="#C8D0DC",
        insertbackground=COLORS["text"],
        selectbackground=COLORS["accent"],
        selectforeground=COLORS["bg_darkest"],
        state=tk.DISABLED, wrap=tk.WORD,
        borderwidth=0, highlightthickness=0,
        padx=14, pady=10,
        relief="flat",
    )
    win._log_text.pack(fill="both", expand=True, padx=2, pady=2)

    # Style the scrollbar via the underlying tk scrollbar
    try:
        vsb = win._log_text.vbar
        vsb.configure(bg=COLORS["bg_dark"], troughcolor=COLORS["bg_darkest"],
                      activebackground=COLORS["border"], width=10,
                      relief="flat", bd=0)
    except Exception:
        pass

    # Level tags with foreground colors
    win._log_text.tag_config("ERROR", foreground=COLORS["danger"])
    win._log_text.tag_config("WARNING", foreground=COLORS["warning"])
    win._log_text.tag_config("INFO", foreground="#C8D0DC")
    win._log_text.tag_config("DEBUG", foreground=COLORS["text_muted"])

    # Timestamp styling — slightly dimmer than the message
    win._log_text.tag_config("TIMESTAMP", foreground=COLORS["text_muted"])

    # Search highlight tag
    win._log_text.tag_config("SEARCH_HIT", background=COLORS["warning"],
                             foreground="#FFFFFF")
    # Hidden tags for level filtering
    win._log_text.tag_config("HIDDEN", elide=True)

    # ── Legacy filter vars (keep for backward compat) ──────────
    win._log_filter_vars = {}
    for level in ("ERROR", "WARNING", "INFO", "DEBUG"):
        win._log_filter_vars[level] = tk.BooleanVar(value=True)

    # ── Status bar ──────────────────────────────────────────────
    status_bar = ctk.CTkFrame(tab, fg_color="transparent", height=20)
    status_bar.pack(fill="x", padx=16, pady=(0, 4))

    win._log_line_count_label = ctk.CTkLabel(
        status_bar, text="0 lines", font=FONTS["tiny"],
        text_color=COLORS["text_muted"])
    win._log_line_count_label.pack(side="left")

    win._log_status_label = ctk.CTkLabel(
        status_bar, text="", font=FONTS["tiny"],
        text_color=COLORS["text_muted"])
    win._log_status_label.pack(side="right")

    # Ctrl+F binding — focus search entry
    try:
        tab.winfo_toplevel().bind("<Control-f>", lambda e: _focus_search(win))
    except Exception:
        pass


def _vsep(parent):
    """Vertical separator bar for toolbar."""
    sep = ctk.CTkFrame(parent, fg_color=COLORS["separator"], width=1, height=20)
    sep.pack(side="left", padx=4, pady=6)


def _apply_dropdown_filter(win):
    """Apply level filter from dropdown selection."""
    selection = win._log_level_filter.get()
    level_map = {
        "All": {"ERROR": True, "WARNING": True, "INFO": True, "DEBUG": True},
        "Info+": {"ERROR": True, "WARNING": True, "INFO": True, "DEBUG": False},
        "Warning+": {"ERROR": True, "WARNING": True, "INFO": False, "DEBUG": False},
        "Error": {"ERROR": True, "WARNING": False, "INFO": False, "DEBUG": False},
    }
    settings = level_map.get(selection, level_map["All"])
    for level, show in settings.items():
        win._log_filter_vars[level].set(show)
    _apply_level_filter(win)


def _toggle_pause(win):
    """Toggle log pause/resume state."""
    win._log_paused = not win._log_paused
    if win._log_paused:
        win._pause_btn.configure(text="\u25b6 Resume",
                                 fg_color=COLORS["accent"],
                                 text_color=COLORS["bg_darkest"])
        try:
            win._log_status_label.configure(text="\u23f8 Paused",
                                            text_color=COLORS["warning"])
        except Exception:
            pass
    else:
        win._pause_btn.configure(text="\u23f8 Pause",
                                 fg_color="transparent",
                                 text_color=COLORS["text_dim"])
        try:
            win._log_status_label.configure(text="",
                                            text_color=COLORS["text_muted"])
            win._log_text.see(tk.END)
        except Exception:
            pass


def _apply_level_filter(win):
    """Show/hide log lines based on level filter checkboxes using elide."""
    try:
        for level, var in win._log_filter_vars.items():
            show = var.get()
            ranges = win._log_text.tag_ranges(level)
            for i in range(0, len(ranges), 2):
                start = ranges[i]
                end = ranges[i + 1]
                if show:
                    win._log_text.tag_remove("HIDDEN", start, end)
                else:
                    win._log_text.tag_add("HIDDEN", start, end)
    except (tk.TclError, Exception):
        pass


def _search_logs(win):
    """Search log text and highlight matches."""
    query = win._log_search_var.get().strip()
    # Clear previous highlights
    win._log_text.tag_remove("SEARCH_HIT", "1.0", tk.END)
    if not query:
        win._search_match_label.configure(text="")
        return

    count = 0
    start = "1.0"
    while True:
        pos = win._log_text.search(query, start, stopindex=tk.END, nocase=True)
        if not pos:
            break
        end = f"{pos}+{len(query)}c"
        win._log_text.tag_add("SEARCH_HIT", pos, end)
        start = end
        count += 1
        if count > 5000:  # safety cap
            break

    if count > 0:
        win._search_match_label.configure(
            text=f"{count} match{'es' if count != 1 else ''}",
            text_color=COLORS["accent"])
        # Scroll to first match
        first = win._log_text.tag_ranges("SEARCH_HIT")
        if first:
            win._log_text.see(first[0])
    else:
        win._search_match_label.configure(
            text="No matches", text_color=COLORS["warning"])


def _clear_search(win):
    """Clear search field and highlights."""
    win._log_search_var.set("")
    win._log_text.tag_remove("SEARCH_HIT", "1.0", tk.END)
    win._search_match_label.configure(text="")


def _focus_search(win):
    """Focus the search entry (Ctrl+F handler)."""
    try:
        win._log_search_entry.focus_set()
    except Exception:
        pass


def _export_logs(win):
    """Export current log content to a text file."""
    try:
        content = win._log_text.get("1.0", tk.END).strip()
        if not content:
            return
        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("Log Files", "*.log"),
                       ("All Files", "*.*")],
            title="Export Logs",
            initialfile="positronic_logs.txt")
        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
    except Exception:
        pass


def setup_log_handler(win):
    """Attach a _TkLogHandler to the positronic root logger."""
    handler = _TkLogHandler(win._log_text, win.root, win)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    handler.setLevel(logging.DEBUG)
    logging.getLogger("positronic").addHandler(handler)


class _TkLogHandler(logging.Handler):
    """Thread-safe logging handler for ScrolledText widget with pause & filter support."""

    def __init__(self, text_widget, root, win):
        super().__init__()
        self._widget = text_widget
        self._root = root
        self._win = win
        self._emitting = False  # re-entrancy guard

    def emit(self, record):
        if self._emitting:
            return  # prevent infinite recursion if RPC client logs during emit
        self._emitting = True
        try:
            # Check pause state
            if getattr(self._win, '_log_paused', False):
                return
            msg = self.format(record) + "\n"
            tag = record.levelname
            self._root.after_idle(self._append, msg, tag)
        except Exception:  # noqa: BLE001
            pass  # INTENTIONAL: Cannot log inside a log handler -- infinite recursion
        finally:
            self._emitting = False

    def _append(self, msg, tag):
        try:
            # Double-check pause (could have changed between emit and after_idle)
            if getattr(self._win, '_log_paused', False):
                return

            self._widget.configure(state=tk.NORMAL)
            self._widget.insert(tk.END, msg, tag)

            # Apply level filter: if this level is unchecked, hide the line
            filters = getattr(self._win, '_log_filter_vars', {})
            var = filters.get(tag)
            if var and not var.get():
                # Hide the just-inserted line
                line_start = self._widget.index(f"end-{len(msg) + 1}c linestart")
                self._widget.tag_add("HIDDEN", line_start, tk.END)

            # Auto-scroll only if enabled
            autoscroll = getattr(self._win, '_log_autoscroll', None)
            if autoscroll is None or autoscroll.get():
                self._widget.see(tk.END)

            # Trim only every 100 lines to reduce UI redraws
            count = int(self._widget.index("end-1c").split(".")[0])
            if count > LOG_TRIM_THRESHOLD:
                self._widget.delete("1.0", f"{count - LOG_TRIM_KEEP}.0")

            self._widget.configure(state=tk.DISABLED)

            # Update line count in status bar
            try:
                line_count_lbl = getattr(self._win, '_log_line_count_label', None)
                if line_count_lbl:
                    line_count_lbl.configure(text=f"{count} lines")
            except Exception:
                pass

        except (tk.TclError, Exception):
            pass  # Widget destroyed or other error — safe to ignore
