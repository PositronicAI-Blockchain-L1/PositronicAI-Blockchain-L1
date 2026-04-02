"""
Animation utilities for CustomTkinter widgets.

Uses tkinter's .after() for frame-by-frame updates.
All animations are non-blocking and cancellable.
"""

import colorsys
import logging

logger = logging.getLogger("positronic.app.ui.animations")


def _hex_to_rgb(hex_color: str) -> tuple:
    """Convert '#RRGGBB' to (r, g, b) floats 0-1."""
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def _rgb_to_hex(r: float, g: float, b: float) -> str:
    """Convert (r, g, b) floats 0-1 to '#RRGGBB'."""
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def _lerp_color(c1: str, c2: str, t: float) -> str:
    """Linear interpolate between two hex colors. t=0 -> c1, t=1 -> c2."""
    r1, g1, b1 = _hex_to_rgb(c1)
    r2, g2, b2 = _hex_to_rgb(c2)
    return _rgb_to_hex(
        r1 + (r2 - r1) * t,
        g1 + (g2 - g1) * t,
        b1 + (b2 - b1) * t,
    )


def pulse_dot(widget, color_on: str, color_off: str, period_ms: int = 2000):
    """Breathing pulse effect for a CTkLabel status dot.

    Smoothly transitions between color_on and color_off using
    a sine-like easing curve. Stores timer ID on widget for cancellation.

    Call stop_pulse(widget) to cancel.
    """
    import math
    steps = 30  # frames per full cycle
    step_ms = period_ms // steps

    def _cancel_existing():
        if hasattr(widget, '_pulse_timer') and widget._pulse_timer is not None:
            try:
                widget.after_cancel(widget._pulse_timer)
            except Exception:
                pass
            widget._pulse_timer = None

    _cancel_existing()

    frame = [0]  # mutable counter

    def _tick():
        try:
            if not widget.winfo_exists():
                return
            # Sine wave easing: 0->1->0 over one cycle
            t = (math.sin(2 * math.pi * frame[0] / steps) + 1) / 2
            color = _lerp_color(color_off, color_on, t)
            widget.configure(text_color=color)
            frame[0] = (frame[0] + 1) % steps
            widget._pulse_timer = widget.after(step_ms, _tick)
        except Exception:
            pass  # Widget destroyed

    _tick()


def stop_pulse(widget):
    """Stop a pulse animation on a widget."""
    if hasattr(widget, '_pulse_timer') and widget._pulse_timer is not None:
        try:
            widget.after_cancel(widget._pulse_timer)
        except Exception:
            pass
        widget._pulse_timer = None


def count_transition(label_widget, old_text: str, new_text: str,
                     duration_ms: int = 400):
    """Smooth numeric count transition on a CTkLabel.

    Parses numeric values from old/new text (handles #N, N,NNN, N.N formats).
    Falls back to instant update if parsing fails.
    """
    old_num = _parse_display_number(old_text)
    new_num = _parse_display_number(new_text)

    if old_num is None or new_num is None or old_num == new_num:
        label_widget.configure(text=new_text)
        return

    steps = 12
    step_ms = max(duration_ms // steps, 16)  # min 16ms (~60fps)
    prefix = ""
    suffix = ""

    # Detect prefix (#) and suffix (ASF, %, etc.)
    stripped = new_text.strip()
    if stripped.startswith("#"):
        prefix = "#"
    # Find suffix after number
    import re
    m = re.search(r'[\d,.]+\s*(.*)', stripped.lstrip("#"))
    if m:
        suffix = m.group(1)

    is_float = "." in new_text
    use_comma = "," in new_text

    def _format_val(v):
        if is_float:
            decimals = len(new_text.split(".")[-1].rstrip("% ASFMmKk"))
            decimals = min(decimals, 4)
            s = f"{v:,.{decimals}f}" if use_comma else f"{v:.{decimals}f}"
        else:
            s = f"{int(v):,}" if use_comma else str(int(v))
        return f"{prefix}{s}" + (f" {suffix}" if suffix else "")

    # Cancel any existing count animation
    if hasattr(label_widget, '_count_timer') and label_widget._count_timer:
        try:
            label_widget.after_cancel(label_widget._count_timer)
        except Exception:
            pass

    step_val = [0]

    def _tick():
        try:
            if not label_widget.winfo_exists():
                return
            step_val[0] += 1
            t = min(step_val[0] / steps, 1.0)
            # Ease-out cubic
            t_ease = 1 - (1 - t) ** 3
            current = old_num + (new_num - old_num) * t_ease
            label_widget.configure(text=_format_val(current))
            if step_val[0] < steps:
                label_widget._count_timer = label_widget.after(step_ms, _tick)
            else:
                label_widget.configure(text=new_text)  # Ensure exact final value
                label_widget._count_timer = None
        except Exception:
            label_widget.configure(text=new_text)

    _tick()


def _parse_display_number(text: str):
    """Parse a display number from stat card text. Returns float or None."""
    if not text or text == "--":
        return None
    import re
    # Remove prefix characters like #
    cleaned = text.strip().lstrip("#").strip()
    # Remove suffix like ASF, %, etc.
    cleaned = re.sub(r'\s*[A-Za-z%/]+.*$', '', cleaned)
    # Remove commas
    cleaned = cleaned.replace(",", "")
    try:
        return float(cleaned)
    except (ValueError, OverflowError):
        return None


class LoadingSpinner:
    """Simple text-based loading spinner for CTkLabel widgets.

    Usage:
        spinner = LoadingSpinner(my_label)
        spinner.start()
        # ... later ...
        spinner.stop("Done!")
    """
    FRAMES = ["\u280b", "\u2819", "\u2839", "\u2838", "\u283c", "\u2834", "\u2826", "\u2827", "\u2807", "\u280f"]

    def __init__(self, label_widget, interval_ms: int = 100):
        self._label = label_widget
        self._interval = interval_ms
        self._idx = 0
        self._timer = None
        self._running = False

    def start(self, prefix: str = ""):
        """Start the spinner animation."""
        self._prefix = prefix
        self._running = True
        self._tick()

    def _tick(self):
        if not self._running:
            return
        try:
            if not self._label.winfo_exists():
                self._running = False
                return
            char = self.FRAMES[self._idx % len(self.FRAMES)]
            text = f"{self._prefix} {char}" if self._prefix else char
            self._label.configure(text=text)
            self._idx += 1
            self._timer = self._label.after(self._interval, self._tick)
        except Exception:
            self._running = False

    def stop(self, final_text: str = ""):
        """Stop the spinner and set final text."""
        self._running = False
        if self._timer is not None:
            try:
                self._label.after_cancel(self._timer)
            except Exception:
                pass
            self._timer = None
        if final_text:
            try:
                self._label.configure(text=final_text)
            except Exception:
                pass
