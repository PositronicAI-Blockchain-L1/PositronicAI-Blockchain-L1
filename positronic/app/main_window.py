"""Backward-compatible re-export.

The actual MainWindow class now lives in positronic.app.ui.main_window.
This shim ensures existing imports (e.g., from positronic.app.main_window
import MainWindow) continue to work without changes.
"""
from positronic.app.ui.main_window import MainWindow  # noqa: F401
