"""
System tray integration using pystray.

Runs in its own daemon thread so it doesn't block tkinter.
Gracefully falls back if pystray/Pillow are not installed.
"""

import logging
import threading
from typing import Callable, Optional

logger = logging.getLogger("positronic.app.tray")


class SystemTray:
    """Cross-platform system tray icon with context menu."""

    def __init__(
        self,
        on_show: Callable,
        on_quit: Callable,
        on_toggle_node: Callable,
    ):
        self.on_show = on_show
        self.on_quit = on_quit
        self.on_toggle_node = on_toggle_node
        self._icon = None
        self._thread: Optional[threading.Thread] = None
        self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def start(self):
        """Start system tray in a daemon thread."""
        try:
            import pystray  # noqa: F401
            from PIL import Image  # noqa: F401
            self._available = True
        except ImportError:
            logger.info(
                "pystray or Pillow not installed — system tray disabled. "
                "Install with: pip install pystray Pillow"
            )
            return

        self._thread = threading.Thread(
            target=self._run_tray,
            daemon=True,
            name="positronic-tray",
        )
        self._thread.start()

    def stop(self):
        """Stop the tray icon."""
        if self._icon:
            try:
                self._icon.stop()
            except Exception as e:
                logger.debug("Tray icon stop error: %s", e)

    def _run_tray(self):
        import pystray
        from PIL import Image, ImageDraw

        # Programmatic icon: purple circle with gold "P" feel
        size = 64
        image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        # Purple filled circle
        draw.ellipse([4, 4, size - 4, size - 4], fill=(123, 47, 247, 255))
        # Inner highlight
        draw.ellipse([16, 16, size - 16, size - 16],
                      fill=(160, 90, 255, 200))

        menu = pystray.Menu(
            pystray.MenuItem("Show Window", self._do_show, default=True),
            pystray.MenuItem("Start / Stop Node", self._do_toggle),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit Positronic", self._do_quit),
        )

        self._icon = pystray.Icon(
            "Positronic", image, "Positronic Node", menu,
        )

        logger.info("System tray started")
        self._icon.run()

    def _do_show(self, icon=None, item=None):
        self.on_show()

    def _do_toggle(self, icon=None, item=None):
        self.on_toggle_node()

    def _do_quit(self, icon=None, item=None):
        self.on_quit()
