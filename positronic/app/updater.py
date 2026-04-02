"""
Positronic - Auto-Update Checker

Checks positronic-ai.network/api/version.json for newer versions.
Notifies user in the GUI — never auto-downloads or auto-installs.
"""

import asyncio
import logging
from typing import Dict, Any

from positronic import __version__

logger = logging.getLogger("positronic.app.updater")

UPDATE_CHECK_URL = "https://positronic-ai.network/api/version.json"
CHECK_INTERVAL_HOURS = 24


async def check_for_updates(timeout: float = 10.0) -> Dict[str, Any]:
    """
    Check for a newer version of Positronic.

    Returns dict with keys:
    - available: bool — True if a newer version exists
    - latest: str — latest version string
    - current: str — current version string
    - download_url: str — URL to download new version
    - changelog: str — changelog summary
    """
    try:
        import aiohttp
        from packaging.version import Version

        async with aiohttp.ClientSession() as session:
            async with session.get(
                UPDATE_CHECK_URL,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    latest = data.get("version", __version__)
                    if Version(latest) > Version(__version__):
                        return {
                            "available": True,
                            "latest": latest,
                            "current": __version__,
                            "download_url": data.get(
                                "download_url",
                                "https://positronic-ai.network/download",
                            ),
                            "changelog": data.get("changelog", ""),
                        }
    except ImportError:
        logger.debug("aiohttp or packaging not available for update check")
    except asyncio.TimeoutError:
        logger.debug("Update check timed out after %ss", timeout)
    except Exception as e:
        logger.debug("Update check failed: %s", e)

    return {
        "available": False,
        "latest": __version__,
        "current": __version__,
        "download_url": "",
        "changelog": "",
    }
