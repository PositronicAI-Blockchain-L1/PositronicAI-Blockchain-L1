"""
Positronic - NAT Traversal
Three-tier approach: UPnP -> STUN -> manual fallback.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("positronic.network.nat")


@dataclass
class ExternalAddress:
    """Discovered external address."""
    ip: str
    port: int
    method: str  # "upnp", "stun", "manual"


class NATTraversal:
    """
    Automatic NAT traversal for P2P nodes.

    Tries in order:
    1. UPnP port mapping (miniupnpc)
    2. STUN external IP detection (pystun3)
    3. Returns None if both fail (node runs in outbound-only mode)
    """

    def __init__(self, port: int = 29335):
        self.port = port
        self._upnp_mapped = False
        self._external: Optional[ExternalAddress] = None

    async def setup(self) -> Optional[ExternalAddress]:
        """Discover external address. Returns None if unreachable."""
        result = await self._try_upnp()
        if result:
            self._external = result
            return result
        result = await self._try_stun()
        if result:
            self._external = result
            return result
        logger.info("NAT: no external address found, outbound-only mode")
        return None

    async def _try_upnp(self) -> Optional[ExternalAddress]:
        """Try UPnP port mapping."""
        try:
            import miniupnpc
        except ImportError:
            logger.debug("miniupnpc not installed, skipping UPnP")
            return None
        try:
            loop = asyncio.get_event_loop()
            u = miniupnpc.UPnP()
            u.discoverdelay = 2000
            devices = await loop.run_in_executor(None, u.discover)
            if devices == 0:
                return None
            await loop.run_in_executor(None, u.selectigd)
            ext_ip = await loop.run_in_executor(None, u.externalipaddress)
            await loop.run_in_executor(None,
                u.addportmapping, self.port, "TCP", u.lanaddr, self.port,
                "Positronic Node", "")
            self._upnp_mapped = True
            logger.info(f"UPnP: mapped {ext_ip}:{self.port}")
            return ExternalAddress(ip=ext_ip, port=self.port, method="upnp")
        except Exception as e:
            logger.debug(f"UPnP failed: {e}")
            return None

    async def _try_stun(self) -> Optional[ExternalAddress]:
        """Try STUN server to detect external IP."""
        try:
            import stun
        except ImportError:
            logger.debug("pystun3 not installed, skipping STUN")
            return None
        try:
            loop = asyncio.get_event_loop()
            nat_type, ext_ip, ext_port = await loop.run_in_executor(
                None, stun.get_ip_info)
            if ext_ip:
                logger.info(f"STUN: {ext_ip}:{ext_port} (NAT: {nat_type})")
                return ExternalAddress(
                    ip=ext_ip, port=ext_port or self.port, method="stun")
            return None
        except Exception as e:
            logger.debug(f"STUN failed: {e}")
            return None

    async def cleanup(self):
        """Remove UPnP mappings on shutdown."""
        if not self._upnp_mapped:
            return
        try:
            import miniupnpc
            loop = asyncio.get_event_loop()
            u = miniupnpc.UPnP()
            await loop.run_in_executor(None, u.discover)
            await loop.run_in_executor(None, u.selectigd)
            await loop.run_in_executor(None,
                u.deleteportmapping, self.port, "TCP")
            self._upnp_mapped = False
            logger.info("UPnP mapping removed")
        except Exception as e:
            logger.debug(f"UPnP cleanup: {e}")

    @property
    def external_address(self) -> Optional[ExternalAddress]:
        return self._external
