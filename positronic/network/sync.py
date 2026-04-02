"""
Positronic - Block Synchronization
Syncs blockchain state with peers using a batch download protocol.

Sync phases:
1. Status Exchange - Compare chain heights with peers
2. Header Download - Download block headers to verify chain
3. Block Download  - Download full blocks in batches
4. Verification    - Validate and add blocks to the chain
"""

import asyncio
import time
import logging
from typing import Optional, List, Callable, Dict
from dataclasses import dataclass, field
from enum import IntEnum

from positronic.network.peer import Peer, PeerManager, PeerState
from positronic.network.messages import (
    make_get_blocks, make_blocks, make_status,
    make_sync_request, make_sync_response,
    NetworkMessage,
)

logger = logging.getLogger("positronic.network.sync")


class SyncPhase(IntEnum):
    """Current phase of sync."""
    IDLE = 0
    STATUS_EXCHANGE = 1
    DOWNLOADING = 2
    PROCESSING = 3
    COMPLETE = 4


@dataclass
class SyncState:
    """Current sync state."""
    syncing: bool = False
    phase: SyncPhase = SyncPhase.IDLE
    target_height: int = 0
    current_height: int = 0
    sync_peer: Optional[str] = None
    blocks_downloaded: int = 0
    blocks_processed: int = 0
    start_time: float = 0.0
    last_activity: float = 0.0
    errors: int = 0

    @property
    def progress(self) -> float:
        if self.target_height <= self.current_height:
            return 1.0
        total = self.target_height - (self.current_height - self.blocks_processed)
        if total <= 0:
            return 1.0
        return min(1.0, self.blocks_processed / total)

    @property
    def is_synced(self) -> bool:
        return not self.syncing or self.current_height >= self.target_height

    @property
    def elapsed(self) -> float:
        """Time elapsed since sync started."""
        if self.start_time == 0:
            return 0
        return time.time() - self.start_time

    @property
    def blocks_per_second(self) -> float:
        """Download rate."""
        elapsed = self.elapsed
        if elapsed <= 0:
            return 0
        return self.blocks_downloaded / elapsed

    @property
    def eta_seconds(self) -> float:
        """Estimated time remaining."""
        rate = self.blocks_per_second
        if rate <= 0:
            return 0
        remaining = self.target_height - self.current_height
        return remaining / rate


class BlockSync:
    """
    Manages block synchronization with peers.

    Flow:
    1. Node starts, checks if peers have higher chain
    2. Picks best peer to sync from
    3. Downloads blocks in batches (BATCH_SIZE at a time)
    4. Validates and adds blocks to chain
    5. Repeats until caught up
    """

    BATCH_SIZE = 50      # Blocks per request
    MAX_RETRIES = 3      # Max retries per batch
    SYNC_TIMEOUT = 30    # Seconds to wait for a batch response
    STALE_TIMEOUT = 60   # Seconds before considering sync stale

    def __init__(self, peer_manager: PeerManager):
        self.peer_manager = peer_manager
        self.state = SyncState()

        # Pending block batches waiting to be processed
        self._pending_blocks: Dict[int, List[dict]] = {}  # start_height -> blocks

        # Callbacks
        self._add_block_callback: Optional[Callable] = None
        self._get_block_callback: Optional[Callable] = None
        self._send_request_callback: Optional[Callable] = None
        self._on_sync_complete_callback: Optional[Callable] = None
        self._on_sync_stalled_callback: Optional[Callable] = None
        self._get_chain_height_callback: Optional[Callable] = None

        # Sync task
        self._sync_task: Optional[asyncio.Task] = None
        self._running = False

    def _is_sync_active(self) -> bool:
        """Check if sync is TRULY active (flag=True AND task running).

        Fixes deadlock where syncing=True but _sync_task finished/crashed,
        which permanently blocks all sync paths (start_sync, _sync_check_loop,
        eager sync).  If flag is stale, auto-reset it.
        """
        if not self.state.syncing:
            return False
        # Flag says syncing, but is the task actually alive?
        if self._sync_task is not None and not self._sync_task.done():
            return True  # genuinely active
        # Stale flag — task is dead/missing.  Reset state so sync can retry.
        logger.warning(
            "Sync flag was True but sync task is not running — resetting stale sync state "
            "(height=%s, target=%s, peer=%s)",
            self.state.current_height,
            self.state.target_height,
            self.state.sync_peer,
        )
        self.state.syncing = False
        self.state.phase = SyncPhase.IDLE
        return False

    def set_callbacks(
        self,
        add_block: Callable,
        get_block: Callable,
        send_request: Callable,
        on_sync_complete: Optional[Callable] = None,
        on_sync_stalled: Optional[Callable] = None,
        get_chain_height: Optional[Callable] = None,
    ):
        """
        Set required callbacks for sync operation.

        add_block(block_dict) -> bool  - Add a block to the chain
        get_block(height) -> dict|None - Get a block by height
        send_request(peer_id, start, count) -> None - Request blocks from peer
        on_sync_complete() -> None - Called after sync finishes (e.g. to broadcast status)
        on_sync_stalled(peer_id) -> None - Called when sync stalls after MAX_RETRIES (e.g. disconnect peer to force reconnect)
        get_chain_height() -> int - Get current blockchain height (for reconciliation)
        """
        self._add_block_callback = add_block
        self._on_sync_complete_callback = on_sync_complete
        self._on_sync_stalled_callback = on_sync_stalled
        self._get_block_callback = get_block
        self._send_request_callback = send_request
        self._get_chain_height_callback = get_chain_height

    def needs_sync(self, local_height: int) -> bool:
        """Check if we need to sync by comparing with best peer."""
        best_peer = self.peer_manager.get_best_peer()
        if not best_peer:
            return False
        return best_peer.chain_height > local_height

    def get_sync_target(self, local_height: int) -> Optional[dict]:
        """Get sync target info."""
        best_peer = self.peer_manager.get_best_peer()
        if not best_peer or best_peer.chain_height <= local_height:
            return None

        return {
            "peer_id": best_peer.peer_id,
            "target_height": best_peer.chain_height,
            "blocks_needed": best_peer.chain_height - local_height,
        }

    async def start_sync(self, local_height: int):
        """
        Begin syncing from peers.
        Finds the best peer and starts downloading blocks.
        """
        if self._is_sync_active():
            logger.debug("Sync already in progress")
            return

        target = self.get_sync_target(local_height)
        if not target:
            logger.debug("No sync target found")
            return

        peer_id = target["peer_id"]
        target_height = target["target_height"]

        logger.info(
            f"Starting sync from peer {peer_id[:8]}: "
            f"height {local_height} -> {target_height} "
            f"({target['blocks_needed']} blocks)"
        )

        self.state = SyncState(
            syncing=True,
            phase=SyncPhase.DOWNLOADING,
            target_height=target_height,
            current_height=local_height,
            sync_peer=peer_id,
            start_time=time.time(),
            last_activity=time.time(),
        )

        # Mark the peer as syncing
        peer = self.peer_manager.get_peer(peer_id)
        if peer:
            peer.state = PeerState.SYNCING

        # Start the sync loop
        if self._sync_task is None or self._sync_task.done():
            self._running = True
            self._sync_task = asyncio.create_task(self._sync_loop())

    async def _sync_loop(self):
        """
        Main sync loop. Downloads blocks in batches.
        """
        retries = 0

        while self._running and self.state.syncing:
            try:
                # Check if sync is complete
                if self.state.current_height >= self.state.target_height:
                    self.finish_sync()
                    break

                # Check if sync peer is still connected
                peer = self.peer_manager.get_peer(self.state.sync_peer)
                if not peer or peer.state == PeerState.DISCONNECTED:
                    # Try to find a new sync peer
                    new_target = self.get_sync_target(self.state.current_height)
                    if new_target:
                        self.state.sync_peer = new_target["peer_id"]
                        self.state.target_height = new_target["target_height"]
                        logger.info(
                            f"Switched sync peer to {new_target['peer_id'][:8]}"
                        )
                    else:
                        logger.warning("No sync peers available, pausing sync")
                        await asyncio.sleep(10)
                        continue

                # Check for stale sync
                if time.time() - self.state.last_activity > self.STALE_TIMEOUT:
                    retries += 1
                    if retries > self.MAX_RETRIES:
                        logger.error("Sync stalled after max retries")
                        self.state.errors += 1
                        stalled_peer = self.state.sync_peer
                        self.state.sync_peer = None
                        self.state.syncing = False
                        retries = 0
                        # Notify node so it can disconnect this peer and let discovery reconnect
                        if stalled_peer and self._on_sync_stalled_callback:
                            try:
                                cb = self._on_sync_stalled_callback(stalled_peer)
                                if asyncio.iscoroutine(cb):
                                    await cb
                            except Exception as e:
                                logger.debug(f"on_sync_stalled callback error: {e}")
                        break
                    logger.warning(
                        f"Sync stalled, retrying ({retries}/{self.MAX_RETRIES})"
                    )

                # Request next batch
                start = self.state.current_height + 1
                count = min(
                    self.BATCH_SIZE,
                    self.state.target_height - self.state.current_height,
                )

                if self._send_request_callback:
                    await self._send_request_callback(
                        self.state.sync_peer, start, count
                    )
                    # NOTE: Do NOT refresh last_activity here.
                    # last_activity tracks when blocks were last RECEIVED
                    # (updated in on_blocks_received / _process_pending_blocks).
                    # Refreshing on send masks unresponsive peers and
                    # prevents STALE_TIMEOUT from ever triggering.

                # Wait for blocks to arrive
                await asyncio.sleep(2)

                # Process any received blocks
                await self._process_pending_blocks()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Sync loop error: {e}")
                self.state.errors += 1
                await asyncio.sleep(5)

    async def _process_pending_blocks(self):
        """Process blocks that have been received and are ready."""
        if not self._add_block_callback:
            return

        # Reconcile sync state with actual blockchain height.
        # P2P broadcast may have advanced the chain beyond what sync knows.
        if self._get_chain_height_callback:
            actual_height = self._get_chain_height_callback()
            if actual_height > self.state.current_height:
                logger.debug(
                    "Sync state reconciled: %d -> %d (P2P advanced chain)",
                    self.state.current_height, actual_height,
                )
                self.state.current_height = actual_height

        # Process blocks in order
        next_height = self.state.current_height + 1

        # Check all pending batches
        heights_to_process = sorted(self._pending_blocks.keys())

        for start_height in heights_to_process:
            if start_height > next_height:
                break  # Gap in the chain, wait for missing blocks

            blocks = self._pending_blocks.pop(start_height)

            for block_dict in blocks:
                block_height = block_dict.get("header", {}).get("height", 0)

                if block_height != next_height:
                    continue  # Skip out-of-order blocks

                try:
                    success = self._add_block_callback(block_dict)
                    if success:
                        self.state.current_height = block_height
                        self.state.blocks_processed += 1
                        self.state.last_activity = time.time()
                        next_height = block_height + 1

                        if block_height % 100 == 0:
                            logger.info(
                                f"Sync progress: block #{block_height} "
                                f"({self.state.progress:.1%})"
                            )
                    else:
                        logger.warning(
                            f"Failed to add block #{block_height}"
                        )
                        self.state.errors += 1

                except Exception as e:
                    logger.error(
                        f"Error processing block #{block_height}: {e}"
                    )
                    self.state.errors += 1

    def on_blocks_received(self, blocks: List[dict], from_peer: str):
        """
        Handle blocks received from a peer.
        Called by the server when BLOCKS or SYNC_RESPONSE messages arrive.
        """
        if not blocks:
            return

        count = len(blocks)
        self.state.blocks_downloaded += count
        self.state.last_activity = time.time()

        # Determine the start height from the first block
        first_height = blocks[0].get("header", {}).get("height", 0)
        self._pending_blocks[first_height] = blocks

        logger.debug(
            f"Received {count} blocks from {from_peer[:8]} "
            f"(starting at height {first_height})"
        )

        # Update peer score (good behavior) and refresh last_seen
        # to prevent stale-peer pruning during active block sync.
        peer = self.peer_manager.get_peer(from_peer)
        if peer:
            peer.adjust_score(2)
            peer.update_seen()

    def on_status_received(self, peer_id: str, height: int, best_hash: str):
        """Handle a STATUS message - check if we need to sync."""
        peer = self.peer_manager.get_peer(peer_id)
        if peer:
            peer.chain_height = height
            peer.best_hash = best_hash

        # Update target height if this peer has a higher chain
        if self.state.syncing and height > self.state.target_height:
            self.state.target_height = height
            logger.debug(
                f"Updated sync target to height {height} from {peer_id[:8]}"
            )

    def finish_sync(self):
        """Mark sync as complete."""
        self.state.syncing = False
        self.state.phase = SyncPhase.COMPLETE
        self._pending_blocks.clear()

        # Reset sync peer state
        if self.state.sync_peer:
            peer = self.peer_manager.get_peer(self.state.sync_peer)
            if peer:
                peer.state = PeerState.CONNECTED

        elapsed = self.state.elapsed
        logger.info(
            f"Sync complete: {self.state.blocks_processed} blocks "
            f"in {elapsed:.1f}s "
            f"({self.state.blocks_per_second:.1f} blocks/s)"
        )

        # Notify the node so it can broadcast updated chain status to
        # peers that may be further downstream (indirect sync / gossip).
        if self._on_sync_complete_callback:
            try:
                self._on_sync_complete_callback()
            except Exception as e:
                logger.debug(f"on_sync_complete callback error: {e}")

    async def stop(self):
        """Stop the sync process."""
        self._running = False
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
        self.state.syncing = False
        self._pending_blocks.clear()

    def handle_sync_request(
        self,
        peer_id: str,
        request_type: str,
        start_height: int,
        count: int,
    ) -> Optional[List[dict]]:
        """
        Handle an incoming sync request from a peer.
        Returns a list of block dicts to send back.
        """
        if not self._get_block_callback:
            return None

        # Limit batch size
        count = min(count, self.BATCH_SIZE)
        blocks = []

        for h in range(start_height, start_height + count):
            block = self._get_block_callback(h)
            if block is None:
                break
            blocks.append(block)

        return blocks if blocks else None

    def get_stats(self) -> dict:
        return {
            "syncing": self.state.syncing,
            "phase": self.state.phase.name,
            "progress": f"{self.state.progress:.1%}",
            "target_height": self.state.target_height,
            "current_height": self.state.current_height,
            "blocks_downloaded": self.state.blocks_downloaded,
            "blocks_processed": self.state.blocks_processed,
            "errors": self.state.errors,
            "sync_peer": (
                self.state.sync_peer[:8] if self.state.sync_peer else None
            ),
            "elapsed": f"{self.state.elapsed:.1f}s",
            "rate": f"{self.state.blocks_per_second:.1f} blocks/s",
            "pending_batches": len(self._pending_blocks),
        }
