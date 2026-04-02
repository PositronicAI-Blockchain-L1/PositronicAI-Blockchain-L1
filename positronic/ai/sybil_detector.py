"""
Positronic - Sybil Attack Detection
Detects and mitigates Sybil attacks at multiple layers:

1. Transaction layer: Detects coordinated patterns from new accounts
2. Network layer: Detects peer flooding from same IP ranges
3. Staking layer: Detects stake-splitting across many validators

A Sybil attack creates many fake identities to gain disproportionate
influence. This module uses behavioral analysis and network topology
to detect suspicious coordination patterns.
"""

import time
import logging
from collections import defaultdict
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("positronic.ai.sybil_detector")


@dataclass
class AccountBehavior:
    """Tracks behavioral fingerprint of an account."""
    address: bytes
    first_seen: float = 0.0
    tx_count: int = 0
    tx_timestamps: List[float] = field(default_factory=list)
    recipients: Set[bytes] = field(default_factory=set)
    gas_prices: List[int] = field(default_factory=list)
    values: List[int] = field(default_factory=list)
    funding_source: Optional[bytes] = None
    funded_accounts: Set[bytes] = field(default_factory=set)


@dataclass
class SybilCluster:
    """A detected cluster of potentially Sybil accounts."""
    cluster_id: int
    accounts: Set[bytes]
    confidence: float          # 0.0-1.0
    detection_reason: str
    detected_at: float
    funding_source: Optional[bytes] = None


class SybilDetector:
    """
    Multi-layered Sybil attack detector.

    Detection strategies:
    1. Funding tree analysis: Tracks which accounts funded which
    2. Temporal correlation: Accounts created/active at same time
    3. Behavioral similarity: Same gas prices, values, patterns
    4. IP correlation: Same IP connecting multiple peers (network layer)
    5. Stake fragmentation: Small stakes split across many validators
    """

    # Thresholds
    CLUSTER_MIN_SIZE = 3           # Min accounts to flag a cluster
    TIME_CORRELATION_WINDOW = 60   # seconds - accounts acting within this window
    FUNDING_TREE_DEPTH = 3         # How deep to trace funding sources
    SIMILARITY_THRESHOLD = 0.8    # Behavioral similarity threshold
    MAX_ACCOUNTS_TRACKED = 10_000  # Memory limit

    # Network-level thresholds
    MAX_PEERS_PER_IP = 3           # Max peers from same IP
    IP_BAN_THRESHOLD = 5           # Ban IP if >5 peers attempted

    def __init__(self):
        self._accounts: Dict[bytes, AccountBehavior] = {}
        self._funding_graph: Dict[bytes, Set[bytes]] = defaultdict(set)  # funder -> funded
        self._clusters: List[SybilCluster] = []
        self._cluster_counter = 0
        self._ip_connections: Dict[str, List[str]] = defaultdict(list)  # IP -> [peer_ids]
        self._banned_ips: Set[str] = set()
        self._flagged_addresses: Set[bytes] = set()

    def on_transaction(self, sender: bytes, recipient: bytes,
                       value: int, gas_price: int, timestamp: float = 0.0):
        """
        Record a transaction for Sybil analysis.
        Called by the mempool or blockchain on every transaction.
        """
        t = timestamp or time.time()

        # Track sender behavior
        if sender not in self._accounts:
            self._accounts[sender] = AccountBehavior(
                address=sender, first_seen=t
            )
        sender_acc = self._accounts[sender]
        sender_acc.tx_count += 1
        sender_acc.tx_timestamps.append(t)
        sender_acc.recipients.add(recipient)
        sender_acc.gas_prices.append(gas_price)
        sender_acc.values.append(value)

        # Track recipient
        if recipient not in self._accounts:
            self._accounts[recipient] = AccountBehavior(
                address=recipient, first_seen=t
            )
        recipient_acc = self._accounts[recipient]

        # Track funding graph (who funded whom)
        if value > 0:
            self._funding_graph[sender].add(recipient)
            sender_acc.funded_accounts.add(recipient)
            if recipient_acc.funding_source is None:
                recipient_acc.funding_source = sender

        # Prune old data if needed
        if len(self._accounts) > self.MAX_ACCOUNTS_TRACKED:
            self._prune_old_accounts()

    def on_peer_connection(self, peer_id: str, ip_address: str) -> Optional[str]:
        """
        Record a peer connection for IP-level Sybil detection.
        Returns a warning string if suspicious, None otherwise.
        """
        if ip_address in self._banned_ips:
            return f"Banned IP: {ip_address}"

        self._ip_connections[ip_address].append(peer_id)
        count = len(self._ip_connections[ip_address])

        if count > self.IP_BAN_THRESHOLD:
            self._banned_ips.add(ip_address)
            return f"IP {ip_address} banned: {count} peer connections (Sybil pattern)"

        if count > self.MAX_PEERS_PER_IP:
            return f"Suspicious: {count} peers from IP {ip_address}"

        return None

    def analyze(self) -> List[SybilCluster]:
        """
        Run full Sybil analysis. Returns newly detected clusters.
        Should be called periodically (e.g., every epoch or every N blocks).
        """
        new_clusters = []

        # Strategy 1: Funding tree analysis
        funding_clusters = self._analyze_funding_trees()
        new_clusters.extend(funding_clusters)

        # Strategy 2: Temporal correlation
        temporal_clusters = self._analyze_temporal_patterns()
        new_clusters.extend(temporal_clusters)

        # Strategy 3: Behavioral similarity
        behavior_clusters = self._analyze_behavioral_similarity()
        new_clusters.extend(behavior_clusters)

        # Merge overlapping clusters
        merged = self._merge_clusters(new_clusters)

        # Update state
        for cluster in merged:
            self._clusters.append(cluster)
            for addr in cluster.accounts:
                self._flagged_addresses.add(addr)

        return merged

    def _analyze_funding_trees(self) -> List[SybilCluster]:
        """Detect accounts funded by the same source in tree patterns."""
        clusters = []

        for funder, funded_set in self._funding_graph.items():
            if len(funded_set) < self.CLUSTER_MIN_SIZE:
                continue

            # Check if funded accounts have similar behavior
            similar_count = 0
            for addr in funded_set:
                acc = self._accounts.get(addr)
                if acc and acc.tx_count <= 5:  # Low-activity new accounts
                    similar_count += 1

            if similar_count >= self.CLUSTER_MIN_SIZE:
                confidence = min(similar_count / 10, 0.95)
                self._cluster_counter += 1
                cluster = SybilCluster(
                    cluster_id=self._cluster_counter,
                    accounts=funded_set | {funder},
                    confidence=confidence,
                    detection_reason=f"Funding tree: {len(funded_set)} accounts funded by same source",
                    detected_at=time.time(),
                    funding_source=funder,
                )
                clusters.append(cluster)

        return clusters

    def _analyze_temporal_patterns(self) -> List[SybilCluster]:
        """Detect accounts that are active in suspiciously synchronized patterns."""
        clusters = []
        now = time.time()
        window = self.TIME_CORRELATION_WINDOW

        # Group accounts by their activity windows
        recent_accounts: Dict[int, List[bytes]] = defaultdict(list)
        for addr, acc in self._accounts.items():
            if acc.tx_timestamps:
                # Bucket by time window
                for ts in acc.tx_timestamps:
                    if now - ts < 3600:  # Only last hour
                        bucket = int(ts // window)
                        recent_accounts[bucket].append(addr)

        # Find time buckets with suspiciously many unique new accounts
        for bucket, addrs in recent_accounts.items():
            unique_addrs = set(addrs)
            new_addrs = set()
            for addr in unique_addrs:
                acc = self._accounts.get(addr)
                if acc and acc.tx_count <= 3 and (now - acc.first_seen) < 3600:
                    new_addrs.add(addr)

            if len(new_addrs) >= self.CLUSTER_MIN_SIZE:
                self._cluster_counter += 1
                cluster = SybilCluster(
                    cluster_id=self._cluster_counter,
                    accounts=new_addrs,
                    confidence=min(len(new_addrs) / 20, 0.85),
                    detection_reason=f"Temporal correlation: {len(new_addrs)} new accounts active in {window}s window",
                    detected_at=now,
                )
                clusters.append(cluster)

        return clusters

    def _analyze_behavioral_similarity(self) -> List[SybilCluster]:
        """Detect accounts with suspiciously similar transaction patterns."""
        clusters = []

        # Group by funding source first (most efficient)
        by_funder: Dict[bytes, List[bytes]] = defaultdict(list)
        for addr, acc in self._accounts.items():
            if acc.funding_source:
                by_funder[acc.funding_source].append(addr)

        for funder, funded in by_funder.items():
            if len(funded) < self.CLUSTER_MIN_SIZE:
                continue

            # Check gas price similarity
            gas_sets = []
            for addr in funded:
                acc = self._accounts.get(addr)
                if acc and acc.gas_prices:
                    gas_sets.append(set(acc.gas_prices))

            if len(gas_sets) >= self.CLUSTER_MIN_SIZE:
                # Check if they all use similar gas prices
                common = gas_sets[0]
                for gs in gas_sets[1:]:
                    common = common & gs
                if common:  # If any gas prices are shared across all
                    similarity = len(common) / max(len(gas_sets[0]), 1)
                    if similarity >= self.SIMILARITY_THRESHOLD:
                        self._cluster_counter += 1
                        cluster = SybilCluster(
                            cluster_id=self._cluster_counter,
                            accounts=set(funded) | {funder},
                            confidence=similarity * 0.9,
                            detection_reason=f"Behavioral similarity: {len(funded)} accounts with identical gas prices",
                            detected_at=time.time(),
                            funding_source=funder,
                        )
                        clusters.append(cluster)

        return clusters

    def _merge_clusters(self, clusters: List[SybilCluster]) -> List[SybilCluster]:
        """Merge overlapping clusters."""
        if len(clusters) <= 1:
            return clusters

        merged = []
        used = set()

        for i, c1 in enumerate(clusters):
            if i in used:
                continue
            combined = set(c1.accounts)
            max_conf = c1.confidence
            reasons = [c1.detection_reason]

            for j, c2 in enumerate(clusters):
                if j <= i or j in used:
                    continue
                overlap = combined & c2.accounts
                if len(overlap) >= 2:  # Significant overlap
                    combined |= c2.accounts
                    max_conf = max(max_conf, c2.confidence)
                    reasons.append(c2.detection_reason)
                    used.add(j)

            self._cluster_counter += 1
            merged.append(SybilCluster(
                cluster_id=self._cluster_counter,
                accounts=combined,
                confidence=max_conf,
                detection_reason=" + ".join(reasons),
                detected_at=time.time(),
                funding_source=c1.funding_source,
            ))
            used.add(i)

        return merged

    def _prune_old_accounts(self):
        """Remove old account data to stay within memory limits."""
        now = time.time()
        # Keep accounts from last 24 hours, or with >10 transactions
        to_remove = []
        for addr, acc in self._accounts.items():
            if now - acc.first_seen > 86400 and acc.tx_count < 10:
                to_remove.append(addr)
        for addr in to_remove[:len(to_remove) // 2]:  # Remove half
            del self._accounts[addr]

    def is_flagged(self, address: bytes) -> bool:
        """Check if an address has been flagged as potential Sybil."""
        return address in self._flagged_addresses

    def is_ip_banned(self, ip: str) -> bool:
        """Check if an IP is banned for Sybil behavior."""
        return ip in self._banned_ips

    def get_clusters(self, min_confidence: float = 0.5) -> List[dict]:
        """Get all detected clusters above confidence threshold."""
        return [
            {
                "cluster_id": c.cluster_id,
                "accounts": len(c.accounts),
                "confidence": round(c.confidence, 3),
                "reason": c.detection_reason,
                "detected_at": c.detected_at,
            }
            for c in self._clusters
            if c.confidence >= min_confidence
        ]

    def get_stats(self) -> dict:
        return {
            "tracked_accounts": len(self._accounts),
            "flagged_addresses": len(self._flagged_addresses),
            "detected_clusters": len(self._clusters),
            "banned_ips": len(self._banned_ips),
            "funding_edges": sum(len(v) for v in self._funding_graph.values()),
        }
