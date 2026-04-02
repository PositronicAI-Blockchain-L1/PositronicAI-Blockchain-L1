"""
Positronic - Node Ranking System (Gamified)
Nodes earn ranks from Probe to Star Positronic based on uptime and performance.
Ranks are evaluated quarterly (every 90 days / 2,592,000 blocks).
"""

from enum import IntEnum
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time


class NodeRank(IntEnum):
    """Sci-fi themed node ranks."""
    PROBE = 1           # Level 1 - New node, just joined
    SENTINEL = 2        # Level 2 - Basic reliability proven
    CIRCUIT = 3         # Level 3 - Solid performer
    RELAY = 4           # Level 4 - Reliable community member
    CORTEX = 5          # Level 5 - Veteran node operator
    NEXUS = 6           # Level 6 - Elite operator
    POSITRONIC = 7      # Level 7 - Master node
    STAR_POSITRONIC = 8 # Level 8 - Legendary (positronic star)


# Rank requirements
NODE_RANK_REQUIREMENTS = {
    NodeRank.PROBE: {"min_uptime_pct": 0.0, "min_blocks_validated": 0, "min_days": 0},
    NodeRank.SENTINEL: {"min_uptime_pct": 50.0, "min_blocks_validated": 100, "min_days": 7},
    NodeRank.CIRCUIT: {"min_uptime_pct": 70.0, "min_blocks_validated": 1000, "min_days": 30},
    NodeRank.RELAY: {"min_uptime_pct": 80.0, "min_blocks_validated": 10000, "min_days": 60},
    NodeRank.CORTEX: {"min_uptime_pct": 90.0, "min_blocks_validated": 50000, "min_days": 90},
    NodeRank.NEXUS: {"min_uptime_pct": 95.0, "min_blocks_validated": 200000, "min_days": 180},
    NodeRank.POSITRONIC: {"min_uptime_pct": 97.0, "min_blocks_validated": 500000, "min_days": 365},
    NodeRank.STAR_POSITRONIC: {"min_uptime_pct": 99.0, "min_blocks_validated": 1000000, "min_days": 730},
}

# Reward multipliers by rank
NODE_REWARD_MULTIPLIER = {
    NodeRank.PROBE: 1.0,
    NodeRank.SENTINEL: 1.3,
    NodeRank.CIRCUIT: 1.6,
    NodeRank.RELAY: 2.0,
    NodeRank.CORTEX: 2.5,
    NodeRank.NEXUS: 3.0,
    NodeRank.POSITRONIC: 4.0,
    NodeRank.STAR_POSITRONIC: 5.0,
}

# Evaluation interval
EVALUATION_INTERVAL_BLOCKS = 648_000      # ~90 days at 12 sec/block
EVALUATION_INTERVAL_DAYS = 90


@dataclass
class NodeProfile:
    """Profile tracking a node operator's performance and rank."""
    address: bytes
    node_id: str = ""
    rank: NodeRank = NodeRank.PROBE
    joined_at: float = 0.0
    last_seen: float = 0.0
    total_uptime_seconds: float = 0.0
    total_possible_seconds: float = 0.0
    blocks_validated: int = 0
    blocks_proposed: int = 0
    ai_scores_computed: int = 0
    penalties: int = 0
    last_evaluation_block: int = 0
    consecutive_evaluations_passed: int = 0

    @property
    def uptime_percentage(self) -> float:
        if self.total_possible_seconds == 0:
            return 0.0
        return (self.total_uptime_seconds / self.total_possible_seconds) * 100

    @property
    def days_active(self) -> int:
        if self.joined_at == 0:
            return 0
        return int((time.time() - self.joined_at) / 86400)

    @property
    def reward_multiplier(self) -> float:
        return NODE_REWARD_MULTIPLIER.get(self.rank, 1.0)

    @property
    def rank_name(self) -> str:
        return self.rank.name.replace("_", " ").title()

    def to_dict(self) -> dict:
        return {
            "address": self.address.hex(),
            "node_id": self.node_id,
            "rank": int(self.rank),
            "rank_name": self.rank_name,
            "uptime_percentage": round(self.uptime_percentage, 2),
            "blocks_validated": self.blocks_validated,
            "days_active": self.days_active,
            "reward_multiplier": self.reward_multiplier,
            "penalties": self.penalties,
            "consecutive_evaluations_passed": self.consecutive_evaluations_passed,
        }


class NodeRankingManager:
    """Manages the gamified node ranking system."""

    def __init__(self):
        self._nodes: Dict[bytes, NodeProfile] = {}

    def register_node(self, address: bytes, node_id: str = "") -> NodeProfile:
        """Register a new node at Probe rank."""
        now = time.time()
        profile = NodeProfile(
            address=address,
            node_id=node_id or address.hex()[:16],
            rank=NodeRank.PROBE,
            joined_at=now,
            last_seen=now,
        )
        self._nodes[address] = profile
        return profile

    def get_node(self, address: bytes) -> Optional[NodeProfile]:
        return self._nodes.get(address)

    def record_heartbeat(self, address: bytes, uptime_seconds: float = 3.0):
        """Record that a node is alive (called each block)."""
        node = self._nodes.get(address)
        if node:
            node.last_seen = time.time()
            node.total_uptime_seconds += uptime_seconds
            node.total_possible_seconds += uptime_seconds

    def record_block_validated(self, address: bytes):
        """Record that a node validated a block."""
        node = self._nodes.get(address)
        if node:
            node.blocks_validated += 1

    def record_block_proposed(self, address: bytes):
        """Record that a node proposed a block."""
        node = self._nodes.get(address)
        if node:
            node.blocks_proposed += 1

    def record_missed_block(self, address: bytes, missed_seconds: float = 3.0):
        """Record that a node missed a block (was offline)."""
        node = self._nodes.get(address)
        if node:
            node.total_possible_seconds += missed_seconds
            node.penalties += 1

    def evaluate_all(self, current_block: int) -> Dict[bytes, dict]:
        """
        Quarterly evaluation of all nodes.
        Returns dict of address -> evaluation result.
        Called every EVALUATION_INTERVAL_BLOCKS blocks.
        """
        results = {}
        for address, node in self._nodes.items():
            result = self._evaluate_node(node, current_block)
            results[address] = result
        return results

    def _evaluate_node(self, node: NodeProfile, current_block: int) -> dict:
        """Evaluate a single node for rank promotion/demotion."""
        old_rank = node.rank

        # Check for promotion
        if node.rank < NodeRank.STAR_POSITRONIC:
            next_rank = NodeRank(node.rank + 1)
            reqs = NODE_RANK_REQUIREMENTS[next_rank]

            if (node.uptime_percentage >= reqs["min_uptime_pct"]
                and node.blocks_validated >= reqs["min_blocks_validated"]
                and node.days_active >= reqs["min_days"]):
                node.rank = next_rank
                node.consecutive_evaluations_passed += 1

        # Check for demotion (uptime dropped below current rank requirement)
        current_reqs = NODE_RANK_REQUIREMENTS[node.rank]
        if (node.rank > NodeRank.PROBE and
            node.uptime_percentage < current_reqs["min_uptime_pct"] * 0.85):
            node.rank = NodeRank(node.rank - 1)
            node.consecutive_evaluations_passed = 0

        node.last_evaluation_block = current_block

        return {
            "address": node.address.hex(),
            "old_rank": old_rank.name,
            "new_rank": node.rank.name,
            "promoted": node.rank > old_rank,
            "demoted": node.rank < old_rank,
            "uptime": round(node.uptime_percentage, 2),
        }

    def get_leaderboard(self, limit: int = 50) -> List[dict]:
        """Get node leaderboard sorted by rank then uptime."""
        sorted_nodes = sorted(
            self._nodes.values(),
            key=lambda n: (-n.rank, -n.uptime_percentage),
        )
        return [n.to_dict() for n in sorted_nodes[:limit]]

    def get_by_rank(self, rank: NodeRank) -> List[NodeProfile]:
        return [n for n in self._nodes.values() if n.rank == rank]

    @property
    def total_nodes(self) -> int:
        return len(self._nodes)

    def get_stats(self) -> dict:
        rank_dist = {}
        for rank in NodeRank:
            rank_dist[rank.name] = len(self.get_by_rank(rank))
        return {
            "total_nodes": self.total_nodes,
            "rank_distribution": rank_dist,
            "evaluation_interval_blocks": EVALUATION_INTERVAL_BLOCKS,
        }
