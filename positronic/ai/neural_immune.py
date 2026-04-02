"""
Positronic - Neural Immune System
AI defense system modeled after biological white blood cells.
Monitors network health and responds to threats with escalating alert levels.

Audit fix: Blocked addresses now have time-based expiry (24h default)
instead of permanent blocking. Appeals mechanism allows governance-gated
unblocking with deposit.
"""

from enum import IntEnum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time

from positronic.constants import IMMUNE_BLOCK_DURATION, IMMUNE_APPEAL_DEPOSIT


class AlertLevel(IntEnum):
    """Network threat alert levels."""
    GREEN = 0       # Normal operation - all healthy
    YELLOW = 1      # Elevated caution - minor anomalies detected
    ORANGE = 2      # High alert - significant suspicious activity
    RED = 3         # Critical - active attack detected
    BLACK = 4       # Emergency - network under severe attack


@dataclass
class ThreatEvent:
    """A detected threat event."""
    event_type: str
    severity: float           # 0.0 to 1.0
    source_address: bytes
    timestamp: float
    description: str
    block_height: int = 0
    resolved: bool = False

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "severity": self.severity,
            "source_address": self.source_address.hex(),
            "timestamp": self.timestamp,
            "description": self.description,
            "block_height": self.block_height,
            "resolved": self.resolved,
        }


@dataclass
class ImmuneResponse:
    """Action taken by the immune system."""
    action_type: str          # "quarantine", "block", "alert", "escalate"
    target_address: bytes
    timestamp: float
    alert_level: AlertLevel
    details: str = ""

    def to_dict(self) -> dict:
        return {
            "action_type": self.action_type,
            "target_address": self.target_address.hex(),
            "timestamp": self.timestamp,
            "alert_level": self.alert_level.name,
            "details": self.details,
        }


@dataclass
class AppealRequest:
    """Appeal request to unblock an address."""
    address: bytes
    deposit: int
    timestamp: float
    status: str = "pending"  # pending, approved, rejected
    votes_for: int = 0
    votes_against: int = 0

    def to_dict(self) -> dict:
        return {
            "address": self.address.hex(),
            "deposit": self.deposit,
            "timestamp": self.timestamp,
            "status": self.status,
            "votes_for": self.votes_for,
            "votes_against": self.votes_against,
        }


class NeuralImmuneSystem:
    """
    AI-powered immune system for the Positronic network.
    Like white blood cells, each node runs this to defend the network.
    Monitors for threats and responds with appropriate escalation.

    Audit fixes applied:
    - Blocked addresses expire after IMMUNE_BLOCK_DURATION (24h default)
    - unblock_address() allows manual unblocking
    - request_appeal() creates governance-gated appeal with deposit
    - clear_old_threats() also expires old blocked addresses
    - is_address_blocked() auto-expires stale blocks

    Integrates with SybilDetector for multi-identity attack detection.
    """

    # Alert level thresholds
    YELLOW_THRESHOLD = 5      # 5+ minor anomalies in window
    ORANGE_THRESHOLD = 10     # 10+ anomalies or 3+ high severity
    RED_THRESHOLD = 20        # 20+ anomalies or active attack pattern
    BLACK_THRESHOLD = 50      # 50+ anomalies or network-wide attack

    # Time window for threat analysis (seconds)
    ANALYSIS_WINDOW = 300     # 5 minutes

    def __init__(self):
        """Initialize the neural immune system with default thresholds."""
        self.alert_level: AlertLevel = AlertLevel.GREEN
        self._threats: List[ThreatEvent] = []
        self._responses: List[ImmuneResponse] = []
        # Timed blocking: address -> blocked_until_timestamp
        self._blocked_addresses: Dict[bytes, float] = {}
        self._threat_counts: Dict[str, int] = {}
        self._last_evaluation: float = time.time()
        self._appeals: Dict[bytes, AppealRequest] = {}

        # Sybil detector integration
        self._sybil_detector = None
        try:
            from positronic.ai.sybil_detector import SybilDetector
            self._sybil_detector = SybilDetector()
        except ImportError:
            pass

    def report_threat(self, event: ThreatEvent) -> ImmuneResponse:
        """Report a new threat event and get immune response."""
        self._threats.append(event)

        # Count by type
        self._threat_counts[event.event_type] = (
            self._threat_counts.get(event.event_type, 0) + 1
        )

        # Evaluate alert level
        self._evaluate_alert_level()

        # Generate response
        response = self._generate_response(event)
        self._responses.append(response)

        return response

    def report_anomaly(self, address: bytes, score: float,
                       description: str, block_height: int = 0) -> ImmuneResponse:
        """Convenience method to report an anomaly."""
        event = ThreatEvent(
            event_type="anomaly",
            severity=score,
            source_address=address,
            timestamp=time.time(),
            description=description,
            block_height=block_height,
        )
        return self.report_threat(event)

    def report_attack(self, attack_type: str, address: bytes,
                      description: str, block_height: int = 0) -> ImmuneResponse:
        """Report a detected attack."""
        event = ThreatEvent(
            event_type=attack_type,
            severity=0.9,
            source_address=address,
            timestamp=time.time(),
            description=description,
            block_height=block_height,
        )
        return self.report_threat(event)

    def _evaluate_alert_level(self):
        """Evaluate current alert level based on recent threats."""
        now = time.time()
        recent = [t for t in self._threats
                  if now - t.timestamp < self.ANALYSIS_WINDOW]

        count = len(recent)
        high_severity = sum(1 for t in recent if t.severity >= 0.8)

        if count >= self.BLACK_THRESHOLD or high_severity >= 20:
            self.alert_level = AlertLevel.BLACK
        elif count >= self.RED_THRESHOLD or high_severity >= 10:
            self.alert_level = AlertLevel.RED
        elif count >= self.ORANGE_THRESHOLD or high_severity >= 3:
            self.alert_level = AlertLevel.ORANGE
        elif count >= self.YELLOW_THRESHOLD:
            self.alert_level = AlertLevel.YELLOW
        else:
            self.alert_level = AlertLevel.GREEN

    def _generate_response(self, event: ThreatEvent) -> ImmuneResponse:
        """Generate an immune response based on threat and alert level."""
        if self.alert_level >= AlertLevel.RED:
            action = "block"
            # Block with time-based expiry instead of permanent
            blocked_until = time.time() + IMMUNE_BLOCK_DURATION
            self._blocked_addresses[event.source_address] = blocked_until
        elif self.alert_level >= AlertLevel.ORANGE:
            action = "quarantine"
        elif event.severity >= 0.8:
            action = "alert"
        else:
            action = "monitor"

        return ImmuneResponse(
            action_type=action,
            target_address=event.source_address,
            timestamp=time.time(),
            alert_level=self.alert_level,
            details=f"Threat: {event.event_type}, Severity: {event.severity:.2f}",
        )

    def on_transaction(self, sender: bytes, recipient: bytes,
                       value: int, gas_price: int, timestamp: float = 0.0):
        """Feed transaction data to Sybil detector."""
        if self._sybil_detector:
            self._sybil_detector.on_transaction(
                sender, recipient, value, gas_price, timestamp
            )

    def on_peer_connection(self, peer_id: str, ip_address: str) -> Optional[str]:
        """Check peer connection for Sybil patterns."""
        if self._sybil_detector:
            return self._sybil_detector.on_peer_connection(peer_id, ip_address)
        return None

    def run_sybil_analysis(self) -> List[dict]:
        """
        Run Sybil analysis and report any detected clusters as threats.
        Should be called periodically (e.g., every epoch).
        """
        if not self._sybil_detector:
            return []

        clusters = self._sybil_detector.analyze()
        results = []

        for cluster in clusters:
            # Report as a threat event
            for addr in cluster.accounts:
                if cluster.confidence >= 0.7:
                    self.report_attack(
                        "sybil_attack",
                        addr,
                        f"Sybil cluster #{cluster.cluster_id}: "
                        f"{cluster.detection_reason} "
                        f"(confidence={cluster.confidence:.2f})",
                    )
            results.append({
                "cluster_id": cluster.cluster_id,
                "accounts": len(cluster.accounts),
                "confidence": cluster.confidence,
                "reason": cluster.detection_reason,
            })

        return results

    def is_address_blocked(self, address: bytes) -> bool:
        """
        Check if an address is blocked by the immune system.
        Auto-expires stale blocks (addresses blocked past their expiry time).
        """
        if address in self._blocked_addresses:
            blocked_until = self._blocked_addresses[address]
            if time.time() < blocked_until:
                return True
            else:
                # Block expired — auto-remove
                del self._blocked_addresses[address]
                return False
        # Also check Sybil flags
        if self._sybil_detector and self._sybil_detector.is_flagged(address):
            return True
        return False

    def unblock_address(self, address: bytes) -> bool:
        """
        Manually unblock an address (governance action).
        Returns True if the address was blocked and has been unblocked.
        """
        if address in self._blocked_addresses:
            del self._blocked_addresses[address]
            return True
        return False

    def request_appeal(self, address: bytes, deposit: int) -> Optional[AppealRequest]:
        """
        Request an appeal to unblock an address.
        Requires a deposit of IMMUNE_APPEAL_DEPOSIT ASF.
        Returns the appeal request if valid, None if deposit too low.
        """
        if deposit < IMMUNE_APPEAL_DEPOSIT:
            return None

        if address not in self._blocked_addresses:
            return None  # Not blocked, no appeal needed

        appeal = AppealRequest(
            address=address,
            deposit=deposit,
            timestamp=time.time(),
        )
        self._appeals[address] = appeal
        return appeal

    def resolve_appeal(self, address: bytes, approved: bool) -> bool:
        """
        Resolve an appeal (called by governance after vote).
        If approved, unblocks the address and returns deposit.
        If rejected, deposit is slashed (kept by protocol).
        """
        appeal = self._appeals.get(address)
        if appeal is None or appeal.status != "pending":
            return False

        if approved:
            appeal.status = "approved"
            self.unblock_address(address)
        else:
            appeal.status = "rejected"
            # Deposit is slashed (handled by caller)

        return True

    def get_appeal(self, address: bytes) -> Optional[dict]:
        """Get appeal status for an address."""
        appeal = self._appeals.get(address)
        return appeal.to_dict() if appeal else None

    def _expire_blocked_addresses(self):
        """Remove expired blocked addresses."""
        now = time.time()
        expired = [addr for addr, until in self._blocked_addresses.items()
                   if now >= until]
        for addr in expired:
            del self._blocked_addresses[addr]

    def clear_old_threats(self, max_age: float = 3600):
        """Clear threats older than max_age seconds and expire old blocks."""
        now = time.time()
        self._threats = [t for t in self._threats if now - t.timestamp < max_age]
        self._evaluate_alert_level()
        # Also expire blocked addresses
        self._expire_blocked_addresses()

    def get_recent_threats(self, limit: int = 50) -> List[dict]:
        """Get recent threat events."""
        return [t.to_dict() for t in self._threats[-limit:]]

    def get_recent_responses(self, limit: int = 50) -> List[dict]:
        """Get recent immune responses."""
        return [r.to_dict() for r in self._responses[-limit:]]

    def get_status(self) -> dict:
        """Get current immune system status."""
        # Auto-expire before reporting
        self._expire_blocked_addresses()
        status = {
            "alert_level": self.alert_level.name,
            "alert_level_value": int(self.alert_level),
            "total_threats": len(self._threats),
            "blocked_addresses": len(self._blocked_addresses),
            "active_appeals": sum(1 for a in self._appeals.values() if a.status == "pending"),
            "threat_types": dict(self._threat_counts),
            "block_duration_seconds": IMMUNE_BLOCK_DURATION,
        }
        if self._sybil_detector:
            status["sybil_detection"] = self._sybil_detector.get_stats()
        return status
