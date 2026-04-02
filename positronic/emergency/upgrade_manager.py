"""
Positronic - Upgrade Manager
Hard fork scheduling, feature flags, and auto-rollback for protocol upgrades.
"""

import time
import uuid
import logging
from enum import IntEnum
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from positronic.constants import (
    UPGRADE_MONITOR_BLOCKS,
    UPGRADE_ROLLBACK_ERROR_THRESHOLD,
    UPGRADE_MIN_ACTIVATION_DELAY,
)

logger = logging.getLogger("positronic.emergency.upgrade_manager")


class UpgradeStatus(IntEnum):
    """Status of a scheduled upgrade."""
    SCHEDULED = 0
    ACTIVATED = 1
    ROLLED_BACK = 2
    CANCELLED = 3


class FeatureStatus(IntEnum):
    """Status of a feature flag."""
    SCHEDULED = 0
    ACTIVE = 1
    LOCKED = 2
    ROLLED_BACK = 3


@dataclass
class ScheduledUpgrade:
    """A protocol upgrade scheduled for activation at a specific block."""
    upgrade_id: str
    name: str
    activation_block: int
    features: List[str]
    min_version: str
    status: UpgradeStatus = UpgradeStatus.SCHEDULED
    checkpoint_block: int = 0  # Block before activation (for rollback)
    created_at: float = 0.0
    activated_at: Optional[float] = None


@dataclass
class FeatureFlag:
    """A feature flag that can be toggled by upgrades."""
    name: str
    activation_block: int
    status: FeatureStatus = FeatureStatus.SCHEDULED
    upgrade_id: str = ""


class UpgradeManager:
    """
    Manages protocol upgrades with feature flags and auto-rollback.

    Lifecycle:
    1. schedule_upgrade()      -- schedule a hard fork at a future block
    2. check_activation()      -- called per-block, activates features at target block
    3. monitor_post_upgrade()  -- monitors error rate for 100 blocks after activation
    4. rollback_upgrade()      -- deactivate features and revert to checkpoint
    """

    def __init__(self):
        self.upgrades: Dict[str, ScheduledUpgrade] = {}
        self.features: Dict[str, FeatureFlag] = {}
        self.current_protocol: int = 1
        self._current_block: int = 0  # Tracked via check_activation calls

    def schedule_upgrade(
        self,
        name: str,
        activation_block: int,
        features: List[str],
        min_version: str,
    ) -> Optional[str]:
        """
        Schedule a new protocol upgrade.
        Returns upgrade_id or None if validation fails.
        """
        # Validate: activation must be far enough in the future
        if activation_block - self._current_block < UPGRADE_MIN_ACTIVATION_DELAY:
            logger.warning(
                f"Upgrade '{name}' too close: block {activation_block} is only "
                f"{activation_block - self._current_block} blocks ahead "
                f"(need {UPGRADE_MIN_ACTIVATION_DELAY})"
            )
            return None

        upgrade_id = str(uuid.uuid4())
        upgrade = ScheduledUpgrade(
            upgrade_id=upgrade_id,
            name=name,
            activation_block=activation_block,
            features=features,
            min_version=min_version,
            checkpoint_block=activation_block - 1,
            created_at=time.time(),
        )
        self.upgrades[upgrade_id] = upgrade

        # Register feature flags
        for feat_name in features:
            self.features[feat_name] = FeatureFlag(
                name=feat_name,
                activation_block=activation_block,
                status=FeatureStatus.SCHEDULED,
                upgrade_id=upgrade_id,
            )

        logger.info(
            f"Upgrade '{name}' scheduled at block {activation_block} "
            f"with features: {features}"
        )
        return upgrade_id

    def check_activation(self, current_block: int) -> List[str]:
        """
        Called every block. Activates features when their activation block is reached.
        Returns list of newly activated feature names.
        """
        self._current_block = current_block
        activated = []

        for uid, upgrade in self.upgrades.items():
            if upgrade.status != UpgradeStatus.SCHEDULED:
                continue
            if current_block >= upgrade.activation_block:
                # Activate all features in this upgrade
                for feat_name in upgrade.features:
                    feat = self.features.get(feat_name)
                    if feat and feat.status == FeatureStatus.SCHEDULED:
                        feat.status = FeatureStatus.ACTIVE
                        activated.append(feat_name)

                upgrade.status = UpgradeStatus.ACTIVATED
                upgrade.activated_at = time.time()
                logger.info(
                    f"Upgrade '{upgrade.name}' activated at block {current_block}: "
                    f"{upgrade.features}"
                )

        return activated

    def is_feature_active(self, feature_name: str) -> bool:
        """Check if a feature flag is currently active."""
        feat = self.features.get(feature_name)
        return feat is not None and feat.status == FeatureStatus.ACTIVE

    def get_active_features(self) -> List[str]:
        """Return all currently active feature names."""
        return [
            name for name, feat in self.features.items()
            if feat.status == FeatureStatus.ACTIVE
        ]

    def rollback_upgrade(self, upgrade_id: str) -> bool:
        """Rollback an upgrade: deactivate all its features."""
        upgrade = self.upgrades.get(upgrade_id)
        if not upgrade:
            logger.warning(f"Upgrade {upgrade_id} not found")
            return False

        if upgrade.status not in (UpgradeStatus.ACTIVATED, UpgradeStatus.SCHEDULED):
            logger.warning(f"Upgrade {upgrade_id} is {upgrade.status.name}, cannot rollback")
            return False

        # Deactivate features
        for feat_name in upgrade.features:
            feat = self.features.get(feat_name)
            if feat:
                feat.status = FeatureStatus.ROLLED_BACK

        upgrade.status = UpgradeStatus.ROLLED_BACK
        logger.warning(f"Upgrade '{upgrade.name}' ROLLED BACK")
        return True

    def cancel_upgrade(self, upgrade_id: str) -> bool:
        """Cancel a scheduled (not yet activated) upgrade."""
        upgrade = self.upgrades.get(upgrade_id)
        if not upgrade:
            return False

        if upgrade.status != UpgradeStatus.SCHEDULED:
            logger.warning(f"Can only cancel SCHEDULED upgrades, got {upgrade.status.name}")
            return False

        for feat_name in upgrade.features:
            if feat_name in self.features:
                del self.features[feat_name]

        upgrade.status = UpgradeStatus.CANCELLED
        logger.info(f"Upgrade '{upgrade.name}' cancelled")
        return True

    def monitor_post_upgrade(self, current_block: int, error_rate: float) -> bool:
        """
        Monitor error rate after an upgrade activation.
        Returns True if auto-rollback should be triggered.
        """
        for uid, upgrade in self.upgrades.items():
            if upgrade.status != UpgradeStatus.ACTIVATED:
                continue

            blocks_since = current_block - upgrade.activation_block
            if blocks_since > UPGRADE_MONITOR_BLOCKS:
                continue  # Monitoring period over, upgrade is stable

            if error_rate > UPGRADE_ROLLBACK_ERROR_THRESHOLD:
                logger.warning(
                    f"Upgrade '{upgrade.name}' error rate {error_rate:.2%} > "
                    f"{UPGRADE_ROLLBACK_ERROR_THRESHOLD:.2%} threshold at "
                    f"block +{blocks_since}. Auto-rollback recommended."
                )
                return True

        return False

    def get_upgrade_status(self, upgrade_id: str) -> dict:
        """Get status of a specific upgrade."""
        upgrade = self.upgrades.get(upgrade_id)
        if not upgrade:
            return {"error": "Upgrade not found"}
        return {
            "upgrade_id": upgrade.upgrade_id,
            "name": upgrade.name,
            "activation_block": upgrade.activation_block,
            "features": upgrade.features,
            "min_version": upgrade.min_version,
            "status": upgrade.status.name,
            "checkpoint_block": upgrade.checkpoint_block,
            "activated_at": upgrade.activated_at,
        }

    def get_scheduled_upgrades(self) -> List[dict]:
        """Get all upgrades (any status)."""
        return [self.get_upgrade_status(uid) for uid in self.upgrades]

    def check_peer_compatibility(self, peer_version: str, peer_features: List[str]) -> dict:
        """
        Check if a peer is compatible with current protocol.
        Called during peer handshake.
        """
        active = set(self.get_active_features())
        peer_set = set(peer_features)
        missing = active - peer_set
        return {
            "compatible": len(missing) == 0,
            "missing_features": list(missing),
            "peer_version": peer_version,
        }
