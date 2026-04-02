"""
Positronic - DePIN (Decentralized Physical Infrastructure)
Register physical devices (cameras, sensors, GPUs) to earn tokens.

Phase 22: Economic Layer
--------------------------
Replaces the flat "1 unit per submission" reward with a three-dimensional
scoring formula (uptime, data quality, freshness) multiplied by a
device-type tier.  A daily cap prevents gaming.

Score Formula
~~~~~~~~~~~~~
    score = uptime_w * uptime_pct
          + data_w   * data_quality
          + fresh_w  * freshness_factor

    reward = tier_multiplier * score        (capped to daily max)

Device Tiers
~~~~~~~~~~~~
    GPU:     5x  (highest -- AI computation)
    STORAGE: 3x
    SENSOR:  2x  (same as CAMERA)
    CAMERA:  2x
    NETWORK: 1x
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List
from enum import IntEnum
import time
import hashlib

from positronic.constants import (
    DEPIN_GPU_MULTIPLIER,
    DEPIN_STORAGE_MULTIPLIER,
    DEPIN_SENSOR_MULTIPLIER,
    DEPIN_NETWORK_MULTIPLIER,
    DEPIN_DAILY_DEVICE_CAP,
    DEPIN_UPTIME_WEIGHT,
    DEPIN_DATA_WEIGHT,
    DEPIN_FRESHNESS_WEIGHT,
    DEPIN_FRESHNESS_DECAY_MINS,
)


# Map device type → reward multiplier
_TIER_MULTIPLIER = {
    2: DEPIN_GPU_MULTIPLIER,      # DeviceType.GPU
    3: DEPIN_STORAGE_MULTIPLIER,  # DeviceType.STORAGE
    0: DEPIN_SENSOR_MULTIPLIER,   # DeviceType.CAMERA
    1: DEPIN_SENSOR_MULTIPLIER,   # DeviceType.SENSOR
    4: DEPIN_NETWORK_MULTIPLIER,  # DeviceType.NETWORK
}


class DeviceType(IntEnum):
    CAMERA = 0
    SENSOR = 1
    GPU = 2
    STORAGE = 3
    NETWORK = 4


class DeviceStatus(IntEnum):
    REGISTERED = 0
    ACTIVE = 1
    OFFLINE = 2
    BANNED = 3


@dataclass
class DePINDevice:
    """A physical device registered on the network."""
    device_id: str
    device_type: DeviceType
    owner: bytes
    location_lat: float = 0.0
    location_lon: float = 0.0
    status: DeviceStatus = DeviceStatus.REGISTERED
    uptime_hours: float = 0.0
    data_submitted: int = 0
    verified_submissions: int = 0
    rewards_earned: int = 0
    pending_rewards: int = 0
    last_heartbeat: float = 0.0
    registered_at: float = 0.0

    # Daily cap tracking
    _daily_rewards: int = 0
    _daily_reset_date: int = 0  # day number (days since epoch)

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    def heartbeat(self):
        """Record device heartbeat."""
        now = time.time()
        if self.last_heartbeat > 0:
            hours = (now - self.last_heartbeat) / 3600
            self.uptime_hours += hours
        self.last_heartbeat = now
        if self.status == DeviceStatus.REGISTERED:
            self.status = DeviceStatus.ACTIVE

    # ------------------------------------------------------------------
    # Phase 22: Device scoring
    # ------------------------------------------------------------------

    def compute_score(self, now: Optional[float] = None) -> float:
        """Compute a device quality score in [0, 1].

        score = uptime_w * uptime_pct
              + data_w   * data_quality
              + fresh_w  * freshness

        Returns 0.0 for banned or never-seen devices.
        """
        if self.status == DeviceStatus.BANNED:
            return 0.0

        if now is None:
            now = time.time()

        # -- Uptime percentage -----------------------------------------
        hours_since_reg = max((now - self.registered_at) / 3600, 1.0)
        uptime_pct = min(self.uptime_hours / hours_since_reg, 1.0)

        # -- Data quality (verified / total) ---------------------------
        if self.data_submitted > 0:
            data_quality = self.verified_submissions / self.data_submitted
        else:
            data_quality = 0.0

        # -- Freshness factor (decays from 1.0 to 0.0) ----------------
        if self.last_heartbeat > 0:
            minutes_since = (now - self.last_heartbeat) / 60
            freshness = max(1.0 - minutes_since / DEPIN_FRESHNESS_DECAY_MINS, 0.0)
        else:
            freshness = 0.0

        score = (
            DEPIN_UPTIME_WEIGHT * uptime_pct
            + DEPIN_DATA_WEIGHT * data_quality
            + DEPIN_FRESHNESS_WEIGHT * freshness
        )
        return round(min(max(score, 0.0), 1.0), 6)

    def compute_reward(self, now: Optional[float] = None) -> int:
        """Compute the reward for a single data submission.

        reward = round(tier_multiplier * score)   (integer, min 0)
        """
        score = self.compute_score(now)
        multiplier = _TIER_MULTIPLIER.get(int(self.device_type), 1)
        return round(multiplier * score)

    # ------------------------------------------------------------------
    # Data submission (replaces flat reward = 1)
    # ------------------------------------------------------------------

    def submit_data(self, data_hash: bytes, verified: bool = True,
                    now: Optional[float] = None) -> int:
        """Submit data and earn a scored, capped reward.

        Parameters
        ----------
        data_hash : bytes
            Hash of the submitted data payload.
        verified : bool
            Whether this submission passed quality checks.
        now : float, optional
            Override current time (for testing).

        Returns
        -------
        int
            Reward earned for this submission (may be 0 if capped or banned).
        """
        if self.status == DeviceStatus.BANNED:
            return 0

        if now is None:
            now = time.time()

        self.data_submitted += 1
        if verified:
            self.verified_submissions += 1

        # Compute reward using scoring formula
        reward = self.compute_reward(now)

        # Enforce daily cap
        today = int(now // 86400)
        if today != self._daily_reset_date:
            self._daily_rewards = 0
            self._daily_reset_date = today

        remaining_cap = max(DEPIN_DAILY_DEVICE_CAP - self._daily_rewards, 0)
        reward = min(reward, remaining_cap)

        self._daily_rewards += reward
        self.rewards_earned += reward
        self.pending_rewards += reward
        return reward

    # ------------------------------------------------------------------
    # Claim rewards
    # ------------------------------------------------------------------

    def claim_rewards(self) -> int:
        """Claim all pending rewards.  Returns claimed amount, resets pending."""
        claimed = self.pending_rewards
        self.pending_rewards = 0
        return claimed

    # ------------------------------------------------------------------
    # Active check
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        if self.status != DeviceStatus.ACTIVE:
            return False
        # Offline if no heartbeat in 10 minutes
        return time.time() - self.last_heartbeat < 600

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "device_id": self.device_id,
            "device_type": self.device_type.name,
            "owner": self.owner.hex(),
            "location": {"lat": self.location_lat, "lon": self.location_lon},
            "status": self.status.name,
            "uptime_hours": round(self.uptime_hours, 2),
            "data_submitted": self.data_submitted,
            "verified_submissions": self.verified_submissions,
            "rewards_earned": self.rewards_earned,
            "pending_rewards": self.pending_rewards,
            "last_heartbeat": self.last_heartbeat,
            "score": self.compute_score(),
        }


class DePINRegistry:
    """Registry for all DePIN devices."""

    def __init__(self):
        self._devices: Dict[str, DePINDevice] = {}
        self._total_rewards: int = 0

    def _generate_id(self, device_type: DeviceType, owner: bytes) -> str:
        data = f"{device_type.name}_{owner.hex()}_{time.time()}".encode()
        return "dev_" + hashlib.sha256(data).hexdigest()[:12]

    def register_device(self, owner: bytes, device_type: DeviceType,
                        lat: float = 0.0, lon: float = 0.0) -> DePINDevice:
        """Register a new physical device."""
        device_id = self._generate_id(device_type, owner)
        device = DePINDevice(
            device_id=device_id,
            device_type=device_type,
            owner=owner,
            location_lat=lat,
            location_lon=lon,
            registered_at=time.time(),
        )
        self._devices[device_id] = device
        return device

    def get_device(self, device_id: str) -> Optional[DePINDevice]:
        return self._devices.get(device_id)

    def get_devices_by_owner(self, owner: bytes) -> List[DePINDevice]:
        return [d for d in self._devices.values() if d.owner == owner]

    def record_heartbeat(self, device_id: str) -> bool:
        device = self.get_device(device_id)
        if device is None:
            return False
        device.heartbeat()
        return True

    def submit_data(self, device_id: str, data_hash: bytes,
                    verified: bool = True) -> int:
        """Submit data from device, returns reward amount."""
        device = self.get_device(device_id)
        if device is None:
            return 0
        reward = device.submit_data(data_hash, verified=verified)
        self._total_rewards += reward
        return reward

    def ban_device(self, device_id: str) -> bool:
        device = self.get_device(device_id)
        if device is None:
            return False
        device.status = DeviceStatus.BANNED
        return True

    # ------------------------------------------------------------------
    # Phase 22: Economic queries
    # ------------------------------------------------------------------

    def get_device_score(self, device_id: str) -> Optional[dict]:
        """Get detailed score breakdown for a device."""
        device = self.get_device(device_id)
        if device is None:
            return None
        now = time.time()
        hours_since_reg = max((now - device.registered_at) / 3600, 1.0)
        uptime_pct = min(device.uptime_hours / hours_since_reg, 1.0)
        data_quality = (device.verified_submissions / device.data_submitted
                        if device.data_submitted > 0 else 0.0)
        if device.last_heartbeat > 0:
            minutes_since = (now - device.last_heartbeat) / 60
            freshness = max(1.0 - minutes_since / DEPIN_FRESHNESS_DECAY_MINS, 0.0)
        else:
            freshness = 0.0
        multiplier = _TIER_MULTIPLIER.get(int(device.device_type), 1)
        return {
            "device_id": device_id,
            "score": device.compute_score(now),
            "uptime_pct": round(uptime_pct, 4),
            "data_quality": round(data_quality, 4),
            "freshness": round(freshness, 4),
            "tier_multiplier": multiplier,
            "reward_estimate": device.compute_reward(now),
        }

    def get_reward_estimate(self, device_id: str) -> int:
        """Estimate next reward for a device submission."""
        device = self.get_device(device_id)
        if device is None:
            return 0
        return device.compute_reward()

    def claim_rewards(self, device_id: str) -> int:
        """Claim pending rewards for a device."""
        device = self.get_device(device_id)
        if device is None:
            return 0
        return device.claim_rewards()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        type_counts = {}
        for dt in DeviceType:
            type_counts[dt.name] = sum(1 for d in self._devices.values() if d.device_type == dt)
        return {
            "total_devices": len(self._devices),
            "active_devices": sum(1 for d in self._devices.values() if d.status == DeviceStatus.ACTIVE),
            "total_rewards": self._total_rewards,
            "pending_rewards": sum(d.pending_rewards for d in self._devices.values()),
            "device_types": type_counts,
        }
