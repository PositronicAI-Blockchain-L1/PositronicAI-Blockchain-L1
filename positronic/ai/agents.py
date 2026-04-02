"""
Positronic - On-Chain AI Agents
Autonomous agents with on-chain identity and permission system.

Phase 23: Enforcement + Autonomous Execution
----------------------------------------------
- Permission enforcement: action_type must be in agent.permissions
  (empty list = unrestricted for backward compat)
- Rate limiting: max N actions per hour (sliding window)
- Spending limits: per-action and daily caps
- Autonomous tick(): each block triggers type-specific actions
- Kill switch: BANNED is permanent (resume() is no-op)
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from enum import IntEnum
import time
import hashlib

from positronic.constants import (
    AGENT_DEFAULT_RATE_LIMIT,
    AGENT_MAX_SPEND_DEFAULT,
    AGENT_DAILY_LIMIT_DEFAULT,
    AGENT_MIN_TRUST_FOR_AUTONOMOUS,
)


class AgentType(IntEnum):
    VALIDATOR = 0   # Validates transactions
    TRADING = 1     # Automated trading
    GAME = 2        # Game bot
    ANALYTICS = 3   # Data analysis
    GOVERNANCE = 4  # DAO voting agent


class AgentStatus(IntEnum):
    ACTIVE = 0
    PAUSED = 1
    BANNED = 2


@dataclass
class AgentAction:
    """Record of an agent action."""
    action_type: str
    timestamp: float
    data: dict = field(default_factory=dict)
    success: bool = True
    spend: int = 0

    def to_dict(self) -> dict:
        return {
            "action_type": self.action_type,
            "timestamp": self.timestamp,
            "data": self.data,
            "success": self.success,
            "spend": self.spend,
        }


@dataclass
class AIAgent:
    """On-chain AI agent with identity."""
    agent_id: str
    owner: bytes
    agent_type: AgentType
    name: str = ""
    permissions: List[str] = field(default_factory=list)
    status: AgentStatus = AgentStatus.ACTIVE
    trust_score: int = 0
    actions_executed: int = 0
    actions_failed: int = 0
    created_at: float = 0.0
    last_active: float = 0.0
    _history: List[AgentAction] = field(default_factory=list)

    # Phase 23: Spending limits
    max_spend_per_action: int = AGENT_MAX_SPEND_DEFAULT
    daily_spend_limit: int = AGENT_DAILY_LIMIT_DEFAULT
    _daily_spend: int = 0
    _daily_reset_date: int = 0  # day number

    # Phase 23: Rate limiting
    rate_limit: int = AGENT_DEFAULT_RATE_LIMIT  # max per hour
    _action_timestamps: deque = field(default_factory=lambda: deque(maxlen=1000))

    # ------------------------------------------------------------------
    # Permission + limit checks
    # ------------------------------------------------------------------

    def _check_permission(self, action_type: str) -> bool:
        """Check if action_type is allowed.

        Empty permissions list = unrestricted (backward compat).
        Non-empty list = action_type must be present.
        """
        if not self.permissions:
            return True
        return action_type in self.permissions

    def _check_rate_limit(self, now: Optional[float] = None) -> bool:
        """Check if agent is under its hourly rate limit."""
        if now is None:
            now = time.time()
        cutoff = now - 3600  # 1 hour window
        # Count actions in the last hour
        recent = sum(1 for ts in self._action_timestamps if ts > cutoff)
        return recent < self.rate_limit

    def _check_spend(self, amount: int, now: Optional[float] = None) -> bool:
        """Check if spend is within per-action and daily limits."""
        if amount > self.max_spend_per_action:
            return False
        if now is None:
            now = time.time()
        today = int(now // 86400)
        if today != self._daily_reset_date:
            self._daily_spend = 0
            self._daily_reset_date = today
        return (self._daily_spend + amount) <= self.daily_spend_limit

    # ------------------------------------------------------------------
    # Execute action (Phase 23: with enforcement)
    # ------------------------------------------------------------------

    def execute_action(self, action_type: str, data: dict = None,
                       spend: int = 0, now: Optional[float] = None) -> bool:
        """Execute an action with permission, rate, and spend enforcement.

        Parameters
        ----------
        action_type : str
            The type of action (e.g., "transfer", "vote", "validate").
        data : dict, optional
            Action-specific data payload.
        spend : int, optional
            Amount of ASF this action spends (default 0).
        now : float, optional
            Override current time (for testing).

        Returns
        -------
        bool
            True if action was executed, False if blocked.
        """
        if self.status != AgentStatus.ACTIVE:
            return False

        if now is None:
            now = time.time()

        # Phase 23: Permission check
        if not self._check_permission(action_type):
            self.record_failure(action_type, {"reason": "permission_denied"})
            return False

        # Phase 23: Rate limit check
        if not self._check_rate_limit(now):
            self.record_failure(action_type, {"reason": "rate_limited"})
            return False

        # Phase 23: Spending limit check
        if spend > 0 and not self._check_spend(spend, now):
            self.record_failure(action_type, {"reason": "spend_limit"})
            return False

        # Execute
        action = AgentAction(
            action_type=action_type,
            timestamp=now,
            data=data or {},
            success=True,
            spend=spend,
        )
        self._history.append(action)
        self._action_timestamps.append(now)
        self.actions_executed += 1
        self.last_active = now
        self.trust_score += 1

        # Track spending
        if spend > 0:
            today = int(now // 86400)
            if today != self._daily_reset_date:
                self._daily_spend = 0
                self._daily_reset_date = today
            self._daily_spend += spend

        return True

    def record_failure(self, action_type: str, data: dict = None):
        """Record a failed action."""
        action = AgentAction(
            action_type=action_type,
            timestamp=time.time(),
            data=data or {},
            success=False,
        )
        self._history.append(action)
        self.actions_failed += 1
        self.trust_score = max(0, self.trust_score - 5)

    def get_history(self, limit: int = 50) -> List[dict]:
        """Get recent action history."""
        return [a.to_dict() for a in self._history[-limit:]]

    # ------------------------------------------------------------------
    # Status management
    # ------------------------------------------------------------------

    def pause(self):
        """Pause agent (can be resumed by owner)."""
        if self.status == AgentStatus.ACTIVE:
            self.status = AgentStatus.PAUSED

    def resume(self):
        """Resume paused agent. BANNED agents CANNOT be resumed."""
        if self.status == AgentStatus.PAUSED:
            self.status = AgentStatus.ACTIVE

    def ban(self):
        """Permanently ban agent. Cannot be reversed."""
        self.status = AgentStatus.BANNED

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def success_rate(self) -> float:
        total = self.actions_executed + self.actions_failed
        if total == 0:
            return 0.0
        return self.actions_executed / total

    @property
    def can_act_autonomously(self) -> bool:
        """Agent needs minimum trust to act autonomously."""
        return (
            self.status == AgentStatus.ACTIVE
            and self.trust_score >= AGENT_MIN_TRUST_FOR_AUTONOMOUS
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "owner": self.owner.hex(),
            "agent_type": self.agent_type.name,
            "name": self.name,
            "permissions": self.permissions,
            "status": self.status.name,
            "trust_score": self.trust_score,
            "actions_executed": self.actions_executed,
            "actions_failed": self.actions_failed,
            "success_rate": round(self.success_rate, 4),
            "created_at": self.created_at,
            "last_active": self.last_active,
            "max_spend_per_action": self.max_spend_per_action,
            "daily_spend_limit": self.daily_spend_limit,
            "rate_limit": self.rate_limit,
        }


class AgentRegistry:
    """Registry for all on-chain AI agents."""

    def __init__(self):
        self._agents: Dict[str, AIAgent] = {}
        self._total_actions: int = 0

    def _generate_id(self, name: str, owner: bytes) -> str:
        data = name.encode() + owner + str(time.time()).encode()
        return "agent_" + hashlib.sha256(data).hexdigest()[:12]

    def register(self, owner: bytes, agent_type: AgentType,
                 name: str = "", permissions: list = None) -> AIAgent:
        """Register a new AI agent."""
        agent_id = self._generate_id(name or "unnamed", owner)
        agent = AIAgent(
            agent_id=agent_id,
            owner=owner,
            agent_type=agent_type,
            name=name,
            permissions=permissions or [],
            created_at=time.time(),
        )
        self._agents[agent_id] = agent
        return agent

    def get(self, agent_id: str) -> Optional[AIAgent]:
        return self._agents.get(agent_id)

    def get_by_owner(self, owner: bytes) -> List[AIAgent]:
        return [a for a in self._agents.values() if a.owner == owner]

    def execute(self, agent_id: str, action_type: str,
                data: dict = None, spend: int = 0) -> bool:
        """Execute an action for an agent with full enforcement."""
        agent = self.get(agent_id)
        if agent is None:
            return False
        result = agent.execute_action(action_type, data, spend=spend)
        if result:
            self._total_actions += 1
        return result

    # ------------------------------------------------------------------
    # Phase 23: Agent limits configuration
    # ------------------------------------------------------------------

    def set_limits(self, agent_id: str, max_spend: int = None,
                   daily_limit: int = None, rate_limit: int = None) -> bool:
        """Set spending / rate limits for an agent."""
        agent = self.get(agent_id)
        if agent is None:
            return False
        if max_spend is not None:
            agent.max_spend_per_action = max_spend
        if daily_limit is not None:
            agent.daily_spend_limit = daily_limit
        if rate_limit is not None:
            agent.rate_limit = rate_limit
        return True

    # ------------------------------------------------------------------
    # Phase 23: Autonomous tick
    # ------------------------------------------------------------------

    def tick(self, block_number: int = 0) -> List[dict]:
        """Called each block — trigger autonomous actions for eligible agents.

        Returns list of action records for monitoring.
        """
        actions = []
        now = time.time()
        for agent in self._agents.values():
            if not agent.can_act_autonomously:
                continue
            if not agent._check_rate_limit(now):
                continue

            action_type = None
            action_data = {"block": block_number, "autonomous": True}

            if agent.agent_type == AgentType.VALIDATOR:
                action_type = "validate"
            elif agent.agent_type == AgentType.GOVERNANCE:
                action_type = "vote"
            elif agent.agent_type == AgentType.ANALYTICS:
                action_type = "analyze"
            elif agent.agent_type == AgentType.TRADING:
                action_type = "trade"
            elif agent.agent_type == AgentType.GAME:
                action_type = "game_move"

            if action_type:
                ok = agent.execute_action(action_type, action_data, now=now)
                actions.append({
                    "agent_id": agent.agent_id,
                    "action": action_type,
                    "success": ok,
                })

        return actions

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        return {
            "total_agents": len(self._agents),
            "active_agents": sum(1 for a in self._agents.values() if a.status == AgentStatus.ACTIVE),
            "total_actions": self._total_actions,
            "autonomous_eligible": sum(1 for a in self._agents.values() if a.can_act_autonomously),
            "agent_types": {
                t.name: sum(1 for a in self._agents.values() if a.agent_type == t)
                for t in AgentType
            },
        }
