"""
Positronic - AI Agent Registry (Phase 29)
Register, approve, and manage on-chain AI agents.
Mirrors the Game Registry pattern: fee → AI review → council vote → activate.
"""

import json
import sqlite3
import time
import hashlib
import secrets
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from positronic.constants import (
    BASE_UNIT,
    AGENT_REGISTRATION_FEE,
    AGENT_CATEGORIES,
    AGENT_AI_MAX_RISK,
    AGENT_COUNCIL_MIN_VOTES,
    AGENT_COUNCIL_APPROVAL_PCT,
    AGENT_MIN_TRUST,
    MAX_AGENTS,
)


class AgentCategory(IntEnum):
    """Category of AI agent capability."""
    ANALYSIS = 0      # Market analysis, portfolio optimization, trend prediction
    AUDIT = 1         # Smart contract audit, code review, vulnerability scanning
    GOVERNANCE = 2    # Proposal analysis, voting recommendations, parameter tuning
    CREATIVE = 3      # Content generation, NFT art, game asset creation
    DATA = 4          # Data indexing, analytics, on-chain forensics
    SECURITY = 5      # Threat detection, anomaly analysis, incident response


class AgentStatus(IntEnum):
    """Lifecycle status of a registered agent."""
    PENDING = 0         # Awaiting AI review
    AI_REVIEWING = 1    # AI risk assessment in progress
    AI_APPROVED = 2     # Passed AI review, awaiting council vote
    AI_REJECTED = 3     # Failed AI review
    COUNCIL_VOTING = 4  # Council members voting
    APPROVED = 5        # Approved but not yet active
    ACTIVE = 6          # Live and accepting tasks
    SUSPENDED = 7       # Temporarily disabled
    REVOKED = 8         # Permanently removed


@dataclass
class AgentInfo:
    """Complete information about a registered AI agent."""
    agent_id: str
    name: str
    owner: bytes                    # Owner's blockchain address
    category: AgentCategory
    status: AgentStatus = AgentStatus.PENDING
    description: str = ""
    endpoint_url: str = ""          # Where to reach the agent (optional, off-chain)
    model_hash: str = ""            # SHA-512 hash of the agent's model/code

    # Pricing
    task_fee: int = 1 * BASE_UNIT   # Fee per task (set by developer, min 1 ASF)

    # Authentication
    api_key_hash: str = ""          # SHA-512 hash of the API key

    # Quality & reputation
    quality_score: int = 7500       # Basis points (0-10000), starts above threshold
    trust_score: int = 100          # Starts at 100, adjusted by behavior
    ai_risk_score: float = 0.0      # AI assessment (0=safe, 1=risky)

    # Statistics
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_earned: int = 0
    total_ratings: int = 0
    rating_sum: int = 0             # Sum of all ratings (for average calculation)

    # Governance
    votes_for: int = 0
    votes_against: int = 0
    voters: List[str] = field(default_factory=list)

    # Timestamps
    registered_at: float = 0.0
    approved_at: float = 0.0
    last_task_at: float = 0.0

    @property
    def average_rating(self) -> float:
        if self.total_ratings == 0:
            return 0.0
        return self.rating_sum / self.total_ratings

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "owner": self.owner.hex(),
            "category": AgentCategory(self.category).name,
            "status": AgentStatus(self.status).name,
            "description": self.description,
            "endpoint_url": self.endpoint_url,
            "model_hash": self.model_hash,
            "task_fee": self.task_fee,
            "quality_score": self.quality_score,
            "trust_score": self.trust_score,
            "ai_risk_score": self.ai_risk_score,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "total_earned": self.total_earned,
            "average_rating": round(self.average_rating, 2),
            "total_ratings": self.total_ratings,
            "registered_at": self.registered_at,
            "approved_at": self.approved_at,
            "last_task_at": self.last_task_at,
        }


class AgentRegistry:
    """
    Manages registration and lifecycle of on-chain AI agents.

    Flow: Register → AI Review → Council Vote → Activate → Monitor

    If ``db_path`` is provided, state is persisted to SQLite and restored
    on init.  Without ``db_path`` the registry is purely in-memory.
    """

    def __init__(self, db_path: Optional[str] = None):
        self._agents: Dict[str, AgentInfo] = {}
        self._api_keys: Dict[str, str] = {}   # api_key_hash → agent_id
        self._council_members: set = set()
        self._agent_counter: int = 0
        self._db_path = db_path
        self._db: Optional[sqlite3.Connection] = None

        if db_path:
            self._init_db()
            self._load_from_db()

    # ---- SQLite persistence ----

    def _init_db(self):
        self._db = sqlite3.connect(self._db_path)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS agent_registry (
                agent_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                owner_hex TEXT NOT NULL,
                category INTEGER NOT NULL,
                status INTEGER NOT NULL,
                description TEXT DEFAULT '',
                endpoint_url TEXT DEFAULT '',
                model_hash TEXT DEFAULT '',
                task_fee INTEGER DEFAULT 0,
                api_key_hash TEXT DEFAULT '',
                quality_score INTEGER DEFAULT 7500,
                trust_score INTEGER DEFAULT 100,
                ai_risk_score REAL DEFAULT 0.0,
                tasks_completed INTEGER DEFAULT 0,
                tasks_failed INTEGER DEFAULT 0,
                total_earned INTEGER DEFAULT 0,
                total_ratings INTEGER DEFAULT 0,
                rating_sum INTEGER DEFAULT 0,
                votes_for INTEGER DEFAULT 0,
                votes_against INTEGER DEFAULT 0,
                voters_json TEXT DEFAULT '[]',
                registered_at REAL DEFAULT 0.0,
                approved_at REAL DEFAULT 0.0,
                last_task_at REAL DEFAULT 0.0
            )
        """)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS agent_registry_council (
                address_hex TEXT PRIMARY KEY
            )
        """)
        self._db.commit()

    def _load_from_db(self):
        if not self._db:
            return
        cur = self._db.execute("SELECT * FROM agent_registry")
        cols = [d[0] for d in cur.description]
        for row in cur.fetchall():
            r = dict(zip(cols, row))
            agent = AgentInfo(
                agent_id=r["agent_id"],
                name=r["name"],
                owner=bytes.fromhex(r["owner_hex"]),
                category=AgentCategory(r["category"]),
                status=AgentStatus(r["status"]),
                description=r["description"],
                endpoint_url=r["endpoint_url"],
                model_hash=r["model_hash"],
                task_fee=r["task_fee"],
                api_key_hash=r["api_key_hash"],
                quality_score=r["quality_score"],
                trust_score=r["trust_score"],
                ai_risk_score=r["ai_risk_score"],
                tasks_completed=r["tasks_completed"],
                tasks_failed=r["tasks_failed"],
                total_earned=r["total_earned"],
                total_ratings=r["total_ratings"],
                rating_sum=r["rating_sum"],
                votes_for=r["votes_for"],
                votes_against=r["votes_against"],
                voters=json.loads(r["voters_json"]),
                registered_at=r["registered_at"],
                approved_at=r["approved_at"],
                last_task_at=r["last_task_at"],
            )
            self._agents[agent.agent_id] = agent
            if agent.api_key_hash:
                self._api_keys[agent.api_key_hash] = agent.agent_id
        if self._agents:
            self._agent_counter = len(self._agents)
        for row in self._db.execute("SELECT address_hex FROM agent_registry_council"):
            self._council_members.add(bytes.fromhex(row[0]))

    def _persist_agent(self, agent: AgentInfo):
        if not self._db:
            return
        self._db.execute("""
            INSERT OR REPLACE INTO agent_registry (
                agent_id, name, owner_hex, category, status, description,
                endpoint_url, model_hash, task_fee, api_key_hash,
                quality_score, trust_score, ai_risk_score,
                tasks_completed, tasks_failed, total_earned,
                total_ratings, rating_sum,
                votes_for, votes_against, voters_json,
                registered_at, approved_at, last_task_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            agent.agent_id, agent.name, agent.owner.hex(),
            int(agent.category), int(agent.status), agent.description,
            agent.endpoint_url, agent.model_hash, agent.task_fee,
            agent.api_key_hash,
            agent.quality_score, agent.trust_score, agent.ai_risk_score,
            agent.tasks_completed, agent.tasks_failed, agent.total_earned,
            agent.total_ratings, agent.rating_sum,
            agent.votes_for, agent.votes_against, json.dumps(agent.voters),
            agent.registered_at, agent.approved_at, agent.last_task_at,
        ))
        self._db.commit()

    def _persist_council(self):
        if not self._db:
            return
        self._db.execute("DELETE FROM agent_registry_council")
        for addr in self._council_members:
            self._db.execute(
                "INSERT INTO agent_registry_council (address_hex) VALUES (?)",
                (addr.hex(),),
            )
        self._db.commit()

    def close(self):
        if self._db:
            self._db.close()
            self._db = None

    # ---- Council management ----

    def add_council_member(self, address: bytes):
        self._council_members.add(address)
        self._persist_council()

    def remove_council_member(self, address: bytes):
        self._council_members.discard(address)
        self._persist_council()

    # ---- Registration ----

    def register_agent(
        self,
        owner: bytes,
        name: str,
        category: AgentCategory,
        description: str = "",
        task_fee: int = 1 * BASE_UNIT,
        endpoint_url: str = "",
        model_hash: str = "",
    ) -> tuple:
        """Register a new AI agent. Returns (AgentInfo, api_key, registration_fee).

        The caller (RPC handler / executor) is responsible for deducting
        AGENT_REGISTRATION_FEE from the owner's account.
        """
        if len(self._agents) >= MAX_AGENTS:
            raise ValueError("Agent registry full")

        # Validate category
        if not isinstance(category, AgentCategory):
            try:
                category = AgentCategory(category)
            except ValueError:
                raise ValueError(f"Invalid category: {category}")

        # Enforce minimum task fee
        task_fee = max(task_fee, 1 * BASE_UNIT)

        self._agent_counter += 1
        agent_id = f"AGENT-{int(time.time())}-{self._agent_counter:04d}"

        # Generate API key
        api_key = f"ak_{secrets.token_hex(32)}"
        api_key_hash = hashlib.sha512(api_key.encode()).hexdigest()

        agent = AgentInfo(
            agent_id=agent_id,
            name=name,
            owner=owner,
            category=category,
            description=description,
            status=AgentStatus.PENDING,
            task_fee=task_fee,
            endpoint_url=endpoint_url,
            model_hash=model_hash,
            api_key_hash=api_key_hash,
            registered_at=time.time(),
        )

        self._agents[agent_id] = agent
        self._api_keys[api_key_hash] = agent_id
        self._persist_agent(agent)

        return agent, api_key, AGENT_REGISTRATION_FEE

    # ---- AI Review ----

    def ai_review(self, agent_id: str, risk_score: float) -> Optional[AgentInfo]:
        """AI reviews an agent registration."""
        agent = self._agents.get(agent_id)
        if not agent or agent.status != AgentStatus.PENDING:
            return None

        agent.ai_risk_score = risk_score
        agent.status = AgentStatus.AI_REVIEWING

        if risk_score <= AGENT_AI_MAX_RISK:
            agent.status = AgentStatus.AI_APPROVED
        else:
            agent.status = AgentStatus.AI_REJECTED

        self._persist_agent(agent)
        return agent

    # ---- Council Vote ----

    def council_vote(self, agent_id: str, voter: bytes, approve: bool) -> Optional[AgentInfo]:
        """Council member votes on an agent."""
        agent = self._agents.get(agent_id)
        if not agent:
            return None
        if agent.status not in (AgentStatus.AI_APPROVED, AgentStatus.COUNCIL_VOTING):
            return None
        if voter not in self._council_members:
            return None

        voter_hex = voter.hex()
        if voter_hex in agent.voters:
            return None  # Already voted

        agent.status = AgentStatus.COUNCIL_VOTING
        agent.voters.append(voter_hex)

        if approve:
            agent.votes_for += 1
        else:
            agent.votes_against += 1

        total = agent.votes_for + agent.votes_against
        if total >= AGENT_COUNCIL_MIN_VOTES:
            ratio = agent.votes_for / total
            if ratio >= AGENT_COUNCIL_APPROVAL_PCT:
                agent.status = AgentStatus.APPROVED
                agent.approved_at = time.time()
            elif total >= len(self._council_members):
                agent.status = AgentStatus.AI_REJECTED

        self._persist_agent(agent)
        return agent

    def activate_agent(self, agent_id: str) -> Optional[AgentInfo]:
        """Activate an approved agent."""
        agent = self._agents.get(agent_id)
        if not agent or agent.status != AgentStatus.APPROVED:
            return None
        agent.status = AgentStatus.ACTIVE
        self._persist_agent(agent)
        return agent

    # ---- Trust Management ----

    def adjust_trust(self, agent_id: str, delta: int):
        """Adjust an agent's trust score."""
        agent = self._agents.get(agent_id)
        if not agent:
            return
        agent.trust_score = max(0, agent.trust_score + delta)
        if agent.trust_score < AGENT_MIN_TRUST and agent.status == AgentStatus.ACTIVE:
            agent.status = AgentStatus.SUSPENDED
        self._persist_agent(agent)

    def update_quality_score(self, agent_id: str, new_score: int):
        """Update agent quality score (basis points 0-10000)."""
        agent = self._agents.get(agent_id)
        if not agent:
            return
        agent.quality_score = max(0, min(10000, new_score))
        self._persist_agent(agent)

    # ---- Lifecycle ----

    def suspend_agent(self, agent_id: str) -> Optional[AgentInfo]:
        agent = self._agents.get(agent_id)
        if agent and agent.status == AgentStatus.ACTIVE:
            agent.status = AgentStatus.SUSPENDED
            self._persist_agent(agent)
        return agent

    def revoke_agent(self, agent_id: str) -> Optional[AgentInfo]:
        agent = self._agents.get(agent_id)
        if agent:
            agent.status = AgentStatus.REVOKED
            self._api_keys.pop(agent.api_key_hash, None)
            self._persist_agent(agent)
        return agent

    # ---- Queries ----

    def get_agent(self, agent_id: str) -> Optional[AgentInfo]:
        return self._agents.get(agent_id)

    def list_active_agents(self) -> List[AgentInfo]:
        return [a for a in self._agents.values() if a.status == AgentStatus.ACTIVE]

    def list_agents_by_category(self, category: AgentCategory) -> List[AgentInfo]:
        return [
            a for a in self._agents.values()
            if a.category == category and a.status == AgentStatus.ACTIVE
        ]

    def list_all_agents(self) -> List[AgentInfo]:
        return list(self._agents.values())

    def get_leaderboard(self, limit: int = 20) -> List[AgentInfo]:
        """Get top agents by quality score."""
        active = self.list_active_agents()
        return sorted(active, key=lambda a: a.quality_score, reverse=True)[:limit]

    def get_stats(self) -> dict:
        agents = list(self._agents.values())
        active = [a for a in agents if a.status == AgentStatus.ACTIVE]
        return {
            "total_registered": len(agents),
            "active_agents": len(active),
            "total_tasks_completed": sum(a.tasks_completed for a in agents),
            "total_earned_all_agents": sum(a.total_earned for a in agents),
            "council_members": len(self._council_members),
            "categories": {
                cat.name: len([a for a in active if a.category == cat])
                for cat in AgentCategory
            },
        }
