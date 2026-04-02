"""
Positronic - AI Agent Marketplace (Phase 29)
Task submission, assignment, result handling, and reward distribution.
"""

import time
import secrets
import hashlib
from enum import IntEnum
from dataclasses import dataclass
from typing import Dict, List, Optional

from positronic.constants import (
    BASE_UNIT,
    AGENT_TASK_FEE_MIN,
    AGENT_QUALITY_THRESHOLD,
    AGENT_REWARD_SHARE,
    AGENT_PLATFORM_FEE,
    AGENT_BURN_FEE,
    AGENT_TASK_TIMEOUT,
)
from positronic.agent.registry import AgentRegistry, AgentInfo, AgentStatus
from positronic.agent.scoring import score_agent_output


class TaskStatus(IntEnum):
    """Lifecycle of a marketplace task."""
    SUBMITTED = 0       # Task submitted, fee paid
    ASSIGNED = 1        # Agent picked up the task
    PROCESSING = 2      # Agent is working on it
    COMPLETED = 3       # Agent returned result, scored
    FAILED = 4          # Agent failed or timed out
    CANCELLED = 5       # Requester cancelled before assignment
    DISPUTED = 6        # Result quality disputed


@dataclass
class TaskInfo:
    """A task submitted to an AI agent."""
    task_id: str
    agent_id: str
    requester: bytes        # Requester's blockchain address
    input_data: str         # Task input (JSON string)
    fee_paid: int           # ASF paid by requester
    status: TaskStatus = TaskStatus.SUBMITTED

    # Result
    result_data: str = ""   # Agent's output (JSON string)
    result_hash: str = ""   # SHA-512 hash of result

    # Quality scoring
    ai_quality_score: int = 0   # PoNC quality score (basis points 0-10000)

    # Reward distribution
    agent_reward: int = 0       # 85% to agent developer
    platform_fee: int = 0       # 10% to treasury
    burn_amount: int = 0        # 5% burned

    # Timestamps
    submitted_at: float = 0.0
    assigned_at: float = 0.0
    completed_at: float = 0.0
    timeout_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "requester": self.requester.hex(),
            "input_data": self.input_data,
            "fee_paid": self.fee_paid,
            "status": TaskStatus(self.status).name,
            "result_data": self.result_data,
            "result_hash": self.result_hash,
            "ai_quality_score": self.ai_quality_score,
            "agent_reward": self.agent_reward,
            "platform_fee": self.platform_fee,
            "burn_amount": self.burn_amount,
            "submitted_at": self.submitted_at,
            "completed_at": self.completed_at,
        }


class AgentMarketplace:
    """
    Manages the lifecycle of tasks in the AI Agent Marketplace.

    Flow: Submit → Assign → Process → Complete (score + reward)
    """

    def __init__(self, registry: AgentRegistry):
        self._registry = registry
        self._tasks: Dict[str, TaskInfo] = {}
        self._agent_tasks: Dict[str, List[str]] = {}  # agent_id → [task_ids]
        self._task_counter: int = 0

    # ---- Task Submission ----

    def submit_task(
        self,
        requester: bytes,
        agent_id: str,
        input_data: str,
        fee: int,
    ) -> Optional[TaskInfo]:
        """Submit a task to a specific agent. Returns TaskInfo or None.

        The caller must verify:
        1. Agent exists and is ACTIVE
        2. Fee >= agent's task_fee (and >= AGENT_TASK_FEE_MIN)
        3. Requester has sufficient balance
        """
        agent = self._registry.get_agent(agent_id)
        if not agent or agent.status != AgentStatus.ACTIVE:
            return None

        if fee < max(agent.task_fee, AGENT_TASK_FEE_MIN):
            return None

        # Quality gate: agent must maintain minimum quality
        if agent.quality_score < AGENT_QUALITY_THRESHOLD:
            return None

        self._task_counter += 1
        task_id = f"TASK-{int(time.time())}-{self._task_counter:04d}"
        now = time.time()

        task = TaskInfo(
            task_id=task_id,
            agent_id=agent_id,
            requester=requester,
            input_data=input_data,
            fee_paid=fee,
            status=TaskStatus.SUBMITTED,
            submitted_at=now,
            timeout_at=now + AGENT_TASK_TIMEOUT,
        )

        self._tasks[task_id] = task
        self._agent_tasks.setdefault(agent_id, []).append(task_id)

        # Auto-assign to the specified agent
        task.status = TaskStatus.ASSIGNED
        task.assigned_at = now

        return task

    # ---- Task Processing ----

    def start_processing(self, task_id: str) -> Optional[TaskInfo]:
        """Mark a task as being processed."""
        task = self._tasks.get(task_id)
        if not task or task.status != TaskStatus.ASSIGNED:
            return None
        task.status = TaskStatus.PROCESSING
        return task

    def execute_task(
        self,
        task_id: str,
        result_data: str,
        ai_gate=None,
    ) -> Optional[TaskInfo]:
        """Execute a task end-to-end: process → score → complete → reward.

        This is the high-level entry point that simulates on-chain agent
        execution.  It scores the output using PoNC (when available) or
        heuristic scoring, then distributes rewards.

        Args:
            task_id: The task to execute.
            result_data: The agent's output for this task.
            ai_gate: Optional AIValidationGate for PoNC-assisted scoring.

        Returns:
            Completed TaskInfo with rewards calculated, or None on error.
        """
        task = self._tasks.get(task_id)
        if not task or task.status not in (TaskStatus.ASSIGNED, TaskStatus.PROCESSING):
            return None

        # Mark as processing
        task.status = TaskStatus.PROCESSING

        # Score the output using PoNC or heuristics
        agent = self._registry.get_agent(task.agent_id)
        category = int(agent.category) if agent else 0
        quality_score = score_agent_output(
            result_data=result_data,
            input_data=task.input_data,
            agent_category=category,
            ai_gate=ai_gate,
        )

        # Complete with the computed score
        return self.complete_task(task_id, result_data, quality_score)

    def complete_task(
        self,
        task_id: str,
        result_data: str,
        ai_quality_score: int,
    ) -> Optional[TaskInfo]:
        """Complete a task with result and quality score.

        Returns the task with reward distribution calculated.
        The caller is responsible for actually transferring the rewards.
        """
        task = self._tasks.get(task_id)
        if not task or task.status not in (TaskStatus.ASSIGNED, TaskStatus.PROCESSING):
            return None

        now = time.time()

        # Check timeout
        if now > task.timeout_at:
            task.status = TaskStatus.FAILED
            return task

        # Record result
        task.result_data = result_data
        task.result_hash = hashlib.sha512(result_data.encode()).hexdigest()
        task.ai_quality_score = max(0, min(10000, ai_quality_score))
        task.completed_at = now
        task.status = TaskStatus.COMPLETED

        # Calculate reward distribution
        task.agent_reward = int(task.fee_paid * AGENT_REWARD_SHARE)
        task.platform_fee = int(task.fee_paid * AGENT_PLATFORM_FEE)
        task.burn_amount = task.fee_paid - task.agent_reward - task.platform_fee

        # Update agent stats
        agent = self._registry.get_agent(task.agent_id)
        if agent:
            agent.tasks_completed += 1
            agent.total_earned += task.agent_reward
            agent.last_task_at = now

            # Update quality score with exponential moving average
            alpha = 0.1  # Smoothing factor
            agent.quality_score = int(
                agent.quality_score * (1 - alpha) + ai_quality_score * alpha
            )

        return task

    def fail_task(self, task_id: str, reason: str = "") -> Optional[TaskInfo]:
        """Mark a task as failed."""
        task = self._tasks.get(task_id)
        if not task or task.status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED):
            return None

        task.status = TaskStatus.FAILED
        task.result_data = reason
        task.completed_at = time.time()

        # Update agent stats
        agent = self._registry.get_agent(task.agent_id)
        if agent:
            agent.tasks_failed += 1

        return task

    def cancel_task(self, task_id: str, requester: bytes) -> Optional[TaskInfo]:
        """Cancel a task (only by requester, only before processing)."""
        task = self._tasks.get(task_id)
        if not task:
            return None
        if task.requester != requester:
            return None
        if task.status not in (TaskStatus.SUBMITTED, TaskStatus.ASSIGNED):
            return None

        task.status = TaskStatus.CANCELLED
        task.completed_at = time.time()
        return task

    # ---- Rating ----

    def rate_agent(
        self,
        agent_id: str,
        rater: bytes,
        score: int,
    ) -> bool:
        """Rate an agent (1-5 stars, stored as 1-5).

        Only callers who have completed tasks with this agent should rate.
        """
        agent = self._registry.get_agent(agent_id)
        if not agent:
            return False

        score = max(1, min(5, score))
        agent.total_ratings += 1
        agent.rating_sum += score
        return True

    # ---- Timeout Management ----

    def check_timeouts(self) -> List[str]:
        """Check for timed-out tasks and mark them as failed."""
        now = time.time()
        timed_out = []
        for task in self._tasks.values():
            if task.status in (TaskStatus.ASSIGNED, TaskStatus.PROCESSING):
                if now > task.timeout_at:
                    task.status = TaskStatus.FAILED
                    task.result_data = "Task timed out"
                    task.completed_at = now
                    timed_out.append(task.task_id)
        return timed_out

    # ---- Queries ----

    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        return self._tasks.get(task_id)

    def get_agent_tasks(self, agent_id: str, limit: int = 20) -> List[TaskInfo]:
        task_ids = self._agent_tasks.get(agent_id, [])
        tasks = [self._tasks[tid] for tid in task_ids if tid in self._tasks]
        return sorted(tasks, key=lambda t: t.submitted_at, reverse=True)[:limit]

    def get_requester_tasks(self, requester: bytes, limit: int = 20) -> List[TaskInfo]:
        tasks = [t for t in self._tasks.values() if t.requester == requester]
        return sorted(tasks, key=lambda t: t.submitted_at, reverse=True)[:limit]

    def get_stats(self) -> dict:
        tasks = list(self._tasks.values())
        completed = [t for t in tasks if t.status == TaskStatus.COMPLETED]
        return {
            "total_tasks": len(tasks),
            "completed_tasks": len(completed),
            "failed_tasks": len([t for t in tasks if t.status == TaskStatus.FAILED]),
            "pending_tasks": len([
                t for t in tasks
                if t.status in (TaskStatus.SUBMITTED, TaskStatus.ASSIGNED, TaskStatus.PROCESSING)
            ]),
            "total_fees_collected": sum(t.fee_paid for t in completed),
            "total_rewards_distributed": sum(t.agent_reward for t in completed),
            "total_burned": sum(t.burn_amount for t in completed),
            "total_platform_fees": sum(t.platform_fee for t in completed),
            "average_quality_score": (
                sum(t.ai_quality_score for t in completed) / len(completed)
                if completed else 0
            ),
        }
