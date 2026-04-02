"""
Positronic - Smart Contract Risk Analyzer (SCRA)
Analyzes contract interactions for security risks.
Detects reentrancy, overflow, unauthorized delegatecalls, and other patterns.
"""

from typing import List, Dict, Set, Optional
from dataclasses import dataclass, field
from enum import IntEnum

from positronic.utils.logging import get_logger
from positronic.core.transaction import Transaction, TxType

logger = get_logger(__name__)
from positronic.constants import (
    SCRA_DELEGATECALL_RISK,
    SCRA_SSTORE_AFTER_CALL_RISK,
    SCRA_SELFDESTRUCT_RISK,
    SCRA_PROXY_PATTERN_RISK,
)


class RiskType(IntEnum):
    """Types of contract risks detected."""
    NONE = 0
    REENTRANCY = 1          # Potential reentrancy attack
    OVERFLOW = 2            # Integer overflow risk
    UNAUTHORIZED_CALL = 3   # Unauthorized delegatecall
    SELF_DESTRUCT = 4       # Self-destruct call
    UNKNOWN_CONTRACT = 5    # Interaction with unverified contract
    HIGH_GAS_CALL = 6       # Suspiciously high gas for simple operation
    RECURSIVE_CALL = 7      # Deeply recursive contract calls
    LARGE_DEPLOY = 8        # Unusually large contract deployment


@dataclass
class ContractRiskReport:
    """Risk analysis report for a contract interaction."""
    risk_type: RiskType = RiskType.NONE
    risk_score: float = 0.0
    description: str = ""
    details: dict = field(default_factory=dict)


class BytecodeAnalyzer:
    """
    Phase 15: Deep bytecode analysis with opcode-level parsing.
    Detects reentrancy, self-destruct drain, proxy patterns,
    gas bombing, and storage-after-call vulnerabilities.
    """

    # EVM Opcodes
    OP_STOP = 0x00
    OP_CALLVALUE = 0x34
    OP_SLOAD = 0x54
    OP_SSTORE = 0x55
    OP_JUMP = 0x56
    OP_JUMPI = 0x57
    OP_JUMPDEST = 0x5b
    OP_PUSH1 = 0x60
    OP_PUSH32 = 0x7f
    OP_CREATE = 0xf0
    OP_CALL = 0xf1
    OP_CALLCODE = 0xf2
    OP_RETURN = 0xf3
    OP_DELEGATECALL = 0xf4
    OP_CREATE2 = 0xf5
    OP_STATICCALL = 0xfa
    OP_REVERT = 0xfd
    OP_INVALID = 0xfe
    OP_SELFDESTRUCT = 0xff

    EXTERNAL_CALLS = {0xf1, 0xf2, 0xf4, 0xfa}  # CALL, CALLCODE, DELEGATECALL, STATICCALL

    def analyze(self, bytecode_hex: str) -> tuple:
        """
        Parse bytecode and detect vulnerability patterns.
        Returns (risk_score, [risk_names]).
        """
        if not bytecode_hex or len(bytecode_hex) < 2:
            return 0.0, []

        opcodes = self._parse_opcodes(bytecode_hex)
        if not opcodes:
            return 0.0, []

        risks = []
        score = 0.0

        # Pattern 1: Reentrancy — external call followed by SSTORE
        if self._detect_reentrancy(opcodes):
            score = max(score, SCRA_SSTORE_AFTER_CALL_RISK)  # 0.40
            risks.append("reentrancy_risk")

        # Pattern 2: Self-destruct with value drain pattern
        if self._detect_selfdestruct_drain(opcodes):
            score = max(score, SCRA_SELFDESTRUCT_RISK)  # 0.35
            risks.append("selfdestruct_drain")

        # Pattern 3: Proxy pattern — DELEGATECALL to variable address
        if self._detect_proxy(opcodes):
            score = max(score, SCRA_PROXY_PATTERN_RISK)  # 0.25
            risks.append("proxy_pattern")

        # Pattern 4: Gas bombing — excessive subcall gas
        if self._detect_gas_bomb(opcodes):
            score += 0.15
            risks.append("gas_bomb")

        # Pattern 5: Storage write after any external call
        if self._detect_storage_after_call(opcodes):
            score = max(score, 0.35)
            risks.append("storage_after_call")

        return min(score, 1.0), risks

    def _parse_opcodes(self, hex_str: str) -> List[int]:
        """Convert hex bytecode to opcode list, skipping PUSH data bytes."""
        try:
            raw = bytes.fromhex(hex_str)
        except ValueError:
            return []

        opcodes = []
        i = 0
        while i < len(raw):
            op = raw[i]
            opcodes.append(op)
            # PUSH1 (0x60) to PUSH32 (0x7f) — skip N data bytes
            if self.OP_PUSH1 <= op <= self.OP_PUSH32:
                skip = op - self.OP_PUSH1 + 1
                i += skip
            i += 1
        return opcodes

    def _detect_reentrancy(self, opcodes: List[int]) -> bool:
        """
        CALL/DELEGATECALL followed by SSTORE without JUMPDEST boundary.
        Classic reentrancy pattern: external call then state change.
        """
        saw_external_call = False
        for op in opcodes:
            if op in self.EXTERNAL_CALLS:
                saw_external_call = True
            elif op == self.OP_JUMPDEST:
                saw_external_call = False  # New code block resets
            elif op == self.OP_SSTORE and saw_external_call:
                return True
        return False

    def _detect_selfdestruct_drain(self, opcodes: List[int]) -> bool:
        """
        SELFDESTRUCT with CALLVALUE check in code — potential drain attack.
        """
        has_callvalue = self.OP_CALLVALUE in opcodes
        has_selfdestruct = self.OP_SELFDESTRUCT in opcodes
        return has_callvalue and has_selfdestruct

    def _detect_proxy(self, opcodes: List[int]) -> bool:
        """
        DELEGATECALL without constant target (no PUSH right before).
        Indicates upgradeable proxy — target can be changed.
        """
        for i, op in enumerate(opcodes):
            if op == self.OP_DELEGATECALL:
                # Check if 2-5 opcodes before contain a PUSH
                has_push = False
                start = max(0, i - 5)
                for j in range(start, i):
                    if self.OP_PUSH1 <= opcodes[j] <= self.OP_PUSH32:
                        has_push = True
                        break
                if not has_push:
                    return True  # Variable target = proxy pattern
        return False

    def _detect_gas_bomb(self, opcodes: List[int]) -> bool:
        """
        Multiple CALL opcodes in sequence — potential gas bombing.
        """
        call_count = sum(1 for op in opcodes if op in self.EXTERNAL_CALLS)
        return call_count >= 5  # 5+ external calls is suspicious

    def _detect_storage_after_call(self, opcodes: List[int]) -> bool:
        """
        SSTORE appearing after any external CALL in the same function.
        More general than reentrancy check — catches indirect patterns.
        """
        call_seen = False
        for op in opcodes:
            if op in (self.OP_CALL, self.OP_CALLCODE):
                call_seen = True
            elif op == self.OP_SSTORE and call_seen:
                return True
            elif op in (self.OP_STOP, self.OP_RETURN, self.OP_REVERT):
                call_seen = False  # Function boundary resets
        return False


class ContractAnalyzer:
    """
    Analyzes smart contract interactions for security risks.
    Maintains a registry of known-safe contracts and their patterns.
    """

    def __init__(self):
        # Known contract patterns (code_hash -> risk_level)
        self.known_contracts: Dict[bytes, float] = {}
        # Call graph tracking: caller -> [callees]
        self.call_graph: Dict[bytes, List[bytes]] = {}
        # Contract deployment history
        self.deployed_contracts: Dict[bytes, dict] = {}
        # Interaction patterns
        self.interaction_count: Dict[bytes, int] = {}
        self.MAX_TRACKED = 10000

        # Phase 15: Deep bytecode analyzer
        self.bytecode_analyzer = BytecodeAnalyzer()

        # Phase 16: Account data tracking for GAT node features
        self._account_data: Dict[bytes, dict] = {}  # addr -> {nonce, balance, code_size, first_seen, total_value_sent}

        # Neural model for graph-based contract risk analysis
        try:
            from positronic.ai.models.graph_attention import GraphAttentionNet
            self._gat = GraphAttentionNet(node_dim=8, hidden_dim=64)
            self._neural_available = True
        except ImportError:
            self._gat = None
            self._neural_available = False
        self._use_neural = False
        self._neural_threshold = 500
        self._neural_samples = 0
        self._neural_errors: int = 0
        self._consecutive_neural_failures: int = 0

    def analyze_transaction(self, tx: Transaction) -> float:
        """
        Analyze a contract-related transaction for risks.
        Returns risk score 0.0 (safe) to 1.0 (high risk).
        """
        if tx.tx_type not in (TxType.CONTRACT_CREATE, TxType.CONTRACT_CALL):
            return 0.0

        reports = []

        if tx.tx_type == TxType.CONTRACT_CREATE:
            reports.append(self._analyze_deployment(tx))
        else:
            reports.append(self._analyze_call(tx))
            reports.append(self._analyze_call_pattern(tx))

        # Combine all risk scores (take maximum)
        rule_score = max(r.risk_score for r in reports) if reports else 0.0

        # Neural augmentation: if active, compute GAT score and take max
        neural_score = 0.0
        if self._use_neural and self._gat is not None:
            try:
                from positronic.ai.models.graph_attention import GraphAttentionNet
                # Phase 16: Pass account_lookup for real node features
                node_features, adj = GraphAttentionNet.build_graph_from_tx(
                    tx, self.call_graph, self.interaction_count,
                    self.known_contracts,
                    account_lookup=self._account_data if self._account_data else None,
                )
                neural_score = self._gat.score(node_features, adj)
                self._consecutive_neural_failures = 0
            except Exception as e:
                logger.debug("GAT scoring error (neural fallback): %s", e)
                neural_score = 0.0
                self._neural_errors += 1  # Neural fallback: GAT scoring
                self._consecutive_neural_failures += 1
                if self._consecutive_neural_failures > 10:
                    self._use_neural = False  # Deactivate neural on degradation

        # Track samples and activate neural model after threshold
        self._neural_samples += 1
        if self._neural_samples >= self._neural_threshold and self._neural_available:
            self._use_neural = True

        return max(rule_score, neural_score)

    def _analyze_deployment(self, tx: Transaction) -> ContractRiskReport:
        """Analyze a contract deployment for risks."""
        report = ContractRiskReport()
        bytecode = tx.data

        if not bytecode:
            return report

        # Check bytecode size
        if len(bytecode) > 48 * 1024:  # > 48KB is suspicious
            report.risk_type = RiskType.LARGE_DEPLOY
            report.risk_score = 0.4
            report.description = "Unusually large contract deployment"
            return report

        # Phase 15: Deep bytecode analysis (replaces naive hex matching)
        bytecode_hex = bytecode.hex()
        deep_score, deep_risks = self.bytecode_analyzer.analyze(bytecode_hex)

        if deep_score > 0 and deep_risks:
            report.risk_score = max(report.risk_score, deep_score)
            # Map to best-fit risk type
            if "reentrancy_risk" in deep_risks or "storage_after_call" in deep_risks:
                report.risk_type = RiskType.REENTRANCY
            elif "selfdestruct_drain" in deep_risks:
                report.risk_type = RiskType.SELF_DESTRUCT
            elif "proxy_pattern" in deep_risks:
                report.risk_type = RiskType.UNAUTHORIZED_CALL
            elif "gas_bomb" in deep_risks:
                report.risk_type = RiskType.HIGH_GAS_CALL
            report.description = f"Bytecode analysis: {', '.join(deep_risks)}"
            report.details["risks"] = deep_risks

        # Legacy fallback: simple hex checks (kept for backward compatibility)
        # Only apply if deep analysis didn't find anything
        if deep_score == 0:
            if "ff" in bytecode_hex:
                report.risk_score = max(report.risk_score, 0.3)
                report.risk_type = RiskType.SELF_DESTRUCT
                report.description = "Contract contains self-destruct capability"

            if "f4" in bytecode_hex:
                report.risk_score = max(report.risk_score, 0.25)
                report.risk_type = RiskType.UNAUTHORIZED_CALL
                report.description = "Contract uses delegatecall"

        return report

    def _analyze_call(self, tx: Transaction) -> ContractRiskReport:
        """Analyze a contract call for risks."""
        report = ContractRiskReport()

        # Check if target contract is known
        target = tx.recipient
        if target not in self.interaction_count:
            report.risk_type = RiskType.UNKNOWN_CONTRACT
            report.risk_score = 0.2
            report.description = "Interaction with previously unseen contract"
        else:
            # Known contract, lower risk
            interactions = self.interaction_count[target]
            if interactions > 100:
                report.risk_score = 0.05  # Well-known contract
            elif interactions > 10:
                report.risk_score = 0.1

        # Check gas limit (very high gas for a call is suspicious)
        if tx.gas_limit > 5_000_000:
            report.risk_score = max(report.risk_score, 0.3)
            report.risk_type = RiskType.HIGH_GAS_CALL
            report.description = "Unusually high gas limit for contract call"

        # Check value with call (potential reentrancy vector)
        if tx.value > 0 and tx.tx_type == TxType.CONTRACT_CALL:
            report.risk_score = max(report.risk_score, 0.15)

        return report

    def _analyze_call_pattern(self, tx: Transaction) -> ContractRiskReport:
        """Analyze the call pattern in the call graph."""
        report = ContractRiskReport()
        sender = tx.sender
        target = tx.recipient

        # Track call graph
        if sender not in self.call_graph:
            self.call_graph[sender] = []
        self.call_graph[sender].append(target)

        # LRU eviction for call_graph and known_contracts
        if len(self.call_graph) > self.MAX_TRACKED:
            oldest_key = next(iter(self.call_graph))
            del self.call_graph[oldest_key]
        if len(self.known_contracts) > self.MAX_TRACKED:
            oldest_key = next(iter(self.known_contracts))
            del self.known_contracts[oldest_key]

        # Check for recursive patterns (A -> B -> A)
        if target in self.call_graph:
            callees = self.call_graph[target]
            if sender in callees:
                report.risk_type = RiskType.RECURSIVE_CALL
                report.risk_score = 0.5
                report.description = "Recursive call pattern detected (potential reentrancy)"

        return report

    def register_contract(self, address: bytes, code_hash: bytes, risk_level: float = 0.0):
        """Register a known contract."""
        self.known_contracts[code_hash] = risk_level
        self.deployed_contracts[address] = {
            "code_hash": code_hash,
            "risk_level": risk_level,
        }

    def record_interaction(self, contract_address: bytes, value: int = 0):
        """Record a successful interaction with a contract."""
        self.interaction_count[contract_address] = \
            self.interaction_count.get(contract_address, 0) + 1

        # Phase 16: Update cumulative value sent for account tracking
        if contract_address in self._account_data:
            self._account_data[contract_address]["total_value_sent"] = (
                self._account_data[contract_address].get("total_value_sent", 0) + value
            )

    def update_account_data(self, address: bytes, nonce: int = 0, balance: int = 0,
                            code_size: int = 0, first_seen: float = 0.0,
                            total_value_sent: int = 0):
        """Phase 16: Update account data for GAT node features."""
        if address not in self._account_data:
            import time as _time
            self._account_data[address] = {
                "nonce": nonce,
                "balance": balance,
                "code_size": code_size,
                "first_seen": first_seen or _time.time(),
                "total_value_sent": total_value_sent,
            }
        else:
            acct = self._account_data[address]
            acct["nonce"] = nonce
            acct["balance"] = balance
            if code_size > 0:
                acct["code_size"] = code_size
            if total_value_sent > acct.get("total_value_sent", 0):
                acct["total_value_sent"] = total_value_sent

        # LRU eviction for account data
        if len(self._account_data) > self.MAX_TRACKED:
            oldest_key = next(iter(self._account_data))
            del self._account_data[oldest_key]

    def get_stats(self) -> dict:
        return {
            "known_contracts": len(self.known_contracts),
            "deployed_contracts": len(self.deployed_contracts),
            "tracked_interactions": sum(self.interaction_count.values()),
            "neural_active": self._use_neural,
            "neural_errors": self._neural_errors,
            "tracked_accounts": len(self._account_data),
        }
