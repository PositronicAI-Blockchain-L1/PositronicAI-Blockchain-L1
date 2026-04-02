"""
Positronic - Shared Type Aliases
Type aliases for return types used across the codebase.
These are plain dicts at runtime; type aliases provide documentation.
"""

from typing import Dict, Any, List

# Treasury spend execution result
TreasurySpendResult = Dict[str, Any]

# Team vesting schedule status
TeamVestingStatus = Dict[str, Any]

# Wallet balance information
WalletBalanceInfo = Dict[str, Any]

# Admin transfer result
AdminTransferResult = Dict[str, Any]

# Treasury transaction log record
TreasuryTransactionRecord = Dict[str, Any]

# Unjail request from slashing governance
UnjailRequest = Dict[str, Any]

# Game session records
PlayerSessionRecord = Dict[str, Any]

# Game session statistics
GameSessionStats = Dict[str, Any]

# Emergency state snapshot
EmergencyStateDict = Dict[str, Any]
