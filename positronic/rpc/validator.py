"""RPC input validation utilities."""
import re
import logging

logger = logging.getLogger(__name__)

# Validation patterns
ADDRESS_PATTERN = re.compile(r'^0x[0-9a-fA-F]{40}$')
HEX_PATTERN = re.compile(r'^0x[0-9a-fA-F]*$')
MAX_PARAM_SIZE = 1_048_576  # 1MB max per parameter


def validate_address(addr: str) -> bool:
    """Validate Ethereum-style hex address."""
    return isinstance(addr, str) and bool(ADDRESS_PATTERN.match(addr))


def validate_hex(data: str) -> bool:
    """Validate hex-encoded data."""
    return isinstance(data, str) and bool(HEX_PATTERN.match(data))


def validate_block_number(num) -> bool:
    """Validate block number (int or 'latest'/'earliest'/'pending')."""
    if isinstance(num, str):
        return num in ('latest', 'earliest', 'pending') or HEX_PATTERN.match(num)
    return isinstance(num, int) and num >= 0


def validate_params_size(params) -> bool:
    """Check total parameter size doesn't exceed limit."""
    import json
    try:
        return len(json.dumps(params)) <= MAX_PARAM_SIZE
    except (TypeError, ValueError):
        return False


def sanitize_rpc_params(params, max_items: int = 100) -> list:
    """Sanitize and limit RPC parameters."""
    if params is None:
        return []
    if not isinstance(params, (list, tuple)):
        return [params]
    return list(params[:max_items])
