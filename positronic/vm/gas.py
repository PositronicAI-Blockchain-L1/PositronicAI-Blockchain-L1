"""
Positronic - PositronicVM Gas Metering

Tracks gas consumption during contract execution.
Provides gas limiting, consumption, and refund mechanisms.
"""


class OutOfGasError(Exception):
    """Raised when execution runs out of gas."""
    pass


class GasMeter:
    """
    Gas metering for PositronicVM execution.

    Tracks:
        - gas_limit: Maximum gas available for execution
        - gas_used: Total gas consumed so far
        - gas_refund: Gas to be refunded after execution (e.g., storage clears)

    The effective gas used after execution is:
        effective = gas_used - min(gas_refund, gas_used // 5)
    (Refund capped at 20% of gas used, similar to EIP-3529.)
    """

    def __init__(self, gas_limit: int):
        """
        Initialize the gas meter.

        Args:
            gas_limit: Maximum gas available for this execution context.
        """
        if gas_limit < 0:
            raise ValueError("Gas limit cannot be negative")
        self._gas_limit = gas_limit
        self._gas_used = 0
        self._gas_refund = 0

    @property
    def gas_limit(self) -> int:
        """Total gas allocated for this execution."""
        return self._gas_limit

    @property
    def gas_used(self) -> int:
        """Total gas consumed so far."""
        return self._gas_used

    @property
    def gas_remaining(self) -> int:
        """Gas still available for consumption."""
        return self._gas_limit - self._gas_used

    @property
    def gas_refund(self) -> int:
        """Accumulated gas refund."""
        return self._gas_refund

    def consume(self, amount: int) -> None:
        """
        Consume gas.

        Args:
            amount: Amount of gas to consume.

        Raises:
            OutOfGasError: If consuming this amount would exceed the gas limit.
        """
        if amount < 0:
            raise ValueError("Cannot consume negative gas")
        if self._gas_used + amount > self._gas_limit:
            # Set gas_used to limit to indicate all gas consumed
            self._gas_used = self._gas_limit
            raise OutOfGasError(
                f"Out of gas: need {amount}, have {self.gas_remaining} remaining "
                f"(used {self._gas_used} of {self._gas_limit})"
            )
        self._gas_used += amount

    def consume_all(self) -> None:
        """
        Consume all remaining gas.
        Used for INVALID opcode and certain error conditions.
        """
        self._gas_used = self._gas_limit

    def return_gas(self, amount: int) -> None:
        """
        Return unused gas (e.g., from a sub-call that did not use all allocated gas).

        Args:
            amount: Amount of gas to return.
        """
        if amount < 0:
            raise ValueError("Cannot return negative gas")
        self._gas_used = max(0, self._gas_used - amount)

    def add_refund(self, amount: int) -> None:
        """
        Add to the gas refund counter.
        Refunds are applied after execution, capped at 20% of gas used.

        Args:
            amount: Amount of gas to add to refund.
        """
        if amount < 0:
            raise ValueError("Cannot add negative refund")
        self._gas_refund += amount

    def sub_refund(self, amount: int) -> None:
        """
        Subtract from the gas refund counter.
        Used when a storage slot is re-dirtied.

        Args:
            amount: Amount to subtract from refund.
        """
        self._gas_refund = max(0, self._gas_refund - amount)

    @property
    def effective_gas_used(self) -> int:
        """
        Calculate effective gas used after applying refunds.
        Refund is capped at 20% of total gas used (EIP-3529 style).

        Returns:
            Effective gas consumed after refund.
        """
        max_refund = self._gas_used // 5  # 20% cap
        actual_refund = min(self._gas_refund, max_refund)
        return self._gas_used - actual_refund

    @property
    def effective_refund(self) -> int:
        """
        Calculate the actual refund that will be applied.

        Returns:
            Capped refund amount.
        """
        max_refund = self._gas_used // 5
        return min(self._gas_refund, max_refund)

    def gas_for_call(self, requested_gas: int) -> int:
        """
        Calculate gas to forward to a sub-call.
        At most 63/64 of remaining gas (EIP-150 rule).

        Args:
            requested_gas: Gas amount requested by the CALL opcode.

        Returns:
            Actual gas to forward (min of requested and 63/64 of remaining).
        """
        remaining = self.gas_remaining
        max_allowed = remaining - (remaining // 64)  # 63/64 rule
        return min(requested_gas, max_allowed)

    def has_gas(self, amount: int) -> bool:
        """
        Check if enough gas is available without consuming it.

        Args:
            amount: Amount of gas to check.

        Returns:
            True if at least `amount` gas is remaining.
        """
        return self.gas_remaining >= amount

    def reset(self, gas_limit: int) -> None:
        """
        Reset the gas meter with a new limit.

        Args:
            gas_limit: New gas limit.
        """
        self._gas_limit = gas_limit
        self._gas_used = 0
        self._gas_refund = 0

    def __repr__(self) -> str:
        return (
            f"GasMeter(limit={self._gas_limit}, "
            f"used={self._gas_used}, "
            f"remaining={self.gas_remaining}, "
            f"refund={self._gas_refund})"
        )
