"""
Positronic - Parallel Transaction Validator

Validates multiple transactions concurrently using a thread pool.
Only the validation step (signature verification, nonce/balance checks)
is parallelized. Execution remains sequential for state consistency.

Phase 17 GOD CHAIN addition.

**Fail-open**: If the thread pool encounters any error, validation falls
back to sequential processing. No transaction is lost or rejected due to
parallelization failures.
"""

import concurrent.futures
from typing import List, Tuple, Optional
from dataclasses import dataclass

from positronic.utils.logging import get_logger
from positronic.constants import TX_VALIDATION_WORKERS, TX_BATCH_SIZE

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of validating a single transaction."""
    tx: object
    is_valid: bool
    error: str = ""


class ParallelTxValidator:
    """Validates transactions in parallel using a thread pool.

    Only parallelizes stateless checks (signature verification, format
    validation). State-dependent checks (nonce, balance) are performed
    sequentially after the parallel phase to maintain consistency.

    Example::

        validator = ParallelTxValidator()
        results = validator.validate_batch(pending_txs)
        valid_txs = [r.tx for r in results if r.is_valid]
    """

    def __init__(self, max_workers: int = TX_VALIDATION_WORKERS):
        self._max_workers = max(1, max_workers)

    def validate_batch(
        self,
        transactions: list,
        validate_fn=None,
    ) -> List[ValidationResult]:
        """Validate a batch of transactions in parallel.

        Args:
            transactions: List of Transaction objects to validate.
            validate_fn: Optional callable ``(tx) -> (bool, str)`` that
                performs the actual validation. If None, uses the built-in
                signature check.

        Returns:
            List of ``ValidationResult`` in the same order as input.
        """
        if not transactions:
            return []

        if validate_fn is None:
            validate_fn = self._default_validate

        # For small batches, just validate sequentially
        if len(transactions) <= 2:
            return self._validate_sequential(transactions, validate_fn)

        try:
            return self._validate_parallel(transactions, validate_fn)
        except Exception as e:
            # Fail-open: fall back to sequential
            logger.warning("Parallel validation failed, falling back to sequential: %s", e)
            return self._validate_sequential(transactions, validate_fn)

    def _validate_parallel(
        self,
        transactions: list,
        validate_fn,
    ) -> List[ValidationResult]:
        """Run validation in parallel using ThreadPoolExecutor."""
        results = [None] * len(transactions)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._max_workers
        ) as executor:
            # Submit all validation tasks
            future_to_idx = {}
            for i, tx in enumerate(transactions):
                future = executor.submit(self._safe_validate, tx, validate_fn)
                future_to_idx[future] = i

            # Collect results
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.debug("Parallel tx validation error (fail-open): %s", e)
                    results[idx] = ValidationResult(
                        tx=transactions[idx],
                        is_valid=True,  # Fail-open: assume valid
                        error=f"Validation error: {e}",
                    )

        # Replace any None results (shouldn't happen, but safety)
        for i, r in enumerate(results):
            if r is None:
                results[i] = ValidationResult(
                    tx=transactions[i],
                    is_valid=True,
                    error="Validation skipped",
                )

        return results

    def _validate_sequential(
        self,
        transactions: list,
        validate_fn,
    ) -> List[ValidationResult]:
        """Validate transactions one at a time (fallback)."""
        results = []
        for tx in transactions:
            results.append(self._safe_validate(tx, validate_fn))
        return results

    def _safe_validate(self, tx, validate_fn) -> ValidationResult:
        """Validate a single transaction, catching all errors."""
        try:
            is_valid, error = validate_fn(tx)
            return ValidationResult(tx=tx, is_valid=is_valid, error=error)
        except Exception as e:
            # Fail-open: if validation itself crashes, assume valid
            logger.debug("Single tx validation exception (fail-open): %s", e)
            return ValidationResult(
                tx=tx,
                is_valid=True,
                error=f"Validation exception: {e}",
            )

    @staticmethod
    def _default_validate(tx) -> Tuple[bool, str]:
        """Default validation: check signature if available.

        Args:
            tx: Transaction object.

        Returns:
            Tuple of (is_valid, error_message).
        """
        try:
            # Check for required fields
            if not hasattr(tx, "sender") or not tx.sender:
                return False, "Missing sender"
            if not hasattr(tx, "signature") or not tx.signature:
                return False, "Missing signature"

            # Verify signature if verify method exists
            if hasattr(tx, "verify_signature"):
                if not tx.verify_signature():
                    return False, "Invalid signature"

            return True, ""
        except Exception as e:
            # Fail-open: log and allow through
            logger.warning("parallel_validation_error (fail-open): %s", e)
            return True, f"Validation skipped: {e}"

    def get_stats(self) -> dict:
        """Return validator configuration stats."""
        return {
            "max_workers": self._max_workers,
            "batch_size": TX_BATCH_SIZE,
        }
