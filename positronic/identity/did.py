"""
Positronic - Decentralized Identity (DID)
W3C-compatible decentralized identity system.

Audit fix: Added social recovery mechanism. Key loss no longer means
permanent DID loss. Guardians can collectively authorize ownership transfer
after a RECOVERY_DELAY waiting period.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List
import time
import hashlib

from positronic.constants import RECOVERY_DELAY


@dataclass
class Credential:
    """A verifiable credential."""
    credential_id: str
    issuer: bytes  # DID of issuer
    subject: bytes  # DID of subject
    claim_type: str  # e.g., "education", "employment", "certification"
    claim_data: dict = field(default_factory=dict)
    issued_at: float = 0.0
    expires_at: float = 0.0  # 0 = never expires
    revoked: bool = False
    signature: bytes = b""

    @property
    def is_valid(self) -> bool:
        if self.revoked:
            return False
        if self.expires_at > 0 and time.time() > self.expires_at:
            return False
        return True

    def to_dict(self) -> dict:
        return {
            "credential_id": self.credential_id,
            "issuer": self.issuer.hex(),
            "subject": self.subject.hex(),
            "claim_type": self.claim_type,
            "claim_data": self.claim_data,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "is_valid": self.is_valid,
            "revoked": self.revoked,
        }


@dataclass
class RecoveryRequest:
    """A pending DID recovery request."""
    did: str
    new_owner: bytes
    guardian_approvals: List[bytes] = field(default_factory=list)
    initiated_at: float = 0.0
    executes_after: float = 0.0  # Timestamp after which recovery can execute
    status: str = "pending"      # pending, executed, cancelled

    def to_dict(self) -> dict:
        return {
            "did": self.did,
            "new_owner": self.new_owner.hex(),
            "approvals": len(self.guardian_approvals),
            "initiated_at": self.initiated_at,
            "executes_after": self.executes_after,
            "status": self.status,
        }


@dataclass
class DecentralizedIdentity:
    """A W3C-compatible DID document."""
    did: str  # "did:asf:0x1234..."
    owner: bytes
    verification_methods: List[bytes] = field(default_factory=list)  # Public keys
    credentials: Dict[str, Credential] = field(default_factory=dict)
    services: Dict[str, str] = field(default_factory=dict)  # service_type -> endpoint
    created_at: float = 0.0
    updated_at: float = 0.0
    deactivated: bool = False
    # Recovery guardians
    recovery_guardians: List[bytes] = field(default_factory=list)
    recovery_threshold: int = 0  # 0 = no recovery configured

    def add_credential(self, credential: Credential):
        """Add a credential to this DID."""
        self.credentials[credential.credential_id] = credential
        self.updated_at = time.time()

    def revoke_credential(self, credential_id: str) -> bool:
        """Revoke a credential."""
        cred = self.credentials.get(credential_id)
        if cred is None:
            return False
        cred.revoked = True
        self.updated_at = time.time()
        return True

    def add_verification_method(self, public_key: bytes):
        """Add a verification method (public key)."""
        if public_key not in self.verification_methods:
            self.verification_methods.append(public_key)
            self.updated_at = time.time()

    def add_service(self, service_type: str, endpoint: str):
        """Add a service endpoint."""
        self.services[service_type] = endpoint
        self.updated_at = time.time()

    def get_valid_credentials(self) -> List[Credential]:
        """Get all valid (non-revoked, non-expired) credentials."""
        return [c for c in self.credentials.values() if c.is_valid]

    def to_dict(self) -> dict:
        return {
            "did": self.did,
            "owner": self.owner.hex(),
            "verification_methods": [vm.hex() for vm in self.verification_methods],
            "credentials_count": len(self.credentials),
            "valid_credentials": len(self.get_valid_credentials()),
            "services": self.services,
            "created_at": self.created_at,
            "deactivated": self.deactivated,
        }


class DIDRegistry:
    """Registry for all decentralized identities."""

    def __init__(self):
        self._identities: Dict[str, DecentralizedIdentity] = {}
        self._address_to_did: Dict[bytes, str] = {}
        self._total_credentials: int = 0
        self._recovery_requests: Dict[str, RecoveryRequest] = {}  # did -> request

    def _generate_did(self, address: bytes) -> str:
        return f"did:asf:{address.hex()}"

    def create_identity(self, owner: bytes, public_key: bytes = b"") -> DecentralizedIdentity:
        """Create a new DID."""
        did_str = self._generate_did(owner)
        if did_str in self._identities:
            return self._identities[did_str]

        identity = DecentralizedIdentity(
            did=did_str,
            owner=owner,
            created_at=time.time(),
            updated_at=time.time(),
        )
        if public_key:
            identity.add_verification_method(public_key)

        self._identities[did_str] = identity
        self._address_to_did[owner] = did_str
        return identity

    def get_identity(self, did: str) -> Optional[DecentralizedIdentity]:
        return self._identities.get(did)

    def get_by_address(self, address: bytes) -> Optional[DecentralizedIdentity]:
        did = self._address_to_did.get(address)
        if did:
            return self._identities.get(did)
        return None

    def issue_credential(self, issuer: bytes, subject: bytes,
                         claim_type: str, claim_data: dict,
                         duration: float = 0) -> Optional[Credential]:
        """Issue a credential from issuer to subject."""
        issuer_did = self._address_to_did.get(issuer)
        subject_did = self._address_to_did.get(subject)
        if issuer_did is None or subject_did is None:
            return None

        cred_id = f"cred_{hashlib.sha256(f'{issuer.hex()}_{subject.hex()}_{time.time()}'.encode()).hexdigest()[:12]}"
        expires = time.time() + duration if duration > 0 else 0

        credential = Credential(
            credential_id=cred_id,
            issuer=issuer,
            subject=subject,
            claim_type=claim_type,
            claim_data=claim_data,
            issued_at=time.time(),
            expires_at=expires,
        )

        # Add to subject's DID
        subject_identity = self._identities.get(subject_did)
        if subject_identity:
            subject_identity.add_credential(credential)

        self._total_credentials += 1
        return credential

    def verify_credential(self, credential_id: str, subject: bytes) -> bool:
        """Verify a credential is valid."""
        identity = self.get_by_address(subject)
        if identity is None:
            return False
        cred = identity.credentials.get(credential_id)
        if cred is None:
            return False
        return cred.is_valid

    def revoke_credential(self, credential_id: str, issuer: bytes) -> bool:
        """Revoke a credential (only issuer can revoke)."""
        for identity in self._identities.values():
            cred = identity.credentials.get(credential_id)
            if cred and cred.issuer == issuer:
                cred.revoked = True
                return True
        return False

    def deactivate_identity(self, did: str, owner: bytes) -> bool:
        """Deactivate a DID."""
        identity = self._identities.get(did)
        if identity is None or identity.owner != owner:
            return False
        identity.deactivated = True
        return True

    # === Recovery Mechanism ===

    def set_recovery_guardians(self, did: str, owner: bytes,
                                guardian_addresses: List[bytes],
                                threshold: int) -> bool:
        """Set recovery guardians for a DID. Only owner can set."""
        identity = self._identities.get(did)
        if identity is None or identity.owner != owner:
            return False
        if threshold < 1 or threshold > len(guardian_addresses):
            return False
        identity.recovery_guardians = list(guardian_addresses)
        identity.recovery_threshold = threshold
        identity.updated_at = time.time()
        return True

    def initiate_recovery(self, did: str, new_owner: bytes,
                          guardian_signatures: List[bytes]) -> Optional[RecoveryRequest]:
        """
        Initiate DID recovery. Requires threshold guardian approvals.
        Recovery executes after RECOVERY_DELAY (7 days).
        """
        identity = self._identities.get(did)
        if identity is None or identity.deactivated:
            return None
        if identity.recovery_threshold == 0:
            return None  # No recovery configured

        # Verify guardian approvals meet threshold
        valid_approvals = [
            g for g in guardian_signatures
            if g in identity.recovery_guardians
        ]
        if len(valid_approvals) < identity.recovery_threshold:
            return None

        now = time.time()
        request = RecoveryRequest(
            did=did,
            new_owner=new_owner,
            guardian_approvals=valid_approvals,
            initiated_at=now,
            executes_after=now + RECOVERY_DELAY,
        )
        self._recovery_requests[did] = request
        return request

    def execute_recovery(self, did: str) -> bool:
        """
        Execute a pending recovery after the delay period.
        Transfers DID ownership to the new owner.
        """
        request = self._recovery_requests.get(did)
        if request is None or request.status != "pending":
            return False
        if time.time() < request.executes_after:
            return False  # Delay not yet passed

        identity = self._identities.get(did)
        if identity is None:
            return False

        old_owner = identity.owner
        identity.owner = request.new_owner
        identity.updated_at = time.time()

        # Update address mapping
        if old_owner in self._address_to_did:
            del self._address_to_did[old_owner]
        self._address_to_did[request.new_owner] = did

        request.status = "executed"
        return True

    def cancel_recovery(self, did: str, owner: bytes) -> bool:
        """Cancel a pending recovery (current owner can cancel)."""
        request = self._recovery_requests.get(did)
        if request is None or request.status != "pending":
            return False
        identity = self._identities.get(did)
        if identity is None or identity.owner != owner:
            return False
        request.status = "cancelled"
        return True

    def get_recovery_request(self, did: str) -> Optional[dict]:
        """Get pending recovery request for a DID."""
        request = self._recovery_requests.get(did)
        return request.to_dict() if request else None

    def get_stats(self) -> dict:
        return {
            "total_identities": len(self._identities),
            "active_identities": sum(1 for i in self._identities.values() if not i.deactivated),
            "total_credentials": self._total_credentials,
            "pending_recoveries": sum(
                1 for r in self._recovery_requests.values() if r.status == "pending"
            ),
        }
