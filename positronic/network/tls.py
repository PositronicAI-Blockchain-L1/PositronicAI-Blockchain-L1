"""
Positronic - TLS/SSL Support
Provides TLS context creation for secure P2P WebSocket connections (wss://).
Supports custom certificates, self-signed generation, and peer verification.
"""

import os
import ssl
import hashlib
import socket
import getpass
import logging
from typing import Optional, Tuple

logger = logging.getLogger("positronic.network.tls")


def _derive_key_password(key_path: str) -> bytes:
    """Derive a machine-specific password for encrypting TLS private keys.

    Uses a deterministic derivation from machine identity (hostname, user,
    absolute key path) so the same machine can always decrypt its own keys
    without storing a separate password file.
    """
    machine_id = f"{socket.gethostname()}:{getpass.getuser()}:{os.path.abspath(key_path)}"
    return hashlib.sha256(machine_id.encode()).digest()


def generate_self_signed_cert(
    cert_path: str,
    key_path: str,
    common_name: str = "positronic-node",
    password: Optional[bytes] = None,
) -> Tuple[str, str]:
    """
    Generate a self-signed certificate for local/testnet use.
    Uses the cryptography library (already a dependency).
    Returns (cert_path, key_path).
    """
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        import datetime

        # Generate EC private key (secp256r1 for TLS)
        private_key = ec.generate_private_key(ec.SECP256R1())

        # Build certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Positronic Network"),
        ])

        now = datetime.datetime.now(datetime.timezone.utc)
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now)
            .not_valid_after(now + datetime.timedelta(days=365))
            .add_extension(
                x509.SubjectAlternativeName([
                    x509.DNSName("localhost"),
                    x509.DNSName("*.positronic.local"),
                    x509.IPAddress(
                        __import__("ipaddress").IPv4Address("127.0.0.1")
                    ),
                ]),
                critical=False,
            )
            .sign(private_key, hashes.SHA256())
        )

        # Ensure directories exist
        os.makedirs(os.path.dirname(cert_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(key_path) or ".", exist_ok=True)

        # Encrypt private key on disk
        key_password = password or _derive_key_password(key_path)
        with open(key_path, "wb") as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.BestAvailableEncryption(key_password),
            ))

        # Write certificate
        with open(cert_path, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))

        logger.info(f"Generated self-signed certificate: {cert_path}")
        return cert_path, key_path

    except ImportError:
        logger.warning(
            "cryptography library not available for cert generation. "
            "Install with: pip install cryptography"
        )
        raise


def create_server_ssl_context(
    cert_path: Optional[str] = None,
    key_path: Optional[str] = None,
    ca_path: Optional[str] = None,
    verify_peers: bool = False,
    data_dir: str = "./data",
    password: Optional[bytes] = None,
) -> ssl.SSLContext:
    """
    Create an SSL context for the P2P server (accepting connections).

    If no cert/key provided and data_dir is given, generates self-signed.
    Private keys are encrypted on disk with a machine-derived password.
    """
    # Generate self-signed if needed
    if not cert_path or not key_path:
        cert_path = os.path.join(data_dir, "tls", "node.crt")
        key_path = os.path.join(data_dir, "tls", "node.key")
        if not os.path.exists(cert_path) or not os.path.exists(key_path):
            generate_self_signed_cert(cert_path, key_path, password=password)

    # Derive password for loading encrypted private key
    key_password = password or _derive_key_password(key_path)

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert_path, key_path, password=key_password)

    # Minimum TLS 1.2
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2

    # Load CA for peer verification if provided
    if ca_path and os.path.exists(ca_path):
        ctx.load_verify_locations(ca_path)
        if verify_peers:
            ctx.verify_mode = ssl.CERT_REQUIRED
        else:
            ctx.verify_mode = ssl.CERT_OPTIONAL
    else:
        ctx.verify_mode = ssl.CERT_NONE

    return ctx


def create_client_ssl_context(
    ca_path: Optional[str] = None,
    verify_peers: bool = False,
) -> ssl.SSLContext:
    """
    Create an SSL context for outbound P2P connections.

    For testnet/development, verify_peers=False allows self-signed certs.
    For mainnet, set verify_peers=True and provide a CA bundle.
    """
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

    # Minimum TLS 1.2
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2

    if ca_path and os.path.exists(ca_path):
        ctx.load_verify_locations(ca_path)
    elif not verify_peers:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    else:
        # Use system CA bundle
        ctx.load_default_certs()

    return ctx
