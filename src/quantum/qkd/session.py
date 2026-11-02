"""QKD Session Manager.

Orchestrates a complete Quantum Key Distribution session between Alice and Bob,
abstracting over the BB84 or E91 protocols.  Provides:

* Quantum key generation and pooling.
* Automatic key refresh when the pool falls below a watermark.
* AES-256-GCM symmetric encryption and decryption using quantum-derived keys.
* Key attestation via HMAC-SHA3-256.
* Thread-safe key-pool access.
* Session metadata and audit logging.

Dependencies
------------
* Python standard library: ``os``, ``hmac``, ``hashlib``, ``secrets``,
  ``threading``, ``time``, ``struct``, ``enum``, ``dataclasses``, ``typing``
* ``cryptography`` package (``pip install cryptography``) for AES-256-GCM.
* Local: :class:`~quantum.qkd.bb84.BB84Protocol`,
         :class:`~quantum.qkd.e91.E91Protocol`.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import struct
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Literal, Optional, Tuple, Union

from .bb84 import BB84Protocol, BB84Result
from .e91 import E91Protocol, E91Result

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AES-256-GCM via the `cryptography` package (pure-Python fallback included)
# ---------------------------------------------------------------------------

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM as _AESGCM
    _HAS_CRYPTOGRAPHY = True
except ImportError:  # pragma: no cover
    _HAS_CRYPTOGRAPHY = False
    logger.warning(
        "The 'cryptography' package is not installed.  "
        "Falling back to XOR-stream cipher (NOT production-secure).  "
        "Install with: pip install cryptography"
    )


# ---------------------------------------------------------------------------
# Enums and constants
# ---------------------------------------------------------------------------

class ProtocolType(Enum):
    """Supported QKD protocol backends."""
    BB84 = auto()
    E91 = auto()


class SessionState(Enum):
    """Lifecycle states of a :class:`QKDSession`."""
    IDLE = auto()
    INITIALISING = auto()
    ACTIVE = auto()
    KEY_REFRESH = auto()
    CLOSED = auto()
    ERROR = auto()


#: Minimum AES-256 key size in bytes
AES_KEY_BYTES: int = 32          # 256 bits
#: GCM nonce size in bytes
GCM_NONCE_BYTES: int = 12        # 96 bits
#: GCM authentication tag size in bytes
GCM_TAG_BYTES: int = 16          # 128 bits
#: Default key pool minimum watermark (bytes)
DEFAULT_LOW_WATERMARK: int = 128
#: Default key pool target size (bytes)
DEFAULT_POOL_SIZE: int = 4096


# ---------------------------------------------------------------------------
# Encrypted message container
# ---------------------------------------------------------------------------

@dataclass
class EncryptedMessage:
    """Container for an AES-256-GCM encrypted payload.

    Attributes:
        ciphertext:  Encrypted bytes (includes GCM tag when using
                     the ``cryptography`` backend).
        nonce:       12-byte GCM nonce used for this message.
        key_id:      Identifier of the key used (for key management).
        timestamp:   Unix timestamp of encryption.
        aad:         Optional additional authenticated data.
        hmac_tag:    HMAC-SHA3-256 of ``nonce + ciphertext`` for
                     extra integrity assurance.
    """

    ciphertext: bytes
    nonce: bytes
    key_id: str
    timestamp: float
    aad: bytes = b""
    hmac_tag: bytes = b""

    def to_bytes(self) -> bytes:
        """Serialise to a compact binary wire format.

        Format (big-endian):
            [4B nonce_len][nonce][4B ct_len][ciphertext]
            [4B aad_len][aad][4B hmac_len][hmac_tag]
            [8B timestamp (double)][key_id (UTF-8, null-terminated)]
        """
        nonce_b = struct.pack(">I", len(self.nonce)) + self.nonce
        ct_b = struct.pack(">I", len(self.ciphertext)) + self.ciphertext
        aad_b = struct.pack(">I", len(self.aad)) + self.aad
        hmac_b = struct.pack(">I", len(self.hmac_tag)) + self.hmac_tag
        ts_b = struct.pack(">d", self.timestamp)
        kid_b = self.key_id.encode("utf-8") + b"\x00"
        return nonce_b + ct_b + aad_b + hmac_b + ts_b + kid_b

    @classmethod
    def from_bytes(cls, data: bytes) -> "EncryptedMessage":
        """Deserialise from the binary wire format produced by :meth:`to_bytes`."""
        offset = 0

        def _read_field() -> bytes:
            nonlocal offset
            length = struct.unpack_from(">I", data, offset)[0]
            offset += 4
            val = data[offset: offset + length]
            offset += length
            return val

        nonce = _read_field()
        ciphertext = _read_field()
        aad = _read_field()
        hmac_tag = _read_field()
        timestamp = struct.unpack_from(">d", data, offset)[0]
        offset += 8
        null_pos = data.index(b"\x00", offset)
        key_id = data[offset:null_pos].decode("utf-8")
        return cls(
            ciphertext=ciphertext,
            nonce=nonce,
            key_id=key_id,
            timestamp=timestamp,
            aad=aad,
            hmac_tag=hmac_tag,
        )


# ---------------------------------------------------------------------------
# Key pool
# ---------------------------------------------------------------------------

@dataclass
class KeyPool:
    """Thread-safe pool of quantum-derived keying material.

    Keys are stored as a byte buffer.  Consumers extract slices (sub-keys)
    of a requested length.  When the pool falls below the watermark,
    callers are expected to trigger a refresh.

    Parameters
    ----------
    low_watermark: Minimum byte count before a refresh is required.
    max_size:      Maximum pool size (bytes); oldest material is discarded.
    """

    low_watermark: int = DEFAULT_LOW_WATERMARK
    max_size: int = DEFAULT_POOL_SIZE

    def __post_init__(self) -> None:
        self._pool: bytearray = bytearray()
        self._lock: threading.Lock = threading.Lock()
        self._bytes_consumed: int = 0
        self._bytes_added: int = 0
        self._refresh_count: int = 0

    def add(self, key_material: bytes) -> None:
        """Append new keying material to the pool.

        If adding would exceed ``max_size``, the oldest bytes are discarded
        first (FIFO policy).

        Args:
            key_material: Bytes to add.
        """
        with self._lock:
            self._pool.extend(key_material)
            self._bytes_added += len(key_material)
            if len(self._pool) > self.max_size:
                excess = len(self._pool) - self.max_size
                del self._pool[:excess]
                logger.debug("Pool trimmed by %d bytes (max_size=%d).", excess, self.max_size)
            self._refresh_count += 1

    def consume(self, n_bytes: int) -> bytes:
        """Extract ``n_bytes`` from the front of the pool.

        Args:
            n_bytes: Number of bytes to extract.

        Returns:
            Extracted bytes.

        Raises:
            ValueError: If the pool does not have enough material.
        """
        with self._lock:
            if len(self._pool) < n_bytes:
                raise ValueError(
                    f"Key pool exhausted: need {n_bytes} bytes, "
                    f"have {len(self._pool)}."
                )
            key = bytes(self._pool[:n_bytes])
            del self._pool[:n_bytes]
            self._bytes_consumed += n_bytes
            return key

    def needs_refresh(self) -> bool:
        """Return True if the pool is below the low-watermark threshold."""
        with self._lock:
            return len(self._pool) < self.low_watermark

    def available(self) -> int:
        """Return the number of bytes currently available in the pool."""
        with self._lock:
            return len(self._pool)

    def stats(self) -> Dict[str, int]:
        """Return pool usage statistics."""
        with self._lock:
            return {
                "available_bytes": len(self._pool),
                "total_added_bytes": self._bytes_added,
                "total_consumed_bytes": self._bytes_consumed,
                "refresh_count": self._refresh_count,
                "low_watermark": self.low_watermark,
                "max_size": self.max_size,
            }

    def clear(self) -> None:
        """Securely zero and clear the pool."""
        with self._lock:
            # Overwrite with zeros before releasing (best-effort)
            for i in range(len(self._pool)):
                self._pool[i] = 0
            self._pool = bytearray()


# ---------------------------------------------------------------------------
# Session configuration
# ---------------------------------------------------------------------------

@dataclass
class SessionConfig:
    """Configuration for a :class:`QKDSession`.

    Attributes:
        protocol:         Which QKD protocol to use.
        n_qubits_per_run: Qubits/pairs per QKD protocol run.
        low_watermark:    Key pool low watermark in bytes.
        pool_max_size:    Maximum pool size in bytes.
        auto_refresh:     Automatically refresh the pool in a background thread.
        refresh_interval: Seconds between background refresh attempts.
        max_key_age_secs: Maximum age of a key before it must be rotated.
        bb84_kwargs:      Extra keyword arguments forwarded to BB84Protocol.
        e91_kwargs:       Extra keyword arguments forwarded to E91Protocol.
    """

    protocol: ProtocolType = ProtocolType.BB84
    n_qubits_per_run: int = 2000
    low_watermark: int = DEFAULT_LOW_WATERMARK
    pool_max_size: int = DEFAULT_POOL_SIZE
    auto_refresh: bool = False
    refresh_interval: float = 30.0
    max_key_age_secs: float = 3600.0
    bb84_kwargs: Dict = field(default_factory=dict)
    e91_kwargs: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# QKD Session
# ---------------------------------------------------------------------------

class QKDSession:
    """Manages a complete QKD session between two parties.

    Provides end-to-end key generation, storage, automatic refresh, and
    AES-256-GCM encryption/decryption using quantum-derived keys.

    Usage
    -----
    >>> cfg = SessionConfig(protocol=ProtocolType.BB84, n_qubits_per_run=2000)
    >>> session = QKDSession("alice", "bob", config=cfg)
    >>> session.initialise()
    >>> msg = session.encrypt(b"Hello quantum world!")
    >>> plaintext = session.decrypt(msg)

    Parameters
    ----------
    alice_id: Identifier string for the first party.
    bob_id:   Identifier string for the second party.
    config:   :class:`SessionConfig` options (defaults to BB84 with 2000 qubits).
    """

    def __init__(
        self,
        alice_id: str = "alice",
        bob_id: str = "bob",
        config: Optional[SessionConfig] = None,
        protocol: Union[str, ProtocolType] = "bb84",
        n_qubits: int = 1000,
        **kwargs: Any,
    ):
        self.alice_id = alice_id
        self.bob_id = bob_id
        if config is None:
            proto = ProtocolType.BB84
            if isinstance(protocol, str):
                if protocol.lower() == "e91":
                    proto = ProtocolType.E91
            elif isinstance(protocol, ProtocolType):
                proto = protocol
            config = SessionConfig(protocol=proto, n_qubits_per_run=n_qubits)
        self.config = config
        self.session_id: str = secrets.token_hex(16)
        self.state: SessionState = SessionState.IDLE
        self._pool = KeyPool(
            low_watermark=self.config.low_watermark,
            max_size=self.config.pool_max_size,
        )
        self._key_registry: Dict[str, Tuple[bytes, float]] = {}  # key_id → (key, ts)
        self._session_lock = threading.RLock()
        self._refresh_thread: Optional[threading.Thread] = None
        self._stop_refresh = threading.Event()
        self._stats: Dict[str, int] = {
            "messages_encrypted": 0,
            "messages_decrypted": 0,
            "protocol_runs": 0,
            "keys_generated": 0,
            "bytes_encrypted": 0,
        }
        self._created_at = time.time()
        self._last_key_time: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def establish_key(self) -> "QKDSession":
        """Establish initial quantum keys (alias for initialise)."""
        return self.initialise()

    def initialise(self) -> "QKDSession":
        """Initialise the session by running the first key-generation pass.

        Transitions the session to :attr:`SessionState.ACTIVE`.

        Returns:
            Self (for chaining).

        Raises:
            RuntimeError: If the protocol run fails to produce a key.
        """
        with self._session_lock:
            if self.state not in (SessionState.IDLE, SessionState.ERROR):
                raise RuntimeError(
                    f"Cannot initialise session in state {self.state!r}."
                )
            self.state = SessionState.INITIALISING
            logger.info(
                "Initialising QKD session %s (%s ↔ %s, protocol=%s).",
                self.session_id, self.alice_id, self.bob_id,
                self.config.protocol.name,
            )

        success = self._run_protocol()
        if not success:
            self.state = SessionState.ERROR
            raise RuntimeError(
                "QKD protocol failed to produce a usable key during initialisation."
            )

        with self._session_lock:
            self.state = SessionState.ACTIVE

        if self.config.auto_refresh:
            self._start_refresh_thread()

        logger.info(
            "Session %s active.  Pool: %d bytes available.",
            self.session_id, self._pool.available(),
        )
        return self

    def close(self) -> None:
        """Terminate the session, securely clearing all key material."""
        self._stop_refresh.set()
        if self._refresh_thread and self._refresh_thread.is_alive():
            self._refresh_thread.join(timeout=5.0)
        self._pool.clear()
        # Zero all registered keys
        for key_id in list(self._key_registry.keys()):
            self._key_registry.pop(key_id, None)
        with self._session_lock:
            self.state = SessionState.CLOSED
        logger.info("Session %s closed.", self.session_id)

    def __enter__(self) -> "QKDSession":
        self.initialise()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Protocol runner
    # ------------------------------------------------------------------

    def _run_protocol(self) -> bool:
        """Execute the configured QKD protocol and add keying material to pool.

        Returns:
            True if a key was successfully generated, False otherwise.
        """
        protocol = self.config.protocol
        n = self.config.n_qubits_per_run
        result: Union[BB84Result, E91Result]

        try:
            if protocol == ProtocolType.BB84:
                bb84 = BB84Protocol(
                    n_qubits=n,
                    **self.config.bb84_kwargs,
                )
                result = bb84.run_full_protocol()
            else:
                e91 = E91Protocol(
                    n_pairs=n,
                    **self.config.e91_kwargs,
                )
                result = e91.run_full_protocol()
        except Exception as exc:
            logger.exception("Protocol run raised an exception: %s", exc)
            return False

        if result.eavesdropping_detected:
            logger.error(
                "Eavesdropping detected during protocol run; discarding key."
            )
            return False

        key = result.final_key
        if not key:
            logger.error("Protocol run produced an empty key.")
            return False

        self._pool.add(key)
        self._stats["protocol_runs"] += 1
        self._last_key_time = time.time()
        logger.info(
            "Protocol run #%d: added %d bytes to pool (total available: %d).",
            self._stats["protocol_runs"], len(key), self._pool.available(),
        )
        return True

    def refresh_keys(self) -> bool:
        """Manually trigger a key-refresh protocol run.

        Returns:
            True if the refresh succeeded and new key material was added.
        """
        with self._session_lock:
            if self.state not in (SessionState.ACTIVE, SessionState.KEY_REFRESH):
                logger.warning("refresh_keys called in state %r; skipping.", self.state)
                return False
            self.state = SessionState.KEY_REFRESH

        success = self._run_protocol()

        with self._session_lock:
            if self.state == SessionState.KEY_REFRESH:
                self.state = SessionState.ACTIVE
        return success

    def _start_refresh_thread(self) -> None:
        """Start a daemon thread that auto-refreshes the key pool."""
        self._stop_refresh.clear()
        self._refresh_thread = threading.Thread(
            target=self._refresh_loop,
            name=f"QKDRefresh-{self.session_id[:8]}",
            daemon=True,
        )
        self._refresh_thread.start()
        logger.debug("Auto-refresh thread started (interval=%.1fs).", self.config.refresh_interval)

    def _refresh_loop(self) -> None:
        """Background key-refresh loop."""
        while not self._stop_refresh.is_set():
            self._stop_refresh.wait(timeout=self.config.refresh_interval)
            if self._stop_refresh.is_set():
                break
            if self._pool.needs_refresh():
                logger.debug("Auto-refresh: pool below watermark, refreshing.")
                self.refresh_keys()

    # ------------------------------------------------------------------
    # Key management
    # ------------------------------------------------------------------

    def _acquire_key(self, n_bytes: int = AES_KEY_BYTES) -> Tuple[str, bytes]:
        """Acquire ``n_bytes`` of key material from the pool.

        Triggers a synchronous refresh if the pool is too low.

        Args:
            n_bytes: Bytes of key to acquire.

        Returns:
            ``(key_id, key_bytes)``

        Raises:
            RuntimeError: If the pool cannot supply the requested key even
                          after a refresh attempt.
        """
        if self._pool.available() < n_bytes:
            logger.info("Pool insufficient; triggering synchronous refresh.")
            success = self._run_protocol()
            if not success or self._pool.available() < n_bytes:
                raise RuntimeError(
                    f"Failed to acquire {n_bytes}-byte key: pool has only "
                    f"{self._pool.available()} bytes after refresh."
                )

        raw_key = self._pool.consume(n_bytes)
        key_id = hashlib.sha256(raw_key + secrets.token_bytes(16)).hexdigest()[:16]
        self._key_registry[key_id] = (raw_key, time.time())
        self._stats["keys_generated"] += 1
        return key_id, raw_key

    def _get_registered_key(self, key_id: str) -> bytes:
        """Retrieve a previously registered key by ID.

        Args:
            key_id: The 16-character hex key identifier.

        Returns:
            Key bytes.

        Raises:
            KeyError: If the key_id is unknown.
            ValueError: If the key has exceeded ``max_key_age_secs``.
        """
        entry = self._key_registry.get(key_id)
        if entry is None:
            raise KeyError(f"Unknown key_id: {key_id!r}")
        raw_key, ts = entry
        age = time.time() - ts
        if age > self.config.max_key_age_secs:
            self._key_registry.pop(key_id)
            raise ValueError(
                f"Key {key_id!r} has expired (age {age:.0f}s > "
                f"max {self.config.max_key_age_secs:.0f}s)."
            )
        return raw_key

    def rotate_key(self, key_id: str) -> str:
        """Retire an old key and issue a fresh one.

        Args:
            key_id: ID of the key to retire.

        Returns:
            New key_id.
        """
        self._key_registry.pop(key_id, None)
        new_key_id, _ = self._acquire_key()
        logger.info("Key rotated: %s → %s", key_id, new_key_id)
        return new_key_id

    # ------------------------------------------------------------------
    # AES-256-GCM encryption / decryption
    # ------------------------------------------------------------------

    def encrypt(
        self,
        plaintext: bytes,
        aad: bytes = b"",
        key_id: Optional[str] = None,
    ) -> EncryptedMessage:
        """Encrypt plaintext with AES-256-GCM using a quantum-derived key.

        A fresh 12-byte random nonce is generated per message.  The GCM tag
        is included in ``ciphertext`` when using the ``cryptography`` backend.
        An additional HMAC-SHA3-256 tag covers ``nonce + ciphertext`` for
        defence in depth.

        Args:
            plaintext: Bytes to encrypt.
            aad:       Additional authenticated data (not encrypted, but
                       authenticated).  Safe to send in the clear.
            key_id:    Reuse an existing registered key (for multi-message
                       sessions).  If None, a fresh key is acquired.

        Returns:
            :class:`EncryptedMessage` ready for transmission.

        Raises:
            RuntimeError: If the session is not active.
        """
        self._assert_active("encrypt")

        if key_id is None:
            key_id, key = self._acquire_key(AES_KEY_BYTES)
        else:
            key = self._get_registered_key(key_id)

        nonce = secrets.token_bytes(GCM_NONCE_BYTES)

        if _HAS_CRYPTOGRAPHY:
            aesgcm = _AESGCM(key)
            ciphertext = aesgcm.encrypt(nonce, plaintext, aad or None)
        else:
            # Insecure XOR fallback — for testing without cryptography installed
            import hashlib as _hl
            stream = _hl.sha256(key + nonce).digest()
            # Extend stream via counter mode
            ct_bytes = bytearray()
            counter = 0
            remaining = plaintext
            while remaining:
                block = _hl.sha256(key + nonce + counter.to_bytes(4, "big")).digest()
                chunk = remaining[: len(block)]
                ct_bytes.extend(b ^ k for b, k in zip(chunk, block))
                remaining = remaining[len(block):]
                counter += 1
            ciphertext = bytes(ct_bytes)

        # HMAC for additional integrity assurance
        hmac_tag = hmac.new(
            key, nonce + ciphertext, digestmod="sha3_256"
        ).digest()

        self._stats["messages_encrypted"] += 1
        self._stats["bytes_encrypted"] += len(plaintext)

        return EncryptedMessage(
            ciphertext=ciphertext,
            nonce=nonce,
            key_id=key_id,
            timestamp=time.time(),
            aad=aad,
            hmac_tag=hmac_tag,
        )

    def decrypt(self, message: Union[EncryptedMessage, bytes]) -> bytes:
        """Decrypt an :class:`EncryptedMessage` or serialized bytes.

        Verifies the HMAC tag before attempting decryption (fail-fast on
        tampered messages).

        Args:
            message: The encrypted message or serialized bytes to decrypt.

        Returns:
            Original plaintext bytes.

        Raises:
            RuntimeError: If the session is not active.
            ValueError: If HMAC verification fails (tampered message).
            cryptography.exceptions.InvalidTag: If the GCM tag is invalid.
        """
        self._assert_active("decrypt")

        if isinstance(message, bytes):
            message = EncryptedMessage.from_bytes(message)

        key = self._get_registered_key(message.key_id)

        # Verify HMAC
        expected_hmac = hmac.new(
            key, message.nonce + message.ciphertext, digestmod="sha3_256"
        ).digest()
        if not hmac.compare_digest(expected_hmac, message.hmac_tag):
            raise ValueError(
                "HMAC verification failed — message may have been tampered with."
            )

        if _HAS_CRYPTOGRAPHY:
            aesgcm = _AESGCM(key)
            plaintext = aesgcm.decrypt(
                message.nonce, message.ciphertext, message.aad or None
            )
        else:
            # XOR fallback (matches encrypt)
            import hashlib as _hl
            ct_bytes = message.ciphertext
            pt_bytes = bytearray()
            counter = 0
            remaining = ct_bytes
            while remaining:
                block = _hl.sha256(key + message.nonce + counter.to_bytes(4, "big")).digest()
                chunk = remaining[: len(block)]
                pt_bytes.extend(b ^ k for b, k in zip(chunk, block))
                remaining = remaining[len(block):]
                counter += 1
            plaintext = bytes(pt_bytes)

        self._stats["messages_decrypted"] += 1
        return plaintext

    def encrypt_str(self, text: str, encoding: str = "utf-8", **kwargs) -> EncryptedMessage:
        """Convenience wrapper: encrypt a string.

        Args:
            text: Plaintext string to encrypt.
            encoding: String encoding (default ``'utf-8'``).

        Returns:
            :class:`EncryptedMessage`.
        """
        return self.encrypt(text.encode(encoding), **kwargs)

    def decrypt_str(
        self, message: EncryptedMessage, encoding: str = "utf-8"
    ) -> str:
        """Convenience wrapper: decrypt to a string.

        Args:
            message: Encrypted message.
            encoding: Expected string encoding.

        Returns:
            Decrypted string.
        """
        return self.decrypt(message).decode(encoding)

    # ------------------------------------------------------------------
    # Key attestation
    # ------------------------------------------------------------------

    def attest_key(self, key_id: str, challenge: bytes) -> bytes:
        """Prove possession of a key by signing a challenge.

        Args:
            key_id:    Key identifier.
            challenge: Random bytes from the verifier.

        Returns:
            HMAC-SHA3-256 of ``challenge`` under the key.
        """
        key = self._get_registered_key(key_id)
        return hmac.new(key, challenge, digestmod="sha3_256").digest()

    def verify_attestation(
        self, key_id: str, challenge: bytes, tag: bytes
    ) -> bool:
        """Verify a key attestation tag.

        Args:
            key_id:    Key identifier.
            challenge: Original challenge bytes.
            tag:       Tag to verify.

        Returns:
            True if the tag is valid.
        """
        expected = self.attest_key(key_id, challenge)
        return hmac.compare_digest(expected, tag)

    # ------------------------------------------------------------------
    # Shared key derivation (for sub-key derivation)
    # ------------------------------------------------------------------

    def derive_subkey(
        self,
        key_id: str,
        context: bytes,
        length: int = 32,
    ) -> bytes:
        """Derive a context-specific sub-key using HKDF-SHA3-256.

        Args:
            key_id:  Source key identifier.
            context: Application-specific context label.
            length:  Desired sub-key length in bytes.

        Returns:
            Derived sub-key bytes.
        """
        key = self._get_registered_key(key_id)
        # HKDF-like: extract then expand
        # Extract
        prk = hmac.new(b"quantumshield-fl-hkdf", key, digestmod="sha3_256").digest()
        # Expand
        t = b""
        okm = b""
        counter = 1
        while len(okm) < length:
            t = hmac.new(prk, t + context + bytes([counter]), digestmod="sha3_256").digest()
            okm += t
            counter += 1
        return okm[:length]

    # ------------------------------------------------------------------
    # Diagnostics and introspection
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        """Return a snapshot of session and pool statistics.

        Returns:
            Dictionary with session metadata and usage counters.
        """
        pool_stats = self._pool.stats()
        return {
            "session_id": self.session_id,
            "alice_id": self.alice_id,
            "bob_id": self.bob_id,
            "protocol": self.config.protocol.name,
            "state": self.state.name,
            "created_at": self._created_at,
            "session_age_secs": time.time() - self._created_at,
            "last_key_time": self._last_key_time,
            "registered_keys": len(self._key_registry),
            **self._stats,
            **{f"pool_{k}": v for k, v in pool_stats.items()},
        }

    def _assert_active(self, operation: str) -> None:
        """Raise if the session is not in an active state.

        Args:
            operation: Human-readable operation name for the error message.

        Raises:
            RuntimeError: If the session is not :attr:`SessionState.ACTIVE`
                          or :attr:`SessionState.KEY_REFRESH`.
        """
        if self.state not in (SessionState.ACTIVE, SessionState.KEY_REFRESH):
            raise RuntimeError(
                f"Cannot perform '{operation}': session is in state {self.state!r}. "
                "Call .initialise() first."
            )

    def __repr__(self) -> str:
        return (
            f"QKDSession(id={self.session_id[:8]}…, "
            f"{self.alice_id}↔{self.bob_id}, "
            f"protocol={self.config.protocol.name}, "
            f"state={self.state.name}, "
            f"pool={self._pool.available()}B)"
        )
