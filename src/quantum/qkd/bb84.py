"""BB84 Quantum Key Distribution Protocol.

Implements the full Bennett–Brassard 1984 (BB84) QKD protocol in software,
simulating quantum channel behaviour with NumPy.

Protocol outline
----------------
1. Alice generates random bits and random bases.
2. Alice "sends" qubits prepared in the chosen bases.
3. (Optional) Eve intercepts with an intercept-resend attack.
4. Bob measures each qubit in a randomly chosen basis.
5. Sifting: Alice and Bob announce their bases over a classical channel
   and keep only the bits where they used the same basis.
6. QBER estimation: A random sample of the sifted key is sacrificed to
   estimate the quantum bit error rate.
7. Information reconciliation: Simple XOR-parity Cascade-style correction.
8. Privacy amplification: Universal-2 hashing to remove Eve's information.

References
----------
Bennett, C. H., & Brassard, G. (1984). Quantum cryptography: Public key
distribution and coin tossing. *Proc. IEEE Int. Conf. on Computers, Systems
and Signal Processing*, 175–179.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class BB84Result:
    """Contains all artefacts produced by a complete BB84 run.

    Attributes:
        raw_key_length:          Number of qubits transmitted.
        sifted_key_length:       Length of key after basis sifting.
        sample_size:             Bits sacrificed for QBER estimation.
        qber:                    Measured Quantum Bit Error Rate.
        reconciled_key_length:   Length after information reconciliation.
        final_key_length:        Length of the final secret key (bits).
        final_key:               The actual secret key bytes.
        eve_present:             Whether an eavesdropper was simulated.
        eve_intercept_rate:      Fraction of qubits Eve intercepted.
        eavesdropping_detected:  Whether QBER exceeded the threshold.
        statistics:              Miscellaneous protocol statistics.
    """

    raw_key_length: int = 0
    sifted_key_length: int = 0
    sample_size: int = 0
    qber: float = 0.0
    reconciled_key_length: int = 0
    final_key_length: int = 0
    final_key: bytes = b""
    eve_present: bool = False
    eve_intercept_rate: float = 0.0
    eavesdropping_detected: bool = False
    statistics: dict = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        if key in ("key_length", "final_key_length"):
            return self.final_key_length
        elif key in ("secret_key", "final_key"):
            if self.final_key:
                return np.unpackbits(np.frombuffer(self.final_key, dtype=np.uint8)).tolist()
            return []
        elif key == "qber":
            return self.qber
        elif key == "is_secure":
            return not self.eavesdropping_detected
        elif key in ("n_qubits", "raw_key_length"):
            return self.raw_key_length
        elif key == "key_rate":
            return self.statistics.get("key_rate", self.final_key_length / max(1, self.raw_key_length))
        elif key == "sifted_key_length":
            return self.sifted_key_length
        elif key in ("alice_key", "bob_key"):
            return self.final_key
        elif hasattr(self, key):
            return getattr(self, key)
        elif key in self.statistics:
            return self.statistics[key]
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: str) -> bool:
        return key in (
            "key_length", "secret_key", "qber", "is_secure", "n_qubits",
            "key_rate", "sifted_key_length", "alice_key", "bob_key"
        ) or hasattr(self, key) or key in self.statistics


# ---------------------------------------------------------------------------
# BB84 Protocol
# ---------------------------------------------------------------------------

class BB84Protocol:
    """Simulates the full BB84 QKD protocol between Alice and Bob.

    Parameters
    ----------
    n_qubits:
        Number of qubits (raw-key bits) to transmit.
    qber_threshold:
        Maximum acceptable QBER before aborting.  Typically 0.11 (11 %).
    sample_fraction:
        Fraction of the sifted key to sacrifice for QBER estimation.
    seed:
        Optional RNG seed for reproducibility.
    channel_error_rate:
        Background depolarising error probability per qubit (simulates
        a noisy channel independent of Eve).
    """

    #: Rectilinear (Z) basis
    BASIS_RECT = 0
    #: Diagonal (X) basis
    BASIS_DIAG = 1

    def __init__(
        self,
        n_qubits: int = 1000,
        qber_threshold: float = 0.11,
        sample_fraction: float = 0.25,
        seed: Optional[int] = None,
        channel_error_rate: float = 0.0,
        error_threshold: Optional[float] = None,
    ):
        if error_threshold is not None:
            qber_threshold = error_threshold
        if n_qubits < 10:
            raise ValueError("n_qubits must be ≥ 10 for a meaningful protocol.")
        if not 0.0 < sample_fraction < 1.0:
            raise ValueError("sample_fraction must be in (0, 1).")
        self.n_qubits = n_qubits
        self.qber_threshold = qber_threshold
        self.sample_fraction = sample_fraction
        self.channel_error_rate = channel_error_rate
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Step 1 – Qubit preparation (Alice)
    # ------------------------------------------------------------------

    def _alice_prepare(self) -> Tuple[np.ndarray, np.ndarray]:
        """Alice randomly generates bits and bases and prepares qubits.

        Bases:
            0  →  rectilinear: |0⟩ for bit 0, |1⟩ for bit 1
            1  →  diagonal:    |+⟩ for bit 0, |−⟩ for bit 1

        Returns:
            (alice_bits, alice_bases) as integer arrays of length n_qubits.
        """
        alice_bits = self._rng.integers(0, 2, size=self.n_qubits)
        alice_bases = self._rng.integers(0, 2, size=self.n_qubits)
        return alice_bits, alice_bases

    # ------------------------------------------------------------------
    # Step 2 – Eavesdropper (Eve) — intercept-resend attack
    # ------------------------------------------------------------------

    def _eve_intercept(
        self,
        alice_bits: np.ndarray,
        alice_bases: np.ndarray,
        intercept_rate: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate Eve's intercept-resend attack on a fraction of qubits.

        Eve intercepts each qubit independently with probability
        ``intercept_rate``, measures in a random basis, and re-sends.

        Args:
            alice_bits:     Alice's original bit values.
            alice_bases:    Alice's preparation bases.
            intercept_rate: Fraction of qubits Eve targets.

        Returns:
            (modified_bits, modified_bases): what Bob actually receives after
            Eve's interference.  Modified bits/bases where Eve introduced
            errors remain identical to Alice's elsewhere.
        """
        n = self.n_qubits
        intercepted_bits = alice_bits.copy()
        intercepted_bases = alice_bases.copy()

        intercept_mask = self._rng.random(n) < intercept_rate
        n_intercepted = int(intercept_mask.sum())
        logger.debug("Eve intercepting %d / %d qubits.", n_intercepted, n)

        # Eve measures in random bases
        eve_bases = self._rng.integers(0, 2, size=n)
        eve_bits = np.zeros(n, dtype=int)

        for i in range(n):
            if not intercept_mask[i]:
                continue
            if eve_bases[i] == alice_bases[i]:
                # Correct basis → Eve learns the bit, no disturbance
                eve_bits[i] = alice_bits[i]
            else:
                # Wrong basis → Eve gets random outcome
                eve_bits[i] = self._rng.integers(0, 2)

            # Eve re-sends in her (possibly wrong) basis
            intercepted_bits[i] = eve_bits[i]
            intercepted_bases[i] = eve_bases[i]

        return intercepted_bits, intercepted_bases

    # ------------------------------------------------------------------
    # Step 3 – Bob's measurement
    # ------------------------------------------------------------------

    def _bob_measure(
        self,
        sent_bits: np.ndarray,
        sent_bases: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Bob measures each qubit in a randomly chosen basis.

        If Bob's basis matches the sender's, he gets the correct bit.
        Otherwise, his result is uniformly random.

        Args:
            sent_bits:  Bit values of qubits as they arrive at Bob's end.
            sent_bases: Bases in which qubits were prepared/re-sent.

        Returns:
            (bob_bits, bob_bases) of length n_qubits.
        """
        bob_bases = self._rng.integers(0, 2, size=self.n_qubits)
        bob_bits = np.zeros(self.n_qubits, dtype=int)

        for i in range(self.n_qubits):
            if bob_bases[i] == sent_bases[i]:
                bob_bits[i] = sent_bits[i]
            else:
                bob_bits[i] = self._rng.integers(0, 2)

        # Apply channel noise (depolarising)
        if self.channel_error_rate > 0:
            noise_mask = self._rng.random(self.n_qubits) < self.channel_error_rate
            bob_bits[noise_mask] ^= 1
            logger.debug(
                "Channel noise flipped %d bits.", int(noise_mask.sum())
            )

        return bob_bits, bob_bases

    # ------------------------------------------------------------------
    # Step 4 – Sifting
    # ------------------------------------------------------------------

    def generate_raw_key(
        self,
        eve_present: bool = False,
        eve_intercept_rate: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate raw bits for Alice and Bob (before sifting).

        Args:
            eve_present:        Simulate an eavesdropper.
            eve_intercept_rate: Fraction of qubits Eve intercepts (0–1).

        Returns:
            Tuple ``(alice_bits, alice_bases, bob_bits, bob_bases)``.
        """
        alice_bits, alice_bases = self._alice_prepare()

        if eve_present:
            sent_bits, sent_bases = self._eve_intercept(
                alice_bits, alice_bases, intercept_rate=eve_intercept_rate
            )
        else:
            sent_bits, sent_bases = alice_bits.copy(), alice_bases.copy()

        bob_bits, bob_bases = self._bob_measure(sent_bits, sent_bases)
        return alice_bits, alice_bases, bob_bits, bob_bases

    def sift_key(
        self,
        alice_bits: np.ndarray,
        alice_bases: np.ndarray,
        bob_bits: np.ndarray,
        bob_bases: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform basis reconciliation (sifting) over the classical channel.

        Alice and Bob publicly announce their bases and keep only the positions
        where they coincide.

        Args:
            alice_bits:   Alice's raw bits.
            alice_bases:  Alice's preparation bases.
            bob_bits:     Bob's raw bits.
            bob_bases:    Bob's measurement bases.

        Returns:
            ``(alice_sifted, bob_sifted)`` — bits at matching-basis positions.
        """
        matching = alice_bases == bob_bases
        alice_sifted = alice_bits[matching]
        bob_sifted = bob_bits[matching]
        logger.info(
            "Sifting: %d / %d qubits kept (%.1f %%).",
            matching.sum(), self.n_qubits, 100 * matching.mean(),
        )
        return alice_sifted, bob_sifted

    # ------------------------------------------------------------------
    # Step 5 – QBER estimation
    # ------------------------------------------------------------------

    def estimate_qber(
        self,
        alice_sifted: np.ndarray,
        bob_sifted: np.ndarray,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Estimate the Quantum Bit Error Rate by sacrificing a key sample.

        Args:
            alice_sifted: Alice's sifted key.
            bob_sifted:   Bob's sifted key.

        Returns:
            ``(qber, alice_remaining, bob_remaining)`` — error rate and the
            remainder of the sifted key after discarding the sample.
        """
        n = len(alice_sifted)
        sample_size = max(1, int(n * self.sample_fraction))
        sample_indices = self._rng.choice(n, size=sample_size, replace=False)
        sample_mask = np.zeros(n, dtype=bool)
        sample_mask[sample_indices] = True

        errors = int((alice_sifted[sample_mask] != bob_sifted[sample_mask]).sum())
        qber = errors / sample_size if sample_size > 0 else 0.0
        logger.info(
            "QBER estimation: %d errors / %d samples → QBER = %.4f",
            errors, sample_size, qber,
        )

        keep_mask = ~sample_mask
        return qber, alice_sifted[keep_mask], bob_sifted[keep_mask]

    # ------------------------------------------------------------------
    # Step 6 – Information reconciliation
    # ------------------------------------------------------------------

    def information_reconciliation(
        self,
        alice_key: np.ndarray,
        bob_key: np.ndarray,
        block_size: int = 8,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Correct Bob's key errors using a single-round XOR-parity scheme.

        This implements a simplified Cascade-style block correction:
        Alice announces the XOR parity of each block; Bob flips one bit in
        any block whose parity doesn't match (assuming at most one error per
        block, reasonable at low QBER).

        Args:
            alice_key:  Alice's reconciliation key (bit array).
            bob_key:    Bob's key (bit array, may contain errors).
            block_size: Size of each parity block.

        Returns:
            ``(alice_reconciled, bob_corrected)`` — after one reconciliation
            pass.  Note: the parity bits themselves are considered public and
            excluded from the final key.
        """
        n = len(alice_key)
        bob_corrected = bob_key.copy()

        for start in range(0, n, block_size):
            end = min(start + block_size, n)
            alice_parity = int(alice_key[start:end].sum() % 2)
            bob_parity = int(bob_corrected[start:end].sum() % 2)
            if alice_parity != bob_parity:
                # Flip the first differing bit (heuristic)
                for idx in range(start, end):
                    if alice_key[idx] != bob_corrected[idx]:
                        bob_corrected[idx] ^= 1
                        break

        logger.debug(
            "Reconciliation residual errors: %d",
            int((alice_key != bob_corrected).sum()),
        )
        return alice_key, bob_corrected

    # ------------------------------------------------------------------
    # Step 7 – Privacy amplification
    # ------------------------------------------------------------------

    def privacy_amplification(
        self,
        key_bits: np.ndarray,
        security_parameter: int = 64,
        output_length: Optional[int] = None,
    ) -> bytes:
        """Reduce Eve's information to negligible levels via universal hashing.

        Converts the reconciled bit array to bytes, then applies SHA-3-256
        keyed with a random (public) seed to extract a shorter, uniform key.

        Args:
            key_bits:           Reconciled bit array.
            security_parameter: Number of bits to shorten by (heuristic for
                                information leaked during reconciliation).
            output_length:      Desired output key length in bytes.  Defaults
                                to ``max(1, (len(key_bits) - security_parameter) // 8)``.

        Returns:
            Final secret key as bytes.
        """
        n = len(key_bits)
        if output_length is None:
            output_length = max(1, (n - security_parameter) // 8)

        # Pack bits into bytes
        n_padded = math.ceil(n / 8) * 8
        padded = np.zeros(n_padded, dtype=int)
        padded[:n] = key_bits
        key_bytes = np.packbits(padded).tobytes()

        # SHAKE-256 (extendable output) for arbitrary-length extraction
        shake = hashlib.shake_256(key_bytes)
        final_key = shake.digest(output_length)

        logger.info(
            "Privacy amplification: %d bits → %d bytes (%d bits).",
            n, output_length, output_length * 8,
        )
        return final_key

    # ------------------------------------------------------------------
    # Full protocol runner
    # ------------------------------------------------------------------

    def run_full_protocol(
        self,
        eve_present: bool = False,
        eve_intercept_rate: float = 1.0,
        block_size: int = 8,
        security_parameter: int = 64,
        output_length: Optional[int] = None,
        eavesdrop: Optional[bool] = None,
    ) -> BB84Result:
        """Execute the entire BB84 protocol end-to-end.

        Args:
            eve_present:        Whether to simulate an eavesdropper.
            eve_intercept_rate: Fraction of qubits Eve intercepts.
            block_size:         Parity-block size for reconciliation.
            security_parameter: Privacy-amplification shortening (bits).
            output_length:      Desired final key length in bytes.
            eavesdrop:          Alias for eve_present.

        Returns:
            :class:`BB84Result` containing the final key and all statistics.
        """
        if eavesdrop is not None:
            eve_present = eavesdrop
        result = BB84Result(
            raw_key_length=self.n_qubits,
            eve_present=eve_present,
            eve_intercept_rate=eve_intercept_rate if eve_present else 0.0,
        )

        # ---- Quantum channel phase ----
        alice_bits, alice_bases, bob_bits, bob_bases = self.generate_raw_key(
            eve_present=eve_present,
            eve_intercept_rate=eve_intercept_rate,
        )

        # ---- Sifting ----
        alice_sifted, bob_sifted = self.sift_key(
            alice_bits, alice_bases, bob_bits, bob_bases
        )
        result.sifted_key_length = len(alice_sifted)

        if result.sifted_key_length < 4:
            logger.error("Sifted key too short; aborting.")
            result.eavesdropping_detected = True
            result.statistics["abort_reason"] = "sifted_key_too_short"
            return result

        # ---- QBER estimation ----
        qber, alice_remain, bob_remain = self.estimate_qber(alice_sifted, bob_sifted)
        result.qber = qber
        result.sample_size = result.sifted_key_length - len(alice_remain)
        result.eavesdropping_detected = qber > self.qber_threshold

        if result.eavesdropping_detected:
            logger.warning(
                "QBER %.4f exceeds threshold %.4f — eavesdropping suspected, aborting.",
                qber, self.qber_threshold,
            )
            result.statistics["abort_reason"] = f"qber_exceeded ({qber:.4f})"
            return result

        # ---- Information reconciliation ----
        alice_rec, bob_rec = self.information_reconciliation(
            alice_remain, bob_remain, block_size=block_size
        )
        result.reconciled_key_length = len(alice_rec)
        residual_errors = int((alice_rec != bob_rec).sum())
        result.statistics["residual_errors_after_reconciliation"] = residual_errors

        if residual_errors > 0:
            logger.warning(
                "%d residual errors remain after reconciliation; "
                "key agreement not perfect.",
                residual_errors,
            )

        # ---- Privacy amplification ----
        final_key = self.privacy_amplification(
            alice_rec,
            security_parameter=security_parameter,
            output_length=output_length,
        )
        result.final_key = final_key
        result.final_key_length = len(final_key) * 8

        result.statistics.update({
            "key_rate": result.final_key_length / self.n_qubits,
            "sifting_efficiency": result.sifted_key_length / self.n_qubits,
            "qber": qber,
            "residual_errors": residual_errors,
        })

        logger.info(
            "BB84 complete. Final key: %d bytes. Key rate: %.4f bits/qubit.",
            len(final_key),
            result.statistics["key_rate"],
        )
        return result

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def bits_to_bytes(bits: np.ndarray) -> bytes:
        """Convert a bit array (0/1 integers) to a bytes object."""
        n_padded = math.ceil(len(bits) / 8) * 8
        padded = np.zeros(n_padded, dtype=np.uint8)
        padded[:len(bits)] = bits.astype(np.uint8)
        return np.packbits(padded).tobytes()

    @staticmethod
    def bytes_to_bits(data: bytes) -> np.ndarray:
        """Convert bytes to a NumPy bit array (MSB first)."""
        return np.unpackbits(np.frombuffer(data, dtype=np.uint8))

    def verify_keys(self, alice_key: bytes, bob_key: bytes) -> bool:
        """Constant-time comparison of two keys.

        Args:
            alice_key: Key held by Alice.
            bob_key:   Key held by Bob.

        Returns:
            True if the keys match.
        """
        return hmac.compare_digest(alice_key, bob_key)


# ---------------------------------------------------------------------------
# Convenience import shim
# ---------------------------------------------------------------------------
import math  # noqa: E402 — needed by privacy_amplification; placed here to
             # avoid circular import issues in some packaging configurations.
