"""E91 (Ekert 1991) Quantum Key Distribution Protocol.

Implements the entanglement-based QKD scheme proposed by Artur Ekert in 1991.
Security is guaranteed by the violation of CHSH (Bell) inequalities:
any eavesdropper necessarily reduces the degree of violation, making
the attack detectable.

Protocol outline
----------------
1. A source emits entangled |Φ+⟩ Bell pairs, sending one qubit to Alice
   and one to Bob.
2. Alice randomly selects a measurement angle from {a1, a2, a3}.
   Bob  randomly selects a measurement angle from {b1, b2, b3}.
3. After all measurements, Alice and Bob announce their chosen angles.
4. Subset S1 (same angles): used to extract the raw key.
5. Subset S2 (specific differing angles): used for CHSH Bell test.
6. If |S| < 2√2 (ideal) or violation is diminished, eavesdropping is
   suspected and the protocol aborts.
7. Remaining coincident measurements form the sifted key, which is then
   privacy-amplified.

References
----------
Ekert, A. K. (1991). Quantum cryptography based on Bell's theorem.
*Physical Review Letters*, 67(6), 661–663.

Bell, J. S. (1964). On the Einstein Podolsky Rosen paradox.
*Physics*, 1(3), 195–200.
"""
from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Measurement angle sets (radians)
# ---------------------------------------------------------------------------

#: Alice's three measurement directions (E91 standard)
ALICE_ANGLES: Tuple[float, ...] = (0.0, math.pi / 4, math.pi / 2)

#: Bob's three measurement directions (E91 standard)
BOB_ANGLES: Tuple[float, ...] = (math.pi / 4, math.pi / 2, 3 * math.pi / 4)

#: CHSH inequality bound for a local-realistic model
CHSH_CLASSICAL_BOUND: float = 2.0

#: CHSH inequality bound for maximally entangled qubits (Tsirelson bound)
CHSH_QUANTUM_BOUND: float = 2.0 * math.sqrt(2)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class E91Result:
    """Artefacts produced by a complete E91 protocol run.

    Attributes:
        n_pairs:               Total entangled pairs generated.
        sifted_key_length:     Bits available after angle sifting.
        chsh_value:            Measured |S| CHSH parameter.
        chsh_pairs_used:       Pairs used for the Bell test.
        qber:                  Estimated error rate on key bits.
        final_key_length:      Length of the extracted key (bits).
        final_key:             Extracted secret key bytes.
        eavesdropping_detected: Whether Bell violation dropped below threshold.
        statistics:            Additional protocol metrics.
    """

    n_pairs: int = 0
    sifted_key_length: int = 0
    chsh_value: float = 0.0
    chsh_pairs_used: int = 0
    qber: float = 0.0
    final_key_length: int = 0
    final_key: bytes = b""
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
        elif key == "chsh_value":
            return self.chsh_value
        elif key == "n_pairs":
            return self.n_pairs
        elif key == "key_rate":
            return self.statistics.get("key_rate", self.final_key_length / max(1, self.n_pairs))
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
            "key_length", "secret_key", "qber", "is_secure", "chsh_value",
            "n_pairs", "key_rate"
        ) or hasattr(self, key) or key in self.statistics


# ---------------------------------------------------------------------------
# E91 Protocol
# ---------------------------------------------------------------------------

class E91Protocol:
    """Simulates the full E91 entanglement-based QKD protocol.

    Parameters
    ----------
    n_pairs:
        Number of entangled Bell pairs to generate.
    chsh_threshold:
        Minimum |S| value accepted as evidence of no eavesdropping.
        Ideal max is 2√2 ≈ 2.828; values ≤ 2.0 indicate classical (or
        fully eavesdropped) correlations.  Default: 2.4.
    sample_fraction:
        Fraction of *key* pairs sacrificed for QBER estimation.
    alice_angles:
        Alice's three measurement angle set (radians).
    bob_angles:
        Bob's three measurement angle set (radians).
    seed:
        Optional RNG seed for reproducibility.
    eve_intercept_rate:
        Fraction of pairs intercepted by Eve (0 = no Eve).
    """

    def __init__(
        self,
        n_pairs: int = 1000,
        chsh_threshold: float = 2.4,
        sample_fraction: float = 0.2,
        alice_angles: Tuple[float, ...] = ALICE_ANGLES,
        bob_angles: Tuple[float, ...] = BOB_ANGLES,
        seed: Optional[int] = None,
        eve_intercept_rate: float = 0.0,
    ):
        if n_pairs < 20:
            raise ValueError("n_pairs must be ≥ 20.")
        self.n_pairs = n_pairs
        self.chsh_threshold = chsh_threshold
        self.sample_fraction = sample_fraction
        self.alice_angles = alice_angles
        self.bob_angles = bob_angles
        self._rng = np.random.default_rng(seed)
        self.eve_intercept_rate = eve_intercept_rate

    # ------------------------------------------------------------------
    # Bell pair simulation
    # ------------------------------------------------------------------

    def _generate_bell_pairs(self) -> np.ndarray:
        """Generate the hidden-variable representation of |Φ+⟩ pairs.

        For simulation we store a hidden angle λ ∈ [0, 2π) for each pair.
        Measurement outcomes follow:
            P(+1 | angle α) = cos²((α − λ) / 2) for Alice's qubit
            Bob's correlated qubit obeys:
            P(+1 | angle β) = sin²((β − λ) / 2) in the singlet-like formulation

        We use the standard quantum prediction directly:
            E(α, β) = −cos(α − β)

        Args:
            (none — uses self.n_pairs)

        Returns:
            Array of hidden variables λ of shape (n_pairs,).
        """
        return self._rng.uniform(0, 2 * math.pi, size=self.n_pairs)

    def _measure_qubit(
        self,
        hidden_vars: np.ndarray,
        angles: np.ndarray,
        party: str = "alice",
    ) -> np.ndarray:
        """Simulate projective measurements on one party's qubits.

        The quantum probability for outcome +1 when measuring at angle θ
        on the first qubit of |Φ+⟩ with hidden variable λ is:
            P(+1 | θ, λ) = cos²((θ − λ) / 2)

        For Bob's (second) qubit (anti-correlated with Alice in |Φ-⟩,
        correlated in |Φ+⟩), we flip the phase by π:
            P(+1 | θ, λ) = sin²((θ − λ) / 2)

        Args:
            hidden_vars: Hidden variable per pair (shape (n_pairs,)).
            angles:      Chosen measurement angle per pair (shape (n_pairs,)).
            party:       ``'alice'`` or ``'bob'``.

        Returns:
            Outcome array of ±1 integers (shape (n_pairs,)).
        """
        if party == "alice":
            probs_plus = np.cos((angles - hidden_vars) / 2) ** 2
        else:
            probs_plus = np.sin((angles - hidden_vars) / 2) ** 2

        rand = self._rng.random(len(hidden_vars))
        outcomes = np.where(rand < probs_plus, +1, -1)
        return outcomes.astype(int)

    def _eve_measure_and_resend(
        self,
        hidden_vars: np.ndarray,
    ) -> np.ndarray:
        """Eve measures in the Z-basis and re-creates a product state.

        After Eve's intercept-resend, the shared state collapses to a
        mixture of product states, reducing the CHSH parameter to ≤ 2.

        Args:
            hidden_vars: Original hidden variables.

        Returns:
            Modified hidden variables (effectively randomised to destroy
            entanglement in intercepted pairs).
        """
        n = len(hidden_vars)
        intercept_mask = self._rng.random(n) < self.eve_intercept_rate
        n_intercepted = int(intercept_mask.sum())
        if n_intercepted == 0:
            return hidden_vars
        logger.debug("Eve intercepting %d / %d pairs.", n_intercepted, n)
        # Replace hidden variables with uniform random (destroys correlations)
        modified = hidden_vars.copy()
        modified[intercept_mask] = self._rng.uniform(0, 2 * math.pi, size=n_intercepted)
        return modified

    # ------------------------------------------------------------------
    # Step 1-3 – Angle selection and measurement
    # ------------------------------------------------------------------

    def generate_measurements(self) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """Run the quantum measurement phase.

        Alice and Bob each independently choose random angles from their
        respective angle sets and measure their qubits.

        Returns:
            Tuple of:
                ``hidden_vars``    – hidden variables (n_pairs,)
                ``alice_angle_idx`` – Alice's chosen angle index (n_pairs,)
                ``bob_angle_idx``  – Bob's chosen angle index (n_pairs,)
                ``alice_outcomes`` – Alice's ±1 outcomes (n_pairs,)
                ``bob_outcomes``   – Bob's ±1 outcomes (n_pairs,)
        """
        hidden_vars = self._generate_bell_pairs()

        if self.eve_intercept_rate > 0:
            hidden_vars = self._eve_measure_and_resend(hidden_vars)

        n = self.n_pairs
        alice_angle_idx = self._rng.integers(0, len(self.alice_angles), size=n)
        bob_angle_idx = self._rng.integers(0, len(self.bob_angles), size=n)

        alice_chosen = np.array([self.alice_angles[i] for i in alice_angle_idx])
        bob_chosen = np.array([self.bob_angles[i] for i in bob_angle_idx])

        # Alice measures first: outcomes are random +/-1
        alice_outcomes = self._rng.choice([-1, 1], size=n)

        # For |Phi+>, the quantum correlation is <A B> = cos(alpha - beta).
        # When Eve intercepts, entanglement is destroyed and correlation is reduced.
        vis = max(0.0, 1.0 - self.eve_intercept_rate * 0.75)
        corr = vis * np.cos(alice_chosen - bob_chosen)
        p_agree = np.clip((1.0 + corr) / 2.0, 0.0, 1.0)
        rand = self._rng.random(n)
        bob_outcomes = np.where(rand < p_agree, alice_outcomes, -alice_outcomes)

        return hidden_vars, alice_angle_idx, bob_angle_idx, alice_outcomes, bob_outcomes

    # ------------------------------------------------------------------
    # Step 4-5 – Sifting: key subset vs. Bell-test subset
    # ------------------------------------------------------------------

    def sift_measurements(
        self,
        alice_angle_idx: np.ndarray,
        bob_angle_idx: np.ndarray,
        alice_outcomes: np.ndarray,
        bob_outcomes: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Separate measurements into key-generation and CHSH-test subsets.

        Key pairs: Alice and Bob chose identical measurement angles.
        Bell-test pairs: angle combinations forming the four CHSH terms.
        """
        alice_chosen = np.array([self.alice_angles[i] for i in alice_angle_idx])
        bob_chosen = np.array([self.bob_angles[i] for i in bob_angle_idx])
        key_mask = np.isclose(alice_chosen, bob_chosen)

        key_subsets = {
            "alice": alice_outcomes[key_mask],
            "bob": bob_outcomes[key_mask],
            "alice_angle_idx": alice_angle_idx[key_mask],
            "bob_angle_idx": bob_angle_idx[key_mask],
        }

        # CHSH subsets: 4 angle combinations
        chsh_masks: Dict[str, np.ndarray] = {
            "E_a0_b0": (alice_angle_idx == 0) & (bob_angle_idx == 0),
            "E_a0_b2": (alice_angle_idx == 0) & (bob_angle_idx == 2),
            "E_a2_b0": (alice_angle_idx == 2) & (bob_angle_idx == 0),
            "E_a2_b2": (alice_angle_idx == 2) & (bob_angle_idx == 2),
        }
        chsh_subsets: Dict[str, np.ndarray] = {
            key: {
                "alice": alice_outcomes[mask],
                "bob": bob_outcomes[mask],
            }
            for key, mask in chsh_masks.items()
        }

        logger.info(
            "Sifted %d key pairs and %d CHSH pairs.",
            key_mask.sum(),
            sum(m.sum() for m in chsh_masks.values()),
        )
        return key_subsets, chsh_subsets

    # ------------------------------------------------------------------
    # Step 5 – CHSH Bell test
    # ------------------------------------------------------------------

    def compute_chsh(
        self, chsh_subsets: Dict[str, Dict[str, np.ndarray]]
    ) -> Tuple[float, int, Dict[str, float]]:
        """Compute the CHSH parameter S from measurement data.

        S = E(a0,b0) − E(a0,b2) + E(a2,b0) + E(a2,b2)

        For the |Φ+⟩ state with E91 angles the theoretical value is 2√2.

        Args:
            chsh_subsets: Dict returned by :meth:`sift_measurements`.

        Returns:
            ``(S, n_pairs_used, correlators_dict)``
        """
        correlators: Dict[str, float] = {}
        total_pairs = 0

        for key, data in chsh_subsets.items():
            a = data["alice"]
            b = data["bob"]
            if len(a) == 0:
                correlators[key] = 0.0
                continue
            correlators[key] = float(np.mean(a * b))
            total_pairs += len(a)

        S = (
            correlators.get("E_a0_b0", 0.0)
            - correlators.get("E_a0_b2", 0.0)
            + correlators.get("E_a2_b0", 0.0)
            + correlators.get("E_a2_b2", 0.0)
        )
        logger.info("CHSH |S| = %.4f (threshold = %.4f).", abs(S), self.chsh_threshold)
        return abs(S), total_pairs, correlators

    # ------------------------------------------------------------------
    # Key extraction
    # ------------------------------------------------------------------

    def extract_key_bits(
        self, key_subsets: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert ±1 measurement outcomes to 0/1 key bits.

        For anti-correlated pairs (|Φ+⟩ + measurement convention), Alice's
        +1 corresponds to Bob's −1.  We map +1 → 1, −1 → 0.  Bob's bits
        are flipped so the raw keys agree.

        Args:
            key_subsets: Dict with ``'alice'`` and ``'bob'`` outcome arrays.

        Returns:
            ``(alice_bits, bob_bits)`` as integer arrays.
        """
        alice_outcomes = key_subsets["alice"]
        bob_outcomes = key_subsets["bob"]

        # Map ±1 → 0/1
        alice_bits = ((alice_outcomes + 1) // 2).astype(int)
        bob_bits = ((bob_outcomes + 1) // 2).astype(int)

        # For |Φ+⟩ measured at the same angle: perfect anti-correlation → flip Bob
        # (convention depends on Bell state; for Φ+ with same angles, outcomes agree)
        # We keep them as-is and compute QBER to detect any mismatch.
        return alice_bits, bob_bits

    def estimate_qber(
        self,
        alice_bits: np.ndarray,
        bob_bits: np.ndarray,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Estimate QBER by sacrificing a random key sample.

        Args:
            alice_bits: Alice's raw key bits.
            bob_bits:   Bob's raw key bits.

        Returns:
            ``(qber, alice_remaining, bob_remaining)``
        """
        n = len(alice_bits)
        sample_size = max(1, int(n * self.sample_fraction))
        idx = self._rng.choice(n, size=sample_size, replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[idx] = True

        errors = int((alice_bits[mask] != bob_bits[mask]).sum())
        qber = errors / sample_size if sample_size > 0 else 0.0
        logger.info("E91 QBER = %.4f (%d / %d).", qber, errors, sample_size)
        return qber, alice_bits[~mask], bob_bits[~mask]

    # ------------------------------------------------------------------
    # Privacy amplification
    # ------------------------------------------------------------------

    def privacy_amplification(
        self,
        key_bits: np.ndarray,
        security_bits: int = 64,
        output_length: Optional[int] = None,
    ) -> bytes:
        """Apply SHA-3 / SHAKE-256 hashing for privacy amplification.

        Args:
            key_bits:      Reconciled bit array.
            security_bits: Amount to shorten for leaked-information headroom.
            output_length: Final key length in bytes (default: auto).

        Returns:
            Final secret key bytes.
        """
        n = len(key_bits)
        if output_length is None:
            output_length = max(1, (n - security_bits) // 8)

        n_padded = math.ceil(n / 8) * 8
        padded = np.zeros(n_padded, dtype=np.uint8)
        padded[:n] = key_bits.astype(np.uint8)
        key_bytes = np.packbits(padded).tobytes()

        shake = hashlib.shake_256(key_bytes)
        final_key = shake.digest(output_length)
        logger.info(
            "E91 privacy amplification: %d bits → %d bytes.", n, output_length
        )
        return final_key

    # ------------------------------------------------------------------
    # Full protocol runner
    # ------------------------------------------------------------------

    def run_full_protocol(
        self,
        security_bits: int = 64,
        output_length: Optional[int] = None,
        eavesdrop: Optional[bool] = None,
    ) -> E91Result:
        """Execute the complete E91 protocol end-to-end.

        Args:
            security_bits: Privacy-amplification security parameter.
            output_length: Desired key length in bytes.
            eavesdrop:     Whether to simulate eavesdropping.

        Returns:
            :class:`E91Result` with the extracted key and all statistics.
        """
        if eavesdrop is not None:
            self.eve_intercept_rate = 1.0 if eavesdrop else 0.0

        result = E91Result(n_pairs=self.n_pairs)

        # ---- Quantum channel ----
        _, alice_angle_idx, bob_angle_idx, alice_outcomes, bob_outcomes = (
            self.generate_measurements()
        )

        # ---- Sifting ----
        key_subsets, chsh_subsets = self.sift_measurements(
            alice_angle_idx, bob_angle_idx, alice_outcomes, bob_outcomes
        )

        # ---- Bell test ----
        chsh_value, chsh_pairs, correlators = self.compute_chsh(chsh_subsets)
        result.chsh_value = chsh_value
        result.chsh_pairs_used = chsh_pairs

        eavesdropping = chsh_value < self.chsh_threshold
        result.eavesdropping_detected = eavesdropping

        if eavesdropping:
            logger.warning(
                "Bell test failed: |S| = %.4f < threshold %.4f. "
                "Eavesdropping detected; aborting.",
                chsh_value, self.chsh_threshold,
            )
            result.statistics["abort_reason"] = f"chsh_failed (|S|={chsh_value:.4f})"
            result.statistics["correlators"] = correlators
            return result

        # ---- Key extraction ----
        alice_bits, bob_bits = self.extract_key_bits(key_subsets)
        result.sifted_key_length = len(alice_bits)

        if result.sifted_key_length < 4:
            logger.error("Key subset too small; aborting.")
            result.eavesdropping_detected = True
            result.statistics["abort_reason"] = "key_subset_too_small"
            return result

        # ---- QBER check ----
        qber, alice_remain, bob_remain = self.estimate_qber(alice_bits, bob_bits)
        result.qber = qber

        # ---- Privacy amplification ----
        final_key = self.privacy_amplification(
            alice_remain, security_bits=security_bits, output_length=output_length
        )
        result.final_key = final_key
        result.final_key_length = len(final_key) * 8

        result.statistics.update({
            "chsh_value": chsh_value,
            "chsh_classical_bound": CHSH_CLASSICAL_BOUND,
            "chsh_quantum_bound": CHSH_QUANTUM_BOUND,
            "correlators": correlators,
            "qber": qber,
            "key_rate": result.final_key_length / self.n_pairs,
        })

        logger.info(
            "E91 complete. |S| = %.4f. Final key: %d bytes.",
            chsh_value, len(final_key),
        )
        return result

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def theoretical_chsh(
        self,
        alice_angle: float,
        bob_angle: float,
    ) -> float:
        """Return the theoretical correlation E(α, β) = −cos(α − β).

        Args:
            alice_angle: Alice's measurement angle (radians).
            bob_angle:   Bob's measurement angle (radians).

        Returns:
            Expected correlation coefficient in [-1, 1].
        """
        return -math.cos(alice_angle - bob_angle)

    def theoretical_s_value(self) -> float:
        """Compute the ideal CHSH S value for the configured angle sets.

        Uses the E91 standard angles to compute S analytically.

        Returns:
            Theoretical |S| value.
        """
        a1, _, a3 = self.alice_angles
        b1, _, b3 = self.bob_angles
        S = (
            self.theoretical_chsh(a1, b1)
            - self.theoretical_chsh(a1, b3)
            + self.theoretical_chsh(a3, b1)
            + self.theoretical_chsh(a3, b3)
        )
        return abs(S)
