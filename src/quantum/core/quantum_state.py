"""Quantum State representation and manipulation."""
from __future__ import annotations
import numpy as np
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class QuantumState:
    """Represents a pure quantum state as a complex statevector.

    Supports up to ~20 qubits via statevector simulation.

    Attributes:
        n_qubits: Number of qubits in the system.
        dim: Hilbert space dimension (2 ** n_qubits).
    """

    def __init__(self, n_qubits: int, state: Optional[np.ndarray] = None):
        """Initialise a quantum state.

        Args:
            n_qubits: Number of qubits.
            state: Optional complex statevector of length 2**n_qubits.
                   Automatically normalised.  Defaults to |0...0⟩.
        """
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        if state is not None:
            self._state = np.array(state, dtype=complex)
            self._normalize()
        else:
            self._state = np.zeros(self.dim, dtype=complex)
            self._state[0] = 1.0  # |0...0⟩ ground state

    # ------------------------------------------------------------------
    # Factory constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_bloch(cls, theta: float, phi: float) -> "QuantumState":
        """Create a single-qubit state from Bloch sphere coordinates.

        Args:
            theta: Polar angle in [0, π].
            phi:   Azimuthal angle in [0, 2π].

        Returns:
            Single-qubit QuantumState on the Bloch sphere.
        """
        s = cls(1)
        s._state = np.array([
            np.cos(theta / 2),
            np.exp(1j * phi) * np.sin(theta / 2)
        ], dtype=complex)
        return s

    @classmethod
    def bell_state(cls, variant: int = 0) -> "QuantumState":
        """Create one of the four Bell (maximally entangled) states.

        Args:
            variant:
                0  →  |Φ+⟩ = (|00⟩ + |11⟩) / √2
                1  →  |Φ-⟩ = (|00⟩ − |11⟩) / √2
                2  →  |Ψ+⟩ = (|01⟩ + |10⟩) / √2
                3  →  |Ψ-⟩ = (|01⟩ − |10⟩) / √2

        Returns:
            Two-qubit QuantumState.

        Raises:
            ValueError: If variant is not in {0, 1, 2, 3}.
        """
        s = cls(2)
        sqrt2 = 1.0 / np.sqrt(2)
        if variant == 0:
            s._state = np.array([sqrt2, 0, 0, sqrt2], dtype=complex)
        elif variant == 1:
            s._state = np.array([sqrt2, 0, 0, -sqrt2], dtype=complex)
        elif variant == 2:
            s._state = np.array([0, sqrt2, sqrt2, 0], dtype=complex)
        elif variant == 3:
            s._state = np.array([0, sqrt2, -sqrt2, 0], dtype=complex)
        else:
            raise ValueError(f"Invalid Bell state variant: {variant}")
        return s

    @classmethod
    def ghz_state(cls, n_qubits: int) -> "QuantumState":
        """Create the Greenberger–Horne–Zeilinger (GHZ) state.

        |GHZ⟩ = (|0...0⟩ + |1...1⟩) / √2

        Args:
            n_qubits: Number of qubits (≥ 2).

        Returns:
            n_qubits-qubit QuantumState in the GHZ configuration.
        """
        s = cls(n_qubits)
        s._state = np.zeros(2 ** n_qubits, dtype=complex)
        s._state[0] = 1.0 / np.sqrt(2)
        s._state[-1] = 1.0 / np.sqrt(2)
        return s

    @classmethod
    def w_state(cls, n_qubits: int) -> "QuantumState":
        """Create the W state: equal superposition of all single-excitation states.

        |W⟩ = (|100...0⟩ + |010...0⟩ + ... + |000...1⟩) / √n

        Args:
            n_qubits: Number of qubits (≥ 2).

        Returns:
            n_qubits-qubit QuantumState in the W configuration.
        """
        s = cls(n_qubits)
        s._state = np.zeros(2 ** n_qubits, dtype=complex)
        amp = 1.0 / np.sqrt(n_qubits)
        for k in range(n_qubits):
            idx = 1 << k  # state with exactly the k-th qubit set
            s._state[idx] = amp
        return s

    @classmethod
    def uniform_superposition(cls, n_qubits: int) -> "QuantumState":
        """Create the uniform superposition |+...+⟩ = H⊗n |0...0⟩.

        Args:
            n_qubits: Number of qubits.

        Returns:
            n_qubits-qubit QuantumState in uniform superposition.
        """
        s = cls(n_qubits)
        s._state = np.ones(2 ** n_qubits, dtype=complex) / np.sqrt(2 ** n_qubits)
        return s

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize(self) -> None:
        """Normalise the statevector in-place."""
        norm = np.linalg.norm(self._state)
        if norm > 1e-12:
            self._state /= norm
        else:
            logger.warning("Attempted to normalise a near-zero statevector.")

    def _build_full_operator(
        self, gate: np.ndarray, target_qubits: List[int]
    ) -> np.ndarray:
        """Build the full 2^n × 2^n unitary for a gate acting on target_qubits.

        For single-qubit gates the operator is placed at each target qubit
        position via Kronecker products.  For multi-qubit gates (e.g. CNOT)
        the qubits are assumed to be consecutive and in the order given.

        Args:
            gate: Unitary matrix of size 2^k × 2^k, where k = len(target_qubits).
            target_qubits: Ordered list of qubit indices on which the gate acts.

        Returns:
            Full 2^n × 2^n unitary operator.
        """
        n = self.n_qubits
        gate_size = gate.shape[0]
        k = int(np.round(np.log2(gate_size)))  # qubits consumed by this gate

        if len(target_qubits) == 1 and k == 1:
            # Efficient single-qubit embedding
            target = target_qubits[0]
            op = np.eye(1, dtype=complex)
            for qubit in range(n):
                if qubit == target:
                    op = np.kron(op, gate)
                else:
                    op = np.kron(op, np.eye(2, dtype=complex))
            return op

        # General multi-qubit gate: permute qubits, apply gate, permute back
        # We reorder qubits so target_qubits are first, apply gate ⊗ I, then undo.
        all_qubits = list(range(n))
        other_qubits = [q for q in all_qubits if q not in target_qubits]
        perm = target_qubits + other_qubits          # target qubits first
        perm_inv = [perm.index(i) for i in all_qubits]  # inverse permutation

        # Permutation matrix
        P = self._permutation_matrix(perm, n)
        P_inv = P.T  # unitary → inverse = transpose-conjugate (real permutation)

        gate_full = np.kron(gate, np.eye(2 ** len(other_qubits), dtype=complex))
        return P_inv @ gate_full @ P

    @staticmethod
    def _permutation_matrix(perm: List[int], n: int) -> np.ndarray:
        """Build a 2^n × 2^n qubit-permutation unitary.

        Maps computational basis state |q_0 q_1 ... q_{n-1}⟩ to
        |q_{perm[0]} q_{perm[1]} ... q_{perm[n-1]}⟩.

        Args:
            perm: Permutation of range(n).
            n:    Total number of qubits.

        Returns:
            2^n × 2^n complex permutation matrix.
        """
        dim = 2 ** n
        P = np.zeros((dim, dim), dtype=complex)
        for col in range(dim):
            # Decompose col into individual qubit bits (MSB = qubit 0)
            bits = [(col >> (n - 1 - q)) & 1 for q in range(n)]
            # Re-assemble according to permutation
            new_bits = [bits[perm[q]] for q in range(n)]
            row = sum(new_bits[q] << (n - 1 - q) for q in range(n))
            P[row, col] = 1.0
        return P

    # ------------------------------------------------------------------
    # Gate application
    # ------------------------------------------------------------------

    def apply_gate(
        self, gate: np.ndarray, target_qubits: List[int]
    ) -> "QuantumState":
        """Apply a unitary gate to specified target qubits.

        Args:
            gate: Unitary matrix (size must match 2^len(target_qubits)).
            target_qubits: Qubit indices on which the gate acts.

        Returns:
            New QuantumState after gate application.
        """
        full_op = self._build_full_operator(gate, target_qubits)
        new_state = full_op @ self._state
        return QuantumState(self.n_qubits, new_state)

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def measure(
        self, qubit: Optional[int] = None
    ) -> Tuple[int, "QuantumState"]:
        """Perform a projective measurement in the computational basis.

        Args:
            qubit: If given, measure only that qubit index and collapse the
                   state accordingly.  If None, measure all qubits at once.

        Returns:
            Tuple (outcome, collapsed_state) where outcome is an integer
            bit-string index (full measurement) or a single bit (0 or 1).
        """
        probs = np.abs(self._state) ** 2
        # Numerical fix: ensure probs sum to exactly 1
        probs /= probs.sum()

        if qubit is None:
            outcome = int(np.random.choice(self.dim, p=probs))
            collapsed = np.zeros(self.dim, dtype=complex)
            collapsed[outcome] = 1.0
            return outcome, QuantumState(self.n_qubits, collapsed)

        # Single-qubit measurement
        prob_0 = float(sum(
            probs[i] for i in range(self.dim)
            if not (i >> (self.n_qubits - 1 - qubit)) & 1
        ))
        outcome = 0 if np.random.random() < prob_0 else 1

        # Collapse: zero out all amplitudes inconsistent with outcome
        new_state = np.copy(self._state)
        for i in range(self.dim):
            bit = (i >> (self.n_qubits - 1 - qubit)) & 1
            if bit != outcome:
                new_state[i] = 0.0
        norm = np.linalg.norm(new_state)
        if norm > 1e-12:
            new_state /= norm
        return outcome, QuantumState(self.n_qubits, new_state)

    def measure_all(self, shots: int = 1024) -> dict:
        """Sample the state distribution for multiple shots.

        Args:
            shots: Number of measurement shots.

        Returns:
            Dictionary mapping binary outcome strings to counts.
        """
        probs = np.abs(self._state) ** 2
        probs /= probs.sum()
        outcomes = np.random.choice(self.dim, size=shots, p=probs)
        counts: dict = {}
        for o in outcomes:
            key = format(o, f"0{self.n_qubits}b")
            counts[key] = counts.get(key, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Information-theoretic quantities
    # ------------------------------------------------------------------

    def partial_trace(self, keep_qubits: List[int]) -> np.ndarray:
        """Compute the reduced density matrix by tracing out all other qubits.

        Args:
            keep_qubits: Qubit indices to retain.

        Returns:
            Dense complex density matrix of size 2^len(keep_qubits).
        """
        n = self.n_qubits
        keep = sorted(keep_qubits)
        trace_out = sorted([q for q in range(n) if q not in keep])

        # Reshape statevector into tensor: axes [q0, q1, ..., q_{n-1}]
        psi = self._state.reshape([2] * n)

        # Contract over traced-out qubits
        # Build density matrix as outer product, then reshape and trace
        rho = np.outer(self._state, self._state.conj()).reshape([2] * (2 * n))

        # Trace out each qubit (in reverse order to keep axis labels stable)
        shift = 0
        for q in trace_out:
            ax1 = q - shift
            ax2 = ax1 + (n - shift)
            # np.trace over axis1, axis2 → reduces tensor rank by 2
            rho = np.trace(rho, axis1=ax1, axis2=ax2)
            shift += 1

        k = len(keep)
        return rho.reshape(2 ** k, 2 ** k)

    def fidelity(self, other: "QuantumState") -> float:
        """Compute state fidelity |⟨ψ|φ⟩|².

        Args:
            other: Another QuantumState with the same number of qubits.

        Returns:
            Fidelity value in [0, 1].
        """
        return float(np.abs(np.dot(self._state.conj(), other._state)) ** 2)

    def entropy(self) -> float:
        """Von Neumann entropy.

        Returns:
            0.0 — always zero for pure states.
        """
        return 0.0

    def entanglement_entropy(self, partition: List[int]) -> float:
        """Bipartite entanglement entropy (von Neumann) across a qubit partition.

        Args:
            partition: List of qubit indices forming subsystem A.

        Returns:
            Entanglement entropy in bits.
        """
        rho_A = self.partial_trace(partition)
        eigenvalues = np.linalg.eigvalsh(rho_A)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        return float(-np.sum(eigenvalues * np.log2(eigenvalues)))

    def expectation_value(self, observable: np.ndarray) -> complex:
        """Compute ⟨ψ|O|ψ⟩ for a given observable O.

        Args:
            observable: Hermitian matrix of size 2^n × 2^n.

        Returns:
            Complex expectation value (imaginary part should be ≈ 0 for
            Hermitian observables).
        """
        return complex(self._state.conj() @ observable @ self._state)

    def bloch_vector(self) -> Tuple[float, float, float]:
        """Return the Bloch vector (rx, ry, rz) for a single-qubit state.

        Raises:
            ValueError: If the state has more than one qubit.

        Returns:
            Tuple (rx, ry, rz) with each component in [-1, 1].
        """
        if self.n_qubits != 1:
            raise ValueError("Bloch vector is only defined for single-qubit states.")
        alpha, beta = self._state
        rx = float(2 * np.real(np.conj(alpha) * beta))
        ry = float(2 * np.imag(np.conj(alpha) * beta))
        rz = float(np.abs(alpha) ** 2 - np.abs(beta) ** 2)
        return rx, ry, rz

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def statevector(self) -> np.ndarray:
        """Return a copy of the statevector."""
        return self._state.copy()

    @property
    def density_matrix(self) -> np.ndarray:
        """Return the density matrix |ψ⟩⟨ψ| as a 2^n × 2^n array."""
        return np.outer(self._state, self._state.conj())

    def __repr__(self) -> str:
        return f"QuantumState(n_qubits={self.n_qubits}, dim={self.dim})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QuantumState):
            return NotImplemented
        return (
            self.n_qubits == other.n_qubits
            and np.allclose(self._state, other._state)
        )
