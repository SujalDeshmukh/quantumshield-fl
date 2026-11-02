"""
Quantum Data Encoders for QuantumShield-FL.

This module provides classical-to-quantum state encoding strategies used to
embed classical feature vectors into quantum state representations, enabling
their processing by quantum circuits.

Supported encoding strategies:
    - Amplitude encoding  : features as probability amplitudes
    - Angle encoding      : features as rotation angles (Rx / Ry / Rz gates)
    - IQP encoding        : Instantaneous Quantum Polynomial circuits
    - Basis encoding      : binary integers mapped to computational basis states
"""

from __future__ import annotations

import math
from typing import List, Literal, Optional, Sequence, Union

import numpy as np


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Array = np.ndarray
EncodingType = Literal["amplitude", "angle", "iqp", "basis"]


# ---------------------------------------------------------------------------
# Low-level gate helpers (return 2x2 numpy matrices)
# ---------------------------------------------------------------------------

def _rx(theta: float) -> Array:
    """Rotation-X gate matrix."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _ry(theta: float) -> Array:
    """Rotation-Y gate matrix."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(theta: float) -> Array:
    """Rotation-Z gate matrix."""
    e_neg = math.exp(-1j * theta / 2)
    e_pos = math.exp(1j * theta / 2)
    return np.array([[e_neg, 0], [0, e_pos]], dtype=complex)


def _hadamard() -> Array:
    """Hadamard gate matrix."""
    return np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)


def _cnot() -> Array:
    """CNOT gate (4x4) for two-qubit systems."""
    return np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1],
         [0, 0, 1, 0]],
        dtype=complex,
    )


def _kron_n(*matrices: Array) -> Array:
    """Compute the Kronecker (tensor) product of an arbitrary sequence of matrices."""
    result = matrices[0]
    for m in matrices[1:]:
        result = np.kron(result, m)
    return result


def _single_qubit_unitary(gate: Array, qubit: int, n_qubits: int) -> Array:
    """
    Embed a single-qubit gate into the full n-qubit Hilbert space.

    Parameters
    ----------
    gate : Array
        2×2 unitary matrix.
    qubit : int
        Index of the target qubit (0 = most significant).
    n_qubits : int
        Total number of qubits.

    Returns
    -------
    Array
        2**n_qubits × 2**n_qubits unitary matrix.
    """
    ops = [np.eye(2, dtype=complex)] * n_qubits
    ops[qubit] = gate
    return _kron_n(*ops)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class QuantumDataEncoder:
    """
    Encodes classical feature vectors into quantum state vectors.

    This class provides four standard encoding strategies used in quantum
    machine learning. All methods operate purely with NumPy and return
    complex-valued state vectors of dimension 2**n_qubits.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the target quantum register.
    encoding : EncodingType, optional
        Default encoding strategy (used by ``encode``). One of
        ``"amplitude"``, ``"angle"``, ``"iqp"``, ``"basis"``.
        Defaults to ``"angle"``.
    rotation_axis : str, optional
        Rotation axis used in angle encoding: ``"X"``, ``"Y"``, or ``"Z"``.
        Defaults to ``"Y"``.
    normalize : bool, optional
        If ``True`` (default), feature vectors are normalised before
        amplitude encoding.

    Examples
    --------
    >>> enc = QuantumDataEncoder(n_qubits=3)
    >>> x = np.array([0.1, 0.5, 0.9])
    >>> state = enc.encode_angle(x)
    >>> state.shape
    (8,)
    """

    def __init__(
        self,
        n_qubits: int,
        encoding: EncodingType = "angle",
        rotation_axis: str = "Y",
        normalize: bool = True,
    ) -> None:
        if n_qubits < 1:
            raise ValueError("n_qubits must be >= 1.")
        if rotation_axis.upper() not in ("X", "Y", "Z"):
            raise ValueError("rotation_axis must be 'X', 'Y', or 'Z'.")

        self.n_qubits: int = n_qubits
        self.encoding: EncodingType = encoding
        self.rotation_axis: str = rotation_axis.upper()
        self.normalize: bool = normalize
        self.dim: int = 2 ** n_qubits

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, x: Array) -> Array:
        """
        Encode a single feature vector using the configured strategy.

        Parameters
        ----------
        x : Array
            1-D feature vector.

        Returns
        -------
        Array
            Complex state vector of shape ``(2**n_qubits,)``.
        """
        dispatch = {
            "amplitude": self.encode_amplitude,
            "angle":     self.encode_angle,
            "iqp":       self.encode_iqp,
            "basis":     self.encode_basis,
        }
        return dispatch[self.encoding](x)

    def encode_batch(
        self,
        X: Array,
        encoding: Optional[EncodingType] = None,
    ) -> Array:
        """
        Encode a batch of feature vectors.

        Parameters
        ----------
        X : Array
            2-D array of shape ``(n_samples, n_features)``.
        encoding : EncodingType, optional
            Override the instance-level encoding strategy for this call.

        Returns
        -------
        Array
            Complex array of shape ``(n_samples, 2**n_qubits)``.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        strategy = encoding or self.encoding
        dispatch = {
            "amplitude": self.encode_amplitude,
            "angle":     self.encode_angle,
            "iqp":       self.encode_iqp,
            "basis":     self.encode_basis,
        }
        fn = dispatch[strategy]
        return np.stack([fn(x) for x in X], axis=0)

    # ------------------------------------------------------------------
    # Amplitude encoding
    # ------------------------------------------------------------------

    def encode_amplitude(self, x: Array) -> Array:
        """
        Amplitude encoding: map a real vector to a quantum state's amplitudes.

        The input vector is padded with zeros to length ``2**n_qubits`` and
        then normalised so its L2 norm equals 1.  The resulting state is:

            |ψ⟩ = Σ_i  (x_i / ‖x‖)  |i⟩

        Parameters
        ----------
        x : Array
            1-D real feature vector of length ≤ ``2**n_qubits``.

        Returns
        -------
        Array
            Normalised complex state vector of shape ``(2**n_qubits,)``.

        Raises
        ------
        ValueError
            If the input vector is longer than ``2**n_qubits``.
        """
        x = np.asarray(x, dtype=float).ravel()
        if x.size > self.dim:
            raise ValueError(
                f"Input length {x.size} exceeds 2**n_qubits = {self.dim}. "
                "Increase n_qubits or truncate the feature vector."
            )
        # Zero-pad to Hilbert space dimension
        padded = np.zeros(self.dim, dtype=float)
        padded[: x.size] = x

        norm = np.linalg.norm(padded)
        if norm < 1e-12:
            raise ValueError("Feature vector has zero norm; cannot encode as amplitudes.")

        state = (padded / norm).astype(complex)
        return state

    # ------------------------------------------------------------------
    # Angle encoding
    # ------------------------------------------------------------------

    def encode_angle(self, x: Array) -> Array:
        """
        Angle encoding: each feature is mapped to a single-qubit rotation angle.

        The circuit applies R_axis(x_i * π) to each qubit i, where R_axis is
        selected by ``self.rotation_axis``.  Features are scaled to [0, π]
        via an optional min-max normalisation.

        Parameters
        ----------
        x : Array
            1-D feature vector.  Length must equal ``n_qubits`` (features
            exceeding this are ignored; missing features are zero-padded).

        Returns
        -------
        Array
            Complex state vector of shape ``(2**n_qubits,)``.
        """
        x = np.asarray(x, dtype=float).ravel()
        # Trim or pad to n_qubits
        features = np.zeros(self.n_qubits, dtype=float)
        n = min(x.size, self.n_qubits)
        features[:n] = x[:n]

        # Scale features to [0, π]
        if self.normalize:
            f_min, f_max = features.min(), features.max()
            if abs(f_max - f_min) > 1e-12:
                features = (features - f_min) / (f_max - f_min) * math.pi

        gate_fn = {"X": _rx, "Y": _ry, "Z": _rz}[self.rotation_axis]

        # Start in |0…0⟩
        state = np.zeros(self.dim, dtype=complex)
        state[0] = 1.0

        for qubit, angle in enumerate(features):
            U = _single_qubit_unitary(gate_fn(angle), qubit, self.n_qubits)
            state = U @ state

        return state

    # ------------------------------------------------------------------
    # IQP encoding
    # ------------------------------------------------------------------

    def encode_iqp(self, x: Array) -> Array:
        """
        IQP (Instantaneous Quantum Polynomial) encoding.

        Implements the two-layer IQP circuit:
            1. Apply Hadamard to every qubit.
            2. Apply diagonal phase gates exp(i * x_i * Z) on each qubit.
            3. Apply cross-terms exp(i * x_i * x_j * ZZ) for neighbouring pairs.
            4. Repeat layers (default: 1 repetition).

        This creates a feature map whose quantum kernel is classically
        hard to simulate for large qubit counts.

        Parameters
        ----------
        x : Array
            1-D feature vector.  Length must equal ``n_qubits``.

        Returns
        -------
        Array
            Complex state vector of shape ``(2**n_qubits,)``.
        """
        x = np.asarray(x, dtype=float).ravel()
        features = np.zeros(self.n_qubits, dtype=float)
        n = min(x.size, self.n_qubits)
        features[:n] = x[:n]

        state = np.zeros(self.dim, dtype=complex)
        state[0] = 1.0

        H = _hadamard()

        # Layer: Hadamard on all qubits
        for q in range(self.n_qubits):
            U = _single_qubit_unitary(H, q, self.n_qubits)
            state = U @ state

        # Layer: single-qubit Z phase gates  exp(i * x_i * Z / 2)
        for q in range(self.n_qubits):
            angle = features[q]
            phase_gate = np.array(
                [[np.exp(1j * angle / 2), 0],
                 [0, np.exp(-1j * angle / 2)]],
                dtype=complex,
            )
            U = _single_qubit_unitary(phase_gate, q, self.n_qubits)
            state = U @ state

        # Layer: two-qubit ZZ interaction on neighbouring pairs
        for q in range(self.n_qubits - 1):
            angle = features[q] * features[q + 1]
            state = self._apply_zz_rotation(state, q, q + 1, angle)

        # Second Hadamard layer
        for q in range(self.n_qubits):
            U = _single_qubit_unitary(H, q, self.n_qubits)
            state = U @ state

        return state

    # ------------------------------------------------------------------
    # Basis encoding
    # ------------------------------------------------------------------

    def encode_basis(self, x: Array) -> Array:
        """
        Basis (binary) encoding: convert a binary integer to a computational
        basis state.

        The input vector is interpreted as a binary string (thresholded at
        0.5) and mapped to the corresponding basis state |b_{n-1} … b_0⟩.

        Parameters
        ----------
        x : Array
            1-D binary (or real-valued, will be binarised) feature vector of
            length ≤ ``n_qubits``.

        Returns
        -------
        Array
            Complex state vector with a single 1 at the encoded basis index.
        """
        x = np.asarray(x, dtype=float).ravel()
        bits = np.zeros(self.n_qubits, dtype=int)
        n = min(x.size, self.n_qubits)
        bits[:n] = (x[:n] >= 0.5).astype(int)

        # Convert bit string to integer index (big-endian: bit[0] = MSB)
        index = int("".join(str(b) for b in bits), 2)

        state = np.zeros(self.dim, dtype=complex)
        state[index] = 1.0
        return state

    # ------------------------------------------------------------------
    # Utility / information methods
    # ------------------------------------------------------------------

    def state_to_density_matrix(self, state: Array) -> Array:
        """
        Convert a pure state vector to a density matrix ρ = |ψ⟩⟨ψ|.

        Parameters
        ----------
        state : Array
            Complex state vector of shape ``(2**n_qubits,)``.

        Returns
        -------
        Array
            Density matrix of shape ``(2**n_qubits, 2**n_qubits)``.
        """
        state = np.asarray(state).ravel()
        return np.outer(state, state.conj())

    def expectation_value(self, state: Array, observable: Array) -> float:
        """
        Compute ⟨ψ|O|ψ⟩ for a given Hermitian observable O.

        Parameters
        ----------
        state : Array
            Complex state vector of shape ``(2**n_qubits,)``.
        observable : Array
            Hermitian matrix of shape ``(2**n_qubits, 2**n_qubits)``.

        Returns
        -------
        float
            Real expectation value.
        """
        state = np.asarray(state, dtype=complex).ravel()
        obs   = np.asarray(observable, dtype=complex)
        ev = state.conj() @ obs @ state
        return float(np.real(ev))

    def fidelity(self, state_a: Array, state_b: Array) -> float:
        """
        Compute the fidelity |⟨ψ_a|ψ_b⟩|² between two pure states.

        Parameters
        ----------
        state_a, state_b : Array
            Complex state vectors.

        Returns
        -------
        float
            Fidelity in [0, 1].
        """
        a = np.asarray(state_a, dtype=complex).ravel()
        b = np.asarray(state_b, dtype=complex).ravel()
        return float(np.abs(np.dot(a.conj(), b)) ** 2)

    def describe(self) -> dict:
        """
        Return a dict summarising the encoder configuration.

        Returns
        -------
        dict
            Keys: ``n_qubits``, ``hilbert_dim``, ``encoding``,
            ``rotation_axis``, ``normalize``.
        """
        return {
            "n_qubits":      self.n_qubits,
            "hilbert_dim":   self.dim,
            "encoding":      self.encoding,
            "rotation_axis": self.rotation_axis,
            "normalize":     self.normalize,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_zz_rotation(
        self,
        state: Array,
        qubit_a: int,
        qubit_b: int,
        angle: float,
    ) -> Array:
        """
        Apply exp(i * angle * Z_a ⊗ Z_b) to a state vector.

        This is a diagonal operation: computational basis states acquire
        phases depending on the parity of the two target qubits.

        Parameters
        ----------
        state : Array
            Current complex state vector.
        qubit_a, qubit_b : int
            Indices of the two qubits.
        angle : float
            Rotation angle.

        Returns
        -------
        Array
            Updated state vector.
        """
        new_state = state.copy()
        mask_a = 1 << (self.n_qubits - 1 - qubit_a)
        mask_b = 1 << (self.n_qubits - 1 - qubit_b)
        for idx in range(self.dim):
            bit_a = 1 if (idx & mask_a) else -1
            bit_b = 1 if (idx & mask_b) else -1
            phase = np.exp(1j * angle * bit_a * bit_b)
            new_state[idx] = state[idx] * phase
        return new_state
