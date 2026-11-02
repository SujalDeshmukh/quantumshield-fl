"""Parameterised Quantum Circuit Builder.

Builds, validates, and simulates quantum circuits using NumPy gate matrices
together with the :class:`~quantum.core.quantum_state.QuantumState`
statevector simulator.

Supported gates
---------------
Single-qubit : I, H, X, Y, Z, S, T, Sdg, Tdg, RX, RY, RZ, P (phase)
Two-qubit    : CNOT (CX), CY, CZ, SWAP, iSWAP, CH
Three-qubit  : Toffoli (CCX), Fredkin (CSWAP)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .quantum_state import QuantumState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gate library – all gates returned as complex NumPy arrays
# ---------------------------------------------------------------------------

_I = np.eye(2, dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_S = np.array([[1, 0], [0, 1j]], dtype=complex)
_T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
_Sdg = _S.conj().T
_Tdg = _T.conj().T
_CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
], dtype=complex)
_CZ = np.array([
    [1, 0, 0,  0],
    [0, 1, 0,  0],
    [0, 0, 1,  0],
    [0, 0, 0, -1],
], dtype=complex)
_SWAP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
], dtype=complex)
_iSWAP = np.array([
    [1,  0,  0, 0],
    [0,  0, 1j, 0],
    [0, 1j,  0, 0],
    [0,  0,  0, 1],
], dtype=complex)
_Toffoli = np.eye(8, dtype=complex)
_Toffoli[6, 6] = 0; _Toffoli[6, 7] = 1
_Toffoli[7, 7] = 0; _Toffoli[7, 6] = 1
_Fredkin = np.eye(8, dtype=complex)
_Fredkin[5, 5] = 0; _Fredkin[5, 6] = 1
_Fredkin[6, 6] = 0; _Fredkin[6, 5] = 1


def _rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(theta: float) -> np.ndarray:
    return np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)],
    ], dtype=complex)


def _phase(lam: float) -> np.ndarray:
    return np.array([[1, 0], [0, np.exp(1j * lam)]], dtype=complex)


def _u3(theta: float, phi: float, lam: float) -> np.ndarray:
    """General single-qubit unitary (IBM U3 gate)."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([
        [c, -np.exp(1j * lam) * s],
        [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c],
    ], dtype=complex)


def _cy() -> np.ndarray:
    """Controlled-Y gate."""
    return np.array([
        [1, 0,  0,  0],
        [0, 1,  0,  0],
        [0, 0,  0, -1j],
        [0, 0, 1j,  0],
    ], dtype=complex)


def _ch() -> np.ndarray:
    """Controlled-Hadamard gate."""
    m = np.eye(4, dtype=complex)
    m[2, 2] = 1 / np.sqrt(2)
    m[2, 3] = 1 / np.sqrt(2)
    m[3, 2] = 1 / np.sqrt(2)
    m[3, 3] = -1 / np.sqrt(2)
    return m


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GateInstruction:
    """A single gate instruction within a circuit."""

    gate_name: str
    """Human-readable gate name (e.g. 'H', 'CNOT', 'RX')."""

    qubits: List[int]
    """Qubit indices on which the gate acts (ordered)."""

    matrix: np.ndarray
    """Unitary matrix for the gate."""

    params: Dict[str, float] = field(default_factory=dict)
    """Symbolic or numeric parameters, e.g. {'theta': 1.57}."""

    layer: int = 0
    """Circuit layer / depth level this instruction belongs to."""

    def __repr__(self) -> str:
        p = ", ".join(f"{k}={v:.4f}" for k, v in self.params.items())
        param_str = f"({p})" if p else ""
        return f"{self.gate_name}{param_str} @ qubits={self.qubits}"


# ---------------------------------------------------------------------------
# QuantumCircuitBuilder
# ---------------------------------------------------------------------------

class QuantumCircuitBuilder:
    """Builds and simulates parameterised quantum circuits.

    The builder maintains an ordered list of :class:`GateInstruction` objects.
    Call :meth:`run` to execute the circuit on an initial state.

    Example
    -------
    >>> cb = QuantumCircuitBuilder(n_qubits=2)
    >>> cb.h(0).cnot(0, 1)
    >>> state = cb.run()
    >>> counts = state.measure_all(shots=1024)

    Parameters
    ----------
    n_qubits:
        Number of qubits in the circuit.
    noise_model:
        Optional callable ``noise_model(state, gate) → state`` applied after
        every gate.  Useful for depolarising/dephasing noise simulation.
    """

    def __init__(
        self,
        n_qubits: int,
        noise_model: Optional[Callable[[QuantumState, GateInstruction], QuantumState]] = None,
    ):
        if n_qubits < 1:
            raise ValueError("n_qubits must be ≥ 1.")
        self.n_qubits = n_qubits
        self._noise_model = noise_model
        self._instructions: List[GateInstruction] = []
        self._current_layer: int = 0
        # Track which layer each qubit was last written to (for depth calc)
        self._qubit_layer: List[int] = [0] * n_qubits

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add(
        self,
        gate_name: str,
        qubits: List[int],
        matrix: np.ndarray,
        params: Optional[Dict[str, float]] = None,
    ) -> "QuantumCircuitBuilder":
        """Register a gate instruction and advance the layer counter."""
        self._validate_qubits(qubits)
        layer = max(self._qubit_layer[q] for q in qubits)
        instr = GateInstruction(
            gate_name=gate_name,
            qubits=qubits,
            matrix=matrix,
            params=params or {},
            layer=layer,
        )
        self._instructions.append(instr)
        for q in qubits:
            self._qubit_layer[q] = layer + 1
        return self

    def _validate_qubits(self, qubits: List[int]) -> None:
        for q in qubits:
            if not (0 <= q < self.n_qubits):
                raise ValueError(
                    f"Qubit index {q} is out of range for {self.n_qubits}-qubit circuit."
                )
        if len(qubits) != len(set(qubits)):
            raise ValueError(f"Duplicate qubit indices: {qubits}")

    # ------------------------------------------------------------------
    # Single-qubit gates
    # ------------------------------------------------------------------

    def i(self, qubit: int) -> "QuantumCircuitBuilder":
        """Identity gate."""
        return self._add("I", [qubit], _I.copy())

    def h(self, qubit: int) -> "QuantumCircuitBuilder":
        """Hadamard gate."""
        return self._add("H", [qubit], _H.copy())

    def x(self, qubit: int) -> "QuantumCircuitBuilder":
        """Pauli-X (NOT) gate."""
        return self._add("X", [qubit], _X.copy())

    def y(self, qubit: int) -> "QuantumCircuitBuilder":
        """Pauli-Y gate."""
        return self._add("Y", [qubit], _Y.copy())

    def z(self, qubit: int) -> "QuantumCircuitBuilder":
        """Pauli-Z gate."""
        return self._add("Z", [qubit], _Z.copy())

    def s(self, qubit: int) -> "QuantumCircuitBuilder":
        """S (√Z) gate."""
        return self._add("S", [qubit], _S.copy())

    def t(self, qubit: int) -> "QuantumCircuitBuilder":
        """T (π/8) gate."""
        return self._add("T", [qubit], _T.copy())

    def sdg(self, qubit: int) -> "QuantumCircuitBuilder":
        """S-dagger gate."""
        return self._add("Sdg", [qubit], _Sdg.copy())

    def tdg(self, qubit: int) -> "QuantumCircuitBuilder":
        """T-dagger gate."""
        return self._add("Tdg", [qubit], _Tdg.copy())

    def rx(self, theta: float, qubit: int) -> "QuantumCircuitBuilder":
        """Rotation around X-axis by angle theta."""
        return self._add("RX", [qubit], _rx(theta), {"theta": theta})

    def ry(self, theta: float, qubit: int) -> "QuantumCircuitBuilder":
        """Rotation around Y-axis by angle theta."""
        return self._add("RY", [qubit], _ry(theta), {"theta": theta})

    def rz(self, theta: float, qubit: int) -> "QuantumCircuitBuilder":
        """Rotation around Z-axis by angle theta."""
        return self._add("RZ", [qubit], _rz(theta), {"theta": theta})

    def p(self, lam: float, qubit: int) -> "QuantumCircuitBuilder":
        """Phase gate P(λ) = diag(1, e^{iλ})."""
        return self._add("P", [qubit], _phase(lam), {"lambda": lam})

    def u3(self, theta: float, phi: float, lam: float, qubit: int) -> "QuantumCircuitBuilder":
        """General single-qubit U3 unitary."""
        return self._add(
            "U3", [qubit], _u3(theta, phi, lam),
            {"theta": theta, "phi": phi, "lambda": lam},
        )

    # ------------------------------------------------------------------
    # Two-qubit gates
    # ------------------------------------------------------------------

    def cnot(self, control: int, target: int) -> "QuantumCircuitBuilder":
        """Controlled-NOT (CX) gate."""
        return self._add("CNOT", [control, target], _CNOT.copy())

    def cx(self, control: int, target: int) -> "QuantumCircuitBuilder":
        """Alias for CNOT."""
        return self.cnot(control, target)

    def cy(self, control: int, target: int) -> "QuantumCircuitBuilder":
        """Controlled-Y gate."""
        return self._add("CY", [control, target], _cy())

    def cz(self, control: int, target: int) -> "QuantumCircuitBuilder":
        """Controlled-Z gate."""
        return self._add("CZ", [control, target], _CZ.copy())

    def ch(self, control: int, target: int) -> "QuantumCircuitBuilder":
        """Controlled-Hadamard gate."""
        return self._add("CH", [control, target], _ch())

    def swap(self, qubit_a: int, qubit_b: int) -> "QuantumCircuitBuilder":
        """SWAP gate."""
        return self._add("SWAP", [qubit_a, qubit_b], _SWAP.copy())

    def iswap(self, qubit_a: int, qubit_b: int) -> "QuantumCircuitBuilder":
        """iSWAP gate."""
        return self._add("iSWAP", [qubit_a, qubit_b], _iSWAP.copy())

    def crx(self, theta: float, control: int, target: int) -> "QuantumCircuitBuilder":
        """Controlled-RX gate."""
        m = np.eye(4, dtype=complex)
        rx_mat = _rx(theta)
        m[2:, 2:] = rx_mat
        return self._add("CRX", [control, target], m, {"theta": theta})

    def cry(self, theta: float, control: int, target: int) -> "QuantumCircuitBuilder":
        """Controlled-RY gate."""
        m = np.eye(4, dtype=complex)
        m[2:, 2:] = _ry(theta)
        return self._add("CRY", [control, target], m, {"theta": theta})

    def crz(self, theta: float, control: int, target: int) -> "QuantumCircuitBuilder":
        """Controlled-RZ gate."""
        m = np.eye(4, dtype=complex)
        m[2:, 2:] = _rz(theta)
        return self._add("CRZ", [control, target], m, {"theta": theta})

    def cp(self, lam: float, control: int, target: int) -> "QuantumCircuitBuilder":
        """Controlled-Phase gate."""
        m = np.eye(4, dtype=complex)
        m[3, 3] = np.exp(1j * lam)
        return self._add("CP", [control, target], m, {"lambda": lam})

    # ------------------------------------------------------------------
    # Three-qubit gates
    # ------------------------------------------------------------------

    def toffoli(self, control1: int, control2: int, target: int) -> "QuantumCircuitBuilder":
        """Toffoli (CCX) gate."""
        return self._add("Toffoli", [control1, control2, target], _Toffoli.copy())

    def ccx(self, control1: int, control2: int, target: int) -> "QuantumCircuitBuilder":
        """Alias for Toffoli."""
        return self.toffoli(control1, control2, target)

    def fredkin(self, control: int, target1: int, target2: int) -> "QuantumCircuitBuilder":
        """Fredkin (CSWAP) gate."""
        return self._add("Fredkin", [control, target1, target2], _Fredkin.copy())

    def cswap(self, control: int, target1: int, target2: int) -> "QuantumCircuitBuilder":
        """Alias for Fredkin."""
        return self.fredkin(control, target1, target2)

    # ------------------------------------------------------------------
    # High-level layer construction
    # ------------------------------------------------------------------

    def add_layer(
        self,
        gate_fn: Callable[["QuantumCircuitBuilder", int, float], "QuantumCircuitBuilder"],
        params: Sequence[float],
        qubits: Optional[Sequence[int]] = None,
    ) -> "QuantumCircuitBuilder":
        """Apply a single-qubit gate with individual parameters to a set of qubits.

        Args:
            gate_fn:
                A bound or unbound method such as ``cb.ry`` or a lambda that
                takes ``(builder, qubit, param)`` and returns ``builder``.
            params:
                Sequence of rotation angles, one per qubit.
            qubits:
                Qubit indices to apply the gate to.  Defaults to all qubits.

        Returns:
            Self (for method chaining).
        """
        target_qubits = list(qubits) if qubits is not None else list(range(self.n_qubits))
        if len(params) != len(target_qubits):
            raise ValueError(
                f"Expected {len(target_qubits)} parameters, got {len(params)}."
            )
        for q, p in zip(target_qubits, params):
            gate_fn(q, p)
        return self

    def add_entanglement_layer(
        self,
        style: str = "linear",
        gate: str = "cnot",
        qubits: Optional[Sequence[int]] = None,
    ) -> "QuantumCircuitBuilder":
        """Add an entangling gate layer between qubits.

        Args:
            style: Entanglement topology:
                * ``'linear'``    – nearest-neighbour pairs (0-1, 1-2, ...)
                * ``'circular'``  – linear + wrap-around last↔first
                * ``'full'``      – all-to-all pairs
                * ``'alternating'`` – alternates between even and odd pairs
            gate: Gate to use: ``'cnot'``, ``'cz'``, ``'swap'``, ``'iswap'``.
            qubits: Subset of qubit indices to entangle.  Defaults to all.

        Returns:
            Self (for method chaining).

        Raises:
            ValueError: On unknown style or gate name.
        """
        target = list(qubits) if qubits is not None else list(range(self.n_qubits))
        n = len(target)

        gate_map: Dict[str, Callable[[int, int], "QuantumCircuitBuilder"]] = {
            "cnot":  self.cnot,
            "cz":    self.cz,
            "swap":  self.swap,
            "iswap": self.iswap,
        }
        if gate not in gate_map:
            raise ValueError(
                f"Unknown entangling gate '{gate}'. Choose from {list(gate_map)}."
            )
        apply = gate_map[gate]

        if style == "linear":
            pairs = [(target[i], target[i + 1]) for i in range(n - 1)]
        elif style == "circular":
            pairs = [(target[i], target[(i + 1) % n]) for i in range(n)]
        elif style == "full":
            pairs = [
                (target[i], target[j])
                for i in range(n)
                for j in range(i + 1, n)
            ]
        elif style == "alternating":
            pairs = [(target[i], target[i + 1]) for i in range(0, n - 1, 2)]
        else:
            raise ValueError(
                f"Unknown entanglement style '{style}'. "
                "Choose from 'linear', 'circular', 'full', 'alternating'."
            )

        for q0, q1 in pairs:
            apply(q0, q1)
        return self

    # ------------------------------------------------------------------
    # Variational Quantum Circuit ansätze
    # ------------------------------------------------------------------

    def build_vqc_ansatz(
        self,
        params: np.ndarray,
        n_layers: int,
        entanglement: str = "linear",
        rotation_gates: Sequence[str] = ("ry", "rz"),
    ) -> "QuantumCircuitBuilder":
        """Construct a hardware-efficient VQC ansatz.

        The ansatz alternates rotation layers (each qubit gets one rotation per
        gate type) and entanglement layers.

        Param layout (for L layers, R rotation types, N qubits):
            ``params`` must have length ``L * R * N``.

        Args:
            params:        Flat array of rotation angles.
            n_layers:      Number of variational layers.
            entanglement:  Entanglement topology string (see :meth:`add_entanglement_layer`).
            rotation_gates: Gate types to apply per rotation sub-layer.
                            Supported: ``'rx'``, ``'ry'``, ``'rz'``.

        Returns:
            Self (for method chaining).

        Raises:
            ValueError: If ``len(params)`` doesn't match the expected count.
        """
        n = self.n_qubits
        r = len(rotation_gates)
        expected = n_layers * r * n
        if len(params) != expected:
            raise ValueError(
                f"Expected {expected} parameters for {n_layers} layers × "
                f"{r} rotations × {n} qubits, got {len(params)}."
            )

        rot_fn_map: Dict[str, Callable[[float, int], "QuantumCircuitBuilder"]] = {
            "rx": self.rx,
            "ry": self.ry,
            "rz": self.rz,
        }
        for rg in rotation_gates:
            if rg not in rot_fn_map:
                raise ValueError(f"Unsupported rotation gate '{rg}'. Use rx/ry/rz.")

        # Initial Hadamard layer
        for q in range(n):
            self.h(q)

        idx = 0
        for _ in range(n_layers):
            for rg in rotation_gates:
                fn = rot_fn_map[rg]
                for q in range(n):
                    fn(params[idx], q)
                    idx += 1
            self.add_entanglement_layer(style=entanglement)

        return self

    def build_qft_circuit(
        self,
        qubits: Optional[Sequence[int]] = None,
        inverse: bool = False,
    ) -> "QuantumCircuitBuilder":
        """Append the Quantum Fourier Transform (or its inverse) sub-circuit.

        Implements the standard textbook QFT decomposition using H and
        controlled-phase gates, followed by SWAP reversals.

        Args:
            qubits:  Qubit subset to apply QFT on.  Defaults to all qubits.
            inverse: If True, build the inverse QFT (IQFT).

        Returns:
            Self (for method chaining).
        """
        target = list(qubits) if qubits is not None else list(range(self.n_qubits))
        n = len(target)

        def _apply_qft(qs: List[int]) -> None:
            for i, q in enumerate(qs):
                self.h(q)
                for j, q2 in enumerate(qs[i + 1:], start=2):
                    lam = 2 * math.pi / (2 ** j)
                    self.cp(lam, q2, q)
            # Reverse qubit order with SWAPs
            for k in range(n // 2):
                self.swap(qs[k], qs[n - 1 - k])

        def _apply_iqft(qs: List[int]) -> None:
            # Inverse: reverse SWAP, then reverse QFT layers
            for k in range(n // 2):
                self.swap(qs[k], qs[n - 1 - k])
            for i in range(n - 1, -1, -1):
                q = qs[i]
                for j, q2 in enumerate(qs[i + 1:], start=2):
                    lam = -2 * math.pi / (2 ** j)
                    self.cp(lam, q2, q)
                self.h(q)

        if inverse:
            _apply_iqft(target)
        else:
            _apply_qft(target)

        return self

    def build_grover_diffuser(
        self, qubits: Optional[Sequence[int]] = None
    ) -> "QuantumCircuitBuilder":
        """Append a Grover diffusion operator (inversion about the mean).

        Args:
            qubits: Qubits to apply the diffuser on.  Defaults to all.

        Returns:
            Self (for method chaining).
        """
        target = list(qubits) if qubits is not None else list(range(self.n_qubits))
        n = len(target)
        for q in target:
            self.h(q)
            self.x(q)
        # Multi-controlled Z
        if n >= 3:
            self.toffoli(target[0], target[1], target[2])
        elif n == 2:
            self.cz(target[0], target[1])
        else:
            self.z(target[0])
        for q in target:
            self.x(q)
            self.h(q)
        return self

    # ------------------------------------------------------------------
    # Measurement-based helpers
    # ------------------------------------------------------------------

    def add_barrier(self) -> "QuantumCircuitBuilder":
        """Insert a logical barrier (no-op, used for readability/grouping)."""
        # Advance all qubits to the same layer
        max_layer = max(self._qubit_layer)
        self._qubit_layer = [max_layer] * self.n_qubits
        return self

    # ------------------------------------------------------------------
    # Circuit execution
    # ------------------------------------------------------------------

    def run(
        self,
        initial_state: Optional[QuantumState] = None,
        params: Optional[Dict[str, float]] = None,
    ) -> QuantumState:
        """Execute the circuit on an initial state.

        Args:
            initial_state:
                Starting :class:`QuantumState`.  Defaults to |0...0⟩.
            params:
                Optional parameter override dict mapping ``gate_name`` to new
                angle value.  Currently applied by position; pass a flat list
                of parameter dicts for per-gate overrides (advanced use).

        Returns:
            The final :class:`QuantumState` after all gate applications.
        """
        if initial_state is None:
            state = QuantumState(self.n_qubits)
        else:
            if initial_state.n_qubits != self.n_qubits:
                raise ValueError(
                    f"Initial state has {initial_state.n_qubits} qubits, "
                    f"but circuit has {self.n_qubits}."
                )
            state = initial_state

        for instr in self._instructions:
            logger.debug("Applying %s", instr)
            state = state.apply_gate(instr.matrix, instr.qubits)
            if self._noise_model is not None:
                state = self._noise_model(state, instr)

        return state

    def run_with_shots(
        self,
        shots: int = 1024,
        initial_state: Optional[QuantumState] = None,
    ) -> Dict[str, int]:
        """Execute the circuit and return measurement counts.

        Args:
            shots:         Number of measurement repetitions.
            initial_state: Starting state (defaults to |0...0⟩).

        Returns:
            Dictionary mapping binary outcome strings to counts.
        """
        final_state = self.run(initial_state)
        return final_state.measure_all(shots=shots)

    def statevector(
        self, initial_state: Optional[QuantumState] = None
    ) -> np.ndarray:
        """Return the final statevector as a NumPy array.

        Args:
            initial_state: Starting state.

        Returns:
            Complex array of shape (2**n_qubits,).
        """
        return self.run(initial_state).statevector

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_circuit_depth(self) -> int:
        """Return the circuit depth (longest path in the dependency DAG).

        Returns:
            Integer depth, equal to the maximum layer index + 1.
        """
        if not self._instructions:
            return 0
        return max(instr.layer for instr in self._instructions) + 1

    def gate_count(self, gate_name: Optional[str] = None) -> int:
        """Count the total number of gate instructions.

        Args:
            gate_name: If provided, count only that gate type.

        Returns:
            Integer count.
        """
        if gate_name is None:
            return len(self._instructions)
        return sum(1 for i in self._instructions if i.gate_name == gate_name)

    def parameter_count(self) -> int:
        """Count the total number of variational parameters in the circuit."""
        return sum(len(i.params) for i in self._instructions)

    def to_dict(self) -> List[dict]:
        """Serialise the circuit to a list of instruction dicts.

        Returns:
            List of dicts with keys ``gate_name``, ``qubits``, ``params``,
            ``layer``.
        """
        return [
            {
                "gate_name": instr.gate_name,
                "qubits": instr.qubits,
                "params": instr.params,
                "layer": instr.layer,
            }
            for instr in self._instructions
        ]

    def draw(self) -> str:
        """Return an ASCII text diagram of the circuit.

        Returns:
            Multi-line string with one row per qubit.
        """
        depth = self.get_circuit_depth()
        if depth == 0:
            return "(empty circuit)"

        # Build a 2-D grid: rows=qubits, cols=layers
        grid: List[List[str]] = [["─" * 6] * depth for _ in range(self.n_qubits)]

        for instr in self._instructions:
            col = instr.layer
            label = instr.gate_name[:4]
            for q in instr.qubits:
                grid[q][col] = f"[{label:<4}]"

        lines = []
        for q, row in enumerate(grid):
            lines.append(f"q{q}: " + "".join(row))
        return "\n".join(lines)

    def reset(self) -> "QuantumCircuitBuilder":
        """Clear all gate instructions and reset the circuit.

        Returns:
            Self (for method chaining).
        """
        self._instructions.clear()
        self._qubit_layer = [0] * self.n_qubits
        return self

    def copy(self) -> "QuantumCircuitBuilder":
        """Return a deep copy of this circuit builder.

        Returns:
            New :class:`QuantumCircuitBuilder` with identical instructions.
        """
        new_cb = QuantumCircuitBuilder(self.n_qubits, self._noise_model)
        new_cb._instructions = [
            GateInstruction(
                gate_name=i.gate_name,
                qubits=list(i.qubits),
                matrix=i.matrix.copy(),
                params=dict(i.params),
                layer=i.layer,
            )
            for i in self._instructions
        ]
        new_cb._qubit_layer = list(self._qubit_layer)
        return new_cb

    def __repr__(self) -> str:
        return (
            f"QuantumCircuitBuilder("
            f"n_qubits={self.n_qubits}, "
            f"depth={self.get_circuit_depth()}, "
            f"gates={self.gate_count()})"
        )
