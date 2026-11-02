"""Quantum circuit visualisation utilities for QuantumShield-FL.

Provides tools to render quantum circuits as:

* ASCII text diagrams (no dependencies beyond the standard library).
* Matplotlib figures (static SVG/PNG exports).
* Bloch sphere representations of single-qubit states.
* Animated quantum state-vector evolution.

All heavy-lifting is done with ``matplotlib`` and ``numpy`` only — no
external quantum-computing library is required.  The ASCII renderer
works with zero external dependencies.

Typical usage
-------------
>>> from src.visualization.circuit_viz import QuantumCircuit, CircuitVisualizer
>>> qc = QuantumCircuit(n_qubits=3)
>>> qc.h(0).cx(0, 1).rx(0.5, 2)
>>> viz = CircuitVisualizer(qc)
>>> print(viz.to_ascii())
>>> fig = viz.to_matplotlib()
>>> fig.savefig("circuit.png", dpi=150)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Gate definitions
# ---------------------------------------------------------------------------

# Standard single-qubit gate matrices
_GATE_MATRICES: Dict[str, np.ndarray] = {
    "H": np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    "S": np.array([[1, 0], [0, 1j]], dtype=complex),
    "T": np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=complex),
    "I": np.eye(2, dtype=complex),
}


def _rx(theta: float) -> np.ndarray:
    """Rotation-X gate matrix."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _ry(theta: float) -> np.ndarray:
    """Rotation-Y gate matrix."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(theta: float) -> np.ndarray:
    """Rotation-Z gate matrix."""
    return np.array(
        [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex
    )


# ---------------------------------------------------------------------------
# Gate instruction dataclass
# ---------------------------------------------------------------------------


@dataclass
class GateInstruction:
    """Representation of a single gate operation in a circuit.

    Attributes:
        name: Gate identifier string (e.g. 'H', 'CNOT', 'RX').
        qubits: List of qubit indices the gate acts on.
        params: Optional numerical parameters (e.g. rotation angles).
        matrix: Optional explicit gate unitary.
        label: Display label for the gate (defaults to ``name``).
    """

    name: str
    qubits: List[int]
    params: List[float] = field(default_factory=list)
    matrix: Optional[np.ndarray] = None
    label: Optional[str] = None

    def display_label(self) -> str:
        """Return the label to show in circuit diagrams."""
        if self.label:
            return self.label
        if self.params:
            param_str = ",".join(f"{p:.2f}" for p in self.params)
            return f"{self.name}({param_str})"
        return self.name


# ---------------------------------------------------------------------------
# Quantum circuit representation
# ---------------------------------------------------------------------------


class QuantumCircuit:
    """A simple quantum circuit builder.

    Supports a fluent interface for adding gates.  The circuit stores
    instructions in order but does not execute them until asked to
    compute the state vector.

    Args:
        n_qubits: Number of qubits in the circuit.
        name: Optional human-readable circuit name.
    """

    def __init__(self, n_qubits: int, name: str = "circuit") -> None:
        if n_qubits < 1:
            raise ValueError("n_qubits must be at least 1.")
        self.n_qubits = n_qubits
        self.name = name
        self._instructions: List[GateInstruction] = []

    # ------------------------------------------------------------------ #
    # Gate methods (fluent API)
    # ------------------------------------------------------------------ #

    def _add(self, instr: GateInstruction) -> "QuantumCircuit":
        """Append an instruction and return self for chaining."""
        for q in instr.qubits:
            if not (0 <= q < self.n_qubits):
                raise ValueError(
                    f"Qubit index {q} is out of range for a {self.n_qubits}-qubit circuit."
                )
        self._instructions.append(instr)
        return self

    def h(self, qubit: int) -> "QuantumCircuit":
        """Apply a Hadamard gate.

        Args:
            qubit: Target qubit index.

        Returns:
            self (for chaining).
        """
        return self._add(GateInstruction("H", [qubit], matrix=_GATE_MATRICES["H"]))

    def x(self, qubit: int) -> "QuantumCircuit":
        """Apply a Pauli-X (NOT) gate."""
        return self._add(GateInstruction("X", [qubit], matrix=_GATE_MATRICES["X"]))

    def y(self, qubit: int) -> "QuantumCircuit":
        """Apply a Pauli-Y gate."""
        return self._add(GateInstruction("Y", [qubit], matrix=_GATE_MATRICES["Y"]))

    def z(self, qubit: int) -> "QuantumCircuit":
        """Apply a Pauli-Z gate."""
        return self._add(GateInstruction("Z", [qubit], matrix=_GATE_MATRICES["Z"]))

    def s(self, qubit: int) -> "QuantumCircuit":
        """Apply an S (phase) gate."""
        return self._add(GateInstruction("S", [qubit], matrix=_GATE_MATRICES["S"]))

    def t(self, qubit: int) -> "QuantumCircuit":
        """Apply a T gate."""
        return self._add(GateInstruction("T", [qubit], matrix=_GATE_MATRICES["T"]))

    def rx(self, theta: float, qubit: int) -> "QuantumCircuit":
        """Apply an RX rotation gate.

        Args:
            theta: Rotation angle in radians.
            qubit: Target qubit index.

        Returns:
            self (for chaining).
        """
        return self._add(
            GateInstruction("RX", [qubit], params=[theta], matrix=_rx(theta))
        )

    def ry(self, theta: float, qubit: int) -> "QuantumCircuit":
        """Apply an RY rotation gate."""
        return self._add(
            GateInstruction("RY", [qubit], params=[theta], matrix=_ry(theta))
        )

    def rz(self, theta: float, qubit: int) -> "QuantumCircuit":
        """Apply an RZ rotation gate."""
        return self._add(
            GateInstruction("RZ", [qubit], params=[theta], matrix=_rz(theta))
        )

    def cx(self, control: int, target: int) -> "QuantumCircuit":
        """Apply a CNOT (controlled-X) gate.

        Args:
            control: Control qubit index.
            target: Target qubit index.

        Returns:
            self (for chaining).
        """
        return self._add(GateInstruction("CNOT", [control, target], label="●─X"))

    def cz(self, control: int, target: int) -> "QuantumCircuit":
        """Apply a controlled-Z gate."""
        return self._add(GateInstruction("CZ", [control, target], label="●─Z"))

    def swap(self, qubit_a: int, qubit_b: int) -> "QuantumCircuit":
        """Apply a SWAP gate."""
        return self._add(GateInstruction("SWAP", [qubit_a, qubit_b], label="✕─✕"))

    def measure(self, qubit: int) -> "QuantumCircuit":
        """Add a measurement operation."""
        return self._add(GateInstruction("M", [qubit], label="M"))

    def barrier(self) -> "QuantumCircuit":
        """Add a visual barrier (no physical operation)."""
        return self._add(GateInstruction("BARRIER", list(range(self.n_qubits)), label="|"))

    # ------------------------------------------------------------------ #
    # State-vector simulation
    # ------------------------------------------------------------------ #

    def statevector(self) -> np.ndarray:
        """Compute the final state vector by applying all gate matrices.

        The initial state is |0⟩^{⊗n}.

        Returns:
            Complex numpy array of shape (2**n_qubits,).
        """
        n = self.n_qubits
        state = np.zeros(2**n, dtype=complex)
        state[0] = 1.0  # |00...0⟩

        for instr in self._instructions:
            if instr.name in ("BARRIER", "M"):
                continue
            if instr.matrix is None:
                continue

            if len(instr.qubits) == 1:
                state = _apply_single_qubit_gate(state, instr.matrix, instr.qubits[0], n)
            elif instr.name == "CNOT":
                state = _apply_cnot(state, instr.qubits[0], instr.qubits[1], n)
            elif instr.name == "SWAP":
                state = _apply_swap(state, instr.qubits[0], instr.qubits[1], n)

        return state

    def statevector_history(self) -> List[np.ndarray]:
        """Return a list of state vectors after each gate application.

        Useful for animating state evolution.

        Returns:
            List of complex state-vector arrays, one per instruction step.
        """
        n = self.n_qubits
        state = np.zeros(2**n, dtype=complex)
        state[0] = 1.0
        history = [state.copy()]

        for instr in self._instructions:
            if instr.name in ("BARRIER", "M"):
                history.append(state.copy())
                continue
            if instr.matrix is None:
                history.append(state.copy())
                continue

            if len(instr.qubits) == 1:
                state = _apply_single_qubit_gate(state, instr.matrix, instr.qubits[0], n)
            elif instr.name == "CNOT":
                state = _apply_cnot(state, instr.qubits[0], instr.qubits[1], n)
            elif instr.name == "SWAP":
                state = _apply_swap(state, instr.qubits[0], instr.qubits[1], n)

            history.append(state.copy())

        return history

    @property
    def depth(self) -> int:
        """Circuit depth: number of time steps (ignoring barriers)."""
        return sum(1 for i in self._instructions if i.name != "BARRIER")

    @property
    def n_gates(self) -> int:
        """Total number of gate operations (excluding barriers and measurements)."""
        return sum(
            1 for i in self._instructions if i.name not in ("BARRIER", "M")
        )


# ---------------------------------------------------------------------------
# Linear algebra helpers
# ---------------------------------------------------------------------------


def _apply_single_qubit_gate(
    state: np.ndarray, gate: np.ndarray, qubit: int, n_qubits: int
) -> np.ndarray:
    """Apply a 2×2 gate to a single qubit of a state vector.

    Args:
        state: Current state vector of shape (2**n_qubits,).
        gate: 2×2 unitary matrix.
        qubit: Target qubit index (0 = least significant bit).
        n_qubits: Total number of qubits.

    Returns:
        New state vector.
    """
    # Reshape into tensor of shape (2, 2, ..., 2)
    shape = [2] * n_qubits
    psi = state.reshape(shape)
    # Axis in the tensor corresponding to this qubit (0-indexed from MSB)
    axis = n_qubits - 1 - qubit
    psi = np.tensordot(gate, psi, axes=([1], [axis]))
    # tensordot returns (2, ...) with the contracted axis first; move it back
    psi = np.moveaxis(psi, 0, axis)
    return psi.reshape(-1)


def _apply_cnot(
    state: np.ndarray, control: int, target: int, n_qubits: int
) -> np.ndarray:
    """Apply a CNOT gate to a state vector.

    Args:
        state: Current state vector.
        control: Control qubit index.
        target: Target qubit index.
        n_qubits: Total number of qubits.

    Returns:
        New state vector.
    """
    new_state = state.copy()
    for i in range(2**n_qubits):
        ctrl_bit = (i >> control) & 1
        if ctrl_bit:
            j = i ^ (1 << target)
            new_state[i], new_state[j] = state[j], state[i]
    return new_state


def _apply_swap(
    state: np.ndarray, qubit_a: int, qubit_b: int, n_qubits: int
) -> np.ndarray:
    """Apply a SWAP gate to a state vector."""
    new_state = state.copy()
    for i in range(2**n_qubits):
        bit_a = (i >> qubit_a) & 1
        bit_b = (i >> qubit_b) & 1
        if bit_a != bit_b:
            j = i ^ (1 << qubit_a) ^ (1 << qubit_b)
            new_state[i], new_state[j] = state[j], state[i]
    return new_state


# ---------------------------------------------------------------------------
# Circuit visualiser
# ---------------------------------------------------------------------------


class CircuitVisualizer:
    """Renders a QuantumCircuit in multiple output formats.

    Args:
        circuit: The quantum circuit to visualise.
    """

    def __init__(self, circuit: QuantumCircuit) -> None:
        self.circuit = circuit

    # ------------------------------------------------------------------ #
    # ASCII rendering
    # ------------------------------------------------------------------ #

    def to_ascii(self) -> str:
        """Render the circuit as an ASCII diagram.

        Returns:
            Multi-line string containing the circuit diagram.

        Example::

            q0: ──H──●──────
                     │
            q1: ─────X──H──
        """
        n = self.circuit.n_qubits
        instructions = self.circuit._instructions

        # Build a column for each instruction
        columns: List[Dict[int, str]] = []
        for instr in instructions:
            col: Dict[int, str] = {}
            if instr.name == "BARRIER":
                for q in range(n):
                    col[q] = "|"
            elif len(instr.qubits) == 1:
                col[instr.qubits[0]] = instr.display_label()
            elif instr.name in ("CNOT", "CZ"):
                ctrl, tgt = instr.qubits
                col[ctrl] = "●"
                col[tgt] = "X" if instr.name == "CNOT" else "Z"
                # Mark intermediate wires
                lo, hi = min(ctrl, tgt), max(ctrl, tgt)
                for q in range(lo + 1, hi):
                    col[q] = "│"
            elif instr.name == "SWAP":
                for q in instr.qubits:
                    col[q] = "✕"
                lo, hi = min(instr.qubits), max(instr.qubits)
                for q in range(lo + 1, hi):
                    col[q] = "│"
            else:
                for q in instr.qubits:
                    col[q] = instr.display_label()
            columns.append(col)

        # Compute column widths
        col_widths = [
            max((len(col.get(q, "─")) for q in range(n)), default=1) + 2
            for col in columns
        ]

        # Wire prefix width
        prefix_width = max(len(f"q{q}: ") for q in range(n))

        lines: List[str] = []
        for q in range(n):
            parts = [f"q{q}: ".ljust(prefix_width)]
            for col_idx, col in enumerate(columns):
                w = col_widths[col_idx]
                cell = col.get(q, "")
                if cell == "│":
                    parts.append(" │ ".center(w, " "))
                elif cell:
                    parts.append(f"[{cell}]".center(w, "─"))
                else:
                    parts.append("─" * w)
            lines.append("".join(parts))

        header = f"Circuit: {self.circuit.name}  ({n} qubits, depth={self.circuit.depth})"
        separator = "─" * len(header)
        return "\n".join([header, separator] + lines)

    # ------------------------------------------------------------------ #
    # Matplotlib rendering
    # ------------------------------------------------------------------ #

    def to_matplotlib(
        self,
        figsize: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
        dpi: int = 100,
    ):
        """Render the circuit as a Matplotlib figure.

        Args:
            figsize: Figure dimensions in inches ``(width, height)``.
                     Computed automatically if None.
            title: Optional title string.
            dpi: Dots per inch for rasterisation.

        Returns:
            ``matplotlib.figure.Figure`` object.
        """
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt

        n = self.circuit.n_qubits
        instructions = self.circuit._instructions
        n_steps = len(instructions)

        if figsize is None:
            figsize = (max(6, n_steps * 0.9 + 2), max(3, n * 0.7 + 1))

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.set_xlim(-0.5, n_steps + 0.5)
        ax.set_ylim(-0.5, n - 0.5)
        ax.invert_yaxis()
        ax.axis("off")
        ax.set_facecolor("#f8f8ff")
        fig.patch.set_facecolor("#f8f8ff")

        # Draw qubit wires
        for q in range(n):
            ax.axhline(y=q, xmin=0.02, xmax=0.98, color="#555", linewidth=1.2, zorder=1)
            ax.text(
                -0.3,
                q,
                f"q{q}",
                ha="right",
                va="center",
                fontsize=9,
                fontfamily="monospace",
                color="#333",
            )

        # Draw gates
        colors = {
            "H": "#4a90d9",
            "X": "#e74c3c",
            "Y": "#e67e22",
            "Z": "#8e44ad",
            "S": "#27ae60",
            "T": "#16a085",
            "RX": "#2980b9",
            "RY": "#c0392b",
            "RZ": "#7f8c8d",
            "CNOT": "#e74c3c",
            "CZ": "#8e44ad",
            "SWAP": "#1abc9c",
            "M": "#2c3e50",
            "BARRIER": "#95a5a6",
        }

        for step, instr in enumerate(instructions):
            x = step
            color = colors.get(instr.name, "#3498db")
            label = instr.display_label()

            if instr.name == "BARRIER":
                ax.axvline(x=x, ymin=0.02, ymax=0.98, color="#aaa", linewidth=1.5,
                           linestyle="--", zorder=2)
                continue

            if len(instr.qubits) == 1:
                q = instr.qubits[0]
                rect = mpatches.FancyBboxPatch(
                    (x - 0.3, q - 0.3), 0.6, 0.6,
                    boxstyle="round,pad=0.05",
                    facecolor=color,
                    edgecolor="#333",
                    linewidth=1,
                    zorder=3,
                )
                ax.add_patch(rect)
                ax.text(x, q, label, ha="center", va="center", fontsize=7,
                        color="white", fontweight="bold", zorder=4)

            elif instr.name == "CNOT":
                ctrl, tgt = instr.qubits
                ax.plot([x, x], [ctrl, tgt], color="#555", linewidth=1.5, zorder=2)
                ax.plot(x, ctrl, "o", color="#333", markersize=8, zorder=3)
                circle = plt.Circle((x, tgt), 0.25, color=color, ec="#333", lw=1.5, zorder=3)
                ax.add_patch(circle)
                ax.plot([x - 0.25, x + 0.25], [tgt, tgt], color="white", lw=1.5, zorder=4)
                ax.plot([x, x], [tgt - 0.25, tgt + 0.25], color="white", lw=1.5, zorder=4)

            elif instr.name in ("CZ", "SWAP"):
                q0, q1 = instr.qubits
                ax.plot([x, x], [q0, q1], color="#555", linewidth=1.5, zorder=2)
                for q in (q0, q1):
                    rect = mpatches.FancyBboxPatch(
                        (x - 0.25, q - 0.25), 0.5, 0.5,
                        boxstyle="round,pad=0.05",
                        facecolor=color,
                        edgecolor="#333",
                        linewidth=1,
                        zorder=3,
                    )
                    ax.add_patch(rect)
                    sym = "●" if instr.name == "CZ" and q == q0 else "✕"
                    ax.text(x, q, sym, ha="center", va="center",
                            fontsize=8, color="white", zorder=4)

        ax.set_title(
            title or f"{self.circuit.name}  —  {n} qubits | depth {self.circuit.depth}",
            fontsize=10,
            pad=8,
        )
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------ #
    # Bloch sphere
    # ------------------------------------------------------------------ #

    def plot_bloch_sphere(
        self,
        qubit: int = 0,
        state: Optional[np.ndarray] = None,
        title: Optional[str] = None,
    ):
        """Plot the Bloch sphere representation of a single qubit state.

        Args:
            qubit: Qubit index to trace out and display.
            state: Optional pre-computed state vector. If None, the circuit
                   is simulated to obtain the final state.
            title: Optional plot title.

        Returns:
            ``matplotlib.figure.Figure`` object.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        if state is None:
            state = self.circuit.statevector()

        # Compute reduced density matrix for the qubit
        n = self.circuit.n_qubits
        rho = _reduced_density_matrix(state, qubit, n)

        # Bloch vector components
        sx = np.real(np.trace(rho @ _GATE_MATRICES["X"]))
        sy = np.real(np.trace(rho @ _GATE_MATRICES["Y"]))
        sz = np.real(np.trace(rho @ _GATE_MATRICES["Z"]))

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")

        # Draw sphere wireframe
        u = np.linspace(0, 2 * np.pi, 40)
        v = np.linspace(0, np.pi, 40)
        xs = np.outer(np.cos(u), np.sin(v))
        ys = np.outer(np.sin(u), np.sin(v))
        zs = np.outer(np.ones(40), np.cos(v))
        ax.plot_wireframe(xs, ys, zs, color="#ccddee", alpha=0.25, linewidth=0.4)

        # Draw axes
        for direction, label, color in [
            ([1, 0, 0], "X", "#e74c3c"),
            ([0, 1, 0], "Y", "#27ae60"),
            ([0, 0, 1], "Z", "#3498db"),
        ]:
            ax.quiver(0, 0, 0, *direction, length=1.2, color=color,
                      arrow_length_ratio=0.15, linewidth=1.5, alpha=0.7)
            ax.text(*(d * 1.35 for d in direction), label, color=color,
                    fontsize=10, ha="center", va="center")

        # Draw equator and meridians
        circle_pts = np.linspace(0, 2 * np.pi, 100)
        ax.plot(np.cos(circle_pts), np.sin(circle_pts), 0, color="#aaa", lw=0.8, alpha=0.6)
        ax.plot(np.cos(circle_pts), np.zeros(100), np.sin(circle_pts), color="#aaa", lw=0.8, alpha=0.6)
        ax.plot(np.zeros(100), np.cos(circle_pts), np.sin(circle_pts), color="#aaa", lw=0.8, alpha=0.6)

        # Draw Bloch vector
        ax.quiver(0, 0, 0, sx, sy, sz, color="#e67e22", linewidth=3,
                  arrow_length_ratio=0.2)
        ax.text(sx * 1.1, sy * 1.1, sz * 1.1,
                f"({sx:.2f}, {sy:.2f}, {sz:.2f})",
                fontsize=7, color="#e67e22")

        # Labels
        ax.text(0, 0, 1.4, "|0⟩", fontsize=10, ha="center", color="#3498db")
        ax.text(0, 0, -1.4, "|1⟩", fontsize=10, ha="center", color="#3498db")

        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_zlim(-1.3, 1.3)
        ax.set_xlabel("X", labelpad=1)
        ax.set_ylabel("Y", labelpad=1)
        ax.set_zlabel("Z", labelpad=1)
        ax.set_title(
            title or f"Bloch sphere – qubit {qubit}  |  "
            f"r=({sx:.2f}, {sy:.2f}, {sz:.2f})",
            fontsize=9,
        )
        ax.grid(False)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------ #
    # State-vector animation
    # ------------------------------------------------------------------ #

    def animate_evolution(
        self,
        interval_ms: int = 600,
        figsize: Tuple[float, float] = (8, 4),
    ):
        """Animate the quantum state-vector evolution through the circuit.

        Each frame shows the probability amplitude bar chart after applying
        one additional gate.

        Args:
            interval_ms: Milliseconds between animation frames.
            figsize: Figure dimensions ``(width, height)``.

        Returns:
            ``matplotlib.animation.FuncAnimation`` object.

        Note:
            Call ``anim.save("evolution.gif")`` or display inline in a
            Jupyter notebook with ``from IPython.display import HTML;
            HTML(anim.to_jshtml())``.
        """
        import matplotlib.animation as animation
        import matplotlib.pyplot as plt

        history = self.circuit.statevector_history()
        n_states = 2**self.circuit.n_qubits
        basis_labels = [
            f"|{format(i, f'0{self.circuit.n_qubits}b')}⟩" for i in range(n_states)
        ]

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.patch.set_facecolor("#f8f8ff")

        # Probability bar chart
        ax_prob = axes[0]
        ax_phase = axes[1]

        bar_probs = ax_prob.bar(
            range(n_states),
            np.abs(history[0]) ** 2,
            color="#4a90d9",
            edgecolor="#333",
            linewidth=0.5,
        )
        ax_prob.set_ylim(0, 1.05)
        ax_prob.set_xticks(range(n_states))
        ax_prob.set_xticklabels(basis_labels, rotation=45, ha="right", fontsize=7)
        ax_prob.set_ylabel("Probability")
        ax_prob.set_title("State probabilities", fontsize=9)
        ax_prob.set_facecolor("#f8f8ff")

        # Phase bar chart
        bar_phases = ax_phase.bar(
            range(n_states),
            np.angle(history[0]) / np.pi,
            color="#e67e22",
            edgecolor="#333",
            linewidth=0.5,
        )
        ax_phase.set_ylim(-1.1, 1.1)
        ax_phase.set_xticks(range(n_states))
        ax_phase.set_xticklabels(basis_labels, rotation=45, ha="right", fontsize=7)
        ax_phase.set_ylabel("Phase (π radians)")
        ax_phase.set_title("State phases", fontsize=9)
        ax_phase.set_facecolor("#f8f8ff")

        step_text = fig.text(0.5, 0.97, "Initial state |0…0⟩", ha="center",
                              fontsize=9, color="#333")

        instructions = self.circuit._instructions

        def _update(frame: int):
            state = history[frame]
            probs = np.abs(state) ** 2
            phases = np.angle(state) / np.pi

            for bar, p in zip(bar_probs, probs):
                bar.set_height(p)
            for bar, ph in zip(bar_phases, phases):
                bar.set_height(ph)

            if frame == 0:
                step_text.set_text("Initial state |0…0⟩")
            else:
                instr = instructions[frame - 1]
                step_text.set_text(
                    f"Step {frame}: {instr.display_label()} on qubit(s) {instr.qubits}"
                )
            return list(bar_probs) + list(bar_phases) + [step_text]

        anim = animation.FuncAnimation(
            fig,
            _update,
            frames=len(history),
            interval=interval_ms,
            blit=False,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        return anim


# ---------------------------------------------------------------------------
# Density matrix helper
# ---------------------------------------------------------------------------


def _reduced_density_matrix(
    state: np.ndarray, qubit: int, n_qubits: int
) -> np.ndarray:
    """Compute the reduced density matrix for a single qubit by tracing out others.

    Args:
        state: Full state vector of shape (2**n_qubits,).
        qubit: Index of the qubit to keep.
        n_qubits: Total number of qubits.

    Returns:
        2×2 complex density matrix for the specified qubit.
    """
    shape = [2] * n_qubits
    psi = state.reshape(shape)
    # Axis index for this qubit (0 = MSB in our convention)
    axis = n_qubits - 1 - qubit
    axes_to_trace = [i for i in range(n_qubits) if i != axis]

    rho = np.zeros((2, 2), dtype=complex)
    for idx0 in range(2):
        for idx1 in range(2):
            # Slice along the qubit axis
            sl0 = [slice(None)] * n_qubits
            sl1 = [slice(None)] * n_qubits
            sl0[axis] = idx0
            sl1[axis] = idx1
            rho[idx0, idx1] = np.sum(
                np.conj(psi[tuple(sl0)]) * psi[tuple(sl1)]
            )
    return rho


# ---------------------------------------------------------------------------
# Convenience factory for common VQC circuits
# ---------------------------------------------------------------------------


def build_vqc(
    n_qubits: int,
    n_layers: int,
    params: Optional[np.ndarray] = None,
    name: str = "VQC",
) -> QuantumCircuit:
    """Build a hardware-efficient Variational Quantum Circuit.

    Architecture:
    * Layer 0: Apply H to all qubits (encoding).
    * For each layer: RY and RZ rotations on every qubit, followed by a
      linear entanglement ladder of CNOT gates.

    Args:
        n_qubits: Number of qubits.
        n_layers: Number of variational layers.
        params: Optional flat array of rotation angles. If None, random
                angles in [0, 2π) are generated.
        name: Circuit name.

    Returns:
        Fully configured QuantumCircuit.
    """
    n_params = n_layers * n_qubits * 2  # RY + RZ per qubit per layer
    if params is None:
        params = np.random.default_rng(0).uniform(0, 2 * math.pi, n_params)
    elif len(params) != n_params:
        raise ValueError(
            f"Expected {n_params} parameters, got {len(params)}."
        )

    qc = QuantumCircuit(n_qubits=n_qubits, name=name)

    # Hadamard encoding layer
    for q in range(n_qubits):
        qc.h(q)

    param_idx = 0
    for _ in range(n_layers):
        qc.barrier()
        # Rotation layer
        for q in range(n_qubits):
            qc.ry(float(params[param_idx]), q)
            param_idx += 1
            qc.rz(float(params[param_idx]), q)
            param_idx += 1
        # Entanglement layer
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)

    return qc
