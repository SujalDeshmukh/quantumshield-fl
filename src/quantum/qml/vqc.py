"""
Variational Quantum Classifier (VQC) for QuantumShield-FL.

This module implements a pure-NumPy simulation of a parameterized quantum
circuit trained as a classifier.  Key features:

- Multiple ansatz types: ``strongly_entangling``, ``hardware_efficient``,
  ``twolocal``
- Parameter-shift rule for exact gradient computation
- Adam optimizer (implemented from scratch with NumPy)
- Binary and multiclass (one-vs-rest) classification
- Cross-entropy loss
- Full training history tracking
- JSON serialisation of model parameters
"""

from __future__ import annotations

import cmath
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

Array = np.ndarray
AnsatzType = Literal["strongly_entangling", "hardware_efficient", "twolocal"]


# ---------------------------------------------------------------------------
# Gate primitives (pure numpy, no framework dependency)
# ---------------------------------------------------------------------------

def _rx(theta: float) -> Array:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _ry(theta: float) -> Array:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(theta: float) -> Array:
    return np.array(
        [[cmath.exp(-1j * theta / 2), 0],
         [0,  cmath.exp(1j * theta / 2)]],
        dtype=complex,
    )


def _hadamard() -> Array:
    return np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)


def _cnot() -> Array:
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                     [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)


def _single_qubit_op(gate: Array, q: int, n: int) -> Array:
    """Embed a 2x2 gate into the n-qubit Hilbert space."""
    ops = [np.eye(2, dtype=complex)] * n
    ops[q] = gate
    result = ops[0]
    for m in ops[1:]:
        result = np.kron(result, m)
    return result


def _two_qubit_cnot(ctrl: int, tgt: int, n: int) -> Array:
    """Build a CNOT unitary in the full n-qubit space."""
    dim = 2 ** n
    U = np.eye(dim, dtype=complex)
    ctrl_mask = 1 << (n - 1 - ctrl)
    tgt_mask  = 1 << (n - 1 - tgt)
    for idx in range(dim):
        if idx & ctrl_mask:
            flipped = idx ^ tgt_mask
            U[idx, idx] = 0
            U[flipped, idx] = 1
            if flipped != idx:
                U[flipped, flipped] = 0
                U[idx, flipped] = 1
    return U


# ---------------------------------------------------------------------------
# Quantum circuit simulator
# ---------------------------------------------------------------------------

class _QuantumCircuit:
    """
    Minimal statevector quantum circuit simulator.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    """

    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits

    def zero_state(self) -> Array:
        """Return |0...0⟩ state."""
        state = np.zeros(self.dim, dtype=complex)
        state[0] = 1.0
        return state

    def apply(self, state: Array, U: Array) -> Array:
        """Apply unitary U to state."""
        return U @ state

    def measure_z(self, state: Array, qubit: int) -> float:
        """Compute ⟨Z_q⟩ = Pr(0) - Pr(1) for qubit q."""
        mask = 1 << (self.n_qubits - 1 - qubit)
        probs = np.abs(state) ** 2
        prob_0 = sum(p for idx, p in enumerate(probs) if not (idx & mask))
        prob_1 = 1.0 - prob_0
        return float(prob_0 - prob_1)

    def measure_all_z(self, state: Array) -> Array:
        """Return ⟨Z_q⟩ for every qubit."""
        return np.array([self.measure_z(state, q) for q in range(self.n_qubits)])


# ---------------------------------------------------------------------------
# Ansatz builders
# ---------------------------------------------------------------------------

def _build_strongly_entangling(
    params: Array,
    n_qubits: int,
    n_layers: int,
    circuit: _QuantumCircuit,
    state: Array,
) -> Array:
    """
    Strongly-entangling layers ansatz.

    Each layer applies Ry(θ) Rz(φ) to every qubit, then a ring of CNOT gates.

    Parameters
    ----------
    params : Array
        Flat parameter array of length ``n_layers * n_qubits * 2``.
    """
    p = params.reshape(n_layers, n_qubits, 2)
    for layer in range(n_layers):
        for q in range(n_qubits):
            ry = _ry(p[layer, q, 0])
            rz = _rz(p[layer, q, 1])
            state = circuit.apply(state, _single_qubit_op(ry, q, n_qubits))
            state = circuit.apply(state, _single_qubit_op(rz, q, n_qubits))
        # Ring CNOT entanglement
        for q in range(n_qubits):
            ctrl, tgt = q, (q + 1) % n_qubits
            if n_qubits > 1:
                state = circuit.apply(state, _two_qubit_cnot(ctrl, tgt, n_qubits))
    return state


def _build_hardware_efficient(
    params: Array,
    n_qubits: int,
    n_layers: int,
    circuit: _QuantumCircuit,
    state: Array,
) -> Array:
    """
    Hardware-efficient ansatz.

    Alternates between single-qubit Ry rotations and CNOT ladders (linear
    topology).

    Parameters
    ----------
    params : Array
        Flat parameter array of length ``n_layers * n_qubits``.
    """
    p = params.reshape(n_layers, n_qubits)
    # Initial Hadamard layer
    H = _hadamard()
    for q in range(n_qubits):
        state = circuit.apply(state, _single_qubit_op(H, q, n_qubits))
    for layer in range(n_layers):
        for q in range(n_qubits):
            ry = _ry(p[layer, q])
            state = circuit.apply(state, _single_qubit_op(ry, q, n_qubits))
        for q in range(n_qubits - 1):
            state = circuit.apply(state, _two_qubit_cnot(q, q + 1, n_qubits))
    return state


def _build_twolocal(
    params: Array,
    n_qubits: int,
    n_layers: int,
    circuit: _QuantumCircuit,
    state: Array,
) -> Array:
    """
    Two-local ansatz (Rx-Ry rotations + CX entanglement).

    Parameters
    ----------
    params : Array
        Flat parameter array of length ``n_layers * n_qubits * 2``.
    """
    p = params.reshape(n_layers, n_qubits, 2)
    for layer in range(n_layers):
        for q in range(n_qubits):
            rx = _rx(p[layer, q, 0])
            ry = _ry(p[layer, q, 1])
            state = circuit.apply(state, _single_qubit_op(rx, q, n_qubits))
            state = circuit.apply(state, _single_qubit_op(ry, q, n_qubits))
        if n_qubits > 1:
            for q in range(0, n_qubits - 1, 2):
                state = circuit.apply(state, _two_qubit_cnot(q, q + 1, n_qubits))
            for q in range(1, n_qubits - 1, 2):
                state = circuit.apply(state, _two_qubit_cnot(q, q + 1, n_qubits))
    return state


# ---------------------------------------------------------------------------
# Adam optimiser (scratch implementation)
# ---------------------------------------------------------------------------

class _Adam:
    """
    Adam optimiser.

    Parameters
    ----------
    lr : float
        Learning rate.
    beta1 : float
        Exponential decay rate for first moment.
    beta2 : float
        Exponential decay rate for second moment.
    eps : float
        Numerical stability constant.
    """

    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.m: Optional[Array] = None
        self.v: Optional[Array] = None
        self.t: int = 0

    def step(self, params: Array, grads: Array) -> Array:
        """
        Compute one Adam update.

        Parameters
        ----------
        params : Array
            Current parameter vector.
        grads : Array
            Gradient vector.

        Returns
        -------
        Array
            Updated parameter vector.
        """
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads ** 2
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def reset(self) -> None:
        """Reset optimiser state."""
        self.m = None
        self.v = None
        self.t = 0


# ---------------------------------------------------------------------------
# VQC
# ---------------------------------------------------------------------------

class VQC:
    """
    Variational Quantum Classifier.

    Trains a parameterised quantum circuit (PQC) as a binary or multiclass
    classifier using the parameter-shift rule for gradient computation and
    the Adam optimiser.

    Parameters
    ----------
    n_qubits : int
        Number of qubits. Should satisfy 2**n_qubits >= n_features.
    n_layers : int
        Number of ansatz repetitions (depth of PQC).
    ansatz : AnsatzType
        Circuit ansatz: ``"strongly_entangling"``, ``"hardware_efficient"``,
        or ``"twolocal"``.
    n_classes : int
        Number of target classes. 2 → binary logistic, >2 → one-vs-rest.
    learning_rate : float
        Adam learning rate.
    n_epochs : int
        Training epochs.
    batch_size : int
        Mini-batch size. 0 → full-batch.
    shift : float
        Parameter-shift amount (default π/2).
    random_state : int, optional
        Seed for reproducibility.

    Examples
    --------
    >>> import numpy as np
    >>> from quantumshield_fl.src.quantum.qml.vqc import VQC
    >>> X = np.random.rand(20, 4)
    >>> y = np.random.randint(0, 2, 20)
    >>> clf = VQC(n_qubits=4, n_layers=2, n_epochs=5)
    >>> clf.fit(X, y)
    >>> preds = clf.predict(X)
    """

    _PARAM_COUNTS: Dict[str, int] = {
        "strongly_entangling": 2,   # params per qubit per layer
        "hardware_efficient":  1,
        "twolocal":            2,
    }

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        ansatz: AnsatzType = "strongly_entangling",
        n_classes: int = 2,
        learning_rate: float = 1e-2,
        n_epochs: int = 50,
        batch_size: int = 16,
        shift: float = math.pi / 2,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_qubits      = n_qubits
        self.n_layers      = n_layers
        self.ansatz        = ansatz
        self.n_classes     = n_classes
        self.learning_rate = learning_rate
        self.n_epochs      = n_epochs
        self.batch_size    = batch_size
        self.shift         = shift
        self.random_state  = random_state

        self._rng      = np.random.default_rng(random_state)
        self._circuit  = _QuantumCircuit(n_qubits)
        self._history: Dict[str, List[float]] = {
            "loss": [], "accuracy": [], "epoch_time": []
        }

        ppc = self._PARAM_COUNTS[ansatz]           # params per qubit per layer
        n_params = n_layers * n_qubits * ppc
        # For OvR we store one parameter set per class
        n_heads = n_classes if n_classes > 2 else 1
        self.params: Array = (
            self._rng.uniform(-math.pi, math.pi, (n_heads, n_params))
        )

        self._optimisers: List[_Adam] = [
            _Adam(lr=learning_rate) for _ in range(n_heads)
        ]

    # ------------------------------------------------------------------
    # Circuit execution
    # ------------------------------------------------------------------

    def _forward_single(self, x: Array, params: Array) -> float:
        """
        Run circuit on encoded state and return ⟨Z_0⟩ expectation value.

        Parameters
        ----------
        x : Array
            Amplitude-encoded state vector of length ``2**n_qubits``.
        params : Array
            Parameter vector for this circuit head.

        Returns
        -------
        float
            ⟨Z_0⟩ in [-1, 1].
        """
        state = x.astype(complex).copy()
        state = self._apply_ansatz(state, params)
        return self._circuit.measure_z(state, 0)

    def _apply_ansatz(self, state: Array, params: Array) -> Array:
        """Apply the chosen ansatz to the state."""
        builders = {
            "strongly_entangling": _build_strongly_entangling,
            "hardware_efficient":  _build_hardware_efficient,
            "twolocal":            _build_twolocal,
        }
        return builders[self.ansatz](
            params, self.n_qubits, self.n_layers, self._circuit, state
        )

    # ------------------------------------------------------------------
    # Parameter shift gradients
    # ------------------------------------------------------------------

    def _gradient(self, x: Array, params: Array) -> Array:
        """
        Compute gradients via the parameter-shift rule.

        ∂f/∂θ_i = [f(θ_i + s) - f(θ_i - s)] / (2 sin(s))

        Parameters
        ----------
        x : Array
            Input state vector.
        params : Array
            Parameter vector.

        Returns
        -------
        Array
            Gradient vector, same shape as ``params``.
        """
        grads = np.zeros_like(params)
        for i in range(len(params)):
            p_plus  = params.copy(); p_plus[i]  += self.shift
            p_minus = params.copy(); p_minus[i] -= self.shift
            f_plus  = self._forward_single(x, p_plus)
            f_minus = self._forward_single(x, p_minus)
            grads[i] = (f_plus - f_minus) / (2 * math.sin(self.shift))
        return grads

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    @staticmethod
    def _sigmoid(z: Union[float, Array]) -> Union[float, Array]:
        """Numerically stable sigmoid."""
        return np.where(z >= 0,
                        1 / (1 + np.exp(-z)),
                        np.exp(z) / (1 + np.exp(z)))

    @staticmethod
    def _binary_cross_entropy(y_true: Array, y_prob: Array, eps: float = 1e-7) -> float:
        """Binary cross-entropy loss."""
        p = np.clip(y_prob, eps, 1 - eps)
        return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))

    # ------------------------------------------------------------------
    # Encoding helper
    # ------------------------------------------------------------------

    def _encode_batch(self, X: Array) -> Array:
        """
        Amplitude-encode each sample in X into a quantum state vector.

        Parameters
        ----------
        X : Array
            Shape ``(n_samples, n_features)``.

        Returns
        -------
        Array
            Shape ``(n_samples, 2**n_qubits)``.
        """
        dim = 2 ** self.n_qubits
        states = []
        for x in X:
            padded = np.zeros(dim)
            n = min(x.size, dim)
            padded[:n] = x[:n]
            norm = np.linalg.norm(padded)
            if norm < 1e-12:
                norm = 1.0
            states.append((padded / norm).astype(complex))
        return np.stack(states, axis=0)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access to training history."""
        return self._history[key]

    def fit(
        self,
        X: Array,
        y: Array,
        n_epochs: Optional[int] = None,
        learning_rate: Optional[float] = None,
        validation_data: Optional[Tuple[Array, Array]] = None,
        verbose: bool = False,
    ) -> "VQC":
        """
        Train the VQC.

        Parameters
        ----------
        X : Array
            Feature matrix of shape ``(n_samples, n_features)``.
        y : Array
            Integer class labels of shape ``(n_samples,)``.
        n_epochs : int, optional
            Number of epochs. Overrides instance n_epochs if provided.
        learning_rate : float, optional
            Learning rate. Overrides instance learning_rate if provided.
        validation_data : tuple, optional
            ``(X_val, y_val)`` for epoch-level validation loss reporting.
        verbose : bool
            Print training progress.

        Returns
        -------
        VQC
            Self (which also supports dict-like history access).
        """
        if n_epochs is not None:
            self.n_epochs = n_epochs
        if learning_rate is not None:
            self.learning_rate = learning_rate
            for opt in self._optimisers:
                opt.lr = learning_rate

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        n_samples = X.shape[0]
        bs = self.batch_size if self.batch_size > 0 else n_samples

        states = self._encode_batch(X)

        n_heads = self.params.shape[0]
        # Map labels to binary targets per head (OvR)
        y_heads = []
        if self.n_classes == 2:
            y_heads.append(y.astype(float))
        else:
            for c in range(self.n_classes):
                y_heads.append((y == c).astype(float))

        for epoch in range(self.n_epochs):
            t0 = time.time()
            idx = self._rng.permutation(n_samples)
            epoch_loss = 0.0

            for start in range(0, n_samples, bs):
                batch_idx = idx[start:start + bs]
                X_b = states[batch_idx]
                n_b = len(batch_idx)

                for head in range(n_heads):
                    y_b = y_heads[head][batch_idx]
                    grad_sum = np.zeros_like(self.params[head])
                    preds = []

                    for xi, yi in zip(X_b, y_b):
                        ev = self._forward_single(xi, self.params[head])
                        prob = float(self._sigmoid((ev + 1) / 2 * 4 - 2))
                        preds.append(prob)
                        # Per-sample gradient
                        g = self._gradient(xi, self.params[head])
                        # Chain-rule: dL/dθ = dL/dσ * dσ/dθ (approx sign)
                        dL = prob - yi
                        grad_sum += dL * g

                    grad_sum /= n_b
                    self.params[head] = self._optimisers[head].step(
                        self.params[head], grad_sum
                    )
                    epoch_loss += self._binary_cross_entropy(y_b, np.array(preds))

            epoch_loss /= (n_heads * math.ceil(n_samples / bs))
            acc = float(np.mean(self.predict(X) == y))
            elapsed = time.time() - t0

            self._history["loss"].append(epoch_loss)
            self._history["accuracy"].append(acc)
            self._history["epoch_time"].append(elapsed)

            if verbose:
                val_str = ""
                if validation_data is not None:
                    X_v, y_v = validation_data
                    val_acc = float(np.mean(self.predict(X_v) == y_v))
                    val_str = f" | val_acc={val_acc:.4f}"
                print(
                    f"Epoch {epoch+1:03d}/{self.n_epochs} "
                    f"loss={epoch_loss:.4f} acc={acc:.4f}"
                    f"{val_str} [{elapsed:.1f}s]"
                )

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(self, X: Array) -> Array:
        """
        Return class probability estimates.

        Parameters
        ----------
        X : Array
            Feature matrix of shape ``(n_samples, n_features)``.

        Returns
        -------
        Array
            Shape ``(n_samples, n_classes)``. Rows sum to 1.
        """
        X = np.asarray(X, dtype=float)
        states = self._encode_batch(X)
        n_heads = self.params.shape[0]

        raw = np.zeros((X.shape[0], n_heads))
        for head in range(n_heads):
            for i, xi in enumerate(states):
                ev = self._forward_single(xi, self.params[head])
                raw[i, head] = float(self._sigmoid((ev + 1) / 2 * 4 - 2))

        if self.n_classes == 2:
            pos = raw[:, 0]
            return np.column_stack([1 - pos, pos])
        else:
            # Softmax over OvR scores
            exp_r = np.exp(raw - raw.max(axis=1, keepdims=True))
            return exp_r / exp_r.sum(axis=1, keepdims=True)

    def predict(self, X: Array) -> Array:
        """
        Predict class labels.

        Parameters
        ----------
        X : Array
            Feature matrix.

        Returns
        -------
        Array
            Integer class predictions of shape ``(n_samples,)``.
        """
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X: Array, y: Array) -> float:
        """
        Return classification accuracy.

        Parameters
        ----------
        X : Array
            Feature matrix.
        y : Array
            True labels.

        Returns
        -------
        float
            Fraction of correct predictions.
        """
        return float(np.mean(self.predict(X) == np.asarray(y, dtype=int)))

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """
        Save model parameters and configuration to a JSON file.

        Parameters
        ----------
        path : str or Path
            Destination file path (should end with ``.json``).
        """
        data: Dict[str, Any] = {
            "config": {
                "n_qubits":      self.n_qubits,
                "n_layers":      self.n_layers,
                "ansatz":        self.ansatz,
                "n_classes":     self.n_classes,
                "learning_rate": self.learning_rate,
                "n_epochs":      self.n_epochs,
                "batch_size":    self.batch_size,
                "shift":         self.shift,
                "random_state":  self.random_state,
            },
            "params":  self.params.tolist(),
            "history": self._history,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def load(self, path: Union[str, Path]) -> "VQC":
        """
        Load model parameters and history from a JSON file into this instance.

        Parameters
        ----------
        path : str or Path
            Source file path.

        Returns
        -------
        VQC
            Self with updated parameters and history.
        """
        data = json.loads(Path(path).read_text())
        self.params = np.array(data["params"])
        self._history = data.get("history", {})
        return self

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "VQC":
        """Construct a new VQC instance from a saved JSON file."""
        data = json.loads(Path(path).read_text())
        model = cls(**data["config"])
        model.params = np.array(data["params"])
        model._history = data.get("history", {})
        return model

    @property
    def history(self) -> Dict[str, List[float]]:
        """Training history dict with keys ``loss``, ``accuracy``, ``epoch_time``."""
        return self._history

    def __repr__(self) -> str:
        return (
            f"VQC(n_qubits={self.n_qubits}, n_layers={self.n_layers}, "
            f"ansatz='{self.ansatz}', n_classes={self.n_classes})"
        )
