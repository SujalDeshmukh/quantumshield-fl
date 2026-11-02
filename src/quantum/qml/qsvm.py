"""
Quantum Support Vector Machine (QSVM) for QuantumShield-FL.

Computes quantum kernels via feature map circuits (ZZFeatureMap and
PauliFeatureMap) and trains a kernel SVM in the dual formulation using
Sequential Minimal Optimisation (SMO) implemented with NumPy/SciPy.

Key components:

    - ``ZZFeatureMap``   – data-dependent phase kick-back feature map
    - ``PauliFeatureMap`` – configurable Pauli-basis feature map
    - ``QSVM``           – assembles the kernel matrix and fits the SVM dual
    - Kernel alignment score for hyperparameter selection
"""

from __future__ import annotations

import math
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize

Array = np.ndarray
KernelType = Literal["zz", "pauli"]


# ---------------------------------------------------------------------------
# Feature map helpers
# ---------------------------------------------------------------------------

def _hadamard() -> Array:
    return np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)


def _embed_gate(gate: Array, q: int, n: int) -> Array:
    """Embed a 2x2 gate as a single-qubit gate in n-qubit space."""
    ops = [np.eye(2, dtype=complex)] * n
    ops[q] = gate
    result = ops[0]
    for m in ops[1:]:
        result = np.kron(result, m)
    return result


def _zz_phase(q1: int, q2: int, angle: float, n: int) -> Array:
    """Return the unitary for exp(i * angle * Z_q1 ⊗ Z_q2)."""
    dim = 2 ** n
    U = np.eye(dim, dtype=complex)
    m1 = 1 << (n - 1 - q1)
    m2 = 1 << (n - 1 - q2)
    for idx in range(dim):
        b1 = 1 if (idx & m1) else -1
        b2 = 1 if (idx & m2) else -1
        U[idx, idx] = np.exp(1j * angle * b1 * b2)
    return U


def _single_z_phase(q: int, angle: float, n: int) -> Array:
    """Return the unitary for exp(i * angle * Z_q)."""
    dim = 2 ** n
    U = np.eye(dim, dtype=complex)
    mask = 1 << (n - 1 - q)
    for idx in range(dim):
        b = 1 if (idx & mask) else -1
        U[idx, idx] = np.exp(1j * angle * b)
    return U


# ---------------------------------------------------------------------------
# ZZFeatureMap
# ---------------------------------------------------------------------------

class ZZFeatureMap:
    """
    ZZ Feature Map (inspired by Havlíček et al., Nature 2019).

    Applies:
        H^⊗n → Rz(2*x_i) → Rzz(2*(π-x_i)(π-x_j)) on pairs → [repeat reps]

    Parameters
    ----------
    n_qubits : int
        Number of qubits (= feature dimension).
    reps : int
        Number of repetitions of the feature map.
    entanglement : str
        ``'linear'`` (chain pairs) or ``'full'`` (all pairs).
    """

    def __init__(
        self,
        n_qubits: int,
        reps: int = 2,
        entanglement: str = "linear",
    ) -> None:
        self.n_qubits     = n_qubits
        self.reps         = reps
        self.entanglement = entanglement
        self.dim          = 2 ** n_qubits
        H = _hadamard()
        # Pre-build Hadamard layer
        H_all = np.eye(self.dim, dtype=complex)
        for q in range(n_qubits):
            H_all = _embed_gate(H, q, n_qubits) @ H_all
        self._H_all = H_all

    def _pairs(self) -> List[Tuple[int, int]]:
        n = self.n_qubits
        if self.entanglement == "linear":
            return [(i, i + 1) for i in range(n - 1)]
        # full
        return [(i, j) for i in range(n) for j in range(i + 1, n)]

    def get_state(self, x: Array) -> Array:
        """
        Compute the feature-mapped quantum state |ψ(x)⟩.

        Parameters
        ----------
        x : Array
            Feature vector of length ``n_qubits``.

        Returns
        -------
        Array
            Complex state vector of shape ``(2**n_qubits,)``.
        """
        x = np.asarray(x, dtype=float).ravel()
        n = self.n_qubits
        # Pad / trim
        x_pad = np.zeros(n)
        x_pad[: min(x.size, n)] = x[: n]

        state = np.zeros(self.dim, dtype=complex)
        state[0] = 1.0

        for _ in range(self.reps):
            # Hadamard
            state = self._H_all @ state
            # Single-qubit Z phases
            for q in range(n):
                state = _single_z_phase(q, 2 * x_pad[q], n) @ state
            # ZZ phases on pairs
            for (q1, q2) in self._pairs():
                angle = 2 * (math.pi - x_pad[q1]) * (math.pi - x_pad[q2])
                state = _zz_phase(q1, q2, angle, n) @ state

        return state

    def kernel(self, x1: Array, x2: Array) -> float:
        """
        Quantum kernel k(x1, x2) = |⟨ψ(x1)|ψ(x2)⟩|².

        Parameters
        ----------
        x1, x2 : Array
            Feature vectors.

        Returns
        -------
        float
            Kernel value in [0, 1].
        """
        s1 = self.get_state(x1)
        s2 = self.get_state(x2)
        return float(np.abs(np.dot(s1.conj(), s2)) ** 2)


# ---------------------------------------------------------------------------
# PauliFeatureMap
# ---------------------------------------------------------------------------

class PauliFeatureMap:
    """
    Pauli Feature Map — generalised version supporting configurable Pauli terms.

    For each repetition:
        H^⊗n → exp(i * x_i * P_i) for each qubit
             → exp(i * x_i * x_j * P_i ⊗ P_j) for neighbouring pairs.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    reps : int
        Number of repetitions.
    paulis : list of str
        Pauli operators to use per qubit. Each entry is ``'X'``, ``'Y'``,
        or ``'Z'``. Default: ``['Z', 'ZZ']`` (Z + ZZ interaction).
    """

    _PAULI: Dict[str, Array] = {
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
        "I": np.eye(2, dtype=complex),
    }

    def __init__(
        self,
        n_qubits: int,
        reps: int = 2,
        paulis: Optional[List[str]] = None,
    ) -> None:
        self.n_qubits = n_qubits
        self.reps     = reps
        self.paulis   = paulis or ["Z", "ZZ"]
        self.dim      = 2 ** n_qubits
        H = _hadamard()
        H_all = np.eye(self.dim, dtype=complex)
        for q in range(n_qubits):
            H_all = _embed_gate(H, q, n_qubits) @ H_all
        self._H_all = H_all

    def _pauli_exp(self, op: str, q: int, angle: float) -> Array:
        """exp(i * angle * P_q)."""
        P = self._PAULI[op]
        # exp(i*angle*P) = cos(angle)*I + i*sin(angle)*P  (since P^2 = I)
        c, s = math.cos(angle), math.sin(angle)
        gate = c * np.eye(2, dtype=complex) + 1j * s * P
        return _embed_gate(gate, q, self.n_qubits)

    def get_state(self, x: Array) -> Array:
        """
        Compute the Pauli feature-mapped state |ψ(x)⟩.

        Parameters
        ----------
        x : Array
            Feature vector of length ``n_qubits``.

        Returns
        -------
        Array
            Complex state vector.
        """
        x = np.asarray(x, dtype=float).ravel()
        n = self.n_qubits
        x_pad = np.zeros(n)
        x_pad[: min(x.size, n)] = x[:n]

        state = np.zeros(self.dim, dtype=complex)
        state[0] = 1.0

        single_pauli = [p for p in self.paulis if len(p) == 1]
        two_pauli    = [p for p in self.paulis if len(p) == 2]

        for _ in range(self.reps):
            state = self._H_all @ state
            # Single-qubit Pauli rotations
            for pauli_op in single_pauli:
                p = pauli_op[0]
                for q in range(n):
                    U = self._pauli_exp(p, q, x_pad[q])
                    state = U @ state
            # Two-qubit Pauli interactions
            for pauli_op in two_pauli:
                p1, p2 = pauli_op[0], pauli_op[1]
                for q in range(n - 1):
                    angle = x_pad[q] * x_pad[q + 1]
                    # Approximate: exp(i*angle*(P_q ⊗ P_{q+1})) as sequential gates
                    U1 = self._pauli_exp(p1, q,     angle)
                    U2 = self._pauli_exp(p2, q + 1, angle)
                    state = U2 @ U1 @ state

        return state

    def kernel(self, x1: Array, x2: Array) -> float:
        """
        Quantum kernel k(x1, x2) = |⟨ψ(x1)|ψ(x2)⟩|².

        Parameters
        ----------
        x1, x2 : Array
        Returns
        -------
        float
        """
        s1 = self.get_state(x1)
        s2 = self.get_state(x2)
        return float(np.abs(np.dot(s1.conj(), s2)) ** 2)


# ---------------------------------------------------------------------------
# Kernel matrix builder
# ---------------------------------------------------------------------------

def build_kernel_matrix(
    feature_map: Union[ZZFeatureMap, PauliFeatureMap],
    X: Array,
    Y: Optional[Array] = None,
) -> Array:
    """
    Construct the Gram (kernel) matrix K[i, j] = k(x_i, y_j).

    Parameters
    ----------
    feature_map : ZZFeatureMap or PauliFeatureMap
        Quantum feature map with a ``kernel`` method.
    X : Array
        Training set of shape ``(n, d)``.
    Y : Array, optional
        Second set of shape ``(m, d)``. If ``None``, compute K(X, X).

    Returns
    -------
    Array
        Kernel matrix of shape ``(n, m)`` or ``(n, n)``.
    """
    symmetric = Y is None
    if symmetric:
        Y = X
    n, m = len(X), len(Y)
    K = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            if symmetric and j < i:
                K[i, j] = K[j, i]
            else:
                K[i, j] = feature_map.kernel(X[i], Y[j])
    return K


# ---------------------------------------------------------------------------
# Kernel alignment
# ---------------------------------------------------------------------------

def kernel_alignment(K1: Array, K2: Array) -> float:
    """
    Compute the (centered) kernel alignment score between two kernel matrices.

    Alignment(K1, K2) = ⟨K1, K2⟩_F / (‖K1‖_F * ‖K2‖_F)

    Parameters
    ----------
    K1, K2 : Array
        Kernel matrices of the same shape.

    Returns
    -------
    float
        Alignment score in [-1, 1].
    """
    n = K1.shape[0]
    # Center: K̃ = K - 1/n * 1 K - 1/n * K 1 + 1/n^2 * 1 K 1
    def center(K: Array) -> Array:
        col_mean = K.mean(axis=0, keepdims=True)
        row_mean = K.mean(axis=1, keepdims=True)
        grand    = K.mean()
        return K - col_mean - row_mean + grand

    K1c = center(K1)
    K2c = center(K2)
    numerator   = np.sum(K1c * K2c)
    denominator = math.sqrt(np.sum(K1c ** 2) * np.sum(K2c ** 2) + 1e-12)
    return float(numerator / denominator)


# ---------------------------------------------------------------------------
# Dual SVM (quadratic programming via scipy.optimize.minimize with L-BFGS-B)
# ---------------------------------------------------------------------------

class _DualSVM:
    """
    SVM trained in the dual formulation (kernel SVM).

    Maximise: Σ α_i - 0.5 Σ_ij α_i α_j y_i y_j K_ij
    Subject to: 0 ≤ α_i ≤ C, Σ α_i y_i = 0

    Parameters
    ----------
    C : float
        Regularisation strength.
    tol : float
        Convergence tolerance.
    """

    def __init__(self, C: float = 1.0, tol: float = 1e-4) -> None:
        self.C    = C
        self.tol  = tol
        self.alphas_: Optional[Array] = None
        self.bias_:   float           = 0.0
        self.sv_idx_: Optional[Array] = None

    def fit(self, K: Array, y: Array) -> "_DualSVM":
        """
        Fit SVM given pre-computed kernel matrix.

        Parameters
        ----------
        K : Array
            Kernel matrix ``(n, n)``.
        y : Array
            Binary labels ``{-1, +1}`` of shape ``(n,)``.

        Returns
        -------
        self
        """
        n = len(y)

        def objective(alphas: Array) -> float:
            return 0.5 * alphas @ (np.outer(y, y) * K) @ alphas - alphas.sum()

        def jac(alphas: Array) -> Array:
            return (np.outer(y, y) * K) @ alphas - np.ones(n)

        # Equality constraint: Σ α_i y_i = 0
        constraints = {
            "type": "eq",
            "fun":  lambda a: np.dot(a, y),
            "jac":  lambda a: y,
        }
        bounds = [(0, self.C)] * n
        alpha0 = np.zeros(n)

        result = minimize(
            objective,
            alpha0,
            jac=jac,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": self.tol, "maxiter": 1000},
        )

        self.alphas_ = result.x
        # Support vectors: α_i > tol
        sv_mask = self.alphas_ > self.tol
        self.sv_idx_ = np.where(sv_mask)[0]

        # Compute bias from support vectors that are not at upper bound
        free_sv = np.where(sv_mask & (self.alphas_ < self.C - self.tol))[0]
        if len(free_sv) == 0:
            free_sv = self.sv_idx_
        if len(free_sv) > 0:
            biases = y[free_sv] - (
                self.alphas_[sv_mask] * y[sv_mask]
            ) @ K[np.ix_(sv_mask, free_sv)]
            self.bias_ = float(biases.mean())
        else:
            self.bias_ = 0.0

        return self

    def decision_function(self, K_test: Array, y_train: Array) -> Array:
        """
        Compute decision scores for test samples.

        Parameters
        ----------
        K_test : Array
            ``(n_test, n_train)`` kernel matrix.
        y_train : Array
            Training labels.

        Returns
        -------
        Array
            Decision scores of shape ``(n_test,)``.
        """
        sv = self.sv_idx_
        return (
            K_test[:, sv] @ (self.alphas_[sv] * y_train[sv]) + self.bias_
        )

    def predict(self, K_test: Array, y_train: Array) -> Array:
        """Return predicted labels {-1, +1}."""
        return np.sign(self.decision_function(K_test, y_train))


# ---------------------------------------------------------------------------
# QSVM
# ---------------------------------------------------------------------------

class QSVM:
    """
    Quantum Support Vector Machine.

    Wraps a quantum feature map (``ZZFeatureMap`` or ``PauliFeatureMap``)
    with a kernel SVM classifier trained in the dual.

    Parameters
    ----------
    n_qubits : int
        Number of qubits = input feature dimension.
    kernel_type : str
        ``'zz'`` for ZZFeatureMap or ``'pauli'`` for PauliFeatureMap.
    reps : int
        Feature map repetition depth.
    C : float
        SVM regularisation parameter.
    entanglement : str
        Entanglement pattern for ZZFeatureMap (``'linear'`` or ``'full'``).
    paulis : list of str, optional
        Pauli operator list for PauliFeatureMap (default ``['Z', 'ZZ']``).

    Examples
    --------
    >>> qsvm = QSVM(n_qubits=4, kernel_type='zz', reps=1, C=1.0)
    >>> qsvm.fit(X_train, y_train)
    >>> qsvm.predict(X_test)
    """

    def __init__(
        self,
        n_qubits: int = 4,
        kernel_type: KernelType = "zz",
        reps: int = 1,
        C: float = 1.0,
        entanglement: str = "linear",
        paulis: Optional[List[str]] = None,
    ) -> None:
        self.n_qubits     = n_qubits
        self.kernel_type  = kernel_type
        self.reps         = reps
        self.C            = C
        self.entanglement = entanglement
        self.paulis       = paulis or ["Z", "ZZ"]

        if kernel_type == "zz":
            self.feature_map: Union[ZZFeatureMap, PauliFeatureMap] = ZZFeatureMap(
                n_qubits, reps=reps, entanglement=entanglement
            )
        else:
            self.feature_map = PauliFeatureMap(
                n_qubits, reps=reps, paulis=self.paulis
            )

        self._svm   = _DualSVM(C=C)
        self._X_train: Optional[Array] = None
        self._y_train: Optional[Array] = None
        self._K_train: Optional[Array] = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X: Array, y: Array) -> "QSVM":
        """
        Build the quantum kernel matrix and fit the dual SVM.

        Parameters
        ----------
        X : Array
            Training data of shape ``(n_samples, n_features)``.
        y : Array
            Binary labels (0/1 or -1/+1). Will be converted to {-1, +1}.

        Returns
        -------
        QSVM
            Self.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        # Convert 0/1 labels to -1/+1 if needed
        if set(np.unique(y.astype(int))).issubset({0, 1}):
            y = 2 * y - 1

        self._X_train = X
        self._y_train = y

        print("Building quantum kernel matrix …")
        self._K_train = build_kernel_matrix(self.feature_map, X)
        print("Fitting dual SVM …")
        self._svm.fit(self._K_train, y)
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, X: Array) -> Array:
        """
        Predict class labels for test samples.

        Parameters
        ----------
        X : Array
            Test data of shape ``(n_test, n_features)``.

        Returns
        -------
        Array
            Predicted labels (0 or 1) of shape ``(n_test,)``.
        """
        K_test = build_kernel_matrix(self.feature_map, np.asarray(X, dtype=float), self._X_train)
        raw = self._svm.predict(K_test, self._y_train)
        return ((raw + 1) / 2).astype(int)  # map -1/+1 → 0/1

    def decision_function(self, X: Array) -> Array:
        """
        Return raw decision scores.

        Parameters
        ----------
        X : Array
            Test data.

        Returns
        -------
        Array
            Scores of shape ``(n_test,)``.
        """
        K_test = build_kernel_matrix(self.feature_map, np.asarray(X, dtype=float), self._X_train)
        return self._svm.decision_function(K_test, self._y_train)

    def score(self, X: Array, y: Array) -> float:
        """
        Return classification accuracy.

        Parameters
        ----------
        X : Array
        y : Array

        Returns
        -------
        float
        """
        y = np.asarray(y, dtype=int)
        if set(np.unique(y)).issubset({-1, 1}):
            y = ((y + 1) / 2).astype(int)
        return float(np.mean(self.predict(X) == y))

    def kernel_alignment_score(self, X: Array, y: Array) -> float:
        """
        Compute the kernel alignment between the quantum kernel and the
        ideal kernel matrix (y_i * y_j).

        Parameters
        ----------
        X : Array
            Data for which to build the kernel matrix.
        y : Array
            True labels.

        Returns
        -------
        float
            Alignment score in [-1, 1].
        """
        y = np.asarray(y, dtype=float)
        if set(np.unique(y.astype(int))).issubset({0, 1}):
            y = 2 * y - 1
        K_q = build_kernel_matrix(self.feature_map, np.asarray(X, dtype=float))
        K_ideal = np.outer(y, y)
        return kernel_alignment(K_q, K_ideal)

    def __repr__(self) -> str:
        return (
            f"QSVM(n_qubits={self.n_qubits}, kernel='{self.kernel_type}', "
            f"reps={self.reps}, C={self.C})"
        )
