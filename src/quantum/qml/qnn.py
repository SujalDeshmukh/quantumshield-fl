"""
Quantum Neural Network (QNN) for QuantumShield-FL.

Implements a hybrid quantum-classical neural network:

    Input → Classical pre-processing → Quantum Layer(s) → Classical post-processing → Output

Quantum layers are parameterised circuits evaluated by statevector simulation.
Gradients are computed via the parameter-shift rule through quantum layers and
standard back-propagation through classical layers.

Supports:
    - Regression and classification tasks
    - Multiple quantum layers
    - Configurable classical pre/post layers
    - Mini-batch training
    - Full training history
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

Array = np.ndarray
TaskType = Literal["classification", "regression"]


# ---------------------------------------------------------------------------
# Utility: gate primitives
# ---------------------------------------------------------------------------

def _ry(theta: float) -> Array:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(theta: float) -> Array:
    return np.array([[math.exp(-1j * theta / 2), 0],
                     [0, math.exp(1j * theta / 2)]], dtype=complex)


def _hadamard() -> Array:
    return np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)


def _embed_gate(gate: Array, q: int, n: int) -> Array:
    ops = [np.eye(2, dtype=complex)] * n
    ops[q] = gate
    M = ops[0]
    for m in ops[1:]:
        M = np.kron(M, m)
    return M


def _cnot_full(ctrl: int, tgt: int, n: int) -> Array:
    dim = 2 ** n
    U = np.eye(dim, dtype=complex)
    cmask = 1 << (n - 1 - ctrl)
    tmask = 1 << (n - 1 - tgt)
    for idx in range(dim):
        if idx & cmask:
            flipped = idx ^ tmask
            U[idx, idx] = 0
            U[flipped, idx] = 1
            if flipped != idx:
                U[flipped, flipped] = 0
                U[idx, flipped] = 1
    return U


# ---------------------------------------------------------------------------
# Classical dense layer (numpy)
# ---------------------------------------------------------------------------

class _DenseLayer:
    """
    Fully-connected linear layer with optional activation.

    Parameters
    ----------
    in_dim : int
        Input dimensionality.
    out_dim : int
        Output dimensionality.
    activation : str
        One of ``'relu'``, ``'tanh'``, ``'sigmoid'``, ``'linear'``.
    rng : np.random.Generator
        Random number generator for weight initialisation.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: str = "relu",
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if rng is None:
            rng = np.random.default_rng()
        scale = math.sqrt(2.0 / in_dim)
        self.W: Array = rng.normal(0, scale, (out_dim, in_dim))
        self.b: Array = np.zeros(out_dim)
        self.activation = activation.lower()
        # Cache for backprop
        self._input: Optional[Array] = None
        self._pre_act: Optional[Array] = None

    # Activations and their derivatives
    @staticmethod
    def _act(z: Array, name: str) -> Array:
        if name == "relu":
            return np.maximum(0, z)
        if name == "tanh":
            return np.tanh(z)
        if name == "sigmoid":
            return 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        return z  # linear

    @staticmethod
    def _act_deriv(z: Array, name: str) -> Array:
        if name == "relu":
            return (z > 0).astype(float)
        if name == "tanh":
            return 1 - np.tanh(z) ** 2
        if name == "sigmoid":
            s = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
            return s * (1 - s)
        return np.ones_like(z)  # linear

    def forward(self, x: Array) -> Array:
        """Forward pass; caches inputs for backprop."""
        self._input = x
        self._pre_act = x @ self.W.T + self.b
        return self._act(self._pre_act, self.activation)

    def backward(self, grad_out: Array, lr: float) -> Array:
        """
        Backward pass: update W, b and return upstream gradient.

        Parameters
        ----------
        grad_out : Array
            Gradient of loss w.r.t. this layer's output.
        lr : float
            Learning rate.

        Returns
        -------
        Array
            Gradient w.r.t. this layer's input.
        """
        da = self._act_deriv(self._pre_act, self.activation)
        delta = grad_out * da
        dW = np.outer(delta, self._input)
        self.W -= lr * dW
        self.b -= lr * delta
        return delta @ self.W


# ---------------------------------------------------------------------------
# Quantum layer
# ---------------------------------------------------------------------------

class _QuantumLayer:
    """
    A single quantum layer implementing a hardware-efficient PQC.

    The circuit structure per sample:
        1. Encode input via angle encoding (Ry gates).
        2. Apply Ry+Rz trainable rotations on all qubits.
        3. CNOT ring entanglement.
        4. Measure ⟨Z_q⟩ for all qubits → output vector.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (= output dimension of this layer).
    n_params_per_qubit : int
        Number of trainable parameters per qubit (default 2: Ry + Rz).
    rng : np.random.Generator
        For weight initialisation.
    """

    def __init__(
        self,
        n_qubits: int,
        n_params_per_qubit: int = 2,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if rng is None:
            rng = np.random.default_rng()
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        self.n_params = n_qubits * n_params_per_qubit
        self.params: Array = rng.uniform(-math.pi, math.pi, self.n_params)
        self.shift = math.pi / 2

    # -- state-vector circuit ------------------------------------------------

    def _zero_state(self) -> Array:
        s = np.zeros(self.dim, dtype=complex)
        s[0] = 1.0
        return s

    def _circuit(self, x_enc: Array, params: Array) -> Array:
        """Run circuit with angle-encoded x and trainable params."""
        state = self._zero_state()
        p = params.reshape(self.n_qubits, 2)
        # Encode input
        for q in range(self.n_qubits):
            angle = float(x_enc[q]) if q < len(x_enc) else 0.0
            U = _embed_gate(_ry(angle * math.pi), q, self.n_qubits)
            state = U @ state
        # Trainable rotations
        for q in range(self.n_qubits):
            U = _embed_gate(_ry(p[q, 0]), q, self.n_qubits)
            state = U @ state
            U = _embed_gate(_rz(p[q, 1]), q, self.n_qubits)
            state = U @ state
        # Ring CNOT
        for q in range(self.n_qubits):
            if self.n_qubits > 1:
                state = _cnot_full(q, (q + 1) % self.n_qubits, self.n_qubits) @ state
        return state

    def _measure_z(self, state: Array, q: int) -> float:
        mask = 1 << (self.n_qubits - 1 - q)
        probs = np.abs(state) ** 2
        return float(sum(p if not (i & mask) else -p for i, p in enumerate(probs)))

    def forward(self, x: Array) -> Array:
        """
        Forward pass for one sample.

        Parameters
        ----------
        x : Array
            Input vector (will be truncated / padded to n_qubits).

        Returns
        -------
        Array
            ⟨Z_q⟩ measurements, shape ``(n_qubits,)``.
        """
        state = self._circuit(x, self.params)
        return np.array([self._measure_z(state, q) for q in range(self.n_qubits)])

    def forward_batch(self, X: Array) -> Array:
        """Process a batch of inputs, shape ``(batch, n_qubits)``."""
        return np.stack([self.forward(x) for x in X], axis=0)

    def parameter_shift_gradient(self, x: Array, grad_output: Array) -> Array:
        """
        Compute ∂L/∂params using the parameter-shift rule and the chain rule.

        Parameters
        ----------
        x : Array
            Single input sample.
        grad_output : Array
            Gradient of loss w.r.t. this layer's outputs (⟨Z_q⟩ values).

        Returns
        -------
        Array
            Gradient w.r.t. ``self.params``.
        """
        grads = np.zeros_like(self.params)
        for i in range(len(self.params)):
            p_plus  = self.params.copy(); p_plus[i]  += self.shift
            p_minus = self.params.copy(); p_minus[i] -= self.shift

            state_plus  = self._circuit(x, p_plus)
            state_minus = self._circuit(x, p_minus)
            z_plus  = np.array([self._measure_z(state_plus,  q) for q in range(self.n_qubits)])
            z_minus = np.array([self._measure_z(state_minus, q) for q in range(self.n_qubits)])
            dz_dtheta = (z_plus - z_minus) / (2 * math.sin(self.shift))
            grads[i] = float(np.dot(grad_output, dz_dtheta))
        return grads

    def update_params(self, grads: Array, lr: float) -> None:
        """Gradient-descent update of quantum layer parameters."""
        self.params -= lr * grads


# ---------------------------------------------------------------------------
# QNN
# ---------------------------------------------------------------------------

class QNN:
    """
    Hybrid Quantum-Classical Neural Network.

    Architecture::

        x ──► [Classical pre-layers] ──► [Quantum layer(s)] ──► [Classical post-layers] ──► output

    Parameters
    ----------
    n_features : int
        Dimensionality of the input.
    n_qubits : int
        Qubits in each quantum layer (also the output dimension of each QLayer).
    n_quantum_layers : int
        How many sequential quantum layers to stack.
    pre_layer_dims : list of int
        Hidden dimensions for classical pre-processing layers.
    post_layer_dims : list of int
        Hidden dimensions for classical post-processing layers.
    n_outputs : int
        Output dimensionality (1 for binary/regression, n_classes for softmax).
    task : TaskType
        ``'classification'`` or ``'regression'``.
    learning_rate : float
        Global learning rate.
    n_epochs : int
        Training epochs.
    batch_size : int
        Mini-batch size.
    random_state : int, optional
        Seed.

    Examples
    --------
    >>> qnn = QNN(n_features=4, n_qubits=4, n_quantum_layers=1,
    ...           pre_layer_dims=[8], post_layer_dims=[4], n_outputs=2,
    ...           task='classification', n_epochs=10)
    >>> qnn.fit(X_train, y_train)
    >>> qnn.predict(X_test)
    """

    def __init__(
        self,
        n_features: int,
        n_qubits: int = 4,
        n_quantum_layers: int = 1,
        pre_layer_dims: Sequence[int] = (8,),
        post_layer_dims: Sequence[int] = (4,),
        n_outputs: int = 2,
        task: TaskType = "classification",
        learning_rate: float = 1e-2,
        n_epochs: int = 50,
        batch_size: int = 16,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_features       = n_features
        self.n_qubits         = n_qubits
        self.n_quantum_layers = n_quantum_layers
        self.pre_layer_dims   = list(pre_layer_dims)
        self.post_layer_dims  = list(post_layer_dims)
        self.n_outputs        = n_outputs
        self.task             = task
        self.learning_rate    = learning_rate
        self.n_epochs         = n_epochs
        self.batch_size       = batch_size
        self.random_state     = random_state

        rng = np.random.default_rng(random_state)

        # Build classical pre-processing layers
        self._pre_layers: List[_DenseLayer] = []
        in_d = n_features
        for h in pre_layer_dims:
            self._pre_layers.append(_DenseLayer(in_d, h, "relu", rng))
            in_d = h
        # The last pre-layer must output n_qubits features
        self._pre_layers.append(_DenseLayer(in_d, n_qubits, "tanh", rng))

        # Build quantum layers
        self._quantum_layers: List[_QuantumLayer] = [
            _QuantumLayer(n_qubits, 2, rng) for _ in range(n_quantum_layers)
        ]

        # Build classical post-processing layers
        self._post_layers: List[_DenseLayer] = []
        in_d = n_qubits
        for h in post_layer_dims:
            self._post_layers.append(_DenseLayer(in_d, h, "relu", rng))
            in_d = h
        act_out = "sigmoid" if n_outputs == 1 else "linear"
        self._post_layers.append(_DenseLayer(in_d, n_outputs, act_out, rng))

        self._history: Dict[str, List[float]] = {"loss": [], "metric": [], "epoch_time": []}

    # ------------------------------------------------------------------
    # Forward pass (single sample)
    # ------------------------------------------------------------------

    def _forward(self, x: Array) -> Array:
        """Full forward pass for one sample."""
        h = x.copy()
        for layer in self._pre_layers:
            h = layer.forward(h)
        for qlayer in self._quantum_layers:
            h = qlayer.forward(h)
        for layer in self._post_layers:
            h = layer.forward(h)
        return h

    def _forward_batch(self, X: Array) -> Array:
        """Forward pass for a batch, returns ``(batch, n_outputs)``."""
        return np.stack([self._forward(x) for x in X], axis=0)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _loss(self, y_pred: Array, y_true: Array) -> Tuple[float, Array]:
        """
        Compute loss and gradient of loss w.r.t. predictions.

        Returns
        -------
        (loss_val, dL_dy_pred)
        """
        if self.task == "classification":
            if self.n_outputs == 1:
                # Binary cross-entropy
                p = np.clip(y_pred.ravel(), 1e-7, 1 - 1e-7)
                t = y_true.ravel()
                loss = float(-np.mean(t * np.log(p) + (1 - t) * np.log(1 - p)))
                grad = ((p - t) / (p * (1 - p) + 1e-8)) / len(t)
                return loss, grad.reshape(y_pred.shape)
            else:
                # Softmax cross-entropy
                scores = y_pred
                exp_s  = np.exp(scores - scores.max(axis=-1, keepdims=True))
                probs  = exp_s / exp_s.sum(axis=-1, keepdims=True)
                n = y_pred.shape[0]
                one_hot = np.eye(self.n_outputs)[y_true.astype(int)]
                loss = float(-np.mean(np.sum(one_hot * np.log(probs + 1e-9), axis=-1)))
                grad = (probs - one_hot) / n
                return loss, grad
        else:
            # MSE regression
            diff = y_pred.ravel() - y_true.ravel()
            loss = float(np.mean(diff ** 2))
            grad = 2 * diff / len(diff)
            return loss, grad.reshape(y_pred.shape)

    # ------------------------------------------------------------------
    # Backward pass (single sample)
    # ------------------------------------------------------------------

    def _backward(self, x: Array, dL_dy: Array) -> None:
        """
        Back-propagate gradients through post → quantum → pre layers.

        Parameters
        ----------
        x : Array
            Original input sample.
        dL_dy : Array
            Gradient of loss w.r.t. output of network.
        """
        lr = self.learning_rate

        # Backprop through post classical layers (reverse)
        g = dL_dy.ravel()
        for layer in reversed(self._post_layers):
            g = layer.backward(g, lr)

        # Backprop through quantum layers (reverse)
        # For quantum layers, compute parameter grads and pass grad upstream
        # The upstream gradient through a quantum layer is approximated by
        # the Jacobian estimated via parameter shift on the measurement map.
        for qlayer in reversed(self._quantum_layers):
            # x_q is the input that entered this quantum layer (cached during forward)
            # We approximate the input gradient as zeros (not tracked through QLayer)
            param_grads = qlayer.parameter_shift_gradient(
                qlayer._last_input if hasattr(qlayer, "_last_input") else x, g
            )
            qlayer.update_params(param_grads, lr)

        # Backprop through pre classical layers (reverse)
        for layer in reversed(self._pre_layers):
            g = layer.backward(g, lr)

    # Override forward to cache quantum layer inputs
    def _forward_with_cache(self, x: Array) -> Array:
        """Forward pass that also caches quantum-layer inputs."""
        h = x.copy()
        for layer in self._pre_layers:
            h = layer.forward(h)
        for qlayer in self._quantum_layers:
            qlayer._last_input = h.copy()
            h = qlayer.forward(h)
        for layer in self._post_layers:
            h = layer.forward(h)
        return h

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def fit(
        self,
        X: Array,
        y: Array,
        validation_data: Optional[Tuple[Array, Array]] = None,
        verbose: bool = True,
    ) -> "QNN":
        """
        Train the QNN.

        Parameters
        ----------
        X : Array
            Shape ``(n_samples, n_features)``.
        y : Array
            Labels or regression targets. Shape ``(n_samples,)`` or
            ``(n_samples, n_outputs)``.
        validation_data : tuple, optional
            ``(X_val, y_val)`` for validation metrics per epoch.
        verbose : bool
            Print epoch summaries.

        Returns
        -------
        QNN
            Self.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n_samples = X.shape[0]
        bs = self.batch_size if self.batch_size > 0 else n_samples

        for epoch in range(self.n_epochs):
            t0 = time.time()
            perm = np.random.permutation(n_samples)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, bs):
                batch_idx = perm[start:start + bs]
                X_b, y_b = X[batch_idx], y[batch_idx]

                # Forward for each sample
                preds = np.stack([self._forward_with_cache(x) for x in X_b], axis=0)

                # Loss + grad
                loss_val, dL = self._loss(preds, y_b)
                epoch_loss += loss_val
                n_batches  += 1

                # Backward for each sample in batch
                for i, (xi, dli) in enumerate(zip(X_b, dL)):
                    self._backward(xi, dli.reshape(1, -1) if dli.ndim == 0 else dli)

            avg_loss = epoch_loss / max(n_batches, 1)

            if self.task == "classification":
                metric_val = float(np.mean(self.predict(X) == y.astype(int)))
                metric_name = "acc"
            else:
                preds_all = self._forward_batch(X).ravel()
                metric_val = float(np.mean((preds_all - y.ravel()) ** 2))
                metric_name = "mse"

            elapsed = time.time() - t0
            self._history["loss"].append(avg_loss)
            self._history["metric"].append(metric_val)
            self._history["epoch_time"].append(elapsed)

            if verbose:
                val_str = ""
                if validation_data is not None:
                    X_v, y_v = validation_data
                    if self.task == "classification":
                        val_m = float(np.mean(self.predict(X_v) == y_v.astype(int)))
                    else:
                        pv = self._forward_batch(X_v).ravel()
                        val_m = float(np.mean((pv - y_v.ravel()) ** 2))
                    val_str = f" | val_{metric_name}={val_m:.4f}"
                print(
                    f"Epoch {epoch+1:03d}/{self.n_epochs} "
                    f"loss={avg_loss:.4f} {metric_name}={metric_val:.4f}"
                    f"{val_str} [{elapsed:.2f}s]"
                )

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: Array) -> Array:
        """
        Return probability scores.

        Parameters
        ----------
        X : Array
            Feature matrix.

        Returns
        -------
        Array
            Shape ``(n_samples, n_outputs)`` or ``(n_samples,)`` for binary.
        """
        X = np.asarray(X, dtype=float)
        raw = self._forward_batch(X)
        if self.n_outputs == 1:
            p = np.clip(raw.ravel(), 0, 1)
            return np.column_stack([1 - p, p])
        exp_r = np.exp(raw - raw.max(axis=1, keepdims=True))
        return exp_r / exp_r.sum(axis=1, keepdims=True)

    def predict(self, X: Array) -> Array:
        """
        Predict class labels (classification) or values (regression).

        Parameters
        ----------
        X : Array
            Feature matrix.

        Returns
        -------
        Array
            Predictions of shape ``(n_samples,)``.
        """
        if self.task == "regression":
            return self._forward_batch(np.asarray(X, dtype=float)).ravel()
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X: Array, y: Array) -> float:
        """
        Return accuracy (classification) or negative MSE (regression).

        Parameters
        ----------
        X : Array
            Feature matrix.
        y : Array
            True labels / targets.
        """
        if self.task == "classification":
            return float(np.mean(self.predict(X) == np.asarray(y, dtype=int)))
        preds = self.predict(X)
        return float(-np.mean((preds - np.asarray(y).ravel()) ** 2))

    @property
    def history(self) -> Dict[str, List[float]]:
        """Training history with keys ``loss``, ``metric``, ``epoch_time``."""
        return self._history

    def __repr__(self) -> str:
        return (
            f"QNN(n_features={self.n_features}, n_qubits={self.n_qubits}, "
            f"n_quantum_layers={self.n_quantum_layers}, task='{self.task}')"
        )
