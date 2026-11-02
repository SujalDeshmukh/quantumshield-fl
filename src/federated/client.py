"""
Federated Learning Client for QuantumShield-FL.

Each ``FederatedClient`` holds a local quantum model (VQC or QNN), trains on
its private dataset, and communicates encrypted model updates to the server.
Local differential privacy (Gaussian mechanism) is applied before sending.

Classes
-------
FederatedClient
    Core federated participant.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

# Lazy imports to avoid circular dependency; these are resolved at runtime.
Array = np.ndarray
ModelType = Literal["vqc", "qnn"]


# ---------------------------------------------------------------------------
# Lightweight XOR-based pseudo-QKD session
# ---------------------------------------------------------------------------

class _QKDSession:
    """
    Simulated QKD (Quantum Key Distribution) session.

    In a real deployment this would interface with BB84/E91 hardware.
    Here we simulate a pre-shared symmetric key generated from a
    quantum-random seed (modelled by ``os.urandom``).

    Parameters
    ----------
    session_id : str
        Unique identifier for this QKD session.
    key_length : int
        Number of random bytes in the shared key.
    """

    def __init__(self, session_id: str, key_length: int = 256) -> None:
        self.session_id = session_id
        # Simulate quantum randomness with os.urandom
        self._key: bytes = os.urandom(key_length)
        self._key_index: int = 0

    def _get_key_bytes(self, n: int) -> bytes:
        """Stream key bytes, cycling if exhausted."""
        result = bytearray()
        while len(result) < n:
            remaining = n - len(result)
            avail = self._key[self._key_index: self._key_index + remaining]
            result.extend(avail)
            self._key_index = (self._key_index + len(avail)) % len(self._key)
        return bytes(result)

    def encrypt(self, data: bytes) -> bytes:
        """
        XOR-encrypt bytes with the QKD key stream.

        Parameters
        ----------
        data : bytes
            Plaintext bytes.

        Returns
        -------
        bytes
            Ciphertext bytes.
        """
        key_stream = self._get_key_bytes(len(data))
        return bytes(a ^ b for a, b in zip(data, key_stream))

    def decrypt(self, data: bytes) -> bytes:
        """
        XOR-decrypt bytes with the QKD key stream.

        Parameters
        ----------
        data : bytes
            Ciphertext bytes.

        Returns
        -------
        bytes
            Plaintext bytes.
        """
        return self.encrypt(data)   # XOR is symmetric

    def derive_shared_secret(self) -> str:
        """Return a hex digest of the key for handshake verification."""
        return hashlib.sha256(self._key).hexdigest()


# ---------------------------------------------------------------------------
# FederatedClient
# ---------------------------------------------------------------------------

class FederatedClient:
    """
    A federated learning participant with quantum-secured model updates.

    The client:
        1. Holds a private local dataset ``(X_local, y_local)``.
        2. Trains the local model for ``local_epochs`` each round.
        3. Clips and noises gradients / parameters with a Gaussian mechanism
           (local differential privacy).
        4. Serialises parameters, encrypts them via a QKD session, and
           returns the encrypted payload to the server.

    Parameters
    ----------
    client_id : str
        Unique identifier for this client.
    model : Any
        A VQC or QNN instance (must expose ``.params``, ``.fit()``,
        ``.predict()``, ``.score()`` attributes).
    X_local : Array
        Local training features.
    y_local : Array
        Local training labels.
    local_epochs : int
        Number of local training epochs per federated round.
    local_batch_size : int
        Batch size for local training.
    dp_epsilon : float
        Local DP noise scale (Gaussian mechanism). 0 → no noise.
    dp_delta : float
        Delta parameter for (ε, δ)-DP.
    clip_norm : float
        L2 norm clipping threshold for parameter updates before DP noise.
    qkd_key_length : int
        Simulated QKD key length in bytes.
    """

    def __init__(
        self,
        client_id: str,
        model: Any,
        X_local: Optional[Array] = None,
        y_local: Optional[Array] = None,
        X_train: Optional[Array] = None,
        y_train: Optional[Array] = None,
        local_epochs: int = 5,
        local_batch_size: int = 16,
        dp_epsilon: float = 1.0,
        dp_delta: float = 1e-5,
        clip_norm: float = 1.0,
        qkd_key_length: int = 256,
        use_qkd: bool = True,
        use_dp: bool = True,
        **kwargs: Any,
    ) -> None:
        if X_local is None and X_train is not None:
            X_local = X_train
        if y_local is None and y_train is not None:
            y_local = y_train
        if not use_dp:
            dp_epsilon = 0.0

        self.client_id       = client_id
        self.model           = model
        self.X_local         = np.asarray(X_local, dtype=float) if X_local is not None else np.zeros((0, 2))
        self.y_local         = np.asarray(y_local) if y_local is not None else np.zeros(0)
        self.local_epochs    = local_epochs
        self.local_batch_size = local_batch_size
        self.dp_epsilon      = dp_epsilon
        self.dp_delta        = dp_delta
        self.clip_norm       = clip_norm
        self.use_qkd         = use_qkd
        self.use_dp          = use_dp

        self._qkd = _QKDSession(session_id=client_id, key_length=qkd_key_length)

        # Metrics
        self._metrics: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc":  [],
            "round_time": [],
        }
        self._selected_rounds: int = 0
        self._initial_params: Optional[Array] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_samples(self) -> int:
        """Number of local training samples."""
        return len(self.X_local)

    @property
    def metrics(self) -> Dict[str, List[float]]:
        """Local training metrics per federated round."""
        return self._metrics

    @property
    def qkd_session_id(self) -> str:
        """QKD session identifier."""
        return self._qkd.session_id

    # ------------------------------------------------------------------
    # Local training
    # ------------------------------------------------------------------

    def train_local(self, verbose: bool = False) -> Dict[str, float]:
        """
        Execute local training for ``local_epochs`` epochs.

        The model is updated in-place. Before training, ``_initial_params``
        is snapshotted to compute the parameter delta for transmission.

        Parameters
        ----------
        verbose : bool
            Print epoch-level training output.

        Returns
        -------
        dict
            ``{'loss': float, 'accuracy': float, 'duration': float}``
        """
        self._selected_rounds += 1
        t0 = time.time()

        # Snapshot parameters before training (for computing delta)
        self._initial_params = copy.deepcopy(self.model.params)

        # Temporarily set epochs + batch size for local training
        orig_epochs = self.model.n_epochs
        orig_bs     = self.model.batch_size
        self.model.n_epochs    = self.local_epochs
        self.model.batch_size  = self.local_batch_size

        self.model.fit(self.X_local, self.y_local, verbose=verbose)

        self.model.n_epochs   = orig_epochs
        self.model.batch_size = orig_bs

        # Metrics
        train_acc  = self.model.score(self.X_local, self.y_local)
        train_loss = (
            self.model.history["loss"][-1]
            if self.model.history.get("loss")
            else float("nan")
        )
        duration = time.time() - t0

        self._metrics["train_loss"].append(train_loss)
        self._metrics["train_acc"].append(train_acc)
        self._metrics["round_time"].append(duration)

        return {"loss": train_loss, "accuracy": train_acc, "duration": duration}

    # ------------------------------------------------------------------
    # DP noise
    # ------------------------------------------------------------------

    def _clip_and_noise(self, delta_params: Array) -> Array:
        """
        Apply gradient clipping and Gaussian DP noise to parameter delta.

        The Gaussian mechanism adds noise calibrated to:
            σ = clip_norm * sqrt(2 * ln(1.25/δ)) / ε

        Parameters
        ----------
        delta_params : Array
            Raw parameter update (current - initial).

        Returns
        -------
        Array
            Clipped and noised parameter delta.
        """
        if self.dp_epsilon <= 0:
            return delta_params

        # L2 clip
        norm = np.linalg.norm(delta_params)
        if norm > self.clip_norm:
            delta_params = delta_params * (self.clip_norm / (norm + 1e-12))

        # Gaussian noise
        sigma = (
            self.clip_norm
            * np.sqrt(2 * np.log(1.25 / (self.dp_delta + 1e-30)))
            / self.dp_epsilon
        )
        noise = np.random.normal(0, sigma, size=delta_params.shape)
        return delta_params + noise

    # ------------------------------------------------------------------
    # Serialise & encrypt
    # ------------------------------------------------------------------

    def get_encrypted_update(self) -> Tuple[bytes, int]:
        """
        Compute, clip, noise, serialise, and QKD-encrypt the parameter delta.

        Returns
        -------
        (encrypted_bytes, n_samples)
            ``encrypted_bytes`` : QKD-encrypted JSON payload (bytes).
            ``n_samples``       : Number of local training samples (for
                                  weighted aggregation).
        """
        if self._initial_params is None:
            raise RuntimeError("Call train_local() before get_encrypted_update().")

        # Compute delta
        delta = self.model.params - self._initial_params
        delta_noised = self._clip_and_noise(delta.ravel()).reshape(delta.shape)

        payload = json.dumps({
            "client_id":  self.client_id,
            "delta":      delta_noised.tolist(),
            "n_samples":  self.n_samples,
            "round":      self._selected_rounds,
        }).encode("utf-8")

        encrypted = self._qkd.encrypt(payload)
        return encrypted, self.n_samples

    def get_raw_update(self) -> Dict[str, Any]:
        """
        Return the noised parameter delta as a plain Python dict (no encryption).

        Useful for testing aggregators without the full QKD pipeline.

        Returns
        -------
        dict
            Keys: ``client_id``, ``delta``, ``n_samples``, ``round``.
        """
        if self._initial_params is None:
            raise RuntimeError("Call train_local() before get_raw_update().")
        delta = self.model.params - self._initial_params
        delta_noised = self._clip_and_noise(delta.ravel()).reshape(delta.shape)
        return {
            "client_id": self.client_id,
            "delta":     delta_noised,
            "n_samples": self.n_samples,
            "round":     self._selected_rounds,
        }

    # ------------------------------------------------------------------
    # Apply global update
    # ------------------------------------------------------------------

    def apply_global_update(self, global_params: Array) -> None:
        """
        Replace local model parameters with the global (aggregated) parameters.

        Parameters
        ----------
        global_params : Array
            Parameter array broadcast from the server.
        """
        self.model.params = global_params.copy()
        # Reset initial snapshot
        self._initial_params = None

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        X_test: Optional[Array] = None,
        y_test: Optional[Array] = None,
    ) -> Dict[str, float]:
        """
        Evaluate the current local model.

        Parameters
        ----------
        X_test : Array, optional
            Test features. Defaults to local training data.
        y_test : Array, optional
            Test labels. Defaults to local training labels.

        Returns
        -------
        dict
            ``{'accuracy': float, 'n_samples': int}``
        """
        X = self.X_local if X_test is None else np.asarray(X_test, dtype=float)
        y = self.y_local if y_test is None else np.asarray(y_test)
        acc = self.model.score(X, y)
        return {"accuracy": acc, "n_samples": len(X)}

    # ------------------------------------------------------------------
    # QKD helpers (server-side mirror)
    # ------------------------------------------------------------------

    def decrypt_message(self, ciphertext: bytes) -> bytes:
        """
        Decrypt a message from the server using the shared QKD key.

        Parameters
        ----------
        ciphertext : bytes
            Encrypted bytes from the server.

        Returns
        -------
        bytes
            Plaintext bytes.
        """
        return self._qkd.decrypt(ciphertext)

    def shared_secret(self) -> str:
        """Return the QKD shared secret digest for handshake verification."""
        return self._qkd.derive_shared_secret()

    def __repr__(self) -> str:
        return (
            f"FederatedClient(id='{self.client_id}', "
            f"n_samples={self.n_samples}, rounds={self._selected_rounds})"
        )
