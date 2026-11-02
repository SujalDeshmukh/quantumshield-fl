"""
Federated Learning Aggregators for QuantumShield-FL.

Provides three aggregation algorithms:

    ``FedAvg``
        Standard federated averaging (weighted by local dataset size).
    ``FedProx``
        FedAvg with a proximal term to limit drift.
    ``QuantumSecureAggregator``
        Aggregation with QKD decryption, differential privacy noise
        filtering, and quantum-inspired Byzantine fault tolerance.

All aggregators accept a list of update dicts::

    {
        'client_id' : str,
        'delta'     : np.ndarray,   # parameter delta (or absolute params)
        'n_samples' : int,
    }

and return a new global parameter array.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

Array = np.ndarray


# ---------------------------------------------------------------------------
# FedAvg
# ---------------------------------------------------------------------------

class FedAvg:
    """
    Federated Averaging aggregator.

    Computes a weighted mean of client parameter deltas, where each client's
    weight is proportional to its local dataset size.

    Parameters
    ----------
    server_lr : float
        Server-side learning rate applied to the aggregated delta.
        Set to 1.0 for vanilla FedAvg.
    """

    def __init__(self, server_lr: float = 1.0) -> None:
        self.server_lr = server_lr

    def aggregate(
        self,
        updates: List[Dict[str, Any]],
        global_params: Array,
    ) -> Array:
        """
        Aggregate client updates into new global parameters.

        Parameters
        ----------
        updates : list of dict
            Each dict must have ``'delta'`` (Array) and ``'n_samples'`` (int).
        global_params : Array
            Current global model parameters (used as base for delta updates).

        Returns
        -------
        Array
            Updated global parameter array.
        """
        if not updates:
            return global_params

        total_samples = sum(u["n_samples"] for u in updates)
        agg_delta = np.zeros_like(global_params, dtype=float)

        for u in updates:
            w = u["n_samples"] / total_samples
            agg_delta += w * np.asarray(u["delta"], dtype=float)

        return global_params + self.server_lr * agg_delta

    def __repr__(self) -> str:
        return f"FedAvg(server_lr={self.server_lr})"


# ---------------------------------------------------------------------------
# FedProx
# ---------------------------------------------------------------------------

class FedProx:
    """
    FedProx aggregator.

    Identical to FedAvg on the server side, but the proximal term is
    enforced on the *client* side during local training.  This aggregator
    stores the global parameters so they can be queried by clients to
    apply the proximal penalty::

        L_client = L_local + (mu / 2) * ‖w - w_global‖²

    Parameters
    ----------
    mu : float
        Proximal regularisation coefficient.
    server_lr : float
        Server-side learning rate.
    """

    def __init__(self, mu: float = 0.1, server_lr: float = 1.0) -> None:
        self.mu        = mu
        self.server_lr = server_lr

    def aggregate(
        self,
        updates: List[Dict[str, Any]],
        global_params: Array,
    ) -> Array:
        """
        Aggregate updates with weighted averaging (same as FedAvg on server).

        The proximal correction is optionally applied by subtracting mu * delta
        to discourage large deviations from the global model.

        Parameters
        ----------
        updates : list of dict
        global_params : Array

        Returns
        -------
        Array
        """
        if not updates:
            return global_params

        total_samples = sum(u["n_samples"] for u in updates)
        agg_delta = np.zeros_like(global_params, dtype=float)

        for u in updates:
            w = u["n_samples"] / total_samples
            delta = np.asarray(u["delta"], dtype=float)
            # Apply proximal correction: reduce large deltas
            corrected_delta = delta - self.mu * delta
            agg_delta += w * corrected_delta

        return global_params + self.server_lr * agg_delta

    def __repr__(self) -> str:
        return f"FedProx(mu={self.mu}, server_lr={self.server_lr})"


# ---------------------------------------------------------------------------
# Quantum-Inspired Byzantine Fault Tolerance
# ---------------------------------------------------------------------------

class _QuantumByzantineTolerance:
    """
    Quantum-inspired Byzantine fault tolerance filter.

    Uses a "quantum measurement" analogy: each client update is treated as
    a quantum state; outliers (Byzantine updates) are detected by computing
    the fidelity of each update with the median update and discarding those
    below a threshold.

    Parameters
    ----------
    byz_threshold : float
        Minimum cosine similarity (fidelity proxy) with median update.
        Updates below this are flagged as Byzantine and excluded.
    max_byz_fraction : float
        Maximum fraction of clients allowed to be flagged as Byzantine.
    """

    def __init__(
        self,
        byz_threshold: float = 0.5,
        max_byz_fraction: float = 0.4,
    ) -> None:
        self.byz_threshold     = byz_threshold
        self.max_byz_fraction  = max_byz_fraction

    def filter(
        self,
        updates: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Filter out likely Byzantine updates.

        Parameters
        ----------
        updates : list of dict
            Client updates with ``'delta'`` arrays.

        Returns
        -------
        (filtered_updates, flagged_ids)
            ``filtered_updates`` : updates that passed the Byzantine filter.
            ``flagged_ids``      : client IDs flagged as Byzantine.
        """
        if len(updates) < 3:
            return updates, []

        deltas = np.stack([np.asarray(u["delta"], dtype=float).ravel() for u in updates])
        median = np.median(deltas, axis=0)
        median_norm = np.linalg.norm(median) + 1e-12

        # Cosine similarity to median as a "fidelity" proxy
        similarities = []
        for d in deltas:
            d_norm = np.linalg.norm(d) + 1e-12
            sim = float(np.dot(d, median) / (d_norm * median_norm))
            similarities.append(sim)

        similarities = np.array(similarities)

        # Flag as Byzantine if similarity < threshold
        byz_mask = similarities < self.byz_threshold
        max_byz  = int(self.max_byz_fraction * len(updates))

        # If too many flagged, only remove the worst ones
        if byz_mask.sum() > max_byz:
            sorted_idx   = np.argsort(similarities)
            byz_mask     = np.zeros(len(updates), dtype=bool)
            byz_mask[sorted_idx[:max_byz]] = True

        flagged_ids     = [updates[i]["client_id"] for i in range(len(updates)) if byz_mask[i]]
        filtered        = [u for i, u in enumerate(updates) if not byz_mask[i]]
        return filtered, flagged_ids


# ---------------------------------------------------------------------------
# QuantumSecureAggregator
# ---------------------------------------------------------------------------

class QuantumSecureAggregator:
    """
    Quantum-secure aggregator combining:

    1. QKD payload decryption (simulated).
    2. Differential privacy noise filtering (Gaussian smoothing of updates).
    3. Quantum-inspired Byzantine fault tolerance.
    4. Weighted parameter averaging.

    Parameters
    ----------
    dp_noise_scale : float
        Standard deviation of the Gaussian noise added during aggregation for
        central DP (applied *after* Byzantine filtering).  Set to 0 to disable.
    byz_threshold : float
        Cosine similarity threshold for Byzantine detection.
    max_byz_fraction : float
        Maximum tolerated fraction of Byzantine clients.
    server_lr : float
        Server learning rate applied to the aggregated delta.
    qkd_sessions : dict, optional
        Mapping of ``client_id → _QKDSession``.  If provided, encrypted bytes
        in ``update['encrypted']`` will be decrypted before aggregation.
    """

    def __init__(
        self,
        dp_noise_scale: float = 0.01,
        byz_threshold: float = 0.5,
        max_byz_fraction: float = 0.3,
        server_lr: float = 1.0,
        qkd_sessions: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.dp_noise_scale  = dp_noise_scale
        self.server_lr       = server_lr
        self.qkd_sessions    = qkd_sessions or {}
        self._byz_filter     = _QuantumByzantineTolerance(
            byz_threshold=byz_threshold,
            max_byz_fraction=max_byz_fraction,
        )
        self._flagged_history: List[List[str]] = []

    # ------------------------------------------------------------------
    # QKD decryption
    # ------------------------------------------------------------------

    def _decrypt_if_needed(self, update: Dict[str, Any]) -> Dict[str, Any]:
        """
        If ``update`` contains an ``'encrypted'`` key, decrypt it.

        Parameters
        ----------
        update : dict
            Client update payload.

        Returns
        -------
        dict
            Update with ``'delta'`` populated.
        """
        if "encrypted" not in update:
            return update

        cid     = update["client_id"]
        session = self.qkd_sessions.get(cid)
        if session is None:
            raise ValueError(f"No QKD session found for client '{cid}'.")

        decrypted = session.decrypt(update["encrypted"])
        payload   = json.loads(decrypted.decode("utf-8"))
        return {**update, **payload}

    # ------------------------------------------------------------------
    # DP noise filtering
    # ------------------------------------------------------------------

    def _apply_central_dp(self, delta: Array) -> Array:
        """
        Add Gaussian noise for central differential privacy.

        Parameters
        ----------
        delta : Array
            Aggregated parameter delta.

        Returns
        -------
        Array
            Noised delta.
        """
        if self.dp_noise_scale <= 0:
            return delta
        noise = np.random.normal(0, self.dp_noise_scale, size=delta.shape)
        return delta + noise

    # ------------------------------------------------------------------
    # Main aggregate method
    # ------------------------------------------------------------------

    def aggregate(
        self,
        updates: List[Dict[str, Any]],
        global_params: Array,
    ) -> Array:
        """
        Secure aggregation pipeline.

        Steps:
            1. Decrypt QKD-encrypted updates (if present).
            2. Byzantine fault tolerance filter.
            3. Weighted FedAvg on surviving updates.
            4. Central DP Gaussian noise.
            5. Apply server learning rate.

        Parameters
        ----------
        updates : list of dict
            Client update dicts (may contain ``'encrypted'`` or plain ``'delta'``).
        global_params : Array
            Current global model parameters.

        Returns
        -------
        Array
            New global parameter array.
        """
        if not updates:
            return global_params

        # Step 1: Decrypt
        decrypted_updates = [self._decrypt_if_needed(u) for u in updates]

        # Step 2: Byzantine filter
        clean_updates, flagged = self._byz_filter.filter(decrypted_updates)
        self._flagged_history.append(flagged)
        if flagged:
            print(f"  [QuantumSecureAggregator] Flagged Byzantine clients: {flagged}")

        if not clean_updates:
            print("  [QuantumSecureAggregator] All updates flagged; returning unchanged params.")
            return global_params

        # Step 3: Weighted average
        total_samples = sum(u["n_samples"] for u in clean_updates)
        agg_delta = np.zeros_like(global_params, dtype=float)
        for u in clean_updates:
            w     = u["n_samples"] / total_samples
            delta = np.asarray(u["delta"], dtype=float)
            agg_delta += w * delta

        # Step 4: Central DP noise
        agg_delta = self._apply_central_dp(agg_delta)

        # Step 5: Apply server lr
        return global_params + self.server_lr * agg_delta

    @property
    def flagged_history(self) -> List[List[str]]:
        """History of Byzantine-flagged client IDs per round."""
        return self._flagged_history

    def __repr__(self) -> str:
        return (
            f"QuantumSecureAggregator("
            f"dp_noise={self.dp_noise_scale}, "
            f"server_lr={self.server_lr})"
        )
