"""
Federated Learning Server for QuantumShield-FL.

``FederatedServer`` orchestrates all federated rounds:

    1. Select a subset of clients (random / loss-based / quantum-random).
    2. Broadcast the current global model parameters.
    3. Wait for encrypted updates.
    4. Aggregate updates (via pluggable aggregator).
    5. Evaluate the updated global model.

Classes
-------
FederatedServer
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Type, Union

import numpy as np

Array = np.ndarray
SelectionStrategy = Literal["random", "loss_based", "quantum_random"]


class FederatedServer:
    """
    Federated learning coordinator.

    Parameters
    ----------
    global_model : Any
        Initialised quantum model (VQC or QNN) that acts as the global model.
        Must expose a ``.params`` attribute.
    aggregator : Any
        An aggregation object with an ``aggregate(updates)`` method.
        See ``aggregator.py`` for ``FedAvg``, ``FedProx``,
        ``QuantumSecureAggregator``.
    clients_per_round : int or float
        If int: exact number of clients selected each round.
        If float in (0, 1]: fraction of total clients.
    selection_strategy : SelectionStrategy
        How to select clients each round.
    min_clients : int
        Minimum number of clients needed to proceed.
    eval_data : tuple, optional
        ``(X_eval, y_eval)`` for global model evaluation each round.
    random_state : int, optional
        Seed for reproducibility.
    """

    def __init__(
        self,
        global_model: Any = None,
        aggregator: Any = None,
        clients: Optional[Sequence[Any]] = None,
        aggregation: Optional[str] = None,
        clients_per_round: Union[int, float] = 1.0,
        selection_strategy: SelectionStrategy = "random",
        min_clients: int = 1,
        eval_data: Optional[Tuple[Array, Array]] = None,
        random_state: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if clients is not None and len(clients) > 0 and global_model is None:
            import copy
            global_model = copy.deepcopy(clients[0].model)
        if aggregator is None:
            from .aggregator import FedAvg, FedProx
            strat = (aggregation or "fedavg").lower()
            if strat == "fedprox":
                aggregator = FedProx()
            else:
                aggregator = FedAvg()
        elif isinstance(aggregator, str):
            from .aggregator import FedAvg, FedProx
            strat = aggregator.lower()
            if strat == "fedprox":
                aggregator = FedProx()
            else:
                aggregator = FedAvg()

        self.global_model         = global_model
        self.aggregator           = aggregator
        self.clients_per_round    = clients_per_round
        self.selection_strategy   = selection_strategy
        self.min_clients          = min_clients
        self.eval_data            = eval_data
        self.random_state         = random_state

        self._rng      = np.random.default_rng(random_state)
        self._clients: List[Any] = []          # registered FederatedClient objects
        self._round: int = 0

        # Global tracking
        self._history: Dict[str, List[Any]] = {
            "round":            [],
            "global_accuracy":  [],
            "n_clients_used":   [],
            "round_time":       [],
            "client_accuracies": [],
        }

        # Last-known client losses for loss-based selection
        self._client_losses: Dict[str, float] = {}

        if clients is not None:
            self.register_clients(clients)

    @property
    def clients(self) -> List[Any]:
        """List of registered clients."""
        return self._clients

    @property
    def n_clients(self) -> int:
        """Total number of registered clients."""
        return len(self._clients)

    # ------------------------------------------------------------------
    # Client registry
    # ------------------------------------------------------------------

    def register_client(self, client: Any) -> None:
        """
        Register a federated client with the server.

        Parameters
        ----------
        client : FederatedClient
            Client to add.
        """
        self._clients.append(client)
        self._client_losses[client.client_id] = float("inf")

    def register_clients(self, clients: Sequence[Any]) -> None:
        """
        Register multiple clients at once.

        Parameters
        ----------
        clients : sequence of FederatedClient
        """
        for c in clients:
            self.register_client(c)

    @property
    def n_clients(self) -> int:
        """Total number of registered clients."""
        return len(self._clients)

    # ------------------------------------------------------------------
    # Client selection
    # ------------------------------------------------------------------

    def _n_to_select(self) -> int:
        """Resolve ``clients_per_round`` to an integer count."""
        cpr = self.clients_per_round
        if isinstance(cpr, float) and 0 < cpr <= 1.0:
            n = max(int(round(cpr * self.n_clients)), self.min_clients)
        else:
            n = int(cpr)
        return min(n, self.n_clients)

    def select_clients(self) -> List[Any]:
        """
        Select clients for the current round according to ``selection_strategy``.

        Strategies
        ----------
        ``'random'``
            Uniform random subset of clients.
        ``'loss_based'``
            Prefer clients with the highest known loss (hard examples first).
        ``'quantum_random'``
            Uses quantum-inspired randomness (QRNG simulation via
            measurement of a uniform superposition state).

        Returns
        -------
        list
            Selected ``FederatedClient`` objects.
        """
        k = self._n_to_select()

        if self.selection_strategy == "random":
            idx = self._rng.choice(self.n_clients, size=k, replace=False)

        elif self.selection_strategy == "loss_based":
            losses = np.array([
                self._client_losses.get(c.client_id, float("inf"))
                for c in self._clients
            ])
            # Higher loss → higher priority; add small random tiebreaker
            priority = losses + self._rng.uniform(0, 1e-6, size=losses.shape)
            idx = np.argsort(priority)[::-1][:k]

        elif self.selection_strategy == "quantum_random":
            # Simulate QRNG: measure a uniform superposition over n_clients qubits
            # Probability of selecting each client is uniform, but the
            # selection indices are derived from quantum measurement outcomes.
            dim = self.n_clients
            # Equal superposition
            amplitudes = np.ones(dim, dtype=complex) / np.sqrt(dim)
            probs = np.abs(amplitudes) ** 2
            # Re-normalise (should already sum to 1)
            probs /= probs.sum()
            idx = self._rng.choice(dim, size=k, replace=False, p=probs)

        else:
            raise ValueError(f"Unknown selection strategy: {self.selection_strategy}")

        return [self._clients[i] for i in idx]

    # ------------------------------------------------------------------
    # Broadcast global model
    # ------------------------------------------------------------------

    def broadcast_model(self, clients: List[Any]) -> None:
        """
        Push the current global model parameters to a list of clients.

        Parameters
        ----------
        clients : list of FederatedClient
        """
        for client in clients:
            client.apply_global_update(self.global_model.params.copy())

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    def aggregate_updates(self, updates: List[Dict[str, Any]]) -> Array:
        """
        Pass collected updates to the aggregator and return the new global params.

        Parameters
        ----------
        updates : list of dict
            Each dict has keys ``'delta'``, ``'n_samples'``, ``'client_id'``.

        Returns
        -------
        Array
            New global parameter array.
        """
        return self.aggregator.aggregate(updates, self.global_model.params.copy())

    # ------------------------------------------------------------------
    # Global evaluation
    # ------------------------------------------------------------------

    def evaluate_global(self, X: Optional[Array] = None, y: Optional[Array] = None) -> float:
        """
        Evaluate the global model.

        Parameters
        ----------
        X : Array, optional
            Test features. Falls back to ``self.eval_data[0]`` if not provided.
        y : Array, optional
            Test labels. Falls back to ``self.eval_data[1]`` if not provided.

        Returns
        -------
        float
            Global model accuracy (or NaN if no eval data is available).
        """
        if X is None or y is None:
            if self.eval_data is None:
                return float("nan")
            X, y = self.eval_data
        return self.global_model.score(np.asarray(X, dtype=float), np.asarray(y))

    # ------------------------------------------------------------------
    # Single round
    # ------------------------------------------------------------------

    def run_round(self, round_num: Optional[int] = None, verbose: bool = False) -> Dict[str, Any]:
        """
        Execute one full federated round.

        Steps:
            1. Select clients.
            2. Broadcast global model.
            3. Each client trains locally and returns an update.
            4. Aggregate updates → new global params.
            5. Evaluate global model.

        Parameters
        ----------
        round_num : int, optional
            Optional round number.
        verbose : bool
            Print round summary.

        Returns
        -------
        dict
            Round summary with keys ``round``, ``global_accuracy``,
            ``n_clients``, ``round_time``, ``client_accuracies``,
            ``accuracy``, and ``loss``.
        """
        if round_num is not None:
            self._round = round_num
        else:
            self._round += 1
        t0 = time.time()

        if self.n_clients < self.min_clients:
            raise RuntimeError(
                f"Not enough clients ({self.n_clients} < {self.min_clients})."
            )

        # 1. Select
        selected = self.select_clients()

        # 2. Broadcast
        self.broadcast_model(selected)

        # 3. Local training + collect updates
        updates: List[Dict[str, Any]] = []
        client_accuracies: Dict[str, float] = {}

        for client in selected:
            metrics = client.train_local(verbose=False)
            update  = client.get_raw_update()
            updates.append(update)
            client_accuracies[client.client_id] = metrics.get("accuracy", 0.0)
            # Track loss for loss-based selection
            self._client_losses[client.client_id] = metrics.get("loss", 0.0)

        # 4. Aggregate
        new_params = self.aggregate_updates(updates)
        self.global_model.params = new_params

        # 5. Evaluate
        global_acc = self.evaluate_global()
        elapsed = time.time() - t0

        # Record history
        self._history["round"].append(self._round)
        self._history["global_accuracy"].append(global_acc)
        self._history["n_clients_used"].append(len(selected))
        self._history["round_time"].append(elapsed)
        self._history["client_accuracies"].append(client_accuracies)

        avg_loss = float(np.mean(list(self._client_losses.values()))) if self._client_losses else 0.0
        summary = {
            "round":             self._round,
            "global_accuracy":   global_acc,
            "accuracy":          global_acc,
            "loss":              avg_loss,
            "n_clients":         len(selected),
            "round_time":        elapsed,
            "client_accuracies": client_accuracies,
        }

        if verbose:
            print(
                f"[Round {self._round:03d}] "
                f"global_acc={global_acc:.4f}  "
                f"clients={len(selected)}  "
                f"time={elapsed:.2f}s  "
                f"client_accs={client_accuracies}"
            )

        return summary

    # ------------------------------------------------------------------
    # Multi-round training loop
    # ------------------------------------------------------------------

    def run(
        self,
        n_rounds: int,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Run the federated learning process for ``n_rounds`` communication rounds.

        Parameters
        ----------
        n_rounds : int
            Number of federated rounds to execute.
        verbose : bool
            Print per-round summaries.

        Returns
        -------
        list of dict
            One summary dict per round (same structure as ``run_round()``).
        """
        histories = []
        for r in range(n_rounds):
            summary = self.run_round(verbose=verbose)
            histories.append(summary)
        return histories

    # ------------------------------------------------------------------
    # Properties / utility
    # ------------------------------------------------------------------

    @property
    def history(self) -> Dict[str, List[Any]]:
        """Full training history across rounds."""
        return self._history

    @property
    def current_round(self) -> int:
        """Current federated round number."""
        return self._round

    def get_global_params(self) -> Array:
        """Return the current global model parameter array."""
        return self.global_model.params.copy()

    def __repr__(self) -> str:
        return (
            f"FederatedServer("
            f"n_clients={self.n_clients}, "
            f"strategy='{self.selection_strategy}', "
            f"round={self._round})"
        )
