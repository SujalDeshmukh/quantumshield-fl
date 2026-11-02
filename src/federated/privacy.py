"""
Privacy Mechanisms for QuantumShield-FL.

Provides three privacy components:

    ``DifferentialPrivacyMechanism``
        Gaussian and Laplace mechanisms for ε-DP and (ε, δ)-DP.
        Tracks consumed privacy budget.

    ``QuantumNoiseChannel``
        Simulates quantum depolarizing channel noise on parameter vectors.
        Used as a quantum-inspired privacy amplification mechanism.

    ``PrivacyAccountant``
        Tracks cumulative privacy loss across multiple FL rounds using
        Rényi Differential Privacy (RDP) composition.

References
----------
- Dwork & Roth (2014) – "The Algorithmic Foundations of Differential Privacy"
- Mironov (2017)      – "Rényi Differential Privacy of the Gaussian Mechanism"
- Abadi et al. (2016) – "Deep Learning with Differential Privacy"
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.special import erfc, gammaincc

Array = np.ndarray


# ---------------------------------------------------------------------------
# Helper: Gaussian CDF tail
# ---------------------------------------------------------------------------

def _gaussian_tail(z: float) -> float:
    """P(N(0,1) > z)."""
    return float(0.5 * erfc(z / math.sqrt(2)))


# ---------------------------------------------------------------------------
# DifferentialPrivacyMechanism
# ---------------------------------------------------------------------------

class DifferentialPrivacyMechanism:
    """
    Classical differential privacy mechanisms (Gaussian and Laplace).

    Supports:
        - Global sensitivity computation for L1 and L2 norms.
        - Gaussian mechanism: achieves (ε, δ)-DP.
        - Laplace mechanism:  achieves ε-DP.
        - Privacy budget tracking (epsilon consumed).

    Parameters
    ----------
    epsilon : float
        Privacy budget ε > 0.
    delta : float
        Failure probability δ ∈ (0, 1) used for Gaussian mechanism.
    sensitivity : float
        Global sensitivity of the query function (Δf).
        Can be overridden per call.

    Examples
    --------
    >>> dp = DifferentialPrivacyMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0)
    >>> noised = dp.gaussian(my_array)
    >>> dp.budget_used
    1.0
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        sensitivity: float = 1.0,
    ) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0.")
        if not (0 <= delta < 1):
            raise ValueError("delta must be in [0, 1).")
        if sensitivity <= 0:
            raise ValueError("sensitivity must be > 0.")

        self.epsilon     = epsilon
        self.delta       = delta
        self.sensitivity = sensitivity

        self._budget_used: float = 0.0
        self._query_log: List[Dict[str, float]] = []

    @property
    def remaining_budget(self) -> float:
        """Remaining privacy budget epsilon."""
        return self.epsilon - self._budget_used

    # ------------------------------------------------------------------
    # Sensitivity helpers
    # ------------------------------------------------------------------

    @staticmethod
    def l2_sensitivity(f_x: Array, f_x_prime: Array) -> float:
        """
        Compute L2 sensitivity ‖f(x) - f(x')‖₂ for two neighbouring datasets.

        Parameters
        ----------
        f_x, f_x_prime : Array
            Query outputs on two neighbouring datasets.

        Returns
        -------
        float
        """
        return float(np.linalg.norm(np.asarray(f_x) - np.asarray(f_x_prime)))

    @staticmethod
    def l1_sensitivity(f_x: Array, f_x_prime: Array) -> float:
        """
        Compute L1 sensitivity ‖f(x) - f(x')‖₁.

        Parameters
        ----------
        f_x, f_x_prime : Array
        Returns
        -------
        float
        """
        return float(np.sum(np.abs(np.asarray(f_x) - np.asarray(f_x_prime))))

    # ------------------------------------------------------------------
    # Gaussian mechanism
    # ------------------------------------------------------------------

    def gaussian_sigma(self, epsilon: Optional[float] = None, delta: Optional[float] = None) -> float:
        """
        Compute the Gaussian noise standard deviation for (ε, δ)-DP.

        Uses the analytic formula:
            σ = Δf * sqrt(2 * ln(1.25 / δ)) / ε

        Parameters
        ----------
        epsilon : float, optional
            Override instance epsilon.
        delta : float, optional
            Override instance delta.

        Returns
        -------
        float
            Required noise standard deviation.
        """
        eps = epsilon or self.epsilon
        dlt = delta if delta is not None and delta > 0 else (self.delta if self.delta > 0 else 1e-5)
        return self.sensitivity * math.sqrt(2 * math.log(1.25 / dlt)) / eps

    def gaussian(
        self,
        query_result: Array,
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
        sensitivity: Optional[float] = None,
    ) -> Array:
        """
        Apply the Gaussian mechanism to a query result.

        Adds i.i.d. Gaussian noise N(0, σ²) where σ is calibrated to
        achieve (ε, δ)-DP.

        Parameters
        ----------
        query_result : Array
            Real-valued query output to protect.
        epsilon : float, optional
            Override ε for this call.
        delta : float, optional
            Override δ for this call.
        sensitivity : float, optional
            Override Δf for this call.

        Returns
        -------
        Array
            Privatised query result.
        """
        eps  = epsilon or self.epsilon
        dlt  = delta if delta is not None and delta > 0 else (self.delta if self.delta > 0 else 1e-5)
        sens = sensitivity or self.sensitivity

        sigma = sens * math.sqrt(2 * math.log(1.25 / dlt)) / eps
        noise = np.random.normal(0.0, sigma, size=np.asarray(query_result).shape)

        self._budget_used += eps
        self._query_log.append({"mechanism": "gaussian", "epsilon": eps, "sigma": sigma})
        return np.asarray(query_result, dtype=float) + noise

    # ------------------------------------------------------------------
    # Laplace mechanism
    # ------------------------------------------------------------------

    def laplace_scale(self, epsilon: Optional[float] = None) -> float:
        """
        Compute the Laplace noise scale b = Δf / ε for ε-DP.

        Parameters
        ----------
        epsilon : float, optional
        Returns
        -------
        float
        """
        eps = epsilon or self.epsilon
        return self.sensitivity / eps

    def laplace(
        self,
        query_result: Array,
        epsilon: Optional[float] = None,
        sensitivity: Optional[float] = None,
    ) -> Array:
        """
        Apply the Laplace mechanism to a query result.

        Adds i.i.d. Laplace noise Lap(0, b) where b = Δf / ε, achieving ε-DP.

        Parameters
        ----------
        query_result : Array
            Real-valued query output.
        epsilon : float, optional
            Override ε for this call.
        sensitivity : float, optional
            Override Δf for this call.

        Returns
        -------
        Array
            Privatised query result.
        """
        eps  = epsilon     or self.epsilon
        sens = sensitivity or self.sensitivity
        b    = sens / eps

        noise = np.random.laplace(0.0, b, size=np.asarray(query_result).shape)

        self._budget_used += eps
        self._query_log.append({"mechanism": "laplace", "epsilon": eps, "b": b})
        return np.asarray(query_result, dtype=float) + noise

    def gaussian_mechanism(self, query_result: Array, *args: Any, **kwargs: Any) -> Array:
        """Alias for gaussian()."""
        return self.gaussian(query_result, *args, **kwargs)

    def laplace_mechanism(self, query_result: Array, *args: Any, **kwargs: Any) -> Array:
        """Alias for laplace()."""
        return self.laplace(query_result, *args, **kwargs)

    # ------------------------------------------------------------------
    # Budget tracking
    # ------------------------------------------------------------------

    @property
    def budget_used(self) -> float:
        """Total privacy budget consumed so far (sequential composition)."""
        return self._budget_used

    @property
    def budget_remaining(self) -> float:
        """Remaining privacy budget under sequential composition."""
        return max(0.0, self.epsilon - self._budget_used)

    @property
    def query_log(self) -> List[Dict[str, float]]:
        """Log of every privatisation call."""
        return self._query_log

    def reset_budget(self) -> None:
        """Reset the consumed privacy budget counter and query log."""
        self._budget_used = 0.0
        self._query_log.clear()

    def __repr__(self) -> str:
        return (
            f"DifferentialPrivacyMechanism("
            f"ε={self.epsilon}, δ={self.delta}, Δf={self.sensitivity})"
        )


# ---------------------------------------------------------------------------
# QuantumNoiseChannel
# ---------------------------------------------------------------------------

class QuantumNoiseChannel:
    """
    Quantum depolarizing channel noise simulator.

    The depolarizing channel maps a state ρ to:

        ε(ρ) = (1 - p) * ρ  +  p * I / d

    where d = 2^n_qubits is the Hilbert space dimension and p is the
    depolarizing probability.

    For a *parameter vector* (classical), this is applied qubit-by-qubit
    as a Pauli twirl: with probability p, a random Pauli error (X, Y, Z)
    is applied to the associated parameter value.

    Parameters
    ----------
    n_qubits : int
        Number of qubits the model represents.
    depolarizing_prob : float
        Depolarizing probability p ∈ [0, 1].  p = 0 → identity channel
        (no noise).  p = 3/4 → fully depolarized (maximally mixed).
    """

    def __init__(self, n_qubits: int, depolarizing_prob: float = 0.05) -> None:
        if not (0.0 <= depolarizing_prob <= 1.0):
            raise ValueError("depolarizing_prob must be in [0, 1].")
        self.n_qubits          = n_qubits
        self.depolarizing_prob = depolarizing_prob
        self.dim               = 2 ** n_qubits

    # ------------------------------------------------------------------
    # Density matrix channel
    # ------------------------------------------------------------------

    def apply_to_density_matrix(self, rho: Array) -> Array:
        """
        Apply the depolarizing channel to a density matrix.

        ε(ρ) = (1 - p) * ρ  +  (p / d) * I

        Parameters
        ----------
        rho : Array
            Density matrix of shape ``(d, d)`` where ``d = 2**n_qubits``.

        Returns
        -------
        Array
            Noised density matrix.
        """
        p   = self.depolarizing_prob
        d   = self.dim
        rho = np.asarray(rho, dtype=complex)
        return (1 - p) * rho + (p / d) * np.eye(d, dtype=complex)

    # ------------------------------------------------------------------
    # Parameter-vector channel (privacy amplification)
    # ------------------------------------------------------------------

    def apply_to_params(self, params: Array) -> Array:
        """
        Simulate depolarizing noise on a classical parameter vector.

        Each parameter θ_i is independently flipped (negated, randomised, or
        zeroed) with probability ``depolarizing_prob`` to model qubit errors
        in a measurement-based or VQE context.

        Concretely:
            - With probability (1 - p): θ_i → θ_i   (identity)
            - With prob p/3:            θ_i → -θ_i  (Z-error analogue)
            - With prob p/3:            θ_i → θ_i + π (X-error analogue)
            - With prob p/3:            θ_i → random Uniform(-π, π) (Y-error)

        Parameters
        ----------
        params : Array
            Parameter array (any shape).

        Returns
        -------
        Array
            Noised parameter array.
        """
        params = np.asarray(params, dtype=float).copy()
        flat   = params.ravel()
        p      = self.depolarizing_prob

        for i in range(len(flat)):
            r = np.random.uniform()
            if r < p / 3:
                flat[i] = -flat[i]
            elif r < 2 * p / 3:
                flat[i] = flat[i] + math.pi
            elif r < p:
                flat[i] = np.random.uniform(-math.pi, math.pi)

        return flat.reshape(params.shape)

    # ------------------------------------------------------------------
    # Fidelity computation
    # ------------------------------------------------------------------

    def fidelity_after_channel(self, rho: Array) -> float:
        """
        Compute the fidelity of the channel output with the input state.

        For a pure input state |ψ⟩⟨ψ|, the fidelity with the depolarized
        output is:

            F = Tr[ρ * ε(ρ)] = 1 - p + p / d

        Parameters
        ----------
        rho : Array
            Input density matrix.

        Returns
        -------
        float
            Fidelity ∈ [0, 1].
        """
        output = self.apply_to_density_matrix(rho)
        rho_arr = np.asarray(rho, dtype=complex)
        return float(np.real(np.trace(rho_arr @ output)))

    def privacy_amplification_factor(self) -> float:
        """
        Estimate the effective DP privacy amplification from the channel.

        The depolarizing noise mixes the quantum state towards the maximally
        mixed state, providing privacy amplification.  The factor is
        heuristically modelled as:

            amplification = 1 / (1 + p * d)

        Returns
        -------
        float
        """
        return 1.0 / (1.0 + self.depolarizing_prob * self.dim)

    def __repr__(self) -> str:
        return (
            f"QuantumNoiseChannel("
            f"n_qubits={self.n_qubits}, "
            f"p={self.depolarizing_prob})"
        )


# ---------------------------------------------------------------------------
# PrivacyAccountant (RDP)
# ---------------------------------------------------------------------------

class PrivacyAccountant:
    """
    Rényi Differential Privacy (RDP) accountant.

    Tracks the cumulative privacy loss across multiple federated rounds using
    RDP composition (Mironov 2017) and converts to (ε, δ)-DP via the RDP-to-DP
    conversion.

    The Gaussian mechanism with noise multiplier σ achieves RDP at order α::

        RDP(α) = α / (2σ²)

    Composition (k rounds):
        RDP_total(α) = k * RDP(α)

    Conversion to (ε, δ)-DP:
        ε(δ) = min_α [ RDP(α) + log(1/δ) / (α - 1) ]

    Parameters
    ----------
    noise_multiplier : float
        Ratio σ / Δf (noise std / L2 sensitivity).
    sampling_probability : float
        Probability that any single record is included in a round's
        mini-batch (q = batch_size / dataset_size).
    delta : float
        Target δ for (ε, δ)-DP conversion.
    orders : list of float, optional
        Rényi orders α to evaluate. Defaults to [2 .. 64].

    Examples
    --------
    >>> acct = PrivacyAccountant(noise_multiplier=1.1, sampling_probability=0.01)
    >>> for round_idx in range(100):
    ...     acct.step()
    >>> print(acct.get_epsilon(delta=1e-5))
    """

    def __init__(
        self,
        noise_multiplier: float = 1.0,
        sampling_probability: float = 0.01,
        delta: float = 1e-5,
        orders: Optional[List[float]] = None,
    ) -> None:
        if noise_multiplier <= 0:
            raise ValueError("noise_multiplier must be > 0.")
        if not (0 < sampling_probability <= 1):
            raise ValueError("sampling_probability must be in (0, 1].")
        if not (0 < delta < 1):
            raise ValueError("delta must be in (0, 1).")

        self.noise_multiplier     = noise_multiplier
        self.sampling_probability = sampling_probability
        self.delta                = delta
        self.orders: List[float]  = orders or list(range(2, 65))

        self._steps: int = 0
        self._rdp_per_step: Dict[float, float] = {}
        self._compute_rdp_per_step()

    # ------------------------------------------------------------------
    # RDP per step
    # ------------------------------------------------------------------

    def _compute_rdp_per_step(self) -> None:
        """Pre-compute RDP(α) for a single step of the Gaussian mechanism."""
        sigma = self.noise_multiplier
        q     = self.sampling_probability
        for alpha in self.orders:
            if q == 1.0:
                rdp = alpha / (2 * sigma ** 2)
            else:
                # Poisson subsampled Gaussian: bound from Mironov (2017) §3
                # Simplified upper bound: q^2 * alpha / (2 * sigma^2)
                rdp = q ** 2 * alpha / (2 * sigma ** 2)
            self._rdp_per_step[alpha] = rdp

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, n_steps: int = 1) -> None:
        """
        Record that ``n_steps`` federated rounds have been completed.

        Parameters
        ----------
        n_steps : int
            Number of additional rounds to account for.
        """
        self._steps += n_steps

    # ------------------------------------------------------------------
    # Cumulative RDP
    # ------------------------------------------------------------------

    def get_rdp(self) -> Dict[float, float]:
        """
        Return cumulative RDP(α) for all tracked orders.

        Returns
        -------
        dict
            Mapping ``α → cumulative RDP(α)``.
        """
        return {alpha: self._steps * rdp for alpha, rdp in self._rdp_per_step.items()}

    # ------------------------------------------------------------------
    # Convert RDP → (ε, δ)-DP
    # ------------------------------------------------------------------

    def get_epsilon(self, delta: Optional[float] = None) -> float:
        """
        Convert cumulative RDP to (ε, δ)-DP via the optimal conversion.

        ε(α) = RDP_total(α) + log(1/δ) / (α - 1)

        The returned ε is the minimum over all Rényi orders.

        Parameters
        ----------
        delta : float, optional
            Target δ. Defaults to ``self.delta``.

        Returns
        -------
        float
            (ε, δ)-DP privacy guarantee ε.
        """
        if self._steps == 0:
            return 0.0

        dlt = delta or self.delta
        rdp_total = self.get_rdp()

        epsilons = []
        for alpha, rdp_val in rdp_total.items():
            if alpha <= 1:
                continue
            eps = rdp_val + math.log(1 / dlt) / (alpha - 1)
            epsilons.append(eps)

        return float(min(epsilons)) if epsilons else float("inf")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def privacy_report(self, delta: Optional[float] = None) -> Dict[str, float]:
        """
        Generate a privacy summary report.

        Parameters
        ----------
        delta : float, optional

        Returns
        -------
        dict
            Keys: ``steps``, ``epsilon``, ``delta``,
            ``noise_multiplier``, ``sampling_prob``.
        """
        dlt = delta or self.delta
        eps = self.get_epsilon(dlt)
        return {
            "steps":            self._steps,
            "epsilon":          eps,
            "delta":            dlt,
            "noise_multiplier": self.noise_multiplier,
            "sampling_prob":    self.sampling_probability,
        }

    @property
    def steps(self) -> int:
        """Number of rounds accounted for."""
        return self._steps

    def reset(self) -> None:
        """Reset accountant to zero rounds."""
        self._steps = 0

    def __repr__(self) -> str:
        return (
            f"PrivacyAccountant("
            f"σ={self.noise_multiplier}, "
            f"q={self.sampling_probability}, "
            f"steps={self._steps})"
        )
