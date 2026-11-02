"""Pydantic schemas for the QuantumShield-FL REST API.

Defines all request and response models used by the FastAPI application,
covering federated learning sessions, QKD key management, model inference,
health checks, and experiment listing.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class FederatedStatus(str, Enum):
    """Possible states of a federated learning session."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class AggregationStrategy(str, Enum):
    """Model aggregation strategies."""

    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    SCAFFOLD = "scaffold"
    QUANTUM_FEDAVG = "quantum_fedavg"


class QKDProtocol(str, Enum):
    """Supported QKD protocols."""

    BB84 = "BB84"
    E91 = "E91"
    B92 = "B92"
    SARG04 = "SARG04"


class QKDSessionStatus(str, Enum):
    """Possible states of a QKD session."""

    INITIALISING = "initialising"
    KEY_EXCHANGE = "key_exchange"
    SIFTING = "sifting"
    ERROR_CORRECTION = "error_correction"
    PRIVACY_AMPLIFICATION = "privacy_amplification"
    COMPLETED = "completed"
    ABORTED = "aborted"


class ModelType(str, Enum):
    """Supported model types."""

    VQC = "vqc"
    CLASSICAL_MLP = "classical_mlp"
    HYBRID = "hybrid"


# ---------------------------------------------------------------------------
# Federated Learning – Request Schemas
# ---------------------------------------------------------------------------


class FederatedStartRequest(BaseModel):
    """Request body for starting a new federated learning session."""

    experiment_name: str = Field(
        default="default-experiment",
        min_length=1,
        max_length=128,
        description="Human-readable name for the experiment.",
        examples=["mnist-vqc-bb84-v1"],
    )

    @model_validator(mode="before")
    @classmethod
    def _handle_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "experiment_name" not in data:
                data["experiment_name"] = "default-experiment"
            if "aggregation" in data and "aggregation_strategy" not in data:
                data["aggregation_strategy"] = data["aggregation"]
            if "protocol" in data and "qkd_protocol" not in data:
                data["qkd_protocol"] = data["protocol"]
            if "qkd_protocol" in data and isinstance(data["qkd_protocol"], str):
                data["qkd_protocol"] = data["qkd_protocol"].upper()
            if "aggregation_strategy" in data and isinstance(data["aggregation_strategy"], str):
                data["aggregation_strategy"] = data["aggregation_strategy"].lower()
        return data
    n_clients: int = Field(
        default=5,
        ge=2,
        le=100,
        description="Number of federated clients.",
    )
    n_rounds: int = Field(
        default=10,
        ge=1,
        le=500,
        description="Total number of federated training rounds.",
    )
    n_qubits: int = Field(
        default=4,
        ge=2,
        le=20,
        description="Number of qubits in the Variational Quantum Circuit.",
    )
    n_layers: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Number of variational layers in the VQC.",
    )
    model_type: ModelType = Field(
        default=ModelType.VQC,
        description="Type of model to train.",
    )
    aggregation_strategy: AggregationStrategy = Field(
        default=AggregationStrategy.FEDAVG,
        description="Strategy used to aggregate client model updates.",
    )
    qkd_protocol: QKDProtocol = Field(
        default=QKDProtocol.BB84,
        description="QKD protocol used to secure parameter transmission.",
    )
    dataset: str = Field(
        default="iris",
        description="Dataset identifier (e.g. 'iris', 'mnist').",
        examples=["iris", "mnist"],
    )
    local_epochs: int = Field(
        default=1,
        ge=1,
        le=50,
        description="Number of local training epochs per round per client.",
    )
    learning_rate: float = Field(
        default=0.01,
        gt=0.0,
        lt=1.0,
        description="Learning rate for local optimisers.",
    )
    differential_privacy: bool = Field(
        default=False,
        description="Whether to apply differential privacy noise.",
    )
    dp_epsilon: float = Field(
        default=1.0,
        gt=0.0,
        description="Differential privacy epsilon (privacy budget).",
    )
    dp_delta: float = Field(
        default=1e-5,
        gt=0.0,
        lt=1.0,
        description="Differential privacy delta.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata attached to the session.",
    )

    @field_validator("dataset")
    @classmethod
    def validate_dataset(cls, v: str) -> str:
        """Ensure the requested dataset is supported."""
        supported = {"iris", "mnist", "breast_cancer", "wine", "digits"}
        if v.lower() not in supported:
            raise ValueError(
                f"Dataset '{v}' is not supported. Choose from: {supported}"
            )
        return v.lower()


class TriggerRoundRequest(BaseModel):
    """Request body for triggering a single federated training round."""

    client_ids: Optional[List[str]] = Field(
        default=None,
        description="Specific client IDs to include; None means all active clients.",
    )
    force: bool = Field(
        default=False,
        description="Force a round even if the session is not in RUNNING state.",
    )


# ---------------------------------------------------------------------------
# Federated Learning – Response Schemas
# ---------------------------------------------------------------------------


class ClientInfo(BaseModel):
    """Summary information about a single federated client."""

    client_id: str = Field(..., description="Unique client identifier.")
    n_samples: int = Field(..., description="Number of training samples on this client.")
    status: str = Field(..., description="Current status of the client.")
    last_seen: Optional[datetime] = Field(
        default=None, description="Timestamp of last communication."
    )
    rounds_participated: int = Field(
        default=0, description="Number of rounds this client has participated in."
    )


class RoundMetrics(BaseModel):
    """Metrics recorded for a single federated training round."""

    round_number: int = Field(..., description="Round index (1-based).")
    global_loss: float = Field(..., description="Aggregated global loss.")
    global_accuracy: float = Field(..., description="Aggregated global accuracy.")
    avg_client_loss: float = Field(..., description="Average loss across clients.")
    avg_client_accuracy: float = Field(..., description="Average accuracy across clients.")
    communication_cost_bytes: int = Field(
        default=0, description="Total bytes transmitted in this round."
    )
    qkd_key_rate_bps: Optional[float] = Field(
        default=None, description="QKD key generation rate for this round (bits/s)."
    )
    qber: Optional[float] = Field(
        default=None, description="Quantum Bit Error Rate measured in this round."
    )
    duration_seconds: float = Field(..., description="Wall-clock time for this round.")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="UTC timestamp of round completion."
    )


class FederatedSessionResponse(BaseModel):
    """Full description of a federated learning session."""

    session_id: str = Field(..., description="Unique session identifier (UUID).")
    experiment_name: str = Field(..., description="Name of the experiment.")
    status: FederatedStatus = Field(..., description="Current session status.")
    n_clients: int = Field(..., description="Number of federated clients.")
    n_rounds: int = Field(..., description="Total planned rounds.")
    current_round: int = Field(..., description="Last completed round index.")
    model_type: ModelType
    aggregation_strategy: AggregationStrategy
    qkd_protocol: QKDProtocol
    dataset: str
    clients: List[ClientInfo] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    config: Dict[str, Any] = Field(default_factory=dict)


class FederatedStartResponse(BaseModel):
    """Response returned when a new federated session is created."""

    session_id: str
    status: FederatedStatus
    message: str
    created_at: datetime
    estimated_duration_seconds: Optional[float] = None


class FederatedStatusResponse(BaseModel):
    """Lightweight status response for a federated session."""

    session_id: str
    status: FederatedStatus
    current_round: int
    n_rounds: int
    progress_pct: float = Field(..., ge=0.0, le=100.0)
    latest_accuracy: Optional[float] = None
    latest_loss: Optional[float] = None
    updated_at: datetime


class FederatedMetricsResponse(BaseModel):
    """Full metrics history for a federated session."""

    session_id: str
    experiment_name: str
    rounds: List[RoundMetrics] = Field(default_factory=list)
    best_accuracy: float = Field(default=0.0)
    best_round: int = Field(default=0)
    total_duration_seconds: float = Field(default=0.0)
    privacy_budget_consumed: Optional[float] = None
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class TriggerRoundResponse(BaseModel):
    """Response returned after triggering a training round."""

    session_id: str
    round_number: int
    status: str
    message: str
    metrics: Optional[RoundMetrics] = None


# ---------------------------------------------------------------------------
# QKD – Request Schemas
# ---------------------------------------------------------------------------


class QKDSessionRequest(BaseModel):
    """Request body for creating a new QKD session."""

    @model_validator(mode="before")
    @classmethod
    def _handle_qkd_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "alice_id" not in data:
                data["alice_id"] = "alice"
            if "bob_id" not in data:
                data["bob_id"] = "bob"
            if "protocol" in data and isinstance(data["protocol"], str):
                data["protocol"] = data["protocol"].upper()
        return data

    alice_id: str = Field(
        default="alice",
        min_length=1,
        max_length=64,
        description="Identifier for Alice (key sender).",
    )
    bob_id: str = Field(
        default="bob",
        min_length=1,
        max_length=64,
        description="Identifier for Bob (key receiver).",
    )
    protocol: QKDProtocol = Field(
        default=QKDProtocol.BB84,
        description="QKD protocol to use.",
    )
    key_length_bits: int = Field(
        default=256,
        ge=64,
        le=4096,
        description="Target secure key length in bits.",
    )
    n_qubits: int = Field(
        default=1000,
        ge=100,
        le=100_000,
        description="Number of qubits (raw quantum bits) to transmit.",
    )
    noise_model: Optional[str] = Field(
        default=None,
        description="Noise model identifier to simulate channel imperfections.",
    )
    eavesdrop_probability: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Probability of eavesdropper interception (simulation only).",
    )


# ---------------------------------------------------------------------------
# QKD – Response Schemas
# ---------------------------------------------------------------------------


class QKDKeyInfo(BaseModel):
    """Information about a generated QKD key (without exposing raw key bytes)."""

    session_id: str
    key_id: str
    key_length_bits: int
    raw_bits_transmitted: int
    sifted_bits: int
    qber: float = Field(..., description="Quantum Bit Error Rate (0–1).")
    secure_key_rate_bps: float
    privacy_amplification_factor: float
    generated_at: datetime
    expires_at: Optional[datetime] = None
    is_secure: bool = Field(
        ...,
        description="True when QBER is below the security threshold (~11% for BB84).",
    )


class QKDSessionResponse(BaseModel):
    """Full QKD session description."""

    session_id: str
    alice_id: str
    bob_id: str
    protocol: QKDProtocol
    status: QKDSessionStatus
    key_length_bits: int
    n_qubits: int
    noise_model: Optional[str]
    eavesdrop_probability: float
    created_at: datetime
    updated_at: datetime
    key_info: Optional[QKDKeyInfo] = None


# ---------------------------------------------------------------------------
# Prediction – Request / Response Schemas
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    """Request body for model inference."""

    @model_validator(mode="before")
    @classmethod
    def _handle_1d_features(cls, data: Any) -> Any:
        if isinstance(data, dict) and "features" in data:
            f = data["features"]
            if isinstance(f, list) and len(f) > 0 and not isinstance(f[0], list):
                data["features"] = [f]
        return data

    session_id: Optional[str] = Field(
        default=None,
        description="Federated session ID whose trained model to use. "
        "Uses the latest completed session if omitted.",
    )
    features: List[List[float]] = Field(
        ...,
        description="Batch of feature vectors. Shape: (batch_size, n_features).",
        min_length=1,
    )
    return_probabilities: bool = Field(
        default=True,
        description="If True, return class probability distribution alongside the label.",
    )

    @field_validator("features")
    @classmethod
    def features_not_empty(cls, v: List[List[float]]) -> List[List[float]]:
        """Validate that each feature vector is non-empty."""
        for i, vec in enumerate(v):
            if len(vec) == 0:
                raise ValueError(f"Feature vector at index {i} is empty.")
        return v


class PredictResponse(BaseModel):
    """Response body for model inference."""

    session_id: Optional[str]
    model_type: ModelType
    predictions: List[int] = Field(..., description="Predicted class indices.")
    class_labels: Optional[List[str]] = Field(
        default=None, description="Human-readable class label names."
    )
    probabilities: Optional[List[List[float]]] = Field(
        default=None,
        description="Class probability distributions. Shape: (batch_size, n_classes).",
    )
    inference_time_ms: float
    n_samples: int


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------


class ComponentHealth(BaseModel):
    """Health status for a single system component."""

    name: str
    status: str = Field(..., description="'healthy', 'degraded', or 'unhealthy'.")
    latency_ms: Optional[float] = None
    details: Optional[str] = None


class HealthResponse(BaseModel):
    """Overall system health response."""

    status: str = Field(
        ..., description="Aggregate status: 'healthy', 'degraded', or 'unhealthy'."
    )
    version: str
    uptime_seconds: float
    components: List[ComponentHealth] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Experiment Listing
# ---------------------------------------------------------------------------


class ExperimentSummary(BaseModel):
    """Summary of a single experiment entry."""

    session_id: str
    experiment_name: str
    status: FederatedStatus
    model_type: ModelType
    dataset: str
    n_clients: int
    n_rounds: int
    current_round: int
    best_accuracy: Optional[float] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    tags: List[str] = Field(default_factory=list)


class ExperimentListResponse(BaseModel):
    """Paginated list of experiments."""

    experiments: List[ExperimentSummary]
    total: int
    page: int
    page_size: int
    has_next: bool


# ---------------------------------------------------------------------------
# WebSocket – Real-time Metrics
# ---------------------------------------------------------------------------


class WSMetricsEvent(BaseModel):
    """Payload pushed over the WebSocket /ws/metrics channel."""

    event_type: str = Field(
        ...,
        description="One of: 'round_complete', 'session_update', 'qkd_update', 'heartbeat'.",
    )
    session_id: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Generic error response
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    """Standard error response body."""

    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
