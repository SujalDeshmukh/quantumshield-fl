"""FastAPI router for Federated Learning endpoints.

Provides endpoints to create and manage federated training sessions,
trigger training rounds, and retrieve session metrics. The business
logic is implemented directly in this module using an in-memory
session store; in production this would be backed by a database and
a distributed task queue (e.g. Celery + Redis).
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, status

from ..schemas import (
    AggregationStrategy,
    ClientInfo,
    FederatedMetricsResponse,
    FederatedSessionResponse,
    FederatedStartRequest,
    FederatedStartResponse,
    FederatedStatus,
    FederatedStatusResponse,
    ModelType,
    QKDProtocol,
    RoundMetrics,
    TriggerRoundRequest,
    TriggerRoundResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/federated",
    tags=["Federated Learning"],
    responses={
        404: {"description": "Session not found."},
        409: {"description": "Session state conflict."},
    },
)

# ---------------------------------------------------------------------------
# In-memory session store (replace with a persistent backend in production)
# ---------------------------------------------------------------------------

_sessions: Dict[str, Dict[str, Any]] = {}
_metrics: Dict[str, List[RoundMetrics]] = {}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _session_or_404(session_id: str) -> Dict[str, Any]:
    """Return the session dict or raise HTTP 404."""
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Federated session '{session_id}' not found.",
        )
    return session


def _make_clients(n_clients: int, dataset: str) -> List[ClientInfo]:
    """Fabricate initial client info entries for a new session."""
    sample_counts = _distribute_samples(n_clients, dataset)
    return [
        ClientInfo(
            client_id=f"client-{i:03d}",
            n_samples=sample_counts[i],
            status="idle",
            last_seen=None,
            rounds_participated=0,
        )
        for i in range(n_clients)
    ]


def _distribute_samples(n_clients: int, dataset: str) -> List[int]:
    """Return a Dirichlet-distributed sample count per client."""
    dataset_sizes = {
        "iris": 150,
        "mnist": 60_000,
        "breast_cancer": 569,
        "wine": 178,
        "digits": 1797,
    }
    total = dataset_sizes.get(dataset, 1000)
    # Dirichlet distribution for non-IID split
    rng = np.random.default_rng(42)
    proportions = rng.dirichlet(alpha=[0.5] * n_clients)
    counts = (proportions * total).astype(int).tolist()
    # Fix rounding so counts sum to total
    diff = total - sum(counts)
    counts[0] += diff
    return counts


def _estimate_duration(req: FederatedStartRequest) -> float:
    """Rough estimate of total training time in seconds."""
    # Very rough heuristic: each round takes ~2 s per client per layer of VQC
    return req.n_rounds * req.n_clients * req.n_layers * 0.5


def _simulate_round(
    session: Dict[str, Any],
    round_num: int,
    rng: np.random.Generator,
) -> RoundMetrics:
    """Simulate a single federated training round and return its metrics.

    In a real deployment this would coordinate actual client training,
    gradient aggregation, and QKD-secured parameter upload.
    """
    t_start = time.monotonic()
    n_clients = session["n_clients"]
    model_type: ModelType = session["model_type"]
    qkd_protocol: QKDProtocol = session["qkd_protocol"]

    # Simulate convergence: loss decays, accuracy climbs
    base_loss = 1.0 * math.exp(-0.15 * round_num) + 0.05 * rng.normal()
    base_acc = 1.0 - 0.9 * math.exp(-0.2 * round_num) + 0.02 * rng.normal()
    base_acc = float(np.clip(base_acc, 0.0, 1.0))
    base_loss = float(max(base_loss, 0.01))

    client_losses = [
        float(max(base_loss + 0.05 * rng.normal(), 0.01))
        for _ in range(n_clients)
    ]
    client_accs = [
        float(np.clip(base_acc + 0.03 * rng.normal(), 0.0, 1.0))
        for _ in range(n_clients)
    ]

    # Simulate QKD parameters
    qber = float(np.clip(0.03 + 0.005 * rng.normal(), 0.0, 0.11))
    key_rate = float(max(1000.0 + 200.0 * rng.normal(), 100.0))

    # Simulate communication cost
    param_bytes = 128 if model_type == ModelType.VQC else 4096
    comm_cost = n_clients * param_bytes * 2  # upload + download

    t_end = time.monotonic()

    return RoundMetrics(
        round_number=round_num,
        global_loss=base_loss,
        global_accuracy=base_acc,
        avg_client_loss=float(np.mean(client_losses)),
        avg_client_accuracy=float(np.mean(client_accs)),
        communication_cost_bytes=comm_cost,
        qkd_key_rate_bps=key_rate,
        qber=qber,
        duration_seconds=t_end - t_start + 0.1,  # add artificial latency floor
        timestamp=datetime.utcnow(),
    )


async def _run_federated_session(session_id: str) -> None:
    """Background task that drives a full federated training run.

    Iterates through all rounds sequentially, updating the session state
    and appending metrics after each round.
    """
    session = _sessions.get(session_id)
    if session is None:
        logger.error("Background task: session %s not found.", session_id)
        return

    rng = np.random.default_rng(int(session_id.replace("-", "")[:8], 16) & 0xFFFFFFFF)
    n_rounds: int = session["n_rounds"]

    logger.info("Starting federated session %s for %d rounds.", session_id, n_rounds)
    _sessions[session_id]["status"] = FederatedStatus.RUNNING
    _sessions[session_id]["updated_at"] = datetime.utcnow()

    try:
        for round_num in range(1, n_rounds + 1):
            # Allow cancellation between rounds
            await asyncio.sleep(0)

            metrics = _simulate_round(session, round_num, rng)
            _metrics.setdefault(session_id, []).append(metrics)

            _sessions[session_id]["current_round"] = round_num
            _sessions[session_id]["updated_at"] = datetime.utcnow()

            # Update client participation counters
            for client in _sessions[session_id]["clients"]:
                client["rounds_participated"] = round_num
                client["last_seen"] = datetime.utcnow().isoformat()
                client["status"] = "idle"

            logger.debug(
                "Session %s | Round %d/%d | loss=%.4f acc=%.4f",
                session_id,
                round_num,
                n_rounds,
                metrics.global_loss,
                metrics.global_accuracy,
            )

        _sessions[session_id]["status"] = FederatedStatus.COMPLETED
        _sessions[session_id]["completed_at"] = datetime.utcnow()
        logger.info("Federated session %s completed successfully.", session_id)

    except asyncio.CancelledError:
        _sessions[session_id]["status"] = FederatedStatus.FAILED
        _sessions[session_id]["updated_at"] = datetime.utcnow()
        logger.warning("Federated session %s was cancelled.", session_id)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/start",
    response_model=FederatedStartResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start a new federated learning session",
    description=(
        "Creates a new federated learning session with the given configuration "
        "and immediately starts training in the background. Returns the session "
        "ID that can be used to query status and metrics."
    ),
)
async def start_session(
    body: FederatedStartRequest,
    background_tasks: BackgroundTasks,
) -> FederatedStartResponse:
    """Start a new federated learning session.

    Args:
        body: Session configuration parameters.
        background_tasks: FastAPI background task runner.

    Returns:
        FederatedStartResponse with the newly created session ID.
    """
    session_id = str(uuid.uuid4())
    now = datetime.utcnow()
    clients = _make_clients(body.n_clients, body.dataset)

    _sessions[session_id] = {
        "session_id": session_id,
        "experiment_name": body.experiment_name,
        "status": FederatedStatus.PENDING,
        "n_clients": body.n_clients,
        "n_rounds": body.n_rounds,
        "current_round": 0,
        "model_type": body.model_type,
        "aggregation_strategy": body.aggregation_strategy,
        "qkd_protocol": body.qkd_protocol,
        "dataset": body.dataset,
        "clients": [c.model_dump() for c in clients],
        "created_at": now,
        "updated_at": now,
        "completed_at": None,
        "config": body.model_dump(
            exclude={"experiment_name", "metadata"}
        ),
    }
    _metrics[session_id] = []

    estimated = _estimate_duration(body)

    # Launch background training
    background_tasks.add_task(_run_federated_session, session_id)

    logger.info(
        "Created federated session %s ('%s') with %d clients, %d rounds.",
        session_id,
        body.experiment_name,
        body.n_clients,
        body.n_rounds,
    )

    return FederatedStartResponse(
        session_id=session_id,
        status=FederatedStatus.PENDING,
        message=(
            f"Federated session '{body.experiment_name}' created. "
            "Training will begin momentarily."
        ),
        created_at=now,
        estimated_duration_seconds=estimated,
    )


@router.get(
    "/{session_id}/status",
    response_model=FederatedStatusResponse,
    summary="Get the status of a federated learning session",
    description="Returns the current status, progress percentage, and latest metrics for a session.",
)
async def get_status(session_id: str) -> FederatedStatusResponse:
    """Retrieve current status of a federated training session.

    Args:
        session_id: UUID of the session to query.

    Returns:
        FederatedStatusResponse with progress and latest metric snapshot.

    Raises:
        HTTPException 404: If no session with the given ID exists.
    """
    session = _session_or_404(session_id)
    round_metrics = _metrics.get(session_id, [])

    latest_acc: Optional[float] = None
    latest_loss: Optional[float] = None
    if round_metrics:
        latest = round_metrics[-1]
        latest_acc = latest.global_accuracy
        latest_loss = latest.global_loss

    n_rounds = session["n_rounds"]
    current_round = session["current_round"]
    progress = (current_round / n_rounds * 100.0) if n_rounds > 0 else 0.0

    return FederatedStatusResponse(
        session_id=session_id,
        status=session["status"],
        current_round=current_round,
        n_rounds=n_rounds,
        progress_pct=round(progress, 2),
        latest_accuracy=latest_acc,
        latest_loss=latest_loss,
        updated_at=session["updated_at"],
    )


@router.post(
    "/{session_id}/round",
    response_model=TriggerRoundResponse,
    summary="Trigger a single federated training round",
    description=(
        "Manually triggers one additional training round for the given session. "
        "Useful when the session was paused or when running step-by-step experiments."
    ),
)
async def trigger_round(
    session_id: str,
    body: TriggerRoundRequest,
) -> TriggerRoundResponse:
    """Manually trigger a single training round.

    Args:
        session_id: UUID of the target session.
        body: Options for this round (optional client subset, force flag).

    Returns:
        TriggerRoundResponse containing the newly computed round metrics.

    Raises:
        HTTPException 404: If the session does not exist.
        HTTPException 409: If the session is not in a state that allows triggering.
    """
    session = _session_or_404(session_id)

    allowable_states = {FederatedStatus.RUNNING, FederatedStatus.PAUSED}
    if session["status"] not in allowable_states and not body.force:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Session is in '{session['status']}' state. "
                "Set force=true to trigger a round anyway."
            ),
        )

    rng = np.random.default_rng()
    next_round = session["current_round"] + 1
    metrics = _simulate_round(session, next_round, rng)

    _metrics.setdefault(session_id, []).append(metrics)
    _sessions[session_id]["current_round"] = next_round
    _sessions[session_id]["updated_at"] = datetime.utcnow()

    logger.info(
        "Manually triggered round %d for session %s.", next_round, session_id
    )

    return TriggerRoundResponse(
        session_id=session_id,
        round_number=next_round,
        status="success",
        message=f"Round {next_round} completed successfully.",
        metrics=metrics,
    )


@router.get(
    "/{session_id}/metrics",
    response_model=FederatedMetricsResponse,
    summary="Get training metrics for a federated session",
    description="Returns the full history of per-round metrics, best accuracy, and privacy budget usage.",
)
async def get_metrics(
    session_id: str,
    last_n: Optional[int] = Query(
        default=None,
        ge=1,
        description="If provided, return only the last N rounds of metrics.",
    ),
) -> FederatedMetricsResponse:
    """Retrieve training metrics for a federated session.

    Args:
        session_id: UUID of the session.
        last_n: Optional limit on how many recent rounds to return.

    Returns:
        FederatedMetricsResponse with per-round metric history.

    Raises:
        HTTPException 404: If the session does not exist.
    """
    session = _session_or_404(session_id)
    all_rounds = _metrics.get(session_id, [])

    rounds = all_rounds if last_n is None else all_rounds[-last_n:]

    best_acc = max((r.global_accuracy for r in all_rounds), default=0.0)
    best_rnd = next(
        (r.round_number for r in all_rounds if r.global_accuracy == best_acc), 0
    )
    total_duration = sum(r.duration_seconds for r in all_rounds)

    # Privacy budget: sum of Gaussian mechanism noise applications (heuristic)
    config = session.get("config", {})
    dp_epsilon: Optional[float] = None
    if config.get("differential_privacy"):
        epsilon_per_round = config.get("dp_epsilon", 1.0)
        dp_epsilon = epsilon_per_round * len(all_rounds)

    return FederatedMetricsResponse(
        session_id=session_id,
        experiment_name=session["experiment_name"],
        rounds=rounds,
        best_accuracy=round(best_acc, 6),
        best_round=best_rnd,
        total_duration_seconds=round(total_duration, 3),
        privacy_budget_consumed=dp_epsilon,
    )


@router.get(
    "",
    response_model=List[FederatedSessionResponse],
    summary="List all federated sessions",
    description="Returns a list of all federated learning sessions known to this server.",
)
async def list_sessions(
    status_filter: Optional[FederatedStatus] = Query(
        default=None, alias="status", description="Filter by session status."
    ),
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> List[FederatedSessionResponse]:
    """List federated learning sessions with optional filtering.

    Args:
        status_filter: When provided, only sessions with this status are returned.
        limit: Maximum number of sessions to return.
        offset: Number of sessions to skip (for pagination).

    Returns:
        List of FederatedSessionResponse objects.
    """
    sessions = list(_sessions.values())
    if status_filter is not None:
        sessions = [s for s in sessions if s["status"] == status_filter]

    sessions = sessions[offset : offset + limit]

    result = []
    for s in sessions:
        clients = [ClientInfo(**c) for c in s.get("clients", [])]
        result.append(
            FederatedSessionResponse(
                session_id=s["session_id"],
                experiment_name=s["experiment_name"],
                status=s["status"],
                n_clients=s["n_clients"],
                n_rounds=s["n_rounds"],
                current_round=s["current_round"],
                model_type=s["model_type"],
                aggregation_strategy=s["aggregation_strategy"],
                qkd_protocol=s["qkd_protocol"],
                dataset=s["dataset"],
                clients=clients,
                created_at=s["created_at"],
                updated_at=s["updated_at"],
                completed_at=s.get("completed_at"),
                config=s.get("config", {}),
            )
        )
    return result


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a federated session",
    description="Removes a session and its metrics from the server. Cannot be undone.",
)
async def delete_session(session_id: str) -> None:
    """Delete a federated learning session.

    Args:
        session_id: UUID of the session to delete.

    Raises:
        HTTPException 404: If the session does not exist.
    """
    _session_or_404(session_id)
    _sessions.pop(session_id, None)
    _metrics.pop(session_id, None)
    logger.info("Deleted federated session %s.", session_id)
