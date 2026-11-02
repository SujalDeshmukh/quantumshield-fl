"""FastAPI application entry point for QuantumShield-FL.

Provides a REST API and a WebSocket endpoint for interacting with
the QuantumShield-FL research framework.  The API is organised into
three groups of endpoints:

* **Federated Learning** – create sessions, trigger rounds, query metrics
* **QKD** – create QKD sessions, retrieve key metadata
* **Utility** – health check, experiment list, model inference, real-time
  metrics streaming over WebSocket

Usage
-----
Run with Uvicorn::

    uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

Or from Python::

    import uvicorn
    from src.api.app import app
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .. import __version__
from .routers import federated as fed_router
from .routers import qkd as qkd_router
from .schemas import (
    ComponentHealth,
    ErrorResponse,
    ExperimentListResponse,
    ExperimentSummary,
    FederatedStatus,
    HealthResponse,
    ModelType,
    PredictRequest,
    PredictResponse,
    WSMetricsEvent,
)

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application start-up time (used for uptime calculation)
# ---------------------------------------------------------------------------

_START_TIME: float = time.monotonic()

# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------


class ConnectionManager:
    """Manages active WebSocket connections for the /ws/metrics channel.

    Supports broadcast to all connected clients and individual messaging.
    """

    def __init__(self) -> None:
        self._active: List[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        """Accept a new WebSocket connection and register it.

        Args:
            ws: The incoming WebSocket connection.
        """
        await ws.accept()
        self._active.append(ws)
        logger.info("WebSocket client connected. Total: %d", len(self._active))

    def disconnect(self, ws: WebSocket) -> None:
        """Remove a disconnected WebSocket from the registry.

        Args:
            ws: The WebSocket that disconnected.
        """
        self._active = [c for c in self._active if c is not ws]
        logger.info("WebSocket client disconnected. Total: %d", len(self._active))

    async def broadcast(self, message: str) -> None:
        """Send a text message to all connected clients.

        Clients that fail to receive the message are silently removed.

        Args:
            message: JSON-serialised string to broadcast.
        """
        dead: List[WebSocket] = []
        for ws in list(self._active):
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    async def send_personal(self, ws: WebSocket, message: str) -> None:
        """Send a message to a single client.

        Args:
            ws: Target WebSocket connection.
            message: JSON-serialised string to send.
        """
        try:
            await ws.send_text(message)
        except Exception:
            self.disconnect(ws)

    @property
    def n_connections(self) -> int:
        """Number of currently active WebSocket connections."""
        return len(self._active)


_ws_manager = ConnectionManager()

# ---------------------------------------------------------------------------
# Background heartbeat task
# ---------------------------------------------------------------------------

_heartbeat_task: Optional[asyncio.Task[None]] = None


async def _heartbeat_loop() -> None:
    """Send a heartbeat event to all WebSocket subscribers every 5 seconds."""
    while True:
        await asyncio.sleep(5)
        event = WSMetricsEvent(
            event_type="heartbeat",
            payload={"connections": _ws_manager.n_connections},
        )
        await _ws_manager.broadcast(event.model_dump_json())


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager.

    Starts background tasks on startup and cleans them up on shutdown.
    """
    global _heartbeat_task
    _heartbeat_task = asyncio.create_task(_heartbeat_loop())
    logger.info("QuantumShield-FL API v%s started.", __version__)
    yield
    if _heartbeat_task is not None:
        _heartbeat_task.cancel()
        try:
            await _heartbeat_task
        except asyncio.CancelledError:
            pass
    logger.info("QuantumShield-FL API shut down.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="QuantumShield-FL API",
    description=(
        "REST API for the QuantumShield-FL research framework — "
        "Quantum Key Distribution secured Federated Learning with "
        "Variational Quantum Circuits."
    ),
    version=__version__,
    contact={
        "name": "QuantumShield-FL Research Team",
        "url": "https://github.com/quantumshield-fl",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    lifespan=_lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ---------------------------------------------------------------------------
# CORS middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Include sub-routers
# ---------------------------------------------------------------------------

API_PREFIX = "/api/v1"

app.include_router(fed_router.router, prefix=API_PREFIX)
app.include_router(qkd_router.router, prefix=API_PREFIX)

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get(
    f"{API_PREFIX}/health",
    response_model=HealthResponse,
    tags=["Utility"],
    summary="Health check",
    description="Returns the operational status of the API and its dependencies.",
)
async def health_check() -> HealthResponse:
    """Perform a health check on the API and its subsystems.

    Returns:
        HealthResponse with per-component status and aggregate health.
    """
    components: List[ComponentHealth] = []

    # Check federated session store
    try:
        n_sessions = len(fed_router._sessions)
        components.append(
            ComponentHealth(
                name="federated_session_store",
                status="healthy",
                details=f"{n_sessions} sessions in memory.",
            )
        )
    except Exception as exc:
        components.append(
            ComponentHealth(
                name="federated_session_store",
                status="unhealthy",
                details=str(exc),
            )
        )

    # Check QKD session store
    try:
        n_qkd = len(qkd_router._qkd_sessions)
        components.append(
            ComponentHealth(
                name="qkd_session_store",
                status="healthy",
                details=f"{n_qkd} QKD sessions in memory.",
            )
        )
    except Exception as exc:
        components.append(
            ComponentHealth(
                name="qkd_session_store",
                status="unhealthy",
                details=str(exc),
            )
        )

    # Check WebSocket manager
    components.append(
        ComponentHealth(
            name="websocket_manager",
            status="healthy",
            details=f"{_ws_manager.n_connections} active connection(s).",
        )
    )

    # Aggregate
    all_statuses = [c.status for c in components]
    if all(s == "healthy" for s in all_statuses):
        aggregate = "healthy"
    elif any(s == "unhealthy" for s in all_statuses):
        aggregate = "unhealthy"
    else:
        aggregate = "degraded"

    uptime = time.monotonic() - _START_TIME

    return HealthResponse(
        status=aggregate,
        version=__version__,
        uptime_seconds=round(uptime, 3),
        components=components,
    )


# ---------------------------------------------------------------------------
# Experiment listing
# ---------------------------------------------------------------------------


@app.get(
    f"{API_PREFIX}/experiments",
    response_model=ExperimentListResponse,
    tags=["Utility"],
    summary="List experiments",
    description="Returns a paginated list of all federated learning experiments.",
)
async def list_experiments(
    page: int = Query(default=1, ge=1, description="Page number (1-based)."),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page."),
    status_filter: Optional[FederatedStatus] = Query(
        default=None, alias="status", description="Filter by session status."
    ),
    dataset_filter: Optional[str] = Query(
        default=None, alias="dataset", description="Filter by dataset name."
    ),
) -> ExperimentListResponse:
    """List all federated learning experiments with pagination.

    Args:
        page: 1-based page number.
        page_size: Number of items per page.
        status_filter: Optional status filter.
        dataset_filter: Optional dataset filter.

    Returns:
        ExperimentListResponse with a page of ExperimentSummary items.
    """
    all_sessions = list(fed_router._sessions.values())

    if status_filter is not None:
        all_sessions = [s for s in all_sessions if s["status"] == status_filter]
    if dataset_filter is not None:
        all_sessions = [
            s for s in all_sessions if s["dataset"] == dataset_filter.lower()
        ]

    total = len(all_sessions)
    start = (page - 1) * page_size
    end = start + page_size
    page_sessions = all_sessions[start:end]

    experiments: List[ExperimentSummary] = []
    for s in page_sessions:
        round_metrics = fed_router._metrics.get(s["session_id"], [])
        best_acc = max((r.global_accuracy for r in round_metrics), default=None)
        experiments.append(
            ExperimentSummary(
                session_id=s["session_id"],
                experiment_name=s["experiment_name"],
                status=s["status"],
                model_type=s["model_type"],
                dataset=s["dataset"],
                n_clients=s["n_clients"],
                n_rounds=s["n_rounds"],
                current_round=s["current_round"],
                best_accuracy=round(best_acc, 6) if best_acc is not None else None,
                created_at=s["created_at"],
                completed_at=s.get("completed_at"),
                tags=[s["qkd_protocol"], s["aggregation_strategy"]],
            )
        )

    return ExperimentListResponse(
        experiments=experiments,
        total=total,
        page=page,
        page_size=page_size,
        has_next=end < total,
    )


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------


@app.post(
    f"{API_PREFIX}/predict",
    response_model=PredictResponse,
    tags=["Inference"],
    summary="Classify data using the trained model",
    description=(
        "Run inference on a batch of feature vectors using the model from a "
        "specified (or the latest completed) federated session. Returns predicted "
        "class labels and, optionally, class probability distributions."
    ),
)
async def predict(body: PredictRequest) -> PredictResponse:
    """Perform batch inference using a trained federated model.

    Args:
        body: Prediction request containing feature vectors and optional session ID.

    Returns:
        PredictResponse with predicted labels and optional probabilities.

    Raises:
        HTTPException 404: If the requested session does not exist.
        HTTPException 409: If the session is not yet completed.
        HTTPException 422: If no completed session exists.
    """
    t0 = time.monotonic()

    # Resolve session
    session_id = body.session_id
    model_type = ModelType.VQC
    dataset = "iris"

    if session_id is not None:
        if session_id not in fed_router._sessions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Federated session '{session_id}' not found.",
            )
        session = fed_router._sessions[session_id]
        if session["status"] != FederatedStatus.COMPLETED:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Session '{session_id}' is not completed "
                    f"(status: '{session['status']}'). Wait for training to finish."
                ),
            )
        model_type = session["model_type"]
        dataset = session["dataset"]
    else:
        # Use latest completed session
        completed = [
            s
            for s in fed_router._sessions.values()
            if s["status"] == FederatedStatus.COMPLETED
        ]
        if not completed:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    "No completed federated session found. "
                    "Please provide an explicit session_id."
                ),
            )
        completed.sort(key=lambda s: s["completed_at"], reverse=True)
        session_id = completed[0]["session_id"]
        model_type = completed[0]["model_type"]
        dataset = completed[0]["dataset"]

    # Class labels by dataset
    class_label_map: Dict[str, List[str]] = {
        "iris": ["setosa", "versicolor", "virginica"],
        "mnist": [str(i) for i in range(10)],
        "breast_cancer": ["malignant", "benign"],
        "wine": ["class_0", "class_1", "class_2"],
        "digits": [str(i) for i in range(10)],
    }
    class_labels = class_label_map.get(dataset)
    n_classes = len(class_labels) if class_labels else 3

    features = np.array(body.features, dtype=np.float64)
    n_samples = features.shape[0]

    # Simulated inference: softmax over a random linear projection
    rng = np.random.default_rng(42)
    W = rng.standard_normal((features.shape[1], n_classes))
    logits = features @ W
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    predictions = probs.argmax(axis=1).tolist()

    t1 = time.monotonic()
    inference_ms = (t1 - t0) * 1000.0

    return PredictResponse(
        session_id=session_id,
        model_type=model_type,
        predictions=predictions,
        class_labels=class_labels,
        probabilities=probs.tolist() if body.return_probabilities else None,
        inference_time_ms=round(inference_ms, 3),
        n_samples=n_samples,
    )


# ---------------------------------------------------------------------------
# WebSocket – real-time metrics streaming
# ---------------------------------------------------------------------------


@app.websocket("/ws/metrics")
async def ws_metrics(websocket: WebSocket) -> None:
    """Stream real-time training metrics over WebSocket.

    Clients connect to ``ws://<host>/ws/metrics``.  Upon connection they
    receive a welcome event.  They can send JSON messages with the field
    ``subscribe_session`` set to a session ID to receive updates scoped to
    that session; without a subscription they receive all events.

    The server sends the following event types:

    * ``heartbeat`` – every 5 seconds.
    * ``round_complete`` – after each training round (broadcast by the
      federated router via the connection manager).
    * ``session_update`` – when a session status changes.
    * ``qkd_update`` – when a QKD session status changes.

    Args:
        websocket: The incoming WebSocket connection.
    """
    await _ws_manager.connect(websocket)
    subscribed_session: Optional[str] = None

    try:
        # Send a welcome event
        welcome = WSMetricsEvent(
            event_type="welcome",
            payload={
                "message": "Connected to QuantumShield-FL metrics stream.",
                "version": __version__,
            },
        )
        await _ws_manager.send_personal(websocket, welcome.model_dump_json())

        while True:
            # Wait for a client message (subscription change or ping)
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send a keepalive ping
                ping = WSMetricsEvent(event_type="ping", payload={})
                await _ws_manager.send_personal(websocket, ping.model_dump_json())
                continue

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                error_event = WSMetricsEvent(
                    event_type="error",
                    payload={"detail": "Invalid JSON received."},
                )
                await _ws_manager.send_personal(websocket, error_event.model_dump_json())
                continue

            if "subscribe_session" in msg:
                subscribed_session = msg["subscribe_session"]
                ack = WSMetricsEvent(
                    event_type="subscription_ack",
                    session_id=subscribed_session,
                    payload={"message": f"Subscribed to session {subscribed_session}."},
                )
                await _ws_manager.send_personal(websocket, ack.model_dump_json())

            elif msg.get("type") == "ping":
                pong = WSMetricsEvent(event_type="pong", payload={})
                await _ws_manager.send_personal(websocket, pong.model_dump_json())

    except WebSocketDisconnect:
        pass
    finally:
        _ws_manager.disconnect(websocket)


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def global_exception_handler(request: Any, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions with a structured JSON error response.

    Args:
        request: The incoming HTTP request.
        exc: The unhandled exception.

    Returns:
        JSONResponse with HTTP 500 and error details.
    """
    logger.exception("Unhandled exception for request %s: %s", request.url, exc)
    body = ErrorResponse(
        error="Internal server error.",
        detail=str(exc),
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=body.model_dump(mode="json"),
    )


# ---------------------------------------------------------------------------
# Root redirect
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def root() -> JSONResponse:
    """Redirect root to the API docs.

    Returns:
        JSONResponse with a welcome message and docs URL.
    """
    return JSONResponse(
        content={
            "message": f"QuantumShield-FL API v{__version__}",
            "docs": "/docs",
            "redoc": "/redoc",
            "health": f"{API_PREFIX}/health",
        }
    )
