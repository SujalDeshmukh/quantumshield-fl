"""FastAPI router for Quantum Key Distribution (QKD) endpoints.

Implements a simulation-backed QKD service that models the BB84, E91,
B92, and SARG04 protocols. In a production deployment the simulation
would be replaced by calls to actual QKD hardware or a photonic
simulator (e.g. NetSquid or SimulaQron).

Session lifecycle
-----------------
1. Client POSTs to ``/qkd/session`` to create a session.
2. The server runs key exchange asynchronously in the background.
3. Client GETs ``/qkd/session/{id}/key`` once the session is COMPLETED
   to retrieve key metadata (not the raw key bytes – those are delivered
   out-of-band to the authenticated parties).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, BackgroundTasks, HTTPException, status

from ..schemas import (
    QKDKeyInfo,
    QKDProtocol,
    QKDSessionRequest,
    QKDSessionResponse,
    QKDSessionStatus,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/qkd",
    tags=["Quantum Key Distribution"],
    responses={
        404: {"description": "QKD session not found."},
    },
)

# ---------------------------------------------------------------------------
# In-memory QKD session store
# ---------------------------------------------------------------------------

_qkd_sessions: Dict[str, Dict[str, Any]] = {}

# Security thresholds per protocol (QBER %)
_QBER_THRESHOLDS: Dict[QKDProtocol, float] = {
    QKDProtocol.BB84: 0.11,
    QKDProtocol.E91: 0.11,
    QKDProtocol.B92: 0.12,
    QKDProtocol.SARG04: 0.125,
}

# Sifting efficiency per protocol (fraction of transmitted qubits kept)
_SIFTING_EFFICIENCY: Dict[QKDProtocol, float] = {
    QKDProtocol.BB84: 0.5,
    QKDProtocol.E91: 0.5,
    QKDProtocol.B92: 0.25,
    QKDProtocol.SARG04: 0.5,
}


# ---------------------------------------------------------------------------
# Simulation utilities
# ---------------------------------------------------------------------------


def _simulate_qber(
    protocol: QKDProtocol,
    eavesdrop_prob: float,
    noise_model: Optional[str],
    rng: np.random.Generator,
) -> float:
    """Compute a realistic simulated QBER.

    Args:
        protocol: The QKD protocol in use.
        eavesdrop_prob: Probability that an eavesdropper intercepts each qubit.
        noise_model: Optional noise model identifier (e.g. 'depolarising').
        rng: NumPy random generator for reproducibility.

    Returns:
        Simulated QBER value in [0, 1].
    """
    # Base channel QBER (detector dark counts + optical loss)
    base_qber = 0.02 + 0.005 * rng.normal()

    # Eavesdropper contribution: BB84 intercept-resend introduces 25% QBER
    eve_contribution = {
        QKDProtocol.BB84: 0.25,
        QKDProtocol.E91: 0.0,    # E91 detects via Bell inequality violation
        QKDProtocol.B92: 0.25,
        QKDProtocol.SARG04: 0.25,
    }
    qber = base_qber + eavesdrop_prob * eve_contribution[protocol]

    # Additional depolarising noise
    if noise_model == "depolarising":
        qber += 0.03 * rng.exponential()
    elif noise_model == "amplitude_damping":
        qber += 0.02 * rng.uniform()

    return float(np.clip(qber, 0.0, 0.5))


def _simulate_secure_key_rate(
    protocol: QKDProtocol,
    n_qubits: int,
    qber: float,
    sifted_bits: int,
) -> float:
    """Estimate the secure key rate in bits per second.

    Uses the standard BB84 asymptotic key rate formula for all protocols
    as a simplification:
        r = sifted_bits * (1 - h(qber) - h(qber))
    where h is the binary entropy function.

    Args:
        protocol: QKD protocol (influences base rate factor).
        n_qubits: Raw transmitted qubit count.
        qber: Estimated Quantum Bit Error Rate.
        sifted_bits: Number of bits remaining after basis sifting.

    Returns:
        Secure key rate in bits per second.
    """
    def binary_entropy(p: float) -> float:
        if p <= 0.0 or p >= 1.0:
            return 0.0
        return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

    h_qber = binary_entropy(qber)
    # Asymptotic secure fraction
    secure_fraction = max(0.0, 1.0 - 2 * h_qber)
    secure_bits = sifted_bits * secure_fraction

    # Assume 1 µs per qubit transmission → rate in bits/second
    transmission_time_s = n_qubits * 1e-6
    if transmission_time_s == 0:
        return 0.0
    return secure_bits / transmission_time_s


def _privacy_amplification_factor(qber: float) -> float:
    """Return the privacy amplification compression factor.

    Args:
        qber: Quantum Bit Error Rate.

    Returns:
        Fraction of sifted key that survives privacy amplification.
    """
    def binary_entropy(p: float) -> float:
        if p <= 0.0 or p >= 1.0:
            return 0.0
        return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

    return max(0.0, 1.0 - 2 * binary_entropy(qber))


def _derive_key_id(session_id: str, alice_id: str, bob_id: str) -> str:
    """Derive a deterministic key identifier from session participants.

    Args:
        session_id: UUID of the QKD session.
        alice_id: Identifier of Alice.
        bob_id: Identifier of Bob.

    Returns:
        Hex-encoded SHA-256 digest truncated to 32 characters.
    """
    raw = f"{session_id}:{alice_id}:{bob_id}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


async def _run_qkd_protocol(session_id: str) -> None:
    """Background coroutine that simulates the full QKD protocol execution.

    Progresses through the QKD lifecycle states, simulating realistic
    processing delays. Updates the session store at each stage.

    Args:
        session_id: UUID of the QKD session to run.
    """
    session = _qkd_sessions.get(session_id)
    if session is None:
        logger.error("QKD background task: session %s not found.", session_id)
        return

    rng = np.random.default_rng(
        int(hashlib.md5(session_id.encode()).hexdigest()[:8], 16)
    )
    protocol: QKDProtocol = session["protocol"]
    n_qubits: int = session["n_qubits"]
    eavesdrop_prob: float = session["eavesdrop_probability"]
    noise_model: Optional[str] = session["noise_model"]

    def _update(new_status: QKDSessionStatus) -> None:
        _qkd_sessions[session_id]["status"] = new_status
        _qkd_sessions[session_id]["updated_at"] = datetime.utcnow()
        logger.debug("QKD session %s → %s", session_id, new_status)

    try:
        # Stage 1: Quantum transmission (key exchange)
        _update(QKDSessionStatus.KEY_EXCHANGE)
        await asyncio.sleep(0.05 * n_qubits / 1000)  # proportional to qubit count

        # Stage 2: Basis sifting
        _update(QKDSessionStatus.SIFTING)
        await asyncio.sleep(0.02)

        sifting_eff = _SIFTING_EFFICIENCY[protocol]
        sifted_bits = int(n_qubits * sifting_eff)

        # Stage 3: Error estimation and correction
        _update(QKDSessionStatus.ERROR_CORRECTION)
        await asyncio.sleep(0.03)

        qber = _simulate_qber(protocol, eavesdrop_prob, noise_model, rng)
        threshold = _QBER_THRESHOLDS[protocol]

        if qber > threshold:
            logger.warning(
                "QKD session %s aborted: QBER %.4f exceeds threshold %.4f.",
                session_id,
                qber,
                threshold,
            )
            _qkd_sessions[session_id]["status"] = QKDSessionStatus.ABORTED
            _qkd_sessions[session_id]["updated_at"] = datetime.utcnow()
            return

        # Stage 4: Privacy amplification
        _update(QKDSessionStatus.PRIVACY_AMPLIFICATION)
        await asyncio.sleep(0.02)

        pa_factor = _privacy_amplification_factor(qber)
        secure_key_length = int(sifted_bits * pa_factor)
        # Clamp to requested key length
        key_length_bits = min(secure_key_length, session["key_length_bits"])

        key_rate = _simulate_secure_key_rate(protocol, n_qubits, qber, sifted_bits)
        key_id = _derive_key_id(session_id, session["alice_id"], session["bob_id"])

        key_info = QKDKeyInfo(
            session_id=session_id,
            key_id=key_id,
            key_length_bits=key_length_bits,
            raw_bits_transmitted=n_qubits,
            sifted_bits=sifted_bits,
            qber=round(qber, 6),
            secure_key_rate_bps=round(key_rate, 2),
            privacy_amplification_factor=round(pa_factor, 6),
            generated_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(hours=24),
            is_secure=qber <= threshold,
        )

        _qkd_sessions[session_id]["key_info"] = key_info.model_dump()
        _update(QKDSessionStatus.COMPLETED)
        logger.info(
            "QKD session %s completed: key_id=%s, key_length=%d bits, QBER=%.4f.",
            session_id,
            key_id,
            key_length_bits,
            qber,
        )

    except asyncio.CancelledError:
        _qkd_sessions[session_id]["status"] = QKDSessionStatus.ABORTED
        _qkd_sessions[session_id]["updated_at"] = datetime.utcnow()
        logger.warning("QKD session %s was cancelled.", session_id)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/session",
    response_model=QKDSessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new QKD session",
    description=(
        "Initiates a QKD key exchange session between Alice and Bob using "
        "the specified protocol. The session runs asynchronously; poll the "
        "GET endpoint to check completion."
    ),
)
async def create_qkd_session(
    body: QKDSessionRequest,
    background_tasks: BackgroundTasks,
) -> QKDSessionResponse:
    """Create and immediately start a QKD session.

    Args:
        body: Session parameters including participants, protocol, and qubit count.
        background_tasks: FastAPI background task runner.

    Returns:
        QKDSessionResponse describing the newly created session.
    """
    session_id = str(uuid.uuid4())
    now = datetime.utcnow()

    _qkd_sessions[session_id] = {
        "session_id": session_id,
        "alice_id": body.alice_id,
        "bob_id": body.bob_id,
        "protocol": body.protocol,
        "status": QKDSessionStatus.INITIALISING,
        "key_length_bits": body.key_length_bits,
        "n_qubits": body.n_qubits,
        "noise_model": body.noise_model,
        "eavesdrop_probability": body.eavesdrop_probability,
        "created_at": now,
        "updated_at": now,
        "key_info": None,
    }

    background_tasks.add_task(_run_qkd_protocol, session_id)

    logger.info(
        "Created QKD session %s (protocol=%s, alice=%s, bob=%s, n_qubits=%d).",
        session_id,
        body.protocol,
        body.alice_id,
        body.bob_id,
        body.n_qubits,
    )

    return _build_session_response(session_id)


@router.get(
    "/session/{session_id}",
    response_model=QKDSessionResponse,
    summary="Get a QKD session",
    description="Returns the full state of a QKD session including key info once completed.",
)
async def get_qkd_session(session_id: str) -> QKDSessionResponse:
    """Retrieve a QKD session by ID.

    Args:
        session_id: UUID of the QKD session.

    Returns:
        QKDSessionResponse with current status and optional key info.

    Raises:
        HTTPException 404: If no session with the given ID exists.
    """
    _require_qkd_session(session_id)
    return _build_session_response(session_id)


@router.get(
    "/session/{session_id}/key",
    response_model=QKDKeyInfo,
    summary="Get generated key information",
    description=(
        "Returns metadata about the QKD-generated key. The raw key bytes are "
        "NOT returned here for security reasons; they are delivered directly "
        "to the authenticated parties via a secure side channel."
    ),
)
async def get_key_info(session_id: str) -> QKDKeyInfo:
    """Retrieve QKD key metadata for a completed session.

    Args:
        session_id: UUID of the completed QKD session.

    Returns:
        QKDKeyInfo with key length, QBER, key rate, and expiry.

    Raises:
        HTTPException 404: If the session does not exist.
        HTTPException 409: If the session has not yet completed or was aborted.
    """
    session = _require_qkd_session(session_id)

    if session["status"] == QKDSessionStatus.ABORTED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "QKD session was aborted (QBER exceeded security threshold "
                "or channel was compromised). Start a new session."
            ),
        )

    if session["status"] != QKDSessionStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"QKD session is not yet completed (current status: "
                f"'{session['status']}'). Retry after a moment."
            ),
        )

    key_data = session.get("key_info")
    if key_data is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Key info is missing despite session being completed. This is a bug.",
        )

    return QKDKeyInfo(**key_data)


@router.get(
    "/sessions",
    response_model=List[QKDSessionResponse],
    summary="List all QKD sessions",
    description="Returns a list of all QKD sessions, optionally filtered by status.",
)
async def list_qkd_sessions(
    status_filter: Optional[QKDSessionStatus] = None,
    limit: int = 20,
    offset: int = 0,
) -> List[QKDSessionResponse]:
    """List QKD sessions with optional status filtering.

    Args:
        status_filter: When provided, only sessions with this status are returned.
        limit: Maximum number of sessions to return.
        offset: Number of sessions to skip.

    Returns:
        List of QKDSessionResponse objects.
    """
    sessions = list(_qkd_sessions.values())
    if status_filter is not None:
        sessions = [s for s in sessions if s["status"] == status_filter]
    sessions = sessions[offset : offset + limit]
    return [_build_session_response(s["session_id"]) for s in sessions]


@router.delete(
    "/session/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a QKD session",
    description="Removes a QKD session from the server. Cannot be undone.",
)
async def delete_qkd_session(session_id: str) -> None:
    """Delete a QKD session.

    Args:
        session_id: UUID of the QKD session to delete.

    Raises:
        HTTPException 404: If the session does not exist.
    """
    _require_qkd_session(session_id)
    _qkd_sessions.pop(session_id, None)
    logger.info("Deleted QKD session %s.", session_id)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_qkd_session(session_id: str) -> Dict[str, Any]:
    """Return session dict or raise HTTP 404."""
    session = _qkd_sessions.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"QKD session '{session_id}' not found.",
        )
    return session


def _build_session_response(session_id: str) -> QKDSessionResponse:
    """Construct a QKDSessionResponse from the internal session dict."""
    s = _qkd_sessions[session_id]
    key_info = None
    if s.get("key_info") is not None:
        key_info = QKDKeyInfo(**s["key_info"])

    return QKDSessionResponse(
        session_id=s["session_id"],
        alice_id=s["alice_id"],
        bob_id=s["bob_id"],
        protocol=s["protocol"],
        status=s["status"],
        key_length_bits=s["key_length_bits"],
        n_qubits=s["n_qubits"],
        noise_model=s["noise_model"],
        eavesdrop_probability=s["eavesdrop_probability"],
        created_at=s["created_at"],
        updated_at=s["updated_at"],
        key_info=key_info,
    )
