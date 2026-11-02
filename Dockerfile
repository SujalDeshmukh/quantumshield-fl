# ─────────────────────────────────────────────
# QuantumShield-FL Dockerfile
# Multi-stage build for minimal production image
# ─────────────────────────────────────────────

# Stage 1: Build dependencies
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ ./src/

# Build wheel
RUN pip install --upgrade pip build \
    && python -m build --wheel --outdir /build/dist

# Stage 2: Production image
FROM python:3.11-slim AS production

LABEL org.opencontainers.image.title="QuantumShield-FL"
LABEL org.opencontainers.image.description="Quantum-Secured Federated Learning with VQC and QKD"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.source="https://github.com/your-username/quantumshield-fl"

# Create non-root user
RUN groupadd -r qsfl && useradd -r -g qsfl -d /app -s /sbin/nologin qsfl

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy built wheel from builder stage
COPY --from=builder /build/dist/*.whl /tmp/

# Install the package
RUN pip install --no-cache-dir /tmp/*.whl \
    && rm /tmp/*.whl

# Copy additional scripts and configs
COPY scripts/ ./scripts/
COPY experiments/ ./experiments/

# Set ownership
RUN chown -R qsfl:qsfl /app

USER qsfl

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

# Expose API port
EXPOSE 8000

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV QSFL_ENV=production
ENV QSFL_HOST=0.0.0.0
ENV QSFL_PORT=8000
ENV QSFL_LOG_LEVEL=INFO

# Default command: run API server
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
