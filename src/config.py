"""
QuantumShield-FL Configuration & Settings
==========================================
Centralized configuration using pydantic-settings.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class QKDSettings(BaseSettings):
    """QKD protocol settings."""

    protocol: Literal["bb84", "e91"] = "bb84"
    n_qubits: int = Field(default=2048, ge=64, le=100_000)
    error_threshold: float = Field(default=0.11, ge=0.0, le=0.5)
    privacy_amplification: bool = True
    error_correction: bool = True
    key_refresh_interval: int = Field(default=100, ge=1)  # rounds

    class Config:
        env_prefix = "QSFL_QKD_"


class VQCSettings(BaseSettings):
    """VQC model settings."""

    n_qubits: int = Field(default=4, ge=1, le=20)
    n_layers: int = Field(default=3, ge=1, le=20)
    ansatz: Literal["strongly_entangling", "hardware_efficient", "twolocal"] = "strongly_entangling"
    measurement_observable: Literal["Z", "X", "Y"] = "Z"

    class Config:
        env_prefix = "QSFL_VQC_"


class FederatedSettings(BaseSettings):
    """Federated learning settings."""

    n_clients: int = Field(default=5, ge=2, le=100)
    n_rounds: int = Field(default=50, ge=1, le=10_000)
    fraction_clients: float = Field(default=1.0, ge=0.1, le=1.0)
    aggregation: Literal["fedavg", "fedprox", "quantum_secure"] = "fedavg"
    fedprox_mu: float = Field(default=0.01, ge=0.0)
    local_epochs: int = Field(default=5, ge=1)
    local_learning_rate: float = Field(default=0.01, gt=0.0)
    local_batch_size: int = Field(default=32, ge=1)
    use_qkd: bool = True
    use_differential_privacy: bool = True

    class Config:
        env_prefix = "QSFL_FL_"


class PrivacySettings(BaseSettings):
    """Differential privacy settings."""

    epsilon: float = Field(default=1.0, gt=0.0)
    delta: float = Field(default=1e-5, gt=0.0, lt=1.0)
    sensitivity: float = Field(default=1.0, gt=0.0)
    mechanism: Literal["gaussian", "laplace"] = "gaussian"
    max_grad_norm: float = Field(default=1.0, gt=0.0)

    class Config:
        env_prefix = "QSFL_DP_"


class APISettings(BaseSettings):
    """API server settings."""

    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    debug: bool = False
    reload: bool = False
    workers: int = Field(default=1, ge=1)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    secret_key: str = Field(default="change-me-in-production")
    allowed_origins: list[str] = ["*"]
    api_prefix: str = "/api/v1"

    class Config:
        env_prefix = "QSFL_"


class Settings(BaseSettings):
    """Master settings object."""

    # Sub-settings
    qkd: QKDSettings = Field(default_factory=QKDSettings)
    vqc: VQCSettings = Field(default_factory=VQCSettings)
    federated: FederatedSettings = Field(default_factory=FederatedSettings)
    privacy: PrivacySettings = Field(default_factory=PrivacySettings)
    api: APISettings = Field(default_factory=APISettings)

    # Paths
    project_root: Path = Path(__file__).parent.parent.parent
    experiments_dir: Path = Field(default=None)
    logs_dir: Path = Field(default=None)
    data_dir: Path = Field(default=None)

    def __init__(self, **data):
        super().__init__(**data)
        root = self.project_root
        if self.experiments_dir is None:
            object.__setattr__(self, "experiments_dir", root / "experiments")
        if self.logs_dir is None:
            object.__setattr__(self, "logs_dir", root / "logs")
        if self.data_dir is None:
            object.__setattr__(self, "data_dir", root / "data")

    def setup_directories(self) -> None:
        """Create all required directories."""
        for d in [self.experiments_dir, self.logs_dir, self.data_dir]:
            d.mkdir(parents=True, exist_ok=True)

    class Config:
        env_prefix = "QSFL_"
        env_file = ".env"
        env_nested_delimiter = "__"


# Global settings instance
settings = Settings()
