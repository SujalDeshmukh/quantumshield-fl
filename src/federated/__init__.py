"""Federated Learning module with quantum-secured communication."""
from .client import FederatedClient
from .server import FederatedServer
from .aggregator import QuantumSecureAggregator, FedAvg, FedProx
from .privacy import DifferentialPrivacyMechanism, QuantumNoiseChannel

__all__ = [
    "FederatedClient", "FederatedServer",
    "QuantumSecureAggregator", "FedAvg", "FedProx",
    "DifferentialPrivacyMechanism", "QuantumNoiseChannel",
]
