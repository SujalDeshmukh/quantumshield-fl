# QuantumShield-FL Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.0.0] - 2024-01-01

### Added
- **BB84 Protocol**: Full simulation including QBER estimation, sifting, information reconciliation, and privacy amplification
- **E91 Protocol**: Entanglement-based QKD with CHSH inequality testing  
- **QKDSession**: AES-256 encryption using quantum-generated keys
- **VQC (Variational Quantum Classifier)**: Hardware-efficient ansatz with parameter shift rule gradients
- **QNN (Quantum Neural Network)**: Hybrid quantum-classical architecture
- **QSVM (Quantum SVM)**: Quantum kernel methods with ZZ and Pauli feature maps
- **QuantumDataEncoder**: Amplitude, angle, IQP, and basis encoding schemes
- **FederatedServer**: Multi-client coordination with FedAvg and FedProx
- **FederatedClient**: Local training with QKD-encrypted updates
- **DifferentialPrivacyMechanism**: Gaussian and Laplace mechanisms with RDP accounting
- **FastAPI REST API**: Full endpoint set for federated sessions, QKD, and inference
- **CLI**: `qsfl` command with `info`, `qkd`, `train`, `serve` subcommands
- **Docker**: Multi-stage Dockerfile + docker-compose with Prometheus/Grafana
- **CI/CD**: GitHub Actions pipeline (lint, test matrix, security audit, Docker, PyPI)
- **Notebooks**: Interactive tutorials for QKD, VQC, and federated learning
- **Benchmarks**: Comparison vs classical federated learning baselines

### Security
- All model updates encrypted with information-theoretically secure QKD keys
- QBER monitoring for real-time eavesdropping detection
- Differential privacy for gradient-level privacy protection

## [Unreleased]

### Planned
- PennyLane hardware backend integration
- Real QKD hardware adapter (ID Quantique Clavis3)
- Asynchronous federated learning
- Quantum Byzantine fault tolerance
- NIST PQC hybrid mode
