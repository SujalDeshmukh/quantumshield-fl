"""
QuantumShield-FL Experiment Runner
====================================
Runs full end-to-end federated quantum ML experiments.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

# ─── Configure logging ────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("quantumshield-fl")


def load_dataset(name: str):
    """Load a dataset from sklearn and return (X, y, feature_names, n_classes)."""
    from sklearn import datasets
    from sklearn.preprocessing import StandardScaler

    if name == "iris":
        data = datasets.load_iris()
        X, y = data.data[:, :2], data.target  # 2 features, 3 classes
        y = (y < 2).astype(int)  # Binary: setosa vs versicolor
        X = X[data.target < 2]
    elif name == "breast_cancer":
        data = datasets.load_breast_cancer()
        X, y = data.data[:, :4], data.target  # 4 features
    elif name == "wine":
        data = datasets.load_wine()
        X, y = data.data[:, :4], data.target
        mask = y < 2
        X, y = X[mask], y[mask]  # Binary
    elif name == "digits":
        data = datasets.load_digits()
        mask = data.target < 2
        X, y = data.data[mask], data.target[mask]
        # Reduce to 4 features via PCA for quantum encoding
        from sklearn.decomposition import PCA
        pca = PCA(n_components=4)
        X = pca.fit_transform(X)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = np.clip(X, -np.pi, np.pi)  # Scale to VQC input range

    n_classes = len(np.unique(y))
    logger.info(f"Loaded {name}: {X.shape[0]} samples, {X.shape[1]} features, {n_classes} classes")
    return X, y, n_classes


def split_data_iid(X, y, n_clients: int, seed: int = 42):
    """Split data i.i.d. among clients."""
    np.random.seed(seed)
    indices = np.random.permutation(len(X))
    splits = np.array_split(indices, n_clients)
    return [(X[idx], y[idx]) for idx in splits]


def split_data_non_iid(X, y, n_clients: int, alpha: float = 0.5, seed: int = 42):
    """Split data non-i.i.d. using Dirichlet distribution (label skew)."""
    np.random.seed(seed)
    n_classes = len(np.unique(y))
    client_data = [[] for _ in range(n_clients)]

    for c in range(n_classes):
        idx = np.where(y == c)[0]
        proportions = np.random.dirichlet(alpha * np.ones(n_clients))
        proportions = (proportions * len(idx)).astype(int)
        proportions[-1] = len(idx) - proportions[:-1].sum()
        start = 0
        for client_id, prop in enumerate(proportions):
            client_data[client_id].extend(idx[start:start + prop])
            start += prop

    result = []
    for indices in client_data:
        if len(indices) == 0:
            indices = [0]  # At least one sample
        result.append((X[indices], y[indices]))
    return result


def run_experiment(
    dataset: str = "iris",
    n_clients: int = 3,
    n_rounds: int = 20,
    n_qubits: int = 2,
    n_layers: int = 3,
    protocol: str | None = "bb84",
    aggregation: str = "fedavg",
    epsilon: float = 1.0,
    use_dp: bool = True,
    output_dir: str = "experiments/run",
    seed: int = 42,
    no_plots: bool = False,
    non_iid: bool = False,
    n_qkd_bits: int = 1024,
):
    """
    Run a complete QuantumShield-FL experiment.

    Args:
        dataset: Dataset name (iris, breast_cancer, wine, digits)
        n_clients: Number of federated clients
        n_rounds: Number of training rounds
        n_qubits: Number of qubits in VQC
        n_layers: Number of VQC layers
        protocol: QKD protocol (bb84, e91, or None for no QKD)
        aggregation: Aggregation method (fedavg, fedprox)
        epsilon: DP privacy budget
        use_dp: Whether to use differential privacy
        output_dir: Directory for saving results
        seed: Random seed
        no_plots: Skip generating plots
        non_iid: Use non-i.i.d. data split
        n_qkd_bits: Number of QKD qubits for key generation
    """
    np.random.seed(seed)
    start_time = time.time()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("QuantumShield-FL Experiment")
    logger.info("=" * 60)
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Clients: {n_clients}, Rounds: {n_rounds}")
    logger.info(f"VQC: {n_qubits} qubits, {n_layers} layers")
    logger.info(f"QKD: {protocol or 'Disabled'}")
    logger.info(f"DP: {'ε=' + str(epsilon) if use_dp else 'Disabled'}")

    # ── Load dataset ──────────────────────────────────────────
    X, y, n_classes = load_dataset(dataset)
    n_features = X.shape[1]

    # ── Data partitioning ─────────────────────────────────────
    if non_iid:
        client_splits = split_data_non_iid(X, y, n_clients, seed=seed)
        logger.info("Data split: Non-IID (Dirichlet)")
    else:
        client_splits = split_data_iid(X, y, n_clients, seed=seed)
        logger.info("Data split: IID")

    # ── QKD Key Establishment ─────────────────────────────────
    qkd_results = {}
    if protocol:
        logger.info(f"\n🔐 Establishing QKD keys ({protocol.upper()})...")
        if protocol == "bb84":
            from src.quantum.qkd import BB84Protocol
            qkd_proto = BB84Protocol(n_qubits=n_qkd_bits)
        elif protocol == "e91":
            from src.quantum.qkd import E91Protocol
            qkd_proto = E91Protocol(n_pairs=n_qkd_bits)

        qkd_result = qkd_proto.run_full_protocol(eavesdrop=False)
        qkd_results = {
            "protocol": protocol,
            "key_length": qkd_result["key_length"],
            "qber": qkd_result["qber"],
            "key_rate": qkd_result["key_rate"],
            "is_secure": qkd_result["is_secure"],
        }
        if not qkd_result["is_secure"]:
            logger.warning("⚠️  QKD security check failed! QBER too high.")
        logger.info(f"   Key length: {qkd_result['key_length']} bits")
        logger.info(f"   QBER: {qkd_result['qber']*100:.2f}%")
        logger.info(f"   Secure: {'✅' if qkd_result['is_secure'] else '❌'}")

    # ── Create clients ─────────────────────────────────────────
    logger.info("\n👥 Initializing federated clients...")
    from src.quantum.qml import VQC
    from src.federated import FederatedClient, FederatedServer

    clients = []
    for i, (X_local, y_local) in enumerate(client_splits):
        model = VQC(n_qubits=min(n_qubits, n_features), n_layers=n_layers)
        client = FederatedClient(
            client_id=f"client_{i}",
            model=model,
            X_train=X_local,
            y_train=y_local,
            use_qkd=(protocol is not None),
            use_dp=use_dp,
            dp_epsilon=epsilon,
        )
        clients.append(client)
        logger.info(f"   Client {i}: {len(X_local)} samples")

    # ── Create server ──────────────────────────────────────────
    global_model = VQC(n_qubits=min(n_qubits, n_features), n_layers=n_layers)
    server = FederatedServer(
        clients=clients,
        global_model=global_model,
        aggregation=aggregation,
    )

    # ── Training loop ──────────────────────────────────────────
    logger.info(f"\n🚀 Starting federated training ({n_rounds} rounds)...")
    history = {
        "round": [],
        "train_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "communication_cost": [],
    }

    # Validation split from full data
    val_size = int(0.2 * len(X))
    X_val, y_val = X[-val_size:], y[-val_size:]

    for r in range(1, n_rounds + 1):
        round_metrics = server.run_round(round_num=r)

        # Evaluate on full validation set
        preds = server.global_model.predict(X_val)
        val_acc = float(np.mean(preds == y_val))

        history["round"].append(r)
        history["train_loss"].append(round_metrics.get("loss", float("nan")))
        history["train_accuracy"].append(round_metrics.get("accuracy", float("nan")))
        history["val_accuracy"].append(val_acc)
        history["communication_cost"].append(
            server.global_model.params.nbytes * n_clients
        )

        if r % max(1, n_rounds // 10) == 0:
            logger.info(
                f"   Round {r:3d}/{n_rounds} | "
                f"Loss: {round_metrics.get('loss', float('nan')):.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

    total_time = time.time() - start_time
    final_accuracy = history["val_accuracy"][-1]

    logger.info(f"\n✅ Training complete!")
    logger.info(f"   Final val accuracy: {final_accuracy:.4f}")
    logger.info(f"   Total time: {total_time:.1f}s")

    # ── Save results ───────────────────────────────────────────
    results = {
        "experiment": {
            "dataset": dataset,
            "n_clients": n_clients,
            "n_rounds": n_rounds,
            "n_qubits": n_qubits,
            "n_layers": n_layers,
            "protocol": protocol,
            "aggregation": aggregation,
            "epsilon": epsilon if use_dp else None,
            "use_dp": use_dp,
            "seed": seed,
            "timestamp": datetime.now().isoformat(),
            "total_time_s": total_time,
        },
        "qkd": qkd_results,
        "history": history,
        "final_accuracy": final_accuracy,
        "final_val_accuracy": final_accuracy,
    }

    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"   Results saved to {results_file}")

    # ── Generate plots ─────────────────────────────────────────
    if not no_plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle(
                f"QuantumShield-FL | {dataset.title()} | {n_clients} clients | {protocol or 'No'} QKD",
                fontsize=13, fontweight="bold"
            )

            rounds = history["round"]

            # Accuracy
            axes[0].plot(rounds, history["train_accuracy"], label="Train", color="#00ff88", linewidth=2)
            axes[0].plot(rounds, history["val_accuracy"], label="Val", color="#4488ff", linewidth=2, linestyle="--")
            axes[0].set_xlabel("Round")
            axes[0].set_ylabel("Accuracy")
            axes[0].set_title("Model Accuracy")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Loss
            axes[1].plot(rounds, history["train_loss"], color="#ff8844", linewidth=2)
            axes[1].set_xlabel("Round")
            axes[1].set_ylabel("Loss")
            axes[1].set_title("Training Loss")
            axes[1].grid(True, alpha=0.3)

            # Communication cost
            comm_mb = [c / 1e6 for c in history["communication_cost"]]
            axes[2].fill_between(rounds, comm_mb, alpha=0.3, color="#aa44ff")
            axes[2].plot(rounds, comm_mb, color="#aa44ff", linewidth=2)
            axes[2].set_xlabel("Round")
            axes[2].set_ylabel("Communication Cost (MB)")
            axes[2].set_title("Communication Overhead")
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            plot_file = output_path / "training_curves.png"
            plt.savefig(plot_file, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"   Plots saved to {plot_file}")
        except Exception as e:
            logger.warning(f"Could not generate plots: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="QuantumShield-FL Experiment Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", choices=["iris", "breast_cancer", "wine", "digits"], default="iris")
    parser.add_argument("--n-clients", type=int, default=3)
    parser.add_argument("--n-rounds", type=int, default=20)
    parser.add_argument("--n-qubits", type=int, default=2)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--protocol", choices=["bb84", "e91", "none"], default="bb84")
    parser.add_argument("--aggregation", choices=["fedavg", "fedprox"], default="fedavg")
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--n-qkd-bits", type=int, default=1024)
    parser.add_argument("--no-dp", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--non-iid", action="store_true")
    parser.add_argument("--output", default="experiments/run")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    run_experiment(
        dataset=args.dataset,
        n_clients=args.n_clients,
        n_rounds=args.n_rounds,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        protocol=None if args.protocol == "none" else args.protocol,
        aggregation=args.aggregation,
        epsilon=args.epsilon,
        use_dp=not args.no_dp,
        output_dir=args.output,
        seed=args.seed,
        no_plots=args.no_plots,
        non_iid=args.non_iid,
        n_qkd_bits=args.n_qkd_bits,
    )


if __name__ == "__main__":
    main()
