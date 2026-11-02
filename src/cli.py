"""
QuantumShield-FL CLI
====================
Command-line interface for running experiments, starting the server,
and managing QKD sessions.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich import print as rprint

console = Console()

BANNER = """
[bold cyan]
  ██████╗ ███████╗██╗  ██╗ █████╗ ███╗   ██╗████████╗██╗   ██╗███╗   ███╗
 ██╔═══██╗██╔════╝██║  ██║██╔══██╗████╗  ██║╚══██╔══╝██║   ██║████╗ ████║
 ██║   ██║███████╗███████║███████║██╔██╗ ██║   ██║   ██║   ██║██╔████╔██║
 ██║▄▄ ██║╚════██║██╔══██║██╔══██║██║╚██╗██║   ██║   ██║   ██║██║╚██╔╝██║
 ╚██████╔╝███████║██║  ██║██║  ██║██║ ╚████║   ██║   ╚██████╔╝██║ ╚═╝ ██║
  ╚══▀▀═╝ ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝    ╚═════╝ ╚═╝     ╚═╝
[/bold cyan]
[bold yellow]  Shield-FL  [/bold yellow][dim]— Quantum-Secured Federated Learning Framework[/dim]
  [dim]v1.0.0 | MIT License | https://github.com/your-username/quantumshield-fl[/dim]
"""


@click.group()
@click.version_option("1.0.0", prog_name="qsfl")
def main():
    """⚛️  QuantumShield-FL: Quantum-Secured Federated Learning CLI."""
    pass


@main.command("info")
def info():
    """Show system information and configuration."""
    console.print(BANNER)
    table = Table(title="System Information", show_header=True)
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="dim")

    try:
        import numpy as np
        table.add_row("NumPy", "✅ Available", np.__version__)
    except ImportError:
        table.add_row("NumPy", "❌ Missing", "pip install numpy")

    try:
        import scipy
        table.add_row("SciPy", "✅ Available", scipy.__version__)
    except ImportError:
        table.add_row("SciPy", "❌ Missing", "pip install scipy")

    try:
        import sklearn
        table.add_row("Scikit-learn", "✅ Available", sklearn.__version__)
    except ImportError:
        table.add_row("Scikit-learn", "❌ Missing", "pip install scikit-learn")

    try:
        import fastapi
        table.add_row("FastAPI", "✅ Available", fastapi.__version__)
    except ImportError:
        table.add_row("FastAPI", "❌ Missing", "pip install fastapi")

    try:
        import pennylane as qml
        table.add_row("PennyLane", "✅ Available", qml.__version__)
    except ImportError:
        table.add_row("PennyLane", "⚠️  Optional", "pip install pennylane")

    try:
        import qiskit
        table.add_row("Qiskit", "✅ Available", qiskit.__version__)
    except ImportError:
        table.add_row("Qiskit", "⚠️  Optional", "pip install qiskit")

    console.print(table)


@main.command("qkd")
@click.option("--protocol", type=click.Choice(["bb84", "e91"]), default="bb84", show_default=True)
@click.option("--n-qubits", type=int, default=1024, show_default=True)
@click.option("--eavesdrop", is_flag=True, default=False, help="Simulate eavesdropper")
@click.option("--output", type=click.Path(), default=None)
def run_qkd(protocol: str, n_qubits: int, eavesdrop: bool, output: str | None):
    """Run a QKD key exchange simulation."""
    console.print(Panel(
        f"[cyan]Protocol:[/cyan] {protocol.upper()}\n"
        f"[cyan]Qubits:[/cyan] {n_qubits}\n"
        f"[cyan]Eavesdropper:[/cyan] {'[red]YES[/red]' if eavesdrop else '[green]NO[/green]'}",
        title="⚛️  QKD Simulation",
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Running QKD protocol...", total=None)

        if protocol == "bb84":
            from src.quantum.qkd import BB84Protocol
            p = BB84Protocol(n_qubits=n_qubits)
        else:
            from src.quantum.qkd import E91Protocol
            p = E91Protocol(n_pairs=n_qubits)

        result = p.run_full_protocol(eavesdrop=eavesdrop)
        progress.update(task, completed=True)

    # Display results
    table = Table(title="QKD Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold")

    table.add_row("Key Length", f"{result['key_length']} bits")
    table.add_row("QBER", f"{result['qber']*100:.3f}%")
    table.add_row("Key Rate", f"{result['key_rate']:.4f} bits/qubit")
    table.add_row("Secure", "✅ YES" if result["is_secure"] else "❌ NO — Eve detected!")

    if protocol == "e91" and "chsh_value" in result:
        table.add_row("CHSH |S|", f"{abs(result['chsh_value']):.4f}")

    console.print(table)

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            # Don't serialize the key itself
            safe_result = {k: v for k, v in result.items() if k != "secret_key"}
            json.dump(safe_result, f, indent=2)
        console.print(f"[green]Results saved to {output}[/green]")


@main.command("train")
@click.option("--dataset", type=click.Choice(["iris", "breast_cancer", "wine", "digits"]), default="iris")
@click.option("--n-clients", type=int, default=3, show_default=True)
@click.option("--n-rounds", type=int, default=20, show_default=True)
@click.option("--n-qubits", type=int, default=2, show_default=True)
@click.option("--n-layers", type=int, default=3, show_default=True)
@click.option("--protocol", type=click.Choice(["bb84", "e91", "none"]), default="bb84")
@click.option("--aggregation", type=click.Choice(["fedavg", "fedprox"]), default="fedavg")
@click.option("--epsilon", type=float, default=1.0, show_default=True, help="DP epsilon")
@click.option("--no-dp", is_flag=True, default=False, help="Disable differential privacy")
@click.option("--no-qkd", is_flag=True, default=False, help="Disable QKD encryption")
@click.option("--output", type=click.Path(), default="experiments/run")
@click.option("--seed", type=int, default=42)
def train(
    dataset, n_clients, n_rounds, n_qubits, n_layers,
    protocol, aggregation, epsilon, no_dp, no_qkd, output, seed
):
    """Run a full federated quantum machine learning experiment."""
    console.print(BANNER)
    console.print(Panel(
        f"[cyan]Dataset:[/cyan] {dataset}\n"
        f"[cyan]Clients:[/cyan] {n_clients}\n"
        f"[cyan]Rounds:[/cyan] {n_rounds}\n"
        f"[cyan]VQC:[/cyan] {n_qubits} qubits, {n_layers} layers\n"
        f"[cyan]QKD:[/cyan] {protocol.upper() if not no_qkd else 'Disabled'}\n"
        f"[cyan]Privacy (ε):[/cyan] {epsilon if not no_dp else 'Disabled'}",
        title="🚀 Starting QuantumShield-FL Training",
    ))

    import numpy as np
    np.random.seed(seed)

    # Import and run experiment
    try:
        from scripts.run_experiment import run_experiment
        results = run_experiment(
            dataset=dataset,
            n_clients=n_clients,
            n_rounds=n_rounds,
            n_qubits=n_qubits,
            n_layers=n_layers,
            protocol=None if no_qkd else protocol,
            aggregation=aggregation,
            epsilon=epsilon,
            use_dp=not no_dp,
            output_dir=output,
            seed=seed,
        )
        console.print(f"[green]✅ Training complete! Final accuracy: {results.get('final_accuracy', 'N/A'):.4f}[/green]")
    except Exception as e:
        console.print(f"[red]Error during training: {e}[/red]")
        raise


@main.command("serve")
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", type=int, default=8000, show_default=True)
@click.option("--reload", is_flag=True, default=False)
@click.option("--workers", type=int, default=1)
def serve(host: str, port: int, reload: bool, workers: int):
    """Start the QuantumShield-FL API server."""
    import uvicorn
    console.print(BANNER)
    console.print(f"[green]🚀 Starting API server at http://{host}:{port}[/green]")
    console.print(f"[dim]📖 API docs: http://{host}:{port}/docs[/dim]")
    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,
        log_level="info",
    )


if __name__ == "__main__":
    main()
