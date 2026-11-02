"""Training visualisation utilities for QuantumShield-FL.

Provides Matplotlib-based plotting functions for analysing and comparing
the results of federated learning experiments:

* Federated learning convergence curves (loss and accuracy over rounds).
* Quantum vs. classical model performance comparison.
* QKD key generation rate over time.
* Quantum Bit Error Rate (QBER) evolution.
* Differential privacy budget consumption.

All functions return ``matplotlib.figure.Figure`` objects so that callers
can save, display, or embed them as needed.

Typical usage
-------------
>>> from src.visualization.training_viz import TrainingVisualizer
>>> viz = TrainingVisualizer(theme="dark")
>>> fig = viz.plot_convergence(rounds, losses, accuracies)
>>> fig.savefig("convergence.png", dpi=150, bbox_inches="tight")
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Theme definitions
# ---------------------------------------------------------------------------

_THEMES: Dict[str, Dict] = {
    "light": {
        "bg": "#ffffff",
        "fg": "#333333",
        "grid": "#dddddd",
        "accent": ["#4a90d9", "#e74c3c", "#27ae60", "#e67e22", "#8e44ad", "#1abc9c"],
        "text_color": "#333333",
        "spine_color": "#aaaaaa",
    },
    "dark": {
        "bg": "#1e1e2e",
        "fg": "#cdd6f4",
        "grid": "#313244",
        "accent": ["#89b4fa", "#f38ba8", "#a6e3a1", "#fab387", "#cba6f7", "#94e2d5"],
        "text_color": "#cdd6f4",
        "spine_color": "#585b70",
    },
    "quantum": {
        "bg": "#0a0a1a",
        "fg": "#00ffff",
        "grid": "#1a1a3a",
        "accent": ["#00ffff", "#ff00ff", "#00ff88", "#ffff00", "#ff8800", "#8888ff"],
        "text_color": "#00ffff",
        "spine_color": "#334455",
    },
}

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _smooth(values: Sequence[float], window: int = 3) -> List[float]:
    """Apply a simple moving-average smoothing to a sequence.

    Args:
        values: Input sequence of floats.
        window: Smoothing window size.

    Returns:
        Smoothed list of the same length.
    """
    arr = np.array(values, dtype=float)
    result = np.convolve(arr, np.ones(window) / window, mode="same")
    # Fix boundary effects by blending with original near edges
    half = window // 2
    result[:half] = arr[:half]
    result[-half:] = arr[-half:]
    return result.tolist()


def _style_axes(ax, theme: Dict, xlabel: str, ylabel: str, title: str) -> None:
    """Apply consistent styling to a Matplotlib Axes object.

    Args:
        ax: Matplotlib Axes to style.
        theme: Theme dictionary from ``_THEMES``.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        title: Plot title.
    """
    ax.set_facecolor(theme["bg"])
    ax.set_xlabel(xlabel, color=theme["text_color"], fontsize=10)
    ax.set_ylabel(ylabel, color=theme["text_color"], fontsize=10)
    ax.set_title(title, color=theme["text_color"], fontsize=11, pad=10)
    ax.tick_params(colors=theme["text_color"])
    ax.grid(True, color=theme["grid"], linewidth=0.6, linestyle="--", alpha=0.7)
    for spine in ax.spines.values():
        spine.set_edgecolor(theme["spine_color"])


# ---------------------------------------------------------------------------
# Main visualiser class
# ---------------------------------------------------------------------------


class TrainingVisualizer:
    """Collection of plotting helpers for QuantumShield-FL training analysis.

    Args:
        theme: One of ``'light'``, ``'dark'``, or ``'quantum'``.
        figsize_default: Default figure size ``(width, height)`` in inches.
        dpi: Default DPI for all figures.
    """

    def __init__(
        self,
        theme: str = "light",
        figsize_default: Tuple[float, float] = (9, 5),
        dpi: int = 100,
    ) -> None:
        if theme not in _THEMES:
            raise ValueError(f"Unknown theme '{theme}'. Choose from: {list(_THEMES)}.")
        self.theme = _THEMES[theme]
        self.figsize_default = figsize_default
        self.dpi = dpi

    # ------------------------------------------------------------------ #
    # 1. Convergence curves
    # ------------------------------------------------------------------ #

    def plot_convergence(
        self,
        rounds: Sequence[int],
        losses: Sequence[float],
        accuracies: Sequence[float],
        *,
        smooth_window: int = 1,
        client_losses: Optional[List[Sequence[float]]] = None,
        title: str = "Federated Learning Convergence",
        figsize: Optional[Tuple[float, float]] = None,
    ):
        """Plot loss and accuracy convergence curves over federated rounds.

        Args:
            rounds: Round indices (x-axis).
            losses: Global loss values per round.
            accuracies: Global accuracy values per round.
            smooth_window: Moving-average window for smoothing (1 = no smoothing).
            client_losses: Optional list of per-client loss sequences to plot
                           as faint background traces.
            title: Figure title.
            figsize: Override figure size.

        Returns:
            ``matplotlib.figure.Figure`` with two subplots (loss, accuracy).
        """
        import matplotlib.pyplot as plt

        fig, (ax_loss, ax_acc) = plt.subplots(
            1, 2, figsize=figsize or self.figsize_default, dpi=self.dpi
        )
        fig.patch.set_facecolor(self.theme["bg"])
        colors = self.theme["accent"]

        r = list(rounds)
        loss_smooth = _smooth(losses, smooth_window)
        acc_smooth = _smooth(accuracies, smooth_window)

        # --- Loss subplot ---
        _style_axes(ax_loss, self.theme, "Round", "Loss", "Global Loss")

        if client_losses:
            for cl in client_losses:
                ax_loss.plot(
                    r, list(cl), color=colors[0], alpha=0.12, linewidth=0.8
                )

        ax_loss.plot(r, losses, color=colors[0], alpha=0.3, linewidth=0.8, label="_raw")
        ax_loss.plot(r, loss_smooth, color=colors[0], linewidth=2.2, label="Global loss")

        # Shade area under the curve
        ax_loss.fill_between(r, loss_smooth, alpha=0.1, color=colors[0])

        # Annotate final value
        ax_loss.annotate(
            f"Final: {losses[-1]:.4f}",
            xy=(r[-1], losses[-1]),
            xytext=(-40, 10),
            textcoords="offset points",
            color=colors[0],
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color=colors[0], lw=1),
        )
        ax_loss.legend(fontsize=8, facecolor=self.theme["bg"], labelcolor=self.theme["text_color"])

        # --- Accuracy subplot ---
        _style_axes(ax_acc, self.theme, "Round", "Accuracy", "Global Accuracy")
        ax_acc.plot(r, accuracies, color=colors[1], alpha=0.3, linewidth=0.8, label="_raw")
        ax_acc.plot(r, acc_smooth, color=colors[1], linewidth=2.2, label="Global accuracy")
        ax_acc.fill_between(r, acc_smooth, alpha=0.1, color=colors[1])
        ax_acc.set_ylim(0, 1.05)
        ax_acc.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:.0%}")
        )

        # Mark best accuracy
        best_round_idx = int(np.argmax(list(accuracies)))
        best_acc = accuracies[best_round_idx]
        ax_acc.axhline(y=best_acc, color=colors[2], linestyle=":", linewidth=1.2, alpha=0.7)
        ax_acc.annotate(
            f"Best: {best_acc:.2%} @ r{rounds[best_round_idx]}",
            xy=(r[best_round_idx], best_acc),
            xytext=(10, -15),
            textcoords="offset points",
            color=colors[2],
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color=colors[2], lw=1),
        )
        ax_acc.legend(fontsize=8, facecolor=self.theme["bg"], labelcolor=self.theme["text_color"])

        fig.suptitle(title, color=self.theme["text_color"], fontsize=13, y=1.01)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------ #
    # 2. Quantum vs. classical comparison
    # ------------------------------------------------------------------ #

    def plot_quantum_vs_classical(
        self,
        rounds: Sequence[int],
        quantum_accuracies: Sequence[float],
        classical_accuracies: Sequence[float],
        quantum_losses: Optional[Sequence[float]] = None,
        classical_losses: Optional[Sequence[float]] = None,
        quantum_label: str = "VQC (Quantum)",
        classical_label: str = "MLP (Classical)",
        title: str = "Quantum vs. Classical Model Performance",
        figsize: Optional[Tuple[float, float]] = None,
    ):
        """Compare quantum (VQC) and classical model performance side by side.

        Args:
            rounds: Round indices.
            quantum_accuracies: Accuracy sequence for the quantum model.
            classical_accuracies: Accuracy sequence for the classical model.
            quantum_losses: Optional loss sequence for the quantum model.
            classical_losses: Optional loss sequence for the classical model.
            quantum_label: Legend label for the quantum model.
            classical_label: Legend label for the classical model.
            title: Figure title.
            figsize: Override figure size.

        Returns:
            ``matplotlib.figure.Figure``.
        """
        import matplotlib.pyplot as plt

        n_plots = 2 if (quantum_losses and classical_losses) else 1
        fig, axes = plt.subplots(
            1, n_plots, figsize=figsize or self.figsize_default, dpi=self.dpi
        )
        if n_plots == 1:
            axes = [axes]
        fig.patch.set_facecolor(self.theme["bg"])
        colors = self.theme["accent"]
        r = list(rounds)

        # Accuracy comparison
        ax = axes[0]
        _style_axes(ax, self.theme, "Round", "Accuracy", "Accuracy Comparison")
        ax.plot(r, list(quantum_accuracies), color=colors[0], linewidth=2.2,
                label=quantum_label, marker="o", markersize=3)
        ax.plot(r, list(classical_accuracies), color=colors[1], linewidth=2.2,
                label=classical_label, marker="s", markersize=3, linestyle="--")
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.legend(fontsize=8, facecolor=self.theme["bg"], labelcolor=self.theme["text_color"])

        # Advantage annotation
        final_diff = list(quantum_accuracies)[-1] - list(classical_accuracies)[-1]
        advantage_text = (
            f"Quantum advantage: +{final_diff:.2%}"
            if final_diff >= 0
            else f"Classical advantage: {-final_diff:.2%}"
        )
        ax.text(
            0.02, 0.05, advantage_text,
            transform=ax.transAxes,
            fontsize=8,
            color=colors[2],
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=self.theme["bg"],
                edgecolor=colors[2],
                alpha=0.8,
            ),
        )

        # Loss comparison (optional)
        if n_plots == 2 and quantum_losses and classical_losses:
            ax2 = axes[1]
            _style_axes(ax2, self.theme, "Round", "Loss", "Loss Comparison")
            ax2.plot(r, list(quantum_losses), color=colors[0], linewidth=2.2,
                     label=quantum_label, marker="o", markersize=3)
            ax2.plot(r, list(classical_losses), color=colors[1], linewidth=2.2,
                     label=classical_label, marker="s", markersize=3, linestyle="--")
            ax2.legend(fontsize=8, facecolor=self.theme["bg"], labelcolor=self.theme["text_color"])

        fig.suptitle(title, color=self.theme["text_color"], fontsize=13, y=1.01)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------ #
    # 3. QKD key generation rate
    # ------------------------------------------------------------------ #

    def plot_key_generation_rate(
        self,
        rounds: Sequence[int],
        key_rates_bps: Sequence[float],
        target_rate_bps: Optional[float] = None,
        protocol_name: str = "BB84",
        title: str = "QKD Key Generation Rate",
        figsize: Optional[Tuple[float, float]] = None,
    ):
        """Plot the QKD secure key generation rate over training rounds.

        Args:
            rounds: Round indices.
            key_rates_bps: Secure key rate (bits/second) per round.
            target_rate_bps: Optional target rate to draw as a horizontal line.
            protocol_name: Protocol name for the legend.
            title: Figure title.
            figsize: Override figure size.

        Returns:
            ``matplotlib.figure.Figure``.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(
            figsize=figsize or self.figsize_default, dpi=self.dpi
        )
        fig.patch.set_facecolor(self.theme["bg"])
        colors = self.theme["accent"]
        r = list(rounds)
        rates = list(key_rates_bps)

        _style_axes(ax, self.theme, "Round", "Key Rate (bits/s)", title)

        ax.plot(r, rates, color=colors[3], linewidth=2.0,
                label=f"{protocol_name} key rate", marker="D", markersize=4)
        ax.fill_between(r, rates, alpha=0.12, color=colors[3])

        if target_rate_bps is not None:
            ax.axhline(
                y=target_rate_bps,
                color=colors[2],
                linestyle="--",
                linewidth=1.5,
                label=f"Target: {target_rate_bps:.0f} bps",
            )

        # Rolling mean
        if len(rates) >= 5:
            rolling = _smooth(rates, window=5)
            ax.plot(r, rolling, color=colors[0], linewidth=1.5, linestyle=":",
                    label="5-round rolling mean")

        # Statistics annotation
        mean_rate = float(np.mean(rates))
        std_rate = float(np.std(rates))
        stats_text = f"μ={mean_rate:.1f} bps\nσ={std_rate:.1f} bps"
        ax.text(
            0.97, 0.97, stats_text,
            transform=ax.transAxes,
            ha="right", va="top", fontsize=8,
            color=self.theme["text_color"],
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor=self.theme["bg"],
                edgecolor=self.theme["spine_color"],
                alpha=0.85,
            ),
        )

        ax.legend(fontsize=8, facecolor=self.theme["bg"], labelcolor=self.theme["text_color"])
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------ #
    # 4. QBER over time
    # ------------------------------------------------------------------ #

    def plot_qber(
        self,
        rounds: Sequence[int],
        qbers: Sequence[float],
        security_threshold: float = 0.11,
        protocol_name: str = "BB84",
        eavesdrop_detected_rounds: Optional[List[int]] = None,
        title: str = "Quantum Bit Error Rate (QBER) Over Time",
        figsize: Optional[Tuple[float, float]] = None,
    ):
        """Plot the QBER evolution and highlight security threshold crossings.

        Args:
            rounds: Round indices.
            qbers: QBER values (0–1) per round.
            security_threshold: The maximum acceptable QBER (default 11% for BB84).
            protocol_name: Protocol name for annotation.
            eavesdrop_detected_rounds: Optional list of round numbers where
                eavesdropping was flagged (marked with red X markers).
            title: Figure title.
            figsize: Override figure size.

        Returns:
            ``matplotlib.figure.Figure``.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(
            figsize=figsize or self.figsize_default, dpi=self.dpi
        )
        fig.patch.set_facecolor(self.theme["bg"])
        colors = self.theme["accent"]
        r = list(rounds)
        q = list(qbers)

        _style_axes(ax, self.theme, "Round", "QBER", title)

        # Shade safe and unsafe zones
        ax.axhspan(0, security_threshold, alpha=0.07, color=colors[2], label="Secure zone")
        ax.axhspan(security_threshold, 0.5, alpha=0.07, color=colors[0], label="Insecure zone")

        # QBER line
        ax.plot(r, q, color=colors[4], linewidth=2.0,
                label=f"{protocol_name} QBER", marker="^", markersize=4)

        # Security threshold line
        ax.axhline(
            y=security_threshold,
            color=colors[0],
            linestyle="--",
            linewidth=1.5,
            label=f"Threshold ({security_threshold:.0%})",
        )

        # Mark eavesdropping detections
        if eavesdrop_detected_rounds:
            eve_idxs = [i for i, rnd in enumerate(rounds) if rnd in eavesdrop_detected_rounds]
            eve_rounds = [rounds[i] for i in eve_idxs]
            eve_qbers = [qbers[i] for i in eve_idxs]
            ax.scatter(eve_rounds, eve_qbers, color=colors[1], marker="X",
                       s=120, zorder=5, label="Eavesdropper detected")

        ax.set_ylim(0, max(max(q) * 1.2, security_threshold * 1.5))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

        n_violations = sum(1 for qber in q if qber > security_threshold)
        ax.text(
            0.02, 0.97,
            f"Threshold violations: {n_violations}/{len(q)}",
            transform=ax.transAxes,
            ha="left", va="top", fontsize=8,
            color=colors[0] if n_violations > 0 else colors[2],
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=self.theme["bg"],
                edgecolor=colors[0] if n_violations > 0 else colors[2],
                alpha=0.85,
            ),
        )

        ax.legend(fontsize=8, facecolor=self.theme["bg"], labelcolor=self.theme["text_color"])
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------ #
    # 5. Differential privacy budget
    # ------------------------------------------------------------------ #

    def plot_privacy_budget(
        self,
        rounds: Sequence[int],
        epsilon_per_round: Sequence[float],
        delta: float = 1e-5,
        total_budget: Optional[float] = None,
        title: str = "Differential Privacy Budget Consumption",
        figsize: Optional[Tuple[float, float]] = None,
    ):
        """Plot cumulative differential privacy epsilon consumption over rounds.

        Args:
            rounds: Round indices.
            epsilon_per_round: Privacy cost ε added per round.
            delta: Privacy δ parameter (shown in annotation).
            total_budget: If provided, draw a horizontal budget cap line.
            title: Figure title.
            figsize: Override figure size.

        Returns:
            ``matplotlib.figure.Figure`` with two subplots:
            (per-round ε, cumulative ε).
        """
        import matplotlib.pyplot as plt

        fig, (ax_per, ax_cum) = plt.subplots(
            1, 2, figsize=figsize or self.figsize_default, dpi=self.dpi
        )
        fig.patch.set_facecolor(self.theme["bg"])
        colors = self.theme["accent"]
        r = list(rounds)
        eps = list(epsilon_per_round)
        cum_eps = list(np.cumsum(eps))

        # Per-round plot
        _style_axes(ax_per, self.theme, "Round", "ε per round", "Per-Round Privacy Cost")
        ax_per.bar(r, eps, color=colors[5], alpha=0.8, edgecolor=self.theme["spine_color"],
                   linewidth=0.5, label="ε per round")
        ax_per.plot(r, eps, color=colors[5], linewidth=1.5, marker="o", markersize=3)
        ax_per.legend(fontsize=8, facecolor=self.theme["bg"], labelcolor=self.theme["text_color"])

        # Cumulative plot
        _style_axes(ax_cum, self.theme, "Round", "Cumulative ε", "Cumulative Privacy Budget")
        ax_cum.plot(r, cum_eps, color=colors[3], linewidth=2.2, label="Cumulative ε",
                    marker="o", markersize=3)
        ax_cum.fill_between(r, cum_eps, alpha=0.12, color=colors[3])

        if total_budget is not None:
            ax_cum.axhline(y=total_budget, color=colors[1], linestyle="--",
                           linewidth=1.5, label=f"Budget cap ε={total_budget}")
            remaining = total_budget - cum_eps[-1]
            pct_used = cum_eps[-1] / total_budget * 100
            ax_cum.text(
                0.97, 0.05,
                f"Used: {pct_used:.1f}%\nRemaining: {remaining:.4f}",
                transform=ax_cum.transAxes,
                ha="right", va="bottom", fontsize=8,
                color=self.theme["text_color"],
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor=self.theme["bg"],
                    edgecolor=self.theme["spine_color"],
                    alpha=0.85,
                ),
            )

        # δ annotation
        ax_cum.text(
            0.02, 0.97,
            f"δ = {delta:.0e}",
            transform=ax_cum.transAxes,
            ha="left", va="top", fontsize=8, color=self.theme["text_color"],
        )
        ax_cum.legend(fontsize=8, facecolor=self.theme["bg"], labelcolor=self.theme["text_color"])

        fig.suptitle(title, color=self.theme["text_color"], fontsize=13, y=1.01)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------ #
    # 6. Multi-client accuracy heatmap
    # ------------------------------------------------------------------ #

    def plot_client_accuracy_heatmap(
        self,
        rounds: Sequence[int],
        client_accuracies: List[Sequence[float]],
        client_labels: Optional[List[str]] = None,
        title: str = "Per-Client Accuracy Heatmap",
        figsize: Optional[Tuple[float, float]] = None,
    ):
        """Plot a heatmap of per-client accuracy across all training rounds.

        Args:
            rounds: Round indices (x-axis).
            client_accuracies: List of accuracy sequences, one per client.
            client_labels: Optional client label strings.
            title: Figure title.
            figsize: Override figure size.

        Returns:
            ``matplotlib.figure.Figure``.
        """
        import matplotlib.pyplot as plt

        n_clients = len(client_accuracies)
        if client_labels is None:
            client_labels = [f"Client {i}" for i in range(n_clients)]

        data = np.array([list(ca) for ca in client_accuracies])
        r = list(rounds)

        default_size = (max(8, len(r) * 0.4 + 2), max(4, n_clients * 0.5 + 1.5))
        fig, ax = plt.subplots(figsize=figsize or default_size, dpi=self.dpi)
        fig.patch.set_facecolor(self.theme["bg"])

        im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0, vmax=1,
                       origin="upper", interpolation="nearest")

        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label("Accuracy", color=self.theme["text_color"])
        cbar.ax.yaxis.set_tick_params(color=self.theme["text_color"])
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=self.theme["text_color"])
        cbar.ax.set_facecolor(self.theme["bg"])

        ax.set_facecolor(self.theme["bg"])
        ax.set_xticks(range(len(r)))
        ax.set_xticklabels([str(rnd) for rnd in r], rotation=45, ha="right",
                           fontsize=7, color=self.theme["text_color"])
        ax.set_yticks(range(n_clients))
        ax.set_yticklabels(client_labels, fontsize=8, color=self.theme["text_color"])
        ax.set_xlabel("Round", color=self.theme["text_color"])
        ax.set_ylabel("Client", color=self.theme["text_color"])
        ax.set_title(title, color=self.theme["text_color"], fontsize=11, pad=10)

        for spine in ax.spines.values():
            spine.set_edgecolor(self.theme["spine_color"])

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------ #
    # 7. Combined dashboard
    # ------------------------------------------------------------------ #

    def plot_dashboard(
        self,
        rounds: Sequence[int],
        losses: Sequence[float],
        accuracies: Sequence[float],
        key_rates: Optional[Sequence[float]] = None,
        qbers: Optional[Sequence[float]] = None,
        epsilon_per_round: Optional[Sequence[float]] = None,
        title: str = "QuantumShield-FL Training Dashboard",
        figsize: Optional[Tuple[float, float]] = None,
    ):
        """Render a 2×3 dashboard summarising the full training run.

        Args:
            rounds: Round indices.
            losses: Global loss per round.
            accuracies: Global accuracy per round.
            key_rates: Optional QKD key rate sequence.
            qbers: Optional QBER sequence.
            epsilon_per_round: Optional per-round privacy cost sequence.
            title: Figure title.
            figsize: Override figure size.

        Returns:
            ``matplotlib.figure.Figure`` with up to 6 subplots.
        """
        import matplotlib.pyplot as plt

        rows, cols = 2, 3
        fs = figsize or (16, 9)
        fig, axes = plt.subplots(rows, cols, figsize=fs, dpi=self.dpi)
        fig.patch.set_facecolor(self.theme["bg"])
        colors = self.theme["accent"]
        r = list(rounds)

        def _ax(row, col):
            return axes[row][col]

        # 0,0 – Loss
        ax = _ax(0, 0)
        _style_axes(ax, self.theme, "Round", "Loss", "Global Loss")
        ax.plot(r, list(losses), color=colors[0], linewidth=2.0)
        ax.fill_between(r, list(losses), alpha=0.1, color=colors[0])

        # 0,1 – Accuracy
        ax = _ax(0, 1)
        _style_axes(ax, self.theme, "Round", "Accuracy", "Global Accuracy")
        ax.plot(r, list(accuracies), color=colors[1], linewidth=2.0)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.fill_between(r, list(accuracies), alpha=0.1, color=colors[1])

        # 0,2 – Key rate or placeholder
        ax = _ax(0, 2)
        if key_rates is not None:
            _style_axes(ax, self.theme, "Round", "Key Rate (bps)", "QKD Key Rate")
            ax.plot(r, list(key_rates), color=colors[3], linewidth=2.0, marker="D", markersize=3)
        else:
            ax.set_facecolor(self.theme["bg"])
            ax.text(0.5, 0.5, "No QKD data", ha="center", va="center",
                    color=self.theme["text_color"], fontsize=10)
            ax.axis("off")

        # 1,0 – QBER
        ax = _ax(1, 0)
        if qbers is not None:
            _style_axes(ax, self.theme, "Round", "QBER", "QBER Evolution")
            ax.plot(r, list(qbers), color=colors[4], linewidth=2.0, marker="^", markersize=3)
            ax.axhline(y=0.11, color=colors[0], linestyle="--", linewidth=1.2, label="Threshold")
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
            ax.legend(fontsize=7, facecolor=self.theme["bg"], labelcolor=self.theme["text_color"])
        else:
            ax.set_facecolor(self.theme["bg"])
            ax.text(0.5, 0.5, "No QBER data", ha="center", va="center",
                    color=self.theme["text_color"], fontsize=10)
            ax.axis("off")

        # 1,1 – Privacy budget
        ax = _ax(1, 1)
        if epsilon_per_round is not None:
            cum = list(np.cumsum(list(epsilon_per_round)))
            _style_axes(ax, self.theme, "Round", "Cumulative ε", "Privacy Budget")
            ax.plot(r, cum, color=colors[5], linewidth=2.0)
            ax.fill_between(r, cum, alpha=0.12, color=colors[5])
        else:
            ax.set_facecolor(self.theme["bg"])
            ax.text(0.5, 0.5, "DP not enabled", ha="center", va="center",
                    color=self.theme["text_color"], fontsize=10)
            ax.axis("off")

        # 1,2 – Summary statistics
        ax = _ax(1, 2)
        ax.set_facecolor(self.theme["bg"])
        ax.axis("off")
        best_acc = max(accuracies)
        final_loss = losses[-1]
        summary_lines = [
            f"Rounds completed: {len(r)}",
            f"Best accuracy:    {best_acc:.2%}",
            f"Final loss:       {final_loss:.4f}",
        ]
        if key_rates:
            summary_lines.append(f"Avg key rate:     {np.mean(list(key_rates)):.1f} bps")
        if qbers:
            summary_lines.append(f"Avg QBER:         {np.mean(list(qbers)):.2%}")
        if epsilon_per_round:
            summary_lines.append(f"Total ε:          {sum(epsilon_per_round):.4f}")

        ax.text(
            0.1, 0.9, "\n".join(summary_lines),
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=9,
            fontfamily="monospace",
            color=self.theme["text_color"],
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor=self.theme["bg"],
                edgecolor=self.theme["spine_color"],
                alpha=0.9,
            ),
        )
        ax.set_title("Summary", color=self.theme["text_color"], fontsize=10)

        fig.suptitle(title, color=self.theme["text_color"], fontsize=14, y=1.01)
        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def plot_convergence(
    rounds: Sequence[int],
    losses: Sequence[float],
    accuracies: Sequence[float],
    theme: str = "light",
    **kwargs,
):
    """Module-level shortcut for ``TrainingVisualizer.plot_convergence``.

    Args:
        rounds: Round indices.
        losses: Global loss per round.
        accuracies: Global accuracy per round.
        theme: Visualisation theme.
        **kwargs: Additional keyword arguments forwarded to ``plot_convergence``.

    Returns:
        ``matplotlib.figure.Figure``.
    """
    return TrainingVisualizer(theme=theme).plot_convergence(
        rounds, losses, accuracies, **kwargs
    )


def plot_qber(
    rounds: Sequence[int],
    qbers: Sequence[float],
    theme: str = "light",
    **kwargs,
):
    """Module-level shortcut for ``TrainingVisualizer.plot_qber``.

    Args:
        rounds: Round indices.
        qbers: QBER values per round.
        theme: Visualisation theme.
        **kwargs: Additional keyword arguments forwarded to ``plot_qber``.

    Returns:
        ``matplotlib.figure.Figure``.
    """
    return TrainingVisualizer(theme=theme).plot_qber(rounds, qbers, **kwargs)


def plot_privacy_budget(
    rounds: Sequence[int],
    epsilon_per_round: Sequence[float],
    theme: str = "light",
    **kwargs,
):
    """Module-level shortcut for ``TrainingVisualizer.plot_privacy_budget``.

    Args:
        rounds: Round indices.
        epsilon_per_round: Per-round privacy cost sequence.
        theme: Visualisation theme.
        **kwargs: Additional keyword arguments forwarded to ``plot_privacy_budget``.

    Returns:
        ``matplotlib.figure.Figure``.
    """
    return TrainingVisualizer(theme=theme).plot_privacy_budget(
        rounds, epsilon_per_round, **kwargs
    )
