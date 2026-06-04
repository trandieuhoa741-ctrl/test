from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .env import DispatchEnv


def plot_training_curves(history: list[dict[str, float]], path: str | Path) -> None:
    if not history:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x_key = "episode" if "episode" in history[0] else "epoch"
    x_label = "Episode" if x_key == "episode" else "Epoch"
    steps = [row[x_key] for row in history]

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(steps, [row["reward"] for row in history], color="#0f766e", linewidth=1.8)
    axes[0].set_ylabel("Avg. Reward")
    axes[0].grid(alpha=0.25)
    axes[1].plot(steps, [row["response_rate"] for row in history], color="#b45309", linewidth=1.8)
    axes[1].set_ylabel("Avg. Response")
    axes[1].set_xlabel(x_label)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_cell_cancellations(env: DispatchEnv, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rates = env.cancellation_rate_by_cell()
    xy = env.grid.xy

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(xy[:, 0], xy[:, 1], c=rates, cmap="magma_r", s=210, edgecolor="#1f2937", linewidth=0.45)
    for i, (x, y) in enumerate(xy):
        ax.text(x, y, str(i), ha="center", va="center", fontsize=6, color="white")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Cancellation rate by cell")
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
