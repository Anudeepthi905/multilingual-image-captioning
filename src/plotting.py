"""Matplotlib helpers shared by training, inference and evaluation."""
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Fixed categorical order (validated palette); series keep their slot across charts.
SERIES = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2",
          "#ec4647"]
TEXT, TEXT_2, GRID = "#191919", "#666666", "#e5e5e5"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(TEXT_2)
    ax.tick_params(colors=TEXT_2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_loss(history: list, out_path: Path, title: str = "") -> Path:
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    for i, key in enumerate(["train_loss", "val_loss"]):
        ys = [h[key] for h in history]
        if all(math.isnan(y) for y in ys):
            continue
        label = key.replace("_", " ")
        ax.plot(epochs, ys, color=SERIES[i], linewidth=2, marker="o", markersize=5,
                label=label)
        ax.annotate(label, (epochs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                    color=TEXT, va="center", fontsize=9)
    ax.set_xlabel("epoch", color=TEXT)
    ax.set_ylabel("masked cross-entropy", color=TEXT)
    ax.set_title(f"Loss - {title}", color=TEXT, loc="left")
    ax.legend(frameon=False, labelcolor=TEXT)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_comparison(rows: list, metrics: list, out_path: Path) -> Path:
    """Grouped bars: one group per metric, one bar (colour) per run."""
    rows = rows[: len(SERIES)]
    n = len(rows)
    width = 0.8 / max(n, 1)
    fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(metrics) + n), 4.5), dpi=120)
    for i, row in enumerate(rows):
        xs = [m + (i - (n - 1) / 2) * width for m in range(len(metrics))]
        ax.bar(xs, [float(row[k]) for k in metrics], width=width * 0.9, color=SERIES[i],
               label=f"{row['run']} ({row.get('decoding', '')})", edgecolor="white", linewidth=1)
    labels = [m + "\n(lower is better)" if m == "WER" else m for m in metrics]
    ax.set_xticks(range(len(metrics)), labels, color=TEXT)
    ax.set_ylabel("score", color=TEXT)
    ax.set_ylim(0, 1)
    ax.set_title("Held-out metrics by run", color=TEXT, loc="left")
    ax.legend(frameon=False, labelcolor=TEXT, fontsize=8, loc="upper right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
