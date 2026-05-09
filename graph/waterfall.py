# graph/waterfall.py
#
# ROLE: Pipeline Step 4b — Node Reduction Waterfall Chart
#
# Renders a waterfall chart showing cumulative node count reduction
# across the optimization pipeline for each model.
#
# ResNet-50:   124 → 91 (fusion) → 91 (dead node)
# MobileNetV2: 172 → 102 (fold)  → 102 (dead node)
#
# OUTPUT: images/graphs/waterfall.png
#
# USAGE:
#   python -m graph.waterfall

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

PIPELINE = {
    "ResNet-50": {
        "steps": ["Baseline", "Conv+Relu\nFusion", "Dead Node\nElimination"],
        "counts": [124, 91, 91],
        "color":  "#4C72B0",
    },
    "MobileNetV2": {
        "steps": ["Baseline", "Constant\nFolding", "Dead Node\nElimination"],
        "counts": [172, 102, 102],
        "color":  "#8172B3",
    },
}

REMOVED_COLOR  = "#FF4444"
UNCHANGED_COLOR = "#2A2A4A"
BG_COLOR       = "#1A1A2E"
PANEL_COLOR    = "#16213E"
TEXT_COLOR     = "white"
GRID_COLOR     = "#2A2A4A"


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render_waterfall(output_path: str = "images/graphs/waterfall.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor(BG_COLOR)

    for ax, (model_name, data) in zip(axes, PIPELINE.items()):
        ax.set_facecolor(PANEL_COLOR)

        steps  = data["steps"]
        counts = data["counts"]
        color  = data["color"]
        n      = len(steps)
        x      = np.arange(n)
        bar_w  = 0.55

        # Draw bars
        for i, (count, step) in enumerate(zip(counts, steps)):
            prev  = counts[i - 1] if i > 0 else count
            delta = prev - count

            # Full bar (surviving nodes)
            ax.bar(i, count, width=bar_w, color=color, alpha=0.85,
                   zorder=3, linewidth=0)

            # Removed portion stacked on top
            if delta > 0:
                ax.bar(i, delta, bottom=count, width=bar_w,
                       color=REMOVED_COLOR, alpha=0.75, zorder=3, linewidth=0)

            # Connector line to next bar
            if i < n - 1:
                ax.hlines(count, i + bar_w / 2, i + 1 - bar_w / 2,
                          colors="#AAAAAA", linewidths=1.2,
                          linestyles="dashed", zorder=2, alpha=0.5)

            # Count label inside bar
            ax.text(i, count / 2, str(count),
                    ha="center", va="center",
                    color=TEXT_COLOR, fontsize=12, fontweight="bold", zorder=4)

            # Delta label above removed portion
            if delta > 0:
                ax.text(i, count + delta / 2, f"−{delta}",
                        ha="center", va="center",
                        color=TEXT_COLOR, fontsize=10, fontweight="bold", zorder=4)

        # Percentage reduction annotation
        total_removed = counts[0] - counts[-1]
        pct = total_removed / counts[0] * 100
        ax.text(n - 1, counts[0] * 1.05,
                f"{pct:.0f}% reduction\n({counts[0]} → {counts[-1]} nodes)",
                ha="center", va="bottom",
                color="#FFD700", fontsize=10, fontweight="bold")

        # Axes styling
        ax.set_xticks(x)
        ax.set_xticklabels(steps, color=TEXT_COLOR, fontsize=10)
        ax.set_ylabel("Node count", color=TEXT_COLOR, fontsize=10)
        ax.tick_params(colors=TEXT_COLOR)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_color(GRID_COLOR)
        ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(0, counts[0] * 1.18)
        ax.set_title(model_name, color=TEXT_COLOR,
                     fontsize=13, fontweight="bold", pad=12)
        ax.tick_params(axis="y", colors=TEXT_COLOR)

    # Legend
    legend_items = [
        mpatches.Patch(color="#4C72B0",   label="ResNet-50 nodes"),
        mpatches.Patch(color="#8172B3",   label="MobileNetV2 nodes"),
        mpatches.Patch(color=REMOVED_COLOR, label="Nodes eliminated by pass"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=3,
               framealpha=0.15, labelcolor=TEXT_COLOR, fontsize=9,
               facecolor=BG_COLOR)

    fig.suptitle("Node Reduction Waterfall",
                 color=TEXT_COLOR, fontsize=15, fontweight="bold", y=0.97)

    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✅ Saved: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\nGenerating waterfall chart...")
    render_waterfall()
    print("  Embed in README with:")
    print("  ![Waterfall](images/graphs/waterfall.png)")