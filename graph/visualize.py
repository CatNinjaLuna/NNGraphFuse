# graph/visualize.py
#
# ROLE: Pipeline Step 4 — Graph Diff Visualization
#
# Renders side-by-side before/after PNG diffs for each optimization pass.
# Nodes are colored by op type. Nodes removed by the pass are highlighted
# in red on the before side; fused nodes show a merge indicator.
#
# OUTPUT FILES (saved to images/graphs/):
#   resnet50_fusion.png       — Conv+Relu fusion diff (ResNet-50)
#   mobilenet_fold.png        — Constant folding diff (MobileNetV2)
#   resnet50_dead_node.png    — Dead node elimination diff (ResNet-50)
#
# USAGE:
#   python -m graph.visualize                  # generate all three PNGs
#   python -m graph.visualize --pass fusion    # single pass
#
# LAYOUT NOTE:
#   Large graphs (124 nodes) are unreadable as full node-link diagrams.
#   This file renders a cluster-based summary graph: one node per op type,
#   edge weight proportional to connection count, with before/after counts
#   annotated directly on each node. This keeps the diff readable at any
#   graph size while still showing the structural change each pass produces.

import os
import copy
import argparse
from collections import Counter, defaultdict

import networkx as nx
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for PNG output
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Color palette — one color per op type, consistent across before/after
# ---------------------------------------------------------------------------

OP_COLORS = {
    "Conv":             "#4C72B0",
    "Relu":             "#DD8452",
    "Add":              "#55A868",
    "Constant":         "#C44E52",
    "Clip":             "#8172B3",
    "Gemm":             "#937860",
    "MaxPool":          "#DA8BC3",
    "GlobalAveragePool":"#8C8C8C",
    "Flatten":          "#CCB974",
    "Reshape":          "#64B5CD",
    "Shape":            "#64B5CD",
    "ReduceMean":       "#8C8C8C",
    "Concat":           "#CCB974",
    "Sub":              "#A6CEE3",
    "Div":              "#B2DF8A",
}
DEFAULT_COLOR    = "#AAAAAA"
REMOVED_COLOR    = "#FF4444"   # red  — node removed by this pass
HIGHLIGHT_COLOR  = "#FFD700"   # gold — node modified (fused)


# ---------------------------------------------------------------------------
# Build a summary graph: one node per op type, edges from data flow
# ---------------------------------------------------------------------------

def _build_summary_graph(ir: dict) -> nx.DiGraph:
    """
    Build a DiGraph where each node is an op type and each edge represents
    at least one data-flow connection between those op types in the IR.
    Edge weight = number of such connections.
    """
    G = nx.DiGraph()

    # Count op types
    op_counts = Counter(n["op"] for n in ir["nodes"])
    for op, count in op_counts.items():
        G.add_node(op, count=count)

    # Build output_tensor → op_type map
    output_to_op = {}
    for node in ir["nodes"]:
        for out in node["outputs"]:
            output_to_op[out] = node["op"]

    # Add edges
    edge_weights = defaultdict(int)
    for node in ir["nodes"]:
        consumer_op = node["op"]
        for inp in node["inputs"]:
            if inp and inp in output_to_op:
                producer_op = output_to_op[inp]
                if producer_op != consumer_op:
                    edge_weights[(producer_op, consumer_op)] += 1

    for (src, dst), weight in edge_weights.items():
        G.add_edge(src, dst, weight=weight)

    return G


def _graph_layout(G: nx.DiGraph) -> dict:
    """
    Use hierarchical layout for DAGs. Falls back to spring layout.
    """
    try:
        # Attempt topological / shell layout
        layers = list(nx.topological_generations(G))
        pos = {}
        for layer_idx, layer_nodes in enumerate(layers):
            for node_idx, node in enumerate(sorted(layer_nodes)):
                x = layer_idx * 2.5
                y = node_idx - len(layer_nodes) / 2
                pos[node] = (x, y)
        return pos
    except Exception:
        return nx.spring_layout(G, seed=42, k=2.5)


# ---------------------------------------------------------------------------
# Core rendering: one before/after subplot pair
# ---------------------------------------------------------------------------

def _render_diff(
    ir_before: dict,
    ir_after: dict,
    title: str,
    removed_ops: list[str],
    modified_ops: list[str],
    output_path: str,
):
    """
    Render a side-by-side before/after summary graph diff and save as PNG.

    Args:
        ir_before:    IR dict before the pass.
        ir_after:     IR dict after the pass.
        title:        Figure title (e.g. "Conv + Relu Fusion — ResNet-50").
        removed_ops:  Op types that were removed or reduced by this pass.
        modified_ops: Op types that were structurally changed (e.g. fused Conv).
        output_path:  Full path to save the PNG.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    G_before = _build_summary_graph(ir_before)
    G_after  = _build_summary_graph(ir_after)

    counts_before = Counter(n["op"] for n in ir_before["nodes"])
    counts_after  = Counter(n["op"] for n in ir_after["nodes"])

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.patch.set_facecolor("#1A1A2E")

    for ax, G, counts, side_label, is_after in [
        (axes[0], G_before, counts_before, "Before", False),
        (axes[1], G_after,  counts_after,  "After",  True),
    ]:
        ax.set_facecolor("#16213E")

        if len(G.nodes) == 0:
            ax.text(0.5, 0.5, "Empty graph", ha="center", va="center",
                    color="white", transform=ax.transAxes)
            ax.set_title(side_label, color="white", fontsize=13, pad=10)
            continue

        pos = _graph_layout(G)

        # Node colors
        node_colors = []
        for node in G.nodes:
            if is_after:
                node_colors.append(OP_COLORS.get(node, DEFAULT_COLOR))
            else:
                if node in removed_ops:
                    node_colors.append(REMOVED_COLOR)
                elif node in modified_ops:
                    node_colors.append(HIGHLIGHT_COLOR)
                else:
                    node_colors.append(OP_COLORS.get(node, DEFAULT_COLOR))

        # Node sizes proportional to count
        node_sizes = [max(800, counts.get(node, 1) * 120) for node in G.nodes]

        # Edge widths proportional to connection count
        edge_weights = [G[u][v].get("weight", 1) * 0.8 for u, v in G.edges]

        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color="#AAAAAA",
            width=edge_weights,
            arrows=True,
            arrowsize=15,
            connectionstyle="arc3,rad=0.1",
            alpha=0.6,
        )

        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.92,
        )

        # Labels: op type + count
        labels = {node: f"{node}\n({counts.get(node, 0)})" for node in G.nodes}
        nx.draw_networkx_labels(
            G, pos, labels=labels, ax=ax,
            font_size=8, font_color="white", font_weight="bold",
        )

        total = sum(counts.values())
        ax.set_title(
            f"{side_label}  ·  {total} nodes",
            color="white", fontsize=13, fontweight="bold", pad=12,
        )
        ax.axis("off")

    # Legend
    legend_items = [
        mpatches.Patch(color=REMOVED_COLOR,   label="Removed / reduced by pass"),
        mpatches.Patch(color=HIGHLIGHT_COLOR,  label="Modified (fused)"),
    ]
    for op, color in sorted(OP_COLORS.items()):
        if op in counts_before or op in counts_after:
            legend_items.append(mpatches.Patch(color=color, label=op))

    fig.legend(
        handles=legend_items,
        loc="lower center",
        ncol=6,
        framealpha=0.15,
        labelcolor="white",
        fontsize=8,
        facecolor="#1A1A2E",
    )

    fig.suptitle(title, color="white", fontsize=15, fontweight="bold", y=0.97)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✅ Saved: {output_path}")


# ---------------------------------------------------------------------------
# Pass-specific visualization functions
# ---------------------------------------------------------------------------

def visualize_fusion(output_dir: str = "images/graphs"):
    """Conv + Relu Fusion diff — ResNet-50."""
    from graph.ir import load_ir
    from passes.fusion import apply_fusion

    print("\n[1/3] Conv + Relu Fusion (ResNet-50)...")

    ir = load_ir("models/resnet50.onnx")
    ir_before = copy.deepcopy(ir)

    # Convert to legacy dict, fuse, convert back
    legacy = {f"node_{i}": n for i, n in enumerate(ir["nodes"])}
    from passes.fusion import apply_fusion as _apply_fusion
    legacy_fused, _ = _apply_fusion(legacy)
    ir_after = {**ir, "nodes": list(legacy_fused.values())}

    _render_diff(
        ir_before=ir_before,
        ir_after=ir_after,
        title="Conv + Relu Fusion  |  ResNet-50",
        removed_ops=["Relu"],
        modified_ops=["Conv"],
        output_path=os.path.join(output_dir, "resnet50_fusion.png"),
    )


def visualize_constant_fold(output_dir: str = "images/graphs"):
    """Constant Folding diff — MobileNetV2."""
    from graph.ir import load_ir
    from passes.constant_fold import run_constant_folding, remove_unused_initializers

    print("\n[2/3] Constant Folding (MobileNetV2)...")

    ir = load_ir("models/mobilenetv2_normalized.onnx")
    ir_before = copy.deepcopy(ir)
    ir_after = run_constant_folding(ir)
    ir_after = remove_unused_initializers(ir_after)

    _render_diff(
        ir_before=ir_before,
        ir_after=ir_after,
        title="Constant Folding  |  MobileNetV2",
        removed_ops=["Constant"],
        modified_ops=[],
        output_path=os.path.join(output_dir, "mobilenet_fold.png"),
    )


def visualize_dead_node(output_dir: str = "images/graphs"):
    """Dead Node Elimination diff — ResNet-50 (post-fusion)."""
    from graph.ir import load_ir
    from passes.dead_node import run_dead_node_elimination

    print("\n[3/3] Dead Node Elimination (ResNet-50, post-fusion)...")

    ir = load_ir("models/resnet50.onnx")

    # Apply fusion first so we're visualizing the correct input state
    legacy = {f"node_{i}": n for i, n in enumerate(ir["nodes"])}
    from passes.fusion import apply_fusion as _apply_fusion
    legacy_fused, _ = _apply_fusion(legacy)
    ir = {**ir, "nodes": list(legacy_fused.values())}

    ir_before = copy.deepcopy(ir)
    ir_after = run_dead_node_elimination(ir)

    _render_diff(
        ir_before=ir_before,
        ir_after=ir_after,
        title="Dead Node Elimination  |  ResNet-50 (post-fusion)",
        removed_ops=[],
        modified_ops=[],
        output_path=os.path.join(output_dir, "resnet50_dead_node.png"),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

PASS_MAP = {
    "fusion":    visualize_fusion,
    "fold":      visualize_constant_fold,
    "dead_node": visualize_dead_node,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NNGraphFuse graph diff visualizer")
    parser.add_argument(
        "--pass",
        dest="pass_name",
        choices=list(PASS_MAP.keys()) + ["all"],
        default="all",
        help="Which pass to visualize (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        default="images/graphs",
        help="Directory to save PNG files (default: images/graphs)",
    )
    args = parser.parse_args()

    print(f"\nSaving PNGs to: {args.output_dir}/")

    if args.pass_name == "all":
        for fn in PASS_MAP.values():
            fn(output_dir=args.output_dir)
    else:
        PASS_MAP[args.pass_name](output_dir=args.output_dir)

    print(f"\nDone. Embed in README with:")
    print(f"  ![fusion](images/graphs/resnet50_fusion.png)")
    print(f"  ![fold](images/graphs/mobilenet_fold.png)")
    print(f"  ![dead_node](images/graphs/resnet50_dead_node.png)")