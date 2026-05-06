# passes/dead_node.py
#
# ROLE: Pipeline Step 3c — Dead Node Elimination Pass
#
# A node is "dead" if none of its outputs are consumed by any other
# node in the graph AND none of its outputs are graph-level outputs.
# Dead nodes are safe to remove — they produce values nobody reads.
#
# WHERE DEAD NODES COME FROM:
# PyTorch's ONNX exporter emits nodes to handle dynamic shapes at
# export time. When you export with a fixed input size (e.g. 224x224),
# those shape-handling ops (Shape, Gather, Unsqueeze, Concat, Reshape)
# compute a shape that is never actually used downstream — their output
# tensors are orphaned. Dead node elimination prunes them.
#
# COMMON DEAD SUBGRAPHS IN RESNET-50 / MOBILENETV2:
#   Shape → Gather → Unsqueeze → Concat → Reshape  (dynamic batch dim path)
#   Constant → Reshape  (when Reshape's shape input is unused)
#
# ALGORITHM (one forward pass, O(N)):
#   1. Collect all tensor names that are consumed as inputs by any node.
#   2. Add all graph-level output tensor names to the "live" set.
#   3. A node is dead if ALL of its outputs are absent from the live set.
#   4. Remove dead nodes; repeat until no more removals (handles chains).
#
# NOTE: This pass operates on the new list-based IR from load_ir().
# It does NOT use the legacy load_graph() dict format.


def _build_consumed_set(nodes: list) -> set:
    """
    Return the set of all tensor names that appear as an input
    to at least one node. These tensors are 'consumed' — live.
    """
    consumed = set()
    for node in nodes:
        for inp in node["inputs"]:
            if inp:  # ONNX uses empty string for optional missing inputs
                consumed.add(inp)
    return consumed


def eliminate_dead_nodes(ir: dict) -> tuple[dict, int]:
    """
    Remove all dead nodes from the IR (in-place).

    A node is dead if every one of its outputs is unconsumed AND
    not a graph-level output. Runs iteratively until convergence
    so that chains of dead nodes (A → B → C, all dead) are fully removed.

    Args:
        ir: IR dict from load_ir() with keys: nodes, initializers,
            inputs, outputs.

    Returns:
        (modified ir, total nodes removed)
    """
    graph_outputs = set(ir["outputs"])
    total_removed = 0

    while True:
        consumed = _build_consumed_set(ir["nodes"])
        # A tensor is "live" if it is consumed by a node OR is a graph output
        live_tensors = consumed | graph_outputs

        surviving = []
        removed_this_round = 0

        for node in ir["nodes"]:
            node_outputs = set(o for o in node["outputs"] if o)

            # Node is dead only if ALL its outputs are outside the live set
            if node_outputs and node_outputs.isdisjoint(live_tensors):
                removed_this_round += 1  # drop this node
            else:
                surviving.append(node)

        ir["nodes"] = surviving
        total_removed += removed_this_round

        # Converged — no more dead nodes found this round
        if removed_this_round == 0:
            break

    return ir, total_removed


def summarize_removed(before_nodes: list, after_nodes: list):
    """
    Print a before/after op-type breakdown, highlighting removed ops.
    Mirrors the style of fusion.py and constant_fold.py summaries.
    """
    from collections import Counter

    before_counts = Counter(n["op"] for n in before_nodes)
    after_counts  = Counter(n["op"] for n in after_nodes)
    all_ops = sorted(before_counts.keys() | after_counts.keys())

    print("\n📊 Op type summary — before vs after dead node elimination:\n")
    print(f"   {'Op':<22} {'Before':>6}  {'After':>6}")
    print(f"   {'-'*22} {'-'*6}  {'-'*6}")
    for op in all_ops:
        b = before_counts.get(op, 0)
        a = after_counts.get(op, 0)
        marker = "  ← removed" if a < b else ""
        print(f"   {op:<22} {b:>6}  {a:>6}{marker}")
    print(f"   {'─'*36}")
    print(f"   {'Total':<22} {sum(before_counts.values()):>6}  {sum(after_counts.values()):>6}")


# ---------------------------------------------------------------------------
# Public alias — matches the import in constant_fold.py:
#   from passes.dead_node import run_dead_node_elimination
# ---------------------------------------------------------------------------

def run_dead_node_elimination(ir: dict) -> dict:
    """
    Thin wrapper around eliminate_dead_nodes() that returns only the IR.
    Use this when chaining passes in a pipeline:

        ir = run_constant_folding(ir)
        ir = run_dead_node_elimination(ir)   # cleans up newly orphaned nodes
    """
    ir, _ = eliminate_dead_nodes(ir)
    return ir


# ---------------------------------------------------------------------------
# Entry point — run directly to test on both models
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import copy
    from graph.ir import load_ir

    # --- Test on ResNet-50 standalone ---------------------------------------
    print("=" * 52)
    print("  Dead Node Elimination — ResNet-50 (standalone)")
    print("=" * 52)

    ir = load_ir("models/resnet50.onnx")
    nodes_before = copy.deepcopy(ir["nodes"])
    ir, removed = eliminate_dead_nodes(ir)
    print(f"\n✅ {len(nodes_before)} → {len(ir['nodes'])} nodes  ({removed} removed)")
    summarize_removed(nodes_before, ir["nodes"])

    if removed == 0:
        print("\n💡 No dead nodes in ResNet-50 standalone.")
        print("   Shape/Reshape outputs feed into live Gemm — not orphaned.")
        print("   Dead nodes appear AFTER constant folding removes their consumers.")

    # --- Test on MobileNetV2: fold first, then eliminate -------------------
    print("\n" + "=" * 52)
    print("  Dead Node Elimination — MobileNetV2 (post-fold)")
    print("=" * 52)
    print("  (Running constant folding first to expose dead nodes)\n")

    from passes.constant_fold import run_constant_folding, remove_unused_initializers

    ir2 = load_ir("models/mobilenetv2_normalized.onnx")
    ir2 = run_constant_folding(ir2)
    ir2 = remove_unused_initializers(ir2)

    nodes_before2 = copy.deepcopy(ir2["nodes"])
    ir2, removed2 = eliminate_dead_nodes(ir2)
    print(f"\n✅ {len(nodes_before2)} → {len(ir2['nodes'])} nodes  ({removed2} removed)")
    summarize_removed(nodes_before2, ir2["nodes"])

    if removed2 == 0:
        print("\n💡 No additional dead nodes after folding on MobileNetV2.")
        print("   Sub and Div remain: they consume the live image tensor at runtime.")
    else:
        print(f"\n🔍 Dead node elimination cleaned up {removed2} node(s) left by constant folding.")

        