# graph/ir.py
#
# ROLE: Pipeline Step 2 — Graph Parsing & IR Construction
#
# This script loads the exported ONNX model and parses it into
# our custom Intermediate Representation (IR) — a plain Python
# dict that makes the graph easy to traverse and rewrite.
#
# WHY A CUSTOM IR?
# ONNX stores graphs as protobuf binary — awkward to traverse
# and mutate directly. We parse it once into a clean structure,
# run all optimization passes on that, then serialize back to ONNX.
#
# This is the same pattern used by production compilers:
#   ONNX (file format) → IR (in-memory workable graph) → passes → ONNX
#
# OUTPUT:
#   {
#     "nodes":        [{"name", "op", "inputs", "outputs", "attrs"}, ...],
#     "initializers": {name -> np.ndarray},   ← weights + constants
#     "inputs":       [graph input names],
#     "outputs":      [graph output names],
#   }
#
# Consumed by passes/fusion.py, passes/constant_fold.py, and future passes.

import onnx
import numpy as np
from onnx import numpy_helper


def load_ir(onnx_path: str) -> dict:
    """
    Parse ONNX model into a workable IR dict.

    Replaces the old load_graph() which returned {node_id: node_info}.
    The new structure is list-based for nodes (preserves topological order)
    and dict-based for initializers (O(1) lookup by name).
    """
    model = onnx.load(onnx_path)
    graph = model.graph

    # --- Nodes (topological order guaranteed by ONNX spec) ------------------
    nodes = []
    for i, node in enumerate(graph.node):
        nodes.append({
            "name":    node.name or f"node_{i}",
            "op":      node.op_type,
            "inputs":  list(node.input),
            "outputs": list(node.output),
            "attrs":   {a.name: a for a in node.attribute},
        })

    # --- Initializers: weights, biases, normalization scalars ---------------
    # These are compile-time constants — the constant folding pass needs them
    # as numpy arrays keyed by name.
    initializers = {
        t.name: numpy_helper.to_array(t)
        for t in graph.initializer
    }

    # --- Graph-level inputs / outputs ---------------------------------------
    # inputs[0] is typically the image tensor ('input'); the rest are weights
    # that are also listed as initializers (ONNX convention).
    inputs  = [inp.name for inp in graph.input]
    outputs = [out.name for out in graph.output]

    ir = {
        "nodes":        nodes,
        "initializers": initializers,
        "inputs":       inputs,
        "outputs":      outputs,
    }

    print(f"✅ Loaded {len(nodes)} nodes, {len(initializers)} initializers from {onnx_path}")
    return ir


# ---------------------------------------------------------------------------
# Legacy shim — keeps existing passes/fusion.py working without changes
# ---------------------------------------------------------------------------

def load_graph(onnx_path: str) -> dict:
    """
    Deprecated. Returns old {node_id: node_info} format.
    Use load_ir() for all new passes.
    """
    ir = load_ir(onnx_path)
    return {
        f"node_{i}": node
        for i, node in enumerate(ir["nodes"])
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def summarize(ir: dict):
    """Print a count of each op type. Accepts both old and new IR formats."""
    from collections import Counter

    # Handle both old dict-of-dicts and new list-of-dicts format
    if isinstance(ir, list) or "nodes" in ir:
        nodes = ir["nodes"] if isinstance(ir, dict) else ir
        op_counts = Counter(n["op"] for n in nodes)
    else:
        op_counts = Counter(v["op"] for v in ir.values())

    print("\n📊 Op type summary:")
    for op, count in op_counts.most_common():
        print(f"   {op:<20} {count}")


if __name__ == "__main__":
    ir = load_ir("models/resnet50.onnx")

    # Print first 10 nodes
    print("\n🔍 First 10 nodes:")
    for node in ir["nodes"][:10]:
        print(f"  {node['name']}: op={node['op']}")
        print(f"       inputs={node['inputs']}")
        print(f"       outputs={node['outputs']}")

    summarize(ir)

    print(f"\n📦 Initializers: {len(ir['initializers'])} tensors")
    print(f"   Sample keys: {list(ir['initializers'].keys())[:5]}")

    '''
    === Constant Folding Pass ===

✅ Loaded 124 nodes, 110 initializers from models/resnet50.onnx

Op summary (before folding)
  Op                    Count
  ----------------------------
  Conv                     53
  Relu                     49
  Add                      16
  Shape                     1
  MaxPool                   1
  ReduceMean                1
  Concat                    1
  Reshape                   1
  Gemm                      1
  ────────────────────────────
  Total                   124

[constant_fold] 124 nodes → 124 nodes  (0 folded)

Op summary (after folding)
  Op                    Count
  ----------------------------
  Conv                     53
  Relu                     49
  Add                      16
  Shape                     1
  MaxPool                   1
  ReduceMean                1
  Concat                    1
  Reshape                   1
  Gemm                      1
  ────────────────────────────
  Total                   124
    
    
    '''