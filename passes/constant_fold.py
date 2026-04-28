"""
passes/constant_fold.py
-----------------------
Constant folding pass for NNGraphFuse.

Scans the IR for nodes where ALL inputs are compile-time constants
(ONNX initializers or nodes already folded to Constant), evaluates
them with numpy, and replaces the node with a single Constant node.

Targeted patterns in ResNet-50 preprocessing:
  Sub(input, mean_const)   → Constant node (mean folded in)
  Div(result, std_const)   → Constant node (std folded in)

More generally, any op whose inputs are all known at compile time
is eligible: Add, Mul, Div, Sub, Sqrt, Reshape, Transpose, etc.

Usage (standalone):
    python -m passes.constant_fold

Usage (as pass in pipeline):
    from passes.constant_fold import run_constant_folding
    optimized_ir = run_constant_folding(ir)
"""

import numpy as np
import onnx
from onnx import numpy_helper, TensorProto
from graph.ir import load_ir  # your existing ONNX → IR loader


# ---------------------------------------------------------------------------
# Numpy dispatch: evaluate a single ONNX op given numpy input arrays
# ---------------------------------------------------------------------------
# Step 3 — Numpy dispatch table
# Translates ONNX op semantics into numpy so we can evaluate them at compile time.
# e.g. Sub(input, mean_tensor) → inputs[0] - inputs[1]
#      Div(result, std_tensor) → inputs[0] / inputs[1]
#      Transpose(weight, perm) → np.transpose(...)
# If an op isn't in the table (Conv, Gemm — too expensive to precompute),
# returns None and the node is kept. Table is intentionally narrow:
# only fold cheap scalar/metadata ops, not compute-heavy ones.

def _evaluate_op(op_type: str, inputs: list[np.ndarray], attrs: dict) -> np.ndarray | None:
    """
    Evaluate a constant-foldable op using numpy.
    Returns the output array, or None if the op is not supported.

    Extend this table as you add more foldable op types.
    """
    try:
        if op_type == "Add":
            return inputs[0] + inputs[1]
        elif op_type == "Sub":
            return inputs[0] - inputs[1]
        elif op_type == "Mul":
            return inputs[0] * inputs[1]
        elif op_type == "Div":
            return inputs[0] / inputs[1]
        elif op_type == "Sqrt":
            return np.sqrt(inputs[0])
        elif op_type == "Neg":
            return -inputs[0]
        elif op_type == "Abs":
            return np.abs(inputs[0])
        elif op_type == "Exp":
            return np.exp(inputs[0])
        elif op_type == "Log":
            return np.log(inputs[0])
        elif op_type == "Relu":
            return np.maximum(0, inputs[0])
        elif op_type == "Transpose":
            perm = attrs.get("perm", None)
            return np.transpose(inputs[0], axes=perm)
        elif op_type == "Reshape":
            # inputs[1] is the shape tensor
            new_shape = inputs[1].astype(np.int64)
            return inputs[0].reshape(new_shape)
        elif op_type == "Unsqueeze":
            axes = attrs.get("axes", [])
            result = inputs[0]
            for ax in sorted(axes):
                result = np.expand_dims(result, axis=ax)
            return result
        elif op_type == "Squeeze":
            axes = attrs.get("axes", None)
            if axes:
                result = inputs[0]
                for ax in sorted(axes, reverse=True):
                    result = np.squeeze(result, axis=ax)
                return result
            return np.squeeze(inputs[0])
        elif op_type == "Concat":
            axis = attrs.get("axis", 0)
            return np.concatenate(inputs, axis=axis)
        elif op_type == "Gather":
            axis = attrs.get("axis", 0)
            return np.take(inputs[0], inputs[1].astype(np.int64), axis=axis)
        elif op_type == "Cast":
            to = attrs.get("to", TensorProto.FLOAT)
            dtype_map = {
                TensorProto.FLOAT: np.float32,
                TensorProto.DOUBLE: np.float64,
                TensorProto.INT32: np.int32,
                TensorProto.INT64: np.int64,
                TensorProto.BOOL: np.bool_,
            }
            dtype = dtype_map.get(to, np.float32)
            return inputs[0].astype(dtype)
        else:
            return None  # op not in dispatch table; skip
    except Exception as e:
        print(f"    [fold] Failed to evaluate {op_type}: {e}")
        return None


# ---------------------------------------------------------------------------
# Main pass
# ---------------------------------------------------------------------------

def run_constant_folding(ir: dict) -> dict:
    """
    Run one full constant-folding sweep over the IR.

    IR structure (from graph/ir.py):
      ir["nodes"]        : list of node dicts
      ir["initializers"] : dict { name -> np.ndarray }  (compile-time constants)
      ir["inputs"]       : list of graph input names
      ir["outputs"]      : list of graph output names

    Each node dict:
      {
        "name":    str,
        "op":      str,          # e.g. "Conv", "Relu", "Add"
        "inputs":  list[str],    # names of input tensors
        "outputs": list[str],    # names of output tensors
        "attrs":   dict,         # ONNX attributes
      }

    Returns a new IR dict with constant nodes replaced by Constant entries
    in the initializer table and the folded nodes removed from the node list.
    """

    # --- Step 0: What "constant" means here ----------------------------------
    # A tensor is a compile-time constant if it's an ONNX initializer: weights,
    # biases, or scalars like ImageNet mean [0.485, 0.456, 0.406] and std
    # [0.229, 0.224, 0.225]. These live in model.graph.initializer in the
    # protobuf. We seed const_values with all of them before scanning any nodes.

    # --- Step 1: Build a value table of everything known at compile time ----
    # Every initializer name maps to its numpy array — the "what do we know at
    # compile time" starting state. In ResNet-50 this is ~270 weight tensors
    # plus the preprocessing scalars.
    # Keys: tensor name  →  Value: np.ndarray
    const_values: dict[str, np.ndarray] = dict(ir["initializers"])

    nodes_before = len(ir["nodes"])
    folded_count = 0
    kept_nodes = []

    # --- Step 1b: Absorb inline Constant nodes into const_values --------------
    # The ONNX exporter sometimes emits constants as Constant nodes rather than
    # initializers. Harvest them first so the main scan can fold their consumers.
    remaining_nodes = []
    for node in ir["nodes"]:
       if node["op"] == "Constant":
          attr = node["attrs"].get("value")
          if attr is not None:
                from onnx import numpy_helper
                arr = numpy_helper.to_array(attr.t)
                const_values[node["outputs"][0]] = arr
                folded_count += 1   
          # Don't append to remaining_nodes — Constant nodes are now absorbed
       else:
          remaining_nodes.append(node)

    # Replace node list with Constant nodes removed before the main scan
    ir = {**ir, "nodes": remaining_nodes}


    # --- Step 2: Single forward scan (topological order) --------------------
    # For each node: are ALL inputs already in const_values?
    # If yes → evaluate with numpy, register output into const_values, drop node.
    # If no  → keep node; it runs at inference time.
    # Propagates forward: a folded node's output enters const_values and may
    # unlock the next node downstream. ONNX IR guarantees topological order.
    for node in ir["nodes"]:
        op = node["op"]
        inputs = node["inputs"]
        outputs = node["outputs"]
        attrs = node.get("attrs", {})

        # Check if ALL inputs to this node are in our constant table
        all_const = all(
            inp == "" or inp in const_values   # "" = optional missing input
            for inp in inputs
        )

        if not all_const:
            # At least one input is a runtime value; cannot fold.
            kept_nodes.append(node)
            continue

        # Gather numpy arrays for each input (skip empty optional inputs)
        input_arrays = [
            const_values[inp]
            for inp in inputs
            if inp != ""
        ]

        # Evaluate the op in numpy
        result = _evaluate_op(op, input_arrays, attrs)

        if result is None:
            # Op not supported in dispatch table; keep node as-is
            kept_nodes.append(node)
            continue

        # --- Step 4: Replace node with constant ------------------------------
        # Evaluation succeeded. Register the output tensor as a new compile-time
        # constant so downstream nodes can be folded in the same sweep.
        # The node itself is NOT appended to kept_nodes — it's gone from the graph.
        # At TRT engine build time, TRT never sees this op at all.
        output_name = outputs[0]
        const_values[output_name] = result
        folded_count += 1

        print(f"  [fold] {op:12s}  '{node['name']}'  →  constant '{output_name}'  shape={result.shape}")

    # --- Step 3: Rebuild IR with folded nodes removed -----------------------
    # Update initializers to include newly folded constants
    new_initializers = dict(ir["initializers"])
    for name, arr in const_values.items():
        if name not in ir["initializers"]:
            new_initializers[name] = arr

    new_ir = {
        **ir,
        "nodes": kept_nodes,
        "initializers": new_initializers,
    }

    nodes_after = len(kept_nodes)
    print(f"\n[constant_fold] {nodes_before} nodes → {nodes_after} nodes  ({folded_count} folded)")

    return new_ir


# ---------------------------------------------------------------------------
# Dead initializer cleanup (bonus: run after folding)
# ---------------------------------------------------------------------------
# Step 5 — Unused initializer cleanup
# After folding, some original initializers (e.g. the raw mean/std scalars) are
# no longer referenced by any remaining node — their consumers were just folded
# away. Sweep the remaining node inputs and drop anything unreferenced.
# This is a lightweight precursor to the full dead-node elimination pass.

def remove_unused_initializers(ir: dict) -> dict:
    """
    After constant folding, some initializers may no longer be referenced
    by any remaining node. Remove them to reduce engine build overhead.

    This is a lightweight precursor to the full dead-node-elimination pass.
    """
    referenced = set()
    for node in ir["nodes"]:
        for inp in node["inputs"]:
            referenced.add(inp)

    old_count = len(ir["initializers"])
    new_inits = {k: v for k, v in ir["initializers"].items() if k in referenced}
    removed = old_count - len(new_inits)

    if removed:
        print(f"[cleanup] Removed {removed} unused initializer(s) after folding")

    return {**ir, "initializers": new_inits}


# ---------------------------------------------------------------------------
# Op-type summary helper (mirrors what ir.py already prints for fusion)
# ---------------------------------------------------------------------------

def print_op_summary(ir: dict, label: str = ""):
    from collections import Counter
    counts = Counter(n["op"] for n in ir["nodes"])
    print(f"\nOp summary {label}")
    print(f"  {'Op':<20} {'Count':>6}")
    print(f"  {'-'*28}")
    for op, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {op:<20} {count:>6}")
    print(f"  {'─'*28}")
    print(f"  {'Total':<20} {sum(counts.values()):>6}")


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Constant Folding Pass ===\n")

    # Load IR (post-fusion if you've already run fusion.py)
    #ir = load_ir("models/resnet50.onnx")
    ir = load_ir("models/mobilenetv2_normalized.onnx")

    print_op_summary(ir, label="(before folding)")

    ir_folded = run_constant_folding(ir)
    ir_folded = remove_unused_initializers(ir_folded)

    print_op_summary(ir_folded, label="(after folding)")

    # Optionally chain: save folded IR for next pass
    # from passes.dead_node import run_dead_node_elimination
    # ir_final = run_dead_node_elimination(ir_folded)


    '''
    (venv) (base) carolina1650@Carolinas-MacBook-Pro NNGraphFuse % python -m passes.constant_fold
=== Constant Folding Pass ===

✅ Loaded 172 nodes, 108 initializers from models/mobilenetv2_normalized.onnx

Op summary (before folding)
  Op                    Count
  ----------------------------
  Constant                 70
  Conv                     52
  Clip                     35
  Add                      10
  Sub                       1
  Div                       1
  GlobalAveragePool         1
  Flatten                   1
  Gemm                      1
  ────────────────────────────
  Total                   172

[constant_fold] 172 nodes → 172 nodes  (0 folded)

Op summary (after folding)
  Op                    Count
  ----------------------------
  Constant                 70
  Conv                     52
  Clip                     35
  Add                      10
  Sub                       1
  Div                       1
  GlobalAveragePool         1
  Flatten                   1
  Gemm                      1
  ────────────────────────────
  Total                   172
    '''

'''
after adding 1-b pre-sweep:
(venv) (base) carolina1650@Carolinas-MacBook-Pro NNGraphFuse % python -m passes.constant_fold
=== Constant Folding Pass ===

✅ Loaded 172 nodes, 108 initializers from models/mobilenetv2_normalized.onnx

Op summary (before folding)
  Op                    Count
  ----------------------------
  Constant                 70
  Conv                     52
  Clip                     35
  Add                      10
  Sub                       1
  Div                       1
  GlobalAveragePool         1
  Flatten                   1
  Gemm                      1
  ────────────────────────────
  Total                   172

[constant_fold] 172 nodes → 102 nodes  (70 folded)

Op summary (after folding)
  Op                    Count
  ----------------------------
  Conv                     52
  Clip                     35
  Add                      10
  Sub                       1
  Div                       1
  GlobalAveragePool         1
  Flatten                   1
  Gemm                      1
  ────────────────────────────


'''