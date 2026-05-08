# pipeline.py
#
# ROLE: Main entry point — runs the full NNGraphFuse optimization pipeline.
#
# Chains all three passes in the correct order for each model and prints
# a unified summary report. GPU benchmark rows are stubbed — to be filled
# after A100 cluster runs via benchmark/runner.py.
#
# PASS ORDER (matters):
#   ResNet-50    : fusion → dead_node
#   MobileNetV2  : constant_fold → dead_node
#
#   Fusion before dead_node: fusion rewires Conv+Relu pairs; dead_node then
#   confirms no orphans remain after rewiring.
#
#   Constant fold before dead_node: folding absorbs Constant nodes and may
#   orphan their consumers; dead_node cleans up any newly unreachable nodes.
#
# IR FORMAT NOTE:
#   fusion.py uses the legacy dict-based IR (load_graph / ir.items()).
#   All other passes use the new list-based IR (load_ir).
#   _run_fusion_on_new_ir() bridges the two formats inside this file,
#   keeping fusion.py untouched for backward compatibility.
#
# USAGE:
#   python pipeline.py                    # run both models, full report
#   python pipeline.py --model resnet     # ResNet-50 only
#   python pipeline.py --model mobilenet  # MobileNetV2 only

import copy
import argparse
import time
from collections import Counter

from graph.ir import load_ir
from passes.fusion import apply_fusion
from passes.constant_fold import run_constant_folding, remove_unused_initializers
from passes.dead_node import run_dead_node_elimination
from benchmark.runner import run_benchmark


# ---------------------------------------------------------------------------
# IR format bridge — fusion.py uses legacy dict IR
# ---------------------------------------------------------------------------

def _new_ir_to_dict(ir: dict) -> dict:
    """Convert list-based IR nodes to legacy dict format for fusion.py."""
    return {f"node_{i}": node for i, node in enumerate(ir["nodes"])}


def _dict_to_new_ir(legacy_dict: dict, ir_template: dict) -> dict:
    """Convert legacy dict IR back to list-based IR after fusion."""
    return {
        **ir_template,
        "nodes": list(legacy_dict.values()),
    }


def _run_fusion_on_new_ir(ir: dict) -> tuple[dict, int]:
    """
    Run fusion.py's apply_fusion() on a new list-based IR.
    Converts to legacy dict format, runs the pass, converts back.
    """
    legacy = _new_ir_to_dict(ir)
    legacy_fused, count = apply_fusion(legacy)
    return _dict_to_new_ir(legacy_fused, ir), count


# ---------------------------------------------------------------------------
# Per-pass result tracking
# ---------------------------------------------------------------------------

def _node_count(ir: dict) -> int:
    return len(ir["nodes"])


def _op_counts(ir: dict) -> Counter:
    return Counter(n["op"] for n in ir["nodes"])


# ---------------------------------------------------------------------------
# ResNet-50 pipeline: fusion → dead_node
# ---------------------------------------------------------------------------

def run_resnet_pipeline(model_path: str) -> dict:
    """
    Run the full optimization pipeline on ResNet-50.
    Returns a result dict with per-pass node counts and benchmark data.
    """
    print("\n" + "=" * 60)
    print("  ResNet-50 Pipeline")
    print("=" * 60)

    ir = load_ir(model_path)
    results = {"model": "ResNet-50", "passes": []}

    # Snapshot baseline
    baseline_count = _node_count(ir)
    results["baseline"] = baseline_count
    print(f"\n  Loaded: {baseline_count} nodes")

    # --- Pass 1: Conv + Relu Fusion -----------------------------------------
    print("\n  [1/2] Conv + Relu Fusion...")
    ir_before = copy.deepcopy(ir)
    t0 = time.perf_counter()
    ir, fused = _run_fusion_on_new_ir(ir)
    elapsed = time.perf_counter() - t0

    results["passes"].append({
        "name":     "Conv+Relu Fusion",
        "before":   _node_count(ir_before),
        "after":    _node_count(ir),
        "removed":  fused,
        "time_ms":  round(elapsed * 1000, 2),
    })
    print(f"  ✅ {_node_count(ir_before)} → {_node_count(ir)} nodes  ({fused} fused)  [{elapsed*1000:.1f}ms]")

    # --- Pass 2: Dead Node Elimination --------------------------------------
    print("\n  [2/2] Dead Node Elimination...")
    ir_before = copy.deepcopy(ir)
    t0 = time.perf_counter()
    ir = run_dead_node_elimination(ir)
    elapsed = time.perf_counter() - t0

    removed = _node_count(ir_before) - _node_count(ir)
    results["passes"].append({
        "name":     "Dead Node Elimination",
        "before":   _node_count(ir_before),
        "after":    _node_count(ir),
        "removed":  removed,
        "time_ms":  round(elapsed * 1000, 2),
    })
    print(f"  ✅ {_node_count(ir_before)} → {_node_count(ir)} nodes  ({removed} removed)  [{elapsed*1000:.1f}ms]")

    results["final_count"] = _node_count(ir)
    results["final_ir"] = ir

    # --- Benchmark (GPU — stubbed until A100 runs) --------------------------
    results["benchmark"] = run_benchmark(ir, model_name="resnet50")

    return results


# ---------------------------------------------------------------------------
# MobileNetV2 pipeline: constant_fold → dead_node
# ---------------------------------------------------------------------------

def run_mobilenet_pipeline(model_path: str) -> dict:
    """
    Run the full optimization pipeline on MobileNetV2.
    Returns a result dict with per-pass node counts and benchmark data.
    """
    print("\n" + "=" * 60)
    print("  MobileNetV2 Pipeline")
    print("=" * 60)

    ir = load_ir(model_path)
    results = {"model": "MobileNetV2", "passes": []}

    baseline_count = _node_count(ir)
    results["baseline"] = baseline_count
    print(f"\n  Loaded: {baseline_count} nodes")

    # --- Pass 1: Constant Folding -------------------------------------------
    print("\n  [1/2] Constant Folding...")
    ir_before = copy.deepcopy(ir)
    t0 = time.perf_counter()
    ir = run_constant_folding(ir)
    ir = remove_unused_initializers(ir)
    elapsed = time.perf_counter() - t0

    removed = _node_count(ir_before) - _node_count(ir)
    results["passes"].append({
        "name":     "Constant Folding",
        "before":   _node_count(ir_before),
        "after":    _node_count(ir),
        "removed":  removed,
        "time_ms":  round(elapsed * 1000, 2),
    })
    print(f"  ✅ {_node_count(ir_before)} → {_node_count(ir)} nodes  ({removed} folded)  [{elapsed*1000:.1f}ms]")

    # --- Pass 2: Dead Node Elimination --------------------------------------
    print("\n  [2/2] Dead Node Elimination...")
    ir_before = copy.deepcopy(ir)
    t0 = time.perf_counter()
    ir = run_dead_node_elimination(ir)
    elapsed = time.perf_counter() - t0

    removed = _node_count(ir_before) - _node_count(ir)
    results["passes"].append({
        "name":     "Dead Node Elimination",
        "before":   _node_count(ir_before),
        "after":    _node_count(ir),
        "removed":  removed,
        "time_ms":  round(elapsed * 1000, 2),
    })
    print(f"  ✅ {_node_count(ir_before)} → {_node_count(ir)} nodes  ({removed} removed)  [{elapsed*1000:.1f}ms]")

    results["final_count"] = _node_count(ir)
    results["final_ir"] = ir

    # --- Benchmark (GPU — stubbed until A100 runs) --------------------------
    results["benchmark"] = run_benchmark(ir, model_name="mobilenetv2")

    return results


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def print_summary(all_results: list[dict]):
    """Print a unified pass + benchmark summary table across all models."""

    print("\n\n" + "=" * 60)
    print("  NNGraphFuse — Optimization Summary")
    print("=" * 60)

    # --- Pass results table -------------------------------------------------
    print(f"\n  {'Model':<14} {'Pass':<26} {'Before':>6}  {'After':>6}  {'Removed':>8}  {'Time':>8}")
    print(f"  {'-'*14} {'-'*26} {'-'*6}  {'-'*6}  {'-'*8}  {'-'*8}")

    for r in all_results:
        model = r["model"]
        for i, p in enumerate(r["passes"]):
            label = model if i == 0 else ""
            print(f"  {label:<14} {p['name']:<26} {p['before']:>6}  {p['after']:>6}  {p['removed']:>8}  {p['time_ms']:>6.1f}ms")
        # Total row
        total_removed = r["baseline"] - r["final_count"]
        pct = total_removed / r["baseline"] * 100
        print(f"  {'':14} {'TOTAL':<26} {r['baseline']:>6}  {r['final_count']:>6}  {total_removed:>7}  ({pct:.0f}%)")
        print()

    # --- Benchmark table ----------------------------------------------------
    print(f"\n  {'Model':<14} {'Precision':<10} {'p50 (ms)':>10}  {'p99 (ms)':>10}  {'Throughput':>12}  {'GPU Mem':>10}")
    print(f"  {'-'*14} {'-'*10} {'-'*10}  {'-'*10}  {'-'*12}  {'-'*10}")

    for r in all_results:
        bm = r.get("benchmark", {})
        for row in bm.get("rows", []):
            print(f"  {r['model']:<14} {row['precision']:<10} {row['p50']:>10}  {row['p99']:>10}  {row['throughput']:>12}  {row['gpu_mem']:>10}")
        if not bm.get("rows"):
            print(f"  {r['model']:<14} {'—':<10} {'TBD':>10}  {'TBD':>10}  {'TBD':>12}  {'TBD':>10}")

    print(f"\n  Benchmark rows marked TBD require GPU — run on A100 cluster:")
    print(f"  $ python pipeline.py  (with TensorRT + CUDA available)\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NNGraphFuse optimization pipeline")
    parser.add_argument(
        "--model",
        choices=["resnet", "mobilenet", "both"],
        default="both",
        help="Which model to run (default: both)",
    )
    args = parser.parse_args()

    all_results = []

    if args.model in ("resnet", "both"):
        all_results.append(run_resnet_pipeline("models/resnet50.onnx"))

    if args.model in ("mobilenet", "both"):
        all_results.append(run_mobilenet_pipeline("models/mobilenetv2_normalized.onnx"))

    print_summary(all_results)