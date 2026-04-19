# NNGraphFuse

**Compiler-style graph optimization passes for neural network inference, benchmarked end-to-end on real image data using TensorRT.**

---

## What it does

NNGraphFuse loads a neural network's computation graph (ONNX), parses it into a custom IR, applies a sequence of optimization passes — operator fusion, constant folding, dead node elimination — then rebuilds a TensorRT engine and benchmarks latency improvement on real image inference.

The core idea: the same computation graph can run significantly faster if you rewrite it before handing it to the runtime. NNGraphFuse makes each optimization pass explicit and measurable.

---

## Why this exists

TensorRT is a graph compiler. It takes a model graph, rewrites it for efficiency, and generates GPU kernels. This project implements a simplified version of that pipeline from scratch — not to replace TensorRT, but to understand it from the inside.

Building the passes manually forces the question: _why does fusing Conv+Relu matter?_ The answer shows up in the benchmark numbers.

---

## Pipeline

```
JPEG Image Input
      ↓
Preprocessing  (resize → normalize → NCHW tensor)
      ↓
ONNX Graph Loader
      ↓
Custom IR  (dict-based node graph)
      ↓
Optimization Passes
  ├── Conv + Relu Fusion
  ├── Constant Folding
  └── Dead Node Elimination
      ↓
TensorRT Engine Builder  (FP32 / FP16 / INT8)
      ↓
GPU Inference
      ↓
Benchmark Report  (latency p50/p99, throughput, memory, node count)
```

---

## Optimization Passes

### Conv + Relu Fusion

ResNet-50 exports with BatchNorm already folded into Conv weights by the ONNX exporter — discovered by inspecting the IR directly. The dominant fusion target is therefore Conv + Relu.

The pass scans the IR for Conv nodes whose output feeds directly into a Relu, rewires the Conv to produce the Relu's output, and removes the Relu node. Of 49 Relu nodes in the graph, 33 are fused. The remaining 16 follow Add nodes (residual branch merges) rather than Conv nodes and are correctly skipped.

**Result: 124 nodes → 91 nodes (33 Relu nodes eliminated)**

```
Op type summary — before vs after fusion:

               Before    After
Conv              53       53
Relu              49       16    ← 33 fused into Conv
Add               16       16
Shape              1        1
MaxPool            1        1
ReduceMean         1        1
Concat             1        1
Reshape            1        1
Gemm               1        1
─────────────────────────────
Total            124       91
```

Without fusion, each op is a separate kernel launch — the intermediate tensor between Conv and Relu gets written to DRAM and read back. Fusing them means the Relu runs inside the same kernel, intermediate values stay in registers or L2 cache, and never touch DRAM.

### Constant Folding

_(in progress)_
Preprocessing constants (mean/std normalization values) are folded directly into the graph at compile time. Eliminates repeated CPU→GPU transfers at inference time.

### Dead Node Elimination

_(in progress)_
Removes identity ops, unused outputs, and no-op reshapes that accumulate during model export.

---

## Benchmark Results (ResNet-50, A100)

_(to be updated after university cluster runs)_

| Configuration      | Latency p50 | Throughput | GPU Memory | Node Count |
| ------------------ | ----------- | ---------- | ---------- | ---------- |
| Baseline TRT       | TBD         | TBD        | TBD        | 124        |
| + Conv+Relu Fusion | TBD         | TBD        | TBD        | 91         |
| + Constant Folding | TBD         | TBD        | TBD        | TBD        |
| + FP16             | TBD         | TBD        | TBD        | TBD        |

---

## Project Structure

```
NNGraphFuse/
├── models/                  # exported ONNX files (gitignored)
├── images/                  # test images (ImageNet validation subset)
├── graph/
│   ├── ir.py                # ONNX → custom IR parser
│   └── visualize.py         # graph diff visualization (before/after)
├── passes/
│   ├── fusion.py            # Conv+Relu fusion pass
│   ├── constant_fold.py     # constant folding pass (in progress)
│   └── dead_node.py         # dead node elimination pass (in progress)
├── benchmark/
│   └── runner.py            # latency/throughput/memory profiling
├── export_model.py          # torch → ONNX export
├── pipeline.py              # main entry point
└── requirements.txt
```

---

## Getting Started

```bash
# 1. Clone and set up environment
git clone https://github.com/CatNinjaLuna/NNGraphFuse
cd NNGraphFuse
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Export ResNet-50 to ONNX
python export_model.py

# 3. Load and inspect the graph IR
python -m graph.ir

# 4. Run Conv+Relu fusion pass
python -m passes.fusion

# 5. Run full optimization pipeline + benchmark (requires GPU)
python pipeline.py
```

---

## Requirements

- Python 3.10+
- `torch`, `torchvision`, `onnx`, `onnxscript`, `onnxruntime`
- `networkx`, `matplotlib`, `numpy`
- TensorRT 10+ (university GPU cluster for benchmark runs)
- CUDA 12+ for TRT engine build and inference

Local development (graph manipulation, IR, pass logic) runs CPU-only. GPU is only required for TRT engine build and benchmark runs.

---

## Key Findings

**BatchNorm folding:** PyTorch's ONNX exporter folds BatchNorm weights into Conv kernels at export time. The expected Conv+BN+Relu pattern does not appear in the graph — only Conv+Relu. Discovered by inspecting the IR op summary, not assumed upfront.

**Residual connections constrain fusion:** The 16 Relu nodes that follow Add nodes (residual branch merges) cannot be fused with Conv using the same pattern. The fusion pass detects this correctly by checking the producer op before fusing.

---

## Technical Background

This project mirrors the internal pipeline of a production graph compiler:

- **Clang** parses C++ → AST → LLVM IR → optimization passes → machine code
- **TensorRT** parses ONNX → internal IR → fusion/optimization passes → CUDA kernels
- **NNGraphFuse** parses ONNX → custom IR → explicit passes → TRT engine → benchmark

The passes implemented here correspond directly to what TRT's graph optimizer does internally — making the implicit explicit.

---

## Milestones

```
✅ Phase 1 — Project Setup
   ✅ Folder structure
   ✅ Virtual environment
   ✅ ResNet-50 exported to ONNX

✅ Phase 2 — Graph IR
   ✅ ONNX parser
   ✅ Op summary
   ✅ Discovered BN folding

✅ Phase 3 — Optimization Passes
   ✅ Conv + Relu Fusion (33 nodes eliminated)
   ⬜ Constant Folding
   ⬜ Dead Node Elimination

⬜ Phase 4 — Graph Visualization
   ⬜ Before/after graph diff (networkx)
   ⬜ Latency waterfall chart

⬜ Phase 5 — TensorRT Benchmark (university cluster)
   ⬜ Baseline TRT engine build
   ⬜ Benchmark each pass individually
   ⬜ Fill in benchmark table in README

⬜ Phase 6 — Polish
   ⬜ pipeline.py (runs all passes end to end)
   ⬜ requirements.txt finalized
   ⬜ GitHub README final
```

---

## Model

ResNet-50 (ImageNet pretrained via `torchvision`). Chosen because its repeating residual block structure produces clear Conv+Relu fusion opportunities that are easy to identify, measure, and explain.

---

## Concepts Covered

This project is designed to build working answers to the following inference engineering interview topics.

---

**Operator fusion: when does TRT fuse automatically vs. when do you force it?**

TRT automatically fuses patterns it has built-in support for — Conv+BN+Relu, element-wise ops after Conv, and select attention patterns. This happens internally during `builder.build_engine()`. You force fusion when TRT does not recognize the pattern — custom ops, unusual topology, or ops that cross plugin boundaries — either by writing a TensorRT plugin or by pre-optimizing the graph before TRT sees it. NNGraphFuse implements the latter: a pre-TRT fusion pass that rewires the graph so TRT receives a cleaner input.

---

**Horizontal vs. vertical fusion — Conv+BN+Relu as the canonical example**

Vertical fusion merges ops along the data flow path, one feeding into the next (Conv → Relu). This is what NNGraphFuse implements. Horizontal fusion merges ops that run in parallel on independent data — parallel Conv branches in Inception, or MoE expert layers. Conv+BN+Relu is the canonical vertical fusion example because it appears in nearly every CNN block and the memory bandwidth savings are directly measurable. In ResNet-50, BN was already folded into Conv weights by the ONNX exporter — discovered by inspecting the IR — making Conv+Relu the actual fusion target.

---

**Structured vs. unstructured pruning — Ampere/Hopper 2:4 sparsity**

Unstructured pruning zeros individual weights anywhere in the matrix. It achieves high sparsity but produces irregular memory access patterns that standard GPU hardware cannot exploit efficiently. Structured pruning removes entire filters or channels, producing a smaller dense matrix that hardware handles natively — easier to deploy but coarser grained. NVIDIA's 2:4 structured sparsity (Ampere+) sits between the two: exactly 2 of every 4 consecutive weights must be zero, producing regular enough structure for Sparse Tensor Cores to decompress and multiply in one step, delivering ~2x throughput at exactly 50% sparsity. The tradeoff: requires retraining or fine-tuning with the sparsity constraint enforced — you cannot prune a dense model post-hoc and expect the hardware speedup.

---

**The roofline model applied to fused vs. unfused graphs**

The roofline model bounds kernel performance by two limits: peak compute (FLOPS) and peak memory bandwidth (GB/s). Relu is almost purely memory bound — minimal arithmetic, full DRAM read and write. Unfused, it sits far left on the roofline: bandwidth limited, compute underutilized. Conv is more compute bound — high arithmetic intensity from multiply-accumulates. Fused, the Relu executes inside the Conv kernel: intermediate values stay in registers, no additional DRAM transaction occurs, and the fused kernel's arithmetic intensity shifts right toward the compute roof. Nsight Systems confirms this directly — fused kernels show higher arithmetic intensity and fewer memory transactions per inference pass.
