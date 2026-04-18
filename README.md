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

## Model

ResNet-50 (ImageNet pretrained via `torchvision`). Chosen because its repeating residual block structure produces clear Conv+Relu fusion opportunities that are easy to identify, measure, and explain.
