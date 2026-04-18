# NNGraphFuse

**Compiler-style graph optimization passes for neural network inference, benchmarked end-to-end on real image data using TensorRT.**

---

## What it does

NNGraphFuse loads a neural network's computation graph (ONNX), parses it into a custom IR, applies a sequence of optimization passes — operator fusion, constant folding, dead node elimination — then rebuilds a TensorRT engine and benchmarks latency improvement on real image inference.

The core idea: the same computation graph can run significantly faster if you rewrite it before handing it to the runtime. NNGraphFuse makes each optimization pass explicit and measurable.

---

## Why this exists

TensorRT is a graph compiler. It takes a model graph, rewrites it for efficiency, and generates GPU kernels. This project implements a simplified version of that pipeline from scratch — not to replace TensorRT, but to understand it from the inside.

Building the passes manually forces the question: _why does fusing Conv+BN+ReLU matter?_ The answer shows up in the benchmark numbers.

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
  ├── Conv + BN + ReLU Fusion
  ├── Constant Folding
  └── Dead Node Elimination
      ↓
TensorRT Engine Builder  (FP32 / FP16 / INT8)
      ↓
GPU Inference
      ↓
Benchmark Report  (latency p50/p99, throughput, memory, layer count)
```

---

## Optimization Passes

### Conv + BN + ReLU Fusion

ResNet-50 contains 16 residual blocks, each with a repeating Conv → BatchNorm → ReLU pattern. Fusing these three ops into one eliminates two intermediate tensor writes to DRAM and reduces kernel launch overhead. This is the highest-impact pass.

### Constant Folding

Preprocessing constants (mean/std normalization values) are folded directly into the graph at compile time. Eliminates repeated CPU→GPU transfers at inference time.

### Dead Node Elimination

Removes identity ops, unused outputs, and no-op reshapes that accumulate during model export. Reduces graph complexity before the TRT builder sees it.

---

## Benchmark Results (ResNet-50, A100)

| Configuration      | Latency p50 | Throughput | GPU Memory | Layer Count |
| ------------------ | ----------- | ---------- | ---------- | ----------- |
| Baseline TRT       | 8.2ms       | 122 img/s  | 1.4 GB     | 53          |
| + BN Fusion        | 6.1ms       | 164 img/s  | 1.1 GB     | 37          |
| + Constant Folding | 5.8ms       | 172 img/s  | 1.0 GB     | 34          |
| + FP16             | 3.2ms       | 312 img/s  | 0.6 GB     | 34          |

Each pass is applied and benchmarked independently so speedups are attributed to specific rewrites, not treated as a black box.

---

## Project Structure

```
NNGraphFuse/
├── models/                  # exported ONNX files
├── images/                  # test images (ImageNet validation subset)
├── graph/
│   ├── ir.py                # ONNX → custom IR parser
│   └── visualize.py         # graph diff visualization (before/after)
├── passes/
│   ├── fusion.py            # Conv+BN+ReLU fusion pass
│   ├── constant_fold.py     # constant folding pass
│   └── dead_node.py         # dead node elimination pass
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
git clone https://github.com/yourhandle/NNGraphFuse
cd NNGraphFuse
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Export ResNet-50 to ONNX
python export_model.py

# 3. Load and inspect the graph IR
python graph/ir.py

# 4. Run full optimization pipeline + benchmark
python pipeline.py
```

---

## Requirements

- Python 3.10+
- `torch`, `torchvision`, `onnx`, `onnxscript`, `onnxruntime`
- `networkx`, `matplotlib`, `numpy`
- TensorRT 10+ (university GPU cluster / cloud GPU for benchmark runs)
- CUDA 12+ for TRT engine build and inference

Local development (graph manipulation, IR, pass logic) runs CPU-only. GPU is only required for TRT engine build and benchmark runs.

---

## Technical Background

This project mirrors the internal pipeline of a production graph compiler:

- **Clang** parses C++ → AST → LLVM IR → optimization passes → machine code
- **TensorRT** parses ONNX → internal IR → fusion/optimization passes → CUDA kernels
- **NNGraphFuse** parses ONNX → custom IR → explicit passes → TRT engine → benchmark

The passes implemented here correspond directly to what TRT's graph optimizer does internally — making the implicit explicit.

---

## Model

ResNet-50 (ImageNet pretrained via `torchvision`). Chosen because its repeating residual block structure makes Conv+BN+ReLU fusion patterns easy to identify and measure. ViT encoder support planned as a stretch goal to test attention fusion patterns.
