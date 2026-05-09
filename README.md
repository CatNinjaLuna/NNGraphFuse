# benchmark/runner.py

#

# ROLE: Pipeline Step 5 — GPU Latency & Throughput Benchmark

#

# Runs the optimized IR through ONNX Runtime on CUDA and measures:

# - Latency p50 / p99 (ms)

# - Throughput (images/sec)

# - GPU memory usage (MB)

#

# PROVIDERS BENCHMARKED:

# FP32 — CUDAExecutionProvider, full precision

# FP16 — CUDAExecutionProvider, half precision

# TRT — TensorrtExecutionProvider (ORT's built-in TRT integration)

#

# METHODOLOGY:

# 50 warmup runs — allow GPU to reach steady-state clock, fill caches

# 200 timed runs — collect latency distribution

# p50/p99 — percentile latencies, robust to outliers

# throughput — 1000 / p50_ms \* batch_size (images/sec)

# GPU memory — peak allocated MB during inference

#

# USAGE:

# Called automatically by pipeline.py.

# Standalone: python -m benchmark.runner

import time
import numpy as np

# ---------------------------------------------------------------------------

# GPU memory helper

# ---------------------------------------------------------------------------

def \_gpu_memory_mb() -> float:
"""Return current GPU peak allocated memory in MB."""
try:
import torch
return torch.cuda.max_memory_allocated() / 1024 \*\* 2
except Exception:
return 0.0

def \_reset_gpu_memory():
try:
import torch
torch.cuda.reset_peak_memory_stats()
except Exception:
pass

# ---------------------------------------------------------------------------

# Single provider benchmark

# ---------------------------------------------------------------------------

def \_benchmark_provider(
onnx_path: str,
providers: list,
provider_options: list,
input_name: str,
dummy_input: np.ndarray,
warmup: int = 50,
runs: int = 200,
) -> dict:
"""
Run warmup + timed inference loop for one provider configuration.
Returns dict with p50, p99, throughput, gpu_mem_mb.
"""
import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3  # suppress warnings

    try:
        sess = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=providers,
            provider_options=provider_options,
        )
    except Exception as e:
        return {"error": str(e)}

    # Warmup
    for _ in range(warmup):
        sess.run(None, {input_name: dummy_input})

    # Timed runs
    _reset_gpu_memory()
    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {input_name: dummy_input})
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

    p50 = float(np.percentile(latencies, 50))
    p99 = float(np.percentile(latencies, 99))
    throughput = round(1000 / p50 * dummy_input.shape[0], 1)
    gpu_mem = round(_gpu_memory_mb(), 1)

    return {
        "p50":        round(p50, 2),
        "p99":        round(p99, 2),
        "throughput": throughput,
        "gpu_mem_mb": gpu_mem,
    }

# ---------------------------------------------------------------------------

# Main benchmark entry point

# ---------------------------------------------------------------------------

def run_benchmark(ir: dict, model_name: str = "model") -> dict:
"""
Benchmark the model under FP32, FP16, and TensorRT providers.

    Args:
        ir:         Optimized IR dict (used to resolve model path).
        model_name: "resnet50" or "mobilenetv2" — selects the ONNX file.

    Returns:
        {"rows": [...]} matching the summary table format in pipeline.py.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return _stub_result()

    available = ort.get_available_providers()
    if "CUDAExecutionProvider" not in available:
        print("  [benchmark] CUDAExecutionProvider not available — skipping")
        return _stub_result()

    path_map = {
        "resnet50":    "models/resnet50.onnx",
        "mobilenetv2": "models/mobilenetv2_normalized.onnx",
    }
    onnx_path = path_map.get(model_name)
    if onnx_path is None:
        return _stub_result()

    dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
    input_name = "input"
    rows = []

    # --- FP32 ---------------------------------------------------------------
    print(f"  [benchmark] {model_name} FP32 (CUDA)...", flush=True)
    r = _benchmark_provider(
        onnx_path=onnx_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        provider_options=[{}, {}],
        input_name=input_name,
        dummy_input=dummy,
    )
    rows.append(_format_row("FP32", r))
    _print_row("FP32", r)

    # --- FP16 ---------------------------------------------------------------
    print(f"  [benchmark] {model_name} FP16 (CUDA)...", flush=True)
    r = _benchmark_provider(
        onnx_path=onnx_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        provider_options=[{"cudnn_conv_algo_search": "DEFAULT"}, {}],
        input_name=input_name,
        dummy_input=dummy,
    )
    rows.append(_format_row("FP16", r))
    _print_row("FP16", r)

    # --- TRT ----------------------------------------------------------------
    if "TensorrtExecutionProvider" in available:
        print(f"  [benchmark] {model_name} TRT (TensorrtExecutionProvider)...", flush=True)
        trt_opts = {
            "trt_fp16_enable": "0",
            "trt_engine_cache_enable": "1",
            "trt_engine_cache_path": f"/tmp/trt_cache_{model_name}",
        }
        r = _benchmark_provider(
            onnx_path=onnx_path,
            providers=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
            provider_options=[trt_opts, {}, {}],
            input_name=input_name,
            dummy_input=dummy,
            warmup=10,
            runs=100,
        )
        rows.append(_format_row("TRT", r))
        _print_row("TRT", r)
    else:
        rows.append(_tbd_row("TRT"))

    return {"rows": rows}

# ---------------------------------------------------------------------------

# Formatting helpers

# ---------------------------------------------------------------------------

def \_format_row(precision: str, r: dict) -> dict:
if "error" in r:
return \_tbd_row(precision)
return {
"precision": precision,
"p50": f"{r['p50']:.2f}",
"p99": f"{r['p99']:.2f}",
"throughput": f"{r['throughput']:.0f}",
"gpu_mem": f"{r['gpu_mem_mb']:.0f} MB",
}

def \_print_row(precision: str, r: dict):
if "error" in r:
print(f" {precision} failed: {r['error']}", flush=True)
else:
print(f" p50={r['p50']:.2f}ms p99={r['p99']:.2f}ms "
f"throughput={r['throughput']:.0f} img/s "
f"mem={r['gpu_mem_mb']:.0f}MB", flush=True)

def \_tbd_row(precision: str) -> dict:
return {
"precision": precision,
"p50": "TBD",
"p99": "TBD",
"throughput": "TBD",
"gpu_mem": "TBD",
}

def \_stub_result() -> dict:
return {"rows": [\_tbd_row(p) for p in ["FP32", "FP16", "TRT"]]}

# ---------------------------------------------------------------------------

# Standalone entry point

# ---------------------------------------------------------------------------

if **name** == "**main**":
print("Running benchmark standalone on ResNet-50...\n")
result = run_benchmark({}, model_name="resnet50")
print("\nResults:")
for row in result["rows"]:
print(f" {row['precision']:<6} p50={row['p50']}ms p99={row['p99']}ms "
f"throughput={row['throughput']} img/s mem={row['gpu_mem']}")
