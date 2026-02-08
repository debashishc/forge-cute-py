"""Run forge-cute-py benchmarks on Modal.

WARNING: Running this script incurs Modal GPU costs. Review the code and
verify timeout/GPU settings before running. Start with `--suite smoke` to
validate your setup. You are responsible for any credits consumed.

References:
- Modal GPU docs: https://modal.com/docs/guide/gpu
- Strict GPU matching (! suffix): https://modal.com/blog/gpu-health
- Inspired by: https://github.com/gpu-mode/kernelbot
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal

from bench.runner import run_benchmarks

APP_NAME = "forge-cute-py-bench"

app = modal.App(APP_NAME)

# Use pre-built CUDA image for faster builds (inspired by kernelbot)
# https://hub.docker.com/r/nvidia/cuda
CUDA_VERSION = "13.1.0"
CUDA_TAG = f"{CUDA_VERSION}-devel-ubuntu24.04"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.13")
    .run_commands("ln -sf $(which python) /usr/local/bin/python3")
    .apt_install(
        "git",
        "gcc-13",
        "g++-13",
        "clang-18",
    )
    .pip_install("uv")
    .uv_pip_install(
        "ninja~=1.11",
        "wheel~=0.45",
        "packaging~=25.0",
        "numpy~=2.3",
        "pyyaml",
    )
    .uv_pip_install(
        "torch==2.9.1",
        index_url="https://download.pytorch.org/whl/cu130",
    )
    .uv_pip_install(
        "nvidia-cutlass-dsl==4.3.5",
        "cuda-python[all]==13.0",
        "apache-tvm-ffi",
    )
    # Only copy the Python packages we need (avoids .env, credentials, etc.)
    .add_local_python_source("forge_cute_py", "bench")
    # Copy the benchmark suite config (not included by add_local_python_source)
    .add_local_file(
        str(Path(__file__).parent / "suites.yaml"),
        remote_path="/root/bench/suites.yaml",
    )
)


@app.function(gpu="B200", image=image, timeout=60 * 60, serialized=True)
def bench_runner(suite: str, out_path: str | None, op: str | None) -> str:
    """Run benchmarks on B200 GPU."""
    return run_benchmarks(suite, out_path, op)


@app.local_entrypoint()
def main(
    suite: str = "smoke",
    op: str | None = None,
    out: str | None = None,
) -> None:
    """Run forge-cute-py benchmarks on Modal B200.

    Args:
        suite: Benchmark suite name (default: smoke)
        op: Filter cases by op name
        out: Save JSON results to this local path
    """
    print(
        f"\n⚠️  WARNING: This will incur Modal GPU costs (B200, suite={suite}).\n"
        "   Review bench/modal_bench.py and verify settings before proceeding.\n"
        "   You are responsible for any credits consumed.\n",
        file=sys.stderr,
    )

    if out:
        remote_out = "/tmp/modal_results.json"
        output = bench_runner.remote(suite, remote_out, op)
        Path(out).write_text(output)
        print(output)
        return
    output = bench_runner.remote(suite, None, op)
    print(output)
