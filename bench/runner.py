"""Benchmark runner logic for Modal execution."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_benchmarks(suite: str, out_path: str | None, op: str | None) -> str:
    """Run benchmarks inside the Modal container."""
    cmd = ["python", "-m", "bench.run", "--suite", suite]
    if op:
        cmd.extend(["--op", op])
    if out_path:
        cmd.extend(["--out", out_path])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"Benchmark failed with exit code {result.returncode}:\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
    if out_path:
        return Path(out_path).read_text()
    return result.stdout
