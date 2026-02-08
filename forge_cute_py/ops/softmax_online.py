"""Online softmax op with backend registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

SoftmaxForwardImpl = Callable[[torch.Tensor, int], torch.Tensor]
SoftmaxBackwardImpl = Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@dataclass(frozen=True)
class _SoftmaxBackend:
    forward_impl: SoftmaxForwardImpl
    backward_impl: SoftmaxBackwardImpl | None = None


_softmax_online_backends: dict[str, _SoftmaxBackend] = {}
_active_softmax_online_backend = "ref"
_kernel_fwd_compile_cache: dict[tuple[object, int], object] = {}
_kernel_bwd_compile_cache: dict[tuple[object, int], object] = {}


def _normalize_dim(dim: int, ndim: int) -> int:
    if dim != -1:
        raise ValueError(f"softmax_online expects dim=-1 (row-wise) for 2D tensors, got {dim}")
    return -1


def _ensure_forward_inputs(x: torch.Tensor, out: torch.Tensor, dim: int) -> int:
    if x.dim() != 2:
        raise ValueError("Input must be 2D")
    if not x.is_cuda:
        raise ValueError("Tensor must be on CUDA device")
    if x.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"Unsupported dtype: {x.dtype}")
    if out.shape != x.shape:
        raise ValueError("Output shape must match input")
    if out.dtype != x.dtype:
        raise ValueError("Output dtype must match input dtype")
    if out.device != x.device:
        raise ValueError("Output device must match input device")
    return _normalize_dim(dim, x.ndim)


def _ensure_backward_inputs(dy: torch.Tensor, y: torch.Tensor, dx: torch.Tensor, dim: int) -> int:
    if dy.dim() != 2 or y.dim() != 2 or dx.dim() != 2:
        raise ValueError("Tensors must be 2D")
    if dy.shape != y.shape or y.shape != dx.shape:
        raise ValueError("All tensors must have same shape")
    if not dy.is_cuda or not y.is_cuda or not dx.is_cuda:
        raise ValueError("Tensors must be on CUDA")
    if dy.dtype != y.dtype or y.dtype != dx.dtype:
        raise ValueError("dy, y, and dx must have same dtype")
    if dy.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"Unsupported dtype: {dy.dtype}")
    return _normalize_dim(dim, dy.ndim)


def _reference_softmax_forward(x: torch.Tensor, dim: int) -> torch.Tensor:
    from forge_cute_py.ref import softmax_online as softmax_online_ref

    return softmax_online_ref(x, dim=dim)


def _reference_softmax_backward(dy: torch.Tensor, y: torch.Tensor, dim: int) -> torch.Tensor:
    dot_product = (dy * y).sum(dim=dim, keepdim=True)
    return y * (dy - dot_product)


def _kernel_forward(x: torch.Tensor, dim: int) -> torch.Tensor:
    del dim
    if x.shape[1] % 32 != 0:
        raise ValueError(f"Inner dimension N must be a multiple of 32, got {x.shape[1]}")

    import cutlass.cute as cute
    from cutlass import BFloat16, Float16, Float32
    from forge_cute_py.kernels.softmax_online import SoftmaxOnline

    dtype_map = {
        torch.float16: Float16,
        torch.float32: Float32,
        torch.bfloat16: BFloat16,
    }
    cute_dtype = dtype_map[x.dtype]
    compile_key = (cute_dtype, x.shape[1])

    if compile_key not in _kernel_fwd_compile_cache:
        m = cute.sym_int()
        n = x.shape[1]
        input_cute = cute.runtime.make_fake_compact_tensor(cute_dtype, (m, n), stride_order=(1, 0))
        output_cute = cute.runtime.make_fake_compact_tensor(cute_dtype, (m, n), stride_order=(1, 0))
        _kernel_fwd_compile_cache[compile_key] = cute.compile(
            SoftmaxOnline(cute_dtype, n),
            input_cute,
            output_cute,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    out = torch.empty_like(x)
    _kernel_fwd_compile_cache[compile_key](x, out)
    return out


def _kernel_backward(dy: torch.Tensor, y: torch.Tensor, dim: int) -> torch.Tensor:
    del dim
    if dy.shape[1] % 32 != 0:
        raise ValueError(f"Inner dimension N must be a multiple of 32, got {dy.shape[1]}")

    import cutlass.cute as cute
    from cutlass import BFloat16, Float16, Float32
    from forge_cute_py.kernels.softmax_online import SoftmaxOnlineBackward

    dtype_map = {
        torch.float16: Float16,
        torch.float32: Float32,
        torch.bfloat16: BFloat16,
    }
    cute_dtype = dtype_map[dy.dtype]
    compile_key = (cute_dtype, dy.shape[1])

    if compile_key not in _kernel_bwd_compile_cache:
        m = cute.sym_int()
        n = dy.shape[1]
        dy_cute = cute.runtime.make_fake_compact_tensor(cute_dtype, (m, n), stride_order=(1, 0))
        y_cute = cute.runtime.make_fake_compact_tensor(cute_dtype, (m, n), stride_order=(1, 0))
        dx_cute = cute.runtime.make_fake_compact_tensor(cute_dtype, (m, n), stride_order=(1, 0))
        _kernel_bwd_compile_cache[compile_key] = cute.compile(
            SoftmaxOnlineBackward(cute_dtype, n),
            dy_cute,
            y_cute,
            dx_cute,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    dx = torch.empty_like(dy)
    _kernel_bwd_compile_cache[compile_key](dy, y, dx)
    return dx


def register_softmax_online_backend(
    name: str,
    forward_impl: SoftmaxForwardImpl,
    backward_impl: SoftmaxBackwardImpl | None = None,
    *,
    overwrite: bool = False,
) -> None:
    if not name:
        raise ValueError("Backend name must be a non-empty string")
    if not callable(forward_impl):
        raise TypeError("forward_impl must be callable")
    if backward_impl is not None and not callable(backward_impl):
        raise TypeError("backward_impl must be callable when provided")
    if name in _softmax_online_backends and not overwrite:
        raise ValueError(f"Backend '{name}' already exists. Pass overwrite=True to replace it.")
    _softmax_online_backends[name] = _SoftmaxBackend(
        forward_impl=forward_impl, backward_impl=backward_impl
    )


def list_softmax_online_backends() -> tuple[str, ...]:
    return tuple(_softmax_online_backends.keys())


def get_softmax_online_backend() -> str:
    return _active_softmax_online_backend


def set_softmax_online_backend(name: str) -> None:
    global _active_softmax_online_backend
    if name not in _softmax_online_backends:
        choices = ", ".join(list_softmax_online_backends()) or "<none>"
        raise ValueError(f"Unknown softmax_online backend '{name}'. Available: {choices}")
    _active_softmax_online_backend = name


def _current_softmax_backend() -> _SoftmaxBackend:
    return _softmax_online_backends[_active_softmax_online_backend]


@torch.library.custom_op("forge_cute_py::_softmax_fwd", mutates_args={"out"})
def _softmax_fwd(x: torch.Tensor, out: torch.Tensor, dim: int = -1) -> None:
    """Softmax forward pass."""
    dim = _ensure_forward_inputs(x, out, dim)
    backend = _current_softmax_backend()
    result = backend.forward_impl(x, dim)
    if result.shape != x.shape:
        raise ValueError(
            f"Backend '{_active_softmax_online_backend}' returned invalid shape {result.shape}, "
            f"expected {x.shape}."
        )
    if result.dtype != x.dtype:
        raise ValueError(
            f"Backend '{_active_softmax_online_backend}' returned invalid dtype {result.dtype}, "
            f"expected {x.dtype}."
        )
    if result.device != x.device:
        raise ValueError(
            f"Backend '{_active_softmax_online_backend}' returned output on {result.device}, "
            f"expected {x.device}."
        )
    out.copy_(result)


_softmax_fwd.compile_cache = _kernel_fwd_compile_cache


def softmax_fwd(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Forward-only softmax (no autograd)."""
    out = torch.empty_like(x)
    _softmax_fwd(x, out, dim)
    return out


@torch.library.custom_op("forge_cute_py::_softmax_backward", mutates_args={"dx"})
def _softmax_backward(dy: torch.Tensor, y: torch.Tensor, dx: torch.Tensor, dim: int = -1) -> None:
    """Softmax backward pass."""
    dim = _ensure_backward_inputs(dy, y, dx, dim)
    backend = _current_softmax_backend()
    if backend.backward_impl is None:
        result = _reference_softmax_backward(dy, y, dim)
    else:
        result = backend.backward_impl(dy, y, dim)

    if result.shape != dy.shape:
        raise ValueError(
            f"Backend '{_active_softmax_online_backend}' returned invalid grad shape {result.shape}, "
            f"expected {dy.shape}."
        )
    if result.dtype != dy.dtype:
        raise ValueError(
            f"Backend '{_active_softmax_online_backend}' returned invalid grad dtype {result.dtype}, "
            f"expected {dy.dtype}."
        )
    if result.device != dy.device:
        raise ValueError(
            f"Backend '{_active_softmax_online_backend}' returned grad on {result.device}, "
            f"expected {dy.device}."
        )
    dx.copy_(result)


_softmax_backward.compile_cache = _kernel_bwd_compile_cache


def softmax_bwd(dy: torch.Tensor, y: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Backward-only softmax (no autograd)."""
    dx = torch.empty_like(dy)
    _softmax_backward(dy, y, dx, dim)
    return dx


class SoftmaxOnlineFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim=-1):
        y = softmax_fwd(x, dim)
        ctx.save_for_backward(y)
        ctx.dim = dim
        return y

    @staticmethod
    def backward(ctx, dy):
        (y,) = ctx.saved_tensors
        dx = softmax_bwd(dy, y, ctx.dim)
        return dx, None


def softmax_online(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Online softmax with automatic differentiation support."""
    return SoftmaxOnlineFunction.apply(x, dim)


register_softmax_online_backend(
    "ref",
    _reference_softmax_forward,
    _reference_softmax_backward,
    overwrite=True,
)
register_softmax_online_backend(
    "kernel",
    _kernel_forward,
    _kernel_backward,
    overwrite=True,
)
