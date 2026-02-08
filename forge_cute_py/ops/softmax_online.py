from __future__ import annotations

import importlib
import os
from types import ModuleType
from typing import Callable

import torch

SoftmaxForwardImpl = Callable[[torch.Tensor, int], torch.Tensor]
SoftmaxBackwardImpl = Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_SUPPORTED_IMPL_MODES = {"auto", "ref", "kernel"}


def _normalize_dim(dim: int, ndim: int) -> int:
    dim = dim if dim >= 0 else ndim + dim
    if dim not in (0, 1):
        raise ValueError(f"softmax_online expects dim in {{-1, 0, 1}} for 2D tensors, got {dim}")
    return dim


def _validate_impl_mode() -> str:
    impl = os.getenv("FORGE_SOFTMAX_IMPL", "auto").strip().lower()
    if impl not in _SUPPORTED_IMPL_MODES:
        choices = ", ".join(sorted(_SUPPORTED_IMPL_MODES))
        raise ValueError(f"FORGE_SOFTMAX_IMPL must be one of {{{choices}}}, got '{impl}'")
    return impl


def _load_kernel_module() -> tuple[ModuleType | None, str | None]:
    module_name = "forge_cute_py.kernels.softmax_online"
    try:
        return importlib.import_module(module_name), None
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            return None, f"module '{module_name}' was not found"
        if exc.name is None:
            return None, f"failed importing '{module_name}': {exc}"
        return None, f"dependency '{exc.name}' missing while importing '{module_name}'"
    except Exception as exc:  # pragma: no cover - defensive runtime diagnostics
        return None, f"failed importing '{module_name}': {exc}"


def _resolve_kernel_forward(module: ModuleType) -> SoftmaxForwardImpl:
    for attr in ("softmax_fwd", "softmax_online"):
        fn = getattr(module, attr, None)
        if callable(fn):
            return fn
    raise AttributeError(
        "kernel module must define a callable 'softmax_fwd(x, dim)' or 'softmax_online(x, dim)'"
    )


def _resolve_kernel_backward(module: ModuleType) -> SoftmaxBackwardImpl | None:
    fn = getattr(module, "softmax_bwd", None)
    if callable(fn):
        return fn
    return None


def _call_forward(fn: SoftmaxForwardImpl, x: torch.Tensor, dim: int) -> torch.Tensor:
    try:
        return fn(x, dim=dim)
    except TypeError:
        return fn(x, dim)


def _call_backward(
    fn: SoftmaxBackwardImpl,
    dy: torch.Tensor,
    y: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    try:
        return fn(dy, y, dim=dim)
    except TypeError:
        return fn(dy, y, dim)


def _reference_softmax_forward(x: torch.Tensor, dim: int) -> torch.Tensor:
    from forge_cute_py.ref import softmax_online as softmax_online_ref

    return softmax_online_ref(x, dim=dim)


def _reference_softmax_backward(dy: torch.Tensor, y: torch.Tensor, dim: int) -> torch.Tensor:
    dot_product = (dy * y).sum(dim=dim, keepdim=True)
    return y * (dy - dot_product)


def _forward_impl(x: torch.Tensor, dim: int) -> torch.Tensor:
    impl = _validate_impl_mode()
    if impl == "ref":
        return _reference_softmax_forward(x, dim)

    module, reason = _load_kernel_module()
    if module is None:
        if impl == "auto":
            return _reference_softmax_forward(x, dim)
        raise NotImplementedError(
            "FORGE_SOFTMAX_IMPL=kernel requested, but softmax kernel is unavailable: "
            f"{reason}. Add forge_cute_py/kernels/softmax_online.py with softmax_fwd()."
        )

    try:
        kernel_forward = _resolve_kernel_forward(module)
    except AttributeError as exc:
        if impl == "auto":
            return _reference_softmax_forward(x, dim)
        raise NotImplementedError(
            "FORGE_SOFTMAX_IMPL=kernel requested, but softmax kernel forward entry point is "
            f"incomplete: {exc}"
        ) from exc

    try:
        return _call_forward(kernel_forward, x, dim)
    except NotImplementedError as exc:
        if impl == "auto":
            return _reference_softmax_forward(x, dim)
        raise NotImplementedError(
            "FORGE_SOFTMAX_IMPL=kernel requested, but softmax kernel forward is not implemented."
        ) from exc


def _backward_impl(dy: torch.Tensor, y: torch.Tensor, dim: int) -> torch.Tensor:
    impl = _validate_impl_mode()
    if impl == "ref":
        return _reference_softmax_backward(dy, y, dim)

    module, reason = _load_kernel_module()
    if module is None:
        if impl == "auto":
            return _reference_softmax_backward(dy, y, dim)
        raise NotImplementedError(
            "FORGE_SOFTMAX_IMPL=kernel requested, but softmax kernel is unavailable: "
            f"{reason}. Add forge_cute_py/kernels/softmax_online.py with softmax_bwd()."
        )

    kernel_backward = _resolve_kernel_backward(module)
    if kernel_backward is None:
        if impl == "auto":
            return _reference_softmax_backward(dy, y, dim)
        raise NotImplementedError(
            "FORGE_SOFTMAX_IMPL=kernel requested, but 'softmax_bwd' is missing in "
            "forge_cute_py.kernels.softmax_online."
        )

    try:
        return _call_backward(kernel_backward, dy, y, dim)
    except NotImplementedError as exc:
        if impl == "auto":
            return _reference_softmax_backward(dy, y, dim)
        raise NotImplementedError(
            "FORGE_SOFTMAX_IMPL=kernel requested, but softmax kernel backward is not implemented."
        ) from exc


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


@torch.library.custom_op("forge_cute_py::_softmax_fwd", mutates_args={"out"})
def _softmax_fwd(x: torch.Tensor, out: torch.Tensor, dim: int = -1) -> None:
    """Softmax forward pass."""
    dim = _ensure_forward_inputs(x, out, dim)
    result = _forward_impl(x, dim)
    if result.shape != x.shape:
        raise ValueError(
            f"softmax forward produced invalid shape {result.shape}, expected {x.shape}"
        )
    if result.dtype != x.dtype:
        raise ValueError(
            f"softmax forward produced invalid dtype {result.dtype}, expected {x.dtype}"
        )
    if result.device != x.device:
        raise ValueError(f"softmax forward produced output on {result.device}, expected {x.device}")
    out.copy_(result)


_softmax_fwd.compile_cache = {}


def softmax_fwd(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Forward-only softmax (no autograd)."""
    out = torch.empty_like(x)
    _softmax_fwd(x, out, dim)
    return out


@torch.library.custom_op("forge_cute_py::_softmax_backward", mutates_args={"dx"})
def _softmax_backward(dy: torch.Tensor, y: torch.Tensor, dx: torch.Tensor, dim: int = -1) -> None:
    """Softmax backward pass."""
    dim = _ensure_backward_inputs(dy, y, dx, dim)
    result = _backward_impl(dy, y, dim)
    if result.shape != dy.shape:
        raise ValueError(
            f"softmax backward produced invalid shape {result.shape}, expected {dy.shape}"
        )
    if result.dtype != dy.dtype:
        raise ValueError(
            f"softmax backward produced invalid dtype {result.dtype}, expected {dy.dtype}"
        )
    if result.device != dy.device:
        raise ValueError(
            f"softmax backward produced output on {result.device}, expected {dy.device}"
        )
    dx.copy_(result)


_softmax_backward.compile_cache = {}


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
    """Online softmax with autograd support.

    Backend selection is controlled by FORGE_SOFTMAX_IMPL:
    - auto (default): try kernel first, fallback to reference
    - ref: force reference implementation
    - kernel: require kernel implementation (raise if unavailable)
    """
    return SoftmaxOnlineFunction.apply(x, dim)
