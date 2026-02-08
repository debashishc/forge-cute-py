import pytest
import torch

from forge_cute_py.ops import (
    get_softmax_online_backend,
    list_softmax_online_backends,
    register_softmax_online_backend,
    set_softmax_online_backend,
    softmax_online,
)
from forge_cute_py.ref import softmax_online as ref_softmax_online


@pytest.fixture(autouse=True)
def _restore_softmax_backend():
    previous = get_softmax_online_backend()
    try:
        set_softmax_online_backend("ref")
        yield
    finally:
        set_softmax_online_backend(previous)


dims = [-1]


@pytest.mark.parametrize("shape", [(4, 8), (2, 128)])
@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.bfloat16, 1e-3, 1e-3),
        (torch.float16, 1e-2, 1e-2),
        (torch.float32, 1e-4, 1e-4),
    ],
)
def test_softmax_online_correctness(shape, dim, dtype, atol, rtol):
    x = (0.1 * torch.randn(*shape, device="cuda", dtype=dtype)).requires_grad_(True)
    y = softmax_online(x, dim)
    y_ref = ref_softmax_online(x, dim=dim)
    torch.testing.assert_close(y, y_ref, atol=atol, rtol=rtol)

    assert torch.isfinite(y).all()


def test_softmax_online_rejects_dim0():
    x = torch.randn(4, 8, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="expects dim=-1"):
        softmax_online(x, dim=0)


def test_softmax_online_rejects_dim1():
    x = torch.randn(4, 8, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="expects dim=-1"):
        softmax_online(x, dim=1)


@pytest.mark.parametrize("shape", [(4, 8), (2, 128)])
@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.bfloat16, 1e-3, 1e-3),
        (torch.float16, 1e-2, 1e-2),
        (torch.float32, 1e-4, 1e-4),
    ],
)
def test_softmax_online_torch_compile(shape, dim, dtype, atol, rtol):
    unsupported_exc = ()
    try:
        from torch._dynamo.exc import Unsupported as DynamoUnsupported

        unsupported_exc = (DynamoUnsupported,)
    except Exception:
        unsupported_exc = ()
    try:
        compiled = torch.compile(softmax_online, fullgraph=True)
    except Exception as exc:
        pytest.skip(f"torch.compile not available for softmax_online: {exc}")
    x = torch.randn(shape, device="cuda", dtype=dtype)
    try:
        y = compiled(x, dim=dim)
    except unsupported_exc as exc:
        pytest.skip(f"torch.compile unsupported for softmax_online op: {exc}")
    y_ref = ref_softmax_online(x, dim=dim)
    torch.testing.assert_close(y, y_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("input_dtype", [torch.float16, torch.float32])
def test_softmax_online_properties(input_dtype):
    x = torch.randn(16, 256, device="cuda", dtype=input_dtype)
    y = softmax_online(x, -1)
    sums = torch.sum(y, dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-3, rtol=1e-3)
    assert (y >= 0).all()
    assert (y <= 1).all()


def test_softmax_online_translation_invariance():
    x = torch.randn(8, 128, device="cuda", dtype=torch.float32)
    y = softmax_online(x, -1)
    y_shifted = softmax_online(x + 100.0, -1)
    torch.testing.assert_close(y, y_shifted, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("input_dtype", [torch.float16, torch.float32])
def test_softmax_online_extreme_values(input_dtype):
    m, n = 8, 256
    x_large = torch.full((m, n), 10.0, device="cuda", dtype=input_dtype)
    out_large = softmax_online(x_large, -1)
    expected = torch.full_like(out_large, 1.0 / n)
    torch.testing.assert_close(out_large, expected, atol=1e-3, rtol=1e-3)
    x_small = torch.full((m, n), -10.0, device="cuda", dtype=input_dtype)
    out_small = softmax_online(x_small, -1)
    torch.testing.assert_close(out_small, expected, atol=1e-3, rtol=1e-3)
    x_mixed = torch.zeros((m, n), device="cuda", dtype=input_dtype)
    x_mixed[:, 0] = 10.0
    x_mixed[:, 1:] = -10.0
    out_mixed = softmax_online(x_mixed, -1)
    assert (out_mixed[:, 0] > 0.99).all()
    assert (out_mixed[:, 1:] < 0.01).all()


@pytest.mark.parametrize("shape", [(4, 8), (16, 128), (32, 256)])
@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.bfloat16, 1e-2, 1e-2),
        (torch.float16, 1e-3, 1e-3),
        (torch.float32, 1e-4, 1e-4),
    ],
)
def test_softmax_online_backward(shape, dim, dtype, atol, rtol):
    """Test backward pass against PyTorch reference."""
    # Create inputs with gradients enabled (scale by 0.1 to avoid overflow)

    x = (0.1 * torch.randn(*shape, device="cuda", dtype=dtype)).requires_grad_(True)
    x_ref = x.detach().clone().requires_grad_(True)

    # Forward pass
    out = softmax_online(x, dim=dim)
    out_ref = ref_softmax_online(x_ref, dim=dim)
    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)

    # Backward pass
    dy = torch.randn_like(out)
    torch.cuda.synchronize()  # Critical: prevents autograd timing issues
    (dx,) = torch.autograd.grad(out, x, grad_outputs=dy)
    (dx_ref,) = torch.autograd.grad(out_ref, x_ref, grad_outputs=dy)
    torch.testing.assert_close(dx, dx_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("shape", [(4, 8), (16, 128)])
@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_softmax_online_backward_torch_compile(shape, dim, dtype):
    """Test backward pass works with torch.compile."""
    unsupported_exc = ()
    try:
        from torch._dynamo.exc import Unsupported as DynamoUnsupported

        unsupported_exc = (DynamoUnsupported,)
    except Exception:
        unsupported_exc = ()

    try:
        compiled = torch.compile(softmax_online, fullgraph=True)
    except Exception as exc:
        pytest.skip(f"torch.compile not available: {exc}")

    # Create inputs
    x = (0.1 * torch.randn(*shape, device="cuda", dtype=dtype)).requires_grad_(True)
    x_ref = x.detach().clone().requires_grad_(True)

    # Forward + backward (compiled)
    try:
        y = compiled(x, dim=dim)
    except unsupported_exc as exc:
        pytest.skip(f"torch.compile unsupported: {exc}")

    dy = torch.randn_like(y)
    torch.cuda.synchronize()
    (dx,) = torch.autograd.grad(y, x, grad_outputs=dy)

    # Forward + backward (reference)
    y_ref = ref_softmax_online(x_ref, dim=dim)
    torch.cuda.synchronize()
    (dx_ref,) = torch.autograd.grad(y_ref, x_ref, grad_outputs=dy)

    # Check gradients
    atol, rtol = (1e-2, 1e-2) if dtype == torch.float16 else (1e-4, 1e-4)
    torch.testing.assert_close(dx, dx_ref, atol=atol, rtol=rtol)


def test_softmax_online_gradient_properties():
    """Test mathematical properties of softmax gradients."""
    m, n = 16, 128

    # Test property: gradient of uniform upstream should have zero row/col sum
    x = torch.randn(m, n, device="cuda", dtype=torch.float32, requires_grad=True)
    y = softmax_online(x, dim=-1)

    # Uniform upstream gradient
    dy_uniform = torch.ones_like(y)
    torch.cuda.synchronize()
    (dx,) = torch.autograd.grad(y, x, grad_outputs=dy_uniform)

    # Row sums should be approximately zero (softmax Jacobian property)
    row_sums = dx.sum(dim=-1)
    torch.testing.assert_close(row_sums, torch.zeros_like(row_sums), atol=1e-6, rtol=1e-6)


def test_softmax_online_backend_registry_exposes_expected_backends():
    backends = list_softmax_online_backends()
    assert "ref" in backends
    assert "kernel" in backends


@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float16, 1e-2, 1e-2),
        (torch.float32, 1e-4, 1e-4),
    ],
)
def test_softmax_online_kernel_backend_forward_matches_ref(dtype, atol, rtol):
    set_softmax_online_backend("kernel")

    x = (0.1 * torch.randn(16, 128, device="cuda", dtype=dtype)).requires_grad_(True)
    y = softmax_online(x, dim=-1)
    y_ref = ref_softmax_online(x, dim=-1)
    torch.testing.assert_close(y, y_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-4, 1e-4),
    ],
)
def test_softmax_online_kernel_backend_backward_matches_ref(dtype, atol, rtol):
    set_softmax_online_backend("kernel")

    x = (0.1 * torch.randn(16, 128, device="cuda", dtype=dtype)).requires_grad_(True)
    x_ref = x.detach().clone().requires_grad_(True)

    y = softmax_online(x, dim=-1)
    y_ref = ref_softmax_online(x_ref, dim=-1)
    dy = torch.randn_like(y)

    torch.cuda.synchronize()
    (dx,) = torch.autograd.grad(y, x, grad_outputs=dy)
    (dx_ref,) = torch.autograd.grad(y_ref, x_ref, grad_outputs=dy)
    torch.testing.assert_close(dx, dx_ref, atol=atol, rtol=rtol)


def test_softmax_online_custom_backend_forward():
    def custom_ref_forward(x: torch.Tensor, dim: int) -> torch.Tensor:
        return ref_softmax_online(x, dim=dim)

    register_softmax_online_backend("test_ref_forward", custom_ref_forward, overwrite=True)
    set_softmax_online_backend("test_ref_forward")

    x = torch.randn(16, 128, device="cuda", dtype=torch.float16)
    y = softmax_online(x, dim=-1)
    y_ref = ref_softmax_online(x, dim=-1)
    torch.testing.assert_close(y, y_ref, atol=1e-2, rtol=1e-2)


def test_softmax_online_custom_backend_backward_fallback():
    def forward_only(x: torch.Tensor, dim: int) -> torch.Tensor:
        return ref_softmax_online(x, dim=dim)

    register_softmax_online_backend("test_fwd_only", forward_only, overwrite=True)
    set_softmax_online_backend("test_fwd_only")

    x = (0.1 * torch.randn(16, 128, device="cuda", dtype=torch.float32)).requires_grad_(True)
    x_ref = x.detach().clone().requires_grad_(True)

    y = softmax_online(x, dim=-1)
    y_ref = ref_softmax_online(x_ref, dim=-1)

    dy = torch.randn_like(y)
    torch.cuda.synchronize()
    (dx,) = torch.autograd.grad(y, x, grad_outputs=dy)
    (dx_ref,) = torch.autograd.grad(y_ref, x_ref, grad_outputs=dy)
    torch.testing.assert_close(dx, dx_ref, atol=1e-4, rtol=1e-4)


def test_softmax_online_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown softmax_online backend"):
        set_softmax_online_backend("missing_backend")
