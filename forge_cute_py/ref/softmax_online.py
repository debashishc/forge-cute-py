import torch


def softmax_online(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError("softmax_online expects a 2D tensor")
    if dim != -1:
        raise ValueError(f"softmax_online expects dim=-1 (row-wise) for 2D tensors, got {dim}")
    x_dtype = x.dtype
    return x.float().softmax(dim=dim).to(x_dtype)
