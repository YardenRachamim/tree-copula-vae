import torch
import torch.nn as nn
import torch.distributions as dist

def get_prob_eps_by_dtype(dtype: torch.dtype) -> float:
    # Pick conservative eps by dtype (you can tune per use-case)
    return 1e-6 if dtype in (torch.float16, torch.bfloat16, torch.float32) else 1e-12

def clamp_probs_by_dtype(p: torch.Tensor, min: float = 0., max: float = 1.) -> torch.Tensor:
    eps = get_prob_eps_by_dtype(p.dtype)
    return p.clamp(min=min + eps, max=max - eps)

def safe_icdf(dist: dist.Distribution, value: torch.Tensor) -> torch.Tensor:
    """
    dist: a torch.distributions instance with .icdf
    p:    probabilities (..., ) in [0,1]
    """
    value64 = value.to(torch.float64)
    out64 = dist.icdf(value64)

    return out64.to(value.dtype)


def safe_cdf(dist, value: torch.Tensor) -> torch.Tensor:
    """
    dist: a torch.distributions instance with .icdf
    p:    probabilities (..., ) in [0,1]
    """
    value64 = value.to(torch.float64)
    out64 = dist.cdf(value64)

    return out64.to(value.dtype)

def tempered_softmax(
        logits: torch.Tensor,
        temperature: torch.Tensor,
        dim: int = -1,
        return_log: bool = False,
        min_temp: float = 1e-8,
) -> torch.Tensor:
    """
    Temperature-scaled softmax: p_i ∝ exp(logit_i / T).

    Args:
        logits: Tensor of shape (N, C) or (N, ... , C). Batch dimension must be 0.
        temperature: Tensor of shape (N,) or broadcastable to (N, 1, ..., 1).
                     Each sample gets its own temperature T_n > 0.
        dim: Dimension over classes (default: -1).
        return_log: If True, returns log-probabilities (log-softmax).
        min_temp: Temperatures are clamped to at least this value for numerical stability.

    Returns:
        probs or log_probs with same shape as logits.
    """
    if logits.dim() < 2:
        raise ValueError("logits must be at least 2D: (N, C) or (N, ..., C)")

    # Make temperature broadcast across all non-batch dims
    temperature = temperature.to(dtype=logits.dtype, device=logits.device)
    temperature = torch.clamp(temperature, min=min_temp)
    # reshape to (N, 1, 1, ..., 1) so it broadcasts across feature dims
    view_shape = (temperature.shape[0],) + (1,) * (logits.dim() - 1)
    temperature = temperature.view(*view_shape)

    scaled = logits / temperature  # elementwise, batched division

    if return_log:
        return nn.functional.log_softmax(scaled, dim=dim)
    else:
        return nn.functional.softmax(scaled, dim=dim)