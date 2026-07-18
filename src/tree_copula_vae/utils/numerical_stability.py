import torch
import torch.nn as nn
import torch.distributions as dist


def get_prob_eps_by_dtype(dtype: torch.dtype) -> float:
    return 1e-6 if dtype in (torch.float16, torch.bfloat16, torch.float32) else 1e-12


def clamp_probs_by_dtype(p: torch.Tensor, min: float = 0., max: float = 1.) -> torch.Tensor:
    eps = get_prob_eps_by_dtype(p.dtype)
    return p.clamp(min=min + eps, max=max - eps)


def safe_icdf(dist: dist.Distribution, value: torch.Tensor) -> torch.Tensor:
    value64 = value.to(torch.float64)
    out64 = dist.icdf(value64)
    return out64.to(value.dtype)


def safe_cdf(dist, value: torch.Tensor) -> torch.Tensor:
    value64 = value.to(torch.float64)
    out64 = dist.cdf(value64)
    return out64.to(value.dtype)


def tempered_softmax(logits: torch.Tensor, temperature: torch.Tensor, dim: int = -1, return_log: bool = False, min_temp: float = 1e-8) -> torch.Tensor:
    if logits.dim() < 2:
        raise ValueError("logits must be at least 2D")
    temperature = temperature.to(dtype=logits.dtype, device=logits.device)
    temperature = torch.clamp(temperature, min=min_temp)
    view_shape = (temperature.shape[0],) + (1,) * (logits.dim() - 1)
    temperature = temperature.view(*view_shape)
    scaled = logits / temperature
    if return_log:
        return nn.functional.log_softmax(scaled, dim=dim)
    else:
        return nn.functional.softmax(scaled, dim=dim)
