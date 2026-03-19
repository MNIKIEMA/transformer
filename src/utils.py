from collections import defaultdict
import torch


def generate_square_subsequent_mask(sz: int):
    """Generate a lower triangular matrix for causal masking."""
    return torch.tril(torch.ones((sz, sz), dtype=torch.bool))


def create_masks(src: torch.Tensor, tgt_input: torch.Tensor, pad_id: int = 0):
    src_mask = (src != pad_id).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]
    tgt_mask = (tgt_input != pad_id).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
    size = tgt_input.size(1)
    causal_mask = generate_square_subsequent_mask(size).to(tgt_input.device)  # [T, T]
    combined_mask = tgt_mask & causal_mask.unsqueeze(0).unsqueeze(0)  # [B, 1, T, T]
    return src_mask, combined_mask


def averager(beta: float = 1):
    """
    Returns a single function that can be called to repeatidly obtain
    a running average from a dictionary of metrics.
    The callback will return the new averaged dict of metrics.

    `beta` is the decay parameter. If `beta == 1`, a regular running
    average is performed. If `beta < 1`, an exponential moving average
    is performed instead.
    """
    count = defaultdict(float)
    total = defaultdict(float)

    def _update(metrics: dict, weight: float = 1) -> dict:
        nonlocal total, count
        for key, value in metrics.items():
            total[key] = total[key] * beta + weight * float(value)
            count[key] = count[key] * beta + weight
        return {key: tot / count[key] for key, tot in total.items()}

    return _update


def compute_rope_params(head_dim: int, theta_base=10_000, context_length=5000, dtype=torch.float32):
    assert head_dim % 2 == 0, "Embedding dimension must be even."
    inv_freq = 1.0 / (
        theta_base
        ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim)
    )
    positions = torch.arange(0, context_length, dtype=dtype)

    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    angles = torch.cat([angles, angles], dim=1)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    B, num_heads, seq_len, head_dim = x.size()
    assert head_dim % 2 == 0, "Embedding dimension must be even."
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)
    return x_rotated.to(dtype=x.dtype)
