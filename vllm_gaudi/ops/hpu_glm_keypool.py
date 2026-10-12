# SPDX-License-Identifier: Apache-2.0
"""Experimental indexer primitive; BF16 storage of E4M3FN values, not a full port."""

import torch


@torch.compile(backend="hpu_backend", fullgraph=True, dynamic=True)
def _fwht128(x: torch.Tensor) -> torch.Tensor:
    rows = x.shape[0]
    for stride in (1, 2, 4, 8, 16, 32, 64):
        a, b = x.reshape(rows, 128 // (2 * stride), 2, stride).unbind(2)
        x = torch.stack((a + b, a - b), dim=2).reshape(rows, 128)
    return x


@torch._dynamo.disable
def fwht128_quant_values(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Preserve official FWHT/FP8 numeric values without Gaudi2's FP8 cast overflow.

    Returns BF16 quantized values and FP32 scales, both on HPU. A native HPU
    KeyPool consumer is still required; this is not the packed CUDA cache format.
    """
    assert q.device.type == "hpu"
    assert q.ndim == 2 and q.shape[1] == 128 and q.dtype == torch.bfloat16
    assert q.is_contiguous()
    x = _fwht128(q.float())
    # Keep the BF16 roundtrip eager so compilation cannot remove its rounding.
    x = (x * 0.08838834764831845).to(torch.bfloat16).float()
    scale = torch.exp2(torch.ceil(torch.log2(x.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448)))
    y = (x / scale).clamp_(-448, 448)
    # Gaudi2's native FP8 cast overflows above 240; BF16 represents every FN value.
    step = torch.exp2(torch.floor(torch.log2(y.abs().clamp_min(2**-6))) - 3)
    values = torch.round(y / step).mul_(step).to(torch.bfloat16)
    return values, scale


@torch._dynamo.disable
def select_pooled_tokens(q, weights, keys, gates, ape, positions, topk):
    """Return causal logical token indices using GLM's learned pooled keys."""
    pool_size, dim = ape.shape
    if dim != 128 or topk % pool_size or keys.shape != gates.shape:
        raise ValueError("Expected matching 128-wide keys/gates and pool-aligned top-k")
    rows, heads, _ = q.shape
    count = keys.shape[0] // pool_size
    width = topk + pool_size - 1
    result = torch.full((rows, width), -1, device=q.device, dtype=torch.long)
    if keys.shape[0] <= topk:
        ids = torch.arange(keys.shape[0], device=q.device)
        result[:, :ids.numel()] = ids[None, :].expand(rows, -1).masked_fill(ids[None, :] > positions[:, None], -1)
        return result
    pooled_keys = keys[:count * pool_size].view(count, pool_size, dim).float()
    logits = gates[:count * pool_size].view(count, pool_size, dim).float() + ape.float()
    pooled = (logits.softmax(dim=1) * pooled_keys).sum(1).to(torch.bfloat16)
    values, scales = fwht128_quant_values(pooled.contiguous())
    pooled = values.float() * scales
    take = min(topk // pool_size, count)
    pool_ids = torch.arange(count, device=q.device)
    offsets = torch.arange(pool_size, device=q.device)
    # shortcut: score pools again each step; replace after pooled-cache parity is established.
    for start in range(0, rows, 32):
        end = min(rows, start + 32)
        n = end - start
        scores = torch.matmul(q[start:end].float(), pooled.T).relu()
        scores = (scores * weights[start:end, :, None]).sum(1)
        completed = (positions[start:end] + 1) // pool_size
        scores.masked_fill_(pool_ids[None, :] >= completed[:, None], -torch.inf)
        ranked, selected = scores.topk(take, dim=-1)
        tokens = (selected[:, :, None] * pool_size + offsets).reshape(n, -1)
        tokens.masked_fill_(ranked.isneginf().repeat_interleave(pool_size, dim=1), -1)
        result[start:end, :take * pool_size] = tokens
        tail = completed[:, None] * pool_size + offsets[:pool_size - 1]
        result[start:end, topk:] = tail.masked_fill_(tail > positions[start:end, None], -1)
    return result
