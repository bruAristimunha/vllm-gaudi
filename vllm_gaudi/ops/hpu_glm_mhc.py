# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native mHC projection and rounding with a compiled FP32 Sinkhorn loop."""

import torch


@torch.compile(backend="hpu_backend", fullgraph=True, dynamic=False)
def _sinkhorn(comp: torch.Tensor, eps: float, iterations: int) -> torch.Tensor:
    comp = comp / (comp.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iterations - 1):
        comp = comp / (comp.sum(dim=-1, keepdim=True) + eps)
        comp = comp / (comp.sum(dim=-2, keepdim=True) + eps)
    return comp


# Keep projection and the final BF16 rounding eager; whole-function fusion changes rounding.
def hpu_mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward pass for mHC pre block.

    Args:
        residual: shape (..., hc_mult, hidden_size), dtype torch.bfloat16
        fn: shape (hc_mult3, hc_mult * hidden_size), dtype torch.float32
        hc_scale: shape (3,), dtype torch.float32
        hc_base: shape (hc_mult3,), dtype torch.float32
        rms_eps: RMS normalization epsilon
        hc_pre_eps: pre-mix epsilon
        hc_sinkhorn_eps: sinkhorn epsilon
        hc_post_mult_value: post-mix multiplier value
        sinkhorn_repeat: number of sinkhorn iterations
        n_splits: split-k factor;

    Returns:
        post_mix: shape (..., hc_mult), dtype torch.float32
        comb_mix: shape (..., hc_mult, hc_mult), dtype torch.float32
        layer_input: shape (..., hidden_size), dtype torch.bfloat16
    """

    # Validate shapes
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2

    hc_hidden_size = hc_mult * hidden_size
    assert fn.shape[0] == hc_mult3
    assert fn.shape[1] == hc_hidden_size
    assert hc_scale.shape == (3, )
    assert hc_base.shape == (hc_mult3, )

    outer_shape = residual.shape[:-2]

    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    fn_t = getattr(fn, "_cached_t", None)
    if fn_t is None:
        fn_t = fn.t()
        try:
            fn._cached_t = fn_t
        except Exception:
            pass

    x = residual_flat.view(num_tokens, hc_mult * hidden_size).to(torch.float32)
    mixes = torch.matmul(x, fn_t)
    sqrsum = x.square().sum(dim=-1, keepdim=True)
    mixes = mixes.mul_(torch.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps))

    base_pre = getattr(hc_base, "_cached_pre", None)
    if base_pre is None or base_pre.shape[0] != hc_mult:
        base_pre = hc_base[:hc_mult]
        base_post = hc_base[hc_mult:2 * hc_mult]
        base_comb = hc_base[2 * hc_mult:].view(1, hc_mult, hc_mult)
        s0, s1, s2 = hc_scale[0], hc_scale[1], hc_scale[2]
        try:
            hc_base._cached_pre = base_pre
            hc_base._cached_post = base_post
            hc_base._cached_comb = base_comb
            hc_scale._cached_s0 = s0
            hc_scale._cached_s1 = s1
            hc_scale._cached_s2 = s2
        except Exception:
            pass
    else:
        base_post = hc_base._cached_post
        base_comb = hc_base._cached_comb
        s0, s1, s2 = hc_scale._cached_s0, hc_scale._cached_s1, hc_scale._cached_s2

    pre_logits = (mixes[:, :hc_mult] * s0).add_(base_pre)
    pre_mix = torch.sigmoid(pre_logits).add_(hc_pre_eps)

    post_logits = (mixes[:, hc_mult:2 * hc_mult] * s1).add_(base_post)
    post_mix = torch.sigmoid(post_logits).mul_(hc_post_mult_value)

    comb_logits = (mixes[:, 2 * hc_mult:].view(num_tokens, hc_mult, hc_mult) * s2).add_(base_comb)
    comb_mix = torch.softmax(comb_logits, dim=-1).add_(hc_sinkhorn_eps)
    comb_mix = _sinkhorn(comp=comb_mix, eps=hc_sinkhorn_eps, iterations=sinkhorn_repeat)

    layer_input = torch.sum(pre_mix.unsqueeze(-1) * residual_flat.to(torch.float32), dim=1).to(torch.bfloat16)
    return (
        post_mix.view(*outer_shape, hc_mult, 1),
        comb_mix.view(*outer_shape, hc_mult, hc_mult),
        layer_input.view(*outer_shape, hidden_size),
    )


def hpu_mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    mixed = torch.matmul(comb_res_mix.transpose(-2, -1), residual.float())
    return mixed.addcmul_(post_layer_mix, x.unsqueeze(-2).float()).to(residual.dtype)


