# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from vllm_gaudi.extension import ops
from vllm_gaudi.ops.hpu_fp8 import HPUFp8MoEMethod
from vllm_gaudi.ops.hpu_fused_moe import HPUUnquantizedFusedMoEMethod
from vllm_gaudi.ops.hpu_moe_combine import _dynamic_quant, _gather_from_quantized, gather_silu_fp8_moe


def test_clamped_experts():
    # Diagonal projections deliberately cross both clamp boundaries.
    eye = torch.eye(128)
    w13 = torch.stack([torch.cat([eye * 32, eye * (-2 if i % 2 else 2)]) for i in range(4)])
    w2 = torch.stack([eye * (i + 1) for i in range(4)])
    for quantized, limit in ((False, 10.), (True, 10.), (False, None), (True, None)):
        dtype = torch.float8_e4m3fn if quantized else torch.bfloat16
        layer = SimpleNamespace(w13_weight=w13.to(dtype).to('hpu'),
                                w2_weight=w2.to(dtype).to('hpu'),
                                w13_weight_scale_inv=torch.ones(4, 256, device='hpu'),
                                w2_weight_scale_inv=torch.ones(4, 128, device='hpu'),
                                local_num_experts=4,
                                use_grouped_topk=False,
                                top_k=2,
                                activation='silu',
                                swiglu_limit=limit,
                                moe_config=SimpleNamespace(dp_size=1, ep_rank=0, is_sequence_parallel=False))
        with patch.object(ops, 'get_config', return_value=SimpleNamespace(moe_chunk=[], moe_token_boundary=[])):
            if quantized:
                layer.moe_op = ops.VllmMixtureOfExpertsOpFP8PerChannel(4, 4, 0, 3)
                ops.fp8_channel_moe_prepare_weights(layer)
            else:
                layer.moe_op = ops.VllmMixtureOfExpertsOp(4, 4, 0, 3)
                for i in range(4):
                    layer.moe_op.w13_list[i].set_weight(layer.w13_weight[i])
                    layer.moe_op.w2_list[i].set_weight(layer.w2_weight[i])
        method = SimpleNamespace(model_type='glm5_next',
                                 _moe_gather_ok=True,
                                 quant_config=SimpleNamespace(activation_scheme='dynamic'))
        apply = HPUFp8MoEMethod.apply_monolithic if quantized else HPUUnquantizedFusedMoEMethod.apply_monolithic
        for tokens in (1, 9):
            x = torch.ones(tokens, 128, dtype=torch.bfloat16)
            logits = torch.tensor([1., 3., 2., 0.]).repeat(tokens, 1)
            weights, indices = logits.softmax(-1).topk(2, dim=-1)
            weights = (weights / weights.sum(-1, keepdim=True)).bfloat16().float()
            expected = torch.zeros_like(x, dtype=torch.float32)
            for row in range(tokens):
                for score, expert in zip(weights[row], indices[row]):
                    gate, up = F.linear(x[row].float(), w13[expert]).chunk(2)
                    if limit is not None:
                        gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
                    act = F.silu(gate) * up
                    expected[row] += score * F.linear(act, w2[expert])
            actual = apply(method, layer, x.to('hpu'), logits.to('hpu')).cpu().float()
            torch.testing.assert_close(actual, expected, rtol=.015, atol=.1)


def test_compiled_clamped_experts_preserve_eager_quantization():
    generator = torch.Generator().manual_seed(819)
    layer = SimpleNamespace(
        w13_weight=(torch.randn(7, 256, 128, generator=generator) * .2).to(torch.float8_e4m3fn).to('hpu'),
        w2_weight=(torch.randn(7, 128, 128, generator=generator) * .2).to(torch.float8_e4m3fn).to('hpu'),
        w13_weight_scale_inv=(torch.rand(7, 256, generator=generator) + .5).to('hpu'),
        w2_weight_scale_inv=(torch.rand(7, 128, generator=generator) + .5).to('hpu'),
        local_num_experts=7,
        moe_config=SimpleNamespace(ep_rank=1))
    with torch.inference_mode():
        for tokens, magnitude in ((1, 1.), (8, 10.), (9, 0.), (9, 1.)):
            x = (torch.randn(tokens, 128, generator=generator) * magnitude).bfloat16().to('hpu')
            # Duplicate routes and nonlocal experts exercise scatter-add and masking.
            ids = torch.tensor([7, 7, 9, 13, 0, 15]).repeat(tokens, 1).to('hpu')
            weights = torch.rand(tokens, 6, generator=generator).softmax(-1).bfloat16().to('hpu')
            expected = []
            for start in range(0, tokens, 8):
                chunk = x[start:start + 8]
                fp8, scale = _dynamic_quant(chunk)
                expected.append(
                    _gather_from_quantized(layer, chunk, ids[start:start + 8], weights[start:start + 8], fp8, scale,
                                           10.))
            expected = torch.cat(expected).float()
            actual = gather_silu_fp8_moe(layer, x, ids, weights, 10.).float()
            assert torch.isfinite(actual).all().item()
            error = actual - expected
            torch.testing.assert_close(actual, expected, rtol=.0078125, atol=.0005)
            relative_rms = error.square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-8)
            assert relative_rms.item() <= .0001


def test_dynamic_quant_per_row_preserves_fp8_bytes_and_scales():
    def reference_quant(data, single_scale=False):
        if single_scale:
            scale = ((torch.abs(data)).max() + 1e-8) / ops.FP8_MAX
        else:
            scale = ((torch.abs(data)).max(dim=-1).values + 1e-8) / ops.FP8_MAX
            scale = scale.unsqueeze(-1)
        data_fp8 = torch.ops.hpu.cast_to_fp8_v2(data, 1.0 / scale, False, False, torch.float8_e4m3fn)[0]
        return data_fp8, scale.float()

    generator = torch.Generator().manual_seed(42)
    width = 4096
    for rows in (1, 8, 256):
        t_rand = torch.randn(rows, width, generator=generator).bfloat16().to('hpu')
        t_zero = torch.zeros(rows, width, dtype=torch.bfloat16, device='hpu')
        t_tied = torch.randn(rows, width, generator=generator).bfloat16().to('hpu')
        t_tied[:, :5] = 42.0
        t_large = (torch.rand(rows, width, generator=generator) * 1e4).bfloat16().to('hpu')
        t_noncontig = torch.randn(rows, width * 2, generator=generator).bfloat16().to('hpu')[:, ::2]

        test_cases = [
            (t_rand, False),
            (t_zero, False),
            (t_tied, False),
            (t_large, False),
            (t_noncontig, False),
            (t_rand, True),
        ]
        for data, single_scale in test_cases:
            ref_fp8, ref_scale = reference_quant(data, single_scale=single_scale)
            cand_fp8, cand_scale = ops.dynamic_quant(data, single_scale=single_scale)
            torch.testing.assert_close(cand_scale, ref_scale, rtol=0, atol=0)
            assert torch.equal(cand_fp8.cpu().view(torch.uint8), ref_fp8.cpu().view(torch.uint8))
