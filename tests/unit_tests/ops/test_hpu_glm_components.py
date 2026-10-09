# SPDX-License-Identifier: Apache-2.0
"""Component regressions on one assigned HPU, not full-model qualification."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F


class GlmComponents(unittest.TestCase):

    def test_existing_sparse_indexer_preserves_request_ownership(self):
        from vllm_gaudi.ops import hpu_sparse_attn_indexer as indexer

        cache = torch.zeros(16, 128, dtype=torch.bfloat16)
        cache[:, 0] = torch.tensor([1, 9, 3, 5, 1000, 1000, 1000, 1000, 7, 2, 1000, 1000, 11, 1000, 1000, 1000])
        cache = cache.to("hpu")
        q = torch.zeros(2, 2, 128, dtype=torch.bfloat16)
        q[:, :, 0] = 1
        meta = SimpleNamespace(block_size=4,
                               slot_mapping=torch.tensor([3, 12], device="hpu"),
                               is_prompt=False,
                               block_list=torch.tensor([2, 0, 3], device="hpu"),
                               block_groups=torch.tensor([1, 0, 1], device="hpu"),
                               block_usage=torch.tensor([2., 4., 1.], device="hpu"))
        holder = SimpleNamespace(k_cache=SimpleNamespace(kv_cache=cache),
                                 skip_k_cache_insert=True,
                                 topk_tokens=4,
                                 topk_indices_buffer=torch.empty(2, 4, dtype=torch.int32, device="hpu"))
        # Only inject metadata; scoring, masking, cache indexing and top-k run unchanged.
        with patch.object(indexer, "get_forward_context", return_value=SimpleNamespace(attn_metadata=meta)):
            result = indexer.forward_hpu(holder, None, q.to("hpu"), cache[:2], torch.full((2, 2), .5, device="hpu"))
        self.assertEqual(result.device.type, "hpu")
        torch.testing.assert_close(result.cpu(), torch.tensor([[1, 3, 2, 0], [12, 8, 9, -1]], dtype=torch.int32))

    def test_existing_sparse_mla_masks_padding(self):
        from vllm_gaudi.attention.backends.hpu_attn import HPUMLAImpl

        generator = torch.Generator().manual_seed(3)
        cache = torch.randn(16, 8, generator=generator, dtype=torch.bfloat16)
        q = torch.randn(2, 2, 8, generator=generator, dtype=torch.bfloat16)
        slots = torch.tensor([[1, 3, 2, 0], [12, 8, 9, -1]], dtype=torch.int32)
        obj = SimpleNamespace(kv_lora_rank=8, qk_rope_head_dim=0, scale=8**-.5, num_heads=2)
        result = HPUMLAImpl.forward_mqa_sparse(obj, q.to("hpu"), cache.to("hpu"), None, slots.to("hpu"))
        expected = []
        for row in range(2):
            selected = cache[slots[row][slots[row] >= 0].long()].float()
            probs = (q[row].float() @ selected.T * obj.scale).softmax(-1)
            expected.append((probs @ selected).flatten())
        self.assertEqual(result.device.type, "hpu")
        torch.testing.assert_close(result.cpu().float(), torch.stack(expected), atol=.02, rtol=.02)
        empty = torch.full_like(slots, -1).to("hpu")
        result = HPUMLAImpl.forward_mqa_sparse(obj, q.to("hpu"), cache.to("hpu"), None, empty)
        torch.testing.assert_close(result.cpu(), torch.zeros(2, 16, dtype=torch.bfloat16))

    def check_prefill(self, rope):
        from vllm_gaudi.attention.backends.hpu_attn import HPUMLAImpl

        generator = torch.Generator().manual_seed(19)
        latent = torch.randn(4, 8 + rope, generator=generator, dtype=torch.bfloat16)
        weight = torch.randn(8, 16, generator=generator, dtype=torch.bfloat16) * .25
        q = torch.randn(4, 2, 4 + rope, generator=generator, dtype=torch.bfloat16)
        scale = (4 + rope)**-.5
        projected = (latent[:, :8].float() @ weight.float()).reshape(4, 2, 8)
        k = torch.cat([projected[:, :, :4], latent[:, 8:].float().unsqueeze(1).expand(-1, 2, -1)], -1)
        v = projected[:, :, 4:]
        expected = F.scaled_dot_product_attention(q.float().transpose(0, 1),
                                                  k.transpose(0, 1),
                                                  v.transpose(0, 1),
                                                  is_causal=True,
                                                  scale=scale)
        weight_hpu = weight.to("hpu")
        obj = SimpleNamespace(kv_lora_rank=8,
                              qk_rope_head_dim=rope,
                              qk_nope_head_dim=4,
                              qk_head_dim=4 + rope,
                              v_head_dim=4,
                              num_heads=2,
                              scale=scale,
                              kv_b_proj=lambda x: (x @ weight_hpu, None),
                              use_merged_prefill=False,
                              prefill_impl="naive_impl",
                              matmul_qk=torch.matmul,
                              softmax=torch.softmax,
                              matmul_av=torch.matmul,
                              latent_cache_k=SimpleNamespace(fetch_from_cache=None),
                              fused_scaled_dot_product_attention=None)
        mask = torch.full((4, 4), -torch.inf).triu(1).to(torch.bfloat16).to("hpu")
        meta = SimpleNamespace(block_list=None, seq_lens_tensor=torch.tensor([4], device="hpu"), attn_bias=mask)
        result = HPUMLAImpl.forward_mha(obj, q.to("hpu"), latent.to("hpu"), None, meta)
        self.assertEqual(result.device.type, "hpu")
        torch.testing.assert_close(result.cpu().float(), expected.transpose(0, 1).reshape(4, 8), atol=.02, rtol=.02)

    def test_mla_prefill_zero_rope(self):
        self.check_prefill(0)

    def test_mla_prefill_existing_rope(self):
        self.check_prefill(64)

    def test_official_fwht_fp8_matches_independent_hadamard(self):
        from vllm_gaudi.ops.hpu_glm_keypool import fwht128_quant_values as fwht128_quant_fp8

        generator = torch.Generator().manual_seed(23)
        # Walsh matrix uses parity, independent of the implementation's butterfly.
        hadamard = torch.tensor([[(-1.)**((i & j).bit_count()) for j in range(128)] for i in range(128)],
                                dtype=torch.float32)
        for name, q in [("empty", torch.empty(0, 128)), ("zero", torch.zeros(2, 128)), ("ones", torch.ones(2, 128)),
                        ("impulses", torch.eye(128)), ("odd_batch", torch.randn(33, 128, generator=generator))]:
            with self.subTest(case=name):
                q = q.to(torch.bfloat16)
                rotated = ((q.float() @ hadamard) * (128**-.5)).to(torch.bfloat16).float()
                scales = torch.exp2(torch.ceil(torch.log2(rotated.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448)))
                expected = (rotated / scales).clamp(-448, 448).to(torch.float8_e4m3fn).float()
                quantized, actual_scales = fwht128_quant_fp8(q.to("hpu"))
                self.assertEqual(quantized.device.type, "hpu")
                self.assertEqual(actual_scales.device.type, "hpu")
                torch.testing.assert_close(actual_scales.cpu(), scales, atol=0, rtol=0)
                torch.testing.assert_close(quantized.to(torch.float32).cpu(), expected, atol=0, rtol=0)
