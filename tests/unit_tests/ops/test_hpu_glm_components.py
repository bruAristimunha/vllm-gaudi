# SPDX-License-Identifier: Apache-2.0
"""Component regressions on one assigned HPU, not full-model qualification."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F


class GlmComponents(unittest.TestCase):

    def test_host_metadata_reuses_transfers_only_within_one_forward(self):
        from vllm_gaudi.models import glm5_next as glm

        source = Mock()
        source.cpu.side_effect = [torch.tensor([[2, 3], [5, 7]]), torch.tensor([[11, 13], [17, 19]])]
        metadata = SimpleNamespace(load_indices_tensor=source)
        first = SimpleNamespace(attn_metadata=metadata, additional_kwargs={})
        second = SimpleNamespace(attn_metadata=metadata, additional_kwargs={})
        with patch.object(glm, "get_forward_context", return_value=first):
            self.assertEqual(glm._cpu_metadata("load_indices_tensor")[0].tolist(), [2, 3])
            self.assertEqual(glm._cpu_metadata("load_indices_tensor")[1].tolist(), [5, 7])
        self.assertEqual(source.cpu.call_count, 1)
        with patch.object(glm, "get_forward_context", return_value=second):
            self.assertEqual(glm._cpu_metadata("load_indices_tensor")[1].tolist(), [17, 19])
        self.assertEqual(source.cpu.call_count, 2)

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

    def test_indexer_decode_avoids_position_host_sync(self):
        from vllm_gaudi.models import glm5_next as glm

        indexer = glm.HpuIndexer.__new__(glm.HpuIndexer)
        indexer.topk_tokens = 4
        indexer.topk_indices_buffer = torch.empty(1, 7, dtype=torch.long, device="hpu")
        indexer.wq_b = lambda x: (torch.zeros(1, 128, device="hpu", dtype=torch.bfloat16), None)
        indexer.n_head = 1
        indexer.wk_weights_proj = Mock()
        indexer.wk_weights_proj.return_value = (torch.zeros(1, 128, device="hpu", dtype=torch.bfloat16), None)
        indexer.wk_weights_proj.weight = torch.zeros(256, 128, device="hpu")
        indexer.k_norm = SimpleNamespace(weight=torch.ones(128, device="hpu"), bias=torch.zeros(128, device="hpu"), eps=1e-5)
        indexer.softmax_scale = 1.0
        indexer.index_kpool_compress_gate = torch.zeros(128, 128, device="hpu", dtype=torch.bfloat16)
        indexer.index_kpool_compress_ape = torch.zeros(4, 128, device="hpu", dtype=torch.bfloat16)
        indexer.k_cache = SimpleNamespace(kv_cache=torch.zeros(100, 256, device="hpu", dtype=torch.bfloat16))

        cached_physical = torch.tensor([10, 11, 12, 13], device="hpu", dtype=torch.long)
        cached_pools = torch.zeros(1, 128, device="hpu", dtype=torch.float32)
        indexer._cached_pools = (cached_pools, cached_physical)

        positions = torch.tensor([4], device="hpu")
        positions_mock = Mock(wraps=positions)
        positions_mock.item.side_effect = AssertionError("Device sync forced via .item()!")
        positions_mock.reshape.return_value = positions_mock

        meta = SimpleNamespace(is_prompt=False, slot_mapping=torch.tensor([14], device="hpu"))
        hidden = torch.zeros(1, 128, device="hpu", dtype=torch.bfloat16)

        with patch.object(glm, "get_forward_context", return_value=SimpleNamespace(attn_metadata=meta)):
            out = indexer.forward(hidden, hidden, positions_mock, None)
        self.assertEqual(out.device.type, "hpu")
        self.assertEqual(indexer._cached_pools[1].numel(), 5)

