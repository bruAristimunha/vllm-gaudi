# SPDX-License-Identifier: Apache-2.0
import torch

from vllm.model_executor.kernels.mhc import mhc_pre_torch
from vllm_gaudi.ops.hpu_glm_mhc import hpu_mhc_pre


def test_mhc_sinkhorn_preserves_native_projection_and_rounding():
    with torch.inference_mode():
        for tokens, seed, magnitude in ((1, 819, 1.), (256, 819, 1.), (1, 991, 1.), (3, 56, 0.), (128, 61, 1.), (1, 715,
                                                                                                                 10.)):
            generator = torch.Generator().manual_seed(seed)
            residual = torch.randn(tokens, 4, 4096, generator=generator, dtype=torch.bfloat16) * magnitude
            mixing = torch.randn(24, 4 * 4096, generator=generator) * .01
            scale = torch.tensor([.2, .4, .8])
            base = torch.randn(24, generator=generator)
            args = (residual, mixing, scale, base, 1e-6, 1e-6, 1e-6, 2., 20)
            hpu_args = tuple(x.to('hpu') if isinstance(x, torch.Tensor) else x for x in args)
            expected = mhc_pre_torch(*hpu_args)
            actual = hpu_mhc_pre(*hpu_args)
            cpu = mhc_pre_torch(*args)
            assert all(x.device.type == 'hpu' and torch.isfinite(x).all().item() for x in actual)
            # The optimization touches only the FP32 combination matrix.
            torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
            torch.testing.assert_close(actual[2], expected[2], atol=0, rtol=0)
            torch.testing.assert_close(actual[1], expected[1], atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(actual[1].cpu(), cpu[1], atol=2e-6, rtol=2e-5)
            assert actual[1].shape == (tokens, 4, 4)
            torch.testing.assert_close(actual[1].sum(-2).cpu(), torch.ones(tokens, 4), atol=1e-5, rtol=0)


def test_hpu_mhc_post_bit_exact():
    from vllm.model_executor.kernels.mhc import mhc_post_torch
    from vllm_gaudi.ops.hpu_glm_mhc import hpu_mhc_post

    with torch.inference_mode():
        for tokens in (1, 2, 8, 33, 128):
            for seed in (1, 42, 99):
                gen = torch.Generator().manual_seed(seed)
                res = torch.randn(tokens, 4, 4096, dtype=torch.bfloat16, generator=gen).to("hpu")
                x = torch.randn(tokens, 4096, dtype=torch.bfloat16, generator=gen).to("hpu")
                post = torch.rand(tokens, 4, 1, dtype=torch.float32, generator=gen).to("hpu")
                comb = torch.rand(tokens, 4, 4, dtype=torch.float32, generator=gen).to("hpu")

                ref = mhc_post_torch(x, res, post, comb)
                actual = hpu_mhc_post(x, res, post, comb)

                assert actual.device.type == "hpu"
                torch.testing.assert_close(actual, ref, atol=0, rtol=0)

