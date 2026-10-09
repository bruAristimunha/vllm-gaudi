# SPDX-License-Identifier: Apache-2.0
import torch

from vllm_gaudi.ops.hpu_kda_pytorch import hpu_chunk_kda


def reference(q, k, v, gate, beta, state):
    q, k, v, gate, beta, state = (x.double() for x in (q, k, v, gate, beta, state))
    q = q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    k = k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    q = q / q.shape[-1]**.5
    outputs = []
    for step in range(q.shape[1]):
        state = state * gate[:, step].exp().unsqueeze(-1)
        residual = v[:, step] - torch.einsum("bhd,bhdv->bhv", k[:, step], state)
        state = state + torch.einsum("bhd,bhv->bhdv", k[:, step], residual * beta[:, step, :, None])
        outputs.append(torch.einsum("bhd,bhdv->bhv", q[:, step], state))
    return torch.stack(outputs, 1), state


def test_prefill_decode_matches_sequential_recurrence():
    for batch, length, decay in [(1, 3, .5), (2, 65, 5.), (1, 129, .01)]:
        generator = torch.Generator().manual_seed(length)
        shape = (batch, length + 1, 2, 128)
        q, k, v = [torch.randn(shape, generator=generator).bfloat16() for _ in range(3)]
        gate = -torch.rand(shape, generator=generator) * decay
        beta = torch.rand(shape[:-1], generator=generator)
        state = torch.randn(batch, 2, 128, 128, generator=generator) * .01
        expected, expected_state = reference(q, k, v, gate, beta, state)
        args = [x[:, :length].contiguous().to('hpu') for x in (q, k, v, gate, beta)]
        out, actual_state = hpu_chunk_kda(*args, initial_state=state.to('hpu'))
        args = [x[:, length:].contiguous().to('hpu') for x in (q, k, v, gate, beta)]
        last, actual_state = hpu_chunk_kda(*args, initial_state=actual_state)
        assert last.device.type == actual_state.device.type == 'hpu'
        actual = torch.cat([out, last], dim=1).cpu().float()
        torch.testing.assert_close(actual, expected.bfloat16().float(), atol=2e-4, rtol=.01)
        torch.testing.assert_close(actual_state.cpu(), expected_state.float(), atol=1e-4, rtol=.001)
