# SPDX-License-Identifier: Apache-2.0
import torch

from vllm_gaudi.ops.hpu_glm_keypool import fwht128_quant_values, select_pooled_tokens


def reference(q, weights, keys, gates, ape, positions, topk):
    hadamard = torch.ones(1, 1)
    for _ in range(7):
        hadamard = torch.cat([torch.cat([hadamard, hadamard], 1), torch.cat([hadamard, -hadamard], 1)], 0)
    pooled = (torch.softmax(gates.view(-1, 4, 128).float() + ape, 1) * keys.view(-1, 4, 128).float()).sum(1)
    transformed = (pooled.bfloat16().float() @ hadamard / 128**0.5).bfloat16().float()
    scale = torch.exp2(torch.ceil(torch.log2(transformed.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448)))
    pooled = (transformed / scale).to(torch.float8_e4m3fn).float() * scale
    result = []
    for row, pos in enumerate(positions.tolist()):
        count = (pos + 1) // 4
        score = ((q[row].float() @ pooled[:count].T).relu() * weights[row, :, None]).sum(0)
        pools = score.topk(min(topk // 4, count)).indices.tolist()
        result.append((set([p * 4 + j for p in pools for j in range(4)] + list(range(count * 4, pos + 1))), score))
    return result


def test_causal_pool_selection_and_tails():
    torch.manual_seed(4)
    keys = torch.randn(28, 128).bfloat16()
    gates = torch.randn(28, 128).bfloat16()
    ape = torch.randn(4, 128)
    q = torch.randn(28, 2, 128).bfloat16()
    weights = torch.rand(28, 2)
    positions = torch.arange(28)
    expected = reference(q, weights, keys, gates, ape, positions, 8)
    actual = select_pooled_tokens(*[x.to('hpu') for x in (q, weights, keys, gates, ape, positions)], 8).cpu()
    for index, (row, (wanted, scores)) in enumerate(zip(actual, expected)):
        selected = set(row[row >= 0].tolist())
        count = (index + 1) // 4
        tail = set(range(count * 4, index + 1))
        assert len(selected) == len(wanted) and tail <= selected
        pools = sorted({token // 4 for token in selected - tail})
        assert selected - tail == {p * 4 + j for p in pools for j in range(4)}
        assert all(p < count for p in pools)
        # ReLU can produce exact ties; different backend top-k tie orders are valid.
        torch.testing.assert_close(scores[pools].sort().values, scores.topk(min(2, count)).values.sort().values)


def test_chunk_boundaries_and_quantization():
    torch.manual_seed(11)
    keys = torch.randn(24, 128).bfloat16()
    gates = torch.randn_like(keys)
    ape = torch.randn(4, 128)
    q = torch.randn(24, 2, 128).bfloat16()
    weights = torch.rand(24, 2)
    pos = torch.arange(24)
    hpu = [x.to('hpu') for x in (q, weights, keys, gates, ape, pos)]
    full = select_pooled_tokens(*hpu, 8).cpu()
    for end in (1, 3, 4, 9, 17, 24):
        got = select_pooled_tokens(hpu[0][end - 1:end], hpu[1][end - 1:end], hpu[2][:end], hpu[3][:end], hpu[4],
                                   hpu[5][end - 1:end], 8).cpu()[0]
        assert set(got[got >= 0].tolist()) == set(full[end - 1][full[end - 1] >= 0].tolist())
    zero, scale = fwht128_quant_values(torch.zeros(3, 128, device='hpu', dtype=torch.bfloat16))
    assert torch.count_nonzero(zero).item() == 0
    assert torch.isfinite(scale).all().item()
