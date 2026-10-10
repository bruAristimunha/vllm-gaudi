# SPDX-License-Identifier: Apache-2.0
"""Eager HPU GLM-5.3 adapter retaining the upstream model and weight loader."""

import torch
import torch.nn.functional as F

from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.models.glm5next.nvidia import attention, kda, model
from vllm_gaudi.ops.hpu_glm_keypool import fwht128_quant_values, select_pooled_tokens
from vllm_gaudi.ops.hpu_glm_mhc import hpu_mhc_pre
from vllm_gaudi.ops.hpu_kda_pytorch import hpu_chunk_kda


def _cpu_metadata(name):
    context = get_forward_context()
    # Scheduling values are shared by layers, but never by forward passes.
    cache = context.additional_kwargs.setdefault("glm_cpu_metadata", {})
    if name not in cache:
        cache[name] = getattr(context.attn_metadata, name).cpu()
    return cache[name]


class HpuIndexerCache(DeepseekV32IndexerCache):

    def __init__(self, *, head_dim, dtype, prefix, cache_config, index_kpool):
        # shortcut: raw K and gates use token slots; compress storage after full-model parity.
        super().__init__(512, torch.bfloat16, prefix, cache_config)


class HpuIndexer(attention.Indexer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        context = get_current_vllm_config().compilation_config.static_forward_context
        context.pop(f"{self.prefix}.tail_cache", None)
        self.tail_cache = None
        self._cached_pools = None

    @torch._dynamo.disable
    def forward(self, hidden_states, qr, positions, rotary_emb):
        meta = get_forward_context().attn_metadata
        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        q = self.wq_b(qr)[0].reshape(-1, 128).contiguous()
        q, scale = fwht128_quant_values(q)
        q = q.view(-1, self.n_head, 128)
        kw = self.wk_weights_proj(x)[0]
        keys = F.layer_norm(kw[:, :128].float(), (128, ), self.k_norm.weight.float(), self.k_norm.bias.float(),
                            self.k_norm.eps).to(x.dtype)
        weights = x.float() @ self.wk_weights_proj.weight[128:].float().T
        weights = weights * scale.view(-1, self.n_head) * (self.softmax_scale * self.n_head**-0.5)
        gates = F.linear(x, self.index_kpool_compress_gate)
        cache = self.k_cache.kv_cache
        if isinstance(cache, (tuple, list)):
            cache = cache[0]
        out = self.topk_indices_buffer
        out[:q.shape[0]].fill_(-1)
        if meta is None or cache is None or cache.numel() == 0:
            return out
        slots = meta.slot_mapping.reshape(-1)
        cache.index_copy_(0, slots, F.pad(torch.cat([keys, gates], dim=-1), (0, cache.shape[-1] - 256)))
        positions = positions.reshape(-1)
        # Only scheduling metadata crosses to the host; model values stay on HPU.
        if meta.is_prompt:
            self._cached_pools = None
            lengths = _cpu_metadata("seq_lens_tensor").tolist()
            batch = len(lengths)
            stride = slots.numel() // batch
            contexts = _cpu_metadata("context_lens_tensor").tolist()
            blocks = _cpu_metadata("block_list").reshape(batch,
                                                         -1).tolist() if meta.block_list is not None else [[]] * batch
            requests = []
            for row, length in enumerate(lengths):
                prefix = [b * meta.block_size + j for b in blocks[row] for j in range(meta.block_size)][:contexts[row]]
                current = slots[row * stride:row * stride + length]
                prefix = torch.tensor(prefix, device=x.device, dtype=torch.long)
                requests.append((row * stride, int(length), torch.cat([prefix, current])))
            for start, length, physical in requests:
                if length == 0 or physical.numel() == 0:
                    continue
                raw = cache.index_select(0, physical)
                logical = select_pooled_tokens(q[start:start + length], weights[start:start + length], raw[:, :128],
                                               raw[:, 128:256], self.index_kpool_compress_ape,
                                               positions[start:start + length], self.topk_tokens)
                selected = physical[logical.clamp_min(0)].masked_fill(logical < 0, -1)
                out[start:start + length, :selected.shape[1]] = selected.to(out.dtype)
                if batch == 1:
                    count = physical.shape[0] // 4
                    if count > 0:
                        pooled_keys = raw[:count * 4, :128].view(count, 4, 128).float()
                        logits = raw[:count * 4, 128:256].view(count, 4, 128).float() + self.index_kpool_compress_ape.float()
                        pooled = (logits.softmax(dim=1) * pooled_keys).sum(1).to(torch.bfloat16)
                        values, scales = fwht128_quant_values(pooled.contiguous())
                        cached_pools = values.float() * scales
                    else:
                        cached_pools = torch.empty(0, 128, device=x.device, dtype=torch.float32)
                    self._cached_pools = (cached_pools, physical)
            return out

        if q.shape[0] == 1 and self._cached_pools is not None:
            cached_pools, cached_physical = self._cached_pools
            pos = positions[0].item()
            if cached_physical.shape[0] == pos:
                cached_physical = torch.cat([cached_physical, slots[:1]])
                total_tokens = cached_physical.shape[0]
                count = total_tokens // 4
                if count > cached_pools.shape[0]:
                    new_slots = cached_physical[(count - 1) * 4 : count * 4]
                    raw_new = cache.index_select(0, new_slots)
                    k_new = raw_new[:, :128].view(1, 4, 128).float()
                    g_new = raw_new[:, 128:256].view(1, 4, 128).float() + self.index_kpool_compress_ape.float()
                    p_new = (g_new.softmax(dim=1) * k_new).sum(1).to(torch.bfloat16)
                    v_new, s_new = fwht128_quant_values(p_new.contiguous())
                    new_pool = v_new.float() * s_new
                    cached_pools = torch.cat([cached_pools, new_pool], dim=0)
                self._cached_pools = (cached_pools, cached_physical)

                width = self.topk_tokens + 3
                result = torch.full((1, width), -1, device=x.device, dtype=torch.long)
                if total_tokens <= self.topk_tokens:
                    ids = torch.arange(total_tokens, device=x.device)
                    result[:, :ids.numel()] = ids[None, :].masked_fill(ids[None, :] > pos, -1)
                else:
                    take = min(self.topk_tokens // 4, count)
                    pool_ids = torch.arange(count, device=x.device)
                    scores = torch.matmul(q.float(), cached_pools.T).relu()
                    scores = (scores * weights[:, :, None]).sum(1)
                    completed = (pos + 1) // 4
                    scores.masked_fill_(pool_ids[None, :] >= completed, -torch.inf)
                    ranked, selected_p = scores.topk(take, dim=-1)
                    offsets = torch.arange(4, device=x.device)
                    tokens = (selected_p[:, :, None] * 4 + offsets).reshape(1, -1)
                    tokens.masked_fill_(ranked.isneginf().repeat_interleave(4, dim=1), -1)
                    result[:, :take * 4] = tokens
                    tail = completed * 4 + offsets[:3]
                    result[:, self.topk_tokens:] = tail.unsqueeze(0).masked_fill(tail.unsqueeze(0) > pos, -1)
                selected = cached_physical[result.clamp_min(0)].masked_fill(result < 0, -1)
                out[:1, :selected.shape[1]] = selected.to(out.dtype)
                return out

        block_list = _cpu_metadata("block_list").tolist()
        owners = _cpu_metadata("block_groups").tolist()
        usage = _cpu_metadata("block_usage").round().long().tolist()
        requests = []
        for row in range(q.shape[0]):
            physical = [
                b * meta.block_size + j for b, owner, used in zip(block_list, owners, usage) if owner == row
                for j in range(used)
            ]
            requests.append((row, 1, torch.tensor(physical, device=x.device, dtype=torch.long)))
        for start, length, physical in requests:
            if length == 0 or physical.numel() == 0:
                continue
            raw = cache.index_select(0, physical)
            logical = select_pooled_tokens(q[start:start + length], weights[start:start + length], raw[:, :128],
                                           raw[:, 128:256], self.index_kpool_compress_ape,
                                           positions[start:start + length], self.topk_tokens)
            selected = physical[logical.clamp_min(0)].masked_fill(logical < 0, -1)
            out[start:start + length, :selected.shape[1]] = selected.to(out.dtype)
        return out


class HpuKda(kda.Glm5NextLinearAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_group_idx = None

    @torch._dynamo.disable
    def forward(self, hidden_states, positions):
        meta = get_forward_context().attn_metadata
        if meta is None:
            return torch.zeros_like(hidden_states)
        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        projected = self.in_proj_qkvbfg_a(x)[0]
        qkv, beta, fa, ga = projected.split(
            [3 * self.local_projection_size, self.local_num_heads, self.head_dim, self.head_dim], dim=-1)
        decay = self.f_b_proj(fa)[0].float() + self.dt_bias
        decay = self.kda_lower_bound * torch.sigmoid(
            self.A_log.float().exp() * decay.view(1, -1, self.local_num_heads, self.head_dim))
        beta = beta.float().sigmoid()
        conv, recurrent = self.kv_cache
        load = _cpu_metadata("load_indices_tensor")
        store = _cpu_metadata("store_indices_tensor")
        if load.ndim > 1:
            if self.cache_group_idx is None:
                raise RuntimeError("KDA cache group was not bound")
            load = load[self.cache_group_idx]
            store = store[self.cache_group_idx]
        load_ids, store_ids = load.tolist(), store.tolist()
        batch = len(load_ids)
        stride = x.shape[0] // batch
        lengths = _cpu_metadata("seq_lens_tensor").tolist() if meta.is_prompt else [1] * batch
        initial = _cpu_metadata("has_initial_states_p").tolist() if meta.is_prompt else [True] * batch
        conv_weights = torch.cat([m.weight.squeeze(1) for m in (self.q_conv1d, self.k_conv1d, self.v_conv1d)])
        core = torch.zeros((x.shape[0], self.local_num_heads, self.head_dim), device=x.device, dtype=x.dtype)
        for row, (source, target, length, has_state) in enumerate(zip(load_ids, store_ids, lengths, initial)):
            if source < 0 or target < 0 or length == 0:
                continue
            start, end = row * stride, row * stride + int(length)
            history = conv[source]
            if not self._conv_state_dim_first:
                history = history.T
            history = history[:, -(self.conv_size - 1):].float()
            if not has_state:
                history = torch.zeros_like(history)
            stream = torch.cat([history, qkv[start:end].float().T], dim=1)
            filtered = F.conv1d(stream.unsqueeze(0), conv_weights.float().unsqueeze(1), groups=stream.shape[0])
            filtered = F.silu(filtered.squeeze(0).T).to(x.dtype)
            q, k, v = [
                t.view(1, int(length), self.local_num_heads, self.head_dim)
                for t in filtered.split(self.local_projection_size, dim=-1)
            ]
            state = recurrent[source:source + 1] if has_state else torch.zeros_like(recurrent[source:source + 1])
            output, final = hpu_chunk_kda(q,
                                          k,
                                          v,
                                          decay[:, start:end],
                                          beta[start:end].unsqueeze(0),
                                          initial_state=state,
                                          chunk_size=64)
            core[start:end] = output[0]
            recurrent[target].copy_(final[0])
            tail = stream[:, -(self.conv_size - 1):].to(conv.dtype)
            if self._conv_state_dim_first:
                conv[target, :, -tail.shape[1]:].copy_(tail)
            else:
                conv[target, -tail.shape[1]:].copy_(tail.T)
        gate = self.g_b_proj(ga)[0].view_as(core).float().sigmoid()
        normed = core.float() * torch.rsqrt(core.float().square().mean(-1, keepdim=True) + self.o_norm.eps)
        normed = (normed * self.o_norm.weight.float() * gate).to(x.dtype)
        return self.o_proj(normed.reshape(x.shape[0], -1))[0].view_as(hidden_states)


class HpuMla(attention.Glm5NextMLAAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mla_attn.mla_attn.hpu_pooled_sparse_prefill = True


class HpuDecoderLayer(model.Glm5NextDecoderLayer):

    def __init__(self, *args, vllm_config, **kwargs):
        if not vllm_config.model_config.enforce_eager:
            raise NotImplementedError("GLM HPU currently requires --enforce-eager")
        if vllm_config.speculative_config is not None:
            raise NotImplementedError("GLM HPU speculative decoding is not qualified")
        if vllm_config.cache_config.enable_prefix_caching and vllm_config.scheduler_config.max_num_seqs != 1:
            raise NotImplementedError("GLM HPU prefix caching currently requires max_num_seqs=1")
        if vllm_config.parallel_config.enable_expert_parallel:
            raise NotImplementedError("GLM HPU currently supports tensor parallelism without expert parallelism")
        warmup = vllm_config.kernel_config.enable_jit_warmup
        vllm_config.kernel_config.enable_jit_warmup = False
        try:
            super().__init__(*args, vllm_config=vllm_config, **kwargs)
        finally:
            vllm_config.kernel_config.enable_jit_warmup = warmup

    def hc_pre(self, x, hc_fn, hc_scale, hc_base, norm_weight=None, norm_eps=0.0):
        post, comb, layer_input = hpu_mhc_pre(x, hc_fn, hc_scale, hc_base, self.rms_norm_eps, self.hc_eps, self.hc_eps,
                                              self.mhc_post_mult_value, self.mhc_sinkhorn_iterations)
        # The upstream native mHC fallback ignores the fused norm arguments.
        if norm_weight is not None:
            value = layer_input.float()
            value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + norm_eps)
            layer_input = (value * norm_weight.float()).to(layer_input.dtype)
        return post, comb, layer_input

    def hc_fused_post_pre(self, x, residual, post, comb, hc_fn, hc_scale, hc_base, norm_weight=None, norm_eps=0.0):
        residual = self.hc_post(x, residual, post, comb)
        post, comb, value = self.hc_pre(residual, hc_fn, hc_scale, hc_base, norm_weight, norm_eps)
        return residual, post, comb, value


attention.Glm5NextIndexerCache = HpuIndexerCache
attention.Indexer = HpuIndexer
model.Glm5NextLinearAttention = HpuKda
model.Glm5NextMLAAttention = HpuMla
model.Glm5NextDecoderLayer = HpuDecoderLayer
Glm5NextForConditionalGeneration = model.Glm5NextForConditionalGeneration
Glm5NextForCausalLM = model.Glm5NextForCausalLM
