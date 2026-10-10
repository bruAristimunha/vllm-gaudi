# SPDX-License-Identifier: Apache-2.0
"""Small seeded GLM architecture: generation, chunking, and request reuse checks."""
import json
import os
import pathlib
import sys

os.environ['VLLM_FLATTEN_INPUT'] = 'true'
os.environ['VLLM_CONTIGUOUS_PA'] = '0'
os.environ['VLLM_KDA_SCAN'] = 'seq'
os.environ['VLLM_GDN_COMPUTE_FP32'] = '1'
os.environ['VLLM_SKIP_WARMUP'] = 'true'
from vllm import LLM, SamplingParams
import torch
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader

_original_dummy_load = DummyModelLoader.load_weights


def initialize_test_weights(self, model, model_config):
    _original_dummy_load(self, model, model_config)
    generator = torch.Generator().manual_seed(819)
    with torch.no_grad():
        for name, parameter in sorted(model.named_parameters()):
            if any(term in name for term in ('A_log', 'dt_bias', '_base', 'bias')):
                value = torch.zeros(parameter.shape)
            elif '_scale' in name:
                value = torch.full(parameter.shape, .2)
            elif parameter.ndim == 1:
                value = torch.ones(parameter.shape)
            else:
                value = torch.randn(parameter.shape, generator=generator) / max(parameter.shape[-1], 1)**.5
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


DummyModelLoader.load_weights = initialize_test_weights


def main():
    prefix_cache = os.environ.get('GLM_SMOKE_PREFIX_CACHE') == '1'
    source = os.environ.get('GLM_SMOKE_CONFIG')
    if source is None:
        from huggingface_hub import hf_hub_download
        source = hf_hub_download('zai-org/GLM-5.3-Flash',
                                 'config.json',
                                 revision='eb9eb208eb0d988989d07a6a12d0fdeb5f52574a')
    config = json.loads(pathlib.Path(source).read_text())
    config.pop('quantization_config', None)
    text = config['text_config']
    text.update(hidden_size=256,
                intermediate_size=512,
                moe_intermediate_size=256,
                n_routed_experts=4,
                num_experts_per_tok=2,
                num_attention_heads=2,
                num_key_value_heads=2,
                num_hidden_layers=4,
                q_lora_rank=128,
                kv_lora_rank=512,
                index_n_heads=2,
                index_topk=8,
                num_nextn_predict_layers=0)
    text['layer_types'] = text['layer_types'][:4]
    text['mlp_layer_types'] = text['mlp_layer_types'][:4]
    text['indexer_types'] = text['indexer_types'][:4]
    text['linear_attn_config'].update(num_heads=2, kda_layers=[0, 1, 2], full_attn_layers=[3])
    config['vision_config'].update(hidden_size=128,
                                   num_heads=2,
                                   depth=1,
                                   intermediate_size=256,
                                   out_hidden_size=256,
                                   projection_intermediate_size=256)
    import tempfile
    directory = tempfile.TemporaryDirectory(prefix='glm53-smoke-')
    path = pathlib.Path(directory.name)
    (path / 'config.json').write_text(json.dumps(config))
    llm = LLM(model=str(path),
              skip_tokenizer_init=True,
              load_format='dummy',
              dtype='bfloat16',
              enforce_eager=True,
              enable_prefix_caching=prefix_cache,
              max_model_len=768,
              max_num_seqs=1 if prefix_cache else 2,
              max_num_batched_tokens=int(sys.argv[1]) if len(sys.argv) > 1 else 256,
              gpu_memory_utilization=0.25,
              num_gpu_blocks_override=64,
              language_model_only=True,
              distributed_executor_backend='uni')
    prompts = [{'prompt_token_ids': [123, 456, 789] * 171}, {'prompt_token_ids': [789, 456, 123] * 87}]
    params = SamplingParams(temperature=0, max_tokens=6, ignore_eos=True, logprobs=1)
    outputs = llm.generate(prompts, params)
    serial = [llm.generate([prompt], params)[0] for prompt in prompts]
    assert [o.outputs[0].token_ids for o in outputs] == [o.outputs[0].token_ids for o in serial]
    if prefix_cache:
        assert all(o.num_cached_tokens > 0 for o in serial), 'Prefix reuse was not exercised'
        for cold, warm in zip(outputs, serial):
            for token, expected, actual in zip(cold.outputs[0].token_ids, cold.outputs[0].logprobs,
                                               warm.outputs[0].logprobs):
                assert abs(expected[token].logprob - actual[token].logprob) <= 1e-4
        print('PREFIX_CACHE_REUSE_PASS', [o.num_cached_tokens for o in serial], flush=True)
    import math
    for output in outputs:
        assert len(output.outputs[0].token_ids) == 6
        for step in output.outputs[0].logprobs:
            assert all(math.isfinite(item.logprob) for item in step.values())
        assert any(
            abs(item.logprob + math.log(154880)) > .01 for step in output.outputs[0].logprobs
            for item in step.values()), 'Degenerate uniform logits'
    print('TINY_GLM_GENERATION_PASS',
          json.dumps([{
              'tokens': o.outputs[0].token_ids,
              'logprobs': [{
                  str(t): p.logprob
                  for t, p in step.items()
              } for step in o.outputs[0].logprobs]
          } for o in outputs]),
          flush=True)


if __name__ == '__main__':
    main()
