"""One real8-expert MoE with K2gate/up and K3down; no full-model quality claim."""
import contextlib
import copy
import hashlib
import json
import os
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors import safe_open
from vllm.config import set_current_vllm_config
from vllm.engine.arg_utils import EngineArgs
from vllm.distributed import init_distributed_environment, ensure_model_parallel_initialized
from vllm.forward_context import set_forward_context
from vllm.models.glm5next.nvidia.model import Glm5NextMoE
from vllm.model_executor.layers.quantization.exl3 import apply_exl3_experts
from vllm.model_executor.weight_transfer import flush_weight_transfers

ROOT=Path('/out'); ROOT.mkdir(exist_ok=True)
report={'state':'STARTED','quality_claim':False,'full_model_claim':False,'layers':[]}
def save(): (ROOT/'gpu-smoke.json').write_text(json.dumps(report,indent=2))
save(); started=time.monotonic()
index=json.loads(Path('/model/model.safetensors.index.json').read_text())['weight_map']
report['index_sha256']=hashlib.sha256(Path('/model/model.safetensors.index.json').read_bytes()).hexdigest()
c=EngineArgs(model='/model',tokenizer='/model',skip_tokenizer_init=True,dtype='bfloat16',quantization='exl3',max_model_len=4096,max_num_seqs=1,max_num_batched_tokens=32,enable_prefix_caching=False).create_engine_config()
config=c.model_config.hf_config.get_text_config()
kept={5:list(range(32,40))}
c.quant_config.raw_config.pop('layer_bits',None)
c.quant_config.raw_config['layer_projection_bits']={'5':{'down_proj':3}}
report['projection_bits']=[2,2,3]
report['native_dispatch_k']=0
init_distributed_environment(world_size=1,rank=0,distributed_init_method='file:///tmp/nonuniform-smoke-dist',local_rank=0,backend='gloo')
with set_current_vllm_config(c): ensure_model_parallel_initialized(1,1)
torch.set_default_dtype(torch.bfloat16)
torch.manual_seed(42)
models=[]
graphs=[]
with set_current_vllm_config(c), torch.device('cuda'), contextlib.ExitStack() as stack:
    handles={}
    k2_index={}
    for part in range(2):
        path=f'/k2-source/layer-05-part-{part}.safetensors'
        handles[path]=stack.enter_context(safe_open(path,framework='pt',device='cpu'))
        for key in handles[path].keys(): k2_index[key]=path
    def get(name):
        use_k2='.layers.5.mlp.experts.' in name and any('.'+proj+'.' in name for proj in ('gate_proj','up_proj'))
        path=k2_index[name] if use_k2 else '/model/'+index[name]
        if path not in handles: handles[path]=stack.enter_context(safe_open(path,framework='pt',device='cpu'))
        return handles[path].get_tensor(name)
    for layer,ids in kept.items():
        cfg=copy.copy(config)
        cfg.n_routed_experts=len(ids)
        prefix=f'model.language_model.layers.{layer}.mlp.'
        m=Glm5NextMoE(cfg,c.parallel_config,c.quant_config,prefix=f'language_model.model.layers.{layer}.mlp')
        router=get(prefix+'gate.weight')[ids]
        bias=get(prefix+'gate.e_score_correction_bias')[ids]
        m.gate.weight.data.copy_(router)
        m.gate.e_score_correction_bias.data.copy_(bias)
        assert torch.equal(m.gate.weight.cpu(),router)
        assert torch.equal(m.gate.e_score_correction_bias.cpu(),bias)
        for proj,sid in [('gate_proj',0),('up_proj',1)]:
            p=m.shared_experts.gate_up_proj.weight
            p.weight_loader(p,get(prefix+'shared_experts.'+proj+'.weight'),sid)
        p=m.shared_experts.down_proj.weight
        p.weight_loader(p,get(prefix+'shared_experts.down_proj.weight'))
        r=m.experts.routed_experts
        loaded=0
        for new_id,original_id in enumerate(ids):
            for proj,packed,sid in [('gate_proj','w13','w1'),('up_proj','w13','w3'),('down_proj','w2','w2')]:
                for rank in range(4):
                    for field in ('trellis','suh','svh','mcg'):
                        old=prefix+f'experts.{original_id}.{proj}.rank{rank}.{field}'
                        remapped=prefix+f'experts.{new_id}.{proj}.rank{rank}.{field}'
                        tensor=get(old)
                        param=getattr(getattr(r,f'{packed}_rank{rank}'),field)
                        param.weight_loader(param,tensor,f'experts.routed_experts.{packed}_rank{rank}.{field}',expert_id=new_id,shard_id=sid)
                        loaded+=1
        flush_weight_transfers()
        r.quant_method.process_weights_after_loading(r)
        assert len(r._exl3_inners)==len(ids)*4
        assert r._exl3_physical_experts==len(ids)
        assert r.w13_rank0.trellis.shape[-1]==32
        assert r.w2_rank0.trellis.shape[-1]==48
        assert r._exl3_projection_bits==(2,2,3)
        assert all((v['gate'].K,v['up'].K,v['down'].K)==(2,2,3) for v in r._exl3_inners)
        models.append((layer,m))
        report['layers'].append({'layer':layer,'experts':len(ids),'retained_original_ids':ids,'loaded_expert_tensors':loaded,'router_rows_exact':True,'fused_pointer_tables':bool(r._exl3_ptrs)})
        save();print('LOADED',layer,len(ids),loaded,flush=True)
    for row,(layer,m) in zip(report['layers'],models):
        r=m.experts.routed_experts
        x=torch.randn(1,4096,device='cuda',dtype=torch.bfloat16)*.1
        logits=m.gate(x)[0]
        scores=logits.float().sigmoid()
        route_ids=(scores+m.gate.e_score_correction_bias.float()).topk(8,dim=-1).indices
        weights=scores.gather(1,route_ids); weights=weights/weights.sum(-1,keepdim=True)*config.routed_scaling_factor
        # Both paths consume the same retained-router selection. The loop uses
        # individual real EXL3 Linear kernels; fused uses the production MoE kernel.
        fused=apply_exl3_experts(x,route_ids,weights,r,fused=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profile:
            apply_exl3_experts(x,route_ids,weights,r,fused=True)
            torch.cuda.synchronize()
        profile.export_chrome_trace(str(ROOT/'kernel-trace.json'))
        kernels=sorted({e.name for e in profile.events() if 'exl3_moe_kernel' in e.name})
        assert any('exl3_moe_kernel<0,' in name for name in kernels),kernels
        report['observed_moe_kernels']=kernels
        loop=apply_exl3_experts(x,route_ids,weights,r,fused=False)
        torch.testing.assert_close(fused,loop,atol=.001,rtol=.01)
        # Independent reconstructed-weight FP32 reference for those same routes.
        ref=torch.zeros_like(x,dtype=torch.float32)
        for pos,e in enumerate(route_ids[0].tolist()):
            for rank in range(4):
                inner=r._exl3_inners[e*4+rank]
                g=x.float()@inner['gate'].get_weight_tensor().float()
                u=x.float()@inner['up'].get_weight_tensor().float()
                a=F.silu(g.clamp(max=10))*u.clamp(min=-10,max=10)
                ref+=(a@inner['down'].get_weight_tensor().float())*weights[0,pos]
        torch.testing.assert_close(fused.float(),ref,atol=.002,rtol=.02)
        # Production router+shared-expert runner, then real graph capture/replay.
        with set_forward_context(None,c,num_tokens=1):
            eager=m(x)
            shared=m.shared_experts(x)
            full_reference=ref+shared.float()
            torch.testing.assert_close(eager.float(),full_reference,atol=.002,rtol=.02)
            torch.cuda.synchronize()
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3): m(x)
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph): output=m(x)
            for _ in range(3): graph.replay()
            torch.cuda.synchronize()
        assert torch.isfinite(eager).all() and torch.isfinite(output).all()
        torch.testing.assert_close(eager,output,atol=.002,rtol=.01)
        row.update({'route_ids':route_ids[0].tolist(),'fused_vs_loop_max_abs':float((fused-loop).abs().max()),'fused_vs_reconstructed_fp32_max_abs':float((fused.float()-ref).abs().max()),'graph_replays':3,'graph_vs_eager_max_abs':float((output-eager).abs().max()),'full_runner_vs_reference_max_abs':float((eager.float()-full_reference).abs().max()),'finite':True})
        graphs.append((graph,output,eager.clone()))
        save(); print('PASSED',json.dumps(row),flush=True)
    for _ in range(3):
        for graph,output,expected in graphs:
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(output,expected,atol=.002,rtol=.01)
    report['alternating_layer_graph_replays']=3
report.update(state='GPU_SMOKE_PASS',elapsed_seconds=time.monotonic()-started,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),global_experts_unchanged=config.n_routed_experts,native_mtp_not_loaded=True)
save();print(json.dumps(report),flush=True)
