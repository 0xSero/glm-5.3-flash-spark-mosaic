"""CPU source/tensor-contract tests; not a full vLLM import or CUDA ABI test."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import types
import unittest

import torch
from torch.nn.parameter import Parameter

HERE = Path(__file__).resolve().parents[1]
UPSTREAM = HERE.parent/'upstream-vllm'


def import_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


port = import_path('port_apply', HERE/'apply_port.py')
transport = import_path('jovian_weight_transport', UPSTREAM/'vllm/model_executor/weight_transfer.py')


def isolated_contract():
    # Execute actual selected method bodies on real CPU tensors. Omit unrelated
    # serving imports/base-class initialization; this is deliberately NOT an
    # import/ABI compatibility substitute or a deployed runtime shim.
    tree = ast.parse((HERE/'exl3.py').read_text())
    nodes = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in (
                '_narrow_tp', 'shard_exl3_col', 'shard_exl3_row',
                '_suffix_from_mapped_name', '_rank_from_mapped_name'):
            nodes.append(copy.deepcopy(node))
        elif isinstance(node, ast.ClassDef) and node.name in ('Exl3Config', 'Exl3MoEMethod'):
            node = copy.deepcopy(node)
            node.bases, node.decorator_list = [], []
            if node.name == 'Exl3MoEMethod':
                node.body = [f for f in node.body if isinstance(f, ast.FunctionDef)
                             and f.name in ('create_weights', '_load_exl3')]
                for f in node.body:
                    f.body = [s for s in f.body if not isinstance(s, ast.ImportFrom)]
            nodes.append(node)
    def attrs(param, values):
        for k, v in values.items():
            setattr(param, k, v)
    env = {'torch': torch, 'Parameter': Parameter, 'Any': object,
           'RANK_STACKED_TP': 4, 'RANK_STACKED_INTERMEDIATE': 512,
           'EXL3_SUFFIXES': ('trellis', 'suh', 'svh', 'mcg'),
           '_RANK_IN_NAME': re.compile(r'(?:^|[._])rank(?P<rank>\d+)(?=[._])'),
           'set_weight_attrs': attrs, 'get_tensor_model_parallel_world_size': lambda: 1,
           'get_tensor_model_parallel_rank': lambda: 0,
           **{name: getattr(transport, name) for name in
              ('copy_weight', 'get_file_tensor_source', 'materialize_weight')}}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 'isolated_exl3_contract', 'exec'), env)
    return env


class PortTests(unittest.TestCase):
    def test_apply_pinned_source_and_reject_existing_or_changed(self):
        manifest = json.loads((HERE/'PORT_MANIFEST.json').read_text())
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in manifest['source_sha256']:
                destination=root/name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(UPSTREAM/name,destination)
            result=port.apply(root)
            self.assertEqual(result['state'],'PORT_APPLIED_NOT_RUNTIME_VALIDATED')
            self.assertEqual((root/port.WRAPPER).read_bytes(),(HERE/'exl3.py').read_bytes())
            registry=(root/port.REGISTRY).read_text()
            self.assertEqual(registry.count('"exl3": Exl3Config'),1)
            with self.assertRaises(ValueError):port.apply(root)
            (root/port.REGISTRY).write_text('changed')
            with self.assertRaisesRegex(ValueError,'source changed'):port.apply(root)

    def test_actual_base_method_argument_compatibility(self):
        base=ast.parse((UPSTREAM/'vllm/model_executor/layers/fused_moe/fused_moe_method_base.py').read_text())
        wrapper=ast.parse((HERE/'exl3.py').read_text())
        def methods(tree, cls):
            return {n.name:n for c in tree.body if isinstance(c,ast.ClassDef) and c.name==cls
                    for n in c.body if isinstance(n,ast.FunctionDef)}
        a,b=methods(base,'FusedMoEMethodBase'),methods(wrapper,'Exl3MoEMethod')
        for name in ('create_weights','apply','get_fused_moe_quant_config'):
            self.assertEqual([x.arg for x in a[name].args.args], [x.arg for x in b[name].args.args])
        self.assertIn('flush_weight_transfers()',ast.unparse(b['process_weights_after_loading']).splitlines()[1])

    def test_config_k2_k3_and_guard(self):
        env=isolated_contract()
        for bits in (2,3):
            cfg=env['Exl3Config'].from_config({'bits':bits,'rank_stacked_tp':4,'quant_method':'exl3'})
            self.assertEqual(cfg.bits,bits)
            self.assertEqual(cfg.rank_stacked_tp,4)
        with self.assertRaises(ValueError):env['Exl3Config'].from_config({'bits':7})
        with self.assertRaises(ValueError):env['Exl3Config'].from_config({'bits':2,'rank_stacked_tp':2})

    def test_native_linear_dispatch_is_not_quantized(self):
        tree=ast.parse((HERE/'exl3.py').read_text())
        method=next(f for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='Exl3Config'
                    for f in c.body if isinstance(f,ast.FunctionDef) and f.name=='get_quant_method')
        method=copy.deepcopy(method)
        method.body=[x for x in method.body if not isinstance(x,ast.ImportFrom)]
        class Routed(torch.nn.Module):pass
        class Linear(torch.nn.Module):pass
        class Native:pass
        env={'torch':torch,'RoutedExperts':Routed,'LinearBase':Linear,
             'UnquantizedLinearMethod':Native,'Exl3MoEMethod':lambda moe,cfg:('packed',moe,cfg)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),
                     'actual_dispatch','exec'),env)
        quant=object()
        self.assertIsInstance(env['get_quant_method'](quant,Linear(),'shared_experts.gate_proj'),Native)
        self.assertIsNone(env['get_quant_method'](quant,torch.nn.Module(),'norm'))
        routed=Routed();routed.moe_config='test'
        self.assertEqual(env['get_quant_method'](quant,routed,'layers.3.mlp.experts')[0],'packed')

    def make_layer(self,bits=2,experts=2):
        env=isolated_contract()
        method=env['Exl3MoEMethod']()
        method.quant_config=env['Exl3Config'](bits=bits,rank_stacked_tp=4)
        method.bits=bits
        layer=torch.nn.Module()
        layer._map_global_expert_id_to_local_expert_id=lambda i:i
        method.create_weights(layer,experts,32,2048,torch.bfloat16)
        return env,method,layer

    def test_four_ranks_mapping_and_exact_load(self):
        tree=ast.parse((UPSTREAM/'vllm/model_executor/layers/fused_moe/routed_experts.py').read_text())
        fn=next(n for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='RoutedExperts'
                for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='build_expert_params_mapping')
        fn=copy.deepcopy(fn);fn.decorator_list=[]
        env={'EplbState':types.SimpleNamespace(build_initial_global_physical_to_logical_map=lambda n,r:list(range(n)))}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'actual_mapping','exec'),env)
        mappings=env[fn.name]('gate_proj','down_proj','up_proj',2)
        for bits in (2,3):
            _,method,layer=self.make_layer(bits)
            self.assertFalse(hasattr(layer,'w13_weight'))
            params=dict(layer.named_parameters())
            for expert in range(2):
                for projection,shard in [('gate_proj','w1'),('up_proj','w3'),('down_proj','w2')]:
                    mapping=next(x for x in mappings if x[2]==expert and x[3]==shard)
                    for rank in range(4):
                        source=f'experts.{expert}.{projection}.rank{rank}.trellis'
                        mapped=source.replace(mapping[1],mapping[0])
                        local=mapped.removeprefix('experts.routed_experts.')
                        param=params[local]
                        target=param.data[expert] if shard=='w2' else param.data[expert,0 if shard=='w1' else 1]
                        data=torch.full_like(target,expert*10+rank)
                        method._load_exl3(param,data,mapped,shard_id=shard,expert_id=expert)
                        self.assertTrue(torch.equal(target,data))
                        self.assertEqual(target.shape[-1],bits*16)
            bad=params['w2_rank0.trellis']
            with self.assertRaisesRegex(RuntimeError,'wrong packed parameter'):
                method._load_exl3(bad,torch.zeros_like(bad[0]),'experts.w2_rank1.trellis',shard_id='w2')

    def test_file_descriptor_preserved_and_scalar_materialized(self):
        _,method,layer=self.make_layer()
        source=torch.empty_like(layer.w2_rank0.trellis[0],device='meta')
        source._vllm_file_tensor_source=transport.FileTensorSource('pack',123,tuple(source.shape),source.dtype)
        class Writer:
            def __init__(self):self.calls=[];self.materialized=[]
            def __call__(self,dest,src):self.calls.append((dest,src));return True
            def materialize(self,src):self.materialized.append(src);return torch.tensor(-877912083,dtype=torch.int32)
        writer=Writer()
        with transport.weight_transfer(writer):
            method._load_exl3(layer.w2_rank0.trellis,source,'experts.w2_rank0.trellis',shard_id='w2')
            self.assertIs(writer.calls[0][1],source)
            self.assertEqual(writer.materialized,[])
            scalar=torch.empty((),dtype=torch.int32,device='meta')
            scalar._vllm_file_tensor_source=transport.FileTensorSource('pack',123,(),scalar.dtype)
            method._load_exl3(layer.w2_rank0.mcg,scalar,'experts.w2_rank0.mcg',shard_id='w2')
            self.assertEqual(len(writer.materialized),1)
            self.assertEqual(writer.calls[1][1].shape,(1,))
            self.assertEqual(writer.calls[1][1].item(),-877912083)
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main()
