import ast
import copy
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest

from apply_patch import apply, patched_source
from layerwise_counts import layer_config, plan, validate_loaded_tensor

HERE = Path(__file__).parent


def config():
    return NS(n_routed_experts=288, num_hidden_layers=45, n_group=1,
              topk_group=1, num_experts_per_token=8, hidden_size=4096,
              mlp_layer_types=['dense'] * 3 + ['sparse'] * 42,
              routed_experts_per_layer={'5': 256, '32': 192},
              retained_expert_ids_by_layer={'5': list(range(32, 288)),
                                            '32': list(range(0, 288, 2)) + list(range(1, 96, 2))})


def parallel():
    return NS(tensor_parallel_size=1, pipeline_parallel_size=1,
              data_parallel_size=1, enable_expert_parallel=False, enable_eplb=False,
              use_sequence_parallel_moe=False,
              eplb_config=NS(num_redundant_experts=0))


class LayerwiseTests(unittest.TestCase):
    def setUp(self):
        self.c = config()
        self.c.retained_expert_ids_by_layer['32'].sort()

    def test_all_layers_route_with_no_shared_config_mutation(self):
        before = copy.deepcopy(vars(self.c))
        for layer in range(45):
            selected = layer_config(self.c, parallel(), layer)
            self.assertEqual(selected.n_routed_experts, {5: 256, 32: 192}.get(layer, 288))
            self.assertEqual(selected.num_experts_per_token, 8)
        self.assertEqual(vars(self.c), before)
        self.assertIs(layer_config(self.c, parallel(), 45, True), self.c)

    def test_wrapper_text_config_copy_does_not_mutate_native_geometry(self):
        class Wrapper:
            def __getattr__(self, name):
                return getattr(object.__getattribute__(self, "text_config"), name)
            def __setattr__(self, name, value):
                if name == "text_config": object.__setattr__(self, name, value)
                else: setattr(self.text_config, name, value)
        wrapper = Wrapper(); wrapper.text_config = self.c
        selected = layer_config(wrapper, parallel(), 5)
        self.assertEqual(selected.n_routed_experts, 256)
        self.assertEqual(wrapper.n_routed_experts, 288)
        self.assertIsNot(selected.text_config, wrapper.text_config)

    def test_uniform_source_unchanged(self):
        del self.c.routed_experts_per_layer
        del self.c.retained_expert_ids_by_layer
        self.assertIs(layer_config(self.c, parallel(), 5), self.c)

    def test_reject_invalid_maps_before_allocation(self):
        cases = [('routed_experts_per_layer', []),
                 ('routed_experts_per_layer', {'05': 256}),
                 ('routed_experts_per_layer', {'5': 7, '32': 192}),
                 ('routed_experts_per_layer', {'5': True, '32': 192}),
                 ('n_group', 2), ('n_routed_experts', 256)]
        for name, value in cases:
            c = copy.deepcopy(self.c); setattr(c, name, value)
            with self.assertRaises(ValueError): plan(c)
        self.c.retained_expert_ids_by_layer['5'][0] = 287
        with self.assertRaises(ValueError): plan(self.c)

    def test_no_dense_or_mtp_pruning(self):
        for layer in ('2', '45'):
            c = copy.deepcopy(self.c)
            c.routed_experts_per_layer[layer] = 256
            c.retained_expert_ids_by_layer[layer] = list(range(256))
            with self.assertRaises(ValueError): plan(c)
        self.c.n_routed_experts = 256
        with self.assertRaises(ValueError): layer_config(self.c, parallel(), 45, True)

    def test_unsupported_parallel_and_eplb_fail(self):
        for name, value in [('tensor_parallel_size', 2), ('data_parallel_size', 2),
                            ('pipeline_parallel_size', 2), ('enable_expert_parallel', True),
                            ('enable_eplb', True), ('use_sequence_parallel_moe', True),
                            ('decode_context_parallel_size', 2)]:
            p = parallel(); setattr(p, name, value)
            with self.assertRaises(ValueError): layer_config(self.c, p, 3)
        p = parallel(); p.eplb_config.num_redundant_experts = 1
        with self.assertRaises(ValueError): layer_config(self.c, p, 3)

    def test_router_and_expert_archive_rows_use_same_count(self):
        for layer, count in [(5, 256), (32, 192), (44, 288)]:
            for prefix in ('layers.', 'model.language_model.layers.'):
                base = f'{prefix}{layer}.mlp.'
                validate_loaded_tensor(self.c, base+'gate.weight', (count, 4096))
                validate_loaded_tensor(self.c, base+'gate.e_score_correction_bias', (count,))
                validate_loaded_tensor(self.c, base+f'experts.{count-1}.gate_proj.rank3.trellis', (1,))
                for name, shape in [('gate.weight', (count+1, 4096)),
                                    ('gate.e_score_correction_bias', (count+1,)),
                                    (f'experts.{count}.up_proj.rank0.mcg', (1,))]:
                    with self.assertRaises(ValueError):
                        validate_loaded_tensor(self.c, base+name, shape)

    def test_patch_is_exact_idempotent_and_rejects_unknown_sources(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d)/'model.py'; original=(HERE/'model.original.py').read_bytes()
            target.write_bytes(original); apply(target, True)
            self.assertEqual(target.read_bytes(), original)
            digest=apply(target); self.assertEqual(apply(target), digest)
            ast.parse(target.read_text())
            target.write_bytes(target.read_bytes()+b'\n# unrelated\n')
            with self.assertRaises(ValueError): apply(target)

    def test_pinned_moe_constructor_uses_selected_router_and_expert_shapes(self):
        # Execute the actual source constructor; only GPU allocation backends
        # are CPU recorders. This tests arguments, not kernels or numerical quality.
        tree=ast.parse((HERE/'model.original.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Glm5NextMoE')
        cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__']
        class Module: pass
        def fake(**kw): return NS(**kw)
        env={'nn':NS(Module=Module,Parameter=lambda x:x),
             'torch':NS(float32='f32',empty=lambda *shape,**kw:shape),
             'allocate_weights':lambda f,*a,**kw:f(*a,**kw),
             'get_tensor_model_parallel_world_size':lambda:1,
             'get_tensor_model_parallel_rank':lambda:0,
             'get_ep_group':lambda:NS(device_group=NS(size=lambda:1),rank_in_group=0),
             '_get_moe_router_dtype':lambda c:'f32',
             'GateLinear':lambda a,b,**kw:NS(shape=(b,a),out_dtype='f32'),
             'Glm5NextMLP':fake,'FusedMoEFactory':fake}
        code=compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),cls],type_ignores=[])),'pinned-moe','exec')
        exec(code,env)
        for layer,count in [(5,256),(32,192),(44,288)]:
            c=layer_config(self.c,parallel(),layer)
            c.n_shared_experts=1;c.hidden_act='silu';c.moe_intermediate_size=2048
            c.topk_method='noaux_tc';c.moe_renormalize=True
            moe=env['Glm5NextMoE'](c,parallel())
            self.assertEqual(moe.gate.shape,(count,4096))
            self.assertEqual(moe.gate.e_score_correction_bias,(count,))
            self.assertEqual(moe.experts.num_experts,count)
            self.assertEqual(moe.experts.top_k,8)
            self.assertEqual(moe.n_local_physical_experts,count)


if __name__=='__main__': unittest.main()
