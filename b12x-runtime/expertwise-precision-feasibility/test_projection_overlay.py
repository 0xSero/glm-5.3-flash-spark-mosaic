"""CPU AST contract checks on the generated actual quant-method dispatch."""
import ast
import copy
from pathlib import Path
import re
from types import SimpleNamespace as NS
import unittest
from projection_contract import resolve_layer_projection_bits

HERE=Path(__file__).parent


class OverlayTests(unittest.TestCase):
    def dispatch(self):
        tree=ast.parse((HERE/'projection-gpu/exl3.projection.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Exl3Config')
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='get_quant_method')
        class ImportRecorder(ast.NodeTransformer):
            def visit_ImportFrom(self,node):
                if node.module in ('vllm.model_executor.layers.fused_moe.routed_experts','glm_exl3_projection_bits'):
                    return ast.Pass()
                return node
        fn=ImportRecorder().visit(fn)
        class Routed: pass
        class Linear: pass
        env={'re':re,'copy':copy,'RoutedExperts':Routed,'LinearBase':Linear,
             'resolve_layer_projection_bits':resolve_layer_projection_bits,
             'Exl3MoEMethod':lambda moe,config:config,
             'UnquantizedLinearMethod':lambda:'native'}
        code=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[])
        exec(compile(ast.fix_missing_locations(code),'actual-projection-dispatch','exec'),env)
        layer=Routed();layer.moe_config=NS()
        return env['get_quant_method'],layer,Linear

    def test_actual_dispatch_preserves_shared_config_and_native_linears(self):
        fn,layer,Linear=self.dispatch()
        c=NS(bits=2,raw_config={'layer_projection_bits':{'5':{'down_proj':3}}})
        d=fn(c,layer,'language_model.model.layers.5.mlp.experts')
        self.assertEqual(d.projection_bits,(2,2,3))
        self.assertFalse(hasattr(c,'projection_bits'))
        self.assertEqual(fn(c,layer,'model.layers.32.mlp.experts').projection_bits,(2,2,2))
        self.assertEqual(fn(c,Linear(),'lm_head'),'native')
        with self.assertRaises(ValueError):fn(c,layer,'model.layers.45.mlp.experts')

    def test_conflicting_whole_layer_metadata_rejected(self):
        fn,layer,_=self.dispatch()
        c=NS(bits=2,raw_config={'layer_bits':{'5':3},'layer_projection_bits':{'5':{'down_proj':3}}})
        with self.assertRaises(ValueError):fn(c,layer,'model.layers.5.mlp.experts')


if __name__=='__main__':unittest.main()
