import importlib.util,unittest,types
from pathlib import Path
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('layer_bits',ROOT/'layer_bits.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
class LayerBitsTests(unittest.TestCase):
 def config(self):return types.SimpleNamespace(bits=2,layer_bits=mod.parse_layer_bits({'5':3,'32':3}),raw_config={'rank_stacked_tp':4})
 def test_all_target_layers(self):
  cfg=self.config()
  for n in range(3,45):
   for prefix in ['model','language_model.model']:
    got=mod.for_routed_prefix(cfg,f'{prefix}.layers.{n}.mlp.experts')
    self.assertEqual(got.bits,3 if n in (5,32) else 2)
    self.assertEqual(got.raw_config,{'rank_stacked_tp':4})
  self.assertEqual(cfg.bits,2);self.assertEqual(cfg.layer_bits,{'5':3,'32':3})
 def test_no_shared_mutation(self):
  cfg=self.config();got=mod.for_routed_prefix(cfg,'model.layers.5.mlp.experts');got.raw_config['rank_stacked_tp']=99
  self.assertEqual(cfg.raw_config,{'rank_stacked_tp':4});self.assertEqual(cfg.bits,2)
 def test_native_mtp_and_unknown_prefix_rejected(self):
  for prefix in ['model.layers.45.mlp.experts','visual.layers.5.mlp.experts','model.layers.5.mlp.shared_experts','model.layers.2.mlp.experts','arbitrary']:
   with self.assertRaises(ValueError):mod.for_routed_prefix(self.config(),prefix)
 def test_invalid_config_rejected(self):
  for value in [[],{'05':3},{'0':3},{'45':3},{'5':True},{'5':3.0},{'5':1},{5:3}]:
   with self.assertRaises(ValueError):mod.parse_layer_bits(value)
 def test_uniform_baseline_unchanged(self):
  cfg=self.config();cfg.layer_bits={};self.assertIs(mod.for_routed_prefix(cfg,'original-prefix'),cfg)
 def test_input_map_copied(self):
  source={'5':3};parsed=mod.parse_layer_bits(source);source['5']=2;self.assertEqual(parsed,{'5':3})
if __name__=='__main__':unittest.main()
