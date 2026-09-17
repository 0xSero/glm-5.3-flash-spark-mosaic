import unittest
from mixed_contract import expected_bits,validate_trellis_shape
class ContractTests(unittest.TestCase):
 def test_all_layers(self):
  for layer in range(3,45):
   p=f'model.language_model.layers.{layer}.mlp.experts.287.down_proj.rank3'
   k=3 if layer in (5,32) else 2
   self.assertEqual(expected_bits(p),k);validate_trellis_shape(p,(16,128,16*k))
 def test_wrong_bits(self):
  for layer in (3,5,32,44):
   p=f'model.language_model.layers.{layer}.mlp.experts.0.gate_proj.rank0'
   with self.assertRaises(ValueError):validate_trellis_shape(p,(16,128,16*(5-expected_bits(p))))
 def test_protected_and_invalid_rejected(self):
  for p in ('lm_head','model.language_model.layers.45.mlp.experts.0.gate_proj.rank0','model.language_model.layers.5.mlp.experts.288.gate_proj.rank0','model.language_model.layers.2.mlp.experts.0.gate_proj.rank0'):
   with self.assertRaises(ValueError):expected_bits(p)
if __name__=='__main__':unittest.main()
