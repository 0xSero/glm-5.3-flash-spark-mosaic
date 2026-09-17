import unittest
from mixed_contract import expected_bits,validate_trellis_shape
class ProjectionContract(unittest.TestCase):
 def test_all_projection_precisions(self):
  for l in range(3,45):
   for proj in ('gate_proj','up_proj','down_proj'):
    key=f'model.language_model.layers.{l}.mlp.experts.287.{proj}.rank3';k=3 if l in (5,26,31,32,35,36) and proj=='down_proj' else 2
    self.assertEqual(expected_bits(key),k);validate_trellis_shape(key,(32,256,16*k))
    with self.assertRaises(ValueError):validate_trellis_shape(key,(32,256,16*(5-k)))
 def test_native_exclusion(self):
  for key in ('lm_head','model.language_model.layers.45.mlp.experts.0.down_proj.rank0','model.language_model.layers.2.mlp.experts.0.gate_proj.rank0'):
   with self.assertRaises(ValueError):expected_bits(key)
if __name__=='__main__':unittest.main()
