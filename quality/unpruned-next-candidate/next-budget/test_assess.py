import unittest
from assess import build
class BudgetTests(unittest.TestCase):
 def test_source_geometry_and_storage(self):
  r=build();m=r['mtp'];self.assertEqual(m['native_routed_bytes'],288*3*4096*2048*2)
  self.assertEqual(m['nominal_saved_bytes'],3170891520)
  self.assertEqual(m['native_nonexpert_bytes'],369670784)
 def test_frozen_nested_allocation_and_increments(self):
  cs=build()['candidates'];previous=set()
  for c in cs:
   current=set(c['selected_layers']);self.assertTrue(previous<=current);previous=current
   self.assertEqual(len(current),c['down_layers']);self.assertEqual(c['extra_target_tensor_bytes'],len(current)*288*2**20)
   self.assertTrue(all(v=={'down_proj':3} for v in c['layer_projection_bits'].values()));self.assertIsNone(c['predicted_agreement'])
if __name__=='__main__':unittest.main()
