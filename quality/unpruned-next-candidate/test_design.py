import json,unittest
from pathlib import Path
class Designs(unittest.TestCase):
 def test_equal_current_memory_expert_design(self):
  x=json.loads(Path(__file__).with_name('expert576-design.json').read_text())
  self.assertEqual(x['upgraded_experts'],576);self.assertEqual(x['additional_tensor_bytes'],1811939328);self.assertEqual(x['projected_tensor_bytes'],113086732152)
  self.assertEqual(x['mixed_precision_layers'],24)
 def test_all_experts_and_native_preserved(self):
  for name in ('expert576','whole3'):
   x=json.loads(Path(__file__).with_name(name+'-design.json').read_text());self.assertEqual(x['experts_retained'],12096);self.assertEqual(x['removed_experts'],0);self.assertEqual(x['native_mtp_experts'],288);self.assertEqual(x['native_protected_tensor_bytes'],33835039608)
   self.assertEqual(set(x['k3_experts_by_layer']),set(map(str,range(3,45))))
   for ids in x['k3_experts_by_layer'].values():self.assertEqual(ids,sorted(set(ids)));self.assertTrue(all(0<=e<288 for e in ids))
   self.assertFalse(x['selection_uses_heldout_quality']);self.assertIsNone(x['quality_prediction'])
 def test_whole3_policy(self):
  x=json.loads(Path(__file__).with_name('whole3-design.json').read_text());self.assertEqual(x['quantization_config_proposal']['layer_bits'],{'5':3,'32':3,'6':3});self.assertEqual(x['additional_tensor_bytes'],2717908992);self.assertEqual(x['mixed_precision_layers'],0)
if __name__=='__main__':unittest.main()
