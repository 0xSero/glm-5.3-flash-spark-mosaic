import unittest
from types import SimpleNamespace
import torch
from full_sparse import capture_raw_setup,FIELDS,BOUNDARY,freeze_check
class SparseContracts(unittest.TestCase):
 def test_source_freezes_and_boundary_ids(self):
  self.assertEqual(len(freeze_check()),5);self.assertEqual(len(set(BOUNDARY)),8)
  self.assertTrue({0,7,8,127,255,287}<=set(BOUNDARY));self.assertTrue(all(0<=x<288 for x in BOUNDARY))
 def test_setup_hook_copies_then_restores_inherited_method(self):
  seen=[]
  class Parent:
   def _setup_kernel(self,layer):seen.append(layer)
  class Child(Parent):pass
  layer=SimpleNamespace(**{n:torch.tensor([i],dtype=torch.int32) for i,n in enumerate(FIELDS)});raw={}
  with capture_raw_setup(Child,raw):Child()._setup_kernel(layer)
  self.assertNotIn('_setup_kernel',Child.__dict__);self.assertEqual(len(seen),1)
  layer.w13_weight.fill_(99);self.assertEqual(int(raw['w13_weight'][0]),0)
 def test_setup_hook_restores_on_failure(self):
  class Method:
   def _setup_kernel(self,layer):raise RuntimeError('sentinel')
  original=Method._setup_kernel;layer=SimpleNamespace(**{n:torch.zeros(1) for n in FIELDS})
  with self.assertRaisesRegex(RuntimeError,'sentinel'):
   with capture_raw_setup(Method,{}):Method()._setup_kernel(layer)
  self.assertIs(Method._setup_kernel,original)
if __name__=='__main__':unittest.main()
