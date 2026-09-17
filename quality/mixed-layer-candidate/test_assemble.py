import importlib.util,tempfile,unittest,json,struct,hashlib
from pathlib import Path
s=importlib.util.spec_from_file_location('mixed',Path(__file__).with_name('assemble.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class AssemblyTests(unittest.TestCase):
 def test_source_hash_failure_never_creates_output(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);src=r/'src';src.write_bytes(b'abc');dst=r/'dst'
   with self.assertRaises(ValueError):m.verified_materialize(src,dst,{'bytes':3,'sha256':'wrong'})
   self.assertFalse(dst.exists());self.assertEqual(src.read_bytes(),b'abc')
 def test_hardlink_preserves_bytes_and_rejects_overwrite(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);src=r/'src';src.write_bytes(b'abc');dst=r/'dst';exp={'bytes':3,'sha256':hashlib.sha256(b'abc').hexdigest()}
   self.assertEqual(m.verified_materialize(src,dst,exp),'hardlink');self.assertEqual(src.read_bytes(),dst.read_bytes())
   with self.assertRaises(ValueError):m.verified_materialize(src,dst,exp)
 def test_config_isolation_native_protected(self):
  config={'text_config':{'n_routed_experts':288},'quantization_config':{'bits':2}};q={'bits':2,'k2_experts_per_layer':288,'quant_method':'exl3'}
  c,new=m.config_with_layers(config,q)
  self.assertEqual(c['text_config']['n_routed_experts'],288);self.assertEqual(c['quantization_config'],new)
  self.assertEqual(new['layer_bits'],{'5':3,'32':3});self.assertEqual(new['bits'],2);self.assertEqual(q['k2_experts_per_layer'],288)
  self.assertNotIn('layer_bits',config['quantization_config']);self.assertEqual(c['native_mtp_n_routed_experts'],288)
 def test_header_rejects_unreferenced_payload(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'file';h=json.dumps({'x':{'dtype':'I16','shape':[1],'data_offsets':[0,2]}}).encode();p.write_bytes(struct.pack('<Q',len(h))+h+b'123')
   with self.assertRaises(ValueError):m.header(p)
if __name__=='__main__':unittest.main()
