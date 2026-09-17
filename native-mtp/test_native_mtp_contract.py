import copy, json, struct, tempfile, unittest
from pathlib import Path
from prepare_native_mtp_view import prepare
ROOT=Path(__file__).parent
class NativeMTPContract(unittest.TestCase):
 def fixture(self, root):
  c=json.loads((ROOT/'source-config.json').read_text());(root/'config.json').write_text(json.dumps(c));h={}
  for e in range(288):
   for proj in ('gate_proj','up_proj','down_proj'):h[f'model.language_model.layers.45.mlp.experts.{e}.{proj}.weight']={'dtype':'BF16','shape':[1],'data_offsets':[0,2]}
  for i in range(23):h[f'model.language_model.layers.45.extra{i}.weight']={'dtype':'BF16','shape':[1],'data_offsets':[0,2]}
  h['model.language_model.layers.45.mlp.gate.weight']={'dtype':'BF16','shape':[288,4096],'data_offsets':[0,2]}
  h['model.language_model.layers.45.mlp.gate.e_score_correction_bias']={'dtype':'F32','shape':[288],'data_offsets':[0,4]}
  for k in ('lm_head.weight','model.language_model.embed_tokens.weight'):h[k]={'dtype':'BF16','shape':[1],'data_offsets':[0,2]}
  data=json.dumps(h).encode();(root/'native.safetensors').write_bytes(struct.pack('<Q',len(data))+data+b'1234');(root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{k:'native.safetensors' for k in h}}));return c
 def test_native_view_retains_unpruned_config(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td);c=self.fixture(p);native=p/'native-config.json';native.write_text(json.dumps(c));c['text_config']['n_routed_experts']=160;(p/'config.json').write_text(json.dumps(c));receipt=prepare(p,p/'draft','/native-source',native);d=json.loads((p/'draft/config.json').read_text());self.assertEqual(d['text_config']['n_routed_experts'],288);self.assertNotIn('quantization_config',d);self.assertEqual(receipt['index_tensor_count'],891);self.assertEqual((p/'draft/native.safetensors').readlink(),Path('/native-source/native.safetensors'));self.assertEqual(json.loads((p/'config.json').read_text())['text_config']['n_routed_experts'],160)
 def test_nested_source_is_flattened_without_changing_source(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td);self.fixture(p);(p/'retained').mkdir();(p/'native.safetensors').rename(p/'retained/native.safetensors');idx=p/'model.safetensors.index.json';d=json.loads(idx.read_text());d['weight_map']={k:'retained/native.safetensors' for k in d['weight_map']};idx.write_text(json.dumps(d));receipt=prepare(p,p/'draft','/native-source');self.assertEqual(receipt['index_tensor_count'],891);self.assertEqual((p/'draft/native.safetensors').readlink(),Path('/native-source/retained/native.safetensors'));self.assertEqual(set(json.loads((p/'draft/model.safetensors.index.json').read_text())['weight_map'].values()),{'native.safetensors'})
 def test_pruned_config_fails_without_original(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td);c=self.fixture(p);c['text_config']['n_routed_experts']=160;(p/'config.json').write_text(json.dumps(c))
   with self.assertRaises(AssertionError):prepare(p,p/'draft','/source')
 def test_missing_companion_fails(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td);self.fixture(p);idx=p/'model.safetensors.index.json';d=json.loads(idx.read_text());d['weight_map'].pop(next(iter(d['weight_map'])));idx.write_text(json.dumps(d))
   with self.assertRaises(AssertionError):prepare(p,p/'draft','/source')
if __name__=='__main__':unittest.main()
