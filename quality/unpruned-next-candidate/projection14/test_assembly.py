import json,struct,tempfile,unittest
from pathlib import Path
from assemble_projection import rewrite,H

def source(path,k,marker):
 header={};payload=bytearray()
 for proj in ('gate_proj','up_proj','down_proj'):
  for field in ('trellis','suh','svh','mcg'):
   shape=[1,1,16*k] if field=='trellis' else ([] if field=='mcg' else [1]);dtype='I16' if field=='trellis' else ('I32' if field=='mcg' else 'F16');size=32*k if field=='trellis' else (4 if field=='mcg' else 2)
   name=f'model.language_model.layers.5.mlp.experts.0.{proj}.rank0.{field}';header[name]={'dtype':dtype,'shape':shape,'data_offsets':[len(payload),len(payload)+size]};payload.extend(bytes([marker])*size)
 h=json.dumps(header,separators=(',',':')).encode();path.write_bytes(struct.pack('<Q',len(h))+h+payload);return {'bytes':path.stat().st_size,'sha256':H.sha(path)}
class AssemblyTests(unittest.TestCase):
 def test_only_down_changes_and_sources_immutable(self):
  with tempfile.TemporaryDirectory() as tmp:
   d=Path(tmp);a=d/'a';b=d/'b';out=d/'out';ra=source(a,2,2);rb=source(b,3,3);self.assertEqual(rewrite(a,b,out,ra,rb),4)
   h=H.header(out)
   with out.open('rb') as f:
    start=8+struct.unpack('<Q',f.read(8))[0]
    for n,v in h.items():
     down='.down_proj.' in n;f.seek(start+v['data_offsets'][0]);data=f.read(v['data_offsets'][1]-v['data_offsets'][0]);self.assertEqual(set(data),{3 if down else 2})
     if n.endswith('.trellis'):self.assertEqual(v['shape'][-1],48 if down else 32)
   self.assertEqual(H.sha(a),ra['sha256']);self.assertEqual(H.sha(b),rb['sha256'])
 def test_source_corruption_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   d=Path(tmp);a=d/'a';b=d/'b';ra=source(a,2,2);rb=source(b,3,3);rb['sha256']='0'*64
   with self.assertRaises(ValueError):rewrite(a,b,d/'out',ra,rb)
   self.assertFalse((d/'out').exists())
if __name__=='__main__':unittest.main()
